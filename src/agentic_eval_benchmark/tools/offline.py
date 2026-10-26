"""In-memory model backends: run the Ragas and TruLens adapters with zero network.

They are the injection points for hermetic tests and offline CI, and double as a
reference for wiring a self-hosted model behind the same interfaces.
"""

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

from pydantic import BaseModel, PrivateAttr
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from trulens.core.feedback.endpoint import Endpoint
from trulens.feedback.llm_provider import LLMProvider

from agentic_eval_benchmark.core._text import stem, tokenize
from agentic_eval_benchmark.tools._isolation import pin_ragas_offline, pin_trulens_offline

pin_ragas_offline()
pin_trulens_offline()

type Payload = BaseModel | Mapping[str, object]
type RagasResponder = Callable[[str, type[BaseModel]], Payload]
type TruLensResponder = Callable[[str, type[BaseModel] | None], str]


class ScriptedRagasLLM(InstructorBaseRagasLLM):
    """A Ragas structured-output LLM answered by a local function.

    ``responder(prompt, response_model)`` returns the model instance (or a mapping
    validated into it). Every call is recorded in :attr:`calls` as
    ``(response_model.__name__, prompt)``.
    """

    def __init__(self, responder: RagasResponder) -> None:
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    @classmethod
    def from_payloads(cls, payloads: Mapping[str, Payload | Sequence[Payload]]) -> Self:
        """Answer by response-model class name; a sequence is replayed in order, cyclically."""
        counters: dict[str, int] = {}

        def respond(_prompt: str, response_model: type[BaseModel]) -> Payload:
            name = response_model.__name__
            if name not in payloads:
                raise KeyError(f"no scripted payload for {name}")
            entry = payloads[name]
            if isinstance(entry, Sequence):
                index = counters.get(name, 0)
                counters[name] = index + 1
                return entry[index % len(entry)]
            return entry

        return cls(respond)

    def generate[M: BaseModel](self, prompt: str, response_model: type[M]) -> M:
        self.calls.append((response_model.__name__, prompt))
        result = self._responder(prompt, response_model)
        if isinstance(result, response_model):
            return result
        return response_model.model_validate(result)

    async def agenerate[M: BaseModel](self, prompt: str, response_model: type[M]) -> M:
        return self.generate(prompt, response_model)


class HashingEmbeddings(BaseRagasEmbedding):
    """Deterministic bag-of-stems feature-hashing embeddings (no model, no download).

    Texts sharing content words get high cosine similarity, which is what Ragas
    answer relevancy needs from an embedding model in offline runs.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if not 8 <= dimensions <= 4096:  # noqa: PLR2004
            raise ValueError("dimensions must be between 8 and 4096")
        super().__init__(cache=None)
        self.dimensions = dimensions

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:  # noqa: ARG002
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = int.from_bytes(hashlib.blake2b(stem(token).encode(), digest_size=8).digest())
            vector[digest % self.dimensions] += 1.0 if (digest >> 63) & 1 else -1.0
        norm = math.sqrt(math.fsum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text, **kwargs)


def chain_of_thought_reply(score: int, reasoning: str, criteria: str = "scripted") -> str:
    """JSON reply in the shape TruLens expects for chain-of-thought scoring prompts."""
    return json.dumps({"criteria": criteria, "supporting_evidence": reasoning, "score": score})


class ScriptedTruLensProvider(LLMProvider):
    """A TruLens LLM provider answered by a local function.

    ``responder(prompt_text, response_format)`` receives the flattened chat messages
    and returns the raw completion text: JSON for structured scoring prompts
    (``response_format`` set, see :func:`chain_of_thought_reply`) or plain text, such
    as one sentence per line for the statement-splitting prompt.
    """

    model_engine: str = "scripted"
    _responder: TruLensResponder = PrivateAttr()

    def __init__(self, responder: TruLensResponder, **kwargs: Any) -> None:
        # No pacing retries offline: TruLens would otherwise sleep 2 s, 4 s, 8 s per fault.
        kwargs.setdefault("endpoint", Endpoint(name="scripted-trulens-endpoint", retries=0))
        super().__init__(**kwargs)  # type: ignore[no-untyped-call]  # untyped upstream
        self._responder = responder

    def _create_chat_completion(
        self,
        prompt: str | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
        **kwargs: Any,  # noqa: ARG002 - temperature etc. are irrelevant offline
    ) -> str:
        parts = [prompt] if prompt else []
        parts += [str(m.get("content", "")) for m in messages or ()]
        return self._responder("\n\n".join(parts), response_format)
