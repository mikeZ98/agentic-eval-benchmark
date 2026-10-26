"""RagasJudge over in-memory backends: scoring, degradation, transient wrapping, factories."""

import asyncio
from collections.abc import Mapping
from pathlib import Path

import openai
import pytest
from pydantic import BaseModel

from agentic_eval_benchmark.core import (
    BenchmarkSettings,
    EvaluationEngine,
    HeuristicJudge,
    JudgeProtocol,
    TransientError,
)
from agentic_eval_benchmark.schemas import GoldenDataset, GoldenSample, MetricScore, MetricType
from agentic_eval_benchmark.tools import ragas_adapter
from agentic_eval_benchmark.tools.offline import HashingEmbeddings, ScriptedRagasLLM
from agentic_eval_benchmark.tools.rag_client import StaticRAGClient
from agentic_eval_benchmark.tools.ragas_adapter import RagasJudge, ragas_embeddings, ragas_llm

F, AR = MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCY
CP, CR = MetricType.CONTEXT_PRECISION, MetricType.CONTEXT_RECALL
Payloads = dict[str, Mapping[str, object]]


async def judged(
    judge: RagasJudge, sample: GoldenSample, contexts: tuple[str, ...] | None = None
) -> dict[MetricType, MetricScore]:
    result = await judge.evaluate_sample(
        sample, sample.ground_truth, sample.contexts if contexts is None else contexts
    )
    return {s.metric: s for s in result}


def responder(
    payloads: Payloads, *, fail: str | None = None, error: Exception | None = None
) -> ScriptedRagasLLM:
    def respond(prompt: str, model: type[BaseModel]) -> Mapping[str, object]:
        if model.__name__ == fail and error is not None:
            raise error
        return payloads[model.__name__]

    return ScriptedRagasLLM(respond)


async def test_scores_all_four_metrics(
    scripted_llm: ScriptedRagasLLM, embeddings: HashingEmbeddings, sample: GoldenSample
) -> None:
    judge = RagasJudge(scripted_llm, embeddings)
    result = await judged(judge, sample)
    assert result[F].score == 0.5
    assert not result[F].passed
    assert result[AR].score > 0.99
    assert result[AR].passed
    assert result[CP].score > 0.99
    assert result[CR].score == 1.0
    assert all(s.latency_ms is not None for s in result.values())
    assert result[F].reasoning.startswith("Ragas faithfulness = 0.500.")
    assert isinstance(judge, JudgeProtocol)
    assert judge.metrics == (F, AR, CP, CR)


async def test_missing_inputs_short_circuit_without_model_calls(
    scripted_llm: ScriptedRagasLLM, embeddings: HashingEmbeddings, sample: GoldenSample
) -> None:
    judge = RagasJudge(scripted_llm, embeddings)
    result = await judged(judge, sample, contexts=())
    assert result[F].reasoning == result[CR].reasoning == "No context was retrieved."
    assert {name for name, _ in scripted_llm.calls} == {"AnswerRelevanceOutput"}
    empty = await judge.evaluate_sample(sample, "   ", sample.contexts)
    assert {s.metric: s.reasoning for s in empty}[AR] == "Answer is empty."


async def test_parse_error_becomes_failure_verdict(
    ragas_payloads: Payloads, embeddings: HashingEmbeddings, sample: GoldenSample
) -> None:
    llm = responder(ragas_payloads, fail="NLIStatementOutput", error=ValueError("could not parse"))
    result = await judged(RagasJudge(llm, embeddings), sample)
    assert result[F].score == 0.0
    assert not result[F].passed
    assert (
        result[F].reasoning
        == "Ragas faithfulness failed (ValueError: could not parse); no fallback."
    )
    assert result[CR].score == 1.0


async def test_nan_score_falls_back_to_heuristic(
    ragas_payloads: Payloads, embeddings: HashingEmbeddings, sample: GoldenSample
) -> None:
    payloads = ragas_payloads | {"StatementGeneratorOutput": {"statements": []}}
    judge = RagasJudge(responder(payloads), embeddings, fallback=HeuristicJudge())
    faithfulness = (await judged(judge, sample))[F]
    assert faithfulness.passed
    assert faithfulness.score == 1.0
    assert "invalid score nan" in faithfulness.reasoning
    assert "fallback judge:" in faithfulness.reasoning


async def test_failing_fallback_keeps_the_failure(
    ragas_payloads: Payloads, embeddings: HashingEmbeddings, sample: GoldenSample
) -> None:
    class BrokenJudge:
        async def evaluate_sample(
            self, sample: GoldenSample, answer: str, contexts: tuple[str, ...]
        ) -> tuple[MetricScore, ...]:
            raise RuntimeError("fallback down")

    llm = responder(ragas_payloads, fail="NLIStatementOutput", error=ValueError("bad"))
    faithfulness = (await judged(RagasJudge(llm, embeddings, fallback=BrokenJudge()), sample))[F]
    assert not faithfulness.passed
    assert "fallback failed (RuntimeError: fallback down)" in faithfulness.reasoning


class _RateLimitedError(Exception):
    status_code = 429


async def test_transient_provider_errors_are_wrapped(
    ragas_payloads: Payloads, embeddings: HashingEmbeddings, sample: GoldenSample
) -> None:
    llm = responder(ragas_payloads, fail="NLIStatementOutput", error=_RateLimitedError("slow down"))
    with pytest.raises(TransientError, match="_RateLimitedError: slow down"):
        await RagasJudge(llm, embeddings).evaluate_sample(
            sample, sample.ground_truth, sample.contexts
        )


async def test_metric_timeout_degrades(
    ragas_payloads: Payloads, embeddings: HashingEmbeddings, sample: GoldenSample
) -> None:
    class SlowLLM(ScriptedRagasLLM):
        async def agenerate[M: BaseModel](self, prompt: str, response_model: type[M]) -> M:
            await asyncio.sleep(1)
            return self.generate(prompt, response_model)

    judge = RagasJudge(
        SlowLLM.from_payloads(ragas_payloads), embeddings, metric_timeout_seconds=0.05
    )
    result = await judged(judge, sample)
    assert all("timed out after 0.05s" in s.reasoning and not s.passed for s in result.values())


async def test_out_of_range_score_is_rejected(
    ragas_payloads: Payloads, sample: GoldenSample
) -> None:
    class Skewed(HashingEmbeddings):
        def embed_text(self, text: str, **kwargs: object) -> list[float]:
            return [3.0] + [0.0] * (self.dimensions - 1)

    judge = RagasJudge(responder(ragas_payloads), Skewed(), metrics=(AR,))
    result = await judged(judge, sample)
    assert result[AR].score == 1.0  # cosine of identical directions, within slack


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metrics": ()}, "non-empty"),
        ({"metrics": (F, F)}, "unique"),
        ({"metrics": (MetricType.ASPECT_CRITIQUE,)}, "does not score"),
        ({"embeddings": None}, "embedding model"),
        ({"answer_relevancy_strictness": 0}, "strictness"),
        ({"metric_timeout_seconds": 0}, "positive"),
    ],
)
def test_configuration(
    scripted_llm: ScriptedRagasLLM,
    embeddings: HashingEmbeddings,
    kwargs: dict[str, object],
    message: str,
) -> None:
    options: dict[str, object] = {"embeddings": embeddings} | kwargs
    with pytest.raises(ValueError, match=message):
        RagasJudge(scripted_llm, **options)  # type: ignore[arg-type]


def test_embeddings_optional_without_answer_relevancy(scripted_llm: ScriptedRagasLLM) -> None:
    assert RagasJudge(scripted_llm, metrics=(F, CR)).metrics == (F, CR)


async def test_engine_end_to_end(
    ragas_payloads: Payloads,
    embeddings: HashingEmbeddings,
    reference_dataset: GoldenDataset,
    settings: BenchmarkSettings,
) -> None:
    grounded = ragas_payloads | {
        "NLIStatementOutput": {"statements": [{"statement": "x", "reason": "ok", "verdict": 1}]}
    }
    engine = EvaluationEngine(
        StaticRAGClient.from_dataset(reference_dataset),
        RagasJudge(ScriptedRagasLLM.from_payloads(grounded), embeddings),
        settings=settings,
    )
    run = await engine.run(reference_dataset, run_id="ragas-run")
    assert run.summary.total_samples == 8
    assert run.errors == ()


def test_factories_wrap_injected_clients_without_network(tmp_path: Path) -> None:
    from ragas import _analytics  # noqa: PLC0415

    _analytics.get_userid.cache_clear()  # the user ID is cached per process
    client = openai.AsyncOpenAI(api_key="sk-test", base_url="http://127.0.0.1:9/v1")
    llm = ragas_llm(client, "test-model")
    embedder = ragas_embeddings(client, "test-embedding")
    assert llm is not None
    assert embedder is not None
    assert (tmp_path / "xdg-data" / "ragas" / "uuid.json").is_file()  # pinned, not in $HOME


def test_factories_reject_unexpected_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ragas_adapter, "llm_factory", lambda *a, **k: object())
    monkeypatch.setattr(ragas_adapter, "embedding_factory", lambda *a, **k: object())
    with pytest.raises(TypeError, match="not a structured LLM"):
        ragas_llm(object(), "m")
    with pytest.raises(TypeError, match="embedding_factory returned"):
        ragas_embeddings(object(), "m")
