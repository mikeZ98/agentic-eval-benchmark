"""Shared fixtures: hermetic environment, settings, datasets, judges, backends, API clients.

Every test runs with sockets disabled (``--disable-socket``), in its own temporary
working directory, with all ``EVAL_*`` / provider / CI variables removed and every
state directory (runs, TruLens, Ragas user ID) pinned under ``tmp_path``.
"""

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr

from agentic_eval_benchmark.api import ApiSettings, EvaluationServices, create_app
from agentic_eval_benchmark.api import services as services_module
from agentic_eval_benchmark.core import (
    BenchmarkSettings,
    EvaluationRun,
    HeuristicJudge,
    summarize,
)
from agentic_eval_benchmark.core.judge import build_score
from agentic_eval_benchmark.schemas import (
    BenchmarkReport,
    BenchmarkRunSummary,
    GoldenDataset,
    GoldenSample,
    MetricScore,
    MetricType,
    SampleEvaluation,
    ThresholdPolicy,
)
from agentic_eval_benchmark.tools import StaticRAGClient, load_reference_dataset
from agentic_eval_benchmark.tools.offline import (
    HashingEmbeddings,
    ScriptedRagasLLM,
    ScriptedTruLensProvider,
    chain_of_thought_reply,
)
from agentic_eval_benchmark.tools.rag_client import HttpRAGClient, RAGEndpointConfig

API_KEY = "k" * 40
T0 = datetime(2026, 10, 1, tzinfo=UTC)
FAITHFULNESS = MetricType.FAITHFULNESS
RELEVANCY = MetricType.ANSWER_RELEVANCY

type MakeRun = Callable[..., EvaluationRun]
type RagHandler = Callable[[httpx.Request], httpx.Response]

_SCRUBBED_PREFIXES = ("EVAL_", "GITHUB_", "OPENAI_", "ANTHROPIC_", "OTEL_", "FASTAPI_OTEL")


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Scrub ambient configuration and pin all state under ``tmp_path``."""
    for name in list(os.environ):
        if name.startswith(_SCRUBBED_PREFIXES):
            monkeypatch.delenv(name)
    monkeypatch.setenv("EVAL_OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("EVAL_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("EVAL_RETRY_BACKOFF_MAX_SECONDS", "0")
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("TRULENS_HOME", str(tmp_path / "trulens"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Settings, policies, datasets, judges
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> BenchmarkSettings:
    return BenchmarkSettings(
        output_dir=tmp_path / "runs",
        timeout_seconds=5.0,
        retry_attempts=3,
        retry_backoff_seconds=0.0,
        retry_backoff_max_seconds=0.0,
    )


@pytest.fixture
def policy() -> ThresholdPolicy:
    return ThresholdPolicy()


@pytest.fixture(scope="session")
def reference_dataset() -> GoldenDataset:
    return load_reference_dataset()


def golden_sample(sample_id: str = "s1", /, **overrides: object) -> GoldenSample:
    fields: dict[str, object] = {
        "sample_id": sample_id,
        "question": "What is the capital of France?",
        "contexts": ("Paris is the capital of France. It has 2.1 million inhabitants.",),
        "ground_truth": "The capital of France is Paris.",
    }
    return GoldenSample.model_validate(fields | overrides)


@pytest.fixture
def sample() -> GoldenSample:
    return golden_sample()


@pytest.fixture
def small_dataset() -> GoldenDataset:
    return GoldenDataset(
        name="demo",
        dataset_version="1.0.0",
        samples=tuple(
            golden_sample(f"s{i}", question=f"What is the capital of France, variant {i}?")
            for i in range(4)
        ),
    )


@pytest.fixture
def judge(policy: ThresholdPolicy) -> HeuristicJudge:
    return HeuristicJudge(policy=policy)


@pytest.fixture
def oracle(reference_dataset: GoldenDataset) -> StaticRAGClient:
    return StaticRAGClient.from_dataset(reference_dataset)


# ---------------------------------------------------------------------------
# Synthetic runs for gate / drift / report tests
# ---------------------------------------------------------------------------


def _dataset(name: str, version: str, size: int) -> GoldenDataset:
    return GoldenDataset(
        name=name,
        dataset_version=version,
        samples=tuple(golden_sample(f"s{i}", question=f"What is item {i}?") for i in range(size)),
    )


def _evaluation(
    index: int, f: float, ar: float, policy: ThresholdPolicy, *, drop_relevancy: bool
) -> SampleEvaluation:
    scores = [
        build_score(
            FAITHFULNESS, f, "faithfulness reasoning", threshold=policy.threshold_for(FAITHFULNESS)
        )
    ]
    if not drop_relevancy:
        scores.append(
            build_score(
                RELEVANCY, ar, "relevancy reasoning", threshold=policy.threshold_for(RELEVANCY)
            )
        )
    return SampleEvaluation(
        sample_id=f"s{index}", metrics=tuple(scores), passed=all(s.passed for s in scores)
    )


@pytest.fixture
def make_run() -> MakeRun:
    """Build a validated run from per-sample faithfulness / relevancy scores."""

    def build(
        run_id: str,
        faithfulness: Sequence[float],
        relevancy: Sequence[float] | None = None,
        *,
        policy: ThresholdPolicy | None = None,
        baseline: BenchmarkRunSummary | None = None,
        day: int = 0,
        name: str = "demo",
        version: str = "1.0.0",
        drop_relevancy: bool = False,
        cost: float | None = None,
    ) -> EvaluationRun:
        gate = policy or ThresholdPolicy()
        ar = list(relevancy) if relevancy is not None else [0.95] * len(faithfulness)
        evaluations = [
            _evaluation(i, f, a, gate, drop_relevancy=drop_relevancy)
            for i, (f, a) in enumerate(zip(faithfulness, ar, strict=True))
        ]
        if cost is not None:
            evaluations = [
                e.model_copy(
                    update={
                        "metrics": tuple(
                            MetricScore.model_validate(m.model_dump() | {"cost_usd": cost})
                            for m in e.metrics
                        )
                    }
                )
                for e in evaluations
            ]
        summary = summarize(
            evaluations,
            run_id=run_id,
            dataset=_dataset(name, version, len(evaluations)),
            policy=gate,
            started_at=T0 + timedelta(days=day),
            duration_seconds=1.5,
            baseline=baseline,
        )
        return EvaluationRun(
            report=BenchmarkReport(summary=summary, evaluations=tuple(evaluations))
        )

    return build


# ---------------------------------------------------------------------------
# In-memory Ragas / TruLens backends
# ---------------------------------------------------------------------------


@pytest.fixture
def ragas_payloads(sample: GoldenSample) -> dict[str, Mapping[str, object]]:
    return {
        "StatementGeneratorOutput": {"statements": ["Paris is the capital.", "Paris is big."]},
        "NLIStatementOutput": {
            "statements": [
                {"statement": "Paris is the capital.", "reason": "stated", "verdict": 1},
                {"statement": "Paris is big.", "reason": "not stated", "verdict": 0},
            ]
        },
        "AnswerRelevanceOutput": {"question": sample.question, "noncommittal": 0},
        "ContextPrecisionOutput": {"reason": "useful", "verdict": 1},
        "ContextRecallOutput": {
            "classifications": [
                {"statement": sample.ground_truth, "reason": "in context", "attributed": 1}
            ]
        },
    }


@pytest.fixture
def scripted_llm(ragas_payloads: dict[str, Mapping[str, object]]) -> ScriptedRagasLLM:
    return ScriptedRagasLLM.from_payloads(ragas_payloads)


@pytest.fixture
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings()


def trulens_responder(text: str, response_format: type[BaseModel] | None) -> str:
    """Splits the answer into two statements; grounds the first, not the second."""
    if response_format is None:
        return "Paris is the capital of France.\nParis has ten million cats."
    if "INFORMATION OVERLAP" in text:
        statement = text.rsplit("Statement:", maxsplit=1)[-1]
        return chain_of_thought_reply(0 if "cats" in statement else 3, "checked against source")
    if "RELEVANCE" in text:
        return chain_of_thought_reply(3, "answers the whole prompt")
    return chain_of_thought_reply(2, "criterion mostly met")


@pytest.fixture
def trulens_provider() -> ScriptedTruLensProvider:
    return ScriptedTruLensProvider(trulens_responder)


# ---------------------------------------------------------------------------
# RAG over a mock HTTP transport (no sockets)
# ---------------------------------------------------------------------------


class RagRecorder:
    """Mutable handler + request log shared with the patched HTTP client."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.handler: RagHandler = self.unreachable

    @staticmethod
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)


@pytest.fixture
def mock_rag(monkeypatch: pytest.MonkeyPatch) -> RagRecorder:
    """Route every ``HttpRAGClient`` built by the service layer through a MockTransport."""
    recorder = RagRecorder()

    class MockedHttpRAGClient(HttpRAGClient):
        def __init__(self, config: RAGEndpointConfig | None = None) -> None:
            super().__init__(
                config, client=httpx.AsyncClient(transport=httpx.MockTransport(recorder))
            )
            self._owns_client = True

    monkeypatch.setattr(services_module, "HttpRAGClient", MockedHttpRAGClient)
    return recorder


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_dir(tmp_path: Path, reference_dataset: GoldenDataset) -> Path:
    root = tmp_path / "datasets"
    root.mkdir()
    (root / "ref.json").write_text(reference_dataset.to_json(), encoding="utf-8")
    (tmp_path / "outside.json").write_text("[]", encoding="utf-8")
    return root


@pytest.fixture
def services(settings: BenchmarkSettings, dataset_dir: Path) -> EvaluationServices:
    return EvaluationServices(
        settings,
        api=ApiSettings(
            api_key=SecretStr(API_KEY),
            dataset_dir=dataset_dir,
            allowed_rag_hosts=("rag.internal",),
            max_request_bytes=200_000,
        ),
    )


@pytest.fixture
def api_client(services: EvaluationServices) -> Iterator[TestClient]:
    with TestClient(create_app(services)) as client:
        yield client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}
