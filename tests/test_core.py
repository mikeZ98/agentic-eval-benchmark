"""Evaluation engine: orchestration, concurrency, retries, fail-safe scoring, aggregation."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_eval_benchmark.core import (
    DEFAULT_METRICS,
    BenchmarkSettings,
    EvaluationEngine,
    HeuristicJudge,
    RAGResponse,
    TransientError,
    aggregate_metrics,
    compare_to_baseline,
    gate_passes,
    read_report,
    write_run,
)
from agentic_eval_benchmark.schemas import (
    GoldenDataset,
    GoldenSample,
    MetricScore,
    MetricType,
    RegressionStatus,
    ThresholdPolicy,
)
from conftest import MakeRun, golden_sample

F, AR = MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCY
GOOD = RAGResponse(
    answer="Paris is the capital of France.",
    contexts=("Paris is the capital of France. It has 2.1 million inhabitants.",),
)
CLOCK = datetime(2026, 10, 26, 12, tzinfo=UTC)


class ScriptedRAG:
    """Per-question behaviours: ok, flaky (2 transient faults), down, bad, slow, wrong."""

    def __init__(self, plan: dict[str, str] | None = None) -> None:
        self.plan = plan or {}
        self.calls: dict[str, int] = {}

    async def query(self, question: str) -> RAGResponse:
        calls = self.calls[question] = self.calls.get(question, 0) + 1
        action = self.plan.get(question, "ok")
        if action == "flaky" and calls <= 2:
            raise TransientError("503")
        if action == "down":
            raise ConnectionError("refused")
        if action == "bad":
            raise ValueError("schema drift")
        if action == "slow":
            await asyncio.sleep(5)
        if action == "wrong":
            return RAGResponse(answer="Lyon is the capital of France.", contexts=GOOD.contexts)
        return GOOD


class TrackingJudge:
    """Wraps a judge and records the peak number of concurrent calls."""

    def __init__(self, inner: HeuristicJudge) -> None:
        self.inner = inner
        self.active = 0
        self.peak = 0

    async def evaluate_sample(
        self, sample: GoldenSample, answer: str, contexts: tuple[str, ...]
    ) -> tuple[MetricScore, ...]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            return await self.inner.evaluate_sample(sample, answer, contexts)
        finally:
            self.active -= 1


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def engine(
    rag: ScriptedRAG,
    judge: object,
    settings: BenchmarkSettings,
    *,
    sleep: Sleeps | None = None,
    metrics: Sequence[MetricType] | None = None,
) -> EvaluationEngine:
    return EvaluationEngine(
        rag,
        judge,  # type: ignore[arg-type]
        settings=settings,
        sleep=sleep or Sleeps(),
        clock=lambda: CLOCK,
        metrics=DEFAULT_METRICS if metrics is None else metrics,
    )


@pytest.fixture
def backoff_settings(tmp_path: Path) -> BenchmarkSettings:
    return BenchmarkSettings(
        output_dir=tmp_path / "runs",
        concurrency_limit=2,
        timeout_seconds=0.2,
        retry_attempts=3,
        retry_backoff_seconds=0.5,
        retry_backoff_max_seconds=8.0,
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_defaults_and_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_RETRY_BACKOFF_SECONDS")
    monkeypatch.delenv("EVAL_RETRY_BACKOFF_MAX_SECONDS")
    defaults = BenchmarkSettings()
    assert (defaults.concurrency_limit, defaults.judge_model, defaults.retry_attempts) == (
        5,
        "heuristic",
        3,
    )
    assert defaults.uses_offline_judge
    assert [defaults.backoff_delay(n) for n in (1, 2, 3, 6)] == [0.5, 1.0, 2.0, 8.0]
    assert defaults.run_dir("r1") == defaults.output_dir / "r1"
    with pytest.raises(ValueError, match="start at 1"):
        defaults.backoff_delay(0)


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVAL_JUDGE_MODEL", " RAGAS ")
    monkeypatch.setenv("EVAL_OUTPUT_DIR", "relative/runs")
    monkeypatch.setenv("EVAL_RAG_URL", "http://ignored.example/")  # other families do not leak in
    loaded = BenchmarkSettings()
    assert loaded.judge_model == "ragas"
    assert not loaded.uses_offline_judge
    assert loaded.output_dir == (tmp_path / "relative/runs").resolve()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"concurrency_limit": 0},
        {"judge_model": "gpt-4o"},
        {"retry_backoff_seconds": 2.0, "retry_backoff_max_seconds": 1.0},
        {"timeout_seconds": 0},
    ],
)
def test_settings_reject(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BenchmarkSettings(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Engine orchestration
# ---------------------------------------------------------------------------


async def test_engine_bounds_concurrency(
    small_dataset: GoldenDataset, backoff_settings: BenchmarkSettings
) -> None:
    tracker = TrackingJudge(HeuristicJudge())
    run = await engine(ScriptedRAG(), tracker, backoff_settings).run(small_dataset, run_id="run-1")
    assert tracker.peak == 2
    assert run.summary.gate_passed
    assert run.summary.pass_rate == 1.0
    assert run.summary.regression_status is RegressionStatus.NO_BASELINE
    assert run.errors == ()


async def test_engine_retries_and_records_failures(
    small_dataset: GoldenDataset, backoff_settings: BenchmarkSettings
) -> None:
    q = [s.question for s in small_dataset.samples]
    rag = ScriptedRAG({q[0]: "flaky", q[1]: "down", q[2]: "bad", q[3]: "slow"})
    sleeps = Sleeps()
    run = await engine(rag, HeuristicJudge(), backoff_settings, sleep=sleeps).run(
        small_dataset, run_id="r"
    )
    errors = {e.sample_id: e for e in run.errors}
    assert "s0" not in errors
    assert rag.calls[q[0]] == 3
    assert (errors["s1"].error_type, errors["s1"].attempts, errors["s1"].stage) == (
        "ConnectionError",
        4,
        "rag",
    )
    assert (errors["s2"].error_type, errors["s2"].attempts) == ("ValueError", 1)
    assert (errors["s3"].error_type, errors["s3"].attempts) == ("TimeoutError", 4)
    assert sorted(sleeps.delays) == sorted([0.5, 1.0] + [0.5, 1.0, 2.0] * 2)
    failed = {e.sample_id: e for e in run.report.evaluations}
    assert all(m.score == 0.0 and not m.passed for m in failed["s1"].metrics)
    assert "Not evaluated: rag failed after 4 attempt(s)" in failed["s1"].metrics[0].reasoning
    assert not run.summary.gate_passed


async def test_engine_regression_and_improvement(
    small_dataset: GoldenDataset, backoff_settings: BenchmarkSettings
) -> None:
    healthy = engine(ScriptedRAG(), HeuristicJudge(), backoff_settings)
    baseline = await healthy.run(small_dataset, run_id="base")
    wrong = {s.question: "wrong" for s in small_dataset.samples}
    regressed = await engine(ScriptedRAG(wrong), HeuristicJudge(), backoff_settings).run(
        small_dataset, run_id="bad", baseline=baseline.summary
    )
    assert regressed.summary.regression_status is RegressionStatus.REGRESSED
    assert "faithfulness" in regressed.summary.regressed_metrics
    improved = await healthy.run(small_dataset, run_id="fixed", baseline=regressed.summary)
    assert improved.summary.regression_status is RegressionStatus.IMPROVED
    stable = await healthy.run(small_dataset, run_id="again", baseline=improved.summary)
    assert stable.summary.regression_status is RegressionStatus.STABLE
    generated = await healthy.run(small_dataset)
    assert generated.summary.run_id.startswith("run-20261026T120000Z-")


async def test_engine_rejects_incomparable_baselines(
    small_dataset: GoldenDataset, backoff_settings: BenchmarkSettings
) -> None:
    eng = engine(ScriptedRAG(), HeuristicJudge(), backoff_settings)
    baseline = (await eng.run(small_dataset, run_id="base")).summary
    other = GoldenDataset(name="other", dataset_version="1.0.0", samples=(golden_sample(),))
    with pytest.raises(ValueError, match="evaluated dataset"):
        await eng.run(other, baseline=baseline)
    with pytest.raises(ValueError, match="own baseline"):
        await eng.run(small_dataset, run_id="base", baseline=baseline)


async def test_policy_owns_thresholds_and_strict_mode_fills_gaps(
    sample: GoldenSample, settings: BenchmarkSettings
) -> None:
    class LenientJudge:
        async def evaluate_sample(
            self, sample: GoldenSample, answer: str, contexts: tuple[str, ...]
        ) -> tuple[MetricScore, ...]:
            return (
                MetricScore(metric=F, score=0.86, threshold=0.5, passed=True, reasoning="lenient"),
                MetricScore(
                    metric=MetricType.CONTEXT_RECALL, score=0.1, passed=True, reasoning="info"
                ),
            )

    evaluation, error = await engine(ScriptedRAG(), LenientJudge(), settings).evaluate_sample(
        sample
    )
    faithfulness, relevancy = evaluation.score_for(F), evaluation.score_for(AR)
    assert error is None
    assert faithfulness is not None
    assert faithfulness.threshold == 0.85
    assert faithfulness.passed
    assert relevancy is not None
    assert not relevancy.passed
    assert "strict mode" in relevancy.reasoning
    lax = EvaluationEngine(
        ScriptedRAG(), LenientJudge(), settings=settings, policy=ThresholdPolicy(strict_mode=False)
    )
    relaxed, _ = await lax.evaluate_sample(sample)
    assert relaxed.score_for(AR) is None


async def test_invalid_judge_output_fails_the_sample(
    sample: GoldenSample, settings: BenchmarkSettings
) -> None:
    class DuplicateJudge:
        async def evaluate_sample(
            self, sample: GoldenSample, answer: str, contexts: tuple[str, ...]
        ) -> tuple[MetricScore, ...]:
            return (MetricScore(metric=F, score=1.0, passed=True, reasoning="a"),) * 2

    evaluation, error = await engine(ScriptedRAG(), DuplicateJudge(), settings).evaluate_sample(
        sample
    )
    assert not evaluation.passed
    assert error is not None
    assert error.stage == "judge"
    assert error.attempts == 1
    assert evaluation.answer == GOOD.answer


async def test_judge_transient_errors_are_retried(
    sample: GoldenSample, settings: BenchmarkSettings
) -> None:
    class FlakyJudge:
        calls = 0

        async def evaluate_sample(
            self, sample: GoldenSample, answer: str, contexts: tuple[str, ...]
        ) -> tuple[MetricScore, ...]:
            FlakyJudge.calls += 1
            if FlakyJudge.calls == 1:
                raise TransientError("rate limited")
            return await HeuristicJudge().evaluate_sample(sample, answer, contexts)

    evaluation, error = await engine(ScriptedRAG(), FlakyJudge(), settings).evaluate_sample(sample)
    assert error is None
    assert evaluation.passed
    assert FlakyJudge.calls == 2


async def test_cancellation_propagates(sample: GoldenSample, settings: BenchmarkSettings) -> None:
    data = GoldenDataset(name="demo", dataset_version="1.0.0", samples=(sample,))
    task = asyncio.create_task(
        engine(ScriptedRAG({sample.question: "slow"}), HeuristicJudge(), settings).run(data)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "metrics",
    [(), (MetricType.FAITHFULNESS, MetricType.FAITHFULNESS), (MetricType.ASPECT_CRITIQUE,)],
)
def test_engine_configuration(metrics: tuple[MetricType, ...], settings: BenchmarkSettings) -> None:
    with pytest.raises(ValueError, match=r"metrics|aspect"):
        engine(ScriptedRAG(), HeuristicJudge(), settings, metrics=metrics)


def test_engine_exposes_policy(settings: BenchmarkSettings) -> None:
    assert engine(ScriptedRAG(), HeuristicJudge(), settings).policy == ThresholdPolicy()


# ---------------------------------------------------------------------------
# Aggregation, regression, gate, persistence
# ---------------------------------------------------------------------------


def test_aggregation_and_cost(make_run: MakeRun) -> None:
    run = make_run("r", [0.9, 0.7], [0.95, 0.85], cost=0.01)
    faithfulness = run.summary.aggregate_for(F)
    assert faithfulness is not None
    assert faithfulness.count == 2
    assert faithfulness.mean == pytest.approx(0.8)
    assert faithfulness.pass_rate == 0.5
    assert run.summary.total_cost_usd == pytest.approx(0.04)
    assert aggregate_metrics(run.report.evaluations) == run.summary.metrics


def test_compare_to_baseline(make_run: MakeRun) -> None:
    base = make_run("base", [0.95, 0.95], [0.95, 0.95]).summary
    current = make_run("cur", [0.94, 0.94], drop_relevancy=True).summary
    assert compare_to_baseline(current.metrics, None, tolerance=0.0) == (
        RegressionStatus.NO_BASELINE,
        (),
    )
    assert compare_to_baseline(current.metrics, base, tolerance=0.02) == (
        RegressionStatus.STABLE,
        (),
    )
    assert (
        compare_to_baseline(current.metrics, base, tolerance=0.0)[0] is RegressionStatus.REGRESSED
    )


def test_gate_passes_rules(make_run: MakeRun) -> None:
    metrics = make_run("r", [0.9, 0.9], drop_relevancy=True).summary.metrics
    strict, lax = ThresholdPolicy(), ThresholdPolicy(strict_mode=False)
    stable = RegressionStatus.STABLE
    assert not gate_passes(
        strict, metrics, failed_samples=0, regression_status=stable
    )  # AR missing
    assert gate_passes(lax, metrics, failed_samples=1, regression_status=stable)
    assert not gate_passes(
        lax, metrics, failed_samples=0, regression_status=RegressionStatus.REGRESSED
    )
    low = make_run("r", [0.5, 0.5]).summary.metrics
    assert not gate_passes(lax, low, failed_samples=0, regression_status=stable)


def test_write_and_read_run(tmp_path: Path, make_run: MakeRun) -> None:
    run = make_run("persisted", [0.9, 0.95])
    run_dir = write_run(run, tmp_path)
    assert {p.name for p in run_dir.iterdir()} == {
        "report.json",
        "summary.parquet",
        "samples.parquet",
    }
    assert read_report(run_dir) == run.report
    with pytest.raises(FileExistsError):
        write_run(run, tmp_path)


async def test_errors_file_is_written(
    tmp_path: Path, small_dataset: GoldenDataset, settings: BenchmarkSettings
) -> None:
    rag = ScriptedRAG({s.question: "bad" for s in small_dataset.samples})
    run = await engine(rag, HeuristicJudge(), settings).run(small_dataset, run_id="broken")
    run_dir = write_run(run, tmp_path)
    assert (run_dir / "errors.json").read_text(encoding="utf-8").count('"stage":"rag"') == 4
