"""Evaluation engine: queries the RAG system under test, judges and aggregates a run."""

import asyncio
import math
import secrets
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import Field

from agentic_eval_benchmark.core.config import BenchmarkSettings
from agentic_eval_benchmark.core.judge import DEFAULT_METRICS, JudgeProtocol, build_score
from agentic_eval_benchmark.schemas import (
    BenchmarkReport,
    BenchmarkRunSummary,
    EvalModel,
    GoldenDataset,
    GoldenSample,
    Identifier,
    MetricAggregate,
    MetricScore,
    MetricType,
    RegressionStatus,
    SampleEvaluation,
    ThresholdPolicy,
)

_MAX_ERROR_CHARS: Final = 500
# Means are compared after float summation; ignore drops smaller than rounding noise.
_REGRESSION_EPSILON: Final = 1e-9


class TransientError(Exception):
    """A retryable fault (rate limit, 5xx, dropped connection) raised by a RAG client
    or judge. The engine also retries :class:`TimeoutError` and :class:`ConnectionError`."""


RETRYABLE_ERRORS: Final = (TransientError, TimeoutError, ConnectionError)


class RAGResponse(EvalModel):
    """What the RAG system under test returned for one question."""

    answer: str = Field(max_length=32_768)
    contexts: tuple[str, ...] = Field(default=(), max_length=256)


@runtime_checkable
class RAGClientProtocol(Protocol):
    """Port to the external RAG system under test (for example a ``rag_documents`` API)."""

    async def query(self, question: str) -> RAGResponse: ...


class SampleError(EvalModel):
    """Why a sample could not be evaluated; the sample is recorded as failed."""

    sample_id: Identifier
    stage: Literal["rag", "judge"]
    error_type: str = Field(min_length=1, max_length=128)
    message: str = Field(max_length=_MAX_ERROR_CHARS)
    attempts: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """The validated report of a run plus the samples that errored."""

    report: BenchmarkReport
    errors: tuple[SampleError, ...] = ()

    @property
    def summary(self) -> BenchmarkRunSummary:
        """The run summary."""
        return self.report.summary


class _StageError(Exception):
    def __init__(self, stage: Literal["rag", "judge"], cause: Exception, attempts: int) -> None:
        super().__init__(f"{stage} failed after {attempts} attempt(s)")
        self.stage: Literal["rag", "judge"] = stage
        self.cause = cause
        self.attempts = attempts


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EvaluationEngine:
    """Evaluates a :class:`GoldenDataset` against a RAG system with an LLM judge.

    - At most ``settings.concurrency_limit`` samples are in flight at once.
    - Each RAG query and judge call gets ``settings.timeout_seconds`` per attempt and
      is retried on :data:`RETRYABLE_ERRORS` with capped exponential backoff.
    - Fail-closed: a sample that still errors is scored ``0.0`` (failed) on every
      expected metric and reported in :attr:`EvaluationRun.errors`, so an unreachable
      endpoint can never pass the gate. Cancellation is never swallowed.
    - The policy, not the judge, decides pass/fail for gated metrics; under
      ``strict_mode`` a gated metric missing from the judge's output fails the sample.
    """

    def __init__(
        self,
        rag_client: RAGClientProtocol,
        judge: JudgeProtocol,
        *,
        settings: BenchmarkSettings | None = None,
        policy: ThresholdPolicy | None = None,
        metrics: Sequence[MetricType] = DEFAULT_METRICS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if not metrics or len(set(metrics)) != len(metrics):
            raise ValueError("metrics must be a non-empty sequence of unique metrics")
        if MetricType.ASPECT_CRITIQUE in metrics:
            raise ValueError("aspect critiques are judge-specific; list only the RAG metrics")
        self._rag = rag_client
        self._judge = judge
        self._settings = settings or BenchmarkSettings()
        self._policy = policy or ThresholdPolicy()
        self._metrics = tuple(metrics)
        self._sleep = sleep
        self._clock = clock

    @property
    def policy(self) -> ThresholdPolicy:
        """The quality-gate policy applied to every run."""
        return self._policy

    async def run(
        self,
        dataset: GoldenDataset,
        *,
        run_id: str | None = None,
        baseline: BenchmarkRunSummary | None = None,
    ) -> EvaluationRun:
        """Evaluate every sample and return the validated report."""
        started_at = self._clock()
        run_id = run_id or f"run-{started_at:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"
        if baseline is not None:
            if baseline.dataset_name != dataset.name:
                raise ValueError(
                    f"baseline {baseline.run_id} evaluated dataset {baseline.dataset_name!r}, "
                    f"not {dataset.name!r}"
                )
            if baseline.run_id == run_id:
                raise ValueError("a run cannot be its own baseline")

        t0 = time.perf_counter()
        semaphore = asyncio.Semaphore(self._settings.concurrency_limit)

        async def bounded(sample: GoldenSample) -> tuple[SampleEvaluation, SampleError | None]:
            async with semaphore:
                return await self.evaluate_sample(sample)

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(bounded(s)) for s in dataset.samples]
        outcomes = [t.result() for t in tasks]
        duration = time.perf_counter() - t0

        evaluations = tuple(e for e, _ in outcomes)
        errors = tuple(err for _, err in outcomes if err is not None)
        summary = summarize(
            evaluations,
            run_id=run_id,
            dataset=dataset,
            policy=self._policy,
            started_at=started_at,
            duration_seconds=duration,
            baseline=baseline,
        )
        return EvaluationRun(
            report=BenchmarkReport(summary=summary, evaluations=evaluations), errors=errors
        )

    async def evaluate_sample(
        self, sample: GoldenSample
    ) -> tuple[SampleEvaluation, SampleError | None]:
        """Query, judge and gate one sample; never raises except on cancellation."""
        answer: str | None = None
        try:
            response = await self._attempt("rag", lambda: self._rag.query(sample.question))
            answer = response.answer
            scores = await self._attempt(
                "judge",
                lambda: self._judge.evaluate_sample(sample, response.answer, response.contexts),
            )
            gated = self._apply_policy(scores)
            evaluation = SampleEvaluation(
                sample_id=sample.sample_id,
                answer=answer,
                metrics=gated,
                passed=all(m.passed for m in gated),
            )
        except _StageError as failure:
            return self._failed(sample, answer, failure.stage, failure.cause, failure.attempts)
        except Exception as exc:  # noqa: BLE001 - fail-safe: invalid judge output fails the sample
            return self._failed(sample, answer, "judge", exc, 1)
        return evaluation, None

    async def _attempt[T](
        self, stage: Literal["rag", "judge"], operation: Callable[[], Awaitable[T]]
    ) -> T:
        attempt = 0
        while True:
            attempt += 1
            try:
                async with asyncio.timeout(self._settings.timeout_seconds):
                    return await operation()
            except RETRYABLE_ERRORS as exc:
                if attempt > self._settings.retry_attempts:
                    raise _StageError(stage, exc, attempt) from exc
                await self._sleep(self._settings.backoff_delay(attempt))
            except Exception as exc:
                raise _StageError(stage, exc, attempt) from exc

    def _apply_policy(self, scores: Iterable[MetricScore]) -> tuple[MetricScore, ...]:
        gated: list[MetricScore] = []
        for score in scores:
            threshold = None if score.aspect else self._policy.threshold_for(score.metric)
            if threshold is None or score.threshold == threshold:
                gated.append(score)
                continue
            gated.append(
                build_score(
                    score.metric,
                    score.score,
                    score.reasoning,
                    threshold=threshold,
                    cost_usd=score.cost_usd,
                    latency_ms=score.latency_ms,
                )
            )
        if self._policy.strict_mode:
            present = {s.metric for s in gated if s.aspect is None}
            for metric, threshold in self._policy.thresholds().items():
                if metric in self._metrics and metric not in present:
                    gated.append(
                        build_score(
                            metric,
                            0.0,
                            "Judge returned no verdict for this gated metric (strict mode).",
                            threshold=threshold,
                        )
                    )
        return tuple(gated)

    def _failed(
        self,
        sample: GoldenSample,
        answer: str | None,
        stage: Literal["rag", "judge"],
        cause: Exception,
        attempts: int,
    ) -> tuple[SampleEvaluation, SampleError]:
        detail = str(cause).strip()
        error = SampleError(
            sample_id=sample.sample_id,
            stage=stage,
            error_type=type(cause).__name__[:128],
            message=detail[:_MAX_ERROR_CHARS],
            attempts=attempts,
        )
        reason = f"Not evaluated: {stage} failed after {attempts} attempt(s) ({error.error_type})."
        # No threshold: a 0.0 score would "pass" an ungated metric (or a 0.0 gate), and
        # an errored sample must fail on every metric.
        metrics = tuple(
            MetricScore(metric=m, score=0.0, passed=False, reasoning=reason) for m in self._metrics
        )
        evaluation = SampleEvaluation(
            sample_id=sample.sample_id, answer=answer, metrics=metrics, passed=False
        )
        return evaluation, error


# ---------------------------------------------------------------------------
# Aggregation, regression and gating
# ---------------------------------------------------------------------------

_METRIC_ORDER: Final = {m: i for i, m in enumerate(MetricType)}


def aggregate_metrics(evaluations: Iterable[SampleEvaluation]) -> tuple[MetricAggregate, ...]:
    """One :class:`MetricAggregate` per metric key, in :class:`MetricType` order."""
    grouped: dict[tuple[MetricType, str | None], list[MetricScore]] = {}
    for evaluation in evaluations:
        for score in evaluation.metrics:
            grouped.setdefault((score.metric, score.aspect), []).append(score)
    ordered = sorted(grouped.items(), key=lambda kv: (_METRIC_ORDER[kv[0][0]], kv[0][1] or ""))
    return tuple(
        MetricAggregate(
            metric=metric,
            aspect=aspect,
            count=len(scores),
            mean=math.fsum(s.score for s in scores) / len(scores),
            minimum=min(s.score for s in scores),
            maximum=max(s.score for s in scores),
            pass_rate=sum(s.passed for s in scores) / len(scores),
        )
        for (metric, aspect), scores in ordered
    )


def compare_to_baseline(
    metrics: Sequence[MetricAggregate],
    baseline: BenchmarkRunSummary | None,
    *,
    tolerance: float,
) -> tuple[RegressionStatus, tuple[str, ...]]:
    """Classify the run against ``baseline`` by metric means.

    A metric regresses when its mean drops by more than ``tolerance``; the run is
    ``IMPROVED`` when nothing regressed and at least one mean rose by more than it.
    Metrics absent from either run are not compared.
    """
    if baseline is None:
        return RegressionStatus.NO_BASELINE, ()
    regressed: list[str] = []
    improved = False
    for aggregate in metrics:
        previous = baseline.aggregate_for(aggregate.metric, aggregate.aspect)
        if previous is None:
            continue
        delta = aggregate.mean - previous.mean
        if delta < -(tolerance + _REGRESSION_EPSILON):
            regressed.append(aggregate.key)
        elif delta > tolerance + _REGRESSION_EPSILON:
            improved = True
    if regressed:
        return RegressionStatus.REGRESSED, tuple(regressed)
    return (RegressionStatus.IMPROVED if improved else RegressionStatus.STABLE), ()


def gate_passes(
    policy: ThresholdPolicy,
    metrics: Sequence[MetricAggregate],
    *,
    failed_samples: int,
    regression_status: RegressionStatus,
) -> bool:
    """Whether a run clears the quality gate under ``policy``.

    It fails on a regression, on any gated mean below its threshold, and in strict
    mode on any failed sample or a gated metric that was not evaluated.
    """
    if regression_status is RegressionStatus.REGRESSED:
        return False
    if policy.strict_mode and failed_samples:
        return False
    means = {m.metric: m.mean for m in metrics if m.aspect is None}
    for metric, minimum in policy.thresholds().items():
        mean = means.get(metric)
        if mean is None:
            if policy.strict_mode:
                return False
        elif mean < minimum:
            return False
    return True


def summarize(
    evaluations: Sequence[SampleEvaluation],
    *,
    run_id: str,
    dataset: GoldenDataset,
    policy: ThresholdPolicy,
    started_at: datetime,
    duration_seconds: float,
    baseline: BenchmarkRunSummary | None = None,
) -> BenchmarkRunSummary:
    """Aggregate per-sample evaluations into a validated :class:`BenchmarkRunSummary`."""
    metrics = aggregate_metrics(evaluations)
    failed = tuple(e.sample_id for e in evaluations if not e.passed)
    status, regressed = compare_to_baseline(
        metrics, baseline, tolerance=policy.regression_tolerance
    )
    costs = [c for e in evaluations if (c := e.total_cost_usd) is not None]
    passed = len(evaluations) - len(failed)
    return BenchmarkRunSummary(
        run_id=run_id,
        dataset_name=dataset.name,
        dataset_version=dataset.dataset_version,
        started_at=started_at,
        duration_seconds=max(0.0, duration_seconds),
        total_samples=len(evaluations),
        passed_samples=passed,
        pass_rate=passed / len(evaluations),
        failed_sample_ids=failed,
        metrics=metrics,
        policy=policy,
        gate_passed=gate_passes(
            policy, metrics, failed_samples=len(failed), regression_status=status
        ),
        regression_status=status,
        baseline_run_id=None if baseline is None else baseline.run_id,
        regressed_metrics=regressed,
        total_cost_usd=math.fsum(costs) if costs else None,
    )


# ---------------------------------------------------------------------------
# Persistence (SSD-pinned via BenchmarkSettings.output_dir)
# ---------------------------------------------------------------------------

REPORT_FILE: Final = "report.json"
ERRORS_FILE: Final = "errors.json"
SUMMARY_PARQUET: Final = "summary.parquet"
SAMPLES_PARQUET: Final = "samples.parquet"


def write_run(run: EvaluationRun, output_dir: Path) -> Path:
    """Persist a run to ``output_dir/<run_id>/`` and return that directory.

    Writes ``report.json`` (round-trips through :func:`read_report`), the summary and
    per-sample Parquet tables and, when samples errored, ``errors.json``. An existing
    run directory is never overwritten.
    """
    run_dir = output_dir / run.summary.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / REPORT_FILE).write_text(run.report.to_json(indent=2) + "\n", encoding="utf-8")
    run.summary.to_parquet(run_dir / SUMMARY_PARQUET)
    run.report.to_parquet(run_dir / SAMPLES_PARQUET)
    if run.errors:
        errors = "[" + ",".join(e.to_json() for e in run.errors) + "]\n"
        (run_dir / ERRORS_FILE).write_text(errors, encoding="utf-8")
    return run_dir


def read_report(run_dir: Path) -> BenchmarkReport:
    """Load and re-validate the report persisted by :func:`write_run`."""
    return BenchmarkReport.from_json((run_dir / REPORT_FILE).read_bytes())
