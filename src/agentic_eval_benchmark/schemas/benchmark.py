"""Benchmark run contracts: the quality-gate policy, run summary and full report."""

import math
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

from pydantic import AwareDatetime, Field, model_validator

from agentic_eval_benchmark.schemas._base import (
    EvalModel,
    Identifier,
    NonNegativeFloat,
    Record,
    SemVer,
    UnitInterval,
    write_parquet,
)
from agentic_eval_benchmark.schemas.metrics import (
    MetricType,
    SampleEvaluation,
    check_aspect,
    metric_key,
)

BLUEPRINT_FAITHFULNESS_MIN: Final = 0.85
BLUEPRINT_RELEVANCY_MIN: Final = 0.85
BLUEPRINT_REGRESSION_TOLERANCE: Final = 0.02

# Aggregates are recomputed from float scores; allow for summation-order rounding.
_FLOAT_TOLERANCE: Final = 1e-9


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE)


class ThresholdPolicy(EvalModel):
    """Quality-gate configuration applied to every benchmark run.

    ``strict_mode`` (the default) pins the blueprint floor: thresholds may be raised
    above 0.85 but never lowered below it, and a gated metric missing from a sample
    counts as a failure. Non-strict mode allows any threshold, for exploratory runs.
    ``regression_tolerance`` (default 0.02, absorbing LLM-judge noise) is the largest drop
    in a metric mean, relative to the baseline run, that is not reported as a regression.
    """

    faithfulness_min: UnitInterval = BLUEPRINT_FAITHFULNESS_MIN
    relevancy_min: UnitInterval = BLUEPRINT_RELEVANCY_MIN
    context_precision_min: UnitInterval | None = None
    context_recall_min: UnitInterval | None = None
    regression_tolerance: UnitInterval = BLUEPRINT_REGRESSION_TOLERANCE
    strict_mode: bool = True

    @model_validator(mode="after")
    def _check_blueprint_floor(self) -> Self:
        if self.strict_mode:
            if self.faithfulness_min < BLUEPRINT_FAITHFULNESS_MIN:
                raise ValueError(
                    f"strict_mode forbids faithfulness_min below {BLUEPRINT_FAITHFULNESS_MIN}"
                )
            if self.relevancy_min < BLUEPRINT_RELEVANCY_MIN:
                raise ValueError(
                    f"strict_mode forbids relevancy_min below {BLUEPRINT_RELEVANCY_MIN}"
                )
        return self

    def thresholds(self) -> dict[MetricType, float]:
        """Minimum score per gated metric; ungated metrics are absent."""
        gates = {
            MetricType.FAITHFULNESS: self.faithfulness_min,
            MetricType.ANSWER_RELEVANCY: self.relevancy_min,
            MetricType.CONTEXT_PRECISION: self.context_precision_min,
            MetricType.CONTEXT_RECALL: self.context_recall_min,
        }
        return {metric: value for metric, value in gates.items() if value is not None}

    def threshold_for(self, metric: MetricType) -> float | None:
        """Minimum score for ``metric``, or ``None`` when it is not gated."""
        return self.thresholds().get(metric)


class RegressionStatus(StrEnum):
    """Outcome of comparing a run's metric means against a baseline run."""

    NO_BASELINE = "no_baseline"
    IMPROVED = "improved"
    STABLE = "stable"
    REGRESSED = "regressed"


class MetricAggregate(EvalModel):
    """Distribution of one metric (or critique aspect) across the samples of a run."""

    metric: MetricType
    aspect: Identifier | None = None
    count: int = Field(ge=1)
    mean: UnitInterval
    minimum: UnitInterval
    maximum: UnitInterval
    pass_rate: UnitInterval

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        check_aspect(self.metric, self.aspect)
        if not (self.minimum - _FLOAT_TOLERANCE <= self.mean <= self.maximum + _FLOAT_TOLERANCE):
            raise ValueError(
                f"{self.key}: expected minimum <= mean <= maximum, got "
                f"{self.minimum} / {self.mean} / {self.maximum}"
            )
        return self

    @property
    def key(self) -> str:
        """See :func:`~agentic_eval_benchmark.schemas.metric_key`."""
        return metric_key(self.metric, self.aspect)


class BenchmarkRunSummary(EvalModel):
    """Aggregated, self-consistent outcome of one benchmark run.

    The validators make contradictory summaries unrepresentable: the pass rate and
    failed IDs must match the sample counts, and ``gate_passed`` cannot be ``True``
    while a gated mean is below its threshold, the run regressed, or (in strict mode)
    any sample failed.
    """

    run_id: Identifier
    dataset_name: Identifier
    dataset_version: SemVer
    started_at: AwareDatetime
    duration_seconds: NonNegativeFloat
    total_samples: int = Field(ge=1)
    passed_samples: int = Field(ge=0)
    pass_rate: UnitInterval
    failed_sample_ids: tuple[Identifier, ...] = ()
    metrics: tuple[MetricAggregate, ...] = Field(min_length=1)
    policy: ThresholdPolicy
    gate_passed: bool
    regression_status: RegressionStatus
    baseline_run_id: Identifier | None = None
    regressed_metrics: tuple[str, ...] = ()
    total_cost_usd: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if self.passed_samples > self.total_samples:
            raise ValueError("passed_samples cannot exceed total_samples")
        if not _close(self.pass_rate, self.passed_samples / self.total_samples):
            raise ValueError(
                f"pass_rate={self.pass_rate} does not match "
                f"{self.passed_samples}/{self.total_samples}"
            )
        if len(set(self.failed_sample_ids)) != len(self.failed_sample_ids):
            raise ValueError("failed_sample_ids must be unique")
        if len(self.failed_sample_ids) != self.total_samples - self.passed_samples:
            raise ValueError(
                f"{len(self.failed_sample_ids)} failed_sample_ids for "
                f"{self.total_samples - self.passed_samples} failed samples"
            )
        keys = [m.key for m in self.metrics]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate metric aggregates: {keys}")
        if any(m.count > self.total_samples for m in self.metrics):
            raise ValueError("a metric aggregate counts more samples than the run has")
        return self

    @model_validator(mode="after")
    def _check_regression(self) -> Self:
        has_baseline = self.regression_status is not RegressionStatus.NO_BASELINE
        if has_baseline != (self.baseline_run_id is not None):
            raise ValueError("baseline_run_id is required exactly when a baseline was compared")
        if self.baseline_run_id == self.run_id:
            raise ValueError("a run cannot be its own baseline")
        regressed = self.regression_status is RegressionStatus.REGRESSED
        if regressed != bool(self.regressed_metrics):
            raise ValueError("regressed_metrics must be non-empty exactly when status is regressed")
        unknown = set(self.regressed_metrics) - {m.key for m in self.metrics}
        if unknown:
            raise ValueError(f"regressed_metrics not in metrics: {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def _check_gate(self) -> Self:
        if not self.gate_passed:
            return self
        if self.regression_status is RegressionStatus.REGRESSED:
            raise ValueError("gate_passed contradicts a regressed run")
        if self.policy.strict_mode and self.failed_sample_ids:
            raise ValueError("gate_passed contradicts failed samples under strict_mode")
        for metric, minimum in self.policy.thresholds().items():
            aggregate = self.aggregate_for(metric)
            if aggregate is None:
                if self.policy.strict_mode:
                    raise ValueError(f"gate_passed but gated metric {metric.value} is missing")
            elif aggregate.mean < minimum:
                raise ValueError(
                    f"gate_passed but {metric.value} mean {aggregate.mean} < {minimum}"
                )
        return self

    @property
    def finished_at(self) -> datetime:
        """``started_at`` plus the run duration."""
        return self.started_at + timedelta(seconds=self.duration_seconds)

    def aggregate_for(
        self, metric: MetricType, aspect: str | None = None
    ) -> MetricAggregate | None:
        """Aggregate for ``metric`` (and ``aspect``), or ``None`` if it was not evaluated."""
        key = metric_key(metric, aspect)
        return next((m for m in self.metrics if m.key == key), None)

    def mean_score(self, metric: MetricType, aspect: str | None = None) -> float | None:
        """Mean score for ``metric`` (and ``aspect``), or ``None`` if it was not evaluated."""
        aggregate = self.aggregate_for(metric, aspect)
        return None if aggregate is None else aggregate.mean

    def to_records(self) -> list[Record]:
        """One flat row per metric aggregate, repeating the run-level columns."""
        run: Record = {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "started_at": self.started_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "total_samples": self.total_samples,
            "passed_samples": self.passed_samples,
            "pass_rate": self.pass_rate,
            "gate_passed": self.gate_passed,
            "regression_status": self.regression_status.value,
            "baseline_run_id": self.baseline_run_id,
        }
        return [
            run
            | {
                "metric": m.metric.value,
                "aspect": m.aspect,
                "count": m.count,
                "mean": m.mean,
                "minimum": m.minimum,
                "maximum": m.maximum,
                "metric_pass_rate": m.pass_rate,
                "threshold": self.policy.threshold_for(m.metric),
                "regressed": m.key in self.regressed_metrics,
            }
            for m in self.metrics
        ]

    def to_parquet(self, path: Path) -> Path:
        """Write :meth:`to_records` to ``path`` and return it."""
        return write_parquet(self.to_records(), path)


class BenchmarkReport(EvalModel):
    """A run summary together with the per-sample evaluations it was computed from.

    The summary's counts, failed IDs and metric aggregates must match the evaluations.
    """

    summary: BenchmarkRunSummary
    evaluations: tuple[SampleEvaluation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_against_summary(self) -> Self:
        summary = self.summary
        ids = [e.sample_id for e in self.evaluations]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sample_id values in evaluations")
        if len(ids) != summary.total_samples:
            raise ValueError(f"{len(ids)} evaluations for total_samples={summary.total_samples}")
        failed = {e.sample_id for e in self.evaluations if not e.passed}
        if failed != set(summary.failed_sample_ids):
            raise ValueError("failed_sample_ids do not match the failed evaluations")

        scores: dict[str, list[float]] = {}
        for evaluation in self.evaluations:
            for score in evaluation.metrics:
                scores.setdefault(score.key, []).append(score.score)
        if set(scores) != {m.key for m in summary.metrics}:
            raise ValueError("summary metrics do not match the evaluated metrics")
        for aggregate in summary.metrics:
            values = scores[aggregate.key]
            if not (
                aggregate.count == len(values)
                and _close(aggregate.mean, math.fsum(values) / len(values))
                and _close(aggregate.minimum, min(values))
                and _close(aggregate.maximum, max(values))
            ):
                raise ValueError(f"aggregate {aggregate.key} does not match the evaluations")
        return self

    def to_records(self) -> list[Record]:
        """One flat row per (sample, metric), prefixed with the run ID."""
        run: Record = {"run_id": self.summary.run_id}
        return [run | row for evaluation in self.evaluations for row in evaluation.to_records()]

    def to_parquet(self, path: Path) -> Path:
        """Write :meth:`to_records` to ``path`` and return it."""
        return write_parquet(self.to_records(), path)
