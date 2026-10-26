"""Quality gate: re-evaluate a run summary against a policy, a baseline and drift.

The gate is deliberately independent of the run: a summary produced under a lax policy
can be re-gated against the blueprint policy in CI. With the run's own policy and
tolerance it reaches the same verdict as ``summary.gate_passed`` (the engine uses the
same rules), but it also says *why*.
"""

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import Field, model_validator

from agentic_eval_benchmark.schemas import (
    BenchmarkRunSummary,
    EvalModel,
    Identifier,
    MetricAggregate,
    MetricScore,
    NonEmptyText,
    RegressionStatus,
    SampleEvaluation,
    ThresholdPolicy,
    UnitInterval,
)

if TYPE_CHECKING:
    from agentic_eval_benchmark.guardrails.drift import DriftReport

# Means are float aggregates; ignore drops smaller than summation noise.
_EPSILON: Final = 1e-9

Delta = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]


class ViolationCode(StrEnum):
    """Why a gate failed."""

    METRIC_BELOW_THRESHOLD = "metric_below_threshold"
    METRIC_MISSING = "metric_missing"
    SAMPLE_FAILURES = "sample_failures"
    REGRESSION = "regression"
    METRIC_DRIFT = "metric_drift"


class GateViolation(EvalModel):
    """One reason the gate failed, with the numbers behind it."""

    code: ViolationCode
    metric: str | None = Field(default=None, max_length=160)
    message: NonEmptyText
    observed: float | None = Field(default=None, allow_inf_nan=False)
    expected: float | None = Field(default=None, allow_inf_nan=False)


class MetricDelta(EvalModel):
    """A metric mean compared with the same metric in the baseline run."""

    metric: str = Field(min_length=1, max_length=160)
    current: UnitInterval
    baseline: UnitInterval
    delta: Delta
    tolerance: UnitInterval
    regressed: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not math.isclose(self.delta, self.current - self.baseline, abs_tol=_EPSILON):
            raise ValueError("delta must equal current - baseline")
        if self.regressed != (self.delta < -(self.tolerance + _EPSILON)):
            raise ValueError("regressed contradicts delta and tolerance")
        return self


class GateResult(EvalModel):
    """Verdict of a :class:`QualityGate`; ``passed`` holds exactly when nothing was violated."""

    run_id: Identifier
    passed: bool
    policy: ThresholdPolicy
    tolerance_delta: UnitInterval
    violations: tuple[GateViolation, ...] = ()
    failing_sample_ids: tuple[Identifier, ...] = ()
    metric_deltas: tuple[MetricDelta, ...] = ()
    baseline_run_id: Identifier | None = None
    notes: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.passed == bool(self.violations):
            raise ValueError("passed must be True exactly when there are no violations")
        return self

    @property
    def regressions(self) -> tuple[MetricDelta, ...]:
        """Deltas beyond the tolerance."""
        return tuple(d for d in self.metric_deltas if d.regressed)


class BaselineMismatchError(ValueError):
    """The baseline evaluated a different dataset, so its scores are not comparable."""


class QualityGate:
    """Gate a :class:`BenchmarkRunSummary` against the blueprint thresholds.

    Violations, in report order:

    - a gated metric mean below its threshold (or, in strict mode, not evaluated);
    - in strict mode, any failed sample;
    - a metric mean more than ``tolerance_delta`` below the baseline's. Without an
      explicit baseline, the regression the run recorded against its own baseline;
    - with a drift report, every metric it marks as decaying.
    """

    def __init__(
        self, policy: ThresholdPolicy | None = None, *, tolerance_delta: float | None = None
    ) -> None:
        self.policy = policy or ThresholdPolicy()
        tolerance = self.policy.regression_tolerance if tolerance_delta is None else tolerance_delta
        if not 0.0 <= tolerance <= 1.0 or not math.isfinite(tolerance):
            raise ValueError("tolerance_delta must be in [0, 1]")
        self.tolerance_delta = tolerance

    def evaluate(
        self,
        summary: BenchmarkRunSummary,
        baseline: BenchmarkRunSummary | None = None,
        *,
        drift: "DriftReport | None" = None,
        evaluations: Sequence[SampleEvaluation] = (),
    ) -> GateResult:
        """Gate ``summary``; raises :class:`BaselineMismatchError` for an incomparable baseline.

        With ``evaluations`` (the run's per-sample results) samples are re-judged under
        this gate's thresholds, so re-gating a lax run also lists the samples that fail
        the stricter policy. A sample the run itself failed always stays failed.
        """
        notes: list[str] = []
        violations = self._threshold_violations(summary)
        failing = self._failing_samples(summary, evaluations)
        if self.policy.strict_mode and failing:
            violations.append(
                GateViolation(
                    code=ViolationCode.SAMPLE_FAILURES,
                    message=(
                        f"{len(failing)} of {summary.total_samples} samples failed; strict "
                        "mode requires every sample to pass."
                    ),
                    observed=1.0 - len(failing) / summary.total_samples,
                    expected=1.0,
                )
            )
        deltas: tuple[MetricDelta, ...] = ()
        baseline_id = summary.baseline_run_id
        if baseline is not None:
            self._check_comparable(summary, baseline, notes)
            deltas = self.compare(summary.metrics, baseline)
            baseline_id = baseline.run_id
            violations += [self._regression(d, baseline.run_id) for d in deltas if d.regressed]
            missing = {m.key for m in baseline.metrics} - {m.key for m in summary.metrics}
            if missing:
                notes.append(f"not evaluated in this run, unlike the baseline: {sorted(missing)}")
        elif summary.regression_status is RegressionStatus.REGRESSED:
            for key in summary.regressed_metrics:
                current = next(m for m in summary.metrics if m.key == key)
                violations.append(
                    GateViolation(
                        code=ViolationCode.REGRESSION,
                        metric=key,
                        message=(
                            f"{key} regressed against baseline {summary.baseline_run_id} "
                            "(recorded by the run)."
                        ),
                        observed=current.mean,
                    )
                )
        if drift is not None:
            violations += [
                GateViolation(
                    code=ViolationCode.METRIC_DRIFT,
                    metric=m.metric,
                    message=m.reason or f"{m.metric} is decaying across recent runs.",
                    observed=m.latest,
                    expected=m.moving_average,
                )
                for m in drift.metrics
                if m.decaying
            ]
        return GateResult(
            run_id=summary.run_id,
            passed=not violations,
            policy=self.policy,
            tolerance_delta=self.tolerance_delta,
            violations=tuple(violations),
            failing_sample_ids=failing,
            metric_deltas=deltas,
            baseline_run_id=baseline_id,
            notes=tuple(notes),
        )

    def failed_metrics(self, evaluation: SampleEvaluation) -> tuple[MetricScore, ...]:
        """Metrics of ``evaluation`` that fail this gate: gated scores below the gate's
        threshold, and every other metric the run itself marked as failed."""
        failed: list[MetricScore] = []
        for score in evaluation.metrics:
            threshold = None if score.aspect else self.policy.threshold_for(score.metric)
            below = threshold is not None and score.score < threshold
            if below or not score.passed:
                failed.append(score)
        return tuple(failed)

    def _failing_samples(
        self, summary: BenchmarkRunSummary, evaluations: Sequence[SampleEvaluation]
    ) -> tuple[str, ...]:
        run_failed = set(summary.failed_sample_ids)
        if not evaluations:
            return summary.failed_sample_ids
        return tuple(
            e.sample_id for e in evaluations if e.sample_id in run_failed or self.failed_metrics(e)
        )

    def compare(
        self, metrics: Sequence[MetricAggregate], baseline: BenchmarkRunSummary
    ) -> tuple[MetricDelta, ...]:
        """Deltas for every metric present in both runs."""
        deltas: list[MetricDelta] = []
        for aggregate in metrics:
            previous = baseline.aggregate_for(aggregate.metric, aggregate.aspect)
            if previous is None:
                continue
            delta = aggregate.mean - previous.mean
            deltas.append(
                MetricDelta(
                    metric=aggregate.key,
                    current=aggregate.mean,
                    baseline=previous.mean,
                    delta=delta,
                    tolerance=self.tolerance_delta,
                    regressed=delta < -(self.tolerance_delta + _EPSILON),
                )
            )
        return tuple(deltas)

    def _threshold_violations(self, summary: BenchmarkRunSummary) -> list[GateViolation]:
        violations: list[GateViolation] = []
        for metric, minimum in self.policy.thresholds().items():
            aggregate = summary.aggregate_for(metric)
            if aggregate is None:
                if self.policy.strict_mode:
                    violations.append(
                        GateViolation(
                            code=ViolationCode.METRIC_MISSING,
                            metric=metric.value,
                            message=f"{metric.value} is gated but was not evaluated.",
                            expected=minimum,
                        )
                    )
            elif aggregate.mean < minimum:
                violations.append(
                    GateViolation(
                        code=ViolationCode.METRIC_BELOW_THRESHOLD,
                        metric=metric.value,
                        message=(
                            f"{metric.value} mean {aggregate.mean:.3f} is below the "
                            f"{minimum:.2f} threshold."
                        ),
                        observed=aggregate.mean,
                        expected=minimum,
                    )
                )
        return violations

    @staticmethod
    def _regression(delta: MetricDelta, baseline_id: str) -> GateViolation:
        return GateViolation(
            code=ViolationCode.REGRESSION,
            metric=delta.metric,
            message=(
                f"{delta.metric} dropped {-delta.delta:.3f} ({delta.baseline:.3f} → "
                f"{delta.current:.3f}) against baseline {baseline_id}, beyond the "
                f"{delta.tolerance:.3f} tolerance."
            ),
            observed=delta.current,
            expected=delta.baseline,
        )

    @staticmethod
    def _check_comparable(
        summary: BenchmarkRunSummary, baseline: BenchmarkRunSummary, notes: list[str]
    ) -> None:
        if baseline.run_id == summary.run_id:
            raise BaselineMismatchError("a run cannot be its own baseline")
        if baseline.dataset_name != summary.dataset_name:
            raise BaselineMismatchError(
                f"baseline {baseline.run_id} evaluated dataset {baseline.dataset_name!r}, "
                f"not {summary.dataset_name!r}"
            )
        if baseline.dataset_version != summary.dataset_version:
            notes.append(
                f"baseline used dataset v{baseline.dataset_version}, this run "
                f"v{summary.dataset_version}: deltas mix model and dataset changes"
            )


def _highest(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def strictest_policy(*policies: ThresholdPolicy) -> ThresholdPolicy:
    """The blueprint policy tightened by ``policies``: the highest threshold per metric,
    the smallest regression tolerance and strict mode. CI built on it can never be
    laxer than either the blueprint or the policy a run was produced with."""
    candidates = (ThresholdPolicy(), *policies)
    return ThresholdPolicy(
        faithfulness_min=max(p.faithfulness_min for p in candidates),
        relevancy_min=max(p.relevancy_min for p in candidates),
        context_precision_min=_highest([p.context_precision_min for p in candidates]),
        context_recall_min=_highest([p.context_recall_min for p in candidates]),
        regression_tolerance=min(p.regression_tolerance for p in candidates),
        strict_mode=True,
    )
