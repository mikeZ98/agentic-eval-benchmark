"""Metric drift: catch gradual decay that each single run's threshold check lets through.

The tracker reads the stored run history (``EVAL_OUTPUT_DIR``, ``./.cache/eval_runs``),
keeps runs of the same dataset that started before the run under test, and compares
every metric of that run with its recent history. A metric is *decaying* when:

1. **drop**: it is more than ``max_decline`` below its moving average over the last
   ``window`` runs;
2. **sustained decline**: it fell at every step across at least ``min_history + 1``
   runs and lost more than ``max_decline`` in total;
3. **eroding headroom**: it still clears its gate by less than ``near_threshold_margin``
   while its trend over the window is negative.

Fewer than ``min_history`` earlier runs means no verdict (not decaying).
"""

import itertools
import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, Self

from pydantic import Field, ValidationError, model_validator

from agentic_eval_benchmark.core.engine import REPORT_FILE
from agentic_eval_benchmark.schemas import (
    BenchmarkRunSummary,
    EvalModel,
    Identifier,
    NonEmptyText,
    SemVer,
    ThresholdPolicy,
    UnitInterval,
)

_EPSILON: Final = 1e-9
_MAX_REPORT_BYTES: Final = 64 * 1024 * 1024


class MetricDrift(EvalModel):
    """Drift analysis of one metric of the latest run."""

    metric: str = Field(min_length=1, max_length=160)
    history_runs: int = Field(ge=0)
    latest: UnitInterval
    moving_average: UnitInterval | None = None
    decline: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    """``moving_average - latest``: positive means the latest run is worse."""
    slope_per_run: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    threshold: UnitInterval | None = None
    decaying: bool = False
    reason: NonEmptyText | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.decaying and self.reason is None:
            raise ValueError("a decaying metric needs a reason")
        return self


class DriftReport(EvalModel):
    """Drift of every metric of ``latest_run_id`` against its recent history."""

    dataset_name: Identifier
    dataset_version: SemVer | None = None
    """The version history was restricted to, or ``None`` for all versions."""
    latest_run_id: Identifier
    window: int = Field(ge=1)
    history_run_ids: tuple[Identifier, ...] = ()
    metrics: tuple[MetricDrift, ...] = ()
    skipped: tuple[str, ...] = ()
    """Run directories whose report could not be read."""

    @property
    def drifting(self) -> bool:
        """Whether any metric is decaying."""
        return any(m.decaying for m in self.metrics)

    @property
    def decaying(self) -> tuple[MetricDrift, ...]:
        return tuple(m for m in self.metrics if m.decaying)


def _slope(values: Sequence[float]) -> float:
    """Least-squares slope of ``values`` against their index (per run)."""
    n = len(values)
    mean_x = (n - 1) / 2
    mean_y = math.fsum(values) / n
    covariance = math.fsum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    variance = math.fsum((i - mean_x) ** 2 for i in range(n))
    return covariance / variance if variance else 0.0


class DriftTracker:
    """Moving-average and trend analysis over stored benchmark runs."""

    def __init__(
        self,
        root: Path,
        *,
        window: int = 5,
        min_history: int = 3,
        max_decline: float = 0.03,
        near_threshold_margin: float = 0.02,
    ) -> None:
        if not 1 <= min_history <= window:
            raise ValueError("need 1 <= min_history <= window")
        for name, value in (("max_decline", max_decline), ("margin", near_threshold_margin)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.root = root
        self.window = window
        self.min_history = min_history
        self.max_decline = max_decline
        self.near_threshold_margin = near_threshold_margin

    def history(
        self,
        dataset_name: str,
        *,
        dataset_version: str | None = None,
        before: datetime | None = None,
        exclude_run_id: str | None = None,
    ) -> tuple[list[BenchmarkRunSummary], list[str]]:
        """Stored summaries of ``dataset_name`` (oldest first) and unreadable run dirs."""
        summaries: list[BenchmarkRunSummary] = []
        skipped: list[str] = []
        if not self.root.is_dir():
            return summaries, skipped
        for run_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if run_dir.name == exclude_run_id:
                continue
            summary = self._read_summary(run_dir / REPORT_FILE)
            if summary is None:
                if (run_dir / REPORT_FILE).exists():
                    skipped.append(run_dir.name)
                continue
            if summary.dataset_name != dataset_name:
                continue
            if dataset_version is not None and summary.dataset_version != dataset_version:
                continue
            if before is not None and summary.started_at >= before:
                continue
            summaries.append(summary)
        summaries.sort(key=lambda s: (s.started_at, s.run_id))
        return summaries, skipped

    @staticmethod
    def _read_summary(file: Path) -> BenchmarkRunSummary | None:
        """Validate only the summary of a stored report (skips per-sample payloads)."""
        try:
            if not file.is_file() or file.stat().st_size > _MAX_REPORT_BYTES:
                return None
            document = json.loads(file.read_bytes())
            return BenchmarkRunSummary.model_validate_json(json.dumps(document["summary"]))
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            return None

    def track(
        self,
        latest: BenchmarkRunSummary,
        *,
        policy: ThresholdPolicy | None = None,
        same_version: bool = True,
    ) -> DriftReport:
        """Load the history of ``latest``'s dataset and analyze it."""
        version = latest.dataset_version if same_version else None
        history, skipped = self.history(
            latest.dataset_name,
            dataset_version=version,
            before=latest.started_at,
            exclude_run_id=latest.run_id,
        )
        return self.analyze(
            latest, history, policy=policy, dataset_version=version, skipped=skipped
        )

    def analyze(
        self,
        latest: BenchmarkRunSummary,
        history: Sequence[BenchmarkRunSummary],
        *,
        policy: ThresholdPolicy | None = None,
        dataset_version: str | None = None,
        skipped: Sequence[str] = (),
    ) -> DriftReport:
        """Pure analysis of ``latest`` against ``history`` (oldest first)."""
        policy = policy or latest.policy
        recent = list(history)[-self.window :]
        metrics = [self._metric(latest, recent, policy, a.key) for a in latest.metrics]
        return DriftReport(
            dataset_name=latest.dataset_name,
            dataset_version=dataset_version,
            latest_run_id=latest.run_id,
            window=self.window,
            history_run_ids=tuple(s.run_id for s in recent),
            metrics=tuple(metrics),
            skipped=tuple(skipped),
        )

    def _metric(
        self,
        latest: BenchmarkRunSummary,
        recent: Sequence[BenchmarkRunSummary],
        policy: ThresholdPolicy,
        key: str,
    ) -> MetricDrift:
        current = next(m for m in latest.metrics if m.key == key)
        points = [a.mean for s in recent for a in s.metrics if a.key == key]
        threshold = None if current.aspect else policy.threshold_for(current.metric)
        if len(points) < self.min_history:
            average = math.fsum(points) / len(points) if points else None
            return MetricDrift(
                metric=key,
                history_runs=len(points),
                latest=current.mean,
                moving_average=average,
                decline=None if average is None else average - current.mean,
                threshold=threshold,
            )
        average = math.fsum(points) / len(points)
        decline = average - current.mean
        series = [*points, current.mean]
        slope = _slope(series)
        reason = self._reason(
            key, series, average=average, decline=decline, slope=slope, threshold=threshold
        )
        return MetricDrift(
            metric=key,
            history_runs=len(points),
            latest=current.mean,
            moving_average=average,
            decline=decline,
            slope_per_run=slope,
            threshold=threshold,
            decaying=reason is not None,
            reason=reason,
        )

    def _reason(
        self,
        key: str,
        series: Sequence[float],
        *,
        average: float,
        decline: float,
        slope: float,
        threshold: float | None,
    ) -> str | None:
        runs = len(series) - 1
        latest = series[-1]
        if decline > self.max_decline + _EPSILON:
            return (
                f"{key} {latest:.3f} is {decline:.3f} below its {runs}-run moving average "
                f"{average:.3f} (limit {self.max_decline:.3f})."
            )
        falling = all(b < a - _EPSILON for a, b in itertools.pairwise(series))
        total = series[0] - latest
        if falling and total > self.max_decline + _EPSILON:
            return (
                f"{key} fell in each of the last {runs} runs, {total:.3f} in total "
                f"({series[0]:.3f} → {latest:.3f})."
            )
        if threshold is not None and slope < -_EPSILON:
            headroom = latest - threshold
            if 0.0 <= headroom < self.near_threshold_margin:
                return (
                    f"{key} {latest:.3f} clears its {threshold:.2f} gate by only {headroom:.3f} "
                    f"while trending {slope:+.4f}/run."
                )
        return None
