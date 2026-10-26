"""Judge metric contracts: per-metric scores and the per-sample verdict."""

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from agentic_eval_benchmark.schemas._base import (
    EvalModel,
    Identifier,
    NonEmptyText,
    NonNegativeFloat,
    Record,
    UnitInterval,
    write_parquet,
)


class MetricType(StrEnum):
    """Evaluation metrics; values match the Ragas metric names."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    ASPECT_CRITIQUE = "aspect_critique"


def metric_key(metric: MetricType, aspect: str | None = None) -> str:
    """Stable column-safe key: ``"faithfulness"`` or ``"aspect_critique:<aspect>"``."""
    return metric.value if aspect is None else f"{metric.value}:{aspect}"


def check_aspect(metric: MetricType, aspect: str | None) -> None:
    """Require ``aspect`` for ``ASPECT_CRITIQUE`` and forbid it for every other metric."""
    if metric is MetricType.ASPECT_CRITIQUE and aspect is None:
        raise ValueError("aspect is required for aspect_critique")
    if metric is not MetricType.ASPECT_CRITIQUE and aspect is not None:
        raise ValueError(f"aspect is only valid for aspect_critique, not {metric.value}")


class MetricScore(EvalModel):
    """One judge verdict for one metric on one sample.

    ``ASPECT_CRITIQUE`` names the critiqued ``aspect`` (e.g. ``harmfulness``) and every
    other metric must leave it unset. When ``threshold`` is given, ``passed`` must
    agree with ``score >= threshold`` so a stored verdict can never contradict its score.
    """

    metric: MetricType
    aspect: Identifier | None = None
    score: UnitInterval
    threshold: UnitInterval | None = None
    passed: bool
    reasoning: NonEmptyText
    cost_usd: NonNegativeFloat | None = None
    latency_ms: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        check_aspect(self.metric, self.aspect)
        if self.threshold is not None and self.passed != (self.score >= self.threshold):
            raise ValueError(
                f"passed={self.passed} contradicts score={self.score} "
                f"against threshold={self.threshold}"
            )
        return self

    @property
    def key(self) -> str:
        """See :func:`metric_key`."""
        return metric_key(self.metric, self.aspect)


class SampleEvaluation(EvalModel):
    """All metric verdicts for one :class:`~agentic_eval_benchmark.schemas.GoldenSample`.

    ``passed`` is the sample verdict and must equal ``all(m.passed for m in metrics)``.
    Each metric (and each critique aspect) appears at most once.
    """

    sample_id: Identifier
    answer: str | None = Field(default=None, max_length=32_768)
    metrics: tuple[MetricScore, ...] = Field(min_length=1)
    passed: bool

    @model_validator(mode="after")
    def _check_metrics(self) -> Self:
        keys = [m.key for m in self.metrics]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"duplicate metrics for sample {self.sample_id}: {duplicates}")
        expected = all(m.passed for m in self.metrics)
        if self.passed != expected:
            raise ValueError(f"passed={self.passed} contradicts metric verdicts (all={expected})")
        return self

    @property
    def failed_metrics(self) -> tuple[MetricScore, ...]:
        """Metrics whose verdict is a failure."""
        return tuple(m for m in self.metrics if not m.passed)

    @property
    def total_cost_usd(self) -> float | None:
        """Summed judge cost, or ``None`` when no metric reported a cost."""
        costs = [m.cost_usd for m in self.metrics if m.cost_usd is not None]
        return sum(costs) if costs else None

    def score_for(self, metric: MetricType, aspect: str | None = None) -> MetricScore | None:
        """Return the verdict for ``metric`` (and ``aspect``), or ``None`` if not evaluated."""
        key = metric_key(metric, aspect)
        return next((m for m in self.metrics if m.key == key), None)

    def to_records(self) -> list[Record]:
        """One flat row per metric, suitable for Parquet / DataFrame export."""
        return [
            {
                "sample_id": self.sample_id,
                "sample_passed": self.passed,
                "metric": m.metric.value,
                "aspect": m.aspect,
                "score": m.score,
                "threshold": m.threshold,
                "passed": m.passed,
                "reasoning": m.reasoning,
                "cost_usd": m.cost_usd,
                "latency_ms": m.latency_ms,
            }
            for m in self.metrics
        ]

    def to_parquet(self, path: Path) -> Path:
        """Write :meth:`to_records` to ``path`` and return it."""
        return write_parquet(self.to_records(), path)
