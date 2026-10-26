"""Immutable, strictly validated data contracts for the evaluation benchmark."""

from agentic_eval_benchmark.schemas._base import (
    EvalModel,
    Identifier,
    Metadata,
    NonEmptyText,
    Record,
    Scalar,
    SemVer,
    UnitInterval,
    write_parquet,
)
from agentic_eval_benchmark.schemas.benchmark import (
    BLUEPRINT_FAITHFULNESS_MIN,
    BLUEPRINT_REGRESSION_TOLERANCE,
    BLUEPRINT_RELEVANCY_MIN,
    BenchmarkReport,
    BenchmarkRunSummary,
    MetricAggregate,
    RegressionStatus,
    ThresholdPolicy,
)
from agentic_eval_benchmark.schemas.dataset import (
    GOLDEN_DATASET_SCHEMA_VERSION,
    GoldenDataset,
    GoldenSample,
)
from agentic_eval_benchmark.schemas.metrics import (
    MetricScore,
    MetricType,
    SampleEvaluation,
    metric_key,
)

__all__ = [
    "BLUEPRINT_FAITHFULNESS_MIN",
    "BLUEPRINT_REGRESSION_TOLERANCE",
    "BLUEPRINT_RELEVANCY_MIN",
    "GOLDEN_DATASET_SCHEMA_VERSION",
    "BenchmarkReport",
    "BenchmarkRunSummary",
    "EvalModel",
    "GoldenDataset",
    "GoldenSample",
    "Identifier",
    "Metadata",
    "MetricAggregate",
    "MetricScore",
    "MetricType",
    "NonEmptyText",
    "Record",
    "RegressionStatus",
    "SampleEvaluation",
    "Scalar",
    "SemVer",
    "ThresholdPolicy",
    "UnitInterval",
    "metric_key",
    "write_parquet",
]
