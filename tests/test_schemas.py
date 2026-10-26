"""Data contracts: strictness, immutability, invariants, versioning, JSON/Parquet round-trips."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from agentic_eval_benchmark.core import EvaluationRun
from agentic_eval_benchmark.schemas import (
    BLUEPRINT_REGRESSION_TOLERANCE,
    GOLDEN_DATASET_SCHEMA_VERSION,
    BenchmarkReport,
    BenchmarkRunSummary,
    GoldenDataset,
    GoldenSample,
    MetricAggregate,
    MetricScore,
    MetricType,
    RegressionStatus,
    SampleEvaluation,
    ThresholdPolicy,
    metric_key,
    write_parquet,
)
from conftest import MakeRun, golden_sample

F, AR = MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCY


def score(**overrides: Any) -> MetricScore:
    fields: dict[str, Any] = {"metric": F, "score": 0.9, "passed": True, "reasoning": "grounded"}
    return MetricScore(**(fields | overrides))


# ---------------------------------------------------------------------------
# GoldenSample / GoldenDataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("question too short", {"question": "ab"}),
        ("blank question", {"question": "     "}),
        ("no contexts", {"contexts": ()}),
        ("list is not a tuple in strict mode", {"contexts": ["a"]}),
        ("blank context", {"contexts": ("  ",)}),
        ("nested metadata", {"metadata": {"k": {"a": 1}}}),
        ("id with a space", {"sample_id": "has space"}),
        ("id starting with a dot", {"sample_id": ".hidden"}),
        ("unknown field", {"answer": "x"}),
        ("empty ground truth", {"ground_truth": ""}),
    ],
)
def test_golden_sample_rejects(label: str, overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        golden_sample(**overrides)


def test_golden_sample_accepts_flat_metadata() -> None:
    sample = golden_sample(metadata={"source": "wiki", "n": 2, "w": 0.5, "flag": True, "x": None})
    assert sample.metadata["n"] == 2


def test_models_are_frozen(sample: GoldenSample) -> None:
    with pytest.raises(ValidationError):
        sample.question = "changed?"  # type: ignore[misc]


def dataset(*samples: GoldenSample, **overrides: Any) -> GoldenDataset:
    fields: dict[str, Any] = {"name": "demo", "dataset_version": "1.2.0", "samples": samples}
    return GoldenDataset(**(fields | overrides))


def test_dataset_requires_unique_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate sample_id values: a"):
        dataset(golden_sample("a"), golden_sample("a"), golden_sample("a"), golden_sample("b"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"samples": ()},
        {"dataset_version": "1.2"},
        {"dataset_version": "01.2.0"},
        {"schema_version": "2.0.0"},
    ],
)
def test_dataset_rejects(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        dataset(golden_sample(), **overrides)


def test_dataset_helpers_and_json_round_trip() -> None:
    data = dataset(golden_sample("a"), golden_sample("b"))
    assert data.schema_version == GOLDEN_DATASET_SCHEMA_VERSION
    assert data.sample_ids == ("a", "b")
    assert data.get("b").sample_id == "b"
    with pytest.raises(KeyError):
        data.get("missing")
    assert GoldenDataset.from_json(data.to_json(indent=2)) == data
    samples = data.to_jsonable()["samples"]
    assert isinstance(samples, list)
    assert samples[0]["contexts"] == list(data.samples[0].contexts)


# ---------------------------------------------------------------------------
# MetricScore / SampleEvaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"score": 1.01},
        {"score": -0.01},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": "0.9"},
        {"metric": "faithfulness"},
        {"reasoning": ""},
        {"cost_usd": -0.1},
        {"latency_ms": float("nan")},
        {"score": 0.8, "threshold": 0.85, "passed": True},
        {"score": 0.9, "threshold": 0.85, "passed": False},
        {"metric": MetricType.ASPECT_CRITIQUE},
        {"aspect": "harmfulness"},
    ],
)
def test_metric_score_rejects(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        score(**overrides)


def test_metric_score_bounds_and_keys() -> None:
    assert score(score=0.85, threshold=0.85, passed=True).passed
    assert score(score=0.0).score == 0.0
    assert score(score=1).score == 1.0  # strict float still accepts an int
    critique = score(metric=MetricType.ASPECT_CRITIQUE, aspect="harmfulness")
    assert critique.key == metric_key(MetricType.ASPECT_CRITIQUE, "harmfulness")
    parsed = MetricScore.from_json(
        '{"metric":"faithfulness","score":1,"passed":true,"reasoning":"r"}'
    )
    assert parsed.metric is F


def test_sample_evaluation_invariants() -> None:
    good, bad = score(), score(metric=AR, score=0.1, passed=False, reasoning="off topic")
    evaluation = SampleEvaluation(sample_id="a", metrics=(good, bad), passed=False)
    assert evaluation.failed_metrics == (bad,)
    assert evaluation.score_for(AR) == bad
    assert evaluation.score_for(MetricType.CONTEXT_RECALL) is None
    assert evaluation.total_cost_usd is None
    with pytest.raises(ValidationError, match="contradicts metric verdicts"):
        SampleEvaluation(sample_id="a", metrics=(good, bad), passed=True)
    with pytest.raises(ValidationError, match="duplicate metrics"):
        SampleEvaluation(sample_id="a", metrics=(good, good), passed=True)
    with pytest.raises(ValidationError):
        SampleEvaluation(sample_id="a", metrics=(), passed=True)
    costly = SampleEvaluation(
        sample_id="a", metrics=(score(cost_usd=0.01), score(metric=AR, cost_usd=0.02)), passed=True
    )
    assert costly.total_cost_usd == pytest.approx(0.03)


def test_two_critique_aspects_are_distinct_metrics() -> None:
    harm = score(metric=MetricType.ASPECT_CRITIQUE, aspect="harmfulness")
    bias = harm.model_copy(update={"aspect": "bias"})
    evaluation = SampleEvaluation(sample_id="a", metrics=(harm, bias), passed=True)
    assert len(evaluation.to_records()) == 2


# ---------------------------------------------------------------------------
# ThresholdPolicy / MetricAggregate / BenchmarkRunSummary / BenchmarkReport
# ---------------------------------------------------------------------------


def test_threshold_policy_blueprint_floor() -> None:
    default = ThresholdPolicy()
    assert (default.faithfulness_min, default.relevancy_min, default.strict_mode) == (
        0.85,
        0.85,
        True,
    )
    assert default.regression_tolerance == BLUEPRINT_REGRESSION_TOLERANCE == 0.02
    assert default.thresholds() == {F: 0.85, AR: 0.85}
    assert default.threshold_for(MetricType.CONTEXT_RECALL) is None
    with pytest.raises(ValidationError, match="faithfulness_min"):
        ThresholdPolicy(faithfulness_min=0.8)
    with pytest.raises(ValidationError, match="relevancy_min"):
        ThresholdPolicy(relevancy_min=0.8)
    assert ThresholdPolicy(relevancy_min=0.9).relevancy_min == 0.9
    lax = ThresholdPolicy(strict_mode=False, faithfulness_min=0.5, context_recall_min=0.7)
    assert lax.thresholds()[MetricType.CONTEXT_RECALL] == 0.7


def aggregate(**overrides: Any) -> MetricAggregate:
    fields: dict[str, Any] = {
        "metric": F, "count": 2, "mean": 0.8, "minimum": 0.7, "maximum": 0.9, "pass_rate": 0.5,
    }  # fmt: skip
    return MetricAggregate(**(fields | overrides))


def test_metric_aggregate_invariants() -> None:
    assert aggregate().key == "faithfulness"
    with pytest.raises(ValidationError, match="minimum <= mean <= maximum"):
        aggregate(mean=0.95)
    with pytest.raises(ValidationError, match="aspect"):
        aggregate(metric=MetricType.ASPECT_CRITIQUE)


SUMMARY: dict[str, Any] = {
    "run_id": "run-2",
    "dataset_name": "demo",
    "dataset_version": "1.0.0",
    "started_at": datetime(2026, 10, 26, tzinfo=UTC),
    "duration_seconds": 12.5,
    "total_samples": 2,
    "passed_samples": 1,
    "pass_rate": 0.5,
    "failed_sample_ids": ("b",),
    "policy": ThresholdPolicy(),
    "gate_passed": False,
    "regression_status": RegressionStatus.NO_BASELINE,
}


def summary(**overrides: Any) -> BenchmarkRunSummary:
    fields = SUMMARY | {
        "metrics": (
            aggregate(),
            aggregate(metric=AR, mean=0.95, minimum=0.95, maximum=0.95, pass_rate=1.0),
        )
    }
    return BenchmarkRunSummary(**(fields | overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"passed_samples": 3}, "cannot exceed"),
        ({"pass_rate": 0.6}, "does not match"),
        ({"failed_sample_ids": ()}, "failed_sample_ids for"),
        ({"failed_sample_ids": ("b", "b"), "passed_samples": 0, "pass_rate": 0.0}, "unique"),
        ({"metrics": (aggregate(), aggregate())}, "duplicate metric aggregates"),
        ({"metrics": (aggregate(count=3),)}, "counts more samples"),
        ({"baseline_run_id": "run-1"}, "baseline_run_id is required"),
        (
            {"regression_status": RegressionStatus.STABLE, "baseline_run_id": "run-2"},
            "own baseline",
        ),
        (
            {"regression_status": RegressionStatus.REGRESSED, "baseline_run_id": "run-1"},
            "regressed_metrics",
        ),
        (
            {
                "regression_status": RegressionStatus.REGRESSED,
                "baseline_run_id": "run-1",
                "regressed_metrics": ("context_recall",),
            },
            "not in metrics",
        ),
        ({"gate_passed": True}, "strict_mode"),
        ({"gate_passed": True, "policy": ThresholdPolicy(strict_mode=False)}, "mean"),
        (
            {
                "gate_passed": True,
                "failed_sample_ids": (),
                "passed_samples": 2,
                "pass_rate": 1.0,
                "metrics": (aggregate(mean=0.9, minimum=0.9, maximum=0.9),),
            },
            "missing",
        ),
        (
            {
                "gate_passed": True,
                "regression_status": RegressionStatus.REGRESSED,
                "baseline_run_id": "run-1",
                "regressed_metrics": ("faithfulness",),
            },
            "regressed run",
        ),
        ({"started_at": datetime(2026, 1, 1)}, "timezone"),
    ],
)
def test_summary_rejects_contradictions(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        summary(**overrides)


def test_summary_accessors_and_records() -> None:
    s = summary(total_cost_usd=0.5)
    assert s.finished_at.isoformat() == "2026-10-26T00:00:12.500000+00:00"
    assert s.mean_score(F) == 0.8
    assert s.mean_score(MetricType.CONTEXT_RECALL) is None
    records = s.to_records()
    assert [r["metric"] for r in records] == ["faithfulness", "answer_relevancy"]
    assert records[0]["threshold"] == 0.85
    assert records[0]["regressed"] is False
    assert BenchmarkRunSummary.from_json(s.to_json()) == s


def test_summary_and_report_parquet_round_trip(tmp_path: Path, make_run: MakeRun) -> None:
    run: EvaluationRun = make_run("run-x", [0.9, 0.7])
    summary_table = pq.read_table(run.summary.to_parquet(tmp_path / "s" / "summary.parquet"))
    report_table = pq.read_table(run.report.to_parquet(tmp_path / "samples.parquet"))
    assert summary_table.num_rows == 2
    assert "metric_pass_rate" in summary_table.column_names
    assert report_table.num_rows == 4
    assert report_table.column("run_id").to_pylist() == ["run-x"] * 4
    one = run.report.evaluations[0]
    assert pq.read_table(one.to_parquet(tmp_path / "one.parquet")).num_rows == 2
    assert BenchmarkReport.from_json(run.report.to_json()) == run.report


def test_write_parquet_requires_uniform_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_parquet([], tmp_path / "x.parquet")
    with pytest.raises(ValueError, match="record 1 has columns"):
        write_parquet([{"a": 1}, {"b": 2}], tmp_path / "x.parquet")


def test_report_must_match_summary(make_run: MakeRun) -> None:
    run = make_run("run-r", [0.9, 0.7])
    evaluations = run.report.evaluations
    mutate: list[tuple[Callable[[], object], str]] = [
        (
            lambda: BenchmarkReport(
                summary=run.summary, evaluations=(evaluations[0], evaluations[0])
            ),
            "duplicate",
        ),
        (
            lambda: BenchmarkReport(summary=run.summary, evaluations=evaluations[:1]),
            "total_samples",
        ),
        (
            lambda: BenchmarkReport(
                summary=run.summary,
                evaluations=(evaluations[0], evaluations[0].model_copy(update={"sample_id": "s1"})),
            ),
            "failed_sample_ids",
        ),
    ]
    for build, message in mutate:
        with pytest.raises(ValidationError, match=message):
            build()
    other = make_run("run-o", [0.9, 0.6])
    with pytest.raises(ValidationError, match="does not match"):
        BenchmarkReport(summary=other.summary, evaluations=evaluations)
    no_relevancy = make_run("run-n", [0.9, 0.7], drop_relevancy=True)
    with pytest.raises(ValidationError, match="evaluated metrics"):
        BenchmarkReport(summary=no_relevancy.summary, evaluations=evaluations)
