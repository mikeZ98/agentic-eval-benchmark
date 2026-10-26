"""Quality gates, regression boundaries, drift tracking and the CI blocker."""

import io
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_eval_benchmark.core import write_run
from agentic_eval_benchmark.guardrails import (
    BaselineMismatchError,
    DriftTracker,
    GateResult,
    GateViolation,
    MetricDelta,
    QualityGate,
    QualityGateFailureError,
    ViolationCode,
    enforce_gate,
    github_annotations,
    in_github_actions,
    render_gate_ansi,
    render_gate_markdown,
    strictest_policy,
    write_step_summary,
)
from agentic_eval_benchmark.guardrails.ci_blocker import _use_color
from agentic_eval_benchmark.guardrails.drift import MetricDrift
from agentic_eval_benchmark.schemas import BenchmarkRunSummary, MetricType, ThresholdPolicy
from conftest import MakeRun

LAX = ThresholdPolicy(strict_mode=False, faithfulness_min=0.5, relevancy_min=0.5)


def codes(result: GateResult) -> list[ViolationCode]:
    return [v.code for v in result.violations]


# ---------------------------------------------------------------------------
# QualityGate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("faithfulness", "kwargs"),
    [
        ([0.9, 0.95], {}),
        ([0.8, 0.82], {}),
        ([0.84, 0.99], {}),
        ([0.84, 0.99], {"policy": ThresholdPolicy(strict_mode=False)}),
        ([0.6, 0.6], {"policy": LAX}),
        ([0.9, 0.9], {"drop_relevancy": True}),
        ([0.9, 0.9], {"drop_relevancy": True, "policy": ThresholdPolicy(strict_mode=False)}),
    ],
)
def test_gate_agrees_with_the_engine(
    make_run: MakeRun, faithfulness: list[float], kwargs: dict[str, object]
) -> None:
    summary = make_run("r", faithfulness, **kwargs).summary
    assert QualityGate(summary.policy).evaluate(summary).passed == summary.gate_passed


def test_gate_agrees_on_recorded_regressions(make_run: MakeRun) -> None:
    base = make_run("base", [0.95, 0.95]).summary
    regressed = make_run("r", [0.9, 0.9], baseline=base).summary
    result = QualityGate(regressed.policy).evaluate(regressed)
    assert not result.passed
    assert not regressed.gate_passed
    assert codes(result) == [ViolationCode.REGRESSION]
    assert result.baseline_run_id == "base"


def test_threshold_violations_carry_numbers(make_run: MakeRun) -> None:
    result = QualityGate().evaluate(make_run("r", [0.8, 0.82]).summary)
    assert codes(result) == [ViolationCode.METRIC_BELOW_THRESHOLD, ViolationCode.SAMPLE_FAILURES]
    below = result.violations[0]
    assert below.metric == "faithfulness"
    assert below.observed == pytest.approx(0.81)
    assert below.expected == 0.85
    assert set(result.failing_sample_ids) == {"s0", "s1"}
    missing = QualityGate().evaluate(make_run("m", [0.9, 0.9], drop_relevancy=True).summary)
    assert ViolationCode.METRIC_MISSING in codes(missing)
    lax_missing = QualityGate(ThresholdPolicy(strict_mode=False)).evaluate(
        make_run("m", [0.9, 0.9], drop_relevancy=True).summary
    )
    assert lax_missing.passed


@pytest.mark.parametrize(("current", "passes"), [(0.935, True), (0.925, False)])
def test_regression_boundary(make_run: MakeRun, current: float, passes: bool) -> None:
    base = make_run("base", [0.95, 0.95]).summary
    result = QualityGate().evaluate(make_run("cur", [current, current]).summary, base)
    delta = {d.metric: d for d in result.metric_deltas}["faithfulness"]
    assert result.passed is passes
    assert delta.delta == pytest.approx(current - 0.95)
    assert delta.regressed is not passes
    assert result.regressions == (() if passes else (delta,))
    if not passes:
        assert "dropped 0.025" in result.violations[0].message


def test_tolerance_override_and_defaults(make_run: MakeRun) -> None:
    base = make_run("base", [0.95, 0.95]).summary
    current = make_run("cur", [0.925, 0.925]).summary
    assert QualityGate(tolerance_delta=0.03).evaluate(current, base).passed
    assert QualityGate().tolerance_delta == ThresholdPolicy().regression_tolerance == 0.02
    with pytest.raises(ValueError, match="tolerance_delta"):
        QualityGate(tolerance_delta=1.5)


def test_incomparable_baselines(make_run: MakeRun) -> None:
    base = make_run("base", [0.95, 0.95]).summary
    with pytest.raises(BaselineMismatchError, match="evaluated dataset"):
        QualityGate().evaluate(make_run("x", [0.9, 0.9], name="other").summary, base)
    with pytest.raises(BaselineMismatchError, match="own baseline"):
        QualityGate().evaluate(base, base)
    upgraded = QualityGate().evaluate(make_run("v2", [0.95, 0.95], version="2.0.0").summary, base)
    assert upgraded.passed
    assert any("v1.0.0" in note for note in upgraded.notes)
    no_relevancy = QualityGate(ThresholdPolicy(strict_mode=False)).evaluate(
        make_run("n", [0.95, 0.95], drop_relevancy=True).summary, base
    )
    assert any("answer_relevancy" in note for note in no_relevancy.notes)


def test_re_gating_a_lax_run_rejudges_samples(make_run: MakeRun) -> None:
    lax = make_run("lax", [0.6, 0.6], policy=LAX)
    assert lax.summary.gate_passed
    summary_only = QualityGate().evaluate(lax.summary)
    assert not summary_only.passed
    assert summary_only.failing_sample_ids == ()
    rejudged = QualityGate().evaluate(lax.summary, evaluations=lax.report.evaluations)
    assert set(rejudged.failing_sample_ids) == {"s0", "s1"}
    assert ViolationCode.SAMPLE_FAILURES in codes(rejudged)
    strict = make_run("s", [0.84, 0.99])
    assert QualityGate().evaluate(
        strict.summary, evaluations=strict.report.evaluations
    ).failing_sample_ids == ("s0",)


def test_result_invariants() -> None:
    violation = GateViolation(code=ViolationCode.REGRESSION, message="m")
    with pytest.raises(ValidationError, match="no violations"):
        GateResult(
            run_id="z",
            passed=True,
            policy=ThresholdPolicy(),
            tolerance_delta=0.02,
            violations=(violation,),
        )
    with pytest.raises(ValidationError, match="current - baseline"):
        MetricDelta(
            metric="f", current=0.9, baseline=0.95, delta=-0.01, tolerance=0.02, regressed=False
        )
    with pytest.raises(ValidationError, match="regressed contradicts"):
        MetricDelta(
            metric="f", current=0.9, baseline=0.95, delta=-0.05, tolerance=0.02, regressed=False
        )


def test_strictest_policy() -> None:
    merged = strictest_policy(
        ThresholdPolicy(faithfulness_min=0.9, regression_tolerance=0.05, context_recall_min=0.7),
        LAX,
    )
    assert (merged.faithfulness_min, merged.relevancy_min, merged.regression_tolerance) == (
        0.9,
        0.85,
        0.02,
    )
    assert merged.context_recall_min == 0.7
    assert merged.context_precision_min is None
    assert merged.strict_mode


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def history(
    root: Path, make_run: MakeRun, means: list[float], *, start: int = 0, name: str = "demo"
) -> None:
    for offset, mean in enumerate(means):
        write_run(
            make_run(f"{name}-{start + offset:02d}", [mean, mean], day=start + offset, name=name),
            root,
        )


def faithfulness_drift(root: Path, latest: BenchmarkRunSummary, **kwargs: object) -> MetricDrift:
    report = DriftTracker(root, **kwargs).track(latest)  # type: ignore[arg-type]
    return {m.metric: m for m in report.metrics}["faithfulness"]


def test_sharp_drop_against_the_moving_average(tmp_path: Path, make_run: MakeRun) -> None:
    history(tmp_path, make_run, [0.95, 0.95, 0.94, 0.95])
    latest = make_run("latest", [0.90, 0.90], day=10).summary
    drift = faithfulness_drift(tmp_path, latest)
    assert drift.decaying
    assert "moving average" in (drift.reason or "")
    assert drift.moving_average == pytest.approx(0.9475)
    assert drift.decline == pytest.approx(0.0475)
    assert QualityGate().evaluate(latest).passed
    report = DriftTracker(tmp_path).track(latest)
    gated = QualityGate().evaluate(latest, drift=report)
    assert codes(gated) == [ViolationCode.METRIC_DRIFT]
    assert report.drifting
    assert report.decaying


def test_sustained_decline(tmp_path: Path, make_run: MakeRun) -> None:
    history(tmp_path, make_run, [0.97, 0.96, 0.95, 0.94])
    drift = faithfulness_drift(
        tmp_path, make_run("latest", [0.935, 0.935], day=10).summary, max_decline=0.03
    )
    assert drift.decaying
    assert "fell in each of the last 4 runs" in (drift.reason or "")
    assert drift.slope_per_run is not None
    assert drift.slope_per_run < 0


def test_eroding_headroom(tmp_path: Path, make_run: MakeRun) -> None:
    history(tmp_path, make_run, [0.875, 0.87, 0.872, 0.868])
    drift = faithfulness_drift(tmp_path, make_run("latest", [0.86, 0.86], day=10).summary)
    assert drift.decaying
    assert "clears its 0.85 gate by only" in (drift.reason or "")


def test_noise_and_short_history_are_not_drift(tmp_path: Path, make_run: MakeRun) -> None:
    stable = tmp_path / "stable"
    history(stable, make_run, [0.93, 0.95, 0.94, 0.95, 0.93])
    assert not faithfulness_drift(stable, make_run("latest", [0.94, 0.94], day=10).summary).decaying
    short = tmp_path / "short"
    history(short, make_run, [0.99, 0.99])
    drift = faithfulness_drift(short, make_run("latest", [0.80, 0.80], day=10).summary)
    assert not drift.decaying
    assert drift.history_runs == 2
    assert drift.moving_average == pytest.approx(0.99)
    empty = faithfulness_drift(tmp_path / "missing", make_run("latest", [0.8, 0.8]).summary)
    assert empty.moving_average is None
    assert empty.decline is None


def test_history_selection(tmp_path: Path, make_run: MakeRun) -> None:
    history(tmp_path, make_run, [0.95, 0.95, 0.95])
    history(tmp_path, make_run, [0.1, 0.1], start=20, name="other")
    history(tmp_path, make_run, [0.1, 0.1], start=30)
    write_run(make_run("old-version", [0.1, 0.1], day=5, version="0.9.0"), tmp_path)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "report.json").write_text("{not json")
    (tmp_path / "empty-dir").mkdir()
    (tmp_path / "stray.txt").write_text("x")
    latest = make_run("latest", [0.95, 0.95], day=10).summary
    report = DriftTracker(tmp_path).track(latest)
    assert report.history_run_ids == ("demo-00", "demo-01", "demo-02")
    assert report.skipped == ("broken",)
    assert not report.drifting
    all_versions = DriftTracker(tmp_path).track(latest, same_version=False)
    assert "old-version" in all_versions.history_run_ids
    assert all_versions.dataset_version is None
    window = DriftTracker(tmp_path, window=2, min_history=1).track(latest)
    assert window.history_run_ids == ("demo-01", "demo-02")


@pytest.mark.parametrize(
    "kwargs",
    [{"window": 2, "min_history": 3}, {"max_decline": 2.0}, {"near_threshold_margin": -1.0}],
)
def test_tracker_configuration(tmp_path: Path, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"min_history|must be in"):
        DriftTracker(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_drift_model_requires_a_reason() -> None:
    with pytest.raises(ValidationError, match="reason"):
        MetricDrift(metric="f", history_runs=3, latest=0.9, decaying=True)


# ---------------------------------------------------------------------------
# CI blocker
# ---------------------------------------------------------------------------


def test_enforce_gate_passes_and_writes_step_summary(
    make_run: MakeRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_file = tmp_path / "step.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    enforce_gate(make_run("ok", [0.95, 0.95]).summary)  # returns normally: no exception
    assert "✅ Quality gate passed: `ok`" in summary_file.read_text()


def test_enforce_gate_blocks_with_actionable_output(
    make_run: MakeRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_file = tmp_path / "step.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    base = make_run("base", [0.95, 0.95]).summary
    current = make_run("cur", [0.925, 0.925])
    stream = io.StringIO()
    with pytest.raises(QualityGateFailureError) as failure:
        enforce_gate(current.summary, base, evaluations=current.report.evaluations, stream=stream)
    markdown = failure.value.markdown
    assert "❌ Quality gate FAILED" in markdown
    assert "### How to fix" in markdown
    assert "| `faithfulness` | 0.925 | 0.85 | 0.950 | -0.025 | - | ❌ |" in markdown
    assert "\033[" not in failure.value.ansi
    assert "FAILED" in failure.value.ansi
    assert "dropped 0.025" in str(failure.value)
    assert stream.getvalue().startswith(
        "::error title=Quality gate%3A regression::faithfulness dropped"
    )
    assert sum(line.startswith("## ") for line in summary_file.read_text().splitlines()) == 1


def test_enforce_gate_is_never_laxer_than_the_blueprint(make_run: MakeRun) -> None:
    lax = make_run("lax", [0.6, 0.6], policy=LAX)
    with pytest.raises(QualityGateFailureError) as failure:
        enforce_gate(lax.summary, evaluations=lax.report.evaluations, stream=io.StringIO())
    assert "### Failing samples" in failure.value.markdown
    assert set(failure.value.result.failing_sample_ids) == {"s0", "s1"}
    stricter = make_run("strict", [0.88, 0.88], policy=ThresholdPolicy(faithfulness_min=0.9))
    with pytest.raises(QualityGateFailureError):
        enforce_gate(stricter.summary, stream=io.StringIO())


def test_markdown_rendering_details(make_run: MakeRun, tmp_path: Path) -> None:
    many = make_run("many", [0.1] * 30)
    result = QualityGate().evaluate(many.summary)
    listed = render_gate_markdown(result, many.summary)
    assert "**Failing samples:**" in listed
    assert "_… 5 more not shown._" in listed
    history(tmp_path, make_run, [0.95, 0.95, 0.95])
    latest = make_run("latest", [0.9, 0.9], day=10)
    drift = DriftTracker(tmp_path).track(latest.summary)
    with_trend = render_gate_markdown(
        QualityGate().evaluate(latest.summary, drift=drift), latest.summary, drift=drift
    )
    assert "metric_drift" in with_trend
    assert "| `faithfulness` | 0.900 | 0.85 | - | - | -0.0" in with_trend


def test_ansi_and_annotation_helpers(make_run: MakeRun, monkeypatch: pytest.MonkeyPatch) -> None:
    passed = QualityGate().evaluate(make_run("ok", [0.95, 0.95]).summary)
    assert "\033[1;32m" in render_gate_ansi(passed)
    failing = QualityGate().evaluate(make_run("bad", [0.1] * 12).summary)
    assert "(+2 more)" in render_gate_ansi(failing, color=False)
    weird = GateResult(
        run_id="z",
        passed=False,
        policy=ThresholdPolicy(),
        tolerance_delta=0.0,
        violations=(
            GateViolation(code=ViolationCode.REGRESSION, message="100% bad\r\nnext, more"),
        ),
    )
    assert github_annotations(weird) == [
        "::error title=Quality gate%3A regression::100%25 bad%0D%0Anext, more"
    ]
    assert not in_github_actions()
    assert write_step_summary("no target") is None
    target = write_step_summary("x", Path("explicit.md"))
    assert target is not None
    assert target.read_text() == "x\n"
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _use_color(io.StringIO())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not _use_color(io.StringIO())


def test_injected_drift_report_becomes_a_violation(make_run: MakeRun) -> None:
    latest = make_run("latest", [0.9, 0.9]).summary
    report = DriftTracker(Path("nowhere")).analyze(latest, [])
    decaying = report.model_copy(
        update={
            "metrics": (
                MetricDrift(
                    metric="faithfulness", history_runs=3, latest=0.9, decaying=True, reason="x"
                ),
            )
        }
    )
    result = QualityGate().evaluate(latest, drift=decaying)
    assert codes(result) == [ViolationCode.METRIC_DRIFT]
    assert result.violations[0].metric == "faithfulness"
    assert result.violations[0].message == "x"
    assert report.metrics[0].metric == MetricType.FAITHFULNESS.value
