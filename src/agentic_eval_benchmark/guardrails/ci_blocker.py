"""CI blocker: fail the pipeline on a quality-gate violation, with actionable output.

:func:`enforce_gate` evaluates a run, appends a Markdown report to
``$GITHUB_STEP_SUMMARY`` and emits ``::error`` annotations when running in GitHub
Actions, then raises :class:`QualityGateFailureError` on failure. The ``agentic-eval
gate`` command turns that into exit code 1, which halts the workflow or blocks the PR.
"""

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TextIO

from agentic_eval_benchmark.guardrails.drift import DriftReport
from agentic_eval_benchmark.guardrails.gates import (
    GateResult,
    QualityGate,
    ViolationCode,
    strictest_policy,
)
from agentic_eval_benchmark.schemas import BenchmarkRunSummary, SampleEvaluation

_MAX_SAMPLES: Final = 25
_MAX_REASON: Final = 160

_HINTS: Final[dict[ViolationCode, str]] = {
    ViolationCode.METRIC_BELOW_THRESHOLD: (
        "Read the failing samples' judge reasoning. Low faithfulness means answers make "
        "claims the retrieved context does not support (tighten the generation prompt, "
        "lower temperature, cite sources); low answer relevancy means answers drift from "
        "the question or hedge."
    ),
    ViolationCode.METRIC_MISSING: (
        "The judge returned no score for a gated metric: check the judge configuration "
        "and errors.json in the run directory."
    ),
    ViolationCode.SAMPLE_FAILURES: (
        "Strict mode fails on any failed sample. Fix the samples listed below, or run "
        "with --non-strict to gate on means only for exploratory work."
    ),
    ViolationCode.REGRESSION: (
        "Diff this change against the baseline run (prompts, retriever settings, model "
        "version). If the drop is intended, refresh the baseline in the same PR."
    ),
    ViolationCode.METRIC_DRIFT: (
        "Scores are sliding across recent runs even though each one passed. Compare "
        "with the oldest run in the window to find the change that started the slide."
    ),
}


class QualityGateFailureError(Exception):
    """The quality gate failed; carries the result and its renderings."""

    def __init__(self, result: GateResult, *, markdown: str, ansi: str) -> None:
        reasons = "; ".join(v.message for v in result.violations)
        super().__init__(f"quality gate failed for {result.run_id}: {reasons}")
        self.result = result
        self.markdown = markdown
        self.ansi = ansi


def _cell(text: str) -> str:
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= _MAX_REASON else flat[: _MAX_REASON - 1] + "…"


def render_gate_markdown(
    result: GateResult,
    summary: BenchmarkRunSummary,
    *,
    evaluations: Sequence[SampleEvaluation] = (),
    drift: DriftReport | None = None,
) -> str:
    """GitHub-flavoured Markdown: verdict, violations with hints, metrics, failing samples."""
    icon, verdict = ("✅", "passed") if result.passed else ("❌", "FAILED")
    lines = [
        f"## {icon} Quality gate {verdict}: `{summary.run_id}`",
        "",
        f"Dataset `{summary.dataset_name}` v{summary.dataset_version} · "
        f"{summary.passed_samples}/{summary.total_samples} samples passed · "
        f"strict_mode={result.policy.strict_mode} · tolerance Δ {result.tolerance_delta:.3f}"
        + (f" · baseline `{result.baseline_run_id}`" if result.baseline_run_id else ""),
        "",
    ]
    if result.violations:
        lines += ["### Violations", ""]
        lines += [f"- **{v.code.value}**: {v.message}" for v in result.violations]
        lines += ["", "### How to fix", ""]
        codes = list(dict.fromkeys(v.code for v in result.violations))
        lines += [f"- **{c.value}**: {_HINTS[c]}" for c in codes]
        lines.append("")
    deltas = {d.metric: d for d in result.metric_deltas}
    drifts = {m.metric: m for m in drift.metrics} if drift else {}
    lines += [
        "### Metrics",
        "",
        "| Metric | Mean | Threshold | Baseline | Δ | Trend | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    failing = {v.metric for v in result.violations if v.metric}
    for m in summary.metrics:
        threshold = None if m.aspect else result.policy.threshold_for(m.metric)
        delta = deltas.get(m.key)
        trend = drifts.get(m.key)
        slope = None if trend is None else trend.slope_per_run
        lines.append(
            f"| `{m.key}` | {m.mean:.3f} | {'-' if threshold is None else f'{threshold:.2f}'} | "
            f"{'-' if delta is None else f'{delta.baseline:.3f}'} | "
            f"{'-' if delta is None else f'{delta.delta:+.3f}'} | "
            f"{'-' if slope is None else f'{slope:+.4f}'} | "
            f"{'❌' if m.key in failing else '✅'} |"
        )
    judge = QualityGate(result.policy, tolerance_delta=result.tolerance_delta)
    failing_ids = set(result.failing_sample_ids)
    failed = [e for e in evaluations if e.sample_id in failing_ids]
    if failed:
        lines += ["", "### Failing samples", "", "| Sample | Metric | Score | Reasoning |"]
        lines.append("|---|---|---:|---|")
        for evaluation in failed[:_MAX_SAMPLES]:
            for score in judge.failed_metrics(evaluation):
                lines.append(
                    f"| `{evaluation.sample_id}` | `{score.key}` | {score.score:.3f} | "
                    f"{_cell(score.reasoning)} |"
                )
    elif result.failing_sample_ids:
        listed = ", ".join(f"`{s}`" for s in result.failing_sample_ids[:_MAX_SAMPLES])
        lines += ["", f"**Failing samples:** {listed}"]
    if len(result.failing_sample_ids) > _MAX_SAMPLES:
        lines.append(f"\n_… {len(result.failing_sample_ids) - _MAX_SAMPLES} more not shown._")
    if result.notes:
        lines += ["", *[f"> {note}" for note in result.notes]]
    return "\n".join(lines) + "\n"


def _use_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return os.environ.get("FORCE_COLOR") is not None or stream.isatty()


def render_gate_ansi(result: GateResult, *, color: bool = True) -> str:
    """Terminal rendering: one line per violation, colored unless ``color`` is False."""

    def paint(code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    if result.passed:
        return paint("1;32", f"✔ quality gate passed: {result.run_id}") + "\n"
    lines = [paint("1;31", f"✖ quality gate FAILED: {result.run_id}")]
    lines += [f"  {paint('31', '•')} [{v.code.value}] {v.message}" for v in result.violations]
    if result.failing_sample_ids:
        shown = ", ".join(result.failing_sample_ids[:10])
        more = len(result.failing_sample_ids) - 10
        lines.append(f"  failing samples: {shown}" + (f" (+{more} more)" if more > 0 else ""))
    return "\n".join(lines) + "\n"


def _escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def github_annotations(result: GateResult) -> list[str]:
    """``::error`` workflow commands, one per violation (escaped per the GitHub spec)."""
    return [
        f"::error title={_escape_property(f'Quality gate: {v.code.value}')}::"
        f"{_escape_data(v.message)}"
        for v in result.violations
    ]


def in_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def write_step_summary(markdown: str, path: Path | None = None) -> Path | None:
    """Append ``markdown`` to ``path`` or ``$GITHUB_STEP_SUMMARY``; no-op when neither is set."""
    target = path or (Path(v) if (v := os.environ.get("GITHUB_STEP_SUMMARY")) else None)
    if target is None:
        return None
    with target.open("a", encoding="utf-8") as handle:
        handle.write(markdown if markdown.endswith("\n") else markdown + "\n")
    return target


def enforce_gate(
    summary: BenchmarkRunSummary,
    baseline: BenchmarkRunSummary | None = None,
    *,
    gate: QualityGate | None = None,
    evaluations: Sequence[SampleEvaluation] = (),
    drift: DriftReport | None = None,
    step_summary: Path | None = None,
    stream: TextIO | None = None,
) -> None:
    """Gate ``summary`` and raise :class:`QualityGateFailureError` unless it passes.

    The default gate uses the *stricter* of the blueprint policy and the run's own
    policy, so CI is never laxer than the run itself. The Markdown report goes to
    ``step_summary`` or ``$GITHUB_STEP_SUMMARY`` when set; under GitHub Actions every
    violation is also printed to ``stream`` (stdout) as an ``::error`` annotation.
    """
    out = stream or sys.stdout
    gate = gate or QualityGate(strictest_policy(summary.policy))
    result = gate.evaluate(summary, baseline, drift=drift, evaluations=evaluations)
    markdown = render_gate_markdown(result, summary, evaluations=evaluations, drift=drift)
    write_step_summary(markdown, step_summary)
    if in_github_actions():
        for line in github_annotations(result):
            out.write(line + "\n")
    if not result.passed:
        raise QualityGateFailureError(
            result, markdown=markdown, ansi=render_gate_ansi(result, color=_use_color(out))
        )
