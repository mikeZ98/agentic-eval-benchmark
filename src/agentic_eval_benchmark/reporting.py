"""Human-readable renderings of benchmark runs: Markdown and rich terminal tables."""

from collections.abc import Sequence
from typing import Final

from rich.table import Table

from agentic_eval_benchmark.schemas import (
    BenchmarkRunSummary,
    RegressionStatus,
    SampleEvaluation,
)

_MAX_REASON_CHARS: Final = 160
_MAX_LISTED_SAMPLES: Final = 25
_STATUS_COLORS: Final = {"ok": "green", "fail": "red", "regressed": "red"}


def _cell(text: str, limit: int = _MAX_REASON_CHARS) -> str:
    """One Markdown table cell: single line, pipes escaped, length capped."""
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _verdict(summary: BenchmarkRunSummary) -> str:
    return "PASSED" if summary.gate_passed else "FAILED"


def _regression(summary: BenchmarkRunSummary) -> str:
    if summary.regression_status is RegressionStatus.NO_BASELINE:
        return "no baseline"
    text = f"{summary.regression_status.value} vs `{summary.baseline_run_id}`"
    if summary.regressed_metrics:
        text += f" ({', '.join(summary.regressed_metrics)})"
    return text


def _threshold(summary: BenchmarkRunSummary, key: str) -> str:
    aggregate = next(m for m in summary.metrics if m.key == key)
    if aggregate.aspect is not None:
        return "-"
    threshold = summary.policy.threshold_for(aggregate.metric)
    return "-" if threshold is None else f"{threshold:.2f}"


def _metric_status(summary: BenchmarkRunSummary, key: str) -> str:
    aggregate = next(m for m in summary.metrics if m.key == key)
    threshold = None if aggregate.aspect else summary.policy.threshold_for(aggregate.metric)
    if key in summary.regressed_metrics:
        return "regressed"
    if threshold is None:
        return "info"
    return "ok" if aggregate.mean >= threshold else "fail"


def render_markdown(
    summary: BenchmarkRunSummary,
    *,
    evaluations: Sequence[SampleEvaluation] = (),
) -> str:
    """Markdown report: verdict, run facts, metric table and (optionally) failed samples."""
    icon = "✅" if summary.gate_passed else "❌"
    lines = [
        f"# Benchmark run `{summary.run_id}`",
        "",
        f"**Quality gate: {icon} {_verdict(summary)}**",
        "",
        "| | |",
        "|---|---|",
        f"| Dataset | `{summary.dataset_name}` v{summary.dataset_version} |",
        f"| Started | {summary.started_at.isoformat(timespec='seconds')} |",
        f"| Duration | {summary.duration_seconds:.2f} s |",
        f"| Samples | {summary.passed_samples}/{summary.total_samples} passed "
        f"({summary.pass_rate:.1%}) |",
        f"| Regression | {_regression(summary)} |",
        f"| Policy | strict_mode={summary.policy.strict_mode}, "
        f"regression_tolerance={summary.policy.regression_tolerance} |",
    ]
    if summary.total_cost_usd is not None:
        lines.append(f"| Judge cost | ${summary.total_cost_usd:.4f} |")
    lines += [
        "",
        "## Metrics",
        "",
        "| Metric | Mean | Min | Max | Pass rate | Threshold | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for m in summary.metrics:
        lines.append(
            f"| `{m.key}` | {m.mean:.3f} | {m.minimum:.3f} | {m.maximum:.3f} | "
            f"{m.pass_rate:.1%} | {_threshold(summary, m.key)} | {_metric_status(summary, m.key)} |"
        )
    failed = [e for e in evaluations if not e.passed]
    if failed:
        lines += ["", "## Failed samples", "", "| Sample | Metric | Score | Reasoning |"]
        lines.append("|---|---|---:|---|")
        for evaluation in failed[:_MAX_LISTED_SAMPLES]:
            for score in evaluation.failed_metrics:
                lines.append(
                    f"| `{evaluation.sample_id}` | `{score.key}` | {score.score:.3f} | "
                    f"{_cell(score.reasoning)} |"
                )
        if len(failed) > _MAX_LISTED_SAMPLES:
            lines.append(f"\n_… and {len(failed) - _MAX_LISTED_SAMPLES} more failed samples._")
    elif summary.failed_sample_ids:
        listed = ", ".join(f"`{s}`" for s in summary.failed_sample_ids[:_MAX_LISTED_SAMPLES])
        lines += ["", f"**Failed samples:** {listed}"]
    return "\n".join(lines) + "\n"


def render_table(summary: BenchmarkRunSummary) -> Table:
    """Rich table of the metric aggregates, titled with the gate verdict."""
    style = "bold green" if summary.gate_passed else "bold red"
    table = Table(
        title=f"{summary.run_id} · gate {_verdict(summary)} · "
        f"{summary.passed_samples}/{summary.total_samples} samples passed · "
        f"{_regression(summary)}",
        title_style=style,
    )
    for column, justify in (
        ("metric", "left"),
        ("mean", "right"),
        ("min", "right"),
        ("max", "right"),
        ("pass rate", "right"),
        ("threshold", "right"),
        ("status", "left"),
    ):
        table.add_column(column, justify=justify)  # type: ignore[arg-type]
    for m in summary.metrics:
        status = _metric_status(summary, m.key)
        color = _STATUS_COLORS.get(status, "dim")
        table.add_row(
            m.key,
            f"{m.mean:.3f}",
            f"{m.minimum:.3f}",
            f"{m.maximum:.3f}",
            f"{m.pass_rate:.1%}",
            _threshold(summary, m.key),
            f"[{color}]{status}[/{color}]",
        )
    return table
