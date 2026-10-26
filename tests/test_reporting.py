"""Markdown and rich-table renderings of a run."""

from rich.console import Console

from agentic_eval_benchmark.reporting import render_markdown, render_table
from agentic_eval_benchmark.schemas import ThresholdPolicy
from conftest import MakeRun


def test_markdown_for_a_passing_run_with_cost(make_run: MakeRun) -> None:
    run = make_run("ok", [0.95, 0.95], cost=0.001)
    text = render_markdown(run.summary, evaluations=run.report.evaluations)
    assert "# Benchmark run `ok`" in text
    assert "✅ PASSED" in text
    assert "| Judge cost | $0.0040 |" in text
    assert "| `faithfulness` | 0.950 | 0.950 | 0.950 | 100.0% | 0.85 | ok |" in text
    assert "Failed samples" not in text


def test_markdown_lists_failures_regressions_and_escapes(make_run: MakeRun) -> None:
    base = make_run("base", [0.99] * 30).summary
    run = make_run("bad", [0.1] * 30, baseline=base)
    text = render_markdown(run.summary, evaluations=run.report.evaluations)
    assert "❌ FAILED" in text
    assert "regressed vs `base` (faithfulness" in text
    assert "| regressed |" in text
    assert "_… and 5 more failed samples._" in text
    ids_only = render_markdown(run.summary)
    assert "**Failed samples:** `s0`" in ids_only


def test_informational_metrics_and_table(make_run: MakeRun) -> None:
    run = make_run(
        "lax",
        [0.6, 0.6],
        policy=ThresholdPolicy(strict_mode=False, faithfulness_min=0.5, relevancy_min=0.5),
    )
    text = render_markdown(run.summary)
    assert "| 0.50 | ok |" in text
    console = Console(record=True, width=160)
    console.print(render_table(run.summary))
    assert "gate PASSED" in console.export_text()
    failing = make_run("f", [0.1, 0.1])
    console.print(render_table(failing.summary))
    assert "fail" in console.export_text()
