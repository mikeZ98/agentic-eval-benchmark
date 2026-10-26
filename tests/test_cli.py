"""Typer CLI: run, report, verify, gate, serve; stdout/stderr split and exit codes 0-3."""

import json
import runpy
import sys
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest
import uvicorn
from typer.testing import CliRunner, Result

from agentic_eval_benchmark.api.services import EvaluationServices
from agentic_eval_benchmark.cli import app
from agentic_eval_benchmark.core import write_run
from conftest import MakeRun, RagRecorder

runner = CliRunner()


def invoke(*args: str, env: dict[str, str] | None = None) -> Result:
    return runner.invoke(app, list(args), env=env)


def summary_of(result: Result) -> dict[str, object]:
    document: dict[str, object] = json.loads(result.stdout)
    return document


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_oracle_run_passes_and_persists(tmp_path: Path) -> None:
    result = invoke("run", "--oracle", "--run-id", "cli-oracle", "--format", "json")
    assert result.exit_code == 0, result.output
    assert summary_of(result)["gate_passed"] is True
    assert "Evaluating synthetic-rag-reference" in result.stderr
    assert "quality gate passed" in result.stderr
    assert (tmp_path / "runs" / "cli-oracle" / "report.json").is_file()


@pytest.mark.parametrize(
    ("fmt", "marker"), [("table", "cli-fmt"), ("markdown", "# Benchmark run `cli-fmt`")]
)
def test_run_output_formats(fmt: str, marker: str) -> None:
    result = invoke("run", "--oracle", "--run-id", "cli-fmt", "--format", fmt)
    assert result.exit_code == 0
    assert marker in result.stdout


def test_run_threshold_and_policy_options() -> None:
    raised = invoke(
        "run",
        "--oracle",
        "--threshold-faithfulness",
        "0.99",
        "--threshold-relevancy",
        "0.99",
        "--format",
        "json",
    )
    assert raised.exit_code in {0, 1}
    assert summary_of(raised)["policy"]["faithfulness_min"] == 0.99  # type: ignore[index]
    lax = invoke(
        "run",
        "--oracle",
        "--non-strict",
        "--threshold-faithfulness",
        "0.5",
        "--regression-tolerance",
        "0.05",
        "--concurrency",
        "2",
    )
    assert lax.exit_code == 0, lax.output


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (("--oracle", "--threshold-faithfulness", "0.5"), "faithfulness_min"),
        (("--oracle", "--rag-url", "http://x"), "--oracle does not call"),
        (("--oracle", "--baseline", "missing-run"), "not found"),
        (("--oracle", "--judge-model", "ragas"), "EVAL_PROVIDER_LLM_MODEL"),
        (("--judge-model", "nope", "--oracle"), "nope"),
    ],
)
def test_run_input_errors_exit_2(args: tuple[str, ...], fragment: str) -> None:
    result = invoke("run", *args)
    assert result.exit_code == 2
    assert fragment in result.output


def test_run_rejects_bad_dataset_and_settings(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('[{"question": "Hi", "contexts": ["a"], "ground_truth": "b"}]')
    result = invoke("run", "--oracle", "--dataset-path", str(bad))
    assert result.exit_code == 2
    assert "sample 1" in result.stderr
    env = invoke("run", "--oracle", env={"EVAL_CONCURRENCY_LIMIT": "0"})
    assert env.exit_code == 2
    assert "concurrency_limit" in env.stderr


def test_duplicate_run_id_exits_2() -> None:
    assert invoke("run", "--oracle", "--run-id", "dup").exit_code == 0
    duplicate = invoke("run", "--oracle", "--run-id", "dup")
    assert duplicate.exit_code == 2
    assert "already exists" in duplicate.stderr


def test_unreachable_rag_fails_closed_and_regresses(mock_rag: RagRecorder) -> None:
    assert invoke("run", "--oracle", "--run-id", "good").exit_code == 0
    down = invoke(
        "run",
        "--rag-url",
        "http://rag.test/answer",
        "--run-id",
        "down",
        "--format",
        "json",
        env={"EVAL_RETRY_ATTEMPTS": "0"},
    )
    assert down.exit_code == 1
    assert summary_of(down)["passed_samples"] == 0
    assert "8 sample(s) could not be evaluated" in down.stderr
    regressed = invoke(
        "run",
        "--rag-url",
        "http://rag.test/answer",
        "--baseline",
        "good",
        "--format",
        "json",
        env={"EVAL_RETRY_ATTEMPTS": "0"},
    )
    assert regressed.exit_code == 1
    assert summary_of(regressed)["regression_status"] == "regressed"
    improved = invoke("run", "--oracle", "--baseline", "down", "--format", "json")
    assert improved.exit_code == 0
    assert summary_of(improved)["regression_status"] == "improved"
    assert mock_rag.requests
    assert mock_rag.requests[0].url.host == "rag.test"


def test_run_answers_from_a_real_endpoint(mock_rag: RagRecorder) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"answer": "No idea.", "sources": [{"preview": "unrelated"}]}
        )

    mock_rag.handler = answer
    result = invoke("run", "--rag-url", "http://rag.test/answer", "--format", "json")
    assert result.exit_code == 1
    assert summary_of(result)["gate_passed"] is False


def test_internal_errors_exit_3(monkeypatch: pytest.MonkeyPatch) -> None:
    async def explode(self: EvaluationServices, plan: object) -> object:
        raise RuntimeError("SECRET")

    monkeypatch.setattr(EvaluationServices, "execute", explode)
    result = invoke("run", "--oracle")
    assert result.exit_code == 3
    assert "internal error (RuntimeError)" in result.stderr
    assert "SECRET" not in result.output


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_formats_and_exports(tmp_path: Path, mock_rag: RagRecorder) -> None:
    invoke(
        "run",
        "--rag-url",
        "http://rag.test/answer",
        "--run-id",
        "down",
        env={"EVAL_RETRY_ATTEMPTS": "0"},
    )
    markdown = invoke("report", "down")
    assert markdown.exit_code == 0
    assert "## Failed samples" in markdown.stdout
    assert "Not evaluated" in markdown.stdout
    parquet = tmp_path / "exports" / "r.parquet"
    by_path = invoke(
        "report", str(tmp_path / "runs" / "down"), "--format", "json", "--export", str(parquet)
    )
    assert by_path.exit_code == 0
    assert "evaluations" in json.loads(by_path.stdout)
    assert pq.read_table(parquet).num_rows == 32
    exported = tmp_path / "exports" / "r.json"
    table = invoke(
        "report",
        str(tmp_path / "runs" / "down" / "report.json"),
        "--export",
        str(exported),
        "--format",
        "table",
    )
    assert table.exit_code == 0
    assert json.loads(exported.read_text())["summary"]["run_id"] == "down"


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (("nope",), "not found"),
        (("../../etc",), "run_id must match"),
        (("down", "--export", "x.csv"), ".json or .parquet"),
    ],
)
def test_report_errors_exit_2(
    args: tuple[str, ...], fragment: str, make_run: MakeRun, tmp_path: Path
) -> None:
    write_run(make_run("down", [0.9]), tmp_path / "runs")
    result = invoke("report", *args)
    assert result.exit_code == 2
    assert fragment in result.stderr


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_offline_passes() -> None:
    result = invoke("verify", "--skip-rag")
    assert result.exit_code == 0, result.output
    assert "FAIL" not in result.stdout
    assert "8 samples" in result.stdout


def test_verify_reports_each_failure(mock_rag: RagRecorder, tmp_path: Path) -> None:
    unreachable = invoke("verify", "--rag-url", "http://rag.test/answer")
    assert unreachable.exit_code == 2
    assert "rag target" in unreachable.stdout
    mock_rag.handler = lambda request: httpx.Response(200, json={"answer": "x", "sources": []})
    assert invoke("verify", "--rag-url", "http://rag.test/answer").exit_code == 0
    unconfigured = invoke("verify", "--skip-rag", "--judge-model", "trulens")
    assert unconfigured.exit_code == 2
    assert "EVAL_PROVIDER" in unconfigured.stdout
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    broken = invoke("verify", "--dataset-path", str(bad))
    assert broken.exit_code == 2
    assert "invalid JSON" in broken.stdout
    assert "no dataset" in broken.stdout
    blocker = tmp_path / "file"
    blocker.write_text("x")
    storage = invoke("verify", "--skip-rag", env={"EVAL_OUTPUT_DIR": str(blocker / "runs")})
    assert storage.exit_code == 2
    assert "PermissionError" in storage.stdout


def test_verify_configured_provider_judges() -> None:
    ragas = {
        "EVAL_PROVIDER_LLM_MODEL": "m",
        "EVAL_PROVIDER_API_KEY": "sk-test",
        "EVAL_PROVIDER_EMBEDDINGS": "hashing",
    }
    assert invoke("verify", "--skip-rag", "--judge-model", "ragas", env=ragas).exit_code == 0
    trulens = {"EVAL_PROVIDER_LLM_MODEL": "m", "EVAL_PROVIDER_BASE_URL": "http://127.0.0.1:9/v1"}
    assert invoke("verify", "--skip-rag", "--judge-model", "trulens", env=trulens).exit_code == 0


def test_verify_flags_missing_provider_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util  # noqa: PLC0415

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "trulens" else real(name, *a)
    )
    result = invoke("verify", "--skip-rag")
    assert result.exit_code == 2
    assert "not importable: trulens" in result.stdout


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


@pytest.fixture
def stored_runs(tmp_path: Path, make_run: MakeRun) -> Path:
    store = tmp_path / "gate-runs"
    for day in range(3):
        write_run(make_run(f"hist-{day}", [0.95, 0.95], day=day), store)
    write_run(make_run("good", [0.95, 0.95], day=10), store)
    write_run(make_run("decayed", [0.90, 0.90], day=11), store)
    write_run(make_run("worse", [0.80, 0.80], day=12), store)
    write_run(make_run("other-ds", [0.95, 0.95], day=13, name="zzz"), store)
    return store


def test_gate_exit_codes(stored_runs: Path, tmp_path: Path) -> None:
    store = ("--output-dir", str(stored_runs))
    passed = invoke("gate", "good", *store)
    assert passed.exit_code == 0
    assert "quality gate passed" in passed.stdout
    drift = invoke("gate", "decayed", *store)
    assert drift.exit_code == 1
    assert "metric_drift" in drift.stdout
    assert invoke("gate", "decayed", *store, "--no-drift").exit_code == 0
    regression = invoke(
        "gate", "decayed", *store, "--no-drift", "--baseline", "good", "--format", "json"
    )
    assert regression.exit_code == 1
    assert json.loads(regression.stdout)["metric_deltas"][0]["regressed"]
    summary_file = tmp_path / "summary.md"
    worse = invoke(
        "gate", "worse", *store, "--format", "markdown", "--step-summary", str(summary_file)
    )
    assert worse.exit_code == 1
    assert "How to fix" in worse.stdout
    assert summary_file.is_file()
    assert invoke("gate", "good", *store, "--baseline", "nope").exit_code == 2
    mismatch = invoke("gate", "good", *store, "--baseline", "other-ds")
    assert mismatch.exit_code == 2
    assert "evaluated dataset" in mismatch.stderr


def test_gate_annotations_in_github_actions(stored_runs: Path, tmp_path: Path) -> None:
    summary = tmp_path / "step.md"
    result = invoke(
        "gate", "worse", "--output-dir", str(stored_runs), "--no-drift",
        env={"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(summary)},
    )  # fmt: skip
    assert result.exit_code == 1
    assert "::error title=Quality gate%3A metric_below_threshold::" in result.stdout
    assert "❌ Quality gate FAILED" in summary.read_text()


def test_run_writes_the_step_summary(tmp_path: Path) -> None:
    summary = tmp_path / "run-summary.md"
    result = invoke("run", "--oracle", env={"GITHUB_STEP_SUMMARY": str(summary)})
    assert result.exit_code == 0
    assert "✅ Quality gate passed" in summary.read_text()


# ---------------------------------------------------------------------------
# serve / entry points
# ---------------------------------------------------------------------------


def test_serve_starts_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    result = invoke("serve", "--port", "9999")
    assert result.exit_code == 0
    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 9999,
            "access_log": False,
            "server_header": False,
            "timeout_graceful_shutdown": 10,
        }
    ]


def test_help_and_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    result = invoke("--help")
    assert all(command in result.stdout for command in ("run", "report", "verify", "gate", "serve"))
    monkeypatch.setattr(sys, "argv", ["agentic-eval", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("agentic_eval_benchmark", run_name="__main__")
    assert exit_info.value.code == 0
