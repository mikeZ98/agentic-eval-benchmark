"""Static checks of the GitHub Actions workflow: structure, pinning, permissions, scripts."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PINNED = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")
TAG_COMMENT = re.compile(r"uses: \S+@[0-9a-f]{40} # v\d+\.\d+\.\d+$")
EXPRESSION = re.compile(r"\$\{\{.*?\}\}")


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    yaml = pytest.importorskip("yaml")
    document: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return document


def steps(workflow: dict[Any, Any]) -> list[tuple[str, dict[Any, Any]]]:
    return [(name, step) for name, job in workflow["jobs"].items() for step in job["steps"]]


def test_triggers_and_concurrency(workflow: dict[Any, Any]) -> None:
    triggers = workflow[True]  # YAML 1.1 parses the bare key `on` as a boolean
    assert triggers == {"push": {"branches": ["main"]}, "pull_request": {"branches": ["main"]}}
    concurrency = workflow["concurrency"]
    assert "github.event.pull_request.number || github.ref" in concurrency["group"]
    assert concurrency["cancel-in-progress"] == "${{ github.ref != 'refs/heads/main' }}"


def test_least_privilege(workflow: dict[Any, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in workflow["jobs"].values())
    for _, step in steps(workflow):
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] is False


def test_every_action_is_pinned_to_a_commit_sha(workflow: dict[Any, Any]) -> None:
    uses = [step["uses"] for _, step in steps(workflow) if "uses" in step]
    assert uses
    assert all(PINNED.match(action) for action in uses), uses
    lines = [line.strip() for line in WORKFLOW.read_text(encoding="utf-8").splitlines()]
    pinned_lines = [line for line in lines if line.startswith(("uses:", "- uses:"))]
    assert all(TAG_COMMENT.search(line) for line in pinned_lines), pinned_lines


def test_job_graph(workflow: dict[Any, Any]) -> None:
    jobs = workflow["jobs"]
    assert set(jobs) == {
        "attribution-audit",
        "lint-and-typecheck",
        "security-audit",
        "test-hermetic",
        "eval-benchmark-gate",
        "docker-build",
    }
    for name in ("lint-and-typecheck", "security-audit", "test-hermetic", "eval-benchmark-gate"):
        assert jobs[name]["needs"] == "attribution-audit"
    assert jobs["docker-build"]["needs"] == ["lint-and-typecheck", "test-hermetic"]
    assert all("timeout-minutes" in job for job in jobs.values())


def test_attribution_audit_sees_the_full_branch_head(workflow: dict[Any, Any]) -> None:
    checkout, audit = workflow["jobs"]["attribution-audit"]["steps"]
    assert checkout["with"]["fetch-depth"] == 0
    assert "pull_request.head.sha" in checkout["with"]["ref"]
    assert audit["run"] == "scripts/audit-attribution.sh HEAD"


def run_scripts(workflow: dict[Any, Any], job: str) -> str:
    return "\n".join(step.get("run", "") for step in workflow["jobs"][job]["steps"])


def test_quality_commands(workflow: dict[Any, Any]) -> None:
    lint = run_scripts(workflow, "lint-and-typecheck")
    assert "ruff check --output-format=github ." in lint
    assert "ruff format --check ." in lint
    assert "mypy --strict src tests" in lint
    security = workflow["jobs"]["security-audit"]["steps"]
    trivy = next(s for s in security if s.get("uses", "").startswith("aquasecurity/trivy-action@"))
    assert trivy["with"]["severity"] == "CRITICAL,HIGH"
    assert trivy["with"]["exit-code"] == "1"
    assert trivy["with"]["skip-dirs"] == "tests"
    assert "bandit -c pyproject.toml -r src" in run_scripts(workflow, "security-audit")
    tests = run_scripts(workflow, "test-hermetic")
    assert "--cov=src/agentic_eval_benchmark --cov-branch --cov-fail-under=90" in tests


def test_evaluation_gate_blocks_the_pipeline(workflow: dict[Any, Any]) -> None:
    gate = run_scripts(workflow, "eval-benchmark-gate")
    assert "agentic-eval run --oracle" in gate
    assert "--format markdown" in gate
    assert 'agentic-eval gate "$RUN_ID" --step-summary "$GITHUB_STEP_SUMMARY"' in gate
    assert gate.count("uv run --no-dev agentic-eval") == 2  # gate the production install
    assert "continue-on-error" not in str(workflow["jobs"]["eval-benchmark-gate"])


def test_docker_job_checks_the_hardened_runtime(workflow: dict[Any, Any]) -> None:
    docker = run_scripts(workflow, "docker-build")
    for fragment in (
        '"10001:10001"',
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges:true",
        '"apt-get"',
        '"uv"',
        '"pip"',
        "/v1/eval/health",
        'test "$status" = "healthy"',
    ):
        assert fragment in docker, fragment


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_run_scripts_are_valid_bash(workflow: dict[Any, Any]) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    for job, step in steps(workflow):
        if "run" not in step:
            continue
        script = EXPRESSION.sub("EXPR", step["run"])
        result = subprocess.run(  # noqa: S603 - fixed interpreter, script from the repo
            [bash, "-n", "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, (job, step.get("name"), result.stderr)
