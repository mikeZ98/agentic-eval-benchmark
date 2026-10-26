"""Static container contract: hardening that must hold without a Docker daemon.

The live checks (healthy under read-only rootfs, uid 10001, no installers) run against a
real build; these keep the Dockerfile, compose spec, build context and env template from
silently regressing.
"""

import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Instructions only: comments may legitimately mention what the image must not contain.
DOCKERFILE = "\n".join(
    line
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    if not line.lstrip().startswith("#")
)
RUNTIME = DOCKERFILE.split("AS runtime", 1)[1]


def test_multi_stage_build_from_python_slim_with_frozen_uv() -> None:
    assert re.findall(r"^FROM python:\$\{PYTHON_VERSION\}-slim AS (\w+)", DOCKERFILE, re.M) == [
        "builder",
        "runtime",
    ]
    assert re.search(r"^ARG PYTHON_VERSION=3\.13$", DOCKERFILE, re.M)
    assert re.search(r"^ARG UV_VERSION=\d+\.\d+\.\d+$", DOCKERFILE, re.M)
    assert "uv sync --frozen --no-dev --no-install-project" in DOCKERFILE
    assert "uv sync --frozen --no-dev --no-editable" in DOCKERFILE
    assert "UV_PROJECT_ENVIRONMENT=/app/.venv" in DOCKERFILE


def test_runtime_stage_has_no_build_tooling() -> None:
    assert "COPY --from=uv" not in RUNTIME
    assert "COPY --from=builder /app/.venv /app/.venv" in RUNTIME
    for removed in ("/usr/local/bin/pip*", "ensurepip", "/usr/bin/apt-*", "/usr/bin/dpkg"):
        assert removed in RUNTIME
    assert "/var/lib/dpkg" not in RUNTIME  # its status file stays for image scanners
    assert not re.search(r"apt-get install|pip install|uv pip", RUNTIME)


def test_runtime_runs_as_unprivileged_appuser() -> None:
    assert "--uid 10001 --gid 10001" in RUNTIME
    assert "--shell /usr/sbin/nologin" in RUNTIME
    assert re.search(r"^USER 10001:10001$", RUNTIME, re.M)
    users = re.findall(r"^USER (.+)$", RUNTIME, re.M)
    assert users == ["10001:10001"]


def test_runtime_state_is_pinned_under_tmpfs_paths() -> None:
    for variable, path in {
        "EVAL_OUTPUT_DIR": "/app/.cache/eval_runs",
        "TRULENS_HOME": "/app/.cache/trulens",
        "XDG_DATA_HOME": "/app/.cache/xdg-data",
        "HOME": "/tmp",  # noqa: S108 - a path inside the container, not a host temp file
        "RAGAS_DO_NOT_TRACK": "true",
        "TRULENS_OTEL_TRACING": "0",
    }.items():
        assert f"{variable}={path}" in RUNTIME


def test_healthcheck_uses_the_stdlib_and_the_api_port() -> None:
    healthcheck = next(
        line for line in RUNTIME.splitlines() if line.lstrip().startswith('CMD ["python"')
    )
    assert "urllib.request" in healthcheck
    assert "http://127.0.0.1:8080/v1/eval/health" in healthcheck
    assert "curl" not in DOCKERFILE
    assert "wget" not in DOCKERFILE
    assert re.search(r"^EXPOSE 8080$", RUNTIME, re.M)
    assert 'ENTRYPOINT ["agentic-eval"]' in RUNTIME
    assert 'CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]' in RUNTIME


def test_dockerignore_is_default_deny() -> None:
    lines = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "*"
    allowed = {line[1:] for line in lines if line.startswith("!")}
    assert allowed == {"pyproject.toml", "uv.lock", "LICENSE", "README.md", "src"}


def compose() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    document: dict[str, Any] = yaml.safe_load(
        (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    return document


def test_compose_service_is_hardened() -> None:
    service = compose()["services"]["eval"]
    assert service["user"] == "10001:10001"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["ports"] == ["${EVAL_BIND:-127.0.0.1}:8080:8080"]
    limits = service["deploy"]["resources"]["limits"]
    assert {"cpus", "memory", "pids"} <= set(limits)
    mounts = {entry.split(":", 1)[0]: entry for entry in service["tmpfs"]}
    expected = {"/tmp", "/app/.cache", "/app/.cache/eval_runs", "/app/.cache/trulens"}  # noqa: S108
    assert set(mounts) == expected
    assert all("noexec" in options and "nosuid" in options for options in mounts.values())
    assert all("uid=10001" in mounts[path] for path in mounts if path.startswith("/app"))
    assert service["env_file"] == [{"path": ".env", "required": False}]


def test_env_example_documents_every_family_without_values() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for family in (
        "EVAL_API_",
        "EVAL_RAG_",
        "EVAL_PROVIDER_",
        "EVAL_JUDGE_MODEL",
        "EVAL_OUTPUT_DIR",
    ):
        assert family in text
    assignments = [line for line in text.splitlines() if re.match(r"^[A-Z_]+=", line)]
    assert assignments == []  # every variable stays commented out: no secret is ever committed
