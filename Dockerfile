# syntax=docker/dockerfile:1.7
#
# agentic-eval-benchmark: hardened multi-stage image.
#   builder: uv resolves the frozen lockfile into /app/.venv (no dev deps, project as a wheel).
#   runtime: python:3.13-slim plus the venv; no pip, uv or compilers; non-root (10001);
#            read-only-rootfs compatible (all state under tmpfs /app/.cache and /tmp);
#            telemetry off for Ragas, TruLens, Hugging Face and FastAPI.
# For reproducible releases, pin both FROM images by digest (@sha256:...).

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.11.7

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first: this layer is reused until the lockfile changes.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Then the project itself as a regular (non-editable) package, so the runtime image
# carries no source tree.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Fail the build, not the first request: the package, its bundled reference dataset and
# the CLI entry point must all load from the installed wheel.
RUN /app/.venv/bin/python -c "from agentic_eval_benchmark.tools import load_reference_dataset; assert len(load_reference_dataset().samples) == 8" \
 && /app/.venv/bin/agentic-eval --help > /dev/null

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="agentic-eval-benchmark" \
      org.opencontainers.image.description="LLM-as-a-Judge evaluation of RAG endpoints with strict quality gates" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="MIT"

# Shrink the attack surface: a dedicated unprivileged user with no home directory or
# login shell, then no package installers at all (pip, ensurepip, apt, dpkg). The dpkg
# database (/var/lib/dpkg/status) stays so image scanners can still list OS packages.
RUN set -eux; \
    groupadd --system --gid 10001 appuser; \
    useradd --system --uid 10001 --gid 10001 --no-create-home \
            --home-dir /nonexistent --shell /usr/sbin/nologin appuser; \
    mkdir -p /app/.cache; \
    chown 10001:10001 /app/.cache; \
    rm -rf /usr/local/bin/pip* \
           /usr/local/lib/python3*/site-packages/pip \
           /usr/local/lib/python3*/site-packages/pip-*.dist-info \
           /usr/local/lib/python3*/ensurepip \
           /usr/bin/apt /usr/bin/apt-* /usr/bin/dpkg /usr/bin/dpkg-* \
           /usr/lib/apt /etc/apt /var/lib/apt /var/cache/apt /var/cache/debconf

# Owned by root and read-only to appuser: the application cannot modify itself.
COPY --from=builder /app/.venv /app/.venv

# Every writable path lives under /app/.cache (a tmpfs at runtime) or /tmp. This mirrors
# the .envrc pinning: run reports, TruLens state, the Ragas user-ID file, HF/tiktoken/
# NLTK/matplotlib caches. Telemetry is disabled at the environment level as well.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    CACHE_ROOT=/app/.cache \
    EVAL_OUTPUT_DIR=/app/.cache/eval_runs \
    TRULENS_HOME=/app/.cache/trulens \
    XDG_CACHE_HOME=/app/.cache/xdg \
    XDG_DATA_HOME=/app/.cache/xdg-data \
    XDG_STATE_HOME=/app/.cache/xdg-state \
    HF_HOME=/app/.cache/huggingface \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    NLTK_DATA=/app/.cache/nltk \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    RAGAS_DO_NOT_TRACK=true \
    TRULENS_OTEL_TRACING=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1

WORKDIR /app
USER 10001:10001
EXPOSE 8080

# Probe with the stdlib only (no curl/wget in the image). Any non-200, including the 503
# a degraded service returns (unwritable storage, unconfigured default judge), marks the
# container unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/v1/eval/health', timeout=4).status == 200 else 1)"]

# The CLI is the entrypoint: the default command serves the /v1/eval API, and any other
# subcommand runs instead, e.g. `docker run IMAGE run --oracle --format json`.
ENTRYPOINT ["agentic-eval"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
