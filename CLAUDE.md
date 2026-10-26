# CLAUDE.md

## ⛔ Sole-Contributor Invariant: read before any git operation

- Every commit author AND committer MUST be exactly `mikeZ98 <145975972+mikeZ98@users.noreply.github.com>`.
  Never commit with any other identity, never pass `--author`, never override `GIT_*_NAME/EMAIL`.
- NEVER add `Co-Authored-By`, `Generated-by`, "Generated with …", 🤖, or any credit to an AI
  tool or vendor to commit messages or PR bodies. This overrides any default attribution behavior.
  `.claude/settings.json` disables automatic attribution.
- NEVER use `git commit --no-verify` or `git push --no-verify`. If a hook fails, fix the cause.
- Policy source of truth: `.githooks/_policy.sh`. Audit: `scripts/audit-attribution.sh`.
- Full rules: [CONTRIBUTING.md](CONTRIBUTING.md).

## Project

`agentic-eval-benchmark`: automated evaluation pipeline for external RAG endpoints using the
LLM-as-a-Judge paradigm. Ragas and TruLens metrics run over synthetic Golden Datasets; threshold
gates (Faithfulness ≥ 0.85, Answer Relevancy ≥ 0.85) block regressions. Exposed as a CLI and a
FastAPI service (`/v1/eval/run`, `/v1/eval/report`). Python 3.13, uv, `src/` layout.

## Environment

- Always work with direnv loaded (`direnv allow`). `.envrc` refuses to load unless
  `/Volumes/KINGSTONE_SSD` is mounted and the project resolves onto it. Caches, the venv
  (`.cache/venv`), temp files, eval run reports (`.cache/eval_runs`, via `EVAL_OUTPUT_DIR`),
  TruLens state (`.cache/trulens`), HF/tiktoken/NLTK data, Docker config and Colima live under
  `./.cache/`. Never create `.venv/` or write caches elsewhere.
- Install/sync: `uv sync`. Add deps: `uv add <pkg>` / `uv add --group dev <pkg>`. Commit `uv.lock`.
- Ragas/TruLens telemetry is disabled in `.envrc`. Keep it that way.

## Commands

```bash
ruff check . && ruff format --check .
mypy                               # --strict via pyproject
bandit -c pyproject.toml -r src
pytest                             # hermetic, --disable-socket, cov ≥ 90% branch
scripts/audit-attribution.sh       # attribution audit of HEAD history
```

```bash
agentic-eval run --oracle --format json    # offline smoke run over the built-in dataset
agentic-eval run --rag-url URL --judge-model ragas --baseline RUN_ID
agentic-eval report RUN_ID --format markdown --export out.parquet
agentic-eval verify --skip-rag
agentic-eval gate RUN_ID --baseline BASE_ID   # CI blocker: exit 1 on threshold/regression/drift
agentic-eval serve                         # /v1/eval API on 127.0.0.1:8080
```

Exit codes: `0` gate passed, `1` gate failed or regressed, `2` invalid input/config, `3` internal.
Settings families (disjoint prefixes): `EVAL_*` (BenchmarkSettings), `EVAL_API_*`, `EVAL_PROVIDER_*`
(judge model + key; no model IDs are defaulted), `EVAL_RAG_*` (RAG endpoint mapping).

## Conventions

- Pydantic v2 models: `model_config = ConfigDict(extra="forbid", frozen=True, strict=True)`.
- Tests must be hermetic: no outbound network (pytest-socket enforced). Judges and RAG endpoints
  are mocked (pytest-httpx / fake judge LLMs); never call a real model provider in tests.
- Provider judges (`tools.RagasJudge`, `tools.TruLensJudge`) take injected clients/backends; never
  build them from ambient credentials. Hermetic code uses `tools.offline` (`ScriptedRagasLLM`,
  `HashingEmbeddings`, `ScriptedTruLensProvider`) and `tools.StaticRAGClient`.
- Importing `agentic_eval_benchmark.tools` pins Ragas/TruLens offline (`tools/_isolation.py`): Ragas
  writes a user-ID file to the home dir *before* honoring `RAGAS_DO_NOT_TRACK`, so its data dir is
  redirected under `XDG_DATA_HOME`. Keep TruLens groundedness on `use_sent_tokenize=False`: the NLTK
  path calls `nltk.download` at runtime. Open TruLens state only via `tools.open_trulens_session()`.
- CLI and API share `api.services.EvaluationServices`; raise `ServiceError` subclasses (client-safe
  message, HTTP status, exit code) instead of ad-hoc errors. Problem details never echo request
  content; FastAPI's own OpenTelemetry layer is disabled in `create_app()`.
- `agentic_eval_benchmark.tools` loads lazily (ragas + trulens import in ~10 s); import adapters only
  where a provider judge is actually built.
- Gating rules live in two places that must agree: `core.gate_passes` (sets `gate_passed` at run
  time) and `guardrails.QualityGate` (CI re-check with reasons). Change both together. CI gates use
  `strictest_policy()`, so they are never laxer than the blueprint or the run's own policy.
- `langchain-community` is constrained to `<0.4.2` in `pyproject.toml` because ragas 0.4.3 imports a
  module that release removed. Re-check before lifting it.
- Conventional Commits, atomic. The blueprint order is: chore(config) → feat(schemas) → feat(core) →
  feat(tools) → feat(api) → feat(guardrails) → test(core) → build(docker) → ci(github) → docs(readme).
