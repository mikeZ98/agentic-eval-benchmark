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
  (`.cache/venv`), temp files, eval artifacts (`.cache/eval`), HF/tiktoken/NLTK data, Docker config
  and Colima live under `./.cache/`. Never create `.venv/` or write caches elsewhere.
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

## Conventions

- Pydantic v2 models: `model_config = ConfigDict(extra="forbid", frozen=True, strict=True)`.
- Tests must be hermetic: no outbound network (pytest-socket enforced). Judges and RAG endpoints
  are mocked (pytest-httpx / fake judge LLMs); never call a real model provider in tests.
- `langchain-community` is constrained to `<0.4.2` in `pyproject.toml` because ragas 0.4.3 imports a
  module that release removed. Re-check before lifting it.
- Conventional Commits, atomic. The blueprint order is: chore(config) → feat(schemas) → feat(core) →
  feat(tools) → feat(api) → feat(guardrails) → test(core) → build(docker) → ci(github) → docs(readme).
