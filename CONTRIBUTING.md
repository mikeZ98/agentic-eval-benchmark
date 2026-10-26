# Contributing to agentic-eval-benchmark

## ⛔ Sole-Contributor Invariant (non-negotiable)

Every commit in this repository, on every branch, MUST have **author and committer** set to exactly:

| Field | Value |
|-------|-------|
| Name  | `mikeZ98` |
| Email | `145975972+mikeZ98@users.noreply.github.com` |

- No other name, no machine username, no personal or local email: zero tolerance.
- Commit messages MUST NOT contain AI attribution or co-author metadata. Rejected:
  - any `Co-Authored-By:` trailer, and `Generated-by:`, `Assisted-by:`, `AI-Generated:`, `Made-with:`
    and similar trailers, whatever their value;
  - trailers keyed by an AI tool (`Claude-Session:`, `Copilot-…:`) and any `*-by:` trailer
    (`Signed-off-by:`, `Reviewed-by:`, …) that credits Claude, Anthropic, ChatGPT, OpenAI, GPT-n,
    Copilot, Gemini, Cursor, Codex or similar tools;
  - signatures such as "Generated with …" / "Written by …" naming those tools, the 🤖 emoji, vendor
    addresses (`noreply@anthropic.com`, `@openai.com`) and `claude.ai` / `chatgpt.com` links;
  - the words **Claude** and **ChatGPT** anywhere. The literal file name `CLAUDE.md` is the only exception.
- Provider names used as product names in a commit body (for example "OpenAI-compatible judge
  client") are allowed: this repository evaluates LLM judges, so they are domain language, not
  attribution.
- Merges made in the GitHub web UI record `GitHub <noreply@github.com>` as committer and **will fail**
  the audit. Rebase or merge locally, then push.

### How it is enforced

| Layer | Where | Checks |
|-------|-------|--------|
| `pre-commit` | `.githooks/pre-commit` | author/committer identity; blocks `.cache/`, venvs, `.env*`, keys, coverage and SQLite files; ruff check and format on staged Python |
| `commit-msg` | `.githooks/commit-msg` | forbidden trailers/terms; Conventional Commit subject; identity re-check (covers `--author`, `--amend` and `git merge`) |
| `pre-push` | `.githooks/pre-push` | full audit of every commit being pushed (catches cherry-picks and rebases, which skip `pre-commit`) |
| CI | `scripts/audit-attribution.sh` in the `attribution-audit` job of `.github/workflows/ci.yml` (`fetch-depth: 0`, PR head commit) | server-side audit of the whole branch history; every other job depends on it |

The policy (allowed identity and banned patterns) lives in one place: `.githooks/_policy.sh`.
Audit any range by hand with `scripts/audit-attribution.sh [<rev-list args>]` (default: `HEAD`).
The audit refuses to run on a shallow clone, so CI cannot silently audit a partial history.

Never bypass hooks with `--no-verify`. CI re-runs the same audit and will reject the branch anyway.

## Environment setup (macOS, external SSD)

All caches, the virtualenv, temp files, evaluation artifacts and the Colima VM live under `./.cache/`
on `/Volumes/KINGSTONE_SSD`. Nothing is written to the internal disk.

```bash
brew install uv direnv colima docker
direnv allow          # verifies the SSD mount, pins caches, sets core.hooksPath + the local git identity
uv sync               # downloads CPython 3.13 into .cache/uv-python and creates .cache/venv
```

`.envrc` fails closed: if `/Volumes/KINGSTONE_SSD` is not mounted, the project resolves outside it, or
it is read-only, nothing is exported. Otherwise it sets `XDG_{CACHE,DATA,STATE}_HOME`, `TMPDIR`,
`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_PROJECT_ENVIRONMENT`, `RUFF_CACHE_DIR`, `MYPY_CACHE_DIR`,
`COVERAGE_FILE`, `PYTHONPYCACHEPREFIX`, `HF_HOME`, `TIKTOKEN_CACHE_DIR`, `NLTK_DATA`,
`EVAL_OUTPUT_DIR` (run reports), `TRULENS_HOME`, `TRULENS_DATABASE_URL`, `TRIVY_CACHE_DIR`,
`COLIMA_HOME`, `LIMA_HOME`, `DOCKER_HOST` and `DOCKER_CONFIG`, and disables Ragas, TruLens and
Hugging Face telemetry. Tool config in `pyproject.toml` also points the pytest, ruff, mypy and
coverage caches at `.cache/`, so they stay there even without direnv. Put personal overrides in
`.envrc.local` (gitignored).

### Containers (Docker via Colima)

Use the wrapper, never Docker Desktop's socket (`/var/run/docker.sock`, `~/.docker`):

```bash
scripts/colima.sh start      # VM disk, images and layers in .cache/lima, base image in .cache/xdg
docker compose up --build    # API on 127.0.0.1:8080, read-only rootfs, uid 10001
docker compose run --rm eval run --oracle --format json
scripts/colima.sh stop
```

The wrapper runs Colima with `HOME=.cache/home` (Lima's download cache ignores every other
variable) and registers the buildx/compose plugin directories in `$DOCKER_CONFIG/config.json`.
`LIMA_HOME` is `.cache/lima`, not `$COLIMA_HOME/_lima`: Lima's socket paths must stay below the
104-byte `UNIX_PATH_MAX`. **Known limitation:** Lima's DNS forwarder hard-codes `/tmp` for its
listener sockets (`/private/tmp/lima-psl-127.0.0.1-53-*`, an empty directory with a 0-byte socket
per VM start) whatever `TMPDIR` says; no file content is written there.

## Quality gates

```bash
ruff check . && ruff format --check .
mypy                      # strict, configured in pyproject.toml
bandit -c pyproject.toml -r src
pytest                    # hermetic: sockets disabled, coverage ≥ 90% (branch)
```

## Continuous integration

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`. After `attribution-audit`,
four jobs run in parallel: `lint-and-typecheck` (ruff, `mypy --strict src tests`), `security-audit`
(Trivy fs scan for CRITICAL/HIGH, bandit), `test-hermetic` (pytest, sockets disabled, branch
coverage ≥ 90%) and `eval-benchmark-gate` (oracle run over the reference dataset, then the blocking
`agentic-eval gate`, reported in the step summary). `docker-build` then builds the image and checks
it runs as 10001, has no installers, and turns healthy with `--network none` and a read-only root.

Every action is pinned to a full commit SHA with its release tag in a comment, and permissions are
`contents: read`. To bump an action, resolve the tag yourself (`git ls-remote https://github.com/OWNER/REPO
refs/tags/vX.Y.Z 'refs/tags/vX.Y.Z^{}'`) and update the SHA and the comment together;
`tests/test_ci_workflow.py` rejects unpinned or uncommented actions.

## Commit discipline

- [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <summary>`, subject
  ≤ 100 characters. Types: `feat fix chore docs test build ci refactor perf style revert`.
- One logical change per commit. Keep commits atomic and green.
- Blueprint order: `chore(config)` → `feat(schemas)` → `feat(core)` → `feat(tools)` → `feat(api)` →
  `feat(guardrails)` → `test(core)` → `build(docker)` → `ci(github)` → `docs(readme)`.
