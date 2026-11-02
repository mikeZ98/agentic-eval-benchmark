#!/usr/bin/env bash
# Audit commits for the sole-contributor identity invariant (author + committer).
#
# Usage: scripts/audit-attribution.sh [<git rev-list args>...]   (default: HEAD)
# Used by .githooks/pre-push locally and by the CI attribution-audit job
# (CI must check out with fetch-depth: 0 so the whole history is audited).
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=../.githooks/_policy.sh
source "$ROOT/.githooks/_policy.sh"

(($#)) || set -- HEAD

if git rev-parse --is-shallow-repository | grep -q true; then
  eval_err "Shallow clone: history is incomplete. Fetch with --unshallow (CI: fetch-depth: 0)."
  exit 1
fi

fail=0
count=0
while IFS= read -r sha; do
  count=$((count + 1))
  IFS=$'\x1f' read -r an ae cn ce <<<"$(git log -1 --format='%an%x1f%ae%x1f%cn%x1f%ce' "$sha")"
  short="${sha:0:12}"
  eval_check_identity "[$short] Author" "$an" "$ae" || fail=1
  eval_check_identity "[$short] Committer" "$cn" "$ce" || fail=1
done < <(git rev-list "$@")

if ((fail)); then
  eval_err "Attribution audit FAILED ($count commit(s) scanned)."
  exit 1
fi
eval_ok "Attribution audit passed ($count commit(s) scanned)."
