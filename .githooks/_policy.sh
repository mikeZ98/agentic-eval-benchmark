# shellcheck shell=bash
# Shared commit policy (sole-contributor identity, Conventional Commits) for git hooks
# and scripts/audit-attribution.sh. Message trailers such as Co-Authored-By are allowed.
# Sourced, never executed. Git ignores this file as a hook (not a hook name).
# Stays compatible with macOS /bin/bash 3.2 (no mapfile, no ${var,,}).

readonly EVAL_ALLOWED_NAME="mikeZ98"
readonly EVAL_ALLOWED_EMAIL="145975972+mikeZ98@users.noreply.github.com"

readonly EVAL_CONVENTIONAL_RE='^(feat|fix|chore|docs|test|build|ci|refactor|perf|style|revert)(\([a-z0-9._/-]+\))?!?: [^ ].{0,98}$'

eval_err() { printf '\033[31m✖ %s\033[0m\n' "$*" >&2; }
eval_ok() { printf '\033[32m✔ %s\033[0m\n' "$*" >&2; }

# eval_check_identity <role> <name> <email>
eval_check_identity() {
  local role="$1" name="$2" email="$3"
  if [[ "$name" != "$EVAL_ALLOWED_NAME" || "$email" != "$EVAL_ALLOWED_EMAIL" ]]; then
    eval_err "$role identity '$name <$email>' violates the sole-contributor invariant."
    eval_err "Expected: '$EVAL_ALLOWED_NAME <$EVAL_ALLOWED_EMAIL>'."
    return 1
  fi
}

# eval_check_pending_identities — validate the identities `git commit` is about to
# record. `git var` resolves --author, GIT_AUTHOR_*/GIT_COMMITTER_* overrides and
# every config layer exactly as the commit will.
eval_check_pending_identities() {
  local fail=0 role ident name email
  for role in AUTHOR COMMITTER; do
    if ! ident="$(git var "GIT_${role}_IDENT" 2>/dev/null)"; then
      eval_err "Cannot resolve the $role identity (user.useConfigOnly is set: run 'direnv allow')."
      fail=1
      continue
    fi
    ident="${ident% * *}" # strip "<unix-time> <tz>"
    name="${ident% <*}"
    email="${ident##*<}"
    email="${email%>}"
    eval_check_identity "$role" "$name" "$email" || fail=1
  done
  return "$fail"
}

# eval_clean_message — strip comments and anything below the scissors line
# (`git commit -v` appends the diff there).
eval_clean_message() {
  sed -e '/^# ------------------------ >8 ------------------------$/,$d' -e '/^#/d'
}

# eval_check_subject <first-line>
eval_check_subject() {
  local subject="$1"
  case "$subject" in
    Merge\ * | fixup!\ * | squash!\ * | amend!\ *) return 0 ;;
  esac
  if ! grep -qE "$EVAL_CONVENTIONAL_RE" <<<"$subject"; then
    eval_err "Subject is not a Conventional Commit (max 100 chars): '$subject'"
    eval_err "Format: <type>(<scope>)?: <summary>   types: feat fix chore docs test build ci refactor perf style revert"
    return 1
  fi
}
