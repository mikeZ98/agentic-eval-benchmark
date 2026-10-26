#!/usr/bin/env bash
# Run Colima with every byte of VM, image and layer state on the project SSD.
#
# Usage: scripts/colima.sh start|stop|status|delete   (run with direnv loaded)
#
# .envrc pins COLIMA_HOME, LIMA_HOME, DOCKER_HOST and DOCKER_CONFIG under ./.cache/.
# Colima caches the VM base image under $XDG_CACHE_HOME (pinned by .envrc). Two leaks
# remain without this wrapper:
#   - Lima's own download cache is os.UserCacheDir() (~/Library/Caches/lima on macOS)
#     with no variable to move it: HOME is overridden for the colima process only.
#   - With DOCKER_CONFIG relocated the CLI no longer finds buildx; its plugin directory
#     is registered (read-only) in $DOCKER_CONFIG/config.json instead.
# The host Docker socket (/var/run/docker.sock, ~/.docker) is never used.
# Known limitation: Lima's DNS forwarder hard-codes /tmp for its listener sockets
# (lima-psl-127.0.0.1-53-*: empty directory + 0-byte socket per start), whatever TMPDIR says.
set -euo pipefail

: "${CACHE_ROOT:?load the project environment first (direnv allow)}"
: "${COLIMA_HOME:?}" "${DOCKER_CONFIG:?}"

COLIMA_FAKE_HOME="$CACHE_ROOT/home"
PLUGIN_DIRS=(/opt/homebrew/lib/docker/cli-plugins /usr/local/lib/docker/cli-plugins
  /Applications/Docker.app/Contents/Resources/cli-plugins)

mkdir -p "$COLIMA_FAKE_HOME" "$DOCKER_CONFIG"

write_docker_config() {
  local dirs=() d joined
  for d in "${PLUGIN_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
      dirs+=("\"$d\"")
    fi
  done
  joined="$(IFS=,; echo "${dirs[*]-}")"
  printf '{\n  "cliPluginsExtraDirs": [%s]\n}\n' "$joined" \
    >"$DOCKER_CONFIG/config.json"
}

colima_cmd() {
  HOME="$COLIMA_FAKE_HOME" colima "$@"
}

case "${1:-status}" in
  start)
    write_docker_config
    colima_cmd start --cpu "${COLIMA_CPUS:-4}" --memory "${COLIMA_MEMORY:-6}" \
      --disk "${COLIMA_DISK:-40}" --vm-type vz --mount-type virtiofs \
      --mount "$CACHE_ROOT/tmp:w"
    ;;
  stop | status | delete)
    colima_cmd "$1"
    ;;
  *)
    echo "usage: $0 start|stop|status|delete" >&2
    exit 2
    ;;
esac
