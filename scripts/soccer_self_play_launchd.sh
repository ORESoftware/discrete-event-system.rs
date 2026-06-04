#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
cd "$repo_dir"

export SOCCER_GAMES="${SOCCER_GAMES:-100}"
export SOCCER_SHARDS="${SOCCER_SHARDS:-1}"
export SOCCER_PARALLEL_SHARDS="${SOCCER_PARALLEL_SHARDS:-$SOCCER_SHARDS}"
export SOCCER_RUN_ID="${SOCCER_RUN_ID:-overnight-$(date -u +%Y%m%dT%H%M%SZ)}"
export SOCCER_OUT_ROOT="${SOCCER_OUT_ROOT:-$repo_dir/out/soccer-self-play/$SOCCER_RUN_ID}"

build_release="${SOCCER_BUILD_RELEASE:-1}"
if [[ -n "${SOCCER_BINARY:-}" ]]; then
  export SOCCER_BINARY
elif (( build_release != 0 )); then
  export SOCCER_BINARY="target/release/main_soccer_learning_run"
else
  export SOCCER_BINARY="target/debug/main_soccer_learning_run"
fi
mkdir -p "$SOCCER_OUT_ROOT"

if (( build_release != 0 )); then
  cargo build --release --bin main_soccer_learning_run
fi

export SOCCER_BUILD_RELEASE=0
export -p | grep 'declare -x SOCCER_' > "$SOCCER_OUT_ROOT/launch.env" || true

label="com.ores.soccer-self-play.$SOCCER_RUN_ID"
printf '%s\n' "$label" > "$SOCCER_OUT_ROOT/launchd.label"

launchctl submit \
  -l "$label" \
  -o "$SOCCER_OUT_ROOT/launchd.stdout.log" \
  -e "$SOCCER_OUT_ROOT/launchd.stderr.log" \
  -- /bin/bash -lc "cd '$repo_dir' && set -a && source '$SOCCER_OUT_ROOT/launch.env' && set +a && exec bash scripts/soccer_self_play_local.sh"

printf 'run_id=%s\n' "$SOCCER_RUN_ID"
printf 'label=%s\n' "$label"
printf 'out_root=%s\n' "$SOCCER_OUT_ROOT"
printf 'status_command=scripts/soccer_self_play_status.sh %s\n' "$SOCCER_OUT_ROOT"
