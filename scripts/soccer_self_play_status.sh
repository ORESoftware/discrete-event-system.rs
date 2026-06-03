#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
cd "$repo_dir"

run_dir="${1:-}"

if [[ -z "$run_dir" ]]; then
  run_env="$(
    find out/soccer-self-play -name run.env -print 2>/dev/null \
      | while IFS= read -r candidate; do
          if modified_at="$(stat -c '%Y' "$candidate" 2>/dev/null)"; then
            :
          else
            modified_at="$(stat -f '%m' "$candidate")"
          fi
          printf '%s\t%s\n' "$modified_at" "$candidate"
        done \
      | sort -n \
      | tail -n 1 \
      | cut -f2- || true
  )"
  if [[ -z "$run_env" ]]; then
    echo "usage: scripts/soccer_self_play_status.sh <run-dir>" >&2
    echo "no run.env found under out/soccer-self-play" >&2
    exit 2
  fi
  run_dir="$(dirname "$run_env")"
fi

if [[ ! -d "$run_dir" ]]; then
  echo "run directory not found: $run_dir" >&2
  exit 2
fi

echo "run_dir=$run_dir"

if [[ -f "$run_dir/run.env" ]]; then
  cat "$run_dir/run.env"
fi

if [[ -f "$run_dir/launchd.label" ]]; then
  label="$(cat "$run_dir/launchd.label")"
  echo "launchd_label=$label"
  if launch_info="$(launchctl print "gui/$(id -u)/$label" 2>/dev/null)"; then
    printf '%s\n' "$launch_info" | grep -E 'state =|pid =|last exit code' || true
  else
    echo "launchd_state=not-loaded"
  fi
fi

expected_games=""
if [[ -f "$run_dir/run.env" ]]; then
  expected_games="$(grep '^games=' "$run_dir/run.env" | cut -d= -f2- || true)"
fi

echo "shards:"
found_shards=0
for shard_dir in "$run_dir"/shard-*-of-*; do
  if [[ ! -d "$shard_dir" ]]; then
    continue
  fi
  found_shards=$((found_shards + 1))
  shard_name="$(basename "$shard_dir")"
  stdout_log="$shard_dir/stdout.log"
  episode_log="$shard_dir/episodes.jsonl"
  artifact="$shard_dir/artifact.json"
  learned_params="$shard_dir/learned-params.json"
  checkpoint_artifact="$shard_dir/artifact.json.checkpoint.json"

  completed_games="0"
  if [[ -f "$episode_log" ]]; then
    completed_games="$(wc -l < "$episode_log" | tr -d ' ')"
  fi

  artifact_status="missing"
  if [[ -f "$artifact" ]]; then
    artifact_bytes="$(wc -c < "$artifact" | tr -d ' ')"
    artifact_status="present:${artifact_bytes}B"
  fi

  learned_params_status="missing"
  if [[ -f "$learned_params" ]]; then
    learned_params_bytes="$(wc -c < "$learned_params" | tr -d ' ')"
    learned_params_status="present:${learned_params_bytes}B"
  fi

  checkpoint_status="missing"
  if [[ -f "$checkpoint_artifact" ]]; then
    checkpoint_bytes="$(wc -c < "$checkpoint_artifact" | tr -d ' ')"
    checkpoint_status="present:${checkpoint_bytes}B"
  fi

  last_line="no-progress-yet"
  if [[ -f "$stdout_log" ]]; then
    last_match="$(grep -E 'progress_game|completed_game|soccer_self_play' "$stdout_log" | tail -n 1 || true)"
    if [[ -n "$last_match" ]]; then
      last_line="$last_match"
    fi
  fi

  if [[ -n "$expected_games" ]]; then
    echo "  $shard_name completed_games=$completed_games/$expected_games artifact=$artifact_status params=$learned_params_status checkpoint=$checkpoint_status"
  else
    echo "  $shard_name completed_games=$completed_games artifact=$artifact_status params=$learned_params_status checkpoint=$checkpoint_status"
  fi
  echo "    last=$last_line"
done

if (( found_shards == 0 )); then
  echo "  no shard directories found"
fi
