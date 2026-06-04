#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
cd "$repo_dir"

if [[ -n "${SOCCER_RUN_DIR:-}" || -n "${SOCCER_ARTIFACT_PATH:-}" || -n "${SOCCER_CHECKPOINT_ARTIFACT_PATH:-}" || -n "${SOCCER_EPISODE_LOG_PATH:-}" || -n "${SOCCER_LEARNED_PARAMS_PATH:-}" ]]; then
  echo "SOCCER_RUN_DIR, SOCCER_ARTIFACT_PATH, SOCCER_CHECKPOINT_ARTIFACT_PATH, SOCCER_EPISODE_LOG_PATH, and SOCCER_LEARNED_PARAMS_PATH are managed per shard by this launcher." >&2
  echo "Use SOCCER_OUT_ROOT or SOCCER_RUN_ID to choose the output namespace." >&2
  exit 2
fi

export SOCCER_GAMES="${SOCCER_GAMES:-100}"
export SOCCER_HALVES="${SOCCER_HALVES:-2}"
export SOCCER_HALF_MINUTES="${SOCCER_HALF_MINUTES:-45}"
export SOCCER_PERIOD_BREAK_RECOVERY_SECONDS="${SOCCER_PERIOD_BREAK_RECOVERY_SECONDS:-900}"
export SOCCER_DT_SECONDS="${SOCCER_DT_SECONDS:-0.2}"
export SOCCER_LEARNING_INTERVAL_TICKS="${SOCCER_LEARNING_INTERVAL_TICKS:-4}"
export SOCCER_CHECKPOINT_INTERVAL_GAMES="${SOCCER_CHECKPOINT_INTERVAL_GAMES:-10}"
export SOCCER_ARTIFACT_MAX_ENTRIES_PER_POLICY="${SOCCER_ARTIFACT_MAX_ENTRIES_PER_POLICY:-10000}"
export SOCCER_ATTACK_SPACING_DELTA_WEIGHT="${SOCCER_ATTACK_SPACING_DELTA_WEIGHT:-0.22}"
export SOCCER_ATTACK_SPACING_SCORE_WEIGHT="${SOCCER_ATTACK_SPACING_SCORE_WEIGHT:-0.06}"
export SOCCER_ATTACK_WIDTH_DELTA_WEIGHT="${SOCCER_ATTACK_WIDTH_DELTA_WEIGHT:-0.52}"
export SOCCER_ATTACK_WIDTH_SCORE_WEIGHT="${SOCCER_ATTACK_WIDTH_SCORE_WEIGHT:-0.14}"
export SOCCER_ATTACK_FLANK_LANE_WEIGHT="${SOCCER_ATTACK_FLANK_LANE_WEIGHT:-0.28}"
export SOCCER_DEFENSE_SPACING_DELTA_WEIGHT="${SOCCER_DEFENSE_SPACING_DELTA_WEIGHT:-0.08}"
export SOCCER_DEFENSE_SPACING_SCORE_WEIGHT="${SOCCER_DEFENSE_SPACING_SCORE_WEIGHT:-0.04}"
export SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT="${SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT:-0.42}"
export SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT="${SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT:-0.14}"

shards="${SOCCER_SHARDS:-1}"
parallel_shards="${SOCCER_PARALLEL_SHARDS:-1}"
run_id="${SOCCER_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
out_root="${SOCCER_OUT_ROOT:-out/soccer-self-play/$run_id}"
build_release="${SOCCER_BUILD_RELEASE:-1}"

if (( shards < 1 )); then
  echo "SOCCER_SHARDS must be at least 1." >&2
  exit 2
fi

if (( parallel_shards < 1 )); then
  echo "SOCCER_PARALLEL_SHARDS must be at least 1." >&2
  exit 2
fi

if (( build_release != 0 )); then
  cargo build --release --bin main_soccer_learning_run
fi

binary="${SOCCER_BINARY:-target/release/main_soccer_learning_run}"
mkdir -p "$out_root"

printf 'run_id=%s\n' "$run_id" > "$out_root/run.env"
printf 'games=%s\n' "$SOCCER_GAMES" >> "$out_root/run.env"
printf 'halves=%s\n' "$SOCCER_HALVES" >> "$out_root/run.env"
printf 'half_minutes=%s\n' "$SOCCER_HALF_MINUTES" >> "$out_root/run.env"
if [[ -n "${SOCCER_MINUTES:-}" ]]; then
  printf 'minutes=%s\n' "$SOCCER_MINUTES" >> "$out_root/run.env"
fi
printf 'period_break_recovery_seconds=%s\n' "$SOCCER_PERIOD_BREAK_RECOVERY_SECONDS" >> "$out_root/run.env"
printf 'dt_seconds=%s\n' "$SOCCER_DT_SECONDS" >> "$out_root/run.env"
printf 'learning_interval_ticks=%s\n' "$SOCCER_LEARNING_INTERVAL_TICKS" >> "$out_root/run.env"
printf 'checkpoint_interval_games=%s\n' "$SOCCER_CHECKPOINT_INTERVAL_GAMES" >> "$out_root/run.env"
printf 'artifact_max_entries_per_policy=%s\n' "$SOCCER_ARTIFACT_MAX_ENTRIES_PER_POLICY" >> "$out_root/run.env"
printf 'artifact_file=artifact.json\n' >> "$out_root/run.env"
printf 'checkpoint_policy_file=checkpoint-policy.json\n' >> "$out_root/run.env"
printf 'learned_params_file=learned-params.json\n' >> "$out_root/run.env"
printf 'episode_log_file=episodes.jsonl\n' >> "$out_root/run.env"
printf 'manifest_file=manifest.json\n' >> "$out_root/run.env"
printf 'attack_spacing_delta_weight=%s\n' "$SOCCER_ATTACK_SPACING_DELTA_WEIGHT" >> "$out_root/run.env"
printf 'attack_spacing_score_weight=%s\n' "$SOCCER_ATTACK_SPACING_SCORE_WEIGHT" >> "$out_root/run.env"
printf 'attack_width_delta_weight=%s\n' "$SOCCER_ATTACK_WIDTH_DELTA_WEIGHT" >> "$out_root/run.env"
printf 'attack_width_score_weight=%s\n' "$SOCCER_ATTACK_WIDTH_SCORE_WEIGHT" >> "$out_root/run.env"
printf 'attack_flank_lane_weight=%s\n' "$SOCCER_ATTACK_FLANK_LANE_WEIGHT" >> "$out_root/run.env"
printf 'defense_spacing_delta_weight=%s\n' "$SOCCER_DEFENSE_SPACING_DELTA_WEIGHT" >> "$out_root/run.env"
printf 'defense_spacing_score_weight=%s\n' "$SOCCER_DEFENSE_SPACING_SCORE_WEIGHT" >> "$out_root/run.env"
printf 'defense_contract_delta_weight=%s\n' "$SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT" >> "$out_root/run.env"
printf 'defense_compactness_score_weight=%s\n' "$SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT" >> "$out_root/run.env"
printf 'shards=%s\n' "$shards" >> "$out_root/run.env"
printf 'parallel_shards=%s\n' "$parallel_shards" >> "$out_root/run.env"

run_shard() {
  local shard_index="$1"
  local shard_dir="$out_root/shard-${shard_index}-of-${shards}"
  mkdir -p "$shard_dir"
  printf 'starting shard %s/%s -> %s\n' "$shard_index" "$shards" "$shard_dir"
  SOCCER_SHARD_INDEX="$shard_index" \
    SOCCER_SHARD_COUNT="$shards" \
    SOCCER_RUN_DIR="$shard_dir" \
    SOCCER_ARTIFACT_PATH="$shard_dir/artifact.json" \
    SOCCER_CHECKPOINT_ARTIFACT_PATH="$shard_dir/checkpoint-policy.json" \
    SOCCER_LEARNED_PARAMS_PATH="$shard_dir/learned-params.json" \
    SOCCER_EPISODE_LOG_PATH="$shard_dir/episodes.jsonl" \
    "$binary" > "$shard_dir/stdout.log" 2> "$shard_dir/stderr.log"
  printf 'finished shard %s/%s -> %s\n' "$shard_index" "$shards" "$shard_dir"
}

status=0
pids=()
running=0

for (( shard_index = 0; shard_index < shards; shard_index += 1 )); do
  if (( parallel_shards == 1 )); then
    if ! run_shard "$shard_index"; then
      status=1
    fi
    continue
  fi

  run_shard "$shard_index" &
  pids+=("$!")
  running=$((running + 1))

  if (( running >= parallel_shards )); then
    batch_status=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        batch_status=1
      fi
    done
    pids=()
    running=0
    if (( batch_status != 0 )); then
      status=1
    fi
  fi
done

if (( running > 0 )); then
  batch_status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      batch_status=1
    fi
  done
  if (( batch_status != 0 )); then
    status=1
  fi
fi

printf 'soccer self-play run complete: %s\n' "$out_root"
exit "$status"
