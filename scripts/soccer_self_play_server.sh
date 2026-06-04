#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
cd "$repo_dir"

export SOCCER_GAMES="${SOCCER_GAMES:-100}"
export SOCCER_HALVES="${SOCCER_HALVES:-2}"
export SOCCER_MINUTES="${SOCCER_MINUTES:-90}"
export SOCCER_PERIOD_BREAK_RECOVERY_SECONDS="${SOCCER_PERIOD_BREAK_RECOVERY_SECONDS:-900}"
export SOCCER_DT_SECONDS="${SOCCER_DT_SECONDS:-1.0}"
export SOCCER_LEARNING_INTERVAL_TICKS="${SOCCER_LEARNING_INTERVAL_TICKS:-4}"
export SOCCER_SEED="${SOCCER_SEED:-2026}"
export SOCCER_ALPHA="${SOCCER_ALPHA:-0.20}"
export SOCCER_GAMMA="${SOCCER_GAMMA:-0.96}"
export SOCCER_ATTACK_SPACING_DELTA_WEIGHT="${SOCCER_ATTACK_SPACING_DELTA_WEIGHT:-0.22}"
export SOCCER_ATTACK_SPACING_SCORE_WEIGHT="${SOCCER_ATTACK_SPACING_SCORE_WEIGHT:-0.06}"
export SOCCER_ATTACK_WIDTH_DELTA_WEIGHT="${SOCCER_ATTACK_WIDTH_DELTA_WEIGHT:-0.52}"
export SOCCER_ATTACK_WIDTH_SCORE_WEIGHT="${SOCCER_ATTACK_WIDTH_SCORE_WEIGHT:-0.14}"
export SOCCER_ATTACK_FLANK_LANE_WEIGHT="${SOCCER_ATTACK_FLANK_LANE_WEIGHT:-0.28}"
export SOCCER_DEFENSE_SPACING_DELTA_WEIGHT="${SOCCER_DEFENSE_SPACING_DELTA_WEIGHT:-0.08}"
export SOCCER_DEFENSE_SPACING_SCORE_WEIGHT="${SOCCER_DEFENSE_SPACING_SCORE_WEIGHT:-0.04}"
export SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT="${SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT:-0.42}"
export SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT="${SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT:-0.14}"
export SOCCER_IMPORT_INTO_SESSION="${SOCCER_IMPORT_INTO_SESSION:-1}"

run_id="${SOCCER_RUN_ID:-server-$(date -u +%Y%m%dT%H%M%SZ)}"
local_out="${SOCCER_SERVER_LOCAL_OUT:-out/soccer-self-play/$run_id}"
shard_out="${SOCCER_SERVER_SHARD_OUT:-$local_out/shard-0-of-1}"
server_artifact_path="${SOCCER_SERVER_ARTIFACT_PATH:-out/soccer-self-play/$run_id/artifact.json}"
server_learned_params_path="${SOCCER_SERVER_LEARNED_PARAMS_PATH:-out/soccer-self-play/$run_id/learned-params.json}"
base_url="${DES_RS_URL:-https://54.91.17.58/des-rs}"
base_url="${base_url%/}"
endpoint="${DES_RS_TRAIN_URL:-$base_url/api/train-self-play}"
auth_header_name="${DES_RS_AUTH_HEADER_NAME:-Auth}"
auth_value="${DES_RS_AUTH:-}"

if [[ -z "$auth_value" && "$endpoint" == https://54.91.17.58* ]]; then
  echo "DES_RS_AUTH must be set for the protected des-rs endpoint." >&2
  exit 2
fi

response_path="${SOCCER_SERVER_RESPONSE_PATH:-$shard_out/response.json}"
artifact_path="${SOCCER_SERVER_LOCAL_ARTIFACT_PATH:-$shard_out/artifact.json}"
learned_params_path="${SOCCER_SERVER_LOCAL_LEARNED_PARAMS_PATH:-$shard_out/learned-params.json}"
episode_log_path="${SOCCER_SERVER_EPISODE_LOG_PATH:-$shard_out/episodes.jsonl}"
server_binary="${SOCCER_SERVER_BINARY:-target/release/main_soccer_learning_server}"
build_release="${SOCCER_SERVER_BUILD_RELEASE:-${SOCCER_BUILD_RELEASE:-1}}"

mkdir -p "$shard_out"
cat > "$local_out/run.env" <<EOF
run_id=$run_id
mode=server
endpoint=$endpoint
server_binary=$server_binary
server_artifact_path=$server_artifact_path
server_learned_params_path=$server_learned_params_path
local_artifact_path=$artifact_path
local_learned_params_path=$learned_params_path
games=$SOCCER_GAMES
halves=$SOCCER_HALVES
minutes=$SOCCER_MINUTES
period_break_recovery_seconds=$SOCCER_PERIOD_BREAK_RECOVERY_SECONDS
dt_seconds=$SOCCER_DT_SECONDS
learning_interval_ticks=$SOCCER_LEARNING_INTERVAL_TICKS
attack_spacing_delta_weight=$SOCCER_ATTACK_SPACING_DELTA_WEIGHT
attack_spacing_score_weight=$SOCCER_ATTACK_SPACING_SCORE_WEIGHT
attack_width_delta_weight=$SOCCER_ATTACK_WIDTH_DELTA_WEIGHT
attack_width_score_weight=$SOCCER_ATTACK_WIDTH_SCORE_WEIGHT
attack_flank_lane_weight=$SOCCER_ATTACK_FLANK_LANE_WEIGHT
defense_spacing_delta_weight=$SOCCER_DEFENSE_SPACING_DELTA_WEIGHT
defense_spacing_score_weight=$SOCCER_DEFENSE_SPACING_SCORE_WEIGHT
defense_contract_delta_weight=$SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT
defense_compactness_score_weight=$SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT
shards=1
parallel_shards=1
EOF
payload_path="$shard_out/payload.json"

if (( build_release != 0 )); then
  cargo build --release --bin main_soccer_learning_server
fi

"$server_binary" \
  --endpoint "$endpoint" \
  --payload "$payload_path" \
  --response "$response_path" \
  --artifact "$artifact_path" \
  --learned-params "$learned_params_path" \
  --episode-log "$episode_log_path" \
  --server-artifact-path "$server_artifact_path" \
  --server-learned-params-path "$server_learned_params_path" \
  --auth-header-name "$auth_header_name" \
  --auth-value "$auth_value"
