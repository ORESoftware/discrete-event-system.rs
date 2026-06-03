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
base_url="${DES_RS_URL:-https://54.91.17.58/des-rs}"
base_url="${base_url%/}"
endpoint="${DES_RS_TRAIN_URL:-$base_url/api/train-self-play}"
auth_header_name="${DES_RS_AUTH_HEADER_NAME:-Auth}"
auth_value="${DES_RS_AUTH:-}"

if [[ -z "$auth_value" && "$endpoint" == https://54.91.17.58* ]]; then
  echo "DES_RS_AUTH must be set for the protected des-rs endpoint." >&2
  exit 2
fi

mkdir -p "$shard_out"
cat > "$local_out/run.env" <<EOF
run_id=$run_id
mode=server
endpoint=$endpoint
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
response_path="${SOCCER_SERVER_RESPONSE_PATH:-$shard_out/response.json}"
artifact_path="${SOCCER_SERVER_LOCAL_ARTIFACT_PATH:-$shard_out/artifact.json}"
episode_log_path="${SOCCER_SERVER_EPISODE_LOG_PATH:-$shard_out/episodes.jsonl}"

python3 - "$server_artifact_path" > "$payload_path" <<'PY'
import json
import os
import sys

def as_float(name: str) -> float:
    return float(os.environ[name])

def as_int(name: str) -> int:
    return int(os.environ[name])

def as_bool(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in {"0", "false", "no", "off"}

artifact_path = sys.argv[1]
payload = {
    "episodes": as_int("SOCCER_GAMES"),
    "minutes": as_float("SOCCER_MINUTES"),
    "periodCount": as_int("SOCCER_HALVES"),
    "periodBreakRecoverySeconds": as_float("SOCCER_PERIOD_BREAK_RECOVERY_SECONDS"),
    "dtSeconds": as_float("SOCCER_DT_SECONDS"),
    "learningIntervalTicks": as_int("SOCCER_LEARNING_INTERVAL_TICKS"),
    "seed": as_int("SOCCER_SEED"),
    "options": {
        "alpha": as_float("SOCCER_ALPHA"),
        "gamma": as_float("SOCCER_GAMMA"),
    },
    "tacticalLearning": {
        "attackSpacingDeltaWeight": as_float("SOCCER_ATTACK_SPACING_DELTA_WEIGHT"),
        "attackSpacingScoreWeight": as_float("SOCCER_ATTACK_SPACING_SCORE_WEIGHT"),
        "attackWidthDeltaWeight": as_float("SOCCER_ATTACK_WIDTH_DELTA_WEIGHT"),
        "attackWidthScoreWeight": as_float("SOCCER_ATTACK_WIDTH_SCORE_WEIGHT"),
        "attackFlankLaneWeight": as_float("SOCCER_ATTACK_FLANK_LANE_WEIGHT"),
        "defenseSpacingDeltaWeight": as_float("SOCCER_DEFENSE_SPACING_DELTA_WEIGHT"),
        "defenseSpacingScoreWeight": as_float("SOCCER_DEFENSE_SPACING_SCORE_WEIGHT"),
        "defenseContractDeltaWeight": as_float("SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT"),
        "defenseCompactnessScoreWeight": as_float("SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT"),
    },
    "artifactPath": artifact_path,
    "importIntoSession": as_bool("SOCCER_IMPORT_INTO_SESSION"),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

curl_args=(
  -fsS
  -X POST
  "$endpoint"
  -H "Content-Type: application/json"
  --data-binary "@$payload_path"
  -o "$response_path"
)
if [[ -n "$auth_value" ]]; then
  curl_args+=(-H "$auth_header_name: $auth_value")
fi

curl "${curl_args[@]}"

python3 - "$response_path" "$artifact_path" "$episode_log_path" <<'PY'
import json
import pathlib
import sys

response_path = pathlib.Path(sys.argv[1])
artifact_path = pathlib.Path(sys.argv[2])
episode_log_path = pathlib.Path(sys.argv[3])
response = json.loads(response_path.read_text())
if response.get("ok") is False:
    raise SystemExit(f"server returned error response: {response.get('error', response)}")
artifact = response.get("artifact")
if not isinstance(artifact, dict):
    raise SystemExit("server response did not include an artifact object")
artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
episodes = artifact.get("episodes") or []
episode_log_path.write_text(
    "".join(json.dumps(episode, sort_keys=True) + "\n" for episode in episodes)
)
print(f"server_response={response_path}")
print(f"artifact={artifact_path}")
print(f"episode_log={episode_log_path}")
print(f"episodes={len(episodes)}")
print(f"home_entries={len(artifact.get('homeEntries') or [])}")
print(f"away_entries={len(artifact.get('awayEntries') or [])}")
PY
