#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

deploy="${SOCCER_K8S_DEPLOYMENT_NAME:-dd-soccer-learning-rds-smoke-20260604a}"
namespace="${SOCCER_K8S_NAMESPACE:-default}"
source_build="${SOCCER_SOURCE_BUILD:-unknown}"
source_sha="${SOCCER_SOURCE_SHA256:-unknown}"
keepalive_seconds="${SOCCER_KEEPALIVE_SECONDS:-86400}"
log_path="/tmp/${SOCCER_RUN_ID:-soccer-learning}.log"

mark() {
  local status="$1"
  local stage="$2"
  local detail_b64="${3:-}"
  local sa api now payload

  if [ ! -f /var/run/secrets/kubernetes.io/serviceaccount/token ]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  sa=/var/run/secrets/kubernetes.io/serviceaccount
  api="https://${KUBERNETES_SERVICE_HOST}:${KUBERNETES_SERVICE_PORT}"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  payload="{\"metadata\":{\"annotations\":{\"codex.ores/run-status\":\"${status}\",\"codex.ores/run-stage\":\"${stage}\",\"codex.ores/run-id\":\"${SOCCER_RUN_ID:-unknown}\",\"codex.ores/source-build\":\"${source_build}\",\"codex.ores/source-sha256\":\"${source_sha}\",\"codex.ores/updated-at\":\"${now}\",\"codex.ores/run-detail-b64\":\"${detail_b64}\"}}}"

  curl -sSk --cacert "${sa}/ca.crt" \
    -H "Authorization: Bearer $(cat "${sa}/token")" \
    -H "Content-Type: application/merge-patch+json" \
    -X PATCH \
    --data "${payload}" \
    "${api}/apis/apps/v1/namespaces/${namespace}/deployments/${deploy}" >/dev/null || true
}

on_exit() {
  local code=$?
  local detail_b64=""
  if [ -f "${log_path}" ]; then
    detail_b64="$(tail -c 1800 "${log_path}" | base64 | tr -d '\n')"
  fi
  if [ "${code}" -eq 0 ]; then
    mark complete sleeping
  else
    mark failed "exit-${code}" "${detail_b64}"
  fi
  exit "${code}"
}

trap on_exit EXIT

echo "source_rev=$(git rev-parse --short HEAD 2>/dev/null || echo archive-no-git)"
echo "soccer smoke config run_id=${SOCCER_RUN_ID:-unset} games=${SOCCER_GAMES:-unset} parallel=${SOCCER_PARALLEL_GAMES:-unset} halves=${SOCCER_HALVES:-unset} half_minutes=${SOCCER_HALF_MINUTES:-unset} dt=${SOCCER_DT_SECONDS:-unset} experiment=${SOCCER_EXPERIMENT_SLUG:-unset} final_artifacts=${SOCCER_WRITE_FINAL_ARTIFACTS:-unset} checkpoint_artifacts=${SOCCER_WRITE_CHECKPOINT_ARTIFACTS:-unset}"

mark running cargo-run
set +e
/usr/local/cargo/bin/cargo run --bin main_soccer_learning_run > >(tee "${log_path}") 2>&1
cargo_status=$?
set -e
if [ "${cargo_status}" -ne 0 ]; then
  exit "${cargo_status}"
fi

trap - EXIT
mark complete sleeping
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "soccer learning smoke complete; sleeping for status inspection"
sleep "${keepalive_seconds}"
