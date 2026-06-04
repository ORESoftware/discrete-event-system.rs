#!/usr/bin/env python3
import argparse
import json


def env(name, value=None, value_from=None):
    item = {"name": name}
    if value_from is not None:
        item["valueFrom"] = value_from
    else:
        item["value"] = value
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", default="dd-soccer-learning-rds-smoke-20260604a")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--patched-at", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-build", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-auth-header", required=True)
    args = parser.parse_args()

    bootstrap = f"""set -euo pipefail
export PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
if ! command -v curl >/dev/null 2>&1; then
  apt-get update
  apt-get install -y curl ca-certificates
fi
run_tag={args.run_id}
source_build={args.source_build}
source_sha={args.source_sha256}
source_url="https://54.91.17.58/builds/${{source_build}}/logs"
work="/tmp/${{run_tag}}-src"
archive="/tmp/${{run_tag}}.tar.gz"
mkdir -p "${{work}}"
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "runner=$(hostname)"
echo "source_build=${{source_build}}"
curl -k -fsSL -H "${{SOCCER_SOURCE_AUTH_HEADER}}" "${{source_url}}" | base64 -d > "${{archive}}"
echo "${{source_sha}}  ${{archive}}" | sha256sum -c -
tar -xzf "${{archive}}" -C "${{work}}"
cd "${{work}}"
exec bash k8s/run-soccer-learning-rds-smoke.sh
"""

    rds_secret_ref = {
        "secretKeyRef": {
            "name": "dd-remote-rest-api-secrets",
            "key": "RDS_DATABASE_URL",
        }
    }
    patch = {
        "metadata": {
            "annotations": {
                "codex.ores/patched-at": args.patched_at,
                "codex.ores/run-status": "patched",
                "codex.ores/run-stage": "waiting-for-rollout",
                "codex.ores/run-id": args.run_id,
                "codex.ores/source-build": args.source_build,
                "codex.ores/source-sha256": args.source_sha256,
            }
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "codex.ores/restarted-at": args.patched_at,
                        "codex.ores/run-id": args.run_id,
                        "codex.ores/source-build": args.source_build,
                    }
                },
                "spec": {
                    "serviceAccountName": "dd-remote-rest-api",
                    "automountServiceAccountToken": True,
                    "containers": [
                        {
                            "name": "soccer-learning",
                            "image": "docker.io/library/rust:1.90-bookworm",
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc"],
                            "args": [bootstrap],
                            "env": [
                                env("HOME", "/tmp"),
                                env("CARGO_HOME", "/tmp/cargo"),
                                env("CARGO_TARGET_DIR", f"/tmp/{args.run_id}-target"),
                                env("CARGO_BUILD_JOBS", "1"),
                                env("SOCCER_DATABASE_URL", value_from=rds_secret_ref),
                                env("SOCCER_RUN_ID", args.run_id),
                                env("SOCCER_EXPERIMENT_SLUG", "soccer-self-play-k8s-smoke"),
                                env("SOCCER_EXPERIMENT_NAME", "Soccer self-play k8s smoke"),
                                env("SOCCER_GAMES", "3"),
                                env("SOCCER_PARALLEL_GAMES", "3"),
                                env("SOCCER_HALVES", "2"),
                                env("SOCCER_HALF_MINUTES", "45"),
                                env("SOCCER_MINUTES", "90"),
                                env("SOCCER_PERIOD_BREAK_RECOVERY_SECONDS", "900"),
                                env("SOCCER_DT_SECONDS", "5"),
                                env("SOCCER_LEARNING_INTERVAL_TICKS", "4"),
                                env("SOCCER_CHECKPOINT_INTERVAL_GAMES", "0"),
                                env("SOCCER_GAME_ARTIFACT_MODE", "summary"),
                                env("SOCCER_WRITE_GAME_ARTIFACTS", "false"),
                                env("SOCCER_WRITE_FINAL_ARTIFACTS", "false"),
                                env("SOCCER_WRITE_CHECKPOINT_ARTIFACTS", "false"),
                                env("SOCCER_RUN_DIR", f"/tmp/{args.run_id}"),
                                env("SOCCER_ARTIFACT_PATH", "/dev/null"),
                                env("SOCCER_CHECKPOINT_ARTIFACT_PATH", "/dev/null"),
                                env("SOCCER_EPISODE_LOG_PATH", "/dev/null"),
                                env("SOCCER_LEARNED_PARAMS_PATH", "/dev/null"),
                                env("SOCCER_K8S_DEPLOYMENT_NAME", args.deployment),
                                env("SOCCER_K8S_NAMESPACE", args.namespace),
                                env("SOCCER_SOURCE_BUILD", args.source_build),
                                env("SOCCER_SOURCE_SHA256", args.source_sha256),
                                env("SOCCER_SOURCE_AUTH_HEADER", args.source_auth_header),
                                env("SOCCER_KEEPALIVE_SECONDS", "86400"),
                            ],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "1Gi"},
                                "limits": {"cpu": "2", "memory": "4Gi"},
                            },
                            "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
                        }
                    ],
                    "volumes": [{"name": "tmp", "emptyDir": {}}],
                },
            }
        },
    }
    print(json.dumps(patch, separators=(",", ":")))


if __name__ == "__main__":
    main()
