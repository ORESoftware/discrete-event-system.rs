# Soccer Self-Play Learning Jobs

This bundle runs the accelerated soccer MDP/POMDP learner as isolated shards.
Each shard plays sequential 90-minute games with two 45-minute halves, updates
its own policy artifact, and writes a separate per-game JSONL log.
The launch scripts and Kubernetes job default to `SOCCER_DT_SECONDS=1.0`, so
each full match is 5,400 simulation ticks.
Checkpoint artifacts default to every 10 completed games per shard via
`SOCCER_CHECKPOINT_INTERVAL_GAMES=10`; set it to `0` to write only the final
artifact.
Artifact exports default to the top 10,000 entries per home/away policy table
via `SOCCER_ARTIFACT_MAX_ENTRIES_PER_POLICY=10000`; learning still uses the
full in-memory tables during a run.
The default tactical reward preset encourages wider attacks, flank-lane use,
and slightly more compact defending. The final `artifact.json` stores both the
learned home/away Q-policy tables and the tactical reward weights.

## Local Overnight Run

From the repository root:

```bash
SOCCER_GAMES=100 \
  scripts/soccer_self_play_local.sh
```

To hand the same run to macOS `launchd` so it keeps running after the shell or
Codex session exits:

```bash
SOCCER_GAMES=100 \
  scripts/soccer_self_play_launchd.sh
```

Outputs are written under `out/soccer-self-play/<run-id>/shard-N-of-M/`:

- `artifact.json`: learned home/away policy weights after that shard's games.
- `learned-params.json`: compact reusable Rust/serde params with tactical
  weights plus home/away Q-policy and target weights.
- `episodes.jsonl`: one episode summary per game for that shard.
- `stdout.log` and `stderr.log`: runner output for the shard.

Monitor the latest run, or pass a specific run directory:

```bash
scripts/soccer_self_play_status.sh
scripts/soccer_self_play_status.sh out/soccer-self-play/<run-id>
```

Verify checkpoint or final artifacts after they appear:

```bash
scripts/soccer_self_play_verify_artifacts.js out/soccer-self-play/<run-id>
```

To continue learning from a previous artifact or learned-params file without
mixing new results into the old output directory:

```bash
SOCCER_RESUME_ARTIFACT_PATH=out/soccer-self-play/<run-id>/shard-0-of-1/learned-params.json \
SOCCER_RUN_ID=resume-$(date -u +%Y%m%dT%H%M%SZ) \
SOCCER_GAMES=100 \
  scripts/soccer_self_play_local.sh
```

For a bigger local run, add shards explicitly:

```bash
SOCCER_GAMES=100 SOCCER_SHARDS=4 SOCCER_PARALLEL_SHARDS=4 \
  scripts/soccer_self_play_local.sh
```

## des-rs Server Run

The live soccer server also exposes synchronous self-play training at
`/api/train-self-play`. The protected des-rs deployment is normally addressed
through `/des-rs`, so the helper defaults to that base URL and saves the server
response plus extracted artifact and learned params locally:

```bash
DES_RS_AUTH=<auth-value> SOCCER_GAMES=100 \
  scripts/soccer_self_play_server.sh
```

To point the same helper at a local live server:

```bash
DES_RS_URL=http://127.0.0.1:6969 SOCCER_GAMES=100 \
  scripts/soccer_self_play_server.sh
```

## Build And Push Image

Set the image tag to a registry reachable from the EC2 Kubernetes cluster:

```bash
docker build -f Dockerfile.soccer-learning -t ghcr.io/ores/discrete-event-system.rs:soccer-learning .
docker push ghcr.io/ores/discrete-event-system.rs:soccer-learning
```

Update `k8s/soccer-self-play/job.yaml` if you use a different image tag.

## Kubernetes Run

The job uses Kubernetes Indexed Jobs. The pod index becomes
`SOCCER_SHARD_INDEX`, and `SOCCER_SHARD_COUNT` must match `spec.completions`.
The default manifest runs one 100-game shard. To shard a larger run, update
`spec.completions`, `spec.parallelism`, and `SOCCER_SHARD_COUNT` together.

The PVC requests `ReadWriteMany`; on EKS this usually means an EFS-backed
storage class. If your cluster only has EBS `ReadWriteOnce`, either install/use
an RWX class or reduce `parallelism` to `1`.

```bash
kubectl apply -k k8s/soccer-self-play
kubectl -n soccer-learning get job,pods,pvc
kubectl -n soccer-learning logs -f job/soccer-self-play
```

Artifacts are stored in the PVC under:

```text
/work/out/soccer-self-play/<job-controller-uid>/shard-N-of-M/
```

For a resumed run, add this env var to the job container and point it at an
artifact already available on the mounted PVC:

```yaml
- name: SOCCER_RESUME_ARTIFACT_PATH
  value: /work/out/soccer-self-play/<run-id>/shard-0-of-1/artifact.json
```
