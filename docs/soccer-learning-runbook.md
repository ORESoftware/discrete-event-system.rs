# Soccer Learning Runbook

The soccer self-play learner supports the same training contract locally, in a
container, and through the des-rs server adapter:

- 100 games by default.
- 2 periods of 45 minutes each.
- 0.2 second simulation ticks.
- Checkpoints every 10 completed games.
- Final policy, learned params, manifest, episode log, and per-game summaries
  stay together under the run/shard directory.

## Local

```bash
scripts/soccer_self_play_local.sh
```

Useful overrides:

```bash
SOCCER_RUN_ID=wide-flanks-audit \
SOCCER_SHARDS=4 \
SOCCER_PARALLEL_SHARDS=2 \
scripts/soccer_self_play_local.sh
```

The launcher owns per-shard output paths. Use `SOCCER_OUT_ROOT` or
`SOCCER_RUN_ID` to choose the output namespace instead of setting
`SOCCER_RUN_DIR`, `SOCCER_ARTIFACT_PATH`, `SOCCER_CHECKPOINT_ARTIFACT_PATH`,
`SOCCER_EPISODE_LOG_PATH`, or `SOCCER_LEARNED_PARAMS_PATH` directly.

Check progress and artifact integrity:

```bash
scripts/soccer_self_play_status.sh out/soccer-self-play/wide-flanks-audit
node scripts/soccer_self_play_verify_artifacts.js out/soccer-self-play/wide-flanks-audit
```

## Docker

```bash
docker build -f Dockerfile.soccer-learning -t des-soccer-learning .
docker run --rm -v "$PWD/out/docker-soccer-learning:/data/soccer-learning" \
  -e SOCCER_RUN_DIR=/data/soccer-learning/audit-100x90 \
  des-soccer-learning
```

## des-rs Server

The protected des-rs endpoint requires the `Auth` header. Keep the value in the
local environment and out of git:

```bash
DES_RS_AUTH='...' scripts/soccer_self_play_server.sh
```

The server launcher writes its request payload, raw response, extracted
artifact, learned params, and episode log under
`out/soccer-self-play/<run-id>/shard-0-of-1/`. The Rust adapter validates the
server response before writing local artifacts.
