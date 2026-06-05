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

## Free-Kick Restart Learning

Use the set-play runner to repeat 10-second indirect and direct free-kicks from
25 yards and train both the MDP/POMDP Q-policy and the bounded neural gradient
learner. Indirect free-kicks are trained first by default:

```bash
SOCCER_SET_PLAY_RUN_ID=free-kick-25y-audit \
SOCCER_SET_PLAY_EPISODES=100 \
cargo run --release --bin main_soccer_set_play_learning_run
```

Useful overrides:

```bash
SOCCER_FREE_KICK_DISTANCE_YARDS=25 \
SOCCER_SET_PLAY_DURATION_SECONDS=10 \
SOCCER_SET_PLAY_RESTARTS=indirect-free-kick,direct-free-kick \
SOCCER_NEURAL_LEARNING_ENABLED=true \
SOCCER_NEURAL_LEARNING_BACKEND=threaded \
SOCCER_RESUME_POSTGRES_POLICY=true \
cargo run --release --bin main_soccer_set_play_learning_run
```

When one of `SOCCER_DATABASE_URL`, `AGENT_TASKS_RDS_DATABASE_URL`,
`RDS_DATABASE_URL`, `DATABASE_URL`, or `PG_DATABASE_URL` is present, the runner
loads the latest active policy for `SOCCER_EXPERIMENT_SLUG` before each newly
submitted simulation and writes the result back to Postgres. The refresh includes
MDP/POMDP policy entries, neural-network snapshots, and tactical-learning
weights, so locally started runs and des-rs runs converge on the same active
Postgres head while still batching completed-run writes. Set-play restart
training uses the same refresh path before each episode and archives stale
output policies if a newer active head lands before the run is persisted.
Evolutionary policy search is enabled by default; tune it with
`SOCCER_EVOLUTION_INTERVAL_GAMES`,
`SOCCER_EVOLUTION_ELITE_GAMES`, `SOCCER_EVOLUTION_MUTATION_RATE`,
`SOCCER_EVOLUTION_MUTATION_SCALE`, `SOCCER_EVOLUTION_CROSSOVER_RATE`,
`SOCCER_EVOLUTION_EXPLORATION_RATE`, and
`SOCCER_EVOLUTION_EXPLORATION_SCALE`. The portable JSONB summaries remain on the
main run and policy rows, but the queryable learning facts are normalized into:

- `des_soccer_learning_policy_versions` and
  `des_soccer_learning_policy_entries` for MDP/POMDP action and target Q-values;
- `des_soccer_learning_runs` for the completed training run;
- `des_soccer_learning_set_play_runs` for typed restart-training run metrics;
- `des_soccer_learning_set_play_restart_mix` for the direct/indirect restart
  schedule;
- `des_soccer_learning_set_play_episode_metrics` for per-episode goal,
  policy-update, and restart metrics;
- `des_soccer_learning_neural_run_metrics` for neural gradient steps, sample
  counts, replay stats, parameter count, and loss.

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
server response before writing local artifacts. When Postgres resume/import is
enabled, it also imports the returned `learnedParams` as a new
`des_soccer_learning_policy_versions` row with `source_kind = 'import'`; stale
remote results are archived instead of replacing a newer active policy.
