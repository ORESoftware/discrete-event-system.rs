# Soccer Learning Job

Runs full 90-minute self-play games as two 45-minute halves and writes isolated
per-game artifacts plus a final MDP/POMDP policy artifact.

## Local overnight run

```bash
cargo build --release --bin main_soccer_learning_run
SOCCER_GAMES=100 \
SOCCER_PARALLEL_GAMES=4 \
SOCCER_MINUTES=90 \
SOCCER_HALVES=2 \
SOCCER_HALF_MINUTES=45 \
SOCCER_RUN_DIR=out/soccer-learning-runs/overnight-local \
target/release/main_soccer_learning_run
```

Outputs:

- `SOCCER_RUN_DIR/games/game-*.json`: one compact summary artifact per game
  by default, keeping results isolated without writing hundreds of MB per game.
- `SOCCER_RUN_DIR/checkpoint-policy.json`: merged learned home/away policy
  weights after each completed batch.
- `SOCCER_RUN_DIR/final-policy.json`: merged learned home/away policy weights.
- `SOCCER_RUN_DIR/manifest.json`: run metadata and per-game paths.

Use `SOCCER_RESUME_ARTIFACT=/path/to/final-policy.json` to continue learning
from a previous run without mixing output directories.
Set `SOCCER_GAME_ARTIFACT_MODE=full` only when you explicitly need every
per-game policy table and have enough disk for very large artifacts.

## Kubernetes run

Build and push the trainer image, then point the kustomization at that image:

```bash
docker build -f Dockerfile.soccer-learning -t <registry>/des-soccer-learning:latest .
docker push <registry>/des-soccer-learning:latest
cd k8s/soccer-learning
kustomize edit set image des-soccer-learning=<registry>/des-soccer-learning:latest
kubectl apply -k .
```

The job writes artifacts to the `soccer-learning-artifacts` PVC under
`/data/soccer-learning/$SOCCER_RUN_ID`. Adjust `SOCCER_GAMES` and
`SOCCER_PARALLEL_GAMES` in `job.yaml` for overnight EC2 capacity.
