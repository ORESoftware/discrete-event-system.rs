# Soccer Policy Snapshots

Curated MDP/POMDP team-policy snapshots can be saved here from the live soccer UI.
Runtime autosaves stay under `out/` and are ignored by git; files in this directory are intended for deliberate version-controlled checkpoints.

Use the UI's action/target/min-visit caps before saving a snapshot here. Snapshot saves apply those caps to a copy of the live policy so the runtime policy can keep learning while the checked-in artifact stays reviewable.

Each saved policy response also reports a `historyPath`. That file is JSONL, with one compact record per save/autosave/checkpoint so learning progress can be tracked over time without storing every transition in the curated snapshot.

The live server exposes recent history through `/api/team-policy/history`, and the UI's Policy History panel reads the same route.

The live UI snapshot path still writes JSON artifacts on disk. Batch self-play and set-play restart learning can also persist to Postgres when a supported database URL env var is present; those runs store policy versions, action/target Q-value rows, run summaries, and neural-learning metrics in the `des_soccer_learning_*` tables.
