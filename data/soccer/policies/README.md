# Soccer Policy Snapshots

Curated MDP/POMDP team-policy snapshots can be saved here from the live soccer UI.
Runtime autosaves stay under `out/` and are ignored by git; files in this directory are intended for deliberate version-controlled checkpoints.

Use the UI's action/target/min-visit caps before saving a snapshot here. Snapshot saves apply those caps to a copy of the live policy so the runtime policy can keep learning while the checked-in artifact stays reviewable.

Each saved policy response also reports a `historyPath`. That file is JSONL, with one compact record per save/autosave/checkpoint so learning progress can be tracked over time without storing every transition in the curated snapshot.

The live server exposes recent history through `/api/team-policy/history`, and the UI's Policy History panel reads the same route.

For now the live soccer learner persists policy weights as JSON artifacts on disk. Q-values, target-grid Q-values, and visit counts are saved; action probabilities are derived at runtime from those values. Postgres policy storage is not wired into this simulator yet.
