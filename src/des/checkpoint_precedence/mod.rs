//! **Checkpoint-precedence ordering** — a token-level ordering enforcer, a
//! complement to the node-level `crate::des::fibonacci_scheduled` scheduler.
//!
//! Where the `DeterministicScheduler` decides *which station runs when*, this
//! module decides *which movable may pass a point, relative to other movables*.
//! Tokens are stamped with a UUID (reusing the movable's existing `moving_uuid`)
//! and a monotonic `seq`, and they declare — by UUID reference — which other
//! tokens must clear a checkpoint before they may pass it. A [`gate`-style
//! station](entities::CheckpointGate) holds arriving tokens in a balanced BST
//! keyed by `seq` and releases them in the deterministic order that satisfies
//! those constraints. See `README.md` in this folder.
//!
//! Additive: reuses the existing movable/entity framework without modifying it.

pub mod entities;
pub mod ledger;
pub mod model;
// A real computation built on the same gate: dependency-ordered task execution
// (a build/job scheduler). See `task_dag` and the README.
pub mod task_dag;

pub use model::{build_and_run, run, CheckpointRun};
