//! **Scheduler-enforced Fibonacci** — a new, independent variant of
//! `crate::des::main_fibonacci_recursion` that does not rely on an implicit,
//! hand-ordered node list for correctness.
//!
//! The original model is deterministic *by accident of insertion order*: its
//! `Vec<dyn Entity>` happens to list `A, B, C, D`, and the `C → B` feedback only
//! lands in the same tick because `C` is stepped after `B`. Nothing enforces or
//! checks that invariant.
//!
//! This module keeps the same recurrence but routes every tick through a
//! [`scheduler::DeterministicScheduler`] (the "enforcer"): the graph topology is
//! declared explicitly, the per-tick execution order is *derived* from it by a
//! deterministic topological sort, and the schedule is *validated* before a
//! single tick runs. See `README.md` in this folder for how the enforcer works.
//!
//! Additive: this does not modify the original model or any shared entity.

pub mod model;
pub mod scheduler;

pub use model::{build_and_run, run, FibonacciRun};
