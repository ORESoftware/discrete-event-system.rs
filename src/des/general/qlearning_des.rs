//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/qlearning-des.ts`
//! Rust target: `src/des/general/qlearning_des.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/qlearning-des.ts",
    "src/des/general/qlearning_des.rs",
    &["RUST MIGRATION: Target module `src/des/general/qlearning_des.rs`.", "RUST MIGRATION: Convert Q-learning options and result interfaces to `serde` structs with explicit numeric state/action newtypes or `usize`.", "RUST MIGRATION: Port `QLearningAgent` as a struct implementing the `RLAgentStation` trait; inherited hooks become trait methods.", "RUST MIGRATION: Keep `runQLearningDES` as a graph runner free function, or wrap it in `PureTransform` only if it is surfaced as a graph-visible transform.", "RUST MIGRATION: Replace epsilon-greedy randomness with an injected RNG trait/closure and return `Result` for invalid learning rates, discounts, or episode counts."],
    &["QLearningAgent", "QLearningOptions", "QLearningResult", "runQLearningDES"],
);
