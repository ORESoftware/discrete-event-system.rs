//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/value-iteration.ts`
//! Rust target: `src/des/general/value_iteration.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/value-iteration.ts",
    "src/des/general/value_iteration.rs",
    &["RUST MIGRATION: Target module `src/des/general/value_iteration.rs`.", "RUST MIGRATION: Convert outcome, MDP spec/options/result interfaces to `serde` structs; use typed state/action IDs instead of raw `number` where practical.", "RUST MIGRATION: Port `ValueIterationStation` as a struct implementing the fixed-point iteration trait, storing values as `Vec<f64>` instead of `Float64Array`.", "RUST MIGRATION: Keep `valueIteration`, `qValue`, and `qValuesAll` as free solver functions; add `PureTransform` only if the solver is wired as a graph-visible block.", "RUST MIGRATION: Return `Result` for invalid probabilities, empty action sets, bad gamma/tolerance, non-finite rewards, and malformed transition outcomes."],
    &["MDPSpec", "Outcome", "VIOptions", "VIResult", "ValueIterationStation", "qValue", "qValuesAll", "valueIteration"],
);
