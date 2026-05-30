//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/belief-state.ts`
//! Rust target: `src/des/general/des_base/belief_state.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/belief-state.ts",
    "src/des/general/des_base/belief_state.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/belief_state.rs",
        "- Keep file-for-file. ActionObservationToken and BeliefToken become token",
        "- BeliefStateStation becomes a trait plus shared state struct holding the",
        "- Bayesian update helpers can be private/associated functions; if exposed as",
        "- Convert invalid dimensions and normalization failures from throws to Result.",
    ],
    &[
        "ActionObservationToken",
        "BeliefStateStation",
        "BeliefToken",
        "POMDPCore",
    ],
);
