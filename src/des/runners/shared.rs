//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/shared.ts`
//! Rust target: `src/des/runners/shared.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/shared.ts",
    "src/des/runners/shared.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/shared.rs.", "- Keep this as the shared runner utility module; TransitionCounter becomes a struct with impl methods.", "- Map TransitionCountMap to HashMap/BTreeMap depending on deterministic iteration needs, and serialize TransitionTables with serde.", "- Keep table builders and population aggregators as private/public pure functions unless lifted into PureTransform-style traits."],
    &["TRANSITION_MATRIX_COLS", "TRANSITION_MATRIX_ROWS", "TransitionCountMap", "TransitionCounter", "TransitionTables", "analyticalTransitionTables", "averageRecord", "buildTransitionTables", "compartmentPopulations", "meanResidence", "updatePeaks", "zeroCompartmentRecord"],
);
