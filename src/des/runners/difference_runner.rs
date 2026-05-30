//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/difference-runner.ts`
//! Rust target: `src/des/runners/difference_runner.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/difference-runner.ts",
    "src/des/runners/difference_runner.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/difference_runner.rs.", "- Keep file-for-file as a library runner; export run_difference_once, analytical_steady_state, max_stable_step, and SteadyState.", "- Convert State and MeanResidences to private structs, preserving COMPARTMENT_ORDER indexing with typed keys or small enums.", "- Pure numerical helpers stay private module functions unless promoted into a PureTransform-style trait implementation."],
    &["SteadyState", "analyticalSteadyState", "maxStableStep", "runDifferenceOnce"],
);
