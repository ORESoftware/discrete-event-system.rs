//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/gillespie-runner.ts`
//! Rust target: `src/des/runners/gillespie_runner.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/gillespie-runner.ts",
    "src/des/runners/gillespie_runner.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/gillespie_runner.rs.", "- Keep file-for-file as a library runner exposing run_gillespie_once; Reaction becomes a concrete struct with typed station IDs.", "- Replace seedable randomness with an RNG trait/object and keep Gillespie propensity calculations as private pure helpers.", "- Convert logging and invalid config paths to Result while preserving deterministic RunResult output structs."],
    &["runGillespieOnce"],
);
