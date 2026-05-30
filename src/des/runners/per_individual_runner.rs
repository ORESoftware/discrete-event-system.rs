//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/per-individual-runner.ts`
//! Rust target: `src/des/runners/per_individual_runner.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/per-individual-runner.ts",
    "src/des/runners/per_individual_runner.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/per_individual_runner.rs.", "- Keep file-for-file as a library runner exposing run_per_individual_once over migrated DES entity traits.", "- Convert graph construction into builder structs; processor callbacks should become PureTransform-style trait implementations where possible.", "- Isolate mathjs, JsonlLogger, and RNG dependencies behind explicit traits so this runner can share Rust types with framework_runner."],
    &["runPerIndividualOnce"],
);
