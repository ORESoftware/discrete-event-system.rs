//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/fel-runner.ts`
//! Rust target: `src/des/runners/fel_runner.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/fel-runner.ts",
    "src/des/runners/fel_runner.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/fel_runner.rs.", "- Keep file-for-file as a library runner exposing run_fel_once; FelEvent becomes an ordered event struct.", "- Replace Math.random fallbacks with an injected RNG trait and keep with_seed behavior as a deterministic RNG adapter.", "- Convert throw/implicit failure points to Result only at construction/logging boundaries; event-loop helpers can stay private."],
    &["runFelOnce"],
);
