//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/max-flow.ts`
//! Rust target: `src/des/max_flow.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/max-flow.ts",
    "src/des/max_flow.rs",
    &["RUST MIGRATION: target src/des/max_flow.rs.", "RUST MIGRATION: Keep max-flow logic in this library module; if runnable behavior remains, add a thin src/bin/max_flow.rs wrapper.", "RUST MIGRATION: Map problem/solver shapes to structs/traits, keep the algorithmic core pure, and return Result for fallible setup.", "RUST MIGRATION: Use clap/std::env/PathBuf only at wrapper boundaries and keep JSON examples/config as serde-deserialized structs."],
    &[],
);
