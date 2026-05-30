//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/program.ts`
//! Rust target: `src/des/program.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/program.ts",
    "src/des/program.rs",
    &["RUST MIGRATION: target src/des/program.rs.", "RUST MIGRATION: Keep this as a library-ish orchestration module; if runnable behavior remains, add a thin src/bin/program.rs wrapper.", "RUST MIGRATION: Map classes/interfaces to structs/traits and keep visual/runtime effects behind explicit ports returning Result.", "RUST MIGRATION: Use clap/std::env/PathBuf only at wrapper boundaries and keep JSON examples/config as serde-deserialized structs."],
    &["getEntities"],
);
