//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/learning-optimization-adapter.ts`
//! Rust target: `src/des/general/adapters/learning_optimization_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/learning-optimization-adapter.ts",
    "src/des/general/adapters/learning_optimization_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/learning_optimization_adapter.rs`.", "RUST MIGRATION: Convert supervised/RL learning optimization registrations and report writers into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map samples, training configs, traces, and run results to `serde` config/result structs; CSV/animation paths become `PathBuf`.", "RUST MIGRATION: Return `Result<_, ValidationError>` for malformed samples, model params, and unsupported learning modes."],
    &[],
);
