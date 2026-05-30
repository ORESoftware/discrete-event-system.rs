//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/advanced-optimization-control-adapter.ts`
//! Rust target: `src/des/general/adapters/advanced_optimization_control_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/advanced-optimization-control-adapter.ts",
    "src/des/general/adapters/advanced_optimization_control_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/advanced_optimization_control_adapter.rs`.", "RUST MIGRATION: Convert the advanced optimization/control registrations into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map parameter schemas, optimizer configs, and run summaries to `serde` config/result structs; paths become `PathBuf`.", "RUST MIGRATION: Replace validation throws or rejected params with `Result<_, ValidationError>`."],
    &[],
);
