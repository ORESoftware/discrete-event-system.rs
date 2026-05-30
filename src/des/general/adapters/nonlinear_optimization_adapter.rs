//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/nonlinear-optimization-adapter.ts`
//! Rust target: `src/des/general/adapters/nonlinear_optimization_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/nonlinear-optimization-adapter.ts",
    "src/des/general/adapters/nonlinear_optimization_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/nonlinear_optimization_adapter.rs`.", "RUST MIGRATION: Convert nonlinear optimization adapter registrations and visualization helpers into structs/functions around `DESModelSpec`.", "RUST MIGRATION: Promote objective configs, curve points, solver traces, and results to `serde` config/result structs; output paths become `PathBuf`.", "RUST MIGRATION: Use `Result<_, ValidationError>` for dimension, expression, and convergence-input validation."],
    &[],
);
