//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/statistical-optimization-adapter.ts`
//! Rust target: `src/des/general/adapters/statistical_optimization_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/statistical-optimization-adapter.ts",
    "src/des/general/adapters/statistical_optimization_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/statistical_optimization_adapter.rs`.", "RUST MIGRATION: Convert statistical optimization and stochastic-LP adapter registrations into structs/functions around `DESModelSpec`.", "RUST MIGRATION: Encode distributions, empirical demand, risk params, alternatives, traces, and solutions as `serde` config/result structs; paths become `PathBuf`.", "RUST MIGRATION: Use `Result<_, ValidationError>` for normalization, probability, range, and stochastic-LP validation."],
    &[],
);
