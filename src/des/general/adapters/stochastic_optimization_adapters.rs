//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/stochastic-optimization-adapters.ts`
//! Rust target: `src/des/general/adapters/stochastic_optimization_adapters.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/stochastic-optimization-adapters.ts",
    "src/des/general/adapters/stochastic_optimization_adapters.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/stochastic_optimization_adapters.rs`.", "RUST MIGRATION: Convert stochastic LP and multi-stage optimization adapter registrations into structs/functions around `DESModelSpec`.", "RUST MIGRATION: Promote pair schemas, demand outcomes, stochastic configs, cuts, and run results to `serde` config/result structs; paths become `PathBuf`.", "RUST MIGRATION: Use `Result<_, ValidationError>` for probability, scenario, vector-shape, and solver input validation."],
    &[],
);
