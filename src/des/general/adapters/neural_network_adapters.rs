//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/neural-network-adapters.ts`
//! Rust target: `src/des/general/adapters/neural_network_adapters.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/neural-network-adapters.ts",
    "src/des/general/adapters/neural_network_adapters.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/neural_network_adapters.rs`.", "RUST MIGRATION: Convert XOR, neural-Q, and neural-ODE adapter registrations into structs/functions around `DESModelSpec`.", "RUST MIGRATION: Represent network configs, samples, traces, learned weights, and run results as `serde` config/result structs; output paths become `PathBuf`.", "RUST MIGRATION: Use `Result<_, ValidationError>` for layer-shape, sample, and hyperparameter validation."],
    &[],
);
