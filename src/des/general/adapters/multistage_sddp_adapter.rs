//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/multistage-sddp-adapter.ts`
//! Rust target: `src/des/general/adapters/multistage_sddp_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/multistage-sddp-adapter.ts",
    "src/des/general/adapters/multistage_sddp_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/multistage_sddp_adapter.rs`.", "RUST MIGRATION: Convert the multi-stage SDDP adapter registration into Rust adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map stage demand outcomes, stochastic program config, cuts, and run results to `serde` structs; filesystem paths become `PathBuf`.", "RUST MIGRATION: Express bad stage data, probabilities, and scenario validation as `Result<_, ValidationError>`."],
    &[],
);
