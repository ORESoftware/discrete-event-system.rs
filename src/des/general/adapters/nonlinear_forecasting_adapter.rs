//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/nonlinear-forecasting-adapter.ts`
//! Rust target: `src/des/general/adapters/nonlinear_forecasting_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/nonlinear-forecasting-adapter.ts",
    "src/des/general/adapters/nonlinear_forecasting_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/nonlinear_forecasting_adapter.rs`.", "RUST MIGRATION: Convert nonlinear forecasting adapter registration and animation helpers into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map forecast params, station panels, traces, and frame captions to `serde` config/result structs; runtime paths become `PathBuf`.", "RUST MIGRATION: Return `Result<_, ValidationError>` for invalid horizon, station, and forecast input validation."],
    &[],
);
