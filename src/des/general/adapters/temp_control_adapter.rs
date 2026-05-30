//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/temp-control-adapter.ts`
//! Rust target: `src/des/general/adapters/temp_control_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/temp-control-adapter.ts",
    "src/des/general/adapters/temp_control_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/temp_control_adapter.rs`.", "RUST MIGRATION: Convert temperature-control adapter registration and controller descriptions into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map controller specs, plant params, telemetry, and run results to `serde` config/result structs; output/runtime paths become `PathBuf`.", "RUST MIGRATION: Return `Result<_, ValidationError>` for invalid controller gains, bounds, schedule, and thermal parameters."],
    &[],
);
