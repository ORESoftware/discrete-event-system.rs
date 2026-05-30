//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/network-flow-adapter.ts`
//! Rust target: `src/des/general/adapters/network_flow_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/network-flow-adapter.ts",
    "src/des/general/adapters/network_flow_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/network_flow_adapter.rs`.", "RUST MIGRATION: Convert max-flow, traffic, and smart-traffic adapter registrations into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Encode graph edges, traffic networks, signals, traces, and animation data as `serde` config/result structs; paths become `PathBuf`.", "RUST MIGRATION: Use `Result<_, ValidationError>` for graph normalization, capacity, lane, and signal validation."],
    &[],
);
