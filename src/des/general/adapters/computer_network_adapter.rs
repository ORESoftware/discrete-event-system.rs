//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/computer-network-adapter.ts`
//! Rust target: `src/des/general/adapters/computer_network_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/computer-network-adapter.ts",
    "src/des/general/adapters/computer_network_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/computer_network_adapter.rs`.", "RUST MIGRATION: Convert the computer-network adapter and built-in problem selection into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map network nodes, links, flows, params, and results to `serde` config/result structs; file/runtime paths become `PathBuf`.", "RUST MIGRATION: Express schema validation and builtin lookup failures as `Result<_, ValidationError>`."],
    &[],
);
