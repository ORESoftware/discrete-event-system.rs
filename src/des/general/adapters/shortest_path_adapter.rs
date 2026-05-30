//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/shortest-path-adapter.ts`
//! Rust target: `src/des/general/adapters/shortest_path_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/shortest-path-adapter.ts",
    "src/des/general/adapters/shortest_path_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/shortest_path_adapter.rs`.", "RUST MIGRATION: Convert shortest-path adapter registration and builtin graph handling into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map graph edges, source/target params, paths, and metrics to `serde` config/result structs; output paths become `PathBuf`.", "RUST MIGRATION: Use `Result<_, ValidationError>` for malformed graph, missing node, and negative/invalid weight errors."],
    &[],
);
