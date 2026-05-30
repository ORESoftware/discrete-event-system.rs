//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/internal-solver-network-adapter.ts`
//! Rust target: `src/des/general/adapters/internal_solver_network_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/internal-solver-network-adapter.ts",
    "src/des/general/adapters/internal_solver_network_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/internal_solver_network_adapter.rs`.", "RUST MIGRATION: Convert internal solver-network adapter registration and progress-frame drawing into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Model graph, shortest-path, knapsack, TSP, cooling, and progress records as `serde` config/result structs; output paths become `PathBuf`.", "RUST MIGRATION: Use `Result<_, ValidationError>` for solver selection, schema, and graph/input validation errors."],
    &[],
);
