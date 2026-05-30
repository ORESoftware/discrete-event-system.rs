//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/simulated-annealing-adapter.ts`
//! Rust target: `src/des/general/adapters/simulated_annealing_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/simulated-annealing-adapter.ts",
    "src/des/general/adapters/simulated_annealing_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/simulated_annealing_adapter.rs`.", "RUST MIGRATION: Convert the simulated-annealing adapter registration and cooling schedule handling into structs/functions around `DESModelSpec`.", "RUST MIGRATION: Promote annealing params, cooling specs, progress traces, and adapter results to `serde` config/result structs; output paths become `PathBuf`.", "RUST MIGRATION: Return `Result<_, ValidationError>` for invalid temperature, cooling, neighbor, and objective settings."],
    &[],
);
