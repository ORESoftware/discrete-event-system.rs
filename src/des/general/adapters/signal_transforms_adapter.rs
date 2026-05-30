//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/signal-transforms-adapter.ts`
//! Rust target: `src/des/general/adapters/signal_transforms_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/signal-transforms-adapter.ts",
    "src/des/general/adapters/signal_transforms_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/signal_transforms_adapter.rs`.", "RUST MIGRATION: Convert z/Laplace/Fourier transform adapters, zod validation, and animation helpers into structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map transform configs, complex points, contributions, traces, and frames to `serde` config/result structs; paths become `PathBuf`.", "RUST MIGRATION: Replace zod/schema failures and expression/sample validation with `Result<_, ValidationError>`."],
    &[],
);
