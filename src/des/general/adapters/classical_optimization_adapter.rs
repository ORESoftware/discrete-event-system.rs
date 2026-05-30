//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/classical-optimization-adapter.ts`
//! Rust target: `src/des/general/adapters/classical_optimization_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/classical-optimization-adapter.ts",
    "src/des/general/adapters/classical_optimization_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/classical_optimization_adapter.rs`.", "RUST MIGRATION: Convert the LP/assignment/routing/job-shop adapter registrations into Rust adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Promote param schemas and solution payloads into `serde` config/result structs; represent output files with `PathBuf`.", "RUST MIGRATION: Surface infeasible or malformed inputs as `Result<_, ValidationError>`."],
    &[],
);
