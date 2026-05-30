//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/milp-bnb-adapter.ts`
//! Rust target: `src/des/general/adapters/milp_bnb_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/milp-bnb-adapter.ts",
    "src/des/general/adapters/milp_bnb_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/milp_bnb_adapter.rs`.", "RUST MIGRATION: Convert MILP branch-and-bound and IP/MIP DES registrations into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Promote MILP params, branch nodes, incumbent traces, and solutions to `serde` config/result structs; output paths become `PathBuf`.", "RUST MIGRATION: Return `Result<_, ValidationError>` for dimension mismatches, invalid bounds, and solver validation failures."],
    &[],
);
