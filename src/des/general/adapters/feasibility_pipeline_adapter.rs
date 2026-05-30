//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/feasibility-pipeline-adapter.ts`
//! Rust target: `src/des/general/adapters/feasibility_pipeline_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/feasibility-pipeline-adapter.ts",
    "src/des/general/adapters/feasibility_pipeline_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/feasibility_pipeline_adapter.rs`.", "RUST MIGRATION: Convert feasibility pipeline adapter registration and drawing helpers into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Promote variables, constraints, candidate evaluations, and improvements to `serde` config/result structs; paths become `PathBuf`.", "RUST MIGRATION: Make constraint validation and infeasible candidate handling explicit with `Result<_, ValidationError>`."],
    &[],
);
