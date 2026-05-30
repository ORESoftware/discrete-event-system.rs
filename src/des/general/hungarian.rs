//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/hungarian.ts`
//! Rust target: `src/des/general/hungarian.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/hungarian.ts",
    "src/des/general/hungarian.rs",
    &["RUST MIGRATION: target module src/des/general/hungarian.rs.", "RUST MIGRATION: AssignmentDirection becomes an enum and AssignmentResult becomes a serde struct.", "RUST MIGRATION: hungarian is a pure solver and should stay a free function returning Result<AssignmentResult, Error> for malformed matrices.", "RUST MIGRATION: Use Vec<Vec<f64>> for cost matrices and keep rectangular padding/dual arrays explicit."],
    &["AssignmentDirection", "AssignmentResult", "hungarian"],
);
