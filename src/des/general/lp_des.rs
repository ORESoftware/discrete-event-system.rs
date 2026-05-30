//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/lp-des.ts`
//! Rust target: `src/des/general/lp_des.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/lp-des.ts",
    "src/des/general/lp_des.rs",
    &["RUST MIGRATION: target module src/des/general/lp_des.rs.", "RUST MIGRATION: DESSimplexTrace, DESSimplexOptions, DESSimplexSolution, and Preprocessed become serde structs; SimplexState is a private mutable state struct.", "RUST MIGRATION: SimplexRoleStation and concrete phase/entering/leaving/pivot/observer stations become structs implementing Station traits instead of inheritance.", "RUST MIGRATION: solveLPViaDES is DES-visible solver orchestration and should be a PureTransform entry struct returning Result<DESSimplexSolution, Error>.", "RUST MIGRATION: Tableau mutation uses Vec<Vec<f64>> with explicit borrowing; preprocessing and pivot errors should be Result/status values."],
    &["DESSimplexOptions", "DESSimplexSolution", "DESSimplexTrace", "solveLPViaDES"],
);
