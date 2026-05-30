//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/lp.ts`
//! Rust target: `src/des/general/lp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/lp.ts",
    "src/des/general/lp.rs",
    &["RUST MIGRATION: target module src/des/general/lp.rs.", "RUST MIGRATION: LPProblem, LPSolution, InternalSimplexOptions, ExternalSolverOptions, and SimplexResult become serde structs where public; LPStatus becomes an enum.", "RUST MIGRATION: solveLPInternal, solveLPExternal, solveLP, and lpToString are solver/adapter free functions unless explicitly registered as PureTransform entrypoints.", "RUST MIGRATION: Replace JS object option merging with explicit defaults/builders, matrix arrays with Vec<Vec<f64>>, and external solver failures with Result.", "RUST MIGRATION: pivot/simplexCore should use mutable slices and clear status enums to avoid exceptions and aliasing surprises."],
    &["ExternalSolverOptions", "InternalSimplexOptions", "LPProblem", "LPSolution", "LPStatus", "lpToString", "solveLP", "solveLPExternal", "solveLPInternal"],
);
