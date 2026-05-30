//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/incremental-lp.ts`
//! Rust target: `src/des/general/incremental_lp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/incremental-lp.ts",
    "src/des/general/incremental_lp.rs",
    &["RUST MIGRATION: target module src/des/general/incremental_lp.rs.", "RUST MIGRATION: LPEvent is a discriminated union and should become an enum with variants; PivotEvent, IncrementalLPInit, and LPSnapshot become serde structs.", "RUST MIGRATION: IncrementalLP becomes a stateful struct with impl methods for stepping/pivoting; expose graph-visible use as a Station/PureTransform wrapper.", "RUST MIGRATION: Tableau arrays map to Vec<Vec<f64>>, and invalid pivots/unbounded states should be Result or enum status values instead of throws."],
    &["IncrementalLP", "IncrementalLPInit", "LPEvent", "LPSnapshot", "PivotEvent"],
);
