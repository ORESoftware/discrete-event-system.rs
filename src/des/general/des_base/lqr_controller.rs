//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/lqr-controller.ts`
//! Rust target: `src/des/general/des_base/lqr_controller.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/lqr-controller.ts",
    "src/des/general/des_base/lqr_controller.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/lqr_controller.rs",
        "- Keep file-for-file. Vec/Mat aliases become Vec<f64>/Vec<Vec<f64>> or a",
        "- LQRController becomes a struct implementing the ControllerStation trait with",
        "- Matrix helpers can stay private module functions or become associated",
        "- Convert shape mismatches and singular-matrix throws to Result.",
    ],
    &["LQRController", "LQRSpec", "Mat", "Vec"],
);
