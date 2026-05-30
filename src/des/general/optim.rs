//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/optim.ts`
//! Rust target: `src/des/general/optim.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/optim.ts",
    "src/des/general/optim.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/optim.rs",
        "- Keep this file as a pure numerical-optimization module. OptimOptions and",
        "- gradientDescent/newtonOptim/bfgs can remain public module functions in",
        "- Callback arguments map to generic `F: Fn(&[f64]) -> f64` / gradient traits.",
        "- Warnings and singular-matrix paths should become Result errors with a small",
    ],
    &[
        "OptimOptions",
        "OptimResult",
        "autoGradient",
        "bfgs",
        "gradientDescent",
        "newtonOptim",
    ],
);
