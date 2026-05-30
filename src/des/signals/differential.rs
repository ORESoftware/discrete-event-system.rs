//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/signals/differential.ts`
//! Rust target: `src/des/signals/differential.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/signals/differential.ts",
    "src/des/signals/differential.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/signals/differential.rs",
        "- DifferentialTimeStepOpts becomes a struct; Differentiator<E,V> becomes a",
        "- The difference calculation is a PureTransform candidate:",
        "- Replace Symbol marker state, LinkedQueue, `any` casts, and console-error",
    ],
    &["DifferentialTimeStepOpts", "Differentiator"],
);
