//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/signals/adder.ts`
//! Rust target: `src/des/signals/adder.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/signals/adder.ts",
    "src/des/signals/adder.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/signals/adder.rs",
        "- IntegratorTimeStepOpts should be renamed or split from integral.rs if this",
        "- The accumulation loop is a PureTransform over queued signal values and",
        "- Replace `any` moving-entity accepts, LinkedQueue storage, Symbol marker, and",
    ],
    &["Adder", "IntegratorTimeStepOpts"],
);
