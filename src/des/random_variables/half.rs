//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/random-variables/half.ts`
//! Rust target: `src/des/random_variables/half.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/random-variables/half.ts",
    "src/des/random_variables/half.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/random_variables/half.rs",
        "- This helper should become a named PureTransform struct, e.g.",
        "- Fix the implicit JS sort before porting: Rust needs an explicit numeric",
        "- Replace `any[]`, tuple accumulator tricks, and console-driven execution",
    ],
    &[],
);
