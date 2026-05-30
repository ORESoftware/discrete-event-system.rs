//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/random-variables/generate.ts`
//! Rust target: `src/des/random_variables/generate.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/random-variables/generate.ts",
    "src/des/random_variables/generate.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/random_variables/generate.rs",
        "- This is a sampling CLI/dev helper. Keep pure sampling in structs such as",
        "- Replace CommonJS require, math.random, dynamic arrays, and console printing",
        "- The inline map/reduce helpers should become named transforms or iterator",
    ],
    &["runExponential", "runUniform"],
);
