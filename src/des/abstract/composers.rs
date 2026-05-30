//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/abstract/composers.ts`
//! Rust target: `src/des/abstract/composers.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/abstract/composers.ts",
    "src/des/abstract/composers.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/abstract/composers.rs",
        "- DoesFanOut<V> becomes a small struct holding a boxed/generic output-endpoint",
        "- Replace `HasManyOutputConnections<any, any>` and `AbstractMovingEntity<any>`",
    ],
    &["DoesFanOut"],
);
