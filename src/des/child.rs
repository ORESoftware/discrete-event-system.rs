//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/child.ts`
//! Rust target: `src/des/child.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/child.ts",
    "src/des/child.rs",
    &["RUST MIGRATION: target src/des/child.rs.", "RUST MIGRATION: Keep this as a library module; map exported classes/interfaces to structs/traits and keep websocket/process effects behind explicit ports.", "RUST MIGRATION: Lift reusable orchestration into shared traits/impls so any Rust binary can call this module without duplicating setup.", "RUST MIGRATION: If this boundary later reads args/env/files, prefer clap/std::env/PathBuf and serde-deserialized config structs."],
    &[],
);
