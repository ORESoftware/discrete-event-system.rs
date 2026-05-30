//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/framework-runner.ts`
//! Rust target: `src/des/runners/framework_runner.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/framework-runner.ts",
    "src/des/runners/framework_runner.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/framework_runner.rs.", "- Keep file-for-file as a library runner exposing run_framework_once over migrated DES entity structs/traits.", "- Replace framework construction helpers with explicit builder structs; DES graph callbacks should become PureTransform-style trait impls.", "- Isolate mathjs/logging/seed behavior behind numeric, JsonlLogger, and RNG traits so the runner ports cleanly."],
    &["runFrameworkOnce"],
);
