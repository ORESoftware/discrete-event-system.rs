//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/reference/compare-epidemic.ts`
//! Rust target: `src/des/reference/compare_epidemic.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/reference/compare-epidemic.ts",
    "src/des/reference/compare_epidemic.rs",
    &["RUST MIGRATION:", "- Target: src/des/reference/compare_epidemic.rs.", "- Keep file-for-file as the reference comparison module; RunSummary and row output become golden comparison structs.", "- Map argv path inputs to clap/std::env, read JSONL through std::fs/std::io, and deserialize with serde/serde_json.", "- Treat framework/FEL event logs as external-adapter fixtures so future Rust and non-Rust references compare through the same schema."],
    &[],
);
