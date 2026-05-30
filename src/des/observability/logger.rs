//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/observability/logger.ts`
//! Rust target: `src/des/observability/logger.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/observability/logger.ts",
    "src/des/observability/logger.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/observability/logger.rs",
        "- LogLevel becomes an enum with Ord; BaseEvent becomes a trait or enum-backed",
        "- JsonlLogger maps to a struct owning a BufWriter<File>, min-level, counters,",
        "- readEvents is a pure IO transform from path -> Vec<Event>; replace",
        "- Append-only line-delimited JSON so the file can be tail'd, jq'd, or",
        "- Cheap when filtered out (level check before stringify).",
        "- Synchronous .write so we don't have to await every event.",
        "- One-shot file path; run-to-run files don't get appended on top of",
    ],
    &["BaseEvent", "JsonlLogger", "LogLevel", "readEvents"],
);
