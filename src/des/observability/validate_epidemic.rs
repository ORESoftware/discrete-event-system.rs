//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/observability/validate-epidemic.ts`
//! Rust target: `src/des/observability/validate_epidemic.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/observability/validate-epidemic.ts",
    "src/des/observability/validate_epidemic.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/observability/validate_epidemic.rs",
        "- This is a CLI validator; keep the module file-for-file, but expose a",
        "- Failure becomes a struct; event records should be serde_json::Value or typed",
        "- Replace process.argv/process.exit, regex helpers, and thrown missing-event",
    ],
    &[],
);
