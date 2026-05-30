//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/reference/main-epidemic-fel.ts`
//! Rust target: `src/des/reference/main_epidemic_fel.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/reference/main-epidemic-fel.ts",
    "src/des/reference/main_epidemic_fel.rs",
    &["RUST MIGRATION:", "- Target: src/des/reference/main_epidemic_fel.rs.", "- Keep file-for-file as the golden FEL reference; FelEvent and emitted summaries become serde comparison structs.", "- Replace Math.random with an injected RNG trait and keep event-log/CSV/matrix output behind std::fs plus serde_json/csv writers.", "- Treat this module as an external adapter-compatible reference implementation, with Result-returning run/main boundaries."],
    &[],
);
