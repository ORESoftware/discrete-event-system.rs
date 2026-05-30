//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/http-server/index.ts`
//! Rust target: `src/des/http_server/mod.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/http-server/index.ts",
    "src/des/http_server/mod.rs",
    &["RUST MIGRATION:", "- Target: src/des/http_server/mod.rs", "- Replace Node http singleton with an axum Router plus a tokio listener; handlers should return Result<impl IntoResponse, ServerError>.", "- Replace program:any with a typed ProgramState/ProgramHandle, likely Arc-owned if shared across async handlers.", "- Template loading should use include_str! for static HTML or PathBuf/std::fs at startup, not per-request loose fs reads.", "- JSON formation payload should become serde structs; send_raw/get_entities boundaries need explicit Rust traits or typed functions."],
    &["getHTTPServer"],
);

// BEGIN GENERATED MODULE DECLARATIONS

// END GENERATED MODULE DECLARATIONS
