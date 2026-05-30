//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/domain-application-adapter.ts`
//! Rust target: `src/des/general/adapters/domain_application_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/domain-application-adapter.ts",
    "src/des/general/adapters/domain_application_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/domain_application_adapter.rs`.", "RUST MIGRATION: Convert domain application adapter registrations, CSV writers, and animation builders into adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Use `serde` config/result structs for domain metrics, candidates, traces, and chart payloads; model output locations as `PathBuf`.", "RUST MIGRATION: Return `Result<_, ValidationError>` for bad domain params and `Result` for filesystem/rendering failures."],
    &[],
);
