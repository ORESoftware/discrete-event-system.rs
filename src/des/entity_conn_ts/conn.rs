//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/entity-conn.ts/conn.ts`
//! Rust target: `src/des/entity_conn_ts/conn.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/entity-conn.ts/conn.ts",
    "src/des/entity_conn_ts/conn.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/entity_conn_ts/conn.rs",
        "- ConnectionOpts becomes a plain struct with typed fields. The odd",
        "- Imported Entity/graph endpoint types are unused here today; decide whether",
    ],
    &["ConnectionOpts"],
);
