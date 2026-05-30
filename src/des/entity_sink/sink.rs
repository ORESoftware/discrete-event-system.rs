//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/entity-sink/sink.ts`
//! Rust target: `src/des/entity_sink/sink.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/entity-sink/sink.ts",
    "src/des/entity_sink/sink.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/entity_sink/sink.rs",
        "- AbstractSinkEntity<S,T> becomes a SinkLike trait plus shared SinkState",
        "- Convert abstract class inheritance and structural HasManyInputConnections",
        "- EntitySinkGraphData and audit/serializable data should be serde structs;",
    ],
    &["AbstractSinkEntity", "EntitySink"],
);
