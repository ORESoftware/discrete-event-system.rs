//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/entity-sink/generic-sink.ts`
//! Rust target: `src/des/entity_sink/generic_sink.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/entity-sink/generic-sink.ts",
    "src/des/entity_sink/generic_sink.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/entity_sink/generic_sink.rs",
        "- EntitySinkGraphData becomes a struct; GenericEntitySink<S,T> becomes a sink",
        "- `StationaryEntity<GenericEntitySink<...>>` is currently used as a structural",
        "- Replace Symbol marker fields, BasicMovingEntity-only takeItem typing,",
    ],
    &["GenericEntitySink"],
);
