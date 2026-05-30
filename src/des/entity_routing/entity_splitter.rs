//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/entity-routing/entity-splitter.ts`
//! Rust target: `src/des/entity_routing/entity_splitter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/entity-routing/entity-splitter.ts",
    "src/des/entity_routing/entity_splitter.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/entity_routing/entity_splitter.rs",
        "- DecisionEntityGraph becomes a graph-data struct; EntitySplitter<S,T>",
        "- Broadcast fan-out is a PureTransform-style operation over one queued item",
        "- Replace LinkedQueue<AbstractMovingEntity<any>>, `any` targets, and",
    ],
    &["DecisionEntityGraph", "EntitySplitter"],
);
