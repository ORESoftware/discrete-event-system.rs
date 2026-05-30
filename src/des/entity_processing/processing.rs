//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/entity-processing/processing.ts`
//! Rust target: `src/des/entity_processing/processing.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/entity-processing/processing.ts",
    "src/des/entity_processing/processing.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/entity_processing/processing.rs",
        "- EntityProcessor<S,T> becomes a processor station struct that composes the",
        "- LinkedQueue/IterableInt/DESMap should become VecDeque/ranges/HashMap or",
        "- The service-completion step is a PureTransform boundary over",
        "- Convert thrown strings/errors, util.inspect hooks, Symbol markers, and",
    ],
    &["EntityProcessor", "ProcessorEntityGraphData", "isProcessor"],
);
