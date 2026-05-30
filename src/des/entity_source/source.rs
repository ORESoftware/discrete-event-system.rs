//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/entity-source/source.ts`
//! Rust target: `src/des/entity_source/source.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/entity-source/source.ts",
    "src/des/entity_source/source.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/entity_source/source.rs",
        "- AbstractSourceEntity<S,T> becomes a SourceLike trait plus SourceState for",
        "- Source emission is a PureTransform boundary: (source state, step size,",
        "- Replace `(global as any).turnOffSources`, uuid, LinkedQueue/Array queues,",
    ],
    &[
        "AbstractSourceEntity",
        "DefiniteFiniteSource",
        "EntitySource",
    ],
);
