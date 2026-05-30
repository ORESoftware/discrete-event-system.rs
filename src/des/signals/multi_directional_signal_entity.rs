//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/signals/multi-directional-signal-entity.ts`
//! Rust target: `src/des/signals/multi_directional_signal_entity.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/signals/multi-directional-signal-entity.ts",
    "src/des/signals/multi_directional_signal_entity.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/signals/multi_directional_signal_entity.rs",
        "- MultiDirectionalSignalEntity<E,V> becomes reusable signal connection state",
        "- `maxQueueSize = null as number` should become Option<usize> or a bounded",
        "- Abstract accept/take hooks should return Result or explicit acceptance",
    ],
    &["MultiDirectionalSignalEntity"],
);
