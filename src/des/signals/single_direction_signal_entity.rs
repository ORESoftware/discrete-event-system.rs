//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/signals/single-direction-signal-entity.ts`
//! Rust target: `src/des/signals/single_direction_signal_entity.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/signals/single-direction-signal-entity.ts",
    "src/des/signals/single_direction_signal_entity.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/signals/single_direction_signal_entity.rs",
        "- SingleInManyOutSignalEntity<E,V> becomes a shared signal station state",
        "- `connectionIn = null as EntityConnection` should become Option<Connection>",
        "- Replace `any` endpoint/moving-entity contracts with associated Item/Input/",
    ],
    &["SingleInManyOutSignalEntity"],
);
