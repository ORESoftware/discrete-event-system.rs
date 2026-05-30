//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/signals/abstract.ts`
//! Rust target: `src/des/signals/abstract.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/signals/abstract.ts",
    "src/des/signals/abstract.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/signals/abstract.rs",
        "- SignalMarker becomes a zero-sized marker type or enum variant;",
        "- SignalEntity<E,V> should be a SignalEntity trait layered on MovingEntity,",
        "- Null/undefined placeholders (`return null as any`) need typed Option/Result",
    ],
    &[
        "SignalEntity",
        "SignalEntityGraphData",
        "SignalMarker",
        "SignalTimeStepOpts",
    ],
);
