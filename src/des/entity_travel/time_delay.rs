//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/entity-travel/time-delay.ts`
//! Rust target: `src/des/entity_travel/time_delay.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/entity-travel/time-delay.ts",
    "src/des/entity_travel/time_delay.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/entity_travel/time_delay.rs",
        "- TimeDelayEntityGraphData and DelayTimeStepOpts become structs;",
        "- Queue storage should be VecDeque with an associated MovingEntity item type;",
        "- The current unimplemented validation/takeItem throws should become",
    ],
    &[
        "DelayTimeStepOpts",
        "TimeDelayEntityGraphData",
        "TimeDelayOrTravelEntity",
    ],
);
