//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/signals/incrementer.ts`
//! Rust target: `src/des/signals/incrementer.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/signals/incrementer.ts",
    "src/des/signals/incrementer.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/signals/incrementer.rs",
        "- IncrementorTimeStepOpts becomes a struct; SignalIncrementor<E,V> becomes a",
        "- Constructor currently passes `null as any` to the parent id; Rust should use",
        "- Current runTimeStep is intentionally inert until increment semantics are",
        "- Queue intake and runningTotal mirror the other signal transforms; in Rust",
    ],
    &["IncrementorTimeStepOpts", "SignalIncrementor"],
);
