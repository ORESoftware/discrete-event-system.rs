//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/signals/integral.ts`
//! Rust target: `src/des/signals/integral.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/signals/integral.ts",
    "src/des/signals/integral.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/signals/integral.rs",
        "- IntegratorTimeStepOpts becomes a struct; Integrator<E,V> becomes a signal",
        "- The integration step is a PureTransform-style accumulation from queued",
        "- Replace LinkedQueue, Symbol marker, broad AbstractMovingEntity<any> inputs,",
    ],
    &["Integrator", "IntegratorTimeStepOpts"],
);
