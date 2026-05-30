//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/entity-processing/value-adder.ts`
//! Rust target: `src/des/entity_processing/value_adder.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/entity-processing/value-adder.ts",
    "src/des/entity_processing/value_adder.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/entity_processing/value_adder.rs",
        "- EntityNumericProcessor<S,T> becomes a small station struct with queue state",
        "- Basic numeric reduction (`k.value + p.value`) should move into a typed",
        "- Replace LinkedQueue, Symbol marker fields, util.inspect customization, and",
    ],
    &["EntityNumericProcessor", "GraphData", "isProcessor"],
);
