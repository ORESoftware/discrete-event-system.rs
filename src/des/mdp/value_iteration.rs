//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/mdp/value-iteration.ts`
//! Rust target: `src/des/mdp/value_iteration.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/mdp/value-iteration.ts",
    "src/des/mdp/value_iteration.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/mdp/value_iteration.rs",
        "- VIOptions and VIResult become structs; Float64Array/Int32Array map to",
        "- buildTransitionTable and valueIteration are pure algorithm functions; they",
        "- Terminal sentinels (-1 policy) should become Option<Action>; warnings on",
    ],
    &[
        "VIOptions",
        "VIResult",
        "buildTransitionTable",
        "valueIteration",
    ],
);
