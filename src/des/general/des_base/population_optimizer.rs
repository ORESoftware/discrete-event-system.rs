//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/population-optimizer.ts`
//! Rust target: `src/des/general/des_base/population_optimizer.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/population-optimizer.ts",
    "src/des/general/des_base/population_optimizer.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/population_optimizer.rs",
        "- Keep file-for-file. Channel constants become pub consts; initial/result",
        "- PopulationSourceStation and PopulationSinkStation become concrete DESStation",
        "- Pure proposal/fitness helpers can stay trait methods; if exposed as graph",
        "- Convert duplicate seed, uninitialized optimizer, and invalid fitness throws",
    ],
    &[
        "POPULATION_INITIAL_CHANNEL",
        "POPULATION_RESULT_CHANNEL",
        "PopulationInitialToken",
        "PopulationOptimizer",
        "PopulationResultSnapshot",
        "PopulationResultToken",
        "PopulationSinkStation",
        "PopulationSourceStation",
    ],
);
