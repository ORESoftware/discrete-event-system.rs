//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/ga-des.ts`
//! Rust target: `src/des/general/ga_des.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/ga-des.ts",
    "src/des/general/ga_des.rs",
    &["RUST MIGRATION: target module src/des/general/ga_des.rs.", "RUST MIGRATION: TSPGAOptions and GADESResult become serde structs; Tour aliases should be Vec<usize>.", "RUST MIGRATION: TSPGAOptimizer becomes a struct implementing the PopulationOptimizer trait hooks instead of extending a base class.", "RUST MIGRATION: runTSPGADES is graph-visible optimization and should be a PureTransform entry struct; helper population/tour builders stay free functions.", "RUST MIGRATION: Keep RNG injected through rand::Rng and return Result for invalid population/tour inputs."],
    &["GADESResult", "TSPGAOptimizer", "TSPGAOptions", "runTSPGADES"],
);
