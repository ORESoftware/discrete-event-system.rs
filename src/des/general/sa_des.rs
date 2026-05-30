//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/sa-des.ts`
//! Rust target: `src/des/general/sa_des.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/sa-des.ts",
    "src/des/general/sa_des.rs",
    &["RUST MIGRATION: Target module `src/des/general/sa_des.rs`.", "RUST MIGRATION: Port simulated-annealing DES leaf classes as structs implementing the `SingleStateOptimizer` trait/template hooks.", "RUST MIGRATION: Convert optimizer options, tick events, and run results to `serde` structs; discriminated problem-specific moves should become enums.", "RUST MIGRATION: Keep DES runner/builders as free functions, but expose graph-visible optimizer steps through trait impls rather than closure-heavy helpers.", "RUST MIGRATION: Inject RNG for candidate generation/acceptance and return `Result` for invalid temperature schedules, dimensions, or TSP/knapsack inputs."],
    &["CoolingSchedule", "SADESResult", "TSPHillClimber", "TSPSAOptimizer", "TSPSAOptions", "runTSPHillClimberDES", "runTSPSADES", "temperatureAt"],
);
