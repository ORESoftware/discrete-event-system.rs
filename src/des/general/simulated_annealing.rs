//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/simulated-annealing.ts`
//! Rust target: `src/des/general/simulated_annealing.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/simulated-annealing.ts",
    "src/des/general/simulated_annealing.rs",
    &["RUST MIGRATION: Target module `src/des/general/simulated_annealing.rs`.", "RUST MIGRATION: Convert `SAProblem`, schedules, options, tick events, results, and knapsack/TSP adapters to `serde` structs where closures are replaced by traits.", "RUST MIGRATION: Represent `CoolingSchedule` as an enum; problem behavior (`energy`, `neighbor`, `clone`) should be a trait implemented by concrete problem structs.", "RUST MIGRATION: Port `SAOptimizer` as a struct implementing the optimizer trait, with candidate acceptance using injected RNG.", "RUST MIGRATION: Keep builder/runner functions free; return `Result` for invalid schedules, empty states, dimension mismatches, and infeasible adapter inputs."],
    &["CoolingSchedule", "KnapsackInstance", "SAProblem", "SAResult", "SASolverOptions", "SATickEvent", "buildKnapsackSAProblem", "buildTSPSAProblem", "runSimulatedAnnealing", "temperatureAt"],
);
