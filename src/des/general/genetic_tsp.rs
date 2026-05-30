//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/genetic-tsp.ts`
//! Rust target: `src/des/general/genetic_tsp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/genetic-tsp.ts",
    "src/des/general/genetic_tsp.rs",
    &["RUST MIGRATION: target module src/des/general/genetic_tsp.rs.", "RUST MIGRATION: TSPInstance, GASolverOptions, GenerationInfo, GASolverResult, and GAPerformanceStats become serde structs; Tour becomes Vec<usize>.", "RUST MIGRATION: GeneticTSPOptimizer becomes a struct implementing PopulationOptimizer<Tour>; avoid subclassing by moving hooks into trait impl methods.", "RUST MIGRATION: Pure operators such as tourLength, crossover, mutation, twoOptImprove, heldKarpExact, and oneTreeLowerBound stay free functions.", "RUST MIGRATION: runGeneticTSP is a solver transform returning Result; all RNG-dependent builders/operators should take injected rand::Rng.", "RUST MIGRATION: Held-Karp dynamic programming maps naturally to HashMap bitmask keys or dense Vec tables; preserve precedence validation with Result."],
    &["GAPerformanceStats", "GASolverOptions", "GASolverResult", "GenerationInfo", "TSPInstance", "Tour", "buildPentagonTSP", "buildRandomTSP", "checkPrecedence", "heldKarpExact", "inversionMutate", "isPermutation", "oneTreeLowerBound", "orderCrossover", "repairPrecedence", "runGeneticTSP", "swapMutate", "tourLength", "tournamentSelect", "twoOptImprove"],
);
