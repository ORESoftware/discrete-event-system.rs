//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/multistage-stochastic.ts`
//! Rust target: `src/des/general/multistage_stochastic.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/multistage-stochastic.ts",
    "src/des/general/multistage_stochastic.rs",
    &["RUST MIGRATION: target module src/des/general/multistage_stochastic.rs.", "RUST MIGRATION: DemandOutcome, MultiStageInventoryProblem, StageDecision, trace/result/options/tree structs become serde structs.", "RUST MIGRATION: SDDPStation becomes a struct implementing FixedPointIterationStation<SDDPState>; MODEL can be an associated const.", "RUST MIGRATION: solveStageDecision, expectedStageValue, solveExactScenarioTree, and evaluatePolicyExact are solver free functions; solveMultiStageSDDP/runMultiStageInventoryDemo can be PureTransform entry structs when graph-visible.", "RUST MIGRATION: Scenario tree and SDDP cut collections should use Vec-backed structs; demand sampling takes injected rand::Rng and validation returns Result."],
    &["DemandOutcome", "ExactTreeNodeResult", "MultiStageInventoryProblem", "MultiStageRunResult", "SDDPIterationTrace", "SDDPOptions", "SDDPResult", "SDDPStation", "StageDecision", "buildDefaultMultiStageInventoryProblem", "evaluatePolicyExact", "expectedStageValue", "runMultiStageInventoryDemo", "solveExactScenarioTree", "solveMultiStageSDDP", "solveStageDecision", "validateMultiStageProblem"],
);
