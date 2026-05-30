//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/statistical-optimization.ts`
//! Rust target: `src/des/general/statistical_optimization.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/statistical-optimization.ts",
    "src/des/general/statistical_optimization.rs",
    &["RUST MIGRATION: Target module `src/des/general/statistical_optimization.rs`.", "RUST MIGRATION: Convert distribution, demand, risk-capacity, SDDP, adaptive-simulation params/results/traces to `serde` structs; family/method unions become enums.", "RUST MIGRATION: Port iteration stations as structs implementing the fixed-point iteration trait, embedding shared station state instead of TS inheritance.", "RUST MIGRATION: Use `HashMap`/`HashSet` for scenario/cut/candidate indexes and typed structs for cuts and adaptive alternatives.", "RUST MIGRATION: Inject RNG for all sampling paths, keep pure fit/scenario/profit helpers as free functions, and return `Result` for invalid distributions, grids, and linear solves."],
    &["AdaptiveAlternative", "AdaptiveSimOptParams", "AdaptiveSimOptResult", "AdaptiveSimulationOptimizerStation", "AdaptiveTraceRow", "AlternativeStats", "CapacityExpansionSDDPStation", "DemandRange", "DemandScenario", "DemandSpec", "DistributionFamily", "DistributionFitParams", "DistributionFitResult", "DistributionFitStation", "EmpiricalPoint", "FitMethod", "FittedDistribution", "OptimizationLogger", "RiskCandidateResult", "RiskCapacityParams", "RiskCapacityResult", "RiskCapacityStation", "SDDPIteration", "SDDPParams", "SDDPResult", "buildDemandScenarios", "capacityProfit", "fitDistribution", "runAdaptiveSimOpt", "runCapacityExpansionSDDP", "runDistributionFit", "runRiskCapacity", "sampleDemandVector", "sampleFittedDistribution"],
);
