//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/advanced-optimization-models.ts`
//! Rust target: `src/des/general/advanced_optimization_models.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/advanced-optimization-models.ts",
    "src/des/general/advanced_optimization_models.rs",
    &["RUST MIGRATION: target module src/des/general/advanced_optimization_models.rs.", "RUST MIGRATION: Param/result interfaces and Point2/WeightedEdge/PortfolioAsset become serde structs; ContinuousObjectiveName becomes an enum.", "RUST MIGRATION: Token and optimizer station classes become structs implementing Token/Station/Optimizer traits; generic OptimizationStartToken<P> maps to a generic struct.", "RUST MIGRATION: Solver helpers that are not graph-visible stay free functions, while runParticleSwarm/runAntColonyTSP/runMapColoringCSP/runMaxSATLocalSearch/runSDPMaxCutRelaxation/runParetoPortfolio become PureTransform entry structs.", "RUST MIGRATION: Replace object/array maps with HashMap/HashSet where needed, keep random search behind injected rand::Rng, and convert thrown validation errors to Result."],
    &["AntColonyTSPParams", "AntColonyTSPResult", "ContinuousObjectiveName", "MapColoringCSPParams", "MapColoringCSPResult", "MaxSATParams", "MaxSATResult", "ParetoPortfolioParams", "ParetoPortfolioPoint", "ParetoPortfolioResult", "ParticleSwarmParams", "ParticleSwarmResult", "Point2", "PortfolioAsset", "SDPMaxCutParams", "SDPMaxCutResult", "WeightedEdge", "paretoFrontIsNondominated", "runAntColonyTSP", "runMapColoringCSP", "runMaxSATLocalSearch", "runParetoPortfolio", "runParticleSwarm", "runSDPMaxCutRelaxation"],
);
