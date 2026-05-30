//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/advanced-optimization.ts`
//! Rust target: `src/des/general/des_base/advanced_optimization.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/advanced-optimization.ts",
    "src/des/general/des_base/advanced_optimization.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/advanced_optimization.rs",
        "- Keep file-for-file. OptimizationCandidateToken, GraphWalkToken,",
        "- NumericSwarmOptimizerStation, PheromoneGraphSearchStation,",
        "- ParetoArchiveStation remains a concrete struct implementing DESStation;",
        "- Keep pure helpers such as dominates, normalize, vectorDot, and gram as module",
    ],
    &[
        "ConstraintAssignmentToken",
        "ConstraintSatisfactionSearchStation",
        "ConstraintSearchNode",
        "GraphWalkToken",
        "NumericSwarmOptimizerStation",
        "NumericSwarmOptions",
        "NumericSwarmParticle",
        "OptimizationCandidateToken",
        "OptimizationTraceRow",
        "ParetoArchiveRow",
        "ParetoArchiveStation",
        "ParetoCandidateToken",
        "PheromoneGraphOptions",
        "PheromoneGraphSearchStation",
        "SourceDrivenConstraintSatisfactionSearchStation",
        "UnitVectorRelaxationOptions",
        "UnitVectorRelaxationStation",
        "UnitVectorRelaxationTraceRow",
        "dominates",
        "gram",
        "normalize",
        "vectorDot",
    ],
);
