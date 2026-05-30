//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/max-flow.ts`
//! Rust target: `src/des/general/max_flow.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/max-flow.ts",
    "src/des/general/max_flow.rs",
    &["RUST MIGRATION: target module src/des/general/max_flow.rs.", "RUST MIGRATION: MaxFlowEdge, MaxFlowProblem, MaxFlowTraceEntry, MaxFlowResult, residual state structs become serde structs where public/private as appropriate.", "RUST MIGRATION: MaxFlowStation becomes a struct implementing FixedPointIterationStation; keep MODEL as an associated const.", "RUST MIGRATION: solveMaxFlow is DES-visible solver orchestration and should be a PureTransform entry struct if registered; buildTextbookMaxFlowProblem remains a free builder.", "RUST MIGRATION: Residual graphs should use Vec<Vec<ResidualEdge>> or HashMap adjacency, and validation/errors should return Result."],
    &["MaxFlowEdge", "MaxFlowProblem", "MaxFlowResult", "MaxFlowStation", "MaxFlowTraceEntry", "buildTextbookMaxFlowProblem", "solveMaxFlow", "validateMaxFlowProblem"],
);
