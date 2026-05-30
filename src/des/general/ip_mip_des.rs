//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/ip-mip-des.ts`
//! Rust target: `src/des/general/ip_mip_des.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/ip-mip-des.ts",
    "src/des/general/ip_mip_des.rs",
    &["RUST MIGRATION: target module src/des/general/ip_mip_des.rs.", "RUST MIGRATION: Relaxation algorithm unions, token state union, trace events, and branch/cut payloads become enums with serde tags.", "RUST MIGRATION: IPMIPProblem, options, solutions, performance stats, topology nodes, and constraints become serde structs with Vec<f64>/Vec<Vec<f64>> matrices.", "RUST MIGRATION: PayloadStatefulToken subclasses and DESStation/CompositeDESStation subclasses become structs implementing Token, StatefulToken, Station, and CompositeStation traits.", "RUST MIGRATION: solveIPMIPWithDES and buildIPMIPSolverTechniquePlan are graph-visible solver transforms; expose them as PureTransform entry structs returning Result.", "RUST MIGRATION: Partial<Record<...>> usage maps to HashMap<ConcreteLPRelaxationAlgorithm, usize>, and all validation/LP relaxation failures should flow through Result/status enums."],
    &["BranchAndCutSolverStation", "BranchOrCutConstraint", "ConcreteLPRelaxationAlgorithm", "IPMIPPerformanceStats", "IPMIPProblem", "IPMIPProblemFeatures", "IPMIPSolution", "IPMIPSolveOptions", "IPMIPSolverTechniquePlan", "IPMIPTraceEvent", "LPRelaxationAlgorithm", "LPRelaxationStation", "SolverTokenStats", "SolverTopologyNode", "analyzeIPMIPProblem", "buildBinaryKnapsackIP", "buildIPMIPSolverTechniquePlan", "buildSmallMixedIP", "solveIPMIPWithDES", "validateIPMIPProblem"],
);
