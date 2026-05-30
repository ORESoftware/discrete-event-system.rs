//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/milp-bnb.ts`
//! Rust target: `src/des/general/milp_bnb.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/milp-bnb.ts",
    "src/des/general/milp_bnb.rs",
    &["RUST MIGRATION: target module src/des/general/milp_bnb.rs.", "RUST MIGRATION: MILPProblem, options, solutions, node events, branches/nodes, and facility-location inputs become serde structs.", "RUST MIGRATION: MILPBnBStation becomes a struct implementing TreeSearchStation<MILPNode>; branch-and-bound status should be explicit enums.", "RUST MIGRATION: solveMILP is graph-visible solver orchestration and should be a PureTransform entry struct returning Result<MILPSolution, Error>.", "RUST MIGRATION: buildKnapsackMILP/buildFacilityLocationMILP are pure builders; validation and LP relaxation failures return Result.", "RUST MIGRATION: Matrix/vector storage maps to Vec<Vec<f64>>/Vec<f64>, integer masks to Vec<bool>, and branch node queues to VecDeque/BinaryHeap as needed."],
    &["FacilityLocationProblem", "MILPProblem", "MILPSolution", "MILPSolveOptions", "NodeEvent", "buildFacilityLocationMILP", "buildKnapsackMILP", "solveMILP"],
);
