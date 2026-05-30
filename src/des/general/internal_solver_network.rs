//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/internal-solver-network.ts`
//! Rust target: `src/des/general/internal_solver_network.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/internal-solver-network.ts",
    "src/des/general/internal_solver_network.rs",
    &["RUST MIGRATION: target module src/des/general/internal_solver_network.rs.", "RUST MIGRATION: InternalSolverKind becomes an enum; progress/node/edge/result/params structs become serde structs.", "RUST MIGRATION: SolverSolutionToken, StopSignalToken, WallClockCheckerStation, SolutionSinkStation, and solver station classes become Token/Station trait impl structs.", "RUST MIGRATION: SnapshotProvider is behavior and should be a Rust trait implemented by each observable solver station.", "RUST MIGRATION: runInternalSolverNetwork is graph-visible orchestration and should be a PureTransform entry struct; buildSolverStation returns boxed trait objects or an enum of station variants.", "RUST MIGRATION: Graph/TSP/knapsack maps and tables should use HashMap/Vec as appropriate, with all validation and required() lookups returning Result."],
    &["InternalSolverKind", "InternalSolverRunParams", "InternalSolverRunResult", "KnapsackDPStation", "KnapsackParams", "KnapsackSAStation", "ObservableTSPGAOptimizer", "ObservableTSPSAOptimizer", "SOLUTION_CHANNEL", "STOP_CHANNEL", "ShortestPathSolverParams", "ShortestPathSolverStation", "SnapshotProvider", "SolutionSinkStation", "SolverNetworkDescription", "SolverNetworkEdge", "SolverNetworkNode", "SolverProgressPayload", "SolverSolutionToken", "StopSignalToken", "TSPHeldKarpStation", "TSPSolverParams", "WallClockCheckerStation", "runInternalSolverNetwork"],
);
