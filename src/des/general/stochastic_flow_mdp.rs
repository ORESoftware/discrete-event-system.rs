//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/stochastic-flow-mdp.ts`
//! Rust target: `src/des/general/stochastic_flow_mdp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/stochastic-flow-mdp.ts",
    "src/des/general/stochastic_flow_mdp.rs",
    &["RUST MIGRATION: Target module `src/des/general/stochastic_flow_mdp.rs`.", "RUST MIGRATION: Convert edge/problem/state/action/decision/sim-step/result interfaces to `serde` structs with typed node/edge IDs.", "RUST MIGRATION: Port `StochasticFlowMDPStation` as a struct implementing the finite-horizon DP station trait; indexed actions should be nominal structs.", "RUST MIGRATION: Use `HashMap`/`HashSet` for state/action indexes, capacity maps, and policy lookups; preserve stable ordering for trace snapshots.", "RUST MIGRATION: Inject RNG for policy simulation and return `Result` from problem validation, solving, and simulation when capacities or transitions are invalid."],
    &["FlowMDPAction", "FlowMDPDecision", "FlowMDPSimStep", "FlowMDPState", "StochasticFlowEdge", "StochasticFlowMDPProblem", "StochasticFlowMDPResult", "StochasticFlowMDPStation", "buildDefaultStochasticFlowMDPProblem", "simulateStochasticFlowPolicy", "solveStochasticFlowMDP", "validateStochasticFlowMDPProblem"],
);
