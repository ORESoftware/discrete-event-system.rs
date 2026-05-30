//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/dispatch.ts`
//! Rust target: `src/des/general/dispatch.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/dispatch.ts",
    "src/des/general/dispatch.rs",
    &["RUST MIGRATION: target module src/des/general/dispatch.rs.", "RUST MIGRATION: DispatchProblem/State/Result and policy option/result interfaces become serde structs; PendingJob and MCTSDispatchState are private structs.", "RUST MIGRATION: DispatchPolicy is behavior and should become a trait or enum of policy implementations rather than a structural interface.", "RUST MIGRATION: simulateDispatch, policy builders, LP/MDP/MCTS builders, evaluatePolicy, and welchT can remain free functions unless registered, then wrap in PureTransform.", "RUST MIGRATION: RNG is already passed as seed/functions; port to injected rand::Rng and use VecDeque/HashMap for queues, tables, and memoized MDP state."],
    &["DispatchPolicy", "DispatchProblem", "DispatchResult", "DispatchState", "EvaluationResult", "FluidLPPolicyResult", "MCTSPolicyOptions", "MDPVIPolicyOptions", "MDPVIPolicyResult", "buildDispatchFluidLP", "evaluatePolicy", "policyFluidLP", "policyMCTS", "policyMDPVI", "policyRandom", "policyRoundRobin", "policySECT", "policyShortestQueue", "simulateDispatch", "welchT"],
);
