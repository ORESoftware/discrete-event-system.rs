//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/classical-optimization-models.ts`
//! Rust target: `src/des/general/classical_optimization_models.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/classical-optimization-models.ts",
    "src/des/general/classical_optimization_models.rs",
    &["RUST MIGRATION: target module src/des/general/classical_optimization_models.rs.", "RUST MIGRATION: QP/assignment/VRP/job-shop/flow-shop params and results become serde structs; rule string unions become enums.", "RUST MIGRATION: Token and DESStation subclasses become structs implementing Token and Station traits; channel constants can be &'static str associated consts.", "RUST MIGRATION: Pure numerical helpers such as qpObjective, solveAssignmentDP, dispatchSchedule, and NEH builders stay free functions.", "RUST MIGRATION: run* model entrypoints assemble DES-visible solver graphs, so expose each as a PureTransform-style struct returning Result for validation errors."],
    &["AssignmentParams", "AssignmentResult", "FlowShopJob", "FlowShopNEHParams", "FlowShopNEHResult", "JobOperation", "JobShopDispatchParams", "JobShopDispatchResult", "JobShopJob", "QPProjectedGradientParams", "QPProjectedGradientResult", "ScheduledOperation", "VRPCustomer", "VRPRoute", "VRPSavingsParams", "VRPSavingsResult", "runAuctionAssignment", "runFlowShopNEH", "runHungarianAssignment", "runJobShopDispatch", "runQPCoordinateDescent", "runQPProjectedGradient", "runVRPNearestNeighbor", "runVRPSavings"],
);
