//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/control-systems/observability-controllability.ts`
//! Rust target: `src/des/general/control_systems/observability_controllability.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/control-systems/observability-controllability.ts",
    "src/des/general/control_systems/observability_controllability.rs",
    &["RUST MIGRATION: Target module `src/des/general/control_systems/observability_controllability.rs`.", "RUST MIGRATION: Convert state-space, MDP, POMDP, source/sink stations, and evaluator stations into structs plus model/controller/evaluator traits.", "RUST MIGRATION: Use `f64` matrices/vectors for dynamics, transitions, beliefs, and metrics; inject model config/RNG rather than reading ambient state.", "RUST MIGRATION: Graph-visible controllability/observability evaluators should become PureTransform-style structs with typed `transform` results."],
    &["ControllabilityEvaluatorStation", "EvaluationKind", "EvaluationSinkStation", "EvaluationToken", "MarkovDecisionProcess", "MdpControllabilityEvaluatorStation", "MdpSourceStation", "MdpSpec", "MdpToken", "ObsCtrlChannels", "ObservabilityEvaluatorStation", "PartiallyObservableProcess", "PomdpObservabilityEvaluatorStation", "PomdpSourceStation", "PomdpSpec", "PomdpToken", "StateSpaceModel", "StateSpaceSourceStation", "StateSpaceSpec", "StateSpaceToken"],
);
