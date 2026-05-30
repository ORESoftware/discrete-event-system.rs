//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/rl-learning-models.ts`
//! Rust target: `src/des/general/rl_learning_models.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/rl-learning-models.ts",
    "src/des/general/rl_learning_models.rs",
    &["RUST MIGRATION: Target module `src/des/general/rl_learning_models.rs`.", "RUST MIGRATION: Convert policy-gradient and Expected-SARSA params/results to `serde` structs; keep `RLTopology` as the Rust graph-summary alias.", "RUST MIGRATION: Port agent/update classes as structs implementing `PolicyGradientAgent`, `PolicyUpdateStation`, and `RLAgentStation` traits.", "RUST MIGRATION: Replace inheritance overrides with trait impls and embed shared base state explicitly.", "RUST MIGRATION: Inject RNG for softmax/action choice and return `Result` for invalid alpha/gamma/episode options or malformed environment dimensions."],
    &["ExpectedSarsaGridParams", "ExpectedSarsaGridResult", "PolicyGradientCorridorParams", "PolicyGradientCorridorResult", "RLTopology", "runExpectedSarsaGridworld", "runPolicyGradientCorridor"],
);
