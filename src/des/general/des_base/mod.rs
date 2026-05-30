//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/index.ts`
//! Rust target: `src/des/general/des_base/mod.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/index.ts",
    "src/des/general/des_base/mod.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/mod.rs",
        "- This is the Rust `mod.rs` boundary for all des_base modules. Each export",
        "- Preserve the class-family hierarchy as trait families: DESStation as the",
        "- Keep the module list one-to-one with the TypeScript files during the first",
    ],
    &[],
);

// BEGIN GENERATED MODULE DECLARATIONS
pub mod actor_critic;
pub mod advanced_optimization;
pub mod adversarial_control;
pub mod argmax;
pub mod belief_state;
pub mod composite_station;
pub mod control_blocks;
pub mod controller;
pub mod cut_pool;
pub mod environment;
pub mod episode_accounting;
pub mod finite_horizon_dp;
pub mod fixed_point;
pub mod learning_optimization;
pub mod linear_vfa;
pub mod lqr_controller;
pub mod model_topology;
pub mod monte_carlo_rl;
pub mod multi_agent;
pub mod neural_network;
pub mod policy_gradient_agent;
pub mod population_optimizer;
pub mod preconditions;
pub mod rl_agent;
pub mod rl_tokens;
pub mod runner;
pub mod semi_mdp;
pub mod single_state_optimizer;
pub mod smart_movable;
pub mod stateful_token;
pub mod station;
pub mod transform_entity;
pub mod tree_search;
pub mod validation;
pub mod visual_block;
// END GENERATED MODULE DECLARATIONS
