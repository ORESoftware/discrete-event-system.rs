//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/actor-critic-gridworld.ts`
//! Rust target: `src/des/general/actor_critic_gridworld.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/actor-critic-gridworld.ts",
    "src/des/general/actor_critic_gridworld.rs",
    &["RUST MIGRATION: target module src/des/general/actor_critic_gridworld.rs.", "RUST MIGRATION: ActorCriticTrainOpts and ActorCriticResult become serde structs with Vec<f64>/Vec<usize> fields.", "RUST MIGRATION: runActorCriticGridworld is a graph-visible training entrypoint; port it as a PureTransform-style struct with transform(opts) -> Result<ActorCriticResult, Error>.", "RUST MIGRATION: Keep all RNG behind an injected rand::Rng/seeded adapter, and model table updates as owned Vec<Vec<f64>> state."],
    &["ActorCriticResult", "ActorCriticTrainOpts", "runActorCriticGridworld"],
);
