//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/actor-critic.ts`
//! Rust target: `src/des/general/des_base/actor_critic.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/actor-critic.ts",
    "src/des/general/des_base/actor_critic.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/actor_critic.rs",
        "- Keep file-for-file. ActorCriticOptions becomes a config struct and",
        "- Preserve the tabular value/logit matrices as Vec<Vec<f64>>; overrideable",
        "- Keep argmax as a private/module helper dependency; if this update is lifted",
        "- Convert constructor validation and impossible transition paths to Result.",
        "- Tabular CRITIC V(s) — closed-form ∇_w V(s) = e_s.",
        "- Tabular softmax ACTOR with parameters logits[s][a]:",
    ],
    &["ActorCriticOptions", "TabularActorCritic"],
);
