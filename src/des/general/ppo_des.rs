//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/ppo-des.ts`
//! Rust target: `src/des/general/ppo_des.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/ppo-des.ts",
    "src/des/general/ppo_des.rs",
    &["RUST MIGRATION: Target module `src/des/general/ppo_des.rs`.", "RUST MIGRATION: Convert PPO options and DES result interfaces to `serde` structs; keep numeric state/action specialization explicit (`usize` or newtypes) instead of `number`.", "RUST MIGRATION: Port `TabularPPOAgent` and `PPOClipUpdateStation` as structs implementing policy-gradient and policy-update traits.", "RUST MIGRATION: Replace inherited override methods with trait impls; shared base-class state should be embedded/composed in the Rust structs.", "RUST MIGRATION: Inject RNG through the environment/agent ports and return `Result` for bad batch sizes, horizons, or non-finite advantages."],
    &["PPOClipUpdateStation", "PPODESResult", "PPOUpdateOptions", "TabularPPOAgent", "runPPODES"],
);
