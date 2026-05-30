//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/stag-hunt.ts`
//! Rust target: `src/des/general/stag_hunt.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/stag-hunt.ts",
    "src/des/general/stag_hunt.rs",
    &["RUST MIGRATION: Target module `src/des/general/stag_hunt.rs`.", "RUST MIGRATION: Convert stag-hunt options/results to `serde` structs and model actions as an enum instead of raw numeric constants.", "RUST MIGRATION: Port `StagHuntEnv` and `StagHuntQLearner` as structs implementing joint-environment and RL-agent traits.", "RUST MIGRATION: Keep `runStagHunt` as a graph runner free function; graph-visible policy/update behavior belongs in trait impls.", "RUST MIGRATION: Inject RNG for independent exploration and return `Result` for invalid payoff, episode, alpha/gamma, or epsilon settings."],
    &["StagHuntOpts", "StagHuntResult", "runStagHunt"],
);
