//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/rl-environments.ts`
//! Rust target: `src/des/general/rl_environments.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/rl-environments.ts",
    "src/des/general/rl_environments.rs",
    &["RUST MIGRATION: Target module `src/des/general/rl_environments.rs`.", "RUST MIGRATION: Convert the `Environment` interface into a Rust trait with associated `State`/`Action` types or explicit numeric aliases.", "RUST MIGRATION: Port `GridWorld` and `Corridor` as structs implementing that trait; constructor option bags should become `serde` config structs.", "RUST MIGRATION: Keep `evalPolicy` as a free function generic over the environment trait, with injected RNG if policy evaluation becomes stochastic.", "RUST MIGRATION: Represent grid/corridor boundary errors as `Result`, and avoid TS-style structural return objects by naming step-result structs."],
    &["Corridor", "Environment", "GridWorld", "evalPolicy"],
);
