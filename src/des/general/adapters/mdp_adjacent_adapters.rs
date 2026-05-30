//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/mdp-adjacent-adapters.ts`
//! Rust target: `src/des/general/adapters/mdp_adjacent_adapters.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/mdp-adjacent-adapters.ts",
    "src/des/general/adapters/mdp_adjacent_adapters.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/mdp_adjacent_adapters.rs`.", "RUST MIGRATION: Convert inventory, mountain-car, tiger, localization, actor-critic, blackjack, and LQR adapters into structs/functions around `DESModelSpec`.", "RUST MIGRATION: Encode MDP/POMDP params, policies, traces, and summaries as `serde` config/result structs; runtime/output paths become `PathBuf`.", "RUST MIGRATION: Use `Result<_, ValidationError>` for probability, grid, transition, and action-space validation."],
    &[],
);
