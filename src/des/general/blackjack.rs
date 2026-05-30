//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/blackjack.ts`
//! Rust target: `src/des/general/blackjack.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/blackjack.ts",
    "src/des/general/blackjack.rs",
    &["RUST MIGRATION: target module src/des/general/blackjack.rs.", "RUST MIGRATION: Blackjack becomes a struct implementing the Environment trait; model hands with Vec<u8> and explicit episode state instead of structural objects.", "RUST MIGRATION: BlackjackTrainOpts and BlackjackResult become serde structs; policy/value tables should use HashMap keys or compact indexed arrays.", "RUST MIGRATION: drawCard and Monte Carlo sampling must take injected rand::Rng; runBlackjackMC is a training transform and should return Result<BlackjackResult, Error>."],
    &["Blackjack", "BlackjackResult", "BlackjackTrainOpts", "runBlackjackMC"],
);
