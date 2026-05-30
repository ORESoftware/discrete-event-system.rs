//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/monte-carlo-rl.ts`
//! Rust target: `src/des/general/des_base/monte_carlo_rl.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/monte-carlo-rl.ts",
    "src/des/general/des_base/monte_carlo_rl.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/monte_carlo_rl.rs",
        "- Keep file-for-file. MonteCarloOptions becomes a config struct and",
        "- Episode traces map to Vec records; Set usage for first-visit tracking maps",
        "- Action-value update helpers can stay private/associated methods; if a",
        "- Convert invalid options and impossible state/action errors to Result.",
        "- No bootstrapping: targets are full returns G_t, NOT r + γ V(s').",
        "- Updates land at end of episode, NOT online.",
        "- Unbiased but high-variance.",
        "- Very natural for episodic problems where the model is unknown",
    ],
    &["MonteCarloAgent", "MonteCarloOptions"],
);
