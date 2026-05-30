//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/episode-accounting.ts`
//! Rust target: `src/des/general/des_base/episode_accounting.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/episode-accounting.ts",
    "src/des/general/des_base/episode_accounting.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/episode_accounting.rs",
        "- Keep file-for-file. EpisodeSummary and VectorEpisodeSummary become data",
        "- Reward histories should use Vec<f64> and vector rewards Vec<Vec<f64>>;",
        "- These are not DES graph nodes today; if reward aggregation becomes a graph",
        "- Convert reward-dimension mismatches from thrown errors to Result.",
    ],
    &[
        "EpisodeAccounting",
        "EpisodeSummary",
        "VectorEpisodeAccounting",
        "VectorEpisodeSummary",
    ],
);
