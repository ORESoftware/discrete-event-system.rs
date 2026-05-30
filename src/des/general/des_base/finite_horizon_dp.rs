//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/finite-horizon-dp.ts`
//! Rust target: `src/des/general/des_base/finite_horizon_dp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/finite-horizon-dp.ts",
    "src/des/general/des_base/finite_horizon_dp.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/finite_horizon_dp.rs",
        "- Keep file-for-file. DPOutcome and DPOptions become data/config structs and",
        "- Preserve horizon tables as Vec<Vec<f64>>/Vec<usize>; model transition and",
        "- Math.random becomes an injected RNG trait/generic. maxArr/minArr stay",
        "- If any DP backup is exposed as a graph node, wrap it in PureTransform or",
    ],
    &["DPOptions", "DPOutcome", "FiniteHorizonDPStation"],
);
