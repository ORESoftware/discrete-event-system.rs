//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/single-state-optimizer.ts`
//! Rust target: `src/des/general/des_base/single_state_optimizer.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/single-state-optimizer.ts",
    "src/des/general/des_base/single_state_optimizer.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/single_state_optimizer.rs",
        "- Keep file-for-file. Channel constants become pub consts; initial/result",
        "- SingleStateSourceStation and SingleStateSinkStation become concrete",
        "- Proposal, cost, accept, clone, and stop hooks map to trait methods; pure",
        "- Convert duplicate seed, uninitialized optimizer, and non-finite cost throws",
        "- SA:           accept if Δ ≤ 0 OR rng() < exp(−Δ/T_iter)",
        "- Hill climb:   accept iff Δ < 0",
        "- Tabu:         accept best non-tabu candidate (uses memory)",
        "- Threshold:    accept if Δ ≤ τ_iter",
    ],
    &[
        "SINGLE_STATE_INITIAL_CHANNEL",
        "SINGLE_STATE_RESULT_CHANNEL",
        "SingleStateInitialToken",
        "SingleStateOptimizer",
        "SingleStateResultSnapshot",
        "SingleStateResultToken",
        "SingleStateSinkStation",
        "SingleStateSourceStation",
    ],
);
