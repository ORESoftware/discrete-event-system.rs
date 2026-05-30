//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-lp-bridge.ts`
//! Rust target: `src/des/general/des_lp_bridge.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-lp-bridge.ts",
    "src/des/general/des_lp_bridge.rs",
    &["RUST MIGRATION: target module src/des/general/des_lp_bridge.rs.", "RUST MIGRATION: MDPLPSolution and RollingHorizonStep become serde structs; generic S/M state-machine types should use explicit trait bounds.", "RUST MIGRATION: solveLPThenSimulate, buildMDPLP, solveMDPAsLP, and lpRollingHorizon are bridge helpers and can remain free functions unless registered as graph transforms.", "RUST MIGRATION: LP tables and policy maps should become Vec<Vec<f64>> or HashMap keyed by state/action; convert infeasible/invalid cases to Result."],
    &["MDPLPSolution", "RollingHorizonStep", "buildMDPLP", "lpRollingHorizon", "solveLPThenSimulate", "solveMDPAsLP"],
);
