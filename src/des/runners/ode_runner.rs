//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/ode-runner.ts`
//! Rust target: `src/des/runners/ode_runner.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/ode-runner.ts",
    "src/des/runners/ode_runner.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/ode_runner.rs.", "- Keep file-for-file as a library runner exposing run_ode_once; State becomes a fixed-field struct or indexed compartment vector.", "- Keep RK/linear-combination helpers private pure functions unless lifted into a numerical Integrator trait.", "- Convert logging/output construction to Result-capable boundaries while preserving RunResult compatibility with other kernels."],
    &["runOdeOnce"],
);
