//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/double-integrator-lqr.ts`
//! Rust target: `src/des/general/double_integrator_lqr.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/double-integrator-lqr.ts",
    "src/des/general/double_integrator_lqr.rs",
    &["RUST MIGRATION: target module src/des/general/double_integrator_lqr.rs.", "RUST MIGRATION: DoubleIntegratorOpts and DoubleIntegratorResult become serde structs; matrix/vector traces should use Vec<[f64; 2]> or Vec<Vec<f64>> consistently.", "RUST MIGRATION: runDoubleIntegratorLQR is a control simulation entrypoint and should be a PureTransform-style struct returning Result.", "RUST MIGRATION: gaussian noise must use an injected rand::Rng plus a normal sampler crate; validation/linear algebra failures should return Result."],
    &["DoubleIntegratorOpts", "DoubleIntegratorResult", "runDoubleIntegratorLQR"],
);
