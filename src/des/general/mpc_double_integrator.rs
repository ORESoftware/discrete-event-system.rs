//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/mpc-double-integrator.ts`
//! Rust target: `src/des/general/mpc_double_integrator.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/mpc-double-integrator.ts",
    "src/des/general/mpc_double_integrator.rs",
    &["RUST MIGRATION: target module src/des/general/mpc_double_integrator.rs.", "RUST MIGRATION: MPCDoubleIntOpts and MPCDoubleIntResult become serde structs; ClosedLoopResult extension should be flattened or composed.", "RUST MIGRATION: DoubleIntegratorPlant and MPCDoubleIntegratorController become Plant/Controller trait impl structs instead of block subclasses.", "RUST MIGRATION: runMPCDoubleIntegrator is a DES/control PureTransform returning Result; projected-gradient QP helper logic can remain private free functions.", "RUST MIGRATION: Use fixed [f64; 2] state/control vectors where possible and return Result for infeasible horizon/bounds settings."],
    &["MPCDoubleIntOpts", "MPCDoubleIntResult", "runMPCDoubleIntegrator"],
);
