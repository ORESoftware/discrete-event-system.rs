//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/mountain-car.ts`
//! Rust target: `src/des/general/mountain_car.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/mountain-car.ts",
    "src/des/general/mountain_car.rs",
    &["RUST MIGRATION: target module src/des/general/mountain_car.rs.", "RUST MIGRATION: MountainCarOpts, MountainCarTrainOpts, and MountainCarResult become serde structs; episode traces should be Vec rows with explicit numeric fields.", "RUST MIGRATION: MountainCarEnv implements Environment and MountainCarLinearVFA implements LinearVFAStation behavior through Rust traits.", "RUST MIGRATION: runMountainCar is a training PureTransform returning Result; tile-coding/value-function tables should use Vec<f64> or HashMap feature indexes.", "RUST MIGRATION: All exploration and environment randomness must use injected rand::Rng."],
    &["MountainCarOpts", "MountainCarResult", "MountainCarTrainOpts", "runMountainCar"],
);
