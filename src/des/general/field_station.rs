//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/field-station.ts`
//! Rust target: `src/des/general/field_station.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/field-station.ts",
    "src/des/general/field_station.rs",
    &["RUST MIGRATION: target module src/des/general/field_station.rs.", "RUST MIGRATION: Station, Census, FieldStation, and FieldSimulation become structs implementing shared TimeSteppedStation/Station traits instead of TS inheritance.", "RUST MIGRATION: FieldUpdater is behavior; model it as a trait object/closure type with explicit borrow rules over mutable field buffers.", "RUST MIGRATION: FieldSimulationOptions and FieldSimulationResult become serde structs; traces/grid states should use Vec<f64> or Vec<Vec<f64>>.", "RUST MIGRATION: shuffleInPlace takes injected rand::Rng, and simulation construction/validation should return Result."],
    &["Census", "FieldSimulation", "FieldSimulationOptions", "FieldSimulationResult", "FieldStation", "FieldUpdater", "Station"],
);
