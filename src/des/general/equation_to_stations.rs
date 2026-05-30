//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/equation-to-stations.ts`
//! Rust target: `src/des/general/equation_to_stations.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/equation-to-stations.ts",
    "src/des/general/equation_to_stations.rs",
    &["RUST MIGRATION: target module src/des/general/equation_to_stations.rs.", "RUST MIGRATION: ODEScheme, Field1DScheme, BC, and Field2DScheme become enums; ODESystemSpec, Field1DSpec, Field1DBuild, Poisson2DSpec, and Poisson2DResult become serde structs.", "RUST MIGRATION: buildODESystem/buildField1D create DES-visible FieldSimulation networks, so expose them as PureTransform-style builders or Result-returning constructors.", "RUST MIGRATION: thomas and solvePoisson2D are numerical solvers and can stay free functions; represent Float64Array as Vec<f64> and validation failures as Result.", "RUST MIGRATION: Expression callbacks need a Rust expression trait or compiled closure object instead of ad hoc JS functions."],
    &["BC", "Field1DBuild", "Field1DScheme", "Field1DSpec", "Field2DScheme", "ODEScheme", "ODESystemSpec", "Poisson2DResult", "Poisson2DSpec", "buildField1D", "buildODESystem", "solvePoisson2D", "thomas"],
);
