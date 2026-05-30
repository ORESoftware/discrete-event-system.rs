//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/cartesian-state-space.ts`
//! Rust target: `src/des/general/cartesian_state_space.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/cartesian-state-space.ts",
    "src/des/general/cartesian_state_space.rs",
    &["RUST MIGRATION: target module src/des/general/cartesian_state_space.rs.", "RUST MIGRATION: CartesianDimension, CoordinateTransition, and CoordinateMDPSpec become serde structs; optional fields become Option<T>.", "RUST MIGRATION: CartesianStateSpace becomes a nominal struct with impl methods for coordinate/index conversion; preserve checked conversions with Result.", "RUST MIGRATION: coordinateMDPToSpec is a pure adapter and can stay a free function unless exposed as a PureTransform in the Rust graph API."],
    &["CartesianDimension", "CartesianStateSpace", "CoordinateMDPSpec", "CoordinateTransition", "coordinateMDPToSpec"],
);
