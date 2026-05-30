//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/time-accrued.ts`
//! Rust target: `src/des/general/time_accrued.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/time-accrued.ts",
    "src/des/general/time_accrued.rs",
    &["RUST MIGRATION: Target module `src/des/general/time_accrued.rs`.", "RUST MIGRATION: Replace the module-level mutable object with an explicit `TimeAccrued` struct whose current time and step size are owned by the simulation context.", "RUST MIGRATION: Map `math.BigNumber` to a chosen Rust numeric type (`f64`, `rust_decimal`, or big rational) consistently with the rest of the time model.", "RUST MIGRATION: Convert getter/setter arrow functions into methods on `TimeAccrued`; avoid global mutable state unless guarded by an explicit runtime handle.", "RUST MIGRATION: Return `Result` from setters/bump methods for negative, zero, or non-finite step sizes."],
    &["bumpTimeAccruedByTimeStep", "getStepSize", "getTimeAccrued", "setStepSize"],
);
