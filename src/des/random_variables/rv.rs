//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/random-variables/rv.ts`
//! Rust target: `src/des/random_variables/rv.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/random-variables/rv.ts",
    "src/des/random_variables/rv.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/random_variables/rv.rs",
        "- RandomVariable becomes a trait with rate/event-count/event-stream methods;",
        "- Event-count generation is a PureTransform boundary over RNG + time step;",
        "- `math.BigNumber` should use the project-wide Decimal/time alias; generator",
    ],
    &[
        "BernoulliRandomVariable",
        "ExponentialRandomVariable",
        "ExponentialRandomVariable2",
        "ExponentialRandomVariable3",
        "PoissonRandomVariable",
        "RandomVariable",
        "UniformRandomVariable",
        "UniformRandomVariable2",
    ],
);
