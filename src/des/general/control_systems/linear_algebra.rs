//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/control-systems/linear-algebra.ts`
//! Rust target: `src/des/general/control_systems/linear_algebra.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/control-systems/linear-algebra.ts",
    "src/des/general/control_systems/linear_algebra.rs",
    &["RUST MIGRATION: Target module `src/des/general/control_systems/linear_algebra.rs`.", "RUST MIGRATION: Convert `Vec`/`Mat` aliases, `LinAlg`, inverse, eigen, and rank helpers into nominal structs/traits over `Vec<f64>` and `Vec<Vec<f64>>`.", "RUST MIGRATION: Keep plant/controller/estimator users on explicit `f64` matrix/vector APIs and pass solver tolerances/config instead of globals.", "RUST MIGRATION: Any graph-visible pure matrix evaluator should be wrapped as a PureTransform-style struct with a `transform` method returning `Result`."],
    &["LinAlg", "Mat", "MatrixInverse", "MatrixRank", "SymmetricEigen", "Vec"],
);
