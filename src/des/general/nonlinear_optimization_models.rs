//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/nonlinear-optimization-models.ts`
//! Rust target: `src/des/general/nonlinear_optimization_models.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/nonlinear-optimization-models.ts",
    "src/des/general/nonlinear_optimization_models.rs",
    &["RUST MIGRATION: Target module `src/des/general/nonlinear_optimization_models.rs`.", "RUST MIGRATION: Convert opt/NLS params, points, and results to `serde` structs; keep the topology alias as the Rust graph-summary type alias.", "RUST MIGRATION: Replace abstract update-station inheritance with traits for unconstrained and least-squares update behavior, implemented by Newton/BFGS/Gauss-Newton/LM structs.", "RUST MIGRATION: Solver runners can remain free functions because they build and execute DES graphs; graph-visible station leaves should be trait objects or enums where dispatch is finite.", "RUST MIGRATION: Turn validation, backtracking failure, and linear-solve singularity into `Result` errors, with vectors/matrices represented as `Vec<f64>`/`Vec<Vec<f64>>`."],
    &["CurveFitPoint", "NonlinearLeastSquaresParams", "NonlinearLeastSquaresResult", "NonlinearTopology", "UnconstrainedOptParams", "UnconstrainedOptResult", "runBFGSRosenbrock", "runGaussNewtonCurveFit", "runLevenbergMarquardtCurveFit", "runNewtonRosenbrock"],
);
