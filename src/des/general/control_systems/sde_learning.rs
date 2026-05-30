//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/control-systems/sde-learning.ts`
//! Rust target: `src/des/general/control_systems/sde_learning.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/control-systems/sde-learning.ts",
    "src/des/general/control_systems/sde_learning.rs",
    &["RUST MIGRATION: Target module `src/des/general/control_systems/sde_learning.rs`.", "RUST MIGRATION: Convert SDE families, MLE, EnKF, MLP, diffusion model, and estimator station into structs plus traits for estimators/controllers.", "RUST MIGRATION: Use `f64` vectors/matrices for trajectories, ensembles, weights, and gradients; inject RNG/config explicitly.", "RUST MIGRATION: Graph-visible pure estimators should follow PureTransform-style `transform` methods and return `Result` for fit/filter failures."],
    &["DenoisingDiffusionModel", "DiffusionOptions", "EnkfOptions", "EnsembleKalmanFilter", "EnsembleKalmanFilterStation", "GbmFamily", "MleFitResult", "Mlp", "OuFamily", "ParametricSdeFamily", "SdeMaximumLikelihoodEstimator"],
);
