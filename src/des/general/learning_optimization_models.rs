//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/learning-optimization-models.ts`
//! Rust target: `src/des/general/learning_optimization_models.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/learning-optimization-models.ts",
    "src/des/general/learning_optimization_models.rs",
    &["RUST MIGRATION: target module src/des/general/learning_optimization_models.rs.", "RUST MIGRATION: SupervisedSample and regression/training params/results become serde structs; inherited params should be flattened explicitly.", "RUST MIGRATION: RegressionFitToken and station classes become Token/Station/GradientOptimizer trait impl structs rather than subclasses.", "RUST MIGRATION: runLinearRegressionLS, runRidgeRegressionLS, runLogisticRegressionSGD, and runBackpropMLPClassifier are graph-visible model transforms; expose as PureTransform entry structs.", "RUST MIGRATION: Linear algebra helpers stay free functions or move to a shared module; matrices are Vec<Vec<f64>> unless using nalgebra.", "RUST MIGRATION: Default sample builders and accuracy helpers stay pure free functions; validation/training failures return Result."],
    &["BackpropMLPParams", "GradientTrainingResult", "LinearRegressionParams", "LinearRegressionResult", "LogisticRegressionSGDParams", "RidgeRegressionParams", "SupervisedSample", "multiclassAccuracy", "runBackpropMLPClassifier", "runLinearRegressionLS", "runLogisticRegressionSGD", "runRidgeRegressionLS"],
);
