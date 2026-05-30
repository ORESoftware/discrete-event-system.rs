//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/neural-network.ts`
//! Rust target: `src/des/general/des_base/neural_network.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/neural-network.ts",
    "src/des/general/des_base/neural_network.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/neural_network.rs",
        "- Keep file-for-file. NumericVector aliases map to Vec<f64>; neural network",
        "- Neural token classes become token structs. NeuralNetworkStation and",
        "- Pure inference/training adapters can stay methods; graph-level inference",
        "- Convert invalid sample shapes and backend failures to Result.",
    ],
    &[
        "NeuralInferenceToken",
        "NeuralNetworkLike",
        "NeuralNetworkStation",
        "NeuralPredictionToken",
        "NeuralSnapshotToken",
        "NeuralTrainingResultToken",
        "NumericVector",
        "SupervisedNeuralNetworkStation",
        "SupervisedNeuralNetworkStationOptions",
        "SupervisedSampleToken",
        "TrainableNeuralNetwork",
    ],
);
