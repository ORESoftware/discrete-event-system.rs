//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/neural-network.ts`
//! Rust target: `src/des/general/neural_network.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/neural-network.ts",
    "src/des/general/neural_network.rs",
    &["RUST MIGRATION: Target module `src/des/general/neural_network.rs`.", "RUST MIGRATION: Convert activation/solver-name unions to enums and configs, samples, options, traces, and results to `serde` structs.", "RUST MIGRATION: Port `FeedForwardNetwork`, supervised stations, Q-learning agent, ODE tokens/station, and prediction sink as structs implementing neural/RL/DES traits.", "RUST MIGRATION: Closure typedefs such as `StateEncoder` should become trait bounds or boxed `Fn` ports; exported runners can stay free functions unless graph-visible wrappers are needed.", "RUST MIGRATION: Replace `Math.random` call sites with an injected RNG trait/closure, and surface training/ODE validation failures as `Result`."],
    &["ActivationName", "DenseLayerConfig", "FeedForwardNetwork", "NeuralODEOptions", "NeuralODESolutionToken", "NeuralODESolveToken", "NeuralODESolverName", "NeuralODESolverStation", "NeuralPredictionSink", "NeuralQLearningAgent", "NeuralQLearningOptions", "NeuralQLearningResult", "StateEncoder", "SupervisedDatasetSource", "SupervisedNeuralNetDESResult", "SupervisedSample", "XOR_DATASET", "XorNeuralNetOptions", "oneHotEncoder", "runNeuralQLearningDES", "runSupervisedNeuralNetDES", "runXorNeuralNetDES", "solveNeuralODE"],
);
