//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/learning-optimization.ts`
//! Rust target: `src/des/general/des_base/learning_optimization.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/learning-optimization.ts",
    "src/des/general/des_base/learning_optimization.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/learning_optimization.rs",
        "- Keep file-for-file. Token classes become token structs; StationGraphSummary,",
        "- Source/sink/evaluator/optimizer station classes become structs implementing",
        "- Pure numeric helpers such as dot, norm2, sigmoid, softmax, zeros, and",
        "- Convert batch-size, learning-rate, and gradient-shape throws to Result.",
    ],
    &[
        "CandidateEvaluatorStation",
        "CandidateSourceStation",
        "CandidateToken",
        "EvaluatedCandidateToken",
        "GradientEvaluation",
        "GradientOptimizerOptions",
        "GradientOptimizerStation",
        "GradientStepToken",
        "GradientTraceSinkStation",
        "IncumbentSinkStation",
        "IncumbentToken",
        "LatestTokenSinkStation",
        "MiniBatchStation",
        "SingleTokenSourceStation",
        "StationGraphSummary",
        "VectorBatchToken",
        "VectorSampleSourceStation",
        "VectorSampleToken",
        "channelEdge",
        "cloneMatrix",
        "dot",
        "emptyStationGraph",
        "nonEmptyArray",
        "norm2",
        "runStateLoopPipeline",
        "sigmoid",
        "softmax",
        "stateLoopTopology",
        "stationGraph",
        "zeros",
    ],
);
