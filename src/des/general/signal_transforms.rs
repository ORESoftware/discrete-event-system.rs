//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/signal-transforms.ts`
//! Rust target: `src/des/general/signal_transforms.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/signal-transforms.ts",
    "src/des/general/signal_transforms.rs",
    &["RUST MIGRATION: Target module `src/des/general/signal_transforms.rs`.", "RUST MIGRATION: Convert transform-kind/quadrature-rule unions to enums and all complex, sample, contribution, params, framework-summary, and result shapes to `serde` structs.", "RUST MIGRATION: Port transform sample/contribution/total tokens plus source/kernel/accumulator/sink stations as structs implementing `Token`/`DESStation`.", "RUST MIGRATION: Keep complex arithmetic and direct/reference transform helpers as private free functions; graph-visible kernels can become `PureTransform` structs with `transform()`.", "RUST MIGRATION: Return `Result` for invalid complex points, constants, expressions, quadrature choices, and non-finite transform outputs."],
    &["ComplexPoint", "ComplexPointInput", "ComplexValue", "FourierTransformParams", "LaplaceTransformParams", "QuadratureRule", "TransformContributionRecord", "TransformEntityFrameworkSummary", "TransformKind", "TransformOutputPoint", "TransformRunResult", "TransformSampleRecord", "ZTransformParams", "formatComplex", "runFourierTransform", "runLaplaceTransform", "runZTransform"],
);
