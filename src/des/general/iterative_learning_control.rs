//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/iterative-learning-control.ts`
//! Rust target: `src/des/general/iterative_learning_control.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/iterative-learning-control.ts",
    "src/des/general/iterative_learning_control.rs",
    &["RUST MIGRATION: target module src/des/general/iterative_learning_control.rs.", "RUST MIGRATION: ILCReferenceKind becomes an enum; params, trial summaries, and results become serde structs.", "RUST MIGRATION: ILC token classes and DESStation subclasses become Token/Station trait impl structs with explicit owned Vec<f64> control/reference programs.", "RUST MIGRATION: runIterativeLearningControl is DES-visible orchestration and should be a PureTransform entry struct; helper reference/RMS functions stay private free functions.", "RUST MIGRATION: Validation and clamp/bounds errors should return Result instead of throwing."],
    &["ILCReferenceKind", "ILCTrialSummary", "IterativeLearningControlParams", "IterativeLearningControlResult", "runIterativeLearningControl"],
);
