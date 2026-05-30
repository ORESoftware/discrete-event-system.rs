//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/feedback-linearization.ts`
//! Rust target: `src/des/general/feedback_linearization.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/feedback-linearization.ts",
    "src/des/general/feedback_linearization.rs",
    &["RUST MIGRATION: target module src/des/general/feedback_linearization.rs.", "RUST MIGRATION: PendulumParams, FeedbackLinearizationOpts, and FeedbackLinearizationResult become serde structs; ClosedLoopResult inheritance becomes composition/flattening.", "RUST MIGRATION: PendulumPlant and FeedbackLinearizationController become structs implementing Plant/Controller traits instead of subclassing blocks.", "RUST MIGRATION: runFeedbackLinearization is a DES/control transform returning Result; rk4 is a private free function over slices/arrays.", "RUST MIGRATION: Keep vector math ownership explicit with [f64; 2] where possible to make borrow checking simple."],
    &["FeedbackLinearizationOpts", "FeedbackLinearizationResult", "PendulumParams", "runFeedbackLinearization"],
);
