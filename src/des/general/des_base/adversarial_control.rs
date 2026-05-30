//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/adversarial-control.ts`
//! Rust target: `src/des/general/des_base/adversarial_control.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/adversarial-control.ts",
    "src/des/general/des_base/adversarial_control.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/adversarial_control.rs",
        "- Keep file-for-file. Channel constants become pub const values; observation,",
        "- ClosedLoopGameTraceRow and ClosedLoopPlantOptions become data structs, while",
        "- wireClosedLoopGame and runClosedLoopGame can remain module functions for",
        "- Convert validation and runner failures from thrown errors to Result.",
    ],
    &[
        "CH_CONTROL",
        "CH_DISTURBANCE",
        "CH_OBSERVATION",
        "ClosedLoopGameRunOptions",
        "ClosedLoopGameTraceRow",
        "ClosedLoopPlantOptions",
        "ClosedLoopPlantStation",
        "ControlMoveToken",
        "DisturbanceMoveToken",
        "DisturbancePolicyStation",
        "FeedbackPolicyStation",
        "StateObservationToken",
        "runClosedLoopGame",
        "wireClosedLoopGame",
    ],
);
