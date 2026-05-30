//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/control-blocks.ts`
//! Rust target: `src/des/general/des_base/control_blocks.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/control-blocks.ts",
    "src/des/general/des_base/control_blocks.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/control_blocks.rs",
        "- Keep file-for-file. VectorSignal becomes a signal/token struct, while",
        "- ClosedLoopOpts and ClosedLoopResult become data structs; vectors/matrices",
        "- runClosedLoop and ensureConnected can remain module functions; pure control",
        "- Replace dt/connection validation throws with Result-returning constructors.",
    ],
    &[
        "ClosedLoopOpts",
        "ClosedLoopResult",
        "ControllerBlock",
        "EstimatorBlock",
        "PlantBlock",
        "VectorSignal",
        "runClosedLoop",
    ],
);
