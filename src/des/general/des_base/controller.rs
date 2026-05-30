//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/controller.ts`
//! Rust target: `src/des/general/des_base/controller.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/controller.ts",
    "src/des/general/des_base/controller.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/controller.rs",
        "- Keep file-for-file. ObservationToken and ControlToken become token structs",
        "- ControllerStation becomes a trait plus shared station-state struct; preserve",
        "- Keep channel routing as explicit enums/consts where possible; any pure",
        "- Convert validation failures to Result instead of throwing.",
    ],
    &["ControlToken", "ControllerStation", "ObservationToken"],
);
