//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/smart-movable.ts`
//! Rust target: `src/des/general/des_base/smart_movable.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/smart-movable.ts",
    "src/des/general/des_base/smart_movable.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/smart_movable.rs",
        "- Keep file-for-file. SmartMovable becomes a trait plus shared state struct",
        "- Token and IterativeDESParticipant implementations map to trait impls; any",
        "- ValidationCheck lists should use Vec<ValidationCheck> from validation.rs.",
        "- No free helpers here; pure behavior lifted into the DES graph should use",
    ],
    &["SmartMovable"],
);
