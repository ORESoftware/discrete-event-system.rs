//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/linear-vfa.ts`
//! Rust target: `src/des/general/des_base/linear_vfa.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/linear-vfa.ts",
    "src/des/general/des_base/linear_vfa.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/linear_vfa.rs",
        "- Keep file-for-file. LinearVFAOptions becomes a config struct and",
        "- Feature extraction and value approximation hooks should become trait",
        "- Pure feature functions can stay associated/private helpers, or become",
        "- Convert feature dimension, action count, and feature-shape throws to Result.",
    ],
    &["LinearVFAOptions", "LinearVFAStation"],
);
