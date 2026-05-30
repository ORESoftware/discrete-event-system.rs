//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/cut-pool.ts`
//! Rust target: `src/des/general/des_base/cut_pool.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/cut-pool.ts",
    "src/des/general/des_base/cut_pool.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/cut_pool.rs",
        "- Keep file-for-file. CutEnvelopeSense should become an enum and AffineCut a",
        "- AffineCutPool becomes a concrete struct with inherent methods; any public",
        "- Keep scalar/vector helper logic private or associated with AffineCutPool; if",
        "- Convert invalid dimensions or envelope misuse to Result-returning methods.",
    ],
    &["AffineCut", "AffineCutPool", "CutEnvelopeSense"],
);
