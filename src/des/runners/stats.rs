//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/stats.ts`
//! Rust target: `src/des/runners/stats.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/stats.ts",
    "src/des/runners/stats.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/stats.rs.", "- Keep this as a small numeric utility module; WelchResult becomes a serde-friendly struct if emitted in reports.", "- Pure helpers such as mean/stddev/welch/erf can remain private or pub module functions with f64 signatures.", "- Avoid panics for empty/degenerate inputs if callers need recoverability; otherwise preserve current NaN/zero semantics explicitly."],
    &["WelchResult", "mean", "sampleVariance", "stddev", "welch"],
);
