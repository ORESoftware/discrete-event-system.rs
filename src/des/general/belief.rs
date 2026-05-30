//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/belief.ts`
//! Rust target: `src/des/general/belief.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/belief.ts",
    "src/des/general/belief.rs",
    &["RUST MIGRATION: target module src/des/general/belief.rs.", "RUST MIGRATION: DiscreteBelief<S> becomes a generic struct with explicit HashMap<S, f64> or Vec<(S, f64)> storage depending on ordering needs.", "RUST MIGRATION: Instance methods map to impl DiscreteBelief<S>; require S: Eq + Hash + Clone if using HashMap, and return Result for invalid probabilities.", "RUST MIGRATION: brierScore and klDivergence are pure numeric helpers and can remain free functions."],
    &["DiscreteBelief", "brierScore", "klDivergence"],
);
