//! Thin binary migration scaffold generated from the TypeScript runner.
//! TypeScript source: `src/des/runners/per-individual-vs-fel.ts`
//! Rust target: `src/bin/per_individual_vs_fel.rs`

#![allow(dead_code)]

use discrete_event_system_rs::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/per-individual-vs-fel.ts",
    "src/bin/per_individual_vs_fel.rs",
    &["RUST MIGRATION:", "- Target: src/bin/per_individual_vs_fel.rs.", "- Keep this as a CLI comparison runner with Result-returning main; map N and other env knobs to clap/std::env.", "- Treat collected statistics as golden comparison structs and reuse the migrated stats module for Welch summaries.", "- Keep kernel invocations as calls into src/des/runners/* modules; formatting helpers remain private."],
    &[],
);

fn main() -> anyhow::Result<()> {
    // The TypeScript runner body is intentionally kept as a thin CLI
    // port target. Shared model construction belongs in the library.
    Ok(())
}
