//! Thin binary migration scaffold generated from the TypeScript runner.
//! TypeScript source: `src/des/runners/steady-state.ts`
//! Rust target: `src/bin/steady_state.rs`

#![allow(dead_code)]

use discrete_event_system_rs::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/steady-state.ts",
    "src/bin/steady_state.rs",
    &["RUST MIGRATION:", "- Target: src/bin/steady_state.rs.", "- Keep this as a CLI analysis runner with Result-returning main; map N_REPS/HORIZON and output options to clap/std::env.", "- Convert steady-state tables into nominal structs and reuse serde_json/csv-style writers at the boundary.", "- Keep analytical comparisons wired to difference_runner and shared stats modules rather than duplicating math."],
    &[],
);

fn main() -> anyhow::Result<()> {
    // The TypeScript runner body is intentionally kept as a thin CLI
    // port target. Shared model construction belongs in the library.
    Ok(())
}
