//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/types.ts`
//! Rust target: `src/des/runners/types.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/types.ts",
    "src/des/runners/types.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/types.rs.", "- Keep this as the shared data-model module; Kernel becomes an enum and SimConfig/RunOpts/RunResult become serde structs.", "- Replace Record<string, ...> with HashMap/BTreeMap or typed compartment enums where deterministic order matters.", "- Constants such as COMPARTMENT_ORDER, DEFAULT_CONFIG, and EDGES should become const/static constructors with explicit ownership."],
    &["COMPARTMENT_GROUPS", "COMPARTMENT_ORDER", "DEFAULT_CONFIG", "DEFAULT_RESIDENCE", "EDGES", "Kernel", "RunOpts", "RunResult", "SimConfig", "buildSuccessors"],
);
