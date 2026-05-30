//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/external-modules.ts`
//! Rust target: `src/des/runners/external_modules.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/external-modules.ts",
    "src/des/runners/external_modules.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/external_modules.rs.", "- Keep this as the built-in external adapter registry; constants become pub const IDs and registration becomes an idempotent initializer.", "- Convert loose ExternalModuleParams extraction into typed adapter config structs with serde validation at the boundary.", "- Preserve Python/SUMO reference modules as external adapters invoked through the migrated external_program process port."],
    &["COMPUTER_NETWORK_FEL_REFERENCE_ID", "COMPUTER_NETWORK_REFERENCE_ID", "IP_MIP_REFERENCE_ID", "NEURAL_NETWORK_REFERENCE_ID", "TRAFFIC_CIW_REFERENCE_ID", "TRAFFIC_FEL_REFERENCE_ID", "TRAFFIC_SIMPY_REFERENCE_ID", "TRAFFIC_SUMO_REFERENCE_ID", "registerBuiltInExternalModules"],
);
