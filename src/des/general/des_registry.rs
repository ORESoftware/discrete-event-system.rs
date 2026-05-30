//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-registry.ts`
//! Rust target: `src/des/general/des_registry.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-registry.ts",
    "src/des/general/des_registry.rs",
    &["RUST MIGRATION: target module src/des/general/des_registry.rs.", "RUST MIGRATION: REGISTRY becomes a HashMap<String, Box<dyn DESModelRegistrationDyn>> or typed enum registry; avoid TS any by introducing an object-safe trait.", "RUST MIGRATION: registerModel/getModel/listModels/runFromSpec/runFromJsonFile are registry/IO free functions; async file reads map to tokio or std fs depending on runtime.", "RUST MIGRATION: RunFromSpecOptions and DESRunSummary-compatible results become serde structs, and validation/lookup/file errors should return Result."],
    &["RunFromSpecOptions", "getModel", "listModels", "registerModel"],
);
