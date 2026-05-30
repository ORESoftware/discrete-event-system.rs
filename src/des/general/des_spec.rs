//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-spec.ts`
//! Rust target: `src/des/general/des_spec.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-spec.ts",
    "src/des/general/des_spec.rs",
    &["RUST MIGRATION: target module src/des/general/des_spec.rs.", "RUST MIGRATION: DESModelSpec, DESRuntimeConfig, DESModelMetadata, ValidationResult, DESModelRegistration, and DESRunSummary become serde structs.", "RUST MIGRATION: ParamSchema is a discriminated union and should become an enum with tagged serde variants; zod integration becomes a validator trait or feature-gated adapter.", "RUST MIGRATION: validate/validateInner/typeOf stay free functions returning Result/ValidationResult, with Record<string, unknown> mapped to serde_json::Map or HashMap<String, Value>."],
    &["DESModelMetadata", "DESModelRegistration", "DESModelSpec", "DESRunSummary", "DESRuntimeConfig", "ParamSchema", "ValidationResult", "validate"],
);
