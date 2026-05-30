//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/entity-registration.ts`
//! Rust target: `src/des/general/entity_registration.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/entity-registration.ts",
    "src/des/general/entity_registration.rs",
    &["RUST MIGRATION: target module src/des/general/entity_registration.rs.", "RUST MIGRATION: vals and reg are runtime registration tables; use HashMap<&'static str, EntityRegistration> or a generated enum-backed registry.", "RUST MIGRATION: Entity classes imported here should be referenced through trait objects or factory functions rather than TS constructor values.", "RUST MIGRATION: Keep this module side-effect-light so Rust can expose an explicit register_entities(registry: &mut Registry) free function."],
    &["reg"],
);
