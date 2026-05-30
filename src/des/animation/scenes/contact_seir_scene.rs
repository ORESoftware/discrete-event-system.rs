//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/contact-seir-scene.ts`
//! Rust target: `src/des/animation/scenes/contact_seir_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/contact-seir-scene.ts",
    "src/des/animation/scenes/contact_seir_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/contact_seir_scene.rs", "- PersonView becomes a Rust struct and layoutGrid/buildContactFrame/buildContactChart remain module helpers.", "- State/color dictionaries should become enums plus match expressions where possible; arrays become Vec<f64>/Vec<PersonView>.", "- Frame/Shape/ChartSpec outputs map to serde structs/enums from animation::types.", "- If contact rendering becomes DES graph-visible, expose a PureTransform that maps epidemic snapshot -> Frame fragment."],
    &["PersonView", "STAGE_H", "STAGE_W", "buildContactChart", "buildContactFrame", "layoutGrid"],
);
