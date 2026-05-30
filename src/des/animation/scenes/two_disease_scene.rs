//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/two-disease-scene.ts`
//! Rust target: `src/des/animation/scenes/two_disease_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/two-disease-scene.ts",
    "src/des/animation/scenes/two_disease_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/two_disease_scene.rs", "- CompartmentCounts should become a Rust struct or enum-indexed map; COLORS can be a match over compartment enum variants.", "- buildBars/buildFrame/buildCompartmentChart stay module helpers returning Vec<Shape>, Frame fragments, and ChartSpec.", "- Animation/Frame/Shape/ChartSpec are serde data from animation::types; optional captions become Option<String>.", "- If this renderer is wired into the DES graph, wrap snapshot-to-frame as a PureTransform implementor."],
    &["COLORS", "CompartmentCounts", "STAGE_H", "STAGE_W", "buildBars", "buildCompartmentChart", "buildFrame"],
);
