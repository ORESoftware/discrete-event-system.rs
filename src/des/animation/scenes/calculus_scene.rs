//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/calculus-scene.ts`
//! Rust target: `src/des/animation/scenes/calculus_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/calculus-scene.ts",
    "src/des/animation/scenes/calculus_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/calculus_scene.rs", "- Keep buildField1DFrame/buildField1DChart/buildPoissonFrame as module helpers returning Frame/ChartSpec serde data.", "- Numeric arrays should become Vec<f64> or typed matrix/grid structs; choose a matrix crate only if later callers need it.", "- valueToColor and projection helpers remain private pure functions.", "- If a PDE field renderer becomes DES graph-visible, wrap it in a PureTransform struct with transform(field_state) -> Frame fragment."],
    &["POISSON_H", "POISSON_W", "STAGE_H", "STAGE_W", "buildField1DChart", "buildField1DFrame", "buildPoissonFrame"],
);
