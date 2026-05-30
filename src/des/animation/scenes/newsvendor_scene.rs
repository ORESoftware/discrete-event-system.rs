//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/newsvendor-scene.ts`
//! Rust target: `src/des/animation/scenes/newsvendor_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/newsvendor-scene.ts",
    "src/des/animation/scenes/newsvendor_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/newsvendor_scene.rs", "- NewsvendorFrameData becomes a Rust struct; buildNewsvendorFrame/buildNewsvendorChart remain module helpers.", "- COLORS should become constants or a small palette struct; optional chart bounds/captions become Option<T>.", "- Keep helper calculations private and return typed Frame/ChartSpec/Shape serde data.", "- If inventory rendering becomes DES graph-visible, wrap NewsvendorFrameData -> Frame in a PureTransform implementor."],
    &["NewsvendorFrameData", "STAGE_H", "STAGE_W", "buildNewsvendorChart", "buildNewsvendorFrame"],
);
