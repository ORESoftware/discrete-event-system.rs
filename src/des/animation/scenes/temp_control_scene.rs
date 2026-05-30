//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/temp-control-scene.ts`
//! Rust target: `src/des/animation/scenes/temp_control_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/temp-control-scene.ts",
    "src/des/animation/scenes/temp_control_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/temp_control_scene.rs", "- SceneData should become a Rust struct and buildTempControlFrame/buildTempControlAnimation remain module helpers.", "- RunResult/TickRecord stay typed imports from general::temp_control; animation output is serde Animation/Frame/Shape data.", "- Private drawing/chart helpers should take &mut Vec<Shape> and return Result only if rendering can fail.", "- If temperature-control frames become graph-visible, use a TempControlSceneTransform implementing PureTransform::transform."],
    &["STAGE_H", "STAGE_W", "buildTempControlAnimation", "buildTempControlFrame"],
);
