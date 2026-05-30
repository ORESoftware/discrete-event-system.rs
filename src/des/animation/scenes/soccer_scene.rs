//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/soccer-scene.ts`
//! Rust target: `src/des/animation/scenes/soccer_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/soccer-scene.ts",
    "src/des/animation/scenes/soccer_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/soccer_scene.rs", "- SoccerFrameInput becomes a Rust struct; buildSoccerFrame/buildSoccerCharts stay module helpers returning Frame/ChartSpec data.", "- POSITION_RELATIVE and color constants can be const arrays/strings; optional events and labels become Option<T>.", "- Keep draw/layout helpers private and pass &mut Vec<Shape> where TS mutates a Shape array.", "- If frame generation is made DES graph-visible, wrap it in a SoccerSceneTransform implementing PureTransform::transform."],
    &["STAGE_H", "STAGE_W", "SoccerFrameInput", "buildSoccerCharts", "buildSoccerFrame"],
);
