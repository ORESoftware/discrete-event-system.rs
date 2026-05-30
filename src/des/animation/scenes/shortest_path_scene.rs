//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/shortest-path-scene.ts`
//! Rust target: `src/des/animation/scenes/shortest_path_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/shortest-path-scene.ts",
    "src/des/animation/scenes/shortest_path_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/shortest_path_scene.rs", "- Keep buildShortestPathFrame/buildShortestPathCharts as module helpers over typed Graph and SPResult structs.", "- Private color/layout helpers remain private functions; arrays of Shape become Vec<Shape>.", "- Distances that are Infinity in TS need an explicit Rust representation such as Option<f64> or a Distance enum before serde.", "- If this scene is exposed as a DES graph node, lift it into a PureTransform struct with transform(result_snapshot) -> Frame."],
    &["STAGE_H", "STAGE_W", "buildShortestPathCharts", "buildShortestPathFrame"],
);
