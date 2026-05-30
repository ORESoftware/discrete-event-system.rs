//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/genetic-tsp-scene.ts`
//! Rust target: `src/des/animation/scenes/genetic_tsp_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/genetic-tsp-scene.ts",
    "src/des/animation/scenes/genetic_tsp_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/genetic_tsp_scene.rs", "- Keep exported STAGE constants plus buildGeneticTSPFrame/buildGeneticTSPCharts as module helpers returning Frame/ChartSpec serde structs.", "- ArchitectureFrameArgs becomes a Rust struct; TSPInstance/Tour imports should stay typed domain structs from general::genetic_tsp.", "- If these scene builders become DES graph-visible, wrap them in a PureTransform struct with transform(...) rather than leaving a bare function.", "- Local draw helpers remain private functions that push into Vec<Shape>."],
    &["ArchitectureFrameArgs", "STAGE_H", "STAGE_W", "buildGeneticTSPCharts", "buildGeneticTSPFrame"],
);
