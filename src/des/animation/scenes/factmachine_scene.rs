//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/factmachine-scene.ts`
//! Rust target: `src/des/animation/scenes/factmachine_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/factmachine-scene.ts",
    "src/des/animation/scenes/factmachine_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/factmachine_scene.rs", "- Keep buildFactMachineFrame/buildFactMachineCharts as module helpers that return Frame/ChartSpec serde structs.", "- ArchitectureFrameArgs, FactMachineParams, and FactMachineResult should be nominal Rust structs instead of structural object shapes.", "- If a scene builder is wired into the DES graph, introduce a FactMachineSceneTransform implementing PureTransform::transform.", "- Local color/draw helpers stay private; arrays of Shape become Vec<Shape> and optional labels become Option<String>."],
    &["ArchitectureFrameArgs", "STAGE_H", "STAGE_W", "buildFactMachineCharts", "buildFactMachineFrame"],
);
