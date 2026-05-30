//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/obs-ctrl-scene.ts`
//! Rust target: `src/des/animation/scenes/obs_ctrl_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/obs-ctrl-scene.ts",
    "src/des/animation/scenes/obs_ctrl_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/obs_ctrl_scene.rs", "- StoryStep aliases Frame-without-time in TS; make it a small Rust struct or reuse a FrameBuilder data type.", "- ObsCtrlScene becomes a storyboard struct with inherent methods returning Vec<Shape>/Frame fragments, not a superclass hierarchy.", "- Matrix imports should stay as typed linear_algebra module values; avoid serde_json::Value except at external boundaries.", "- If this storyboard becomes graph-visible, expose a PureTransform implementation that maps control-system state to StoryStep."],
    &["OC_STAGE_H", "OC_STAGE_W", "ObsCtrlScene", "StoryStep"],
);
