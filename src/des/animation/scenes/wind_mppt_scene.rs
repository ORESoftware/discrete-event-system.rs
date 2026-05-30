//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/wind-mppt-scene.ts`
//! Rust target: `src/des/animation/scenes/wind_mppt_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/wind-mppt-scene.ts",
    "src/des/animation/scenes/wind_mppt_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/wind_mppt_scene.rs", "- WindSceneOpts becomes a config struct and WindMpptScene becomes a Rust builder struct with inherent frame/chart methods.", "- TurbineStateToken stays a typed import from general::control_systems::wind_mppt; optional series become Option<Vec<_>>.", "- Shape and ChartSpec outputs are serde structs/enums; keep private drawing helpers as functions over &mut Vec<Shape>.", "- If used as a DES graph node, implement PureTransform for WindMpptScene with transform(token) -> Frame fragment."],
    &["WIND_STAGE_H", "WIND_STAGE_W", "WindMpptScene", "WindSceneOpts"],
);
