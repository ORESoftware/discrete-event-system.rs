//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/elevator-scene.ts`
//! Rust target: `src/des/animation/scenes/elevator_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/elevator-scene.ts",
    "src/des/animation/scenes/elevator_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/elevator_scene.rs", "- Keep buildElevatorFrame/buildElevatorChart as module helpers over a typed Building struct imported from main_elevator.", "- Direction/state string unions should become enums; color selection becomes match expressions.", "- Shape/ChartSpec outputs are serde structs/enums and private drawing helpers should push into Vec<Shape>.", "- If the elevator scene participates in the DES graph, wrap Building -> Frame generation as a PureTransform implementor."],
    &["STAGE_H", "STAGE_W", "buildElevatorChart", "buildElevatorFrame"],
);
