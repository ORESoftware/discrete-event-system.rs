//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/dc-motor-scene.ts`
//! Rust target: `src/des/animation/scenes/dc_motor_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/dc-motor-scene.ts",
    "src/des/animation/scenes/dc_motor_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/dc_motor_scene.rs", "- DcMotorSceneOpts becomes a Rust config struct and DcMotorScene becomes a state/lightweight builder struct with inherent methods.", "- MotorStateToken and DcMotorParams should stay typed imports from general::control_systems::dc_motor.", "- Returned Frame/ChartSpec/Shape values should be serde structs/enums; optional controls become Option<T>.", "- If the class is inserted into the DES graph, implement PureTransform for DcMotorScene with transform(state) -> Frame fragment."],
    &["DcMotorScene", "DcMotorSceneOpts", "MOTOR_STAGE_H", "MOTOR_STAGE_W"],
);
