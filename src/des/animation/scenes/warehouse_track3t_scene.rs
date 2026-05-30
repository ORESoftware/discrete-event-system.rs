//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/warehouse-track3t-scene.ts`
//! Rust target: `src/des/animation/scenes/warehouse_track3t_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/warehouse-track3t-scene.ts",
    "src/des/animation/scenes/warehouse_track3t_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/warehouse_track3t_scene.rs", "- Keep buildWarehouseComparisonFrame/frameCount/frameTime/buildWarehouseComparisonCharts as module helpers.", "- PanelGeom, MotionFrame, ReserveRow, and route/layout records become private Rust structs; Shape/ChartSpec are serde data from animation::types.", "- TS arrays/maps of traces and stations should become Vec plus HashMap/BTreeMap only where lookup or deterministic order is required.", "- If frame generation becomes DES graph-visible, wrap it as a WarehouseComparisonSceneTransform with transform(...)."],
    &["WAREHOUSE_TRACK3T_STAGE_H", "WAREHOUSE_TRACK3T_STAGE_W", "buildWarehouseComparisonCharts", "buildWarehouseComparisonFrame", "warehouseComparisonFrameCount", "warehouseComparisonFrameTime"],
);
