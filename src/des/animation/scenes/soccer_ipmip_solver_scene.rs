//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/soccer-ipmip-solver-scene.ts`
//! Rust target: `src/des/animation/scenes/soccer_ipmip_solver_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/soccer-ipmip-solver-scene.ts",
    "src/des/animation/scenes/soccer_ipmip_solver_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/soccer_ipmip_solver_scene.rs", "- Keep frame count, frame builder, and chart builder as module helpers over typed IPMIP solution/event structs.", "- NodeBox and Edge become private structs; NODE_BY_ID should become a lazy static map or a small lookup helper.", "- Shape/ChartSpec outputs map directly to serde data; nullable trace events become Option<IPMIPTraceEvent>.", "- If solver animation is graph-visible, wrap the event-to-frame logic in a PureTransform implementor."],
    &["SOCCER_IPMIP_SOLVER_H", "SOCCER_IPMIP_SOLVER_W", "SOLVER_FRAMES_PER_EVENT", "buildSoccerIPMIPSolverCharts", "buildSoccerIPMIPSolverFrame", "soccerIPMIPSolverFrameCount"],
);
