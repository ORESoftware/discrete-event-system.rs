//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/grid-localization-pomdp.ts`
//! Rust target: `src/des/general/grid_localization_pomdp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/grid-localization-pomdp.ts",
    "src/des/general/grid_localization_pomdp.rs",
    &["RUST MIGRATION: target module src/des/general/grid_localization_pomdp.rs.", "RUST MIGRATION: GridLocalizationActionKind and GridLocalizationObservation become enums; params/actions/trace/result/model become serde structs.", "RUST MIGRATION: buildGridLocalizationPOMDP and runGridLocalizationPOMDP are POMDP model/transformation entrypoints; expose graph-visible ones as PureTransform structs.", "RUST MIGRATION: Dense transition/observation/belief tables should use Vec matrices; sampling takes injected rand::Rng and validation returns Result."],
    &["GridLocalizationAction", "GridLocalizationActionKind", "GridLocalizationObservation", "GridLocalizationParams", "GridLocalizationResult", "GridLocalizationTraceRow", "buildGridLocalizationPOMDP", "runGridLocalizationPOMDP"],
);
