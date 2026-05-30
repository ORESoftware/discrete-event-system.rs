//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/advanced-control-models.ts`
//! Rust target: `src/des/general/advanced_control_models.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/advanced-control-models.ts",
    "src/des/general/advanced_control_models.rs",
    &["RUST MIGRATION: target module src/des/general/advanced_control_models.rs.", "RUST MIGRATION: HInfinityRobustControlParams/Result and PursuitEvasionGameParams/Result become serde structs; advancedControlChannels can be associated consts.", "RUST MIGRATION: ScalarRobustPlant, PursuitEvasionPlant, controllers, and disturbance policies become structs implementing closed-loop station/policy traits instead of TS inheritance.", "RUST MIGRATION: runHInfinityRobustControl and runPursuitEvasionGame should be PureTransform entry structs because they assemble DES-visible closed-loop graphs; return Result for validation."],
    &["HInfinityRobustControlParams", "HInfinityRobustControlResult", "PursuitEvasionGameParams", "PursuitEvasionGameResult", "advancedControlChannels", "runHInfinityRobustControl", "runPursuitEvasionGame"],
);
