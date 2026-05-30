//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/tiger-pomdp.ts`
//! Rust target: `src/des/general/tiger_pomdp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/tiger-pomdp.ts",
    "src/des/general/tiger_pomdp.rs",
    &["RUST MIGRATION: Target module `src/des/general/tiger_pomdp.rs`.", "RUST MIGRATION: Convert tiger constants to state/action/observation enums and params/results/options to `serde` structs.", "RUST MIGRATION: Port `QMDPStation` and `OneStepLookAheadStation` as structs implementing the belief-state station trait, with shared belief logic composed explicitly.", "RUST MIGRATION: Keep spec builders and simulators as free functions; use `HashMap` for label/index conversion where classic specs cross into numeric core specs.", "RUST MIGRATION: Inject RNG for noisy listening and reset behavior, and return `Result` for malformed specs, invalid beliefs, or unsupported action labels."],
    &["ACT_LISTEN", "ACT_OPEN_LEFT", "ACT_OPEN_RIGHT", "OBS_HEAR_LEFT", "OBS_HEAR_RIGHT", "OneStepLookAheadStation", "QMDPStation", "TIGER_LEFT", "TIGER_RIGHT", "TigerOpts", "TigerSimOpts", "TigerSimResult", "buildTigerSpec", "simulateTiger"],
);
