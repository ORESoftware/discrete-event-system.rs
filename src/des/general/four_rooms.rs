//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/four-rooms.ts`
//! Rust target: `src/des/general/four_rooms.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/four-rooms.ts",
    "src/des/general/four_rooms.rs",
    &["RUST MIGRATION: target module src/des/general/four_rooms.rs.", "RUST MIGRATION: FourRoomsEnv becomes a struct implementing Environment; FourRoomsSMDPAgent becomes a station struct implementing the SemiMDPAgent trait.", "RUST MIGRATION: FourRoomsOpts, FourRoomsTrainOpts, and FourRoomsResult become serde structs; option/action identifiers should use enums or usize newtypes.", "RUST MIGRATION: buildFourRoomsOptions and hallway helpers are free builders; runFourRoomsSMDP is a training PureTransform returning Result.", "RUST MIGRATION: HALLWAY_FIRST_ACTION uses Map/Set today; port to HashMap/HashSet or precomputed Vec<Option<Action>> for deterministic grid indexing."],
    &["FourRoomsEnv", "FourRoomsOpts", "FourRoomsResult", "FourRoomsTrainOpts", "buildFourRoomsOptions", "runFourRoomsSMDP"],
);
