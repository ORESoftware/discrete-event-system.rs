//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/temp-control.ts`
//! Rust target: `src/des/general/temp_control.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/temp-control.ts",
    "src/des/general/temp_control.rs",
    &["RUST MIGRATION: Target module `src/des/general/temp_control.rs`.", "RUST MIGRATION: Convert house/outdoor/controller/simulation/tick/result interfaces to `serde` structs; `ControllerSpec`, fuzzy terms, and output levels should become enums.", "RUST MIGRATION: Replace `TempControllerBase` inheritance with a controller trait plus shared embedded state; concrete controllers become structs implementing the trait.", "RUST MIGRATION: Keep physical model and controller helper functions as free functions, or wrap controller steps as `PureTransform` when graph-visible.", "RUST MIGRATION: Inject RNG for outdoor noise, represent BigNumber-free values as `f64`, and return `Result` for invalid time steps, gains, or controller specs."],
    &["BangBangController", "ControllerSpec", "ControllerState", "DEFAULT_HOUSE", "DEFAULT_OUTDOOR", "FuzzyController", "HouseParams", "MdpMpcController", "OutdoorPattern", "PIDController", "RunResult", "SimConfig", "TempControllerBase", "TempObs", "TickRecord", "controllerStep", "fuzzyDeltaController", "houseStep", "makeTempController", "mdpMPCController", "mulberry32", "runTempControl", "trueOutdoorTemp"],
);
