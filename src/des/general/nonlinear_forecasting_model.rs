//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/nonlinear-forecasting-model.ts`
//! Rust target: `src/des/general/nonlinear_forecasting_model.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/nonlinear-forecasting-model.ts",
    "src/des/general/nonlinear_forecasting_model.rs",
    &["RUST MIGRATION: Target module `src/des/general/nonlinear_forecasting_model.rs`.", "RUST MIGRATION: Convert regime/observation/source unions to enums and all params, normalized params, scenario, trace, equation, and projection shapes to `serde` structs.", "RUST MIGRATION: Port the forecast token/station pipeline as structs implementing `Token` and `DESStation`; keep private feature/equation helpers as module-local free functions.", "RUST MIGRATION: Map MDP/POMDP discovery state to nominal Rust structs instead of structural records; use `HashMap`/`HashSet` for candidate and feature indexes.", "RUST MIGRATION: Return `Result` from validation, linear solves, ridge fitting, and projection paths that can currently throw or produce non-finite numbers."],
    &["FineTuneTraceRow", "ForecastProjectionPoint", "LatentBeliefPoint", "LatentBeliefTrace", "MDPDiscoveryStep", "NonlinearMDPPOMDPForecastParams", "NonlinearMDPPOMDPForecastResult", "TunedEquation", "VariableDiscoveryResult", "runNonlinearMDPPOMDPForecast"],
);
