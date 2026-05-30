//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/traffic-flow.ts`
//! Rust target: `src/des/general/traffic_flow.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/traffic-flow.ts",
    "src/des/general/traffic_flow.rs",
    &["RUST MIGRATION: Target module `src/des/general/traffic_flow.rs`.", "RUST MIGRATION: Convert traffic node/link/source/problem/snapshot/stats/time-sample/result interfaces to `serde` structs; `SignalAxis` becomes an enum.", "RUST MIGRATION: Port `TrafficCar`, `IntersectionStation`, `RoadLinkStation`, and `TrafficGridStation` as structs implementing moving-entity/token and DES-station traits.", "RUST MIGRATION: Use `HashMap`/`HashSet` for road graph, outgoing adjacency, station lookup, and path validation indexes; preserve stable ordering for traces.", "RUST MIGRATION: Keep validation, simulation, max-flow conversion, and default-problem builders as free functions returning `Result` where validation can fail."],
    &["IntersectionStation", "RoadLinkStation", "SignalAxis", "TrafficCar", "TrafficCarSnapshot", "TrafficGridStation", "TrafficLinkSpec", "TrafficLinkStats", "TrafficNodeSpec", "TrafficProblem", "TrafficSimulationResult", "TrafficSourceSpec", "TrafficTimeSample", "buildDefaultTrafficProblem", "buildTrafficMaxFlowProblem", "runTrafficSimulation", "validateTrafficProblem"],
);
