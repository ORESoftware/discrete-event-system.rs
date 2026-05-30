//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/smart-traffic-flow.ts`
//! Rust target: `src/des/general/smart_traffic_flow.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/smart-traffic-flow.ts",
    "src/des/general/smart_traffic_flow.rs",
    &["RUST MIGRATION: Target module `src/des/general/smart_traffic_flow.rs`.", "RUST MIGRATION: Convert smart-traffic params, fault-mode unions, snapshots, traces, accidents, stats, and result interfaces to `serde` structs/enums.", "RUST MIGRATION: Port `SmartTrafficCellStation`, `SmartTrafficCar`, and `SmartTrafficWorldStation` as structs implementing DES/smart-movable traits.", "RUST MIGRATION: Use `HashMap`/`HashSet` for spatial indexes, active-car maps, lane occupancy, and route lookups; keep deterministic iteration order where traces depend on it.", "RUST MIGRATION: Inject RNG for driver traits/faults and return `Result` from network/trip validation, routing, and capacity calculations."],
    &["SmartTrafficAccident", "SmartTrafficCar", "SmartTrafficCarSnapshot", "SmartTrafficCellStation", "SmartTrafficCellStats", "SmartTrafficExecutionStats", "SmartTrafficFaultMode", "SmartTrafficParams", "SmartTrafficResult", "SmartTrafficTraceRow", "SmartTrafficWorldStation", "runSmartTrafficFlow"],
);
