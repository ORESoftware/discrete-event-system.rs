//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/factory-floor-track3t.ts`
//! Rust target: `src/des/general/factory_floor_track3t.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/factory-floor-track3t.ts",
    "src/des/general/factory_floor_track3t.rs",
    &["RUST MIGRATION: target module src/des/general/factory_floor_track3t.rs.", "RUST MIGRATION: Warehouse station/action/observation/config/metrics/result interfaces become serde structs; WarehouseStationKind becomes an enum.", "RUST MIGRATION: WarehousePallet, WarehouseStation/Source/Sink, WarehouseForklift, WarehouseQMDPSolver, and WarehousePlanner become structs with Token/Station/Planner trait impls.", "RUST MIGRATION: POMDP and belief tables should use Vec matrices where dense and HashMap<String, usize> for dynamic station indexes.", "RUST MIGRATION: simulateWarehouseScenario and runWarehouseComparison are DES-visible simulation transforms; inject rand::Rng and return Result for layout/scenario validation.", "RUST MIGRATION: Keep helper math and summarization as free functions; use serde-friendly Vec traces instead of structural object literals."],
    &["BASELINE_WAREHOUSE_SCENARIO", "StationDefinition", "TRACK3T_ARCHIVE_GROUNDING", "TRACK3T_WAREHOUSE_SCENARIO", "WarehouseAction", "WarehouseComparisonResult", "WarehouseDecisionState", "WarehouseForklift", "WarehouseJobSummary", "WarehouseLayout", "WarehouseMetrics", "WarehouseObservation", "WarehousePOMDPModel", "WarehousePallet", "WarehousePlanner", "WarehouseQMDPSolver", "WarehouseScenarioConfig", "WarehouseScenarioResult", "WarehouseSimulationOptions", "WarehouseSink", "WarehouseSource", "WarehouseStation", "WarehouseStationKind", "WarehouseStepTrace", "beliefByStation", "buildWarehouseFloor", "buildWarehousePOMDP", "defaultWarehouseLayout", "initialWarehouseBelief", "runWarehouseComparison", "simulateWarehouseScenario", "summarizeWarehouseComparison", "travelMinutes"],
);
