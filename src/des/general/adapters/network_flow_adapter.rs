//! Port of `src/des/general/adapters/network-flow-adapter.ts`
//! (module `des::general::adapters::network_flow_adapter`).
//!
//! Registers the `max-flow`, `traffic-flow`, and `smart-traffic-flow` JSON
//! adapters. Each follows the [`DESModelRegistration`] contract.
//!
//! ## Conversion notes
//!
//!   * `MaxFlowAdapterParams = MaxFlowParams & {builtin?; problem?}` flattens to
//!     [`MaxFlowAdapterParams`] with the direct [`MaxFlowParams`] fields made
//!     `Option` (the "direct" code path checks for their presence).
//!   * `normalizeMaxFlowParams`'s `throw` on missing direct fields →
//!     `panic!` (an invariant violation in `run`).
//!
//! PORT NOTE: `registerModel` / the registry is not wired here; each model is
//! exposed via an `adapter_*()` constructor (matching the sibling adapters).
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder` + the
//! SVG `Shape` frame builders) is not ported, so the elaborate `animate`
//! methods — and the layout helpers (`nodeLayout`, `trafficPoint`,
//! `trafficVector`, `trafficPolyline`, `trafficCarColor`, `fmtMetric`) that only
//! the animators used — are omitted; `animate` is a no-op (matching the sibling
//! `computer_network_adapter`).
//!
//! PORT NOTE: the TS `run` wraps the call in `withLogger(runtime, ..)`; the
//! engine's runners accept an `Option<_>` observability logger, but this adapter
//! passes `None` (the sibling convention — observability logger wiring is not
//! threaded through the adapters).

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, validation_line, write_csv_lines};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};

use crate::des::general::network_flow::{
    build_five_intersection_traffic_network, run_max_flow, run_traffic_flow, FlowEdge,
    MaxFlowParams, MaxFlowResult, TrafficNodeKind, TrafficParams, TrafficResult,
};
use crate::des::general::smart_traffic_flow::{
    run_smart_traffic_flow, SmartTrafficParams, SmartTrafficResult,
};

// =============================================================================
// Schema builders.
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default, description: None }
}
fn string_field() -> ParamSchema {
    ParamSchema::String { allowed: None, default: None, description: None }
}
fn str_enum(allowed: &[&str], default: &str) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: Some(default.to_string()),
        description: None,
    }
}
fn arr(items: ParamSchema, min_length: Option<usize>, max_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length, description: None }
}
fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

fn flow_edge_schema() -> ParamSchema {
    obj(
        vec![
            ("from", num(Some(0.0), None, Some(true), None)),
            ("to", num(Some(0.0), None, Some(true), None)),
            ("capacity", num(Some(0.0), None, None, None)),
            ("name", string_field()),
        ],
        vec!["from", "to", "capacity"],
    )
}

fn max_flow_problem_fields() -> Vec<(&'static str, ParamSchema)> {
    vec![
        ("numNodes", num(Some(2.0), None, Some(true), None)),
        ("source", num(Some(0.0), None, Some(true), None)),
        ("sink", num(Some(0.0), None, Some(true), None)),
        ("edges", arr(flow_edge_schema(), Some(1), None)),
        ("maxAugmentations", num(Some(1.0), None, Some(true), None)),
        (
            "nodeCoordinates",
            arr(arr(num(None, None, None, None), Some(2), Some(2)), None, None),
        ),
        ("nodeNames", arr(string_field(), None, None)),
    ]
}

fn max_flow_problem_schema() -> ParamSchema {
    obj(max_flow_problem_fields(), vec!["numNodes", "source", "sink", "edges"])
}

fn max_flow_schema() -> ParamSchema {
    let mut fields = vec![
        ("builtin", str_enum(&["textbook"], "textbook")),
        ("problem", max_flow_problem_schema()),
    ];
    fields.extend(max_flow_problem_fields());
    obj(fields, vec![])
}

fn traffic_fields() -> Vec<(&'static str, ParamSchema)> {
    let traffic_node = obj(
        vec![
            ("id", string_field()),
            ("kind", str_enum(&["source", "intersection", "sink"], "intersection")),
            ("x", num(None, None, None, None)),
            ("y", num(None, None, None, None)),
        ],
        vec!["id", "kind", "x", "y"],
    );
    let traffic_lane = obj(
        vec![
            ("id", string_field()),
            ("from", string_field()),
            ("to", string_field()),
            ("lengthM", num(Some(0.0), None, None, None)),
            ("speedLimitMps", num(Some(0.0), None, None, None)),
            ("capacity", num(Some(1.0), None, Some(true), None)),
        ],
        vec!["id", "from", "to", "lengthM", "speedLimitMps"],
    );
    let signal_phase = obj(
        vec![
            ("name", string_field()),
            ("greenLanes", arr(string_field(), Some(1), None)),
            ("durationSec", num(Some(0.0), None, None, None)),
        ],
        vec!["name", "greenLanes", "durationSec"],
    );
    let traffic_signal = obj(
        vec![
            ("nodeId", string_field()),
            ("phases", arr(signal_phase, Some(1), None)),
            ("offsetSec", num(None, None, None, Some(0.0))),
        ],
        vec!["nodeId", "phases"],
    );
    let traffic_source = obj(
        vec![
            ("id", string_field()),
            ("nodeId", string_field()),
            ("ratePerMin", num(Some(0.0), None, None, None)),
            ("destinationSinkIds", arr(string_field(), Some(1), None)),
        ],
        vec!["id", "nodeId", "ratePerMin"],
    );
    let traffic_sink = obj(
        vec![("id", string_field()), ("nodeId", string_field())],
        vec!["id", "nodeId"],
    );
    let traffic_network = obj(
        vec![
            ("nodes", arr(traffic_node, Some(1), None)),
            ("lanes", arr(traffic_lane, Some(1), None)),
            ("signals", arr(traffic_signal, None, None)),
            ("sources", arr(traffic_source, Some(1), None)),
            ("sinks", arr(traffic_sink, Some(1), None)),
        ],
        vec!["nodes", "lanes", "sources", "sinks"],
    );
    vec![
        ("builtin", str_enum(&["five-intersection"], "five-intersection")),
        ("network", traffic_network),
        ("durationSec", num(Some(1.0), None, None, Some(180.0))),
        ("dtSec", num(Some(0.01), None, None, Some(1.0))),
        ("seed", num(None, None, Some(true), Some(19.0))),
        ("maxCars", num(Some(1.0), Some(299.0), Some(true), Some(250.0))),
        ("carLengthM", num(Some(0.1), None, None, Some(4.8))),
        ("carWidthM", num(Some(0.1), None, None, Some(1.8))),
        ("laneWidthM", num(Some(0.1), None, None, Some(3.7))),
        ("minGapM", num(Some(0.0), None, None, Some(2.5))),
        ("maxAccelMps2", num(Some(0.1), None, None, Some(2.2))),
        ("maxDecelMps2", num(Some(0.1), None, None, Some(4.0))),
        ("maxJerkMps3", num(Some(0.1), None, None, Some(6.0))),
        ("reactionTimeSec", num(Some(0.0), None, None, Some(0.8))),
        ("timeHeadwaySec", num(Some(0.0), None, None, Some(1.1))),
        ("gridCellSizeM", num(Some(0.01), None, None, Some(0.3048))),
        ("gridLookAheadM", num(Some(0.1), None, None, None)),
        ("spawnRateMultiplier", num(Some(0.0), None, None, Some(1.0))),
    ]
}

fn traffic_schema() -> ParamSchema {
    obj(traffic_fields(), vec!["durationSec", "dtSec", "seed", "maxCars"])
}

fn smart_traffic_schema() -> ParamSchema {
    let mut fields = traffic_fields();
    // Override dtSec default (TS spreads then re-sets it).
    if let Some(slot) = fields.iter_mut().find(|(k, _)| *k == "dtSec") {
        slot.1 = num(Some(0.01), None, None, Some(0.1));
    }
    fields.extend(vec![
        ("smartCarPoolSize", num(Some(1.0), Some(10000.0), Some(true), Some(250.0))),
        ("actorShuffleSeed", num(None, None, Some(true), None)),
        ("accidentRiskScale", num(Some(0.0), None, None, Some(0.0))),
        ("accidentProbability", num(Some(0.0), Some(1.0), None, Some(0.0))),
        ("accidentAccelBoostMps2", num(Some(0.0), None, None, Some(10.0))),
        ("accidentFaultDurationSec", num(Some(0.1), None, None, Some(1.0))),
        ("distancePreferenceSpread", num(Some(0.0), Some(1.5), None, Some(0.0))),
        ("startPreferenceSpread", num(Some(0.0), Some(1.5), None, Some(0.0))),
        ("accidentFlashSeconds", num(Some(0.1), None, None, Some(2.0))),
    ]);
    obj(fields, vec!["durationSec", "dtSec", "seed", "maxCars"])
}

// =============================================================================
// 1. max-flow
// =============================================================================

/// `MaxFlowParams & {builtin?: 'textbook'; problem?: MaxFlowParams}`. The direct
/// fields are `Option` because the adapter's "direct" path tests their presence.
#[derive(Clone, Debug, Default)]
pub struct MaxFlowAdapterParams {
    pub builtin: Option<String>,
    pub problem: Option<MaxFlowParams>,
    pub num_nodes: Option<usize>,
    pub source: Option<usize>,
    pub sink: Option<usize>,
    pub edges: Option<Vec<FlowEdge>>,
    pub max_augmentations: Option<usize>,
    pub node_coordinates: Option<Vec<(f64, f64)>>,
    pub node_names: Option<Vec<String>>,
}

fn build_teaching_max_flow_params() -> MaxFlowParams {
    let edge = |from, to, capacity: f64, name: &str| FlowEdge {
        from,
        to,
        capacity,
        name: Some(name.to_string()),
    };
    MaxFlowParams {
        num_nodes: 6,
        source: 0,
        sink: 5,
        node_names: Some(vec!["s", "a", "b", "c", "d", "t"].into_iter().map(String::from).collect()),
        node_coordinates: Some(vec![
            (90.0, 260.0),
            (260.0, 160.0),
            (260.0, 360.0),
            (520.0, 160.0),
            (520.0, 360.0),
            (760.0, 260.0),
        ]),
        edges: vec![
            edge(0, 1, 16.0, "s-a"),
            edge(0, 2, 13.0, "s-b"),
            edge(1, 2, 10.0, "a-b"),
            edge(2, 1, 4.0, "b-a"),
            edge(1, 3, 12.0, "a-c"),
            edge(3, 2, 9.0, "c-b"),
            edge(2, 4, 14.0, "b-d"),
            edge(4, 3, 7.0, "d-c"),
            edge(3, 5, 20.0, "c-t"),
            edge(4, 5, 4.0, "d-t"),
        ],
        max_augmentations: None,
    }
}

fn has_direct_max_flow_params(p: &MaxFlowAdapterParams) -> bool {
    p.num_nodes.is_some()
        || p.source.is_some()
        || p.sink.is_some()
        || p.edges.as_ref().map(|e| !e.is_empty()).unwrap_or(false)
}

fn normalize_max_flow_params(p: MaxFlowAdapterParams) -> MaxFlowParams {
    if let Some(problem) = p.problem {
        return problem;
    }
    if has_direct_max_flow_params(&p) {
        match (p.num_nodes, p.source, p.sink, p.edges) {
            (Some(num_nodes), Some(source), Some(sink), Some(edges)) => MaxFlowParams {
                num_nodes,
                source,
                sink,
                edges,
                max_augmentations: p.max_augmentations,
                node_coordinates: p.node_coordinates,
                node_names: p.node_names,
            },
            _ => panic!("max-flow: direct parameters require numNodes, source, sink, and edges"),
        }
    } else {
        build_teaching_max_flow_params()
    }
}

pub struct MaxFlowAdapter;
pub fn adapter_max_flow() -> MaxFlowAdapter {
    MaxFlowAdapter
}

impl DESModelRegistration<MaxFlowAdapterParams, MaxFlowResult> for MaxFlowAdapter {
    fn id(&self) -> &str {
        "max-flow"
    }
    fn description(&self) -> &str {
        "Maximum s-t flow via augmenting-path DES ticks with min-cut validation."
    }
    fn schema(&self) -> ParamSchema {
        max_flow_schema()
    }
    fn run(&self, params: MaxFlowAdapterParams, _runtime: &DESRuntimeConfig) -> MaxFlowResult {
        run_max_flow(normalize_max_flow_params(params), None)
    }
    fn summarize(&self, r: &MaxFlowResult, _p: &MaxFlowAdapterParams) -> String {
        let cut_edges = r.min_cut.cut_edges.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
        [
            "MAX FLOW".to_string(),
            "------------------------".to_string(),
            format!("  nodes={} edges={}", r.params.num_nodes, r.params.edges.len()),
            format!("  max flow={:.4} augmentations={}", r.max_flow, r.trace.len()),
            format!("  min cut capacity={:.4} cut edges=[{cut_edges}]", r.min_cut.capacity),
            format!("  validation: {}", validation_line(&r.validation)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &MaxFlowResult, csv_path: &str) {
        let mut lines = vec!["edge,from,to,capacity,flow,residual".to_string()];
        for (i, e) in r.edge_flows.iter().enumerate() {
            lines.push(csv_row([
                e.name.clone().unwrap_or_else(|| i.to_string()),
                e.from.to_string(),
                e.to.to_string(),
                e.capacity.to_string(),
                e.flow.to_string(),
                e.residual.to_string(),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(&self, _result: &MaxFlowResult, _params: &MaxFlowAdapterParams, _runtime: &DESRuntimeConfig) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<MaxFlowAdapterParams>> {
        let edge = |from, to, capacity: f64| FlowEdge { from, to, capacity, name: None };
        vec![RegistrationExample {
            name: "six-node teaching network".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "max-flow".to_string(),
                description: None,
                parameters: MaxFlowAdapterParams {
                    num_nodes: Some(6),
                    source: Some(0),
                    sink: Some(5),
                    node_names: Some(
                        vec!["s", "a", "b", "c", "d", "t"].into_iter().map(String::from).collect(),
                    ),
                    node_coordinates: Some(vec![
                        (90.0, 260.0),
                        (260.0, 160.0),
                        (260.0, 360.0),
                        (520.0, 160.0),
                        (520.0, 360.0),
                        (760.0, 260.0),
                    ]),
                    edges: Some(vec![
                        edge(0, 1, 16.0),
                        edge(0, 2, 13.0),
                        edge(1, 2, 10.0),
                        edge(2, 1, 4.0),
                        edge(1, 3, 12.0),
                        edge(3, 2, 9.0),
                        edge(2, 4, 14.0),
                        edge(4, 3, 7.0),
                        edge(3, 5, 20.0),
                        edge(4, 5, 4.0),
                    ]),
                    ..Default::default()
                },
                runtime: Some(DESRuntimeConfig { animate: Some(true), ..Default::default() }),
                metadata: None,
            },
        }]
    }
}

// =============================================================================
// 2. traffic-flow
// =============================================================================

fn count_intersections(network: &crate::des::general::network_flow::TrafficNetwork) -> usize {
    network.nodes.iter().filter(|n| n.kind == TrafficNodeKind::Intersection).count()
}

pub struct TrafficFlowAdapter;
pub fn adapter_traffic_flow() -> TrafficFlowAdapter {
    TrafficFlowAdapter
}

impl DESModelRegistration<TrafficParams, TrafficResult> for TrafficFlowAdapter {
    fn id(&self) -> &str {
        "traffic-flow"
    }
    fn description(&self) -> &str {
        "Continuous-time traffic flow on a stationary grid with moving cars, signals, sources, and sinks."
    }
    fn schema(&self) -> ParamSchema {
        traffic_schema()
    }
    fn run(&self, params: TrafficParams, _runtime: &DESRuntimeConfig) -> TrafficResult {
        run_traffic_flow(params, None)
    }
    fn summarize(&self, r: &TrafficResult, _p: &TrafficParams) -> String {
        [
            "TRAFFIC FLOW".to_string(),
            "------------------------".to_string(),
            format!(
                "  network nodes={} lanes={} intersections={}",
                r.network.nodes.len(),
                r.network.lanes.len(),
                count_intersections(&r.network)
            ),
            format!(
                "  entered={} exited={} active={} dropped={}",
                r.entered, r.exited, r.final_cars.len(), r.dropped
            ),
            format!(
                "  max active cars={} mean speed={:.2} m/s mean travel={:.1} s",
                r.max_active_cars, r.mean_speed_mps, r.mean_travel_time_sec
            ),
            format!(
                "  grid cell={:.4} m active cells={} created stations={}",
                r.cell_stats.cell_size_m, r.cell_stats.active_cells, r.cell_stats.created_cell_stations
            ),
            format!("  validation: {}", validation_line(&r.validation)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &TrafficResult, csv_path: &str) {
        let mut lines = vec![
            "tick,time_sec,active_cars,entered,exited,mean_speed_mps,mean_travel_time_sec,queue_length"
                .to_string(),
        ];
        for t in &r.trace {
            lines.push(csv_row([
                t.tick.to_string(),
                t.time_sec.to_string(),
                t.active_cars.to_string(),
                t.entered.to_string(),
                t.exited.to_string(),
                t.mean_speed_mps.to_string(),
                t.mean_travel_time_sec.to_string(),
                t.queue_length.to_string(),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(&self, _result: &TrafficResult, _params: &TrafficParams, _runtime: &DESRuntimeConfig) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<TrafficParams>> {
        vec![RegistrationExample {
            name: "five intersections under signal control".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "traffic-flow".to_string(),
                description: None,
                parameters: TrafficParams {
                    builtin: Some("five-intersection".to_string()),
                    network: None,
                    duration_sec: 180.0,
                    dt_sec: 1.0,
                    seed: 19.0,
                    max_cars: 250,
                    car_length_m: None,
                    car_width_m: None,
                    lane_width_m: None,
                    min_gap_m: None,
                    max_accel_mps2: None,
                    max_decel_mps2: None,
                    max_jerk_mps3: None,
                    reaction_time_sec: None,
                    time_headway_sec: None,
                    grid_cell_size_m: None,
                    grid_look_ahead_m: None,
                    spawn_rate_multiplier: Some(1.0),
                    scheduled_trips: None,
                },
                runtime: Some(DESRuntimeConfig { animate: Some(true), ..Default::default() }),
                metadata: None,
            },
        }]
    }
}

// =============================================================================
// 3. smart-traffic-flow
// =============================================================================

pub struct SmartTrafficFlowAdapter;
pub fn adapter_smart_traffic_flow() -> SmartTrafficFlowAdapter {
    SmartTrafficFlowAdapter
}

impl DESModelRegistration<SmartTrafficParams, SmartTrafficResult> for SmartTrafficFlowAdapter {
    fn id(&self) -> &str {
        "smart-traffic-flow"
    }
    fn description(&self) -> &str {
        "Traffic flow where each car is a smart movable participant with its own shuffled runTimeStep."
    }
    fn schema(&self) -> ParamSchema {
        smart_traffic_schema()
    }
    fn run(&self, params: SmartTrafficParams, _runtime: &DESRuntimeConfig) -> SmartTrafficResult {
        run_smart_traffic_flow(params, None)
    }
    fn summarize(&self, r: &SmartTrafficResult, _p: &SmartTrafficParams) -> String {
        let accident_scale = r
            .params
            .accident_risk_scale
            .or(r.params.accident_probability)
            .unwrap_or(0.0);
        [
            "SMART TRAFFIC FLOW".to_string(),
            "------------------------".to_string(),
            format!(
                "  network nodes={} lanes={} intersections={}",
                r.network.nodes.len(),
                r.network.lanes.len(),
                count_intersections(&r.network)
            ),
            format!(
                "  participants={} smart movables={} shuffled={}",
                r.execution.participant_count, r.execution.smart_movable_count, r.execution.shuffled_by_runner
            ),
            format!(
                "  entered={} exited={} crashed={} active={} dropped={}",
                r.entered, r.exited, r.crashed, r.final_cars.len(), r.dropped
            ),
            format!(
                "  max active cars={} mean speed={:.2} m/s mean travel={:.1} s",
                r.max_active_cars, r.mean_speed_mps, r.mean_travel_time_sec
            ),
            format!(
                "  accidents={} accident risk scale={:.2}",
                r.accidents.len(), accident_scale
            ),
            format!(
                "  distance preference spread={:.2} start preference spread={:.2}",
                r.params.distance_preference_spread.unwrap_or(0.0),
                r.params.start_preference_spread.unwrap_or(0.0)
            ),
            format!(
                "  smart movable runs={} max per tick={}",
                r.execution.total_smart_movable_runs, r.execution.max_smart_movable_runs_per_tick
            ),
            format!(
                "  grid cell={:.4} m active cells={} created stations={}",
                r.cell_stats.cell_size_m, r.cell_stats.active_cells, r.cell_stats.created_cell_stations
            ),
            format!("  validation: {}", validation_line(&r.validation)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &SmartTrafficResult, csv_path: &str) {
        let mut lines = vec![
            "tick,time_sec,active_cars,scheduled_smart_cars,smart_movable_runs,entered,exited,crashed,accidents_this_tick,mean_speed_mps,mean_travel_time_sec,queue_length"
                .to_string(),
        ];
        for t in &r.trace {
            lines.push(csv_row([
                t.tick.to_string(),
                t.time_sec.to_string(),
                t.active_cars.to_string(),
                t.scheduled_smart_cars.to_string(),
                t.smart_movable_runs.to_string(),
                t.entered.to_string(),
                t.exited.to_string(),
                t.crashed.to_string(),
                t.accidents.len().to_string(),
                t.mean_speed_mps.to_string(),
                t.mean_travel_time_sec.to_string(),
                t.queue_length.to_string(),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(&self, _result: &SmartTrafficResult, _params: &SmartTrafficParams, _runtime: &DESRuntimeConfig) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<SmartTrafficParams>> {
        vec![RegistrationExample {
            name: "smart movable cars on five intersections".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "smart-traffic-flow".to_string(),
                description: None,
                parameters: SmartTrafficParams {
                    base: TrafficParams {
                        builtin: Some("five-intersection".to_string()),
                        network: None,
                        duration_sec: 180.0,
                        dt_sec: 0.1,
                        seed: 19.0,
                        max_cars: 250,
                        car_length_m: None,
                        car_width_m: None,
                        lane_width_m: None,
                        min_gap_m: None,
                        max_accel_mps2: None,
                        max_decel_mps2: None,
                        max_jerk_mps3: None,
                        reaction_time_sec: None,
                        time_headway_sec: None,
                        grid_cell_size_m: None,
                        grid_look_ahead_m: None,
                        spawn_rate_multiplier: Some(3.0),
                        scheduled_trips: None,
                    },
                    smart_car_pool_size: Some(400),
                    actor_shuffle_seed: Some(2026.0),
                    accident_risk_scale: Some(16.0),
                    accident_probability: None,
                    accident_accel_boost_mps2: Some(12.0),
                    accident_fault_duration_sec: Some(1.0),
                    distance_preference_spread: Some(0.54),
                    start_preference_spread: Some(0.65),
                    accident_flash_seconds: Some(2.5),
                },
                runtime: Some(DESRuntimeConfig { animate: Some(true), ..Default::default() }),
                metadata: None,
            },
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_expose_stable_ids() {
        assert_eq!(adapter_max_flow().id(), "max-flow");
        assert_eq!(adapter_traffic_flow().id(), "traffic-flow");
        assert_eq!(adapter_smart_traffic_flow().id(), "smart-traffic-flow");
    }

    #[test]
    fn normalize_defaults_to_teaching_network() {
        let p = normalize_max_flow_params(MaxFlowAdapterParams::default());
        assert_eq!(p.num_nodes, 6);
        assert_eq!(p.source, 0);
        assert_eq!(p.sink, 5);
        assert_eq!(p.edges.len(), 10);
    }

    #[test]
    fn normalize_prefers_explicit_problem() {
        let problem = build_teaching_max_flow_params();
        let p = normalize_max_flow_params(MaxFlowAdapterParams {
            problem: Some(problem.clone()),
            ..Default::default()
        });
        assert_eq!(p.num_nodes, problem.num_nodes);
    }

    #[test]
    #[should_panic(expected = "direct parameters require")]
    fn normalize_partial_direct_panics() {
        let _ = normalize_max_flow_params(MaxFlowAdapterParams {
            num_nodes: Some(4),
            ..Default::default()
        });
    }
}
