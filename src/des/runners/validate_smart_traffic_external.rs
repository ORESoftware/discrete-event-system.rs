//! Port of `src/des/runners/validate-smart-traffic-external.ts`.
//!
//! Cross-checks the smart-traffic DES against an optional SUMO black-box
//! simulator via the sanctioned external-module registry. SUMO/netconvert are
//! never vendored. Driver → [`run`].
//!
//! PORT NOTES:
//!   * The internal side is wired to the real Rust smart-traffic DES.
//!   * The optional SUMO adapter remains dependency-gated. The validator writes
//!     a real normalized problem payload, but only calls the Python-backed
//!     external-module registry when explicitly enabled; missing scripts, SUMO,
//!     or netconvert are reported as clean skips.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::des::general::network_flow::{
    TrafficLane, TrafficNetwork, TrafficNodeKind, TrafficParams, TrafficSink, TrafficSource,
};
use crate::des::general::smart_traffic_flow::{
    run_smart_traffic_flow, SmartTrafficParams, SmartTrafficResult,
};
use crate::des::observability::logger::{parse_json, JsonValue};

use super::external_modules::{register_built_in_external_modules, TRAFFIC_SUMO_REFERENCE_ID};
use super::external_program::{
    run_external_module, ExternalModuleParams, ExternalProgramResult, ParamValue,
};

const ENABLE_SUMO_REFERENCE_ENV: &str = "VALIDATE_SMART_TRAFFIC_ENABLE_SUMO";

// =============================================================================
// Demand routing (faithful).
// =============================================================================

#[derive(Clone, Debug)]
struct TrafficDemandRow {
    id: String,
    source_id: String,
    sink_id: String,
    route: Vec<String>,
    vehicles: i64,
    begin_sec: f64,
    end_sec: f64,
}

fn demand_weight(source: &TrafficSource, sink_count: usize) -> f64 {
    source.rate_per_min / (sink_count.max(1) as f64)
}

fn shortest_lane_path(network: &TrafficNetwork, start_node: &str, end_node: &str) -> Vec<String> {
    let mut outgoing: HashMap<String, Vec<TrafficLane>> = HashMap::new();
    for lane in &network.lanes {
        outgoing
            .entry(lane.from.clone())
            .or_default()
            .push(lane.clone());
    }
    let nodes: HashSet<String> = network.nodes.iter().map(|n| n.id.clone()).collect();
    let mut dist: HashMap<String, f64> = HashMap::new();
    let mut prev_node: HashMap<String, String> = HashMap::new();
    let mut prev_lane: HashMap<String, String> = HashMap::new();
    for node in &nodes {
        dist.insert(node.clone(), f64::INFINITY);
    }
    dist.insert(start_node.to_string(), 0.0);
    let mut pending: HashSet<String> = nodes.clone();
    while !pending.is_empty() {
        let mut current: Option<String> = None;
        let mut best = f64::INFINITY;
        for node in &pending {
            let d = *dist.get(node).unwrap_or(&f64::INFINITY);
            if d < best {
                best = d;
                current = Some(node.clone());
            }
        }
        let current = match current {
            Some(c) if best.is_finite() => c,
            _ => break,
        };
        pending.remove(&current);
        if current == end_node {
            break;
        }
        if let Some(lanes) = outgoing.get(&current) {
            for lane in lanes {
                let nd = best + lane.length_m;
                if nd < *dist.get(&lane.to).unwrap_or(&f64::INFINITY) {
                    dist.insert(lane.to.clone(), nd);
                    prev_node.insert(lane.to.clone(), current.clone());
                    prev_lane.insert(lane.to.clone(), lane.id.clone());
                }
            }
        }
    }
    if !prev_lane.contains_key(end_node) {
        return Vec::new();
    }
    let mut route: Vec<String> = Vec::new();
    let mut cur = end_node.to_string();
    while cur != start_node {
        let lane_id = match prev_lane.get(&cur) {
            Some(l) => l.clone(),
            None => return Vec::new(),
        };
        let parent = match prev_node.get(&cur) {
            Some(p) => p.clone(),
            None => return Vec::new(),
        };
        route.push(lane_id);
        cur = parent;
    }
    route.reverse();
    route
}

fn build_demand(
    network: &TrafficNetwork,
    params: &SmartTrafficParams,
    total_vehicles: i64,
) -> Vec<TrafficDemandRow> {
    struct Row {
        row: TrafficDemandRow,
        weight: f64,
        fractional: f64,
    }
    let mut rows: Vec<Row> = Vec::new();
    let sink_by_id: HashMap<String, TrafficSink> = network
        .sinks
        .iter()
        .map(|s| (s.id.clone(), s.clone()))
        .collect();
    for source in &network.sources {
        let sink_ids: Vec<String> = source
            .destination_sink_ids
            .clone()
            .unwrap_or_else(|| network.sinks.iter().map(|s| s.id.clone()).collect());
        for sink_id in &sink_ids {
            let sink = match sink_by_id.get(sink_id) {
                Some(s) => s,
                None => continue,
            };
            let route = shortest_lane_path(network, &source.node_id, &sink.node_id);
            if route.is_empty() {
                continue;
            }
            rows.push(Row {
                row: TrafficDemandRow {
                    id: format!("{}-{}", source.id, sink_id),
                    source_id: source.id.clone(),
                    sink_id: sink_id.clone(),
                    route,
                    vehicles: 0,
                    begin_sec: 0.0,
                    end_sec: params.base.duration_sec,
                },
                weight: demand_weight(source, sink_ids.len()),
                fractional: 0.0,
            });
        }
    }
    let total_weight: f64 = rows.iter().map(|r| r.weight).sum();
    if total_weight <= 0.0 || rows.is_empty() {
        return Vec::new();
    }
    let mut assigned: i64 = 0;
    for r in rows.iter_mut() {
        let raw = total_vehicles as f64 * r.weight / total_weight;
        r.row.vehicles = raw.floor() as i64;
        r.fractional = raw - r.row.vehicles as f64;
        assigned += r.row.vehicles;
    }
    rows.sort_by(|a, b| {
        b.fractional
            .partial_cmp(&a.fractional)
            .unwrap()
            .then(a.row.id.cmp(&b.row.id))
    });
    let mut i = 0usize;
    while assigned < total_vehicles {
        let n = rows.len();
        rows[i % n].row.vehicles += 1;
        assigned += 1;
        i += 1;
    }
    rows.sort_by(|a, b| a.row.id.cmp(&b.row.id));
    rows.into_iter().map(|r| r.row).collect()
}

// =============================================================================
// External SUMO payload.
// =============================================================================

#[derive(Clone, Debug, Default)]
struct ExternalTrafficResult {
    generated_demand: f64,
    departed: f64,
    arrived: f64,
    active_at_end: f64,
    mean_travel_time_sec: f64,
    mean_speed_mps: f64,
    mean_waiting_time_sec: f64,
    collision_count: f64,
}

#[derive(Clone, Debug, Default)]
struct ExternalTrafficPayload {
    status: String,
    message: Option<String>,
    result: Option<ExternalTrafficResult>,
}

fn opt_num(v: Option<f64>) -> JsonValue {
    v.map(JsonValue::Number).unwrap_or(JsonValue::Null)
}

fn opt_usize(v: Option<usize>) -> JsonValue {
    v.map(|n| JsonValue::Number(n as f64))
        .unwrap_or(JsonValue::Null)
}

fn opt_string_array(v: Option<&Vec<String>>) -> JsonValue {
    match v {
        Some(items) => {
            JsonValue::Array(items.iter().map(|s| JsonValue::String(s.clone())).collect())
        }
        None => JsonValue::Null,
    }
}

fn node_kind_label(kind: TrafficNodeKind) -> &'static str {
    match kind {
        TrafficNodeKind::Source => "source",
        TrafficNodeKind::Intersection => "intersection",
        TrafficNodeKind::Sink => "sink",
    }
}

fn network_json(network: &TrafficNetwork) -> JsonValue {
    let nodes = network
        .nodes
        .iter()
        .map(|node| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(node.id.clone())),
                (
                    "kind".to_string(),
                    JsonValue::String(node_kind_label(node.kind).to_string()),
                ),
                ("x".to_string(), JsonValue::Number(node.x)),
                ("y".to_string(), JsonValue::Number(node.y)),
            ])
        })
        .collect();
    let lanes = network
        .lanes
        .iter()
        .map(|lane| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(lane.id.clone())),
                ("from".to_string(), JsonValue::String(lane.from.clone())),
                ("to".to_string(), JsonValue::String(lane.to.clone())),
                ("lengthM".to_string(), JsonValue::Number(lane.length_m)),
                (
                    "speedLimitMps".to_string(),
                    JsonValue::Number(lane.speed_limit_mps),
                ),
                ("capacity".to_string(), opt_usize(lane.capacity)),
            ])
        })
        .collect();
    let signals = network
        .signals
        .as_ref()
        .map(|signals| {
            JsonValue::Array(
                signals
                    .iter()
                    .map(|signal| {
                        JsonValue::Object(vec![
                            (
                                "nodeId".to_string(),
                                JsonValue::String(signal.node_id.clone()),
                            ),
                            (
                                "phases".to_string(),
                                JsonValue::Array(
                                    signal
                                        .phases
                                        .iter()
                                        .map(|phase| {
                                            JsonValue::Object(vec![
                                                (
                                                    "name".to_string(),
                                                    JsonValue::String(phase.name.clone()),
                                                ),
                                                (
                                                    "greenLanes".to_string(),
                                                    JsonValue::Array(
                                                        phase
                                                            .green_lanes
                                                            .iter()
                                                            .map(|lane| {
                                                                JsonValue::String(lane.clone())
                                                            })
                                                            .collect(),
                                                    ),
                                                ),
                                                (
                                                    "durationSec".to_string(),
                                                    JsonValue::Number(phase.duration_sec),
                                                ),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                            ("offsetSec".to_string(), opt_num(signal.offset_sec)),
                        ])
                    })
                    .collect(),
            )
        })
        .unwrap_or(JsonValue::Null);
    let sources = network
        .sources
        .iter()
        .map(|source| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(source.id.clone())),
                (
                    "nodeId".to_string(),
                    JsonValue::String(source.node_id.clone()),
                ),
                (
                    "ratePerMin".to_string(),
                    JsonValue::Number(source.rate_per_min),
                ),
                (
                    "destinationSinkIds".to_string(),
                    opt_string_array(source.destination_sink_ids.as_ref()),
                ),
            ])
        })
        .collect();
    let sinks = network
        .sinks
        .iter()
        .map(|sink| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(sink.id.clone())),
                (
                    "nodeId".to_string(),
                    JsonValue::String(sink.node_id.clone()),
                ),
            ])
        })
        .collect();

    JsonValue::Object(vec![
        ("nodes".to_string(), JsonValue::Array(nodes)),
        ("lanes".to_string(), JsonValue::Array(lanes)),
        ("signals".to_string(), signals),
        ("sources".to_string(), JsonValue::Array(sources)),
        ("sinks".to_string(), JsonValue::Array(sinks)),
    ])
}

fn demand_json(demand: &[TrafficDemandRow]) -> JsonValue {
    JsonValue::Array(
        demand
            .iter()
            .map(|row| {
                JsonValue::Object(vec![
                    ("id".to_string(), JsonValue::String(row.id.clone())),
                    (
                        "sourceId".to_string(),
                        JsonValue::String(row.source_id.clone()),
                    ),
                    ("sinkId".to_string(), JsonValue::String(row.sink_id.clone())),
                    (
                        "route".to_string(),
                        JsonValue::Array(
                            row.route
                                .iter()
                                .map(|lane| JsonValue::String(lane.clone()))
                                .collect(),
                        ),
                    ),
                    (
                        "vehicles".to_string(),
                        JsonValue::Number(row.vehicles as f64),
                    ),
                    ("beginSec".to_string(), JsonValue::Number(row.begin_sec)),
                    ("endSec".to_string(), JsonValue::Number(row.end_sec)),
                ])
            })
            .collect(),
    )
}

fn params_json(params: &SmartTrafficParams) -> JsonValue {
    JsonValue::Object(vec![
        (
            "builtin".to_string(),
            params
                .base
                .builtin
                .clone()
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Null),
        ),
        (
            "durationSec".to_string(),
            JsonValue::Number(params.base.duration_sec),
        ),
        ("dtSec".to_string(), JsonValue::Number(params.base.dt_sec)),
        ("seed".to_string(), JsonValue::Number(params.base.seed)),
        (
            "maxCars".to_string(),
            JsonValue::Number(params.base.max_cars as f64),
        ),
        ("carLengthM".to_string(), opt_num(params.base.car_length_m)),
        ("carWidthM".to_string(), opt_num(params.base.car_width_m)),
        ("laneWidthM".to_string(), opt_num(params.base.lane_width_m)),
        ("minGapM".to_string(), opt_num(params.base.min_gap_m)),
        (
            "maxAccelMps2".to_string(),
            opt_num(params.base.max_accel_mps2),
        ),
        (
            "maxDecelMps2".to_string(),
            opt_num(params.base.max_decel_mps2),
        ),
        (
            "maxJerkMps3".to_string(),
            opt_num(params.base.max_jerk_mps3),
        ),
        (
            "reactionTimeSec".to_string(),
            opt_num(params.base.reaction_time_sec),
        ),
        (
            "timeHeadwaySec".to_string(),
            opt_num(params.base.time_headway_sec),
        ),
        (
            "gridCellSizeM".to_string(),
            opt_num(params.base.grid_cell_size_m),
        ),
        (
            "gridLookAheadM".to_string(),
            opt_num(params.base.grid_look_ahead_m),
        ),
        (
            "spawnRateMultiplier".to_string(),
            opt_num(params.base.spawn_rate_multiplier),
        ),
        (
            "smartCarPoolSize".to_string(),
            opt_num(params.smart_car_pool_size.map(|n| n as f64)),
        ),
        (
            "actorShuffleSeed".to_string(),
            opt_num(params.actor_shuffle_seed),
        ),
        (
            "accidentRiskScale".to_string(),
            opt_num(params.accident_risk_scale),
        ),
        (
            "accidentProbability".to_string(),
            opt_num(params.accident_probability),
        ),
        (
            "accidentAccelBoostMps2".to_string(),
            opt_num(params.accident_accel_boost_mps2),
        ),
        (
            "accidentFaultDurationSec".to_string(),
            opt_num(params.accident_fault_duration_sec),
        ),
        (
            "distancePreferenceSpread".to_string(),
            opt_num(params.distance_preference_spread),
        ),
        (
            "startPreferenceSpread".to_string(),
            opt_num(params.start_preference_spread),
        ),
        (
            "accidentFlashSeconds".to_string(),
            opt_num(params.accident_flash_seconds),
        ),
    ])
}

fn problem_json(
    internal: &SmartTrafficResult,
    demand: &[TrafficDemandRow],
    demand_vehicles: i64,
) -> JsonValue {
    JsonValue::Object(vec![
        (
            "model".to_string(),
            JsonValue::String("smart-traffic-sumo".to_string()),
        ),
        ("network".to_string(), network_json(&internal.network)),
        ("demand".to_string(), demand_json(demand)),
        ("params".to_string(), params_json(&internal.params)),
        (
            "internalBaseline".to_string(),
            JsonValue::Object(vec![
                (
                    "entered".to_string(),
                    JsonValue::Number(internal.entered as f64),
                ),
                (
                    "exited".to_string(),
                    JsonValue::Number(internal.exited as f64),
                ),
                (
                    "demandVehicles".to_string(),
                    JsonValue::Number(demand_vehicles as f64),
                ),
                (
                    "meanTravelTimeSec".to_string(),
                    JsonValue::Number(internal.mean_travel_time_sec),
                ),
                (
                    "meanSpeedMps".to_string(),
                    JsonValue::Number(internal.mean_speed_mps),
                ),
            ]),
        ),
    ])
}

fn run_external_module_safe(id: &str, params: &ExternalModuleParams) -> ExternalProgramResult {
    match run_external_module(id, params) {
        Ok(result) => result,
        Err(error) => ExternalProgramResult {
            command: String::new(),
            args: Vec::new(),
            status: None,
            stdout: String::new(),
            stderr: error,
            module_id: Some(id.to_string()),
        },
    }
}

fn status_str(status: Option<i32>) -> String {
    status
        .map(|code| code.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn sumo_reference_enabled() -> bool {
    std::env::var(ENABLE_SUMO_REFERENCE_ENV)
        .map(|value| sumo_reference_flag_value_enabled(&value))
        .unwrap_or(false)
}

fn sumo_reference_flag_value_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "y" | "on"
    )
}

fn slice_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

fn optional_external_unavailable(ext: &ExternalProgramResult) -> Option<String> {
    let stderr = ext.stderr.trim();
    let stdout = ext.stdout.trim();
    let message = if stderr.is_empty() { stdout } else { stderr };
    let lower = message.to_ascii_lowercase();
    let unavailable = lower.contains("unknown external module")
        || lower.contains("not registered")
        || lower.contains("external script not found")
        || lower.contains("no such file")
        || lower.contains("no module named")
        || lower.contains("modulenotfounderror")
        || lower.contains("not installed")
        || lower.contains("sumo not found")
        || lower.contains("netconvert not found")
        || lower.contains("unavailable");

    if unavailable {
        Some(if message.is_empty() {
            "optional external dependency unavailable".to_string()
        } else {
            slice_chars(message, 500)
        })
    } else {
        None
    }
}

fn number_any(value: &JsonValue, names: &[&str]) -> Option<f64> {
    names
        .iter()
        .find_map(|name| value.get(name).and_then(|v| v.as_f64()))
}

fn parse_external_result(value: &JsonValue) -> ExternalTrafficResult {
    ExternalTrafficResult {
        generated_demand: number_any(value, &["generatedDemand", "generated_demand"])
            .unwrap_or(f64::NAN),
        departed: number_any(value, &["departed"]).unwrap_or(f64::NAN),
        arrived: number_any(value, &["arrived"]).unwrap_or(f64::NAN),
        active_at_end: number_any(value, &["activeAtEnd", "active_at_end"]).unwrap_or(0.0),
        mean_travel_time_sec: number_any(value, &["meanTravelTimeSec", "mean_travel_time_sec"])
            .unwrap_or(f64::NAN),
        mean_speed_mps: number_any(value, &["meanSpeedMps", "mean_speed_mps"]).unwrap_or(0.0),
        mean_waiting_time_sec: number_any(value, &["meanWaitingTimeSec", "mean_waiting_time_sec"])
            .unwrap_or(0.0),
        collision_count: number_any(value, &["collisionCount", "collision_count"])
            .unwrap_or(f64::NAN),
    }
}

fn load_external_payload(path: &PathBuf) -> Result<ExternalTrafficPayload, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let root = parse_json(&text)?;
    let status = root
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("ok")
        .to_string();
    let message = root
        .get("message")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let result = if status == "ok" {
        root.get("result").map(parse_external_result)
    } else {
        None
    };
    Ok(ExternalTrafficPayload {
        status,
        message,
        result,
    })
}

// =============================================================================
// Driver.
// =============================================================================

struct CheckRow {
    name: String,
    passed: bool,
    detail: Option<String>,
}

struct Driver {
    checks: Vec<CheckRow>,
}

impl Driver {
    fn check(&mut self, name: &str, passed: bool, detail: Option<String>) {
        let tail = detail
            .as_ref()
            .map(|d| format!(" - {}", d))
            .unwrap_or_default();
        println!(
            "  {}  {}{}",
            if passed { "PASS" } else { "FAIL" },
            name,
            tail
        );
        self.checks.push(CheckRow {
            name: name.to_string(),
            passed,
            detail,
        });
    }

    fn relative_close(&mut self, name: &str, actual: f64, expected: f64, tolerance: f64) {
        let diff = (actual - expected).abs();
        let rel = diff / actual.abs().max(expected.abs()).max(1.0);
        self.check(
            name,
            rel <= tolerance,
            Some(format!(
                "actual={:.4} expected={:.4} rel={:.3} tol={}",
                actual, expected, rel, tolerance
            )),
        );
    }

    fn compare_sumo(&mut self, internal: &SmartTrafficResult, external: &ExternalTrafficResult) {
        let internal_entered = internal.entered as f64;
        let internal_exited = internal.exited as f64;
        self.check(
            "SUMO generated at least one vehicle",
            external.generated_demand > 0.0,
            Some(format!("generated={}", external.generated_demand)),
        );
        self.relative_close(
            "SUMO departures align with DES entered count",
            external.departed,
            internal_entered,
            0.15,
        );
        let internal_exit_rate = if internal_entered > 0.0 {
            internal_exited / internal_entered
        } else {
            0.0
        };
        let external_exit_rate = if external.departed > 0.0 {
            external.arrived / external.departed
        } else {
            0.0
        };
        self.check(
            "SUMO and DES completion rates are in the same broad band",
            (internal_exit_rate - external_exit_rate).abs() <= 0.4,
            Some(format!(
                "internal={:.3} external={:.3}",
                internal_exit_rate, external_exit_rate
            )),
        );
        self.check(
            "SUMO mean travel time is finite",
            external.mean_travel_time_sec.is_finite() && external.mean_travel_time_sec >= 0.0,
            Some(format!("mean={}", external.mean_travel_time_sec)),
        );
        if internal.mean_travel_time_sec > 0.0 && external.mean_travel_time_sec > 0.0 {
            let ratio = internal.mean_travel_time_sec / external.mean_travel_time_sec;
            self.check(
                "SUMO and DES mean travel times are comparable order-of-magnitude",
                (0.2..=5.0).contains(&ratio),
                Some(format!(
                    "internal={:.3} external={:.3} ratio={:.3}",
                    internal.mean_travel_time_sec, external.mean_travel_time_sec, ratio
                )),
            );
        }
        self.check(
            "SUMO collision count is reported",
            external.collision_count >= 0.0,
            Some(format!("collisions={}", external.collision_count)),
        );
    }
}

/// `validate-smart-traffic-external.ts` `main`.
pub fn run() {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let out_dir = root.join("out").join("external").join("traffic");
    let mut d = Driver { checks: Vec::new() };

    println!("Smart traffic DES: optional SUMO external simulator cross-check");
    println!("================================================================");

    let params = SmartTrafficParams {
        base: TrafficParams {
            builtin: Some("five-intersection".to_string()),
            network: None,
            duration_sec: 60.0,
            dt_sec: 0.5,
            seed: 19.0,
            max_cars: 80,
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
        smart_car_pool_size: None,
        actor_shuffle_seed: None,
        accident_risk_scale: Some(0.0),
        accident_probability: Some(0.0),
        accident_accel_boost_mps2: None,
        accident_fault_duration_sec: None,
        distance_preference_spread: None,
        start_preference_spread: None,
        accident_flash_seconds: None,
    };

    let internal = run_smart_traffic_flow(params, None);
    d.check(
        "internal smart traffic validators pass",
        internal.validation.iter().all(|c| c.passed),
        None,
    );
    let accident_risk_scale = internal.params.accident_risk_scale.unwrap_or(1.0);
    let accident_probability = internal.params.accident_probability.unwrap_or(1.0);
    d.check(
        "external cross-check uses no-accident baseline",
        (accident_risk_scale == 0.0 || accident_probability == 0.0) && internal.crashed == 0,
        Some(format!("crashed={}", internal.crashed)),
    );
    d.check(
        "internal baseline has observable throughput",
        internal.entered > 0 && internal.exited > 0,
        Some(format!(
            "entered={} exited={}",
            internal.entered, internal.exited
        )),
    );

    std::fs::create_dir_all(&out_dir).ok();
    let problem_path = out_dir.join("smart-traffic-sumo-problem.json");
    let out_path = out_dir.join("smart-traffic-sumo-reference.json");
    let demand = build_demand(&internal.network, &internal.params, internal.entered as i64);
    let demand_vehicles: i64 = demand.iter().map(|row| row.vehicles).sum();
    d.check(
        "normalized external traffic demand is routed",
        !demand.is_empty()
            && demand_vehicles == internal.entered as i64
            && demand.iter().all(|row| !row.route.is_empty()),
        Some(format!(
            "rows={} vehicles={} entered={}",
            demand.len(),
            demand_vehicles,
            internal.entered
        )),
    );
    let problem = problem_json(&internal, &demand, demand_vehicles);
    std::fs::write(&problem_path, format!("{}\n", problem.to_string_pretty(2))).ok();
    d.check(
        "writes normalized external traffic problem",
        problem_path.exists(),
        Some(problem_path.display().to_string()),
    );

    let registration = register_built_in_external_modules();
    d.check(
        "built-in external module registry loads",
        registration.is_ok(),
        registration.err(),
    );
    std::fs::write(&out_path, "{\"status\":\"pending\"}\n").ok();
    let mut external_params = ExternalModuleParams::new();
    external_params.insert(
        "problem".to_string(),
        ParamValue::Str(problem_path.display().to_string()),
    );
    external_params.insert(
        "out".to_string(),
        ParamValue::Str(out_path.display().to_string()),
    );
    external_params.insert(
        "collisionAction".to_string(),
        ParamValue::Str("warn".to_string()),
    );
    if !sumo_reference_enabled() {
        d.check(
            "external SUMO adapter disabled by default",
            true,
            Some(format!(
                "set {ENABLE_SUMO_REFERENCE_ENV}=1 to run the Python-backed SUMO adapter"
            )),
        );
    } else {
        let ext = run_external_module_safe(TRAFFIC_SUMO_REFERENCE_ID, &external_params);
        if !ext.stdout.trim().is_empty() {
            println!("  external stdout: {}", ext.stdout.trim());
        }
        if !ext.stderr.trim().is_empty() {
            eprintln!("{}", ext.stderr.trim());
        }

        if ext.status != Some(0) {
            if let Some(message) = optional_external_unavailable(&ext) {
                d.check(
                    "SUMO dependency is optional and reported cleanly",
                    true,
                    Some(message),
                );
            } else {
                d.check(
                    "external SUMO adapter process exits cleanly",
                    false,
                    Some(format!("status={}", status_str(ext.status))),
                );
            }
        } else {
            d.check(
                "external SUMO adapter process exits cleanly",
                true,
                Some(format!("status={}", status_str(ext.status))),
            );
            d.check(
                "external SUMO adapter writes JSON payload",
                out_path.exists(),
                Some(out_path.display().to_string()),
            );
            match load_external_payload(&out_path) {
                Ok(payload) => {
                    d.check(
                        "external payload has known status",
                        ["ok", "unavailable", "error"].contains(&payload.status.as_str()),
                        Some(format!("status={}", payload.status)),
                    );
                    if payload.status == "unavailable" {
                        d.check(
                            "SUMO dependency is optional and reported cleanly",
                            true,
                            payload.message.clone(),
                        );
                    } else if payload.status == "ok" && payload.result.is_some() {
                        let result = payload.result.clone().unwrap();
                        d.compare_sumo(&internal, &result);
                    } else {
                        d.check(
                            "SUMO run completed without adapter error",
                            false,
                            Some(
                                payload.message.clone().unwrap_or_else(|| {
                                    "unknown external adapter error".to_string()
                                }),
                            ),
                        );
                    }
                }
                Err(error) => d.check("external payload parses as JSON", false, Some(error)),
            }
        }
    }

    println!();
    let passed = d.checks.iter().filter(|c| c.passed).count();
    println!(
        "validate-smart-traffic-external: {}/{} checks passed.",
        passed,
        d.checks.len()
    );
    if passed < d.checks.len() {
        println!("FAILED:");
        for c in &d.checks {
            if !c.passed {
                println!(
                    "  - {}{}",
                    c.name,
                    c.detail
                        .as_ref()
                        .map(|x| format!(": {}", x))
                        .unwrap_or_default()
                );
            }
        }
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sumo_reference_flag_requires_explicit_enable_value() {
        for value in ["1", "true", "TRUE", " yes ", "y", "on"] {
            assert!(
                sumo_reference_flag_value_enabled(value),
                "{value:?} should enable the SUMO external reference"
            );
        }

        for value in ["", "0", "false", "no", "off", "python", "auto", "fallback"] {
            assert!(
                !sumo_reference_flag_value_enabled(value),
                "{value:?} should keep the Python-backed SUMO adapter disabled"
            );
        }
    }
}
