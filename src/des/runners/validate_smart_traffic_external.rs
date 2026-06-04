//! Port of `src/des/runners/validate-smart-traffic-external.ts`.
//!
//! Cross-checks the real smart-traffic DES against an optional SUMO black-box
//! simulator. SUMO and its Python adapter are never vendored; when they are not
//! present this runner records a skip instead of fabricating an external result.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::des::general::network_flow::{
    TrafficLane, TrafficNetwork, TrafficNodeKind, TrafficParams, TrafficSignal, TrafficSignalPhase,
    TrafficSink, TrafficSource,
};
use crate::des::general::smart_traffic_flow::{
    run_smart_traffic_flow, SmartTrafficParams, SmartTrafficResult,
};

const SUMO_MODULE_ID: &str = "traffic-sumo-reference";
const SUMO_SCRIPT: &str = "external-references/traffic/sumo_traffic_reference.py";

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TrafficDemandRow {
    id: String,
    source_id: String,
    sink_id: String,
    route: Vec<String>,
    vehicles: i64,
    begin_sec: f64,
    end_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProblemJson {
    network: NetworkJson,
    params: ParamsJson,
    demand: Vec<TrafficDemandRow>,
    summary: ProblemSummaryJson,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProblemSummaryJson {
    generated_by: &'static str,
    demand_rows: usize,
    total_vehicles: i64,
}

#[derive(Clone, Debug, Serialize)]
struct NetworkJson {
    nodes: Vec<NodeJson>,
    lanes: Vec<LaneJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    signals: Option<Vec<SignalJson>>,
    sources: Vec<SourceJson>,
    sinks: Vec<SinkJson>,
}

#[derive(Clone, Debug, Serialize)]
struct NodeJson {
    id: String,
    kind: &'static str,
    x: f64,
    y: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LaneJson {
    id: String,
    from: String,
    to: String,
    length_m: f64,
    speed_limit_mps: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    capacity: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SignalJson {
    node_id: String,
    phases: Vec<SignalPhaseJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    offset_sec: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SignalPhaseJson {
    name: String,
    green_lanes: Vec<String>,
    duration_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SourceJson {
    id: String,
    node_id: String,
    rate_per_min: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    destination_sink_ids: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SinkJson {
    id: String,
    node_id: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ParamsJson {
    duration_sec: f64,
    dt_sec: f64,
    seed: f64,
    max_cars: usize,
    actor_shuffle_seed: Option<f64>,
    smart_car_pool_size: Option<usize>,
    spawn_rate_multiplier: Option<f64>,
    car_length_m: Option<f64>,
    car_width_m: Option<f64>,
    lane_width_m: Option<f64>,
    min_gap_m: Option<f64>,
    max_accel_mps2: Option<f64>,
    max_decel_mps2: Option<f64>,
    max_jerk_mps3: Option<f64>,
    reaction_time_sec: Option<f64>,
    time_headway_sec: Option<f64>,
    grid_cell_size_m: Option<f64>,
    accident_risk_scale: Option<f64>,
    accident_probability: Option<f64>,
    distance_preference_spread: Option<f64>,
    start_preference_spread: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExternalTrafficResult {
    #[serde(alias = "generated_demand")]
    generated_demand: f64,
    departed: f64,
    arrived: f64,
    #[serde(alias = "active_at_end")]
    active_at_end: f64,
    #[serde(alias = "mean_travel_time_sec")]
    mean_travel_time_sec: f64,
    #[serde(alias = "mean_speed_mps")]
    mean_speed_mps: f64,
    #[serde(default, alias = "mean_waiting_time_sec")]
    mean_waiting_time_sec: f64,
    #[serde(default, alias = "collision_count")]
    collision_count: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExternalTrafficPayload {
    status: String,
    message: Option<String>,
    result: Option<ExternalTrafficResult>,
}

#[derive(Clone, Debug)]
struct ExternalRun {
    status: Option<i32>,
    stdout: String,
    stderr: String,
}

#[derive(Clone, Debug)]
struct CheckRow {
    name: String,
    passed: bool,
    detail: Option<String>,
}

struct Driver {
    checks: Vec<CheckRow>,
    skipped: usize,
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

    fn skip(&mut self, name: &str, detail: Option<String>) {
        let tail = detail
            .as_ref()
            .map(|d| format!(" - {}", d))
            .unwrap_or_default();
        println!("  SKIP  {}{}", name, tail);
        self.skipped += 1;
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
        self.check(
            "SUMO generated at least one vehicle",
            external.generated_demand > 0.0,
            Some(format!("generated={}", external.generated_demand)),
        );
        self.relative_close(
            "SUMO departures align with DES entered count",
            external.departed,
            internal.entered as f64,
            0.15,
        );
        let internal_exit_rate = if internal.entered > 0 {
            internal.exited as f64 / internal.entered as f64
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
            "SUMO mean speed is finite",
            external.mean_speed_mps.is_finite() && external.mean_speed_mps >= 0.0,
            Some(format!("meanSpeed={}", external.mean_speed_mps)),
        );
        self.check(
            "SUMO active-at-end is finite",
            external.active_at_end.is_finite() && external.active_at_end >= 0.0,
            Some(format!("active={}", external.active_at_end)),
        );
        self.check(
            "SUMO waiting time is reported",
            external.mean_waiting_time_sec.is_finite() && external.mean_waiting_time_sec >= 0.0,
            Some(format!("waiting={}", external.mean_waiting_time_sec)),
        );
        self.check(
            "SUMO collision count is reported",
            external.collision_count >= 0.0,
            Some(format!("collisions={}", external.collision_count)),
        );
    }
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
    let mut pending: HashSet<String> = nodes;
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
    if total_weight <= 0.0 || rows.is_empty() || total_vehicles <= 0 {
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

fn build_params() -> SmartTrafficParams {
    SmartTrafficParams {
        base: TrafficParams {
            builtin: Some("five-intersection".to_string()),
            network: None,
            duration_sec: 180.0,
            dt_sec: 0.2,
            seed: 19.0,
            max_cars: 250,
            car_length_m: None,
            car_width_m: None,
            lane_width_m: None,
            min_gap_m: None,
            max_accel_mps2: None,
            max_decel_mps2: None,
            max_jerk_mps3: Some(4.0),
            reaction_time_sec: Some(0.8),
            time_headway_sec: Some(1.1),
            grid_cell_size_m: Some(0.3048),
            grid_look_ahead_m: None,
            spawn_rate_multiplier: Some(3.0),
            scheduled_trips: None,
        },
        smart_car_pool_size: Some(400),
        actor_shuffle_seed: Some(2026.0),
        accident_risk_scale: Some(0.0),
        accident_probability: Some(0.0),
        accident_accel_boost_mps2: None,
        accident_fault_duration_sec: None,
        distance_preference_spread: Some(0.54),
        start_preference_spread: Some(0.65),
        accident_flash_seconds: None,
    }
}

fn node_kind(kind: TrafficNodeKind) -> &'static str {
    match kind {
        TrafficNodeKind::Source => "source",
        TrafficNodeKind::Intersection => "intersection",
        TrafficNodeKind::Sink => "sink",
    }
}

fn signal_phase_json(p: &TrafficSignalPhase) -> SignalPhaseJson {
    SignalPhaseJson {
        name: p.name.clone(),
        green_lanes: p.green_lanes.clone(),
        duration_sec: p.duration_sec,
    }
}

fn signal_json(s: &TrafficSignal) -> SignalJson {
    SignalJson {
        node_id: s.node_id.clone(),
        phases: s.phases.iter().map(signal_phase_json).collect(),
        offset_sec: s.offset_sec,
    }
}

fn network_json(network: &TrafficNetwork) -> NetworkJson {
    NetworkJson {
        nodes: network
            .nodes
            .iter()
            .map(|n| NodeJson {
                id: n.id.clone(),
                kind: node_kind(n.kind),
                x: n.x,
                y: n.y,
            })
            .collect(),
        lanes: network
            .lanes
            .iter()
            .map(|l| LaneJson {
                id: l.id.clone(),
                from: l.from.clone(),
                to: l.to.clone(),
                length_m: l.length_m,
                speed_limit_mps: l.speed_limit_mps,
                capacity: l.capacity,
            })
            .collect(),
        signals: network
            .signals
            .as_ref()
            .map(|signals| signals.iter().map(signal_json).collect()),
        sources: network
            .sources
            .iter()
            .map(|s| SourceJson {
                id: s.id.clone(),
                node_id: s.node_id.clone(),
                rate_per_min: s.rate_per_min,
                destination_sink_ids: s.destination_sink_ids.clone(),
            })
            .collect(),
        sinks: network
            .sinks
            .iter()
            .map(|s| SinkJson {
                id: s.id.clone(),
                node_id: s.node_id.clone(),
            })
            .collect(),
    }
}

fn params_json(params: &SmartTrafficParams) -> ParamsJson {
    ParamsJson {
        duration_sec: params.base.duration_sec,
        dt_sec: params.base.dt_sec,
        seed: params.base.seed,
        max_cars: params.base.max_cars,
        actor_shuffle_seed: params.actor_shuffle_seed,
        smart_car_pool_size: params.smart_car_pool_size,
        spawn_rate_multiplier: params.base.spawn_rate_multiplier,
        car_length_m: params.base.car_length_m,
        car_width_m: params.base.car_width_m,
        lane_width_m: params.base.lane_width_m,
        min_gap_m: params.base.min_gap_m,
        max_accel_mps2: params.base.max_accel_mps2,
        max_decel_mps2: params.base.max_decel_mps2,
        max_jerk_mps3: params.base.max_jerk_mps3,
        reaction_time_sec: params.base.reaction_time_sec,
        time_headway_sec: params.base.time_headway_sec,
        grid_cell_size_m: params.base.grid_cell_size_m,
        accident_risk_scale: params.accident_risk_scale,
        accident_probability: params.accident_probability,
        distance_preference_spread: params.distance_preference_spread,
        start_preference_spread: params.start_preference_spread,
    }
}

fn write_problem(
    path: &Path,
    internal: &SmartTrafficResult,
    demand: Vec<TrafficDemandRow>,
) -> Result<(), String> {
    let total_vehicles = demand.iter().map(|d| d.vehicles).sum();
    let problem = ProblemJson {
        network: network_json(&internal.network),
        params: params_json(&internal.params),
        summary: ProblemSummaryJson {
            generated_by: "validate_smart_traffic_external",
            demand_rows: demand.len(),
            total_vehicles,
        },
        demand,
    };
    let parent = path
        .parent()
        .ok_or_else(|| format!("missing parent directory for {}", path.display()))?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let text = serde_json::to_string_pretty(&problem).map_err(|e| e.to_string())?;
    fs::write(path, format!("{text}\n")).map_err(|e| e.to_string())
}

fn run_sumo_adapter(
    root: &Path,
    problem_path: &Path,
    out_path: &Path,
) -> Result<ExternalRun, String> {
    let script = root.join(SUMO_SCRIPT);
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let output = Command::new(&python)
        .arg(&script)
        .arg("--problem")
        .arg(problem_path)
        .arg("--out")
        .arg(out_path)
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run {python}: {e}"))?;
    Ok(ExternalRun {
        status: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn read_external_payload(path: &Path) -> Result<ExternalTrafficPayload, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

fn root_from_env() -> PathBuf {
    std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

/// `validate-smart-traffic-external.ts` `main`.
pub fn run() {
    let root = root_from_env();
    let out_dir = root.join("out").join("external").join("traffic");
    let mut d = Driver {
        checks: Vec::new(),
        skipped: 0,
    };

    println!("Smart traffic DES: optional SUMO external simulator cross-check");
    println!("================================================================");

    let internal = run_smart_traffic_flow(build_params(), None);
    d.check(
        "internal smart traffic validators pass",
        internal.validation.iter().all(|c| c.passed),
        Some(format!(
            "failed={}",
            internal.validation.iter().filter(|c| !c.passed).count()
        )),
    );
    d.check(
        "external cross-check uses no-accident baseline",
        internal.params.accident_risk_scale.unwrap_or(0.0) == 0.0
            && internal.params.accident_probability.unwrap_or(0.0) == 0.0
            && internal.crashed == 0,
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

    let demand = build_demand(&internal.network, &internal.params, internal.entered as i64);
    let demand_total: i64 = demand.iter().map(|r| r.vehicles).sum();
    d.check(
        "normalized external demand is non-empty",
        !demand.is_empty(),
        Some(format!("rows={} vehicles={}", demand.len(), demand_total)),
    );
    d.check(
        "normalized external demand preserves DES entered count",
        demand_total == internal.entered as i64,
        Some(format!(
            "demandVehicles={} entered={}",
            demand_total, internal.entered
        )),
    );
    d.check(
        "normalized external demand routes are valid",
        demand.iter().all(|row| !row.route.is_empty()),
        Some(format!(
            "emptyRoutes={}",
            demand.iter().filter(|row| row.route.is_empty()).count()
        )),
    );

    let problem_path = out_dir.join("smart-traffic-sumo-problem.json");
    let out_path = out_dir.join("smart-traffic-sumo-reference.json");
    match write_problem(&problem_path, &internal, demand) {
        Ok(()) => d.check(
            "writes normalized external traffic problem",
            true,
            Some(problem_path.display().to_string()),
        ),
        Err(err) => d.check(
            "writes normalized external traffic problem",
            false,
            Some(err),
        ),
    }

    let script = root.join(SUMO_SCRIPT);
    if !script.exists() {
        d.skip(
            &format!("{SUMO_MODULE_ID}: optional SUMO adapter unavailable"),
            Some(script.display().to_string()),
        );
    } else {
        match run_sumo_adapter(&root, &problem_path, &out_path) {
            Ok(ext) => {
                d.check(
                    "external SUMO adapter process exits cleanly",
                    ext.status == Some(0),
                    Some(format!("status={}", ext.status.unwrap_or(-1))),
                );
                if !ext.stdout.trim().is_empty() {
                    println!("  external stdout: {}", ext.stdout.trim());
                }
                if !ext.stderr.trim().is_empty() {
                    eprintln!("{}", ext.stderr.trim());
                }
                d.check(
                    "external SUMO adapter writes JSON payload",
                    out_path.exists(),
                    Some(out_path.display().to_string()),
                );
                if ext.status == Some(0) && out_path.exists() {
                    match read_external_payload(&out_path) {
                        Ok(payload) => {
                            d.check(
                                "external payload has known status",
                                ["ok", "unavailable", "error"].contains(&payload.status.as_str()),
                                Some(format!("status={}", payload.status)),
                            );
                            if payload.status == "unavailable" {
                                d.skip(
                                    "SUMO dependency is optional and reported cleanly",
                                    payload.message.clone(),
                                );
                            } else if payload.status == "ok" {
                                match payload.result {
                                    Some(result) => d.compare_sumo(&internal, &result),
                                    None => d.check(
                                        "SUMO ok payload includes result",
                                        false,
                                        Some("payload.result missing".to_string()),
                                    ),
                                }
                            } else {
                                d.check(
                                    "SUMO run completed without adapter error",
                                    false,
                                    Some(
                                        payload
                                            .message
                                            .clone()
                                            .unwrap_or_else(|| "unknown adapter error".to_string()),
                                    ),
                                );
                            }
                        }
                        Err(err) => d.check("parses SUMO adapter payload", false, Some(err)),
                    }
                }
            }
            Err(err) => d.check(
                "external SUMO adapter process exits cleanly",
                false,
                Some(err),
            ),
        }
    }

    println!();
    let passed = d.checks.iter().filter(|c| c.passed).count();
    println!(
        "validate-smart-traffic-external: {}/{} checks passed, {} skipped.",
        passed,
        d.checks.len(),
        d.skipped
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
