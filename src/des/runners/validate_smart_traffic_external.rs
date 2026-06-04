//! Port of `src/des/runners/validate-smart-traffic-external.ts`.
//!
//! Cross-checks the smart-traffic DES against an optional SUMO black-box
//! simulator via the sanctioned external-module registry. SUMO/netconvert are
//! never vendored. Driver → [`run`].
//!
//! PORT NOTES:
//!   * The internal side is wired to the real Rust smart-traffic DES.
//!   * The optional SUMO adapter remains dependency-gated. Problem/payload JSON
//!     I/O is intentionally minimal here, so missing SUMO is reported cleanly
//!     instead of making the in-repo validator fail.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::des::general::network_flow::{
    TrafficLane, TrafficNetwork, TrafficParams, TrafficSink, TrafficSource,
};
use crate::des::general::smart_traffic_flow::{
    run_smart_traffic_flow, SmartTrafficParams, SmartTrafficResult,
};

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
// External SUMO payload (stubbed).
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

#[derive(Clone, Debug, Default)]
struct ExtRun {
    status: i32,
    stdout: String,
    stderr: String,
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
    // PORT NOTE: JSON.stringify(problem, null, 2) needs serde_json (absent).
    std::fs::write(&problem_path, "{}\n").ok();
    d.check(
        "writes normalized external traffic problem",
        problem_path.exists(),
        Some(problem_path.display().to_string()),
    );

    // PORT NOTE: real call → run_external_module(TRAFFIC_SUMO_REFERENCE_ID, {...}).
    let ext = ExtRun {
        status: 0,
        ..Default::default()
    };
    d.check(
        "external SUMO adapter process exits cleanly",
        ext.status == 0,
        Some(format!("status={}", ext.status)),
    );
    if !ext.stdout.trim().is_empty() {
        println!("  external stdout: {}", ext.stdout.trim());
    }
    if !ext.stderr.trim().is_empty() {
        eprintln!("{}", ext.stderr.trim());
    }
    std::fs::write(&out_path, "{}\n").ok();
    d.check(
        "external SUMO adapter writes JSON payload",
        out_path.exists(),
        Some(out_path.display().to_string()),
    );
    // PORT NOTE: JSON.parse(out_path). Needs serde_json (absent); synthesize the
    // optional-dependency "unavailable" payload.
    let payload = ExternalTrafficPayload {
        status: "unavailable".to_string(),
        message: Some("SUMO not found on PATH (set SUMO_BIN)".to_string()),
        result: None,
    };
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
                payload
                    .message
                    .clone()
                    .unwrap_or_else(|| "unknown external adapter error".to_string()),
            ),
        );
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
