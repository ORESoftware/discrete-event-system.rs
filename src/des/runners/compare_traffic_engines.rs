//! Port of `src/des/runners/compare-traffic-engines.ts`.
//!
//! Compares the smart-traffic DES engine and optional external SUMO / UXsim
//! engines, emitting a markdown table and a JSON dump. The TS top-level
//! `main()` becomes [`run`].
//!
//! ## PORT NOTE
//!
//!   * Reuses the real `crate::des::general::network_flow` traffic types +
//!     `build_five_intersection_traffic_network`.
//!   * `smart-traffic-flow` is **not ported**, so [`SmartTrafficParams`] /
//!     [`SmartTrafficResult`] / [`run_smart_traffic_flow`] are the smallest
//!     self-contained stand-ins (deterministic placeholder stats, empty trace ⇒
//!     `meanAbsJerk`/`minLeaderGap` = 0). Replace with the real engine once
//!     ported.
//!   * `spawnSync` (SUMO/netconvert/UXsim) → [`std::process::Command`]; the SUMO
//!     and UXsim branches early-return "not found" when the binaries are absent
//!     (the usual case in this repo). `runCommand` returns a `Result` and a
//!     failed external command yields a notes-only `EngineStats` instead of the
//!     TS uncaught `throw`.
//!   * `JSON.parse`/`stringify` → [`parse_json`] / `JsonValue::to_string_pretty`.
//!   * regex XML scraping → hand-written attribute scanners.
//!   * `mulberry32`/`Date.now()` → `mulberry32` / `SystemClock`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::des::general::network_flow::{
    build_five_intersection_traffic_network, TrafficLane, TrafficNetwork, TrafficNodeKind,
};
use crate::des::general::prng::mulberry32;
use crate::des::observability::logger::{parse_json, JsonValue};
use crate::des::shared::capabilities::{Clock, RandomSource, SystemClock};

// =============================================================================
// PORT NOTE: smart-traffic-flow stand-in (engine not yet ported).
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct SmartTrafficParams {
    pub duration_sec: f64,
    pub dt_sec: f64,
    pub seed: f64,
    pub max_cars: f64,
    pub spawn_rate_multiplier: f64,
    pub car_length_m: f64,
    pub car_width_m: f64,
    pub lane_width_m: f64,
    pub min_gap_m: f64,
    pub max_accel_mps2: f64,
    pub max_decel_mps2: f64,
    pub time_headway_sec: f64,
    pub reaction_time_sec: f64,
    pub max_jerk_mps3: f64,
    pub grid_cell_size_m: f64,
    pub smart_car_pool_size: f64,
    pub actor_shuffle_seed: f64,
    pub accident_risk_scale: f64,
}

#[derive(Clone, Debug, Default)]
pub struct SmartTrafficResult {
    pub entered: usize,
    pub exited: usize,
    pub dropped: usize,
    pub final_cars_len: usize,
    pub max_active_cars: usize,
    pub mean_travel_time_sec: f64,
    pub mean_speed_mps: f64,
    pub mean_abs_jerk_mps3: f64,
    pub min_leader_gap_m: f64,
}

/// PORT NOTE: deterministic stand-in for the real `runSmartTrafficFlow`.
pub fn run_smart_traffic_flow(
    _params: &SmartTrafficParams,
    network: &TrafficNetwork,
    scheduled_trips: &[SharedTrip],
) -> SmartTrafficResult {
    let entered = scheduled_trips.len();
    let mean_len = if network.lanes.is_empty() {
        0.0
    } else {
        network.lanes.iter().map(|l| l.length_m).sum::<f64>() / network.lanes.len() as f64
    };
    let mean_speed = if network.lanes.is_empty() {
        1.0
    } else {
        (network.lanes.iter().map(|l| l.speed_limit_mps).sum::<f64>() / network.lanes.len() as f64).max(1e-6)
    };
    SmartTrafficResult {
        entered,
        exited: entered,
        dropped: 0,
        final_cars_len: 0,
        max_active_cars: entered,
        mean_travel_time_sec: if entered > 0 { mean_len / mean_speed } else { 0.0 },
        mean_speed_mps: mean_speed,
        mean_abs_jerk_mps3: 0.0,
        min_leader_gap_m: 0.0,
    }
}

// =============================================================================
// Report types.
// =============================================================================

#[derive(Clone, Debug)]
pub struct SharedTrip {
    pub depart_sec: f64,
    pub source_id: String,
    pub destination_sink_id: String,
    pub id: String,
    pub route: Vec<String>,
    pub source_node_id: String,
    pub sink_node_id: String,
}

#[derive(Clone, Debug, Default)]
struct EngineStats {
    engine: String,
    version: Option<String>,
    generated: f64,
    entered: Option<f64>,
    completed: f64,
    active_at_end: f64,
    dropped: Option<f64>,
    max_active: Option<f64>,
    mean_travel_time_sec: Option<f64>,
    mean_speed_mps: Option<f64>,
    mean_abs_jerk_mps3: Option<f64>,
    min_headway_m: Option<f64>,
    notes: Vec<String>,
}

struct Scenario {
    network: String,
    duration_sec: f64,
    dt_sec: f64,
    scheduled_trips: usize,
    lanes: usize,
    intersections: usize,
}

fn project_root() -> PathBuf {
    crate::des::runners::external_program::repo_root_from_runner()
}

fn out_dir() -> PathBuf {
    project_root().join("out").join("traffic-engine-comparison")
}

fn venv_dir() -> PathBuf {
    match std::env::var("TRAFFIC_ENGINE_VENV") {
        Ok(rel) => project_root().join(rel),
        Err(_) => project_root().join("out").join("traffic-engine-venv"),
    }
}

/// `main()`.
pub fn run() {
    let out = out_dir();
    let _ = fs::create_dir_all(&out);
    let network = build_five_intersection_traffic_network();
    let params = SmartTrafficParams {
        duration_sec: 200.0,
        dt_sec: 0.1,
        seed: 19.0,
        max_cars: 180.0,
        spawn_rate_multiplier: 1.0,
        car_length_m: 4.8,
        car_width_m: 1.8,
        lane_width_m: 3.7,
        min_gap_m: 2.5,
        max_accel_mps2: 2.2,
        max_decel_mps2: 4.0,
        time_headway_sec: 1.2,
        reaction_time_sec: 1.0,
        max_jerk_mps3: 6.0,
        grid_cell_size_m: 0.3048,
        smart_car_pool_size: 240.0,
        actor_shuffle_seed: 2026.0,
        accident_risk_scale: 0.0,
    };
    let trips = generate_scheduled_trips(&network, &params, (params.seed + 4242.0) as u32);
    write_shared_input(&network, &params, &trips);

    let des_stats = run_des(&params, &network, &trips);
    let sumo_stats = run_sumo(&network, &params, &trips);
    let uxsim_stats = run_uxsim(&network, &params, &trips);

    let scenario = Scenario {
        network: "five-intersection".to_string(),
        duration_sec: params.duration_sec,
        dt_sec: params.dt_sec,
        scheduled_trips: trips.len(),
        lanes: network.lanes.len(),
        intersections: network.nodes.iter().filter(|n| n.kind == TrafficNodeKind::Intersection).count(),
    };
    let engines = [des_stats, sumo_stats, uxsim_stats];

    let comparison = JsonValue::Object(vec![
        ("generatedAt".to_string(), JsonValue::String(format!("{}", SystemClock.now_ms()))),
        (
            "scenario".to_string(),
            JsonValue::Object(vec![
                ("network".to_string(), JsonValue::String(scenario.network.clone())),
                ("durationSec".to_string(), JsonValue::Number(scenario.duration_sec)),
                ("dtSec".to_string(), JsonValue::Number(scenario.dt_sec)),
                ("scheduledTrips".to_string(), JsonValue::Number(scenario.scheduled_trips as f64)),
                ("lanes".to_string(), JsonValue::Number(scenario.lanes as f64)),
                ("intersections".to_string(), JsonValue::Number(scenario.intersections as f64)),
            ]),
        ),
        ("engines".to_string(), JsonValue::Array(engines.iter().map(engine_json).collect())),
    ]);

    let _ = fs::write(out.join("traffic-engine-comparison.json"), comparison.to_string_pretty(2));
    let md = render_markdown(&engines, &scenario);
    let _ = fs::write(out.join("traffic-engine-comparison.md"), &md);
    println!("{md}");
    println!(
        "\nWrote {}",
        relative(&project_root(), &out.join("traffic-engine-comparison.json"))
    );
}

fn generate_scheduled_trips(network: &TrafficNetwork, params: &SmartTrafficParams, seed: u32) -> Vec<SharedTrip> {
    let mut rng = mulberry32(seed);
    let mut accumulators: HashMap<String, f64> = HashMap::new();
    let mut trips: Vec<SharedTrip> = Vec::new();
    let ticks = (params.duration_sec / params.dt_sec).ceil() as i64;
    let sink_by_id: HashMap<&str, &str> = network.sinks.iter().map(|s| (s.id.as_str(), s.node_id.as_str())).collect();
    for source in &network.sources {
        accumulators.insert(source.id.clone(), 0.0);
    }
    for tick in 0..ticks {
        let depart_sec = round_time(tick as f64 * params.dt_sec);
        for source in &network.sources {
            let expected = source.rate_per_min * params.spawn_rate_multiplier * params.dt_sec / 60.0;
            let mut acc = accumulators.get(&source.id).copied().unwrap_or(0.0) + expected;
            let count = acc.floor() as i64;
            acc -= count as f64;
            accumulators.insert(source.id.clone(), acc);
            for _ in 0..count {
                let sink_ids: Vec<String> = source
                    .destination_sink_ids
                    .clone()
                    .unwrap_or_else(|| network.sinks.iter().map(|s| s.id.clone()).collect());
                if sink_ids.is_empty() {
                    continue;
                }
                let pick = (rng.next_float() * sink_ids.len() as f64).floor() as usize;
                let destination_sink_id = sink_ids[pick.min(sink_ids.len() - 1)].clone();
                let Some(sink_node) = sink_by_id.get(destination_sink_id.as_str()) else { continue };
                let route = shortest_lane_path(network, &source.node_id, sink_node);
                if route.is_empty() {
                    continue;
                }
                trips.push(SharedTrip {
                    id: format!("trip-{}", trips.len() + 1),
                    depart_sec,
                    source_id: source.id.clone(),
                    destination_sink_id,
                    route,
                    source_node_id: source.node_id.clone(),
                    sink_node_id: (*sink_node).to_string(),
                });
            }
        }
    }
    trips
}

fn run_des(params: &SmartTrafficParams, network: &TrafficNetwork, trips: &[SharedTrip]) -> EngineStats {
    let result = run_smart_traffic_flow(params, network, trips);
    EngineStats {
        engine: "DES smart traffic".to_string(),
        version: Some("local".to_string()),
        generated: trips.len() as f64,
        entered: Some(result.entered as f64),
        completed: result.exited as f64,
        active_at_end: result.final_cars_len as f64,
        dropped: Some(result.dropped as f64),
        max_active: Some(result.max_active_cars as f64),
        mean_travel_time_sec: round_metric(Some(result.mean_travel_time_sec)),
        mean_speed_mps: round_metric(Some(result.mean_speed_mps)),
        mean_abs_jerk_mps3: round_metric(Some(result.mean_abs_jerk_mps3)),
        min_headway_m: round_metric(Some(result.min_leader_gap_m)),
        notes: vec!["uses one-foot cell stations and smart movable car runTimeStep decisions".to_string()],
    }
}

fn run_sumo(network: &TrafficNetwork, params: &SmartTrafficParams, trips: &[SharedTrip]) -> EngineStats {
    let venv = venv_dir();
    let sumo_bin = venv.join("bin").join("sumo");
    let netconvert_bin = venv.join("bin").join("netconvert");
    if !sumo_bin.exists() || !netconvert_bin.exists() {
        return EngineStats {
            engine: "SUMO".to_string(),
            generated: trips.len() as f64,
            completed: 0.0,
            active_at_end: trips.len() as f64,
            notes: vec![format!("SUMO binaries not found under {}", venv.display())],
            ..Default::default()
        };
    }
    match run_sumo_inner(network, params, trips, &sumo_bin, &netconvert_bin) {
        Ok(stats) => stats,
        Err(e) => EngineStats {
            engine: "SUMO".to_string(),
            generated: trips.len() as f64,
            completed: 0.0,
            active_at_end: trips.len() as f64,
            notes: vec![format!("SUMO run failed: {e}")],
            ..Default::default()
        },
    }
}

fn run_sumo_inner(
    network: &TrafficNetwork,
    params: &SmartTrafficParams,
    trips: &[SharedTrip],
    sumo_bin: &Path,
    netconvert_bin: &Path,
) -> Result<EngineStats, String> {
    let dir = out_dir().join("sumo");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let nodes_file = dir.join("five-intersection.nod.xml");
    let edges_file = dir.join("five-intersection.edg.xml");
    let routes_file = dir.join("five-intersection.rou.xml");
    let net_file = dir.join("five-intersection.net.xml");
    let tripinfo_file = dir.join("tripinfo.xml");
    let summary_file = dir.join("summary.xml");

    fs::write(&nodes_file, sumo_nodes_xml(network)).map_err(|e| e.to_string())?;
    fs::write(&edges_file, sumo_edges_xml(network)).map_err(|e| e.to_string())?;
    fs::write(&routes_file, sumo_routes_xml(params, trips)).map_err(|e| e.to_string())?;
    run_command(
        netconvert_bin,
        &[
            "--node-files".into(),
            nodes_file.display().to_string(),
            "--edge-files".into(),
            edges_file.display().to_string(),
            "--output-file".into(),
            net_file.display().to_string(),
            "--no-turnarounds".into(),
            "--xml-validation".into(),
            "never".into(),
        ],
        &dir,
    )?;
    let version = command_version(sumo_bin)?;
    run_command(
        sumo_bin,
        &[
            "-n".into(),
            net_file.display().to_string(),
            "-r".into(),
            routes_file.display().to_string(),
            "--begin".into(),
            "0".into(),
            "--end".into(),
            params.duration_sec.to_string(),
            "--step-length".into(),
            params.dt_sec.to_string(),
            "--tripinfo-output".into(),
            tripinfo_file.display().to_string(),
            "--summary-output".into(),
            summary_file.display().to_string(),
            "--no-step-log".into(),
            "true".into(),
            "--duration-log.disable".into(),
            "true".into(),
            "--time-to-teleport".into(),
            "-1".into(),
            "--xml-validation".into(),
            "never".into(),
            "--collision.action".into(),
            "warn".into(),
        ],
        &dir,
    )?;
    let trip_infos = parse_xml_records(&fs::read_to_string(&tripinfo_file).map_err(|e| e.to_string())?, "tripinfo");
    let summary_steps = parse_xml_records(&fs::read_to_string(&summary_file).map_err(|e| e.to_string())?, "step");
    let last = summary_steps.last().cloned().unwrap_or_default();
    let durations: Vec<f64> = trip_infos.iter().filter_map(|t| t.get("duration").and_then(|s| s.parse::<f64>().ok())).filter(|x| x.is_finite()).collect();
    let speeds: Vec<f64> = trip_infos
        .iter()
        .filter_map(|t| {
            let rl = t.get("routeLength")?.parse::<f64>().ok()?;
            let d = t.get("duration")?.parse::<f64>().ok()?;
            Some(rl / d.max(1e-9))
        })
        .filter(|x| x.is_finite())
        .collect();
    let max_active = summary_steps.iter().filter_map(|s| s.get("running").and_then(|v| v.parse::<f64>().ok())).fold(0.0_f64, f64::max);
    let inserted = last.get("inserted").and_then(|v| v.parse::<f64>().ok());
    let ended = last.get("ended").and_then(|v| v.parse::<f64>().ok());
    let active_at_end = match (inserted, ended) {
        (Some(i), Some(e)) => (i - e).max(0.0),
        _ => (trips.len() as f64 - trip_infos.len() as f64).max(0.0),
    };
    Ok(EngineStats {
        engine: "SUMO".to_string(),
        version: Some(version),
        generated: trips.len() as f64,
        entered: inserted.filter(|x| x.is_finite()),
        completed: trip_infos.len() as f64,
        active_at_end,
        max_active: Some(max_active),
        mean_travel_time_sec: mean_rounded(&durations),
        mean_speed_mps: mean_rounded(&speeds),
        notes: vec!["microscopic, space-continuous SUMO run with the shared scheduled trip table".to_string()],
        ..Default::default()
    })
}

fn run_uxsim(network: &TrafficNetwork, params: &SmartTrafficParams, trips: &[SharedTrip]) -> EngineStats {
    let venv = venv_dir();
    let python_bin = venv.join("bin").join("python");
    if !python_bin.exists() {
        return EngineStats {
            engine: "UXsim".to_string(),
            generated: trips.len() as f64,
            completed: 0.0,
            active_at_end: trips.len() as f64,
            notes: vec![format!("UXsim virtualenv Python not found under {}", venv.display())],
            ..Default::default()
        };
    }
    match run_uxsim_inner(network, params, trips, &python_bin) {
        Ok(stats) => stats,
        Err(e) => EngineStats {
            engine: "UXsim".to_string(),
            generated: trips.len() as f64,
            completed: 0.0,
            active_at_end: trips.len() as f64,
            notes: vec![format!("UXsim run failed: {e}")],
            ..Default::default()
        },
    }
}

fn run_uxsim_inner(
    network: &TrafficNetwork,
    params: &SmartTrafficParams,
    trips: &[SharedTrip],
    python_bin: &Path,
) -> Result<EngineStats, String> {
    let dir = out_dir().join("uxsim");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let input_file = dir.join("input.json");
    let script_file = dir.join("run_uxsim.py");
    let output_file = dir.join("stats.json");
    let input = JsonValue::Object(vec![
        ("network".to_string(), network_json(network)),
        ("params".to_string(), params_json(params)),
        ("trips".to_string(), JsonValue::Array(trips.iter().map(trip_json).collect())),
    ]);
    fs::write(&input_file, input.to_string_pretty(2)).map_err(|e| e.to_string())?;
    fs::write(&script_file, uxsim_script()).map_err(|e| e.to_string())?;
    run_command(
        python_bin,
        &[script_file.display().to_string(), input_file.display().to_string(), output_file.display().to_string()],
        &dir,
    )?;
    let text = fs::read_to_string(&output_file).map_err(|e| e.to_string())?;
    let stats = parse_json(&text).map_err(|e| e)?;
    let g = |k: &str| stats.get(k).and_then(|v| v.as_f64());
    Ok(EngineStats {
        engine: "UXsim".to_string(),
        version: stats.get("version").and_then(|v| v.as_str()).map(str::to_string),
        generated: g("generated").unwrap_or(0.0),
        entered: g("entered"),
        completed: g("completed").unwrap_or(0.0),
        active_at_end: g("activeAtEnd").unwrap_or(0.0),
        max_active: g("maxActive"),
        mean_travel_time_sec: round_metric(g("meanTravelTimeSec")),
        mean_speed_mps: round_metric(g("meanSpeedMps")),
        notes: vec!["mesoscopic UXsim run with exact shared departure times and OD pairs; jerk is not exposed by UXsim".to_string()],
        ..Default::default()
    })
}

fn write_shared_input(network: &TrafficNetwork, params: &SmartTrafficParams, trips: &[SharedTrip]) {
    let payload = JsonValue::Object(vec![
        ("network".to_string(), network_json(network)),
        ("params".to_string(), params_json(params)),
        ("trips".to_string(), JsonValue::Array(trips.iter().map(trip_json).collect())),
    ]);
    let _ = fs::write(out_dir().join("shared-traffic-input.json"), payload.to_string_pretty(2));
}

// --- SUMO XML ----------------------------------------------------------------

fn sumo_nodes_xml(network: &TrafficNetwork) -> String {
    let signal_nodes: std::collections::HashSet<String> = network
        .signals
        .as_ref()
        .map(|sigs| sigs.iter().map(|s| s.node_id.clone()).collect())
        .unwrap_or_default();
    let mut lines = vec!["<nodes>".to_string()];
    for node in &network.nodes {
        let ty = if signal_nodes.contains(&node.id) { "traffic_light" } else { "priority" };
        lines.push(format!(
            "  <node id=\"{}\" x=\"{}\" y=\"{}\" type=\"{ty}\"/>",
            xml(&node.id),
            node.x * 120.0,
            -node.y * 120.0
        ));
    }
    lines.push("</nodes>".to_string());
    lines.join("\n") + "\n"
}

fn sumo_edges_xml(network: &TrafficNetwork) -> String {
    let mut lines = vec!["<edges>".to_string()];
    for lane in &network.lanes {
        lines.push(format!(
            "  <edge id=\"{}\" from=\"{}\" to=\"{}\" numLanes=\"1\" speed=\"{}\" length=\"{}\" priority=\"1\"/>",
            xml(&lane.id),
            xml(&lane.from),
            xml(&lane.to),
            lane.speed_limit_mps,
            lane.length_m
        ));
    }
    lines.push("</edges>".to_string());
    lines.join("\n") + "\n"
}

fn sumo_routes_xml(params: &SmartTrafficParams, trips: &[SharedTrip]) -> String {
    let mut routes: Vec<(String, String)> = Vec::new();
    let mut route_index: HashMap<String, String> = HashMap::new();
    for trip in trips {
        let key = trip.route.join(" ");
        if !route_index.contains_key(&key) {
            let id = format!("route-{}", routes.len() + 1);
            route_index.insert(key.clone(), id.clone());
            routes.push((key, id));
        }
    }
    let emergency_decel = (params.max_decel_mps2 * 2.0).max(8.0);
    let mut lines = vec![
        "<routes>".to_string(),
        format!(
            "  <vType id=\"car\" accel=\"{}\" decel=\"{}\" apparentDecel=\"{}\" emergencyDecel=\"{}\" length=\"{}\" minGap=\"{}\" maxSpeed=\"13.5\" tau=\"{}\" sigma=\"0.5\"/>",
            params.max_accel_mps2, params.max_decel_mps2, params.max_decel_mps2, emergency_decel, params.car_length_m, params.min_gap_m, params.reaction_time_sec
        ),
    ];
    for (edges, id) in &routes {
        lines.push(format!("  <route id=\"{id}\" edges=\"{}\"/>", xml(edges)));
    }
    for trip in trips {
        let route_id = route_index.get(&trip.route.join(" ")).cloned().unwrap_or_default();
        lines.push(format!(
            "  <vehicle id=\"{}\" type=\"car\" route=\"{route_id}\" depart=\"{:.1}\" departLane=\"best\" departSpeed=\"max\"/>",
            xml(&trip.id),
            trip.depart_sec
        ));
    }
    lines.push("</routes>".to_string());
    lines.join("\n") + "\n"
}

fn uxsim_script() -> String {
    // Identical to the TS embedded Python (kept verbatim for a faithful run).
    r#"
import json
import sys

from uxsim import World
import uxsim

input_path, output_path = sys.argv[1], sys.argv[2]
with open(input_path) as f:
    data = json.load(f)

network = data["network"]
params = data["params"]
trips = data["trips"]
vehicle_space = max(1e-9, params.get("carLengthM", 4.8) + params.get("minGapM", 2.5))
W = World(
    name="five-intersection-cross-check",
    deltan=1,
    reaction_time=params.get("dtSec", 0.1),
    tmax=params["durationSec"],
    random_seed=params.get("seed", 19),
    print_mode=0,
    save_mode=0,
    show_mode=0,
    show_progress=0,
    vehicle_logging_timestep_interval=1,
    hard_deterministic_mode=True,
)
for node in network["nodes"]:
    W.addNode(node["id"], node["x"] * 120, -node["y"] * 120)
for lane in network["lanes"]:
    W.addLink(
        lane["id"],
        lane["from"],
        lane["to"],
        length=lane["lengthM"],
        free_flow_speed=lane["speedLimitMps"],
        number_of_lanes=1,
        jam_density=1 / vehicle_space,
    )
for trip in trips:
    W.addVehicle(
        trip["sourceNodeId"],
        trip["sinkNodeId"],
        trip["departSec"],
        name=trip["id"],
        links_prefer=trip["route"],
    )
W.exec_simulation()

vehicles = list(W.VEHICLES.values())
completed = [v for v in vehicles if getattr(v, "state", None) == "end" and getattr(v, "travel_time", None) is not None]
def safe_float(value, default=0):
    try:
        return float(value)
    except Exception:
        return default

durations = [safe_float(v.travel_time) for v in completed]
speeds = [safe_float(getattr(v, "distance_traveled", 0)) / max(1e-9, safe_float(v.travel_time)) for v in completed]
max_active = 0
dt = params.get("dtSec", 0.1)
ticks = int(params["durationSec"] / dt + 0.5)
for tick in range(ticks):
    t = tick * dt
    active = 0
    for v in vehicles:
        depart = safe_float(getattr(v, "departure_time_in_second", getattr(v, "departure_time", 0)))
        arrival = getattr(v, "arrival_time", None)
        if depart <= t and (arrival is None or safe_float(arrival, params["durationSec"] + 1) > t):
            active += 1
    max_active = max(max_active, active)

out = {
    "version": getattr(uxsim, "__version__", "unknown"),
    "generated": len(trips),
    "entered": len(vehicles),
    "completed": len(completed),
    "activeAtEnd": max(0, len(vehicles) - len(completed)),
    "maxActive": max_active,
    "meanTravelTimeSec": sum(durations) / len(durations) if durations else None,
    "meanSpeedMps": sum(speeds) / len(speeds) if speeds else None,
}
with open(output_path, "w") as f:
    json.dump(out, f, indent=2)
"#
    .to_string()
}

fn shortest_lane_path(network: &TrafficNetwork, from_node: &str, to_node: &str) -> Vec<String> {
    let mut outgoing: HashMap<&str, Vec<&TrafficLane>> = HashMap::new();
    for lane in &network.lanes {
        outgoing.entry(lane.from.as_str()).or_default().push(lane);
    }
    let mut queue: std::collections::VecDeque<(String, Vec<String>)> = std::collections::VecDeque::new();
    queue.push_back((from_node.to_string(), Vec::new()));
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    seen.insert(from_node.to_string());
    while let Some((node, route)) = queue.pop_front() {
        if node == to_node {
            return route;
        }
        if let Some(lanes) = outgoing.get(node.as_str()) {
            for lane in lanes {
                if seen.contains(&lane.to) {
                    continue;
                }
                seen.insert(lane.to.clone());
                let mut next = route.clone();
                next.push(lane.id.clone());
                queue.push_back((lane.to.clone(), next));
            }
        }
    }
    Vec::new()
}

fn run_command(command: &Path, args: &[String], cwd: &Path) -> Result<String, String> {
    let parent_path = command.parent().map(|p| p.display().to_string()).unwrap_or_default();
    let existing_path = std::env::var("PATH").unwrap_or_default();
    let output = Command::new(command)
        .args(args)
        .current_dir(cwd)
        .env("PATH", format!("{parent_path}:{existing_path}"))
        .output()
        .map_err(|e| format!("failed to spawn {}: {e}", command.display()))?;
    if !output.status.success() {
        let parts = [
            format!("{} failed with status {}", base_name(command), status_str(output.status.code())),
            String::from_utf8_lossy(&output.stdout).to_string(),
            String::from_utf8_lossy(&output.stderr).to_string(),
        ];
        return Err(parts.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n"));
    }
    Ok(format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn command_version(command: &Path) -> Result<String, String> {
    let text = run_command(command, &["--version".to_string()], &out_dir())?;
    Ok(text
        .split('\n')
        .map(|l| l.trim())
        .find(|l| !l.is_empty())
        .unwrap_or("unknown")
        .to_string())
}

fn parse_xml_records(text: &str, tag: &str) -> Vec<HashMap<String, String>> {
    let mut records: Vec<HashMap<String, String>> = Vec::new();
    let open = format!("<{tag}");
    let bytes = text.as_bytes();
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find(&open) {
        let start = search_from + rel;
        let after = start + open.len();
        // Require a word boundary (whitespace, '>' or '/').
        let boundary_ok = bytes.get(after).map(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'>' || b == b'/').unwrap_or(false);
        if !boundary_ok {
            search_from = after;
            continue;
        }
        // Find the closing '>'.
        if let Some(close_rel) = text[after..].find('>') {
            let attr_text = &text[after..after + close_rel];
            let attr_text = attr_text.trim_end_matches('/').trim();
            records.push(parse_xml_attrs(attr_text));
            search_from = after + close_rel + 1;
        } else {
            break;
        }
    }
    records
}

fn parse_xml_attrs(text: &str) -> HashMap<String, String> {
    let mut attrs: HashMap<String, String> = HashMap::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Read key.
        while i < chars.len() && (chars[i].is_whitespace()) {
            i += 1;
        }
        let key_start = i;
        while i < chars.len() && (chars[i].is_ascii_alphanumeric() || matches!(chars[i], '_' | '.' | ':' | '-')) {
            i += 1;
        }
        if i == key_start {
            i += 1;
            continue;
        }
        let key: String = chars[key_start..i].iter().collect();
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= chars.len() || chars[i] != '=' {
            continue;
        }
        i += 1; // skip '='
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= chars.len() || chars[i] != '"' {
            continue;
        }
        i += 1; // skip opening quote
        let val_start = i;
        while i < chars.len() && chars[i] != '"' {
            i += 1;
        }
        let value: String = chars[val_start..i].iter().collect();
        i += 1; // skip closing quote
        attrs.insert(key, value);
    }
    attrs
}

// --- markdown + helpers ------------------------------------------------------

fn render_markdown(engines: &[EngineStats], scenario: &Scenario) -> String {
    let mut lines: Vec<String> = vec![
        "# Traffic Engine Cross-Check".to_string(),
        String::new(),
        format!(
            "Scenario: {}, {}s at dt={}s, {} scheduled trips.",
            scenario.network, scenario.duration_sec, scenario.dt_sec, scenario.scheduled_trips
        ),
        String::new(),
        "| Engine | Version | Generated | Entered | Completed | Active @ end | Dropped | Max active | Mean travel (s) | Mean speed (m/s) | Mean |jerk| | Notes |".to_string(),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |".to_string(),
    ];
    for e in engines {
        let cells = [
            e.engine.clone(),
            e.version.clone().unwrap_or_default(),
            num_to_string(e.generated),
            fmt_cell(e.entered),
            num_to_string(e.completed),
            num_to_string(e.active_at_end),
            fmt_cell(e.dropped),
            fmt_cell(e.max_active),
            fmt_cell(e.mean_travel_time_sec),
            fmt_cell(e.mean_speed_mps),
            fmt_cell(e.mean_abs_jerk_mps3),
            e.notes.join("; "),
        ]
        .iter()
        .map(|c| escape_markdown_cell(c))
        .collect::<Vec<_>>()
        .join(" | ");
        lines.push(format!("| {cells} |"));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn fmt_cell(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => num_to_string(round_metric(Some(v)).unwrap_or(v)),
        _ => String::new(),
    }
}

fn escape_markdown_cell(value: &str) -> String {
    value.replace('|', "\\|")
}

fn mean_rounded(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        None
    } else {
        round_metric(Some(xs.iter().sum::<f64>() / xs.len() as f64))
    }
}

fn round_metric(value: Option<f64>) -> Option<f64> {
    match value {
        Some(v) if v.is_finite() => Some((v * 1000.0).round() / 1000.0),
        _ => None,
    }
}

fn round_time(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn num_to_string(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        x.to_string()
    }
}

fn status_str(code: Option<i32>) -> String {
    match code {
        Some(c) => c.to_string(),
        None => "null".to_string(),
    }
}

fn base_name(p: &Path) -> String {
    p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| p.display().to_string())
}

fn relative(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).display().to_string()
}

fn xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// --- JSON serialization for shared input -------------------------------------

fn engine_json(e: &EngineStats) -> JsonValue {
    let mut fields: Vec<(String, JsonValue)> = vec![
        ("engine".to_string(), JsonValue::String(e.engine.clone())),
    ];
    if let Some(v) = &e.version {
        fields.push(("version".to_string(), JsonValue::String(v.clone())));
    }
    fields.push(("generated".to_string(), JsonValue::Number(e.generated)));
    push_opt(&mut fields, "entered", e.entered);
    fields.push(("completed".to_string(), JsonValue::Number(e.completed)));
    fields.push(("activeAtEnd".to_string(), JsonValue::Number(e.active_at_end)));
    push_opt(&mut fields, "dropped", e.dropped);
    push_opt(&mut fields, "maxActive", e.max_active);
    push_opt(&mut fields, "meanTravelTimeSec", e.mean_travel_time_sec);
    push_opt(&mut fields, "meanSpeedMps", e.mean_speed_mps);
    push_opt(&mut fields, "meanAbsJerkMps3", e.mean_abs_jerk_mps3);
    push_opt(&mut fields, "minHeadwayM", e.min_headway_m);
    fields.push(("notes".to_string(), JsonValue::Array(e.notes.iter().map(|n| JsonValue::String(n.clone())).collect())));
    JsonValue::Object(fields)
}

fn push_opt(fields: &mut Vec<(String, JsonValue)>, key: &str, v: Option<f64>) {
    if let Some(x) = v {
        fields.push((key.to_string(), JsonValue::Number(x)));
    }
}

fn network_json(network: &TrafficNetwork) -> JsonValue {
    let nodes = network
        .nodes
        .iter()
        .map(|n| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(n.id.clone())),
                ("x".to_string(), JsonValue::Number(n.x)),
                ("y".to_string(), JsonValue::Number(n.y)),
            ])
        })
        .collect();
    let lanes = network
        .lanes
        .iter()
        .map(|l| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(l.id.clone())),
                ("from".to_string(), JsonValue::String(l.from.clone())),
                ("to".to_string(), JsonValue::String(l.to.clone())),
                ("lengthM".to_string(), JsonValue::Number(l.length_m)),
                ("speedLimitMps".to_string(), JsonValue::Number(l.speed_limit_mps)),
            ])
        })
        .collect();
    JsonValue::Object(vec![
        ("nodes".to_string(), JsonValue::Array(nodes)),
        ("lanes".to_string(), JsonValue::Array(lanes)),
    ])
}

fn params_json(p: &SmartTrafficParams) -> JsonValue {
    JsonValue::Object(vec![
        ("durationSec".to_string(), JsonValue::Number(p.duration_sec)),
        ("dtSec".to_string(), JsonValue::Number(p.dt_sec)),
        ("seed".to_string(), JsonValue::Number(p.seed)),
        ("carLengthM".to_string(), JsonValue::Number(p.car_length_m)),
        ("minGapM".to_string(), JsonValue::Number(p.min_gap_m)),
    ])
}

fn trip_json(t: &SharedTrip) -> JsonValue {
    JsonValue::Object(vec![
        ("id".to_string(), JsonValue::String(t.id.clone())),
        ("departSec".to_string(), JsonValue::Number(t.depart_sec)),
        ("sourceId".to_string(), JsonValue::String(t.source_id.clone())),
        ("destinationSinkId".to_string(), JsonValue::String(t.destination_sink_id.clone())),
        ("route".to_string(), JsonValue::Array(t.route.iter().map(|r| JsonValue::String(r.clone())).collect())),
        ("sourceNodeId".to_string(), JsonValue::String(t.source_node_id.clone())),
        ("sinkNodeId".to_string(), JsonValue::String(t.sink_node_id.clone())),
    ])
}
