//! Port of `src/des/runners/compare-external-fel-models.ts`.
//!
//! Source/sink DES comparisons vs external FEL reference models. Every scenario
//! writes one shared JSON input and feeds that same file into the internal
//! runner and the external program. The TS top-level `main()` becomes [`run`],
//! which returns the intended process exit code.
//!
//! ## PORT NOTE
//!
//!   * Reuses the real `crate::des::general::network_flow` traffic types +
//!     `build_five_intersection_traffic_network`, and the real
//!     `crate::des::general::computer_network` problem/result/simulation.
//!   * The internal smart-traffic side delegates to the real
//!     `crate::des::general::smart_traffic_flow` module, then reduces the result
//!     to the source/sink summary shape used by the external comparisons.
//!   * `import './external-modules'` (import-time registration) → explicit
//!     [`register_built_in_external_modules`] call; if it fails (missing
//!     `external-references/` scripts) we log and continue, so external engines
//!     simply report `failed`/`skipped` instead of crashing the whole driver.
//!   * `JSON.parse`/`stringify` → [`parse_json`] / `JsonValue::to_string_pretty`.
//!   * `mulberry32`/`Date.now()` → `mulberry32` / `SystemClock`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::des::general::computer_network::{
    build_bottleneck_computer_network_problem, build_default_computer_network_problem,
    run_computer_network_simulation, validate_computer_network_problem, ComputerNetworkProblem,
    ComputerNetworkResult,
};
use crate::des::general::network_flow::{
    build_five_intersection_traffic_network, TrafficLane, TrafficNetwork, TrafficParams,
    TrafficScheduledTrip,
};
use crate::des::general::prng::mulberry32;
use crate::des::general::smart_traffic_flow::{
    run_smart_traffic_flow as run_smart_traffic_flow_model,
    SmartTrafficParams as SmartTrafficModelParams,
};
use crate::des::observability::logger::{parse_json, JsonValue};
use crate::des::shared::capabilities::RandomSource;

use super::external_modules::{
    register_built_in_external_modules, COMPUTER_NETWORK_FEL_REFERENCE_ID,
    TRAFFIC_CIW_REFERENCE_ID, TRAFFIC_FEL_REFERENCE_ID, TRAFFIC_SIMPY_REFERENCE_ID,
    TRAFFIC_SUMO_REFERENCE_ID,
};
use super::external_program::{
    repo_root_from_runner, run_external_module, ExternalModuleParams, ExternalProgramResult,
    ParamValue,
};

// =============================================================================
// Smart-traffic source/sink comparison params + summary.
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct SmartTrafficParams {
    pub duration_sec: f64,
    pub dt_sec: f64,
    pub seed: f64,
    pub actor_shuffle_seed: Option<f64>,
    pub max_cars: Option<f64>,
    pub smart_car_pool_size: Option<f64>,
    pub spawn_rate_multiplier: Option<f64>,
    pub car_length_m: Option<f64>,
    pub car_width_m: Option<f64>,
    pub lane_width_m: Option<f64>,
    pub min_gap_m: Option<f64>,
    pub max_accel_mps2: Option<f64>,
    pub max_decel_mps2: Option<f64>,
    pub max_jerk_mps3: Option<f64>,
    pub reaction_time_sec: Option<f64>,
    pub time_headway_sec: Option<f64>,
    pub grid_cell_size_m: Option<f64>,
    pub accident_risk_scale: Option<f64>,
    pub accident_probability: Option<f64>,
    pub distance_preference_spread: Option<f64>,
    pub start_preference_spread: Option<f64>,
    pub scheduled_trips_len: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SmartTrafficResult {
    pub entered: usize,
    pub exited: usize,
    pub dropped: usize,
    pub final_cars_len: usize,
    pub mean_travel_time_sec: f64,
    pub mean_speed_mps: f64,
    pub scheduled_trips_len: usize,
}

pub fn run_smart_traffic_flow(
    params: &SmartTrafficParams,
    network: &TrafficNetwork,
    scheduled_trips: &[TrafficScheduledTrip],
) -> SmartTrafficResult {
    let result = run_smart_traffic_flow_model(
        SmartTrafficModelParams {
            base: TrafficParams {
                builtin: None,
                network: Some(network.clone()),
                duration_sec: params.duration_sec,
                dt_sec: params.dt_sec,
                seed: params.seed,
                max_cars: params.max_cars.unwrap_or(100.0).max(1.0) as usize,
                car_length_m: params.car_length_m,
                car_width_m: params.car_width_m,
                lane_width_m: params.lane_width_m,
                min_gap_m: params.min_gap_m,
                max_accel_mps2: params.max_accel_mps2,
                max_decel_mps2: params.max_decel_mps2,
                max_jerk_mps3: params.max_jerk_mps3,
                reaction_time_sec: params.reaction_time_sec,
                time_headway_sec: params.time_headway_sec,
                grid_cell_size_m: params.grid_cell_size_m,
                grid_look_ahead_m: None,
                spawn_rate_multiplier: Some(0.0),
                scheduled_trips: Some(scheduled_trips.to_vec()),
            },
            smart_car_pool_size: params.smart_car_pool_size.map(|v| v.max(1.0) as usize),
            actor_shuffle_seed: params.actor_shuffle_seed,
            accident_risk_scale: params.accident_risk_scale,
            accident_probability: params.accident_probability,
            accident_accel_boost_mps2: None,
            accident_fault_duration_sec: None,
            distance_preference_spread: params.distance_preference_spread,
            start_preference_spread: params.start_preference_spread,
            accident_flash_seconds: None,
        },
        None,
    );
    SmartTrafficResult {
        entered: result.entered,
        exited: result.exited,
        dropped: result.dropped,
        final_cars_len: result.final_cars.len(),
        mean_travel_time_sec: result.mean_travel_time_sec,
        mean_speed_mps: result.mean_speed_mps,
        scheduled_trips_len: scheduled_trips.len(),
    }
}

// =============================================================================
// Report types.
// =============================================================================

#[derive(Clone, Debug)]
struct SharedTrafficTrip {
    depart_sec: f64,
    source_id: String,
    destination_sink_id: String,
    id: String,
    route: Vec<String>,
    source_node_id: String,
    sink_node_id: String,
}

#[derive(Clone, Debug)]
struct CheckRow {
    name: String,
    passed: bool,
    detail: String,
}

#[derive(Clone, Debug)]
struct EngineReport {
    domain: String,
    scenario: String,
    engine: String,
    status: String,
    input_path: String,
    output_path: Option<String>,
    checks: Vec<CheckRow>,
    notes: Vec<String>,
}

fn check_row(name: &str, passed: bool, detail: String) -> CheckRow {
    CheckRow {
        name: name.to_string(),
        passed,
        detail,
    }
}

fn out_dir() -> PathBuf {
    repo_root_from_runner()
        .join("out")
        .join("external-fel-comparison")
}

/// `main()` — returns the intended exit code (1 if any report failed).
pub fn run() -> i32 {
    if let Err(e) = register_built_in_external_modules() {
        eprintln!("[compare-external-fel-models] external modules not registered: {e}");
    }
    let dir = out_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[compare-external-fel-models] could not create out dir: {e}");
        return 1;
    }
    let mut reports: Vec<EngineReport> = Vec::new();
    reports.extend(compare_traffic());
    reports.extend(compare_computer_network());

    let root = repo_root_from_runner();
    let report = JsonValue::Object(vec![
        ("generatedAt".to_string(), JsonValue::String(timestamp())),
        (
            "sharedInputContract".to_string(),
            JsonValue::String(
                "Internal and external runs read the same JSON scenario files from out/external-fel-comparison.".to_string(),
            ),
        ),
        ("reports".to_string(), JsonValue::Array(reports.iter().map(report_to_json).collect())),
    ]);
    let json_path = dir.join("comparison-report.json");
    let md_path = dir.join("comparison-report.md");
    let _ = fs::write(&json_path, report.to_string_pretty(2));
    let md = render_markdown(&reports, &root);
    let _ = fs::write(&md_path, &md);
    println!("{md}");
    println!("Wrote {}", relative(&root, &json_path));
    if reports.iter().any(|r| r.status == "failed") {
        return 1;
    }
    0
}

fn compare_traffic() -> Vec<EngineReport> {
    let network = build_five_intersection_traffic_network();
    let params = SmartTrafficParams {
        duration_sec: 180.0,
        dt_sec: 0.1,
        seed: 19.0,
        actor_shuffle_seed: Some(2026.0),
        max_cars: Some(180.0),
        smart_car_pool_size: Some(260.0),
        spawn_rate_multiplier: Some(0.35),
        car_length_m: Some(4.8),
        car_width_m: Some(1.8),
        lane_width_m: Some(3.7),
        min_gap_m: Some(2.5),
        max_accel_mps2: Some(2.2),
        max_decel_mps2: Some(4.0),
        max_jerk_mps3: Some(6.0),
        reaction_time_sec: Some(0.8),
        time_headway_sec: Some(1.1),
        grid_cell_size_m: Some(0.3048),
        accident_risk_scale: Some(0.0),
        accident_probability: Some(0.0),
        distance_preference_spread: Some(0.0),
        start_preference_spread: Some(0.0),
        scheduled_trips_len: 0,
    };
    let demand_end_sec = 120.0;
    let trips = generate_scheduled_trips(
        &network,
        &params,
        (params.seed + 4242.0) as u32,
        demand_end_sec,
    );
    let input_path = out_dir().join("traffic-shared-input.json");
    let input_json = shared_traffic_input_json(&network, &params, &trips, demand_end_sec);
    let _ = fs::write(&input_path, input_json.to_string_pretty(2));

    let source_checks = validate_shared_traffic_input(&network, &params, &trips);
    let input_path_str = input_path.display().to_string();
    if source_checks.iter().any(|c| !c.passed) {
        return vec![EngineReport {
            domain: "traffic".to_string(),
            scenario: "five-intersection-scheduled-trips".to_string(),
            engine: "source/sink input".to_string(),
            status: "failed".to_string(),
            input_path: input_path_str,
            output_path: None,
            checks: source_checks,
            notes: vec!["shared traffic input failed before simulation".to_string()],
        }];
    }

    let internal_trips: Vec<TrafficScheduledTrip> = trips
        .iter()
        .map(|t| TrafficScheduledTrip {
            depart_sec: t.depart_sec,
            source_id: t.source_id.clone(),
            destination_sink_id: t.destination_sink_id.clone(),
        })
        .collect();
    let mut internal_params = params.clone();
    internal_params.scheduled_trips_len = internal_trips.len();
    let internal = run_smart_traffic_flow(&internal_params, &network, &internal_trips);

    let mut reports: Vec<EngineReport> = vec![
        run_traffic_external(
            "Python traffic FEL",
            TRAFFIC_FEL_REFERENCE_ID,
            &input_path_str,
            &internal,
        ),
        run_traffic_external(
            "SimPy",
            TRAFFIC_SIMPY_REFERENCE_ID,
            &input_path_str,
            &internal,
        ),
        run_traffic_external("Ciw", TRAFFIC_CIW_REFERENCE_ID, &input_path_str, &internal),
        run_traffic_external(
            "SUMO",
            TRAFFIC_SUMO_REFERENCE_ID,
            &input_path_str,
            &internal,
        ),
    ];
    reports.insert(
        0,
        EngineReport {
            domain: "traffic".to_string(),
            scenario: "five-intersection-scheduled-trips".to_string(),
            engine: "source/sink input".to_string(),
            status: "passed".to_string(),
            input_path: input_path_str,
            output_path: None,
            checks: source_checks,
            notes: vec![format!(
                "internal DES entered={} exited={} dropped={}",
                internal.entered, internal.exited, internal.dropped
            )],
        },
    );
    reports
}

fn run_external_module_safe(id: &str, params: &ExternalModuleParams) -> ExternalProgramResult {
    match run_external_module(id, params) {
        Ok(r) => r,
        Err(e) => ExternalProgramResult {
            command: String::new(),
            args: Vec::new(),
            status: None,
            stdout: String::new(),
            stderr: e,
            module_id: Some(id.to_string()),
        },
    }
}

fn optional_external_unavailable(ext: &ExternalProgramResult) -> Option<String> {
    let stderr = ext.stderr.trim();
    let stdout = ext.stdout.trim();
    let message = if stderr.is_empty() { stdout } else { stderr };
    let lower = message.to_ascii_lowercase();
    let looks_unavailable = lower.contains("unknown external module")
        || lower.contains("not registered")
        || lower.contains("external script not found")
        || lower.contains("no such file")
        || lower.contains("no module named")
        || lower.contains("modulenotfounderror")
        || lower.contains("not installed")
        || lower.contains("unavailable");

    if looks_unavailable {
        Some(if message.is_empty() {
            "optional external dependency unavailable".to_string()
        } else {
            slice_chars(message, 500)
        })
    } else {
        None
    }
}

fn run_traffic_external(
    engine: &str,
    module_id: &str,
    input_path: &str,
    internal: &SmartTrafficResult,
) -> EngineReport {
    let slug = engine
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>();
    let slug = collapse_dashes(&slug);
    let output_path = out_dir().join(format!("{slug}.json"));
    let output_path_str = output_path.display().to_string();

    let mut params = ExternalModuleParams::new();
    params.insert(
        "problem".to_string(),
        ParamValue::Str(input_path.to_string()),
    );
    params.insert("out".to_string(), ParamValue::Str(output_path_str.clone()));
    params.insert(
        "collisionAction".to_string(),
        ParamValue::Str("warn".to_string()),
    );
    let ext = run_external_module_safe(module_id, &params);

    let mut notes = vec![format!(
        "external command status={}",
        status_str(ext.status)
    )];
    if !ext.stderr.trim().is_empty() {
        notes.push(slice_chars(ext.stderr.trim(), 500));
    }
    let base = |status: &str, checks: Vec<CheckRow>, notes: Vec<String>| EngineReport {
        domain: "traffic".to_string(),
        scenario: "five-intersection-scheduled-trips".to_string(),
        engine: engine.to_string(),
        status: status.to_string(),
        input_path: input_path.to_string(),
        output_path: Some(output_path_str.clone()),
        checks,
        notes,
    };

    if ext.status != Some(0) || !output_path.exists() {
        if let Some(message) = optional_external_unavailable(&ext) {
            return base(
                "skipped",
                vec![check_row(
                    "optional external dependency unavailable",
                    true,
                    message,
                )],
                notes,
            );
        }
        return base(
            "failed",
            vec![check_row(
                "external process writes output JSON",
                false,
                format!("status={}", status_str(ext.status)),
            )],
            notes,
        );
    }
    let text = match fs::read_to_string(&output_path) {
        Ok(t) => t,
        Err(e) => {
            return base(
                "failed",
                vec![check_row("read output JSON", false, e.to_string())],
                notes,
            )
        }
    };
    let payload = match parse_json(&text) {
        Ok(v) => v,
        Err(e) => {
            return base(
                "failed",
                vec![check_row("parse output JSON", false, e)],
                notes,
            )
        }
    };
    let status_field = payload.get("status").and_then(|v| v.as_str());
    if status_field == Some("unavailable") {
        let msg = payload
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unavailable")
            .to_string();
        return base(
            "skipped",
            vec![check_row(
                "external dependency reported unavailable cleanly",
                true,
                msg,
            )],
            notes,
        );
    }
    if let Some(s) = status_field {
        if s != "ok" {
            let msg = payload
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or(s)
                .to_string();
            return base(
                "failed",
                vec![check_row("external payload status is ok", false, msg)],
                notes,
            );
        }
    }
    let result = payload.get("result").cloned().unwrap_or(JsonValue::Null);
    let checks = compare_traffic_stats(internal, &result);
    let all = checks.iter().all(|c| c.passed);
    let result_notes = result
        .get("notes")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or(notes);
    base(if all { "passed" } else { "failed" }, checks, result_notes)
}

fn compare_computer_network() -> Vec<EngineReport> {
    let scenarios: Vec<(&str, ComputerNetworkProblem)> = vec![
        ("small-enterprise", build_default_computer_network_problem()),
        (
            "bottleneck-lab",
            build_bottleneck_computer_network_problem(),
        ),
    ];
    let mut reports: Vec<EngineReport> = Vec::new();
    for (name, problem) in scenarios {
        validate_computer_network_problem(&problem).expect("invalid computer-network problem");
        let input = JsonValue::Object(vec![
            (
                "$schema".to_string(),
                JsonValue::String("des/model-spec/v1".to_string()),
            ),
            (
                "model".to_string(),
                JsonValue::String("computer-network".to_string()),
            ),
            (
                "description".to_string(),
                JsonValue::String(format!("{name} shared source/sink packet-flow comparison")),
            ),
            // PORT NOTE: the problem is run from the in-memory builder; we do not
            // round-trip it through JSON (no Serialize for ComputerNetworkProblem).
            (
                "parameters".to_string(),
                JsonValue::Object(vec![(
                    "problem".to_string(),
                    JsonValue::String(name.to_string()),
                )]),
            ),
            (
                "runtime".to_string(),
                JsonValue::Object(vec![("verbose".to_string(), JsonValue::Bool(false))]),
            ),
        ]);
        let input_path = out_dir().join(format!("computer-network-{name}.json"));
        let output_path =
            out_dir().join(format!("computer-network-{name}-python-fel-reference.json"));
        let _ = fs::write(&input_path, input.to_string_pretty(2));

        let internal = run_computer_network_simulation(&problem);
        let mut params = ExternalModuleParams::new();
        params.insert(
            "problem".to_string(),
            ParamValue::Str(input_path.display().to_string()),
        );
        params.insert(
            "out".to_string(),
            ParamValue::Str(output_path.display().to_string()),
        );
        let ext = run_external_module_safe(COMPUTER_NETWORK_FEL_REFERENCE_ID, &params);
        let mut notes = vec![format!(
            "external command status={}",
            status_str(ext.status)
        )];
        if !ext.stderr.trim().is_empty() {
            notes.push(slice_chars(ext.stderr.trim(), 500));
        }
        if ext.status != Some(0) || !output_path.exists() {
            if let Some(message) = optional_external_unavailable(&ext) {
                reports.push(EngineReport {
                    domain: "computer-network".to_string(),
                    scenario: name.to_string(),
                    engine: "Python computer-network FEL".to_string(),
                    status: "skipped".to_string(),
                    input_path: input_path.display().to_string(),
                    output_path: Some(output_path.display().to_string()),
                    checks: vec![check_row(
                        "optional external dependency unavailable",
                        true,
                        message,
                    )],
                    notes,
                });
                continue;
            }
            reports.push(EngineReport {
                domain: "computer-network".to_string(),
                scenario: name.to_string(),
                engine: "Python computer-network FEL".to_string(),
                status: "failed".to_string(),
                input_path: input_path.display().to_string(),
                output_path: Some(output_path.display().to_string()),
                checks: vec![check_row(
                    "external process writes output JSON",
                    false,
                    format!("status={}", status_str(ext.status)),
                )],
                notes,
            });
            continue;
        }
        let text = fs::read_to_string(&output_path).unwrap_or_default();
        let external = parse_json(&text)
            .ok()
            .and_then(|v| v.get("result").cloned())
            .unwrap_or(JsonValue::Null);
        let checks = compare_computer_network_stats(&internal, &external);
        let all = checks.iter().all(|c| c.passed);
        reports.push(EngineReport {
            domain: "computer-network".to_string(),
            scenario: name.to_string(),
            engine: "Python computer-network FEL".to_string(),
            status: if all {
                "passed".to_string()
            } else {
                "failed".to_string()
            },
            input_path: input_path.display().to_string(),
            output_path: Some(output_path.display().to_string()),
            checks,
            notes,
        });
    }
    reports
}

fn validate_shared_traffic_input(
    network: &TrafficNetwork,
    params: &SmartTrafficParams,
    trips: &[SharedTrafficTrip],
) -> Vec<CheckRow> {
    let mut checks: Vec<CheckRow> = Vec::new();
    let lane_by_id: HashMap<&str, &TrafficLane> =
        network.lanes.iter().map(|l| (l.id.as_str(), l)).collect();
    checks.push(check_row(
        "has source entities",
        !network.sources.is_empty(),
        format!("sources={}", network.sources.len()),
    ));
    checks.push(check_row(
        "has sink entities",
        !network.sinks.is_empty(),
        format!("sinks={}", network.sinks.len()),
    ));
    checks.push(check_row(
        "has scheduled source trips",
        !trips.is_empty(),
        format!("trips={}", trips.len()),
    ));
    for trip in trips {
        let source = network.sources.iter().find(|s| s.id == trip.source_id);
        let sink = network
            .sinks
            .iter()
            .find(|s| s.id == trip.destination_sink_id);
        let route_ok = match (source, sink) {
            (Some(src), Some(snk)) => {
                route_connects(&trip.route, &src.node_id, &snk.node_id, &lane_by_id)
            }
            _ => false,
        };
        let allowed = match source {
            Some(src) => src
                .destination_sink_ids
                .clone()
                .unwrap_or_else(|| network.sinks.iter().map(|s| s.id.clone()).collect())
                .contains(&trip.destination_sink_id),
            None => false,
        };
        checks.push(check_row(
            &format!("{}: source exists", trip.id),
            source
                .map(|s| s.node_id == trip.source_node_id)
                .unwrap_or(false),
            format!("{}@{}", trip.source_id, trip.source_node_id),
        ));
        checks.push(check_row(
            &format!("{}: sink exists", trip.id),
            sink.map(|s| s.node_id == trip.sink_node_id)
                .unwrap_or(false),
            format!("{}@{}", trip.destination_sink_id, trip.sink_node_id),
        ));
        checks.push(check_row(
            &format!("{}: sink allowed by source", trip.id),
            allowed,
            format!("{}->{}", trip.source_id, trip.destination_sink_id),
        ));
        checks.push(check_row(
            &format!("{}: route connects source to sink", trip.id),
            route_ok,
            trip.route.join("->"),
        ));
        checks.push(check_row(
            &format!("{}: departSec is in horizon", trip.id),
            trip.depart_sec >= 0.0 && trip.depart_sec <= params.duration_sec,
            format!("depart={}", trip.depart_sec),
        ));
    }
    checks
}

fn compare_traffic_stats(internal: &SmartTrafficResult, external: &JsonValue) -> Vec<CheckRow> {
    let g = |k: &str| external.get(k).and_then(|v| v.as_f64());
    let mut checks = vec![
        exact_number(
            "generated demand matches internal scheduled input",
            g("generatedDemand").unwrap_or(f64::NAN),
            internal.scheduled_trips_len as f64,
        ),
        relative_number(
            "departures align with internal entered count",
            g("departed").unwrap_or(f64::NAN),
            internal.entered as f64,
            0.2,
        ),
        relative_number(
            "arrivals align with internal exited count",
            g("arrived").unwrap_or(f64::NAN),
            internal.exited as f64,
            0.45,
        ),
        close_number(
            "active-at-end aligns with internal final cars",
            g("activeAtEnd").unwrap_or(f64::NAN),
            internal.final_cars_len as f64,
            2.0,
        ),
        finite_number(
            "external mean travel time is finite",
            g("meanTravelTimeSec").unwrap_or(f64::NAN),
        ),
        finite_number(
            "external mean speed is finite",
            g("meanSpeedMps").unwrap_or(f64::NAN),
        ),
    ];
    let ext_mt = g("meanTravelTimeSec").unwrap_or(0.0);
    if internal.mean_travel_time_sec > 0.0 && ext_mt > 0.0 {
        checks.push(ratio_band(
            "mean travel time same order of magnitude",
            ext_mt,
            internal.mean_travel_time_sec,
            0.2,
            5.0,
        ));
    }
    checks
}

fn compare_computer_network_stats(
    internal: &ComputerNetworkResult,
    external: &JsonValue,
) -> Vec<CheckRow> {
    let g = |k: &str| external.get(k).and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
    let mut checks = vec![
        exact_number(
            "generated packets",
            g("generatedPackets"),
            internal.generated_packets,
        ),
        exact_number(
            "delivered packets",
            g("deliveredPackets"),
            internal.delivered_packets,
        ),
        exact_number(
            "dropped packets",
            g("droppedPackets"),
            internal.dropped_packets,
        ),
        exact_number(
            "active packets",
            g("activePackets"),
            internal.active_packets,
        ),
        exact_number(
            "max active packets",
            g("maxActivePackets"),
            internal.max_active_packets,
        ),
        close_number(
            "delivery ratio",
            g("deliveryRatio"),
            internal.delivery_ratio,
            1e-9,
        ),
        close_number(
            "offered load Mbps",
            g("offeredLoadMbps"),
            internal.offered_load_mbps,
            1e-9,
        ),
        close_number(
            "wire throughput Mbps",
            g("throughputMbps"),
            internal.throughput_mbps,
            1e-9,
        ),
        close_number(
            "goodput Mbps",
            g("goodputMbps"),
            internal.goodput_mbps,
            1e-9,
        ),
        close_number(
            "mean latency ms",
            g("meanLatencyMs"),
            internal.mean_latency_ms,
            1e-9,
        ),
        close_number(
            "p95 latency ms",
            g("p95LatencyMs"),
            internal.p95_latency_ms,
            1e-9,
        ),
        close_number("total cost", g("totalCost"), internal.total_cost, 1e-9),
    ];
    let ext_b0 = external
        .get("bottlenecks")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first());
    let ext_kind = ext_b0.and_then(|b| b.get("kind")).and_then(|v| v.as_str());
    let ext_id = ext_b0.and_then(|b| b.get("id")).and_then(|v| v.as_str());
    let ext_reason = ext_b0
        .and_then(|b| b.get("reason"))
        .and_then(|v| v.as_str());
    let int_b0 = internal.bottlenecks.first();
    let agree = ext_kind == int_b0.map(|b| b.kind.as_str())
        && ext_id == int_b0.map(|b| b.id.as_str())
        && ext_reason == int_b0.map(|b| b.reason.as_str());
    checks.push(check_row(
        "top bottleneck agrees",
        agree,
        format!(
            "internal={} external={}",
            bottleneck_label_internal(internal),
            bottleneck_label_json(external)
        ),
    ));
    checks
}

fn generate_scheduled_trips(
    network: &TrafficNetwork,
    params: &SmartTrafficParams,
    seed: u32,
    demand_end_sec: f64,
) -> Vec<SharedTrafficTrip> {
    let mut rng = mulberry32(seed);
    let mut accumulators: HashMap<String, f64> = HashMap::new();
    let mut trips: Vec<SharedTrafficTrip> = Vec::new();
    let ticks = (demand_end_sec / params.dt_sec).ceil() as i64;
    let sink_by_id: HashMap<&str, &str> = network
        .sinks
        .iter()
        .map(|s| (s.id.as_str(), s.node_id.as_str()))
        .collect();
    for source in &network.sources {
        accumulators.insert(source.id.clone(), 0.0);
    }
    let spawn_mult = params.spawn_rate_multiplier.unwrap_or(1.0);
    for tick in 0..ticks {
        let depart_sec = round_time(tick as f64 * params.dt_sec);
        for source in &network.sources {
            let expected = source.rate_per_min * spawn_mult * params.dt_sec / 60.0;
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
                let Some(sink_node) = sink_by_id.get(destination_sink_id.as_str()) else {
                    continue;
                };
                let route = shortest_lane_path(network, &source.node_id, sink_node);
                if route.is_empty() {
                    continue;
                }
                trips.push(SharedTrafficTrip {
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

fn shortest_lane_path(network: &TrafficNetwork, from_node: &str, to_node: &str) -> Vec<String> {
    let mut outgoing: HashMap<&str, Vec<&TrafficLane>> = HashMap::new();
    for lane in &network.lanes {
        outgoing.entry(lane.from.as_str()).or_default().push(lane);
    }
    let mut queue: std::collections::VecDeque<(String, Vec<String>)> =
        std::collections::VecDeque::new();
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

fn route_connects(
    route: &[String],
    source_node: &str,
    sink_node: &str,
    lane_by_id: &HashMap<&str, &TrafficLane>,
) -> bool {
    if route.is_empty() {
        return false;
    }
    let mut current = source_node.to_string();
    for lane_id in route {
        let Some(lane) = lane_by_id.get(lane_id.as_str()) else {
            return false;
        };
        if lane.from != current {
            return false;
        }
        current = lane.to.clone();
    }
    current == sink_node
}

// --- check helpers -----------------------------------------------------------

fn exact_number(name: &str, actual: f64, expected: f64) -> CheckRow {
    check_row(
        name,
        actual == expected,
        format!("actual={} expected={}", fmt_num(actual), fmt_num(expected)),
    )
}

fn close_number(name: &str, actual: f64, expected: f64, tolerance: f64) -> CheckRow {
    let diff = (actual - expected).abs();
    check_row(
        name,
        diff <= tolerance,
        format!(
            "actual={} expected={} diff={:e} tol={tolerance}",
            fmt_num(actual),
            fmt_num(expected),
            diff
        ),
    )
}

fn relative_number(name: &str, actual: f64, expected: f64, tolerance: f64) -> CheckRow {
    let diff = (actual - expected).abs();
    let rel = diff / actual.abs().max(expected.abs()).max(1.0);
    check_row(
        name,
        rel <= tolerance,
        format!(
            "actual={} expected={} rel={:.3} tol={tolerance}",
            fmt_num(actual),
            fmt_num(expected),
            rel
        ),
    )
}

fn ratio_band(name: &str, actual: f64, expected: f64, min_ratio: f64, max_ratio: f64) -> CheckRow {
    let ratio = actual / expected.max(1e-9);
    check_row(
        name,
        ratio >= min_ratio && ratio <= max_ratio,
        format!(
            "actual={} expected={} ratio={ratio:.3}",
            fmt_num(actual),
            fmt_num(expected)
        ),
    )
}

fn finite_number(name: &str, actual: f64) -> CheckRow {
    check_row(
        name,
        actual.is_finite(),
        format!("actual={}", fmt_num(actual)),
    )
}

fn bottleneck_label_internal(result: &ComputerNetworkResult) -> String {
    match result.bottlenecks.first() {
        Some(b) => format!("{}:{}:{}", b.kind, b.id, b.reason),
        None => "none".to_string(),
    }
}

fn bottleneck_label_json(external: &JsonValue) -> String {
    let b0 = external
        .get("bottlenecks")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first());
    match b0 {
        Some(b) => format!(
            "{}:{}:{}",
            b.get("kind").and_then(|v| v.as_str()).unwrap_or(""),
            b.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            b.get("reason").and_then(|v| v.as_str()).unwrap_or("")
        ),
        None => "none".to_string(),
    }
}

fn fmt_num(n: f64) -> String {
    if n.is_finite() {
        ((n * 1e6).round() / 1e6).to_string()
    } else {
        n.to_string()
    }
}

fn round_time(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn status_str(status: Option<i32>) -> String {
    match status {
        Some(c) => c.to_string(),
        None => "null".to_string(),
    }
}

fn slice_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

fn collapse_dashes(s: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for c in s.chars() {
        if c == '-' {
            if !prev_dash {
                out.push('-');
            }
            prev_dash = true;
        } else {
            out.push(c);
            prev_dash = false;
        }
    }
    out.trim_matches('-').to_string()
}

fn timestamp() -> String {
    // PORT NOTE: `new Date().toISOString()` → epoch-millis string (no chrono dep).
    use crate::des::shared::capabilities::{Clock, SystemClock};
    format!("{}", SystemClock.now_ms())
}

fn relative(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).display().to_string()
}

// --- shared-input + report JSON ----------------------------------------------

fn opt(v: Option<f64>) -> JsonValue {
    match v {
        Some(x) => JsonValue::Number(x),
        None => JsonValue::Null,
    }
}

fn shared_traffic_input_json(
    network: &TrafficNetwork,
    params: &SmartTrafficParams,
    trips: &[SharedTrafficTrip],
    demand_end_sec: f64,
) -> JsonValue {
    JsonValue::Object(vec![
        (
            "schema".to_string(),
            JsonValue::String("des/shared-traffic-source-sink/v1".to_string()),
        ),
        (
            "model".to_string(),
            JsonValue::String("smart-traffic-flow".to_string()),
        ),
        ("params".to_string(), params_json(params)),
        ("network".to_string(), network_json(network)),
        (
            "trips".to_string(),
            JsonValue::Array(trips.iter().map(trip_json).collect()),
        ),
        (
            "sourceInitialConditions".to_string(),
            JsonValue::Object(vec![
                (
                    "sourceCount".to_string(),
                    JsonValue::Number(network.sources.len() as f64),
                ),
                (
                    "sinkCount".to_string(),
                    JsonValue::Number(network.sinks.len() as f64),
                ),
                (
                    "scheduledTrips".to_string(),
                    JsonValue::Number(trips.len() as f64),
                ),
                (
                    "demandEndSec".to_string(),
                    JsonValue::Number(demand_end_sec),
                ),
            ]),
        ),
    ])
}

fn params_json(p: &SmartTrafficParams) -> JsonValue {
    JsonValue::Object(vec![
        ("durationSec".to_string(), JsonValue::Number(p.duration_sec)),
        ("dtSec".to_string(), JsonValue::Number(p.dt_sec)),
        ("seed".to_string(), JsonValue::Number(p.seed)),
        ("actorShuffleSeed".to_string(), opt(p.actor_shuffle_seed)),
        ("maxCars".to_string(), opt(p.max_cars)),
        ("smartCarPoolSize".to_string(), opt(p.smart_car_pool_size)),
        (
            "spawnRateMultiplier".to_string(),
            opt(p.spawn_rate_multiplier),
        ),
        ("carLengthM".to_string(), opt(p.car_length_m)),
        ("carWidthM".to_string(), opt(p.car_width_m)),
        ("laneWidthM".to_string(), opt(p.lane_width_m)),
        ("minGapM".to_string(), opt(p.min_gap_m)),
        ("maxAccelMps2".to_string(), opt(p.max_accel_mps2)),
        ("maxDecelMps2".to_string(), opt(p.max_decel_mps2)),
        ("maxJerkMps3".to_string(), opt(p.max_jerk_mps3)),
        ("reactionTimeSec".to_string(), opt(p.reaction_time_sec)),
        ("timeHeadwaySec".to_string(), opt(p.time_headway_sec)),
        ("gridCellSizeM".to_string(), opt(p.grid_cell_size_m)),
        ("accidentRiskScale".to_string(), opt(p.accident_risk_scale)),
        (
            "accidentProbability".to_string(),
            opt(p.accident_probability),
        ),
        (
            "distancePreferenceSpread".to_string(),
            opt(p.distance_preference_spread),
        ),
        (
            "startPreferenceSpread".to_string(),
            opt(p.start_preference_spread),
        ),
    ])
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
                (
                    "speedLimitMps".to_string(),
                    JsonValue::Number(l.speed_limit_mps),
                ),
            ])
        })
        .collect();
    let sources = network
        .sources
        .iter()
        .map(|s| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(s.id.clone())),
                ("nodeId".to_string(), JsonValue::String(s.node_id.clone())),
                ("ratePerMin".to_string(), JsonValue::Number(s.rate_per_min)),
            ])
        })
        .collect();
    let sinks = network
        .sinks
        .iter()
        .map(|s| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(s.id.clone())),
                ("nodeId".to_string(), JsonValue::String(s.node_id.clone())),
            ])
        })
        .collect();
    JsonValue::Object(vec![
        ("nodes".to_string(), JsonValue::Array(nodes)),
        ("lanes".to_string(), JsonValue::Array(lanes)),
        ("sources".to_string(), JsonValue::Array(sources)),
        ("sinks".to_string(), JsonValue::Array(sinks)),
    ])
}

fn trip_json(t: &SharedTrafficTrip) -> JsonValue {
    JsonValue::Object(vec![
        ("id".to_string(), JsonValue::String(t.id.clone())),
        ("departSec".to_string(), JsonValue::Number(t.depart_sec)),
        (
            "sourceId".to_string(),
            JsonValue::String(t.source_id.clone()),
        ),
        (
            "destinationSinkId".to_string(),
            JsonValue::String(t.destination_sink_id.clone()),
        ),
        (
            "route".to_string(),
            JsonValue::Array(
                t.route
                    .iter()
                    .map(|r| JsonValue::String(r.clone()))
                    .collect(),
            ),
        ),
        (
            "sourceNodeId".to_string(),
            JsonValue::String(t.source_node_id.clone()),
        ),
        (
            "sinkNodeId".to_string(),
            JsonValue::String(t.sink_node_id.clone()),
        ),
    ])
}

fn report_to_json(r: &EngineReport) -> JsonValue {
    JsonValue::Object(vec![
        ("domain".to_string(), JsonValue::String(r.domain.clone())),
        (
            "scenario".to_string(),
            JsonValue::String(r.scenario.clone()),
        ),
        ("engine".to_string(), JsonValue::String(r.engine.clone())),
        ("status".to_string(), JsonValue::String(r.status.clone())),
        (
            "inputPath".to_string(),
            JsonValue::String(r.input_path.clone()),
        ),
        (
            "outputPath".to_string(),
            match &r.output_path {
                Some(p) => JsonValue::String(p.clone()),
                None => JsonValue::Null,
            },
        ),
        (
            "checks".to_string(),
            JsonValue::Array(
                r.checks
                    .iter()
                    .map(|c| {
                        JsonValue::Object(vec![
                            ("name".to_string(), JsonValue::String(c.name.clone())),
                            ("passed".to_string(), JsonValue::Bool(c.passed)),
                            ("detail".to_string(), JsonValue::String(c.detail.clone())),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "notes".to_string(),
            JsonValue::Array(
                r.notes
                    .iter()
                    .map(|n| JsonValue::String(n.clone()))
                    .collect(),
            ),
        ),
    ])
}

fn render_markdown(reports: &[EngineReport], root: &Path) -> String {
    let mut lines: Vec<String> = vec![
        "# External FEL Comparison".to_string(),
        String::new(),
        "| Domain | Scenario | Engine | Status | Checks | Input | Output |".to_string(),
        "| --- | --- | --- | --- | ---: | --- | --- |".to_string(),
    ];
    for report in reports {
        let passed = report.checks.iter().filter(|c| c.passed).count();
        let cells = [
            report.domain.clone(),
            report.scenario.clone(),
            report.engine.clone(),
            report.status.clone(),
            format!("{}/{}", passed, report.checks.len()),
            relative(root, Path::new(&report.input_path)),
            report
                .output_path
                .as_ref()
                .map(|p| relative(root, Path::new(p)))
                .unwrap_or_default(),
        ]
        .iter()
        .map(|c| escape_markdown_cell(c))
        .collect::<Vec<_>>()
        .join(" | ");
        lines.push(format!("| {cells} |"));
    }
    let failed: Vec<String> = reports
        .iter()
        .flat_map(|report| {
            report.checks.iter().filter(|c| !c.passed).map(move |c| {
                format!(
                    "- {}/{}/{}: {} ({})",
                    report.domain, report.scenario, report.engine, c.name, c.detail
                )
            })
        })
        .collect();
    if !failed.is_empty() {
        lines.push(String::new());
        lines.push("## Failed Checks".to_string());
        lines.push(String::new());
        lines.extend(failed);
    }
    let skipped: Vec<&EngineReport> = reports.iter().filter(|r| r.status == "skipped").collect();
    if !skipped.is_empty() {
        lines.push(String::new());
        lines.push("## Skipped Optional Engines".to_string());
        lines.push(String::new());
        for report in skipped {
            let detail = report
                .checks
                .first()
                .map(|c| c.detail.clone())
                .unwrap_or_else(|| "skipped".to_string());
            lines.push(format!("- {}: {detail}", report.engine));
        }
    }
    lines.push(String::new());
    lines.join("\n")
}

fn escape_markdown_cell(value: &str) -> String {
    value.replace('|', "\\|")
}
