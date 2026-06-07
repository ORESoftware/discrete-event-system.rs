use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use des_engine::des::general::network_flow::{
    TrafficLane, TrafficNetwork, TrafficNode, TrafficNodeKind, TrafficParams, TrafficScheduledTrip,
    TrafficSink, TrafficSource,
};
use des_engine::des::general::smart_traffic_flow::{run_smart_traffic_flow, SmartTrafficParams};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug)]
struct CliError(String);

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Args {
    problem: Option<PathBuf>,
    out: Option<PathBuf>,
    collision_action: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SharedTrafficInput {
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    model: Option<String>,
    params: RawTrafficParams,
    network: RawTrafficNetwork,
    #[serde(default)]
    trips: Vec<RawTrafficTrip>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrafficParams {
    duration_sec: f64,
    dt_sec: f64,
    seed: f64,
    #[serde(default)]
    actor_shuffle_seed: Option<f64>,
    #[serde(default)]
    max_cars: Option<f64>,
    #[serde(default)]
    smart_car_pool_size: Option<f64>,
    #[serde(default)]
    car_length_m: Option<f64>,
    #[serde(default)]
    car_width_m: Option<f64>,
    #[serde(default)]
    lane_width_m: Option<f64>,
    #[serde(default)]
    min_gap_m: Option<f64>,
    #[serde(default)]
    max_accel_mps2: Option<f64>,
    #[serde(default)]
    max_decel_mps2: Option<f64>,
    #[serde(default)]
    max_jerk_mps3: Option<f64>,
    #[serde(default)]
    reaction_time_sec: Option<f64>,
    #[serde(default)]
    time_headway_sec: Option<f64>,
    #[serde(default)]
    grid_cell_size_m: Option<f64>,
    #[serde(default)]
    accident_risk_scale: Option<f64>,
    #[serde(default)]
    accident_probability: Option<f64>,
    #[serde(default)]
    distance_preference_spread: Option<f64>,
    #[serde(default)]
    start_preference_spread: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrafficNetwork {
    nodes: Vec<RawTrafficNode>,
    lanes: Vec<RawTrafficLane>,
    #[serde(default)]
    sources: Vec<RawTrafficSource>,
    #[serde(default)]
    sinks: Vec<RawTrafficSink>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrafficNode {
    id: String,
    #[serde(default)]
    kind: Option<String>,
    x: f64,
    y: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrafficLane {
    id: String,
    from: String,
    to: String,
    length_m: f64,
    speed_limit_mps: f64,
    #[serde(default)]
    capacity: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrafficSource {
    id: String,
    node_id: String,
    rate_per_min: f64,
    #[serde(default)]
    destination_sink_ids: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrafficSink {
    id: String,
    node_id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrafficTrip {
    #[serde(default)]
    id: Option<String>,
    depart_sec: f64,
    source_id: String,
    destination_sink_id: String,
    #[serde(default)]
    route: Vec<String>,
}

fn usage(program: &str) -> String {
    format!("usage: {program} --problem PATH [--out PATH] [--collision-action warn|ignore|fail]")
}

fn parse_args<I>(program: &str, raw_args: I) -> Result<Args, CliError>
where
    I: IntoIterator<Item = String>,
{
    let mut args = raw_args.into_iter();
    let mut parsed = Args::default();
    while let Some(raw) = args.next() {
        let (key, inline_value) = raw
            .split_once('=')
            .map(|(key, value)| (key.to_string(), Some(value.to_string())))
            .unwrap_or((raw, None));
        match key.as_str() {
            "--problem" => {
                parsed.problem = Some(PathBuf::from(next_value(
                    program,
                    &mut args,
                    "--problem",
                    inline_value,
                )?));
            }
            "--out" => {
                parsed.out = Some(PathBuf::from(next_value(
                    program,
                    &mut args,
                    "--out",
                    inline_value,
                )?));
            }
            "--collision-action" => {
                parsed.collision_action = Some(next_value(
                    program,
                    &mut args,
                    "--collision-action",
                    inline_value,
                )?);
            }
            "-h" | "--help" => return Err(CliError(usage(program))),
            other => {
                return Err(CliError(format!(
                    "unknown argument {other}; {}",
                    usage(program)
                )));
            }
        }
    }
    if parsed.problem.is_none() {
        return Err(CliError(format!(
            "--problem is required; {}",
            usage(program)
        )));
    }
    Ok(parsed)
}

fn next_value<I>(
    program: &str,
    args: &mut I,
    flag: &str,
    inline_value: Option<String>,
) -> Result<String, CliError>
where
    I: Iterator<Item = String>,
{
    inline_value
        .or_else(|| args.next())
        .ok_or_else(|| CliError(format!("{flag} requires a value; {}", usage(program))))
}

fn load_input(path: &Path) -> Result<SharedTrafficInput, CliError> {
    let text = fs::read_to_string(path)
        .map_err(|err| CliError(format!("read {}: {err}", path.display())))?;
    let input = serde_json::from_str::<SharedTrafficInput>(&text)
        .map_err(|err| CliError(format!("parse {}: {err}", path.display())))?;
    validate_input_header(&input)?;
    Ok(input)
}

fn validate_input_header(input: &SharedTrafficInput) -> Result<(), CliError> {
    if let Some(schema) = input.schema.as_deref() {
        if schema != "des/shared-traffic-source-sink/v1" {
            return Err(CliError(format!("unsupported traffic schema {schema}")));
        }
    }
    if let Some(model) = input.model.as_deref() {
        if model != "smart-traffic-flow" {
            return Err(CliError(format!("unsupported traffic model {model}")));
        }
    }
    Ok(())
}

fn ensure_finite(value: f64, label: &str) -> Result<f64, CliError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CliError(format!("{label} must be finite")))
    }
}

fn finite_option(value: Option<f64>, label: &str) -> Result<Option<f64>, CliError> {
    value.map(|number| ensure_finite(number, label)).transpose()
}

fn usize_option_from_f64(value: Option<f64>, label: &str) -> Result<Option<usize>, CliError> {
    value
        .map(|number| {
            let number = ensure_finite(number, label)?;
            if number < 0.0 {
                return Err(CliError(format!("{label} must be non-negative")));
            }
            Ok(number.max(1.0) as usize)
        })
        .transpose()
}

fn traffic_node_kind(
    raw: &RawTrafficNode,
    source_nodes: &HashSet<String>,
    sink_nodes: &HashSet<String>,
) -> Result<TrafficNodeKind, CliError> {
    if let Some(kind) = raw.kind.as_deref() {
        return match kind.to_ascii_lowercase().as_str() {
            "source" => Ok(TrafficNodeKind::Source),
            "intersection" => Ok(TrafficNodeKind::Intersection),
            "sink" => Ok(TrafficNodeKind::Sink),
            other => Err(CliError(format!(
                "unsupported node kind {other} for {}",
                raw.id
            ))),
        };
    }
    let is_source = source_nodes.contains(&raw.id);
    let is_sink = sink_nodes.contains(&raw.id);
    match (is_source, is_sink) {
        (true, false) => Ok(TrafficNodeKind::Source),
        (false, true) => Ok(TrafficNodeKind::Sink),
        (false, false) => Ok(TrafficNodeKind::Intersection),
        (true, true) => Err(CliError(format!(
            "node {} cannot be both source and sink",
            raw.id
        ))),
    }
}

fn network_from_raw(
    raw: RawTrafficNetwork,
    inferred_destination_sink_ids: &HashMap<String, Vec<String>>,
) -> Result<TrafficNetwork, CliError> {
    let source_nodes = raw
        .sources
        .iter()
        .map(|source| source.node_id.clone())
        .collect::<HashSet<_>>();
    let sink_nodes = raw
        .sinks
        .iter()
        .map(|sink| sink.node_id.clone())
        .collect::<HashSet<_>>();
    let nodes = raw
        .nodes
        .iter()
        .map(|node| {
            Ok(TrafficNode {
                id: node.id.clone(),
                kind: traffic_node_kind(node, &source_nodes, &sink_nodes)?,
                x: ensure_finite(node.x, &format!("node {} x", node.id))?,
                y: ensure_finite(node.y, &format!("node {} y", node.id))?,
            })
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    let lanes = raw
        .lanes
        .into_iter()
        .map(|lane| {
            Ok(TrafficLane {
                id: lane.id.clone(),
                from: lane.from,
                to: lane.to,
                length_m: ensure_finite(lane.length_m, &format!("lane {} lengthM", lane.id))?,
                speed_limit_mps: ensure_finite(
                    lane.speed_limit_mps,
                    &format!("lane {} speedLimitMps", lane.id),
                )?,
                capacity: lane.capacity,
            })
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    let sources = raw
        .sources
        .into_iter()
        .map(|source| {
            let destination_sink_ids = source
                .destination_sink_ids
                .or_else(|| inferred_destination_sink_ids.get(&source.id).cloned());
            Ok(TrafficSource {
                id: source.id.clone(),
                node_id: source.node_id,
                rate_per_min: ensure_finite(
                    source.rate_per_min,
                    &format!("source {} ratePerMin", source.id),
                )?,
                destination_sink_ids,
            })
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    let sinks = raw
        .sinks
        .into_iter()
        .map(|sink| TrafficSink {
            id: sink.id,
            node_id: sink.node_id,
        })
        .collect::<Vec<_>>();
    Ok(TrafficNetwork {
        nodes,
        lanes,
        signals: None,
        sources,
        sinks,
    })
}

fn inferred_destination_sinks_by_source(trips: &[RawTrafficTrip]) -> HashMap<String, Vec<String>> {
    let mut by_source = HashMap::<String, Vec<String>>::new();
    for trip in trips {
        let destinations = by_source.entry(trip.source_id.clone()).or_default();
        if !destinations.contains(&trip.destination_sink_id) {
            destinations.push(trip.destination_sink_id.clone());
        }
    }
    by_source
}

fn trips_from_raw(raw: Vec<RawTrafficTrip>) -> Result<Vec<TrafficScheduledTrip>, CliError> {
    raw.into_iter()
        .map(|trip| {
            let _ = trip.id;
            let _ = trip.route;
            Ok(TrafficScheduledTrip {
                depart_sec: ensure_finite(trip.depart_sec, "trip departSec")?,
                source_id: trip.source_id,
                destination_sink_id: trip.destination_sink_id,
            })
        })
        .collect()
}

fn params_from_raw(
    raw: RawTrafficParams,
    network: TrafficNetwork,
    trips: Vec<TrafficScheduledTrip>,
) -> Result<SmartTrafficParams, CliError> {
    Ok(SmartTrafficParams {
        base: TrafficParams {
            builtin: None,
            network: Some(network),
            duration_sec: ensure_finite(raw.duration_sec, "durationSec")?,
            dt_sec: ensure_finite(raw.dt_sec, "dtSec")?,
            seed: ensure_finite(raw.seed, "seed")?,
            max_cars: usize_option_from_f64(raw.max_cars, "maxCars")?.unwrap_or(100),
            car_length_m: finite_option(raw.car_length_m, "carLengthM")?,
            car_width_m: finite_option(raw.car_width_m, "carWidthM")?,
            lane_width_m: finite_option(raw.lane_width_m, "laneWidthM")?,
            min_gap_m: finite_option(raw.min_gap_m, "minGapM")?,
            max_accel_mps2: finite_option(raw.max_accel_mps2, "maxAccelMps2")?,
            max_decel_mps2: finite_option(raw.max_decel_mps2, "maxDecelMps2")?,
            max_jerk_mps3: finite_option(raw.max_jerk_mps3, "maxJerkMps3")?,
            reaction_time_sec: finite_option(raw.reaction_time_sec, "reactionTimeSec")?,
            time_headway_sec: finite_option(raw.time_headway_sec, "timeHeadwaySec")?,
            grid_cell_size_m: finite_option(raw.grid_cell_size_m, "gridCellSizeM")?,
            grid_look_ahead_m: None,
            spawn_rate_multiplier: Some(0.0),
            scheduled_trips: Some(trips),
        },
        smart_car_pool_size: usize_option_from_f64(raw.smart_car_pool_size, "smartCarPoolSize")?,
        actor_shuffle_seed: finite_option(raw.actor_shuffle_seed, "actorShuffleSeed")?,
        accident_risk_scale: finite_option(raw.accident_risk_scale, "accidentRiskScale")?,
        accident_probability: finite_option(raw.accident_probability, "accidentProbability")?,
        accident_accel_boost_mps2: None,
        accident_fault_duration_sec: None,
        distance_preference_spread: finite_option(
            raw.distance_preference_spread,
            "distancePreferenceSpread",
        )?,
        start_preference_spread: finite_option(
            raw.start_preference_spread,
            "startPreferenceSpread",
        )?,
        accident_flash_seconds: None,
    })
}

fn reference_output(input: SharedTrafficInput) -> Result<Value, CliError> {
    let inferred_destination_sink_ids = inferred_destination_sinks_by_source(&input.trips);
    let network = network_from_raw(input.network, &inferred_destination_sink_ids)?;
    let trips = trips_from_raw(input.trips)?;
    let generated_demand = trips.len();
    let params = params_from_raw(input.params, network, trips)?;
    let result = run_smart_traffic_flow(params, None);
    Ok(json!({
        "status": "ok",
        "backend": "rust",
        "simulator": "rust:traffic-fel-reference",
        "result": {
            "generatedDemand": generated_demand,
            "departed": result.entered,
            "arrived": result.exited,
            "dropped": result.dropped,
            "crashed": result.crashed,
            "activeAtEnd": result.final_cars.len(),
            "maxActiveCars": result.max_active_cars,
            "meanTravelTimeSec": result.mean_travel_time_sec,
            "meanSpeedMps": result.mean_speed_mps,
            "notes": [
                "Rust traffic FEL reference ran the crate-native smart traffic simulator",
                "Random source spawning disabled so scheduled trips are the shared demand"
            ]
        }
    }))
}

fn run(args: &Args) -> Result<Value, CliError> {
    let problem = args
        .problem
        .as_deref()
        .ok_or_else(|| CliError("--problem is required".to_string()))?;
    let input = load_input(problem)?;
    reference_output(input)
}

fn write_output(path: &Path, output: &Value) -> Result<(), CliError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|err| CliError(format!("create {}: {err}", parent.display())))?;
    }
    fs::write(path, format!("{}\n", pretty_json(output)))
        .map_err(|err| CliError(format!("write {}: {err}", path.display())))
}

fn pretty_json(output: &Value) -> String {
    serde_json::to_string_pretty(output).unwrap_or_else(|_| output.to_string())
}

fn compact_stdout(args: &Args, output: &Value) -> Value {
    json!({
        "status": output.get("status").cloned().unwrap_or_else(|| json!("ok")),
        "backend": "rust",
        "out": args.out.as_ref().map(|path| path.display().to_string()),
    })
}

fn error_json(message: String) -> Value {
    json!({
        "status": "error",
        "backend": "rust",
        "simulator": "rust:traffic-fel-reference",
        "message": message,
        "result": {},
    })
}

fn main() {
    let raw_args = env::args().collect::<Vec<_>>();
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "traffic_fel_reference".to_string());
    if raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage(&program));
        return;
    }
    let args = match parse_args(&program, raw_args.into_iter().skip(1)) {
        Ok(args) => args,
        Err(err) => {
            println!("{}", error_json(err.to_string()));
            std::process::exit(1);
        }
    };
    match run(&args) {
        Ok(output) => {
            if let Some(path) = &args.out {
                if let Err(err) = write_output(path, &output) {
                    println!("{}", error_json(err.to_string()));
                    std::process::exit(1);
                }
                println!("{}", compact_stdout(&args, &output));
            } else {
                println!("{}", pretty_json(&output));
            }
        }
        Err(err) => {
            println!("{}", error_json(err.to_string()));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TRAFFIC_FEL_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, previous }
        }

        fn clear(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => std::env::set_var(self.key, previous),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn traffic_reference_python_off_guards() -> Vec<EnvVarGuard> {
        vec![
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-traffic-fel"),
            EnvVarGuard::set("PYTHON", "/definitely/not-python-for-traffic-fel"),
            EnvVarGuard::set("SUMO_HOME", "/definitely/not-sumo-for-traffic-fel"),
            EnvVarGuard::clear("TRAFFIC_FEL_REFERENCE_FORCE_PYTHON"),
            EnvVarGuard::clear("SUMO_REFERENCE_FORCE_EXTERNAL"),
            EnvVarGuard::clear("ORES_EXTERNAL_REFERENCE_FORCE_PYTHON"),
        ]
    }

    fn minimal_input() -> SharedTrafficInput {
        serde_json::from_value(json!({
            "schema": "des/shared-traffic-source-sink/v1",
            "model": "smart-traffic-flow",
            "params": {
                "durationSec": 5.0,
                "dtSec": 0.5,
                "seed": 3.0,
                "actorShuffleSeed": 17.0,
                "maxCars": 4.0,
                "smartCarPoolSize": 4.0,
                "carLengthM": 4.8,
                "carWidthM": 1.8,
                "laneWidthM": 3.7,
                "minGapM": 1.0,
                "maxAccelMps2": 3.0,
                "maxDecelMps2": 5.0,
                "maxJerkMps3": 8.0,
                "reactionTimeSec": 0.5,
                "timeHeadwaySec": 0.8,
                "gridCellSizeM": 1.0,
                "accidentRiskScale": 0.0,
                "accidentProbability": 0.0,
                "distancePreferenceSpread": 0.0,
                "startPreferenceSpread": 0.0
            },
            "network": {
                "nodes": [
                    {"id": "A", "x": 0.0, "y": 0.0},
                    {"id": "B", "x": 1.0, "y": 0.0}
                ],
                "lanes": [
                    {"id": "A-B", "from": "A", "to": "B", "lengthM": 10.0, "speedLimitMps": 10.0}
                ],
                "sources": [
                    {"id": "src", "nodeId": "A", "ratePerMin": 0.0}
                ],
                "sinks": [
                    {"id": "sink", "nodeId": "B"}
                ]
            },
            "trips": [
                {"id": "trip-1", "departSec": 0.0, "sourceId": "src", "destinationSinkId": "sink", "route": ["A-B"]}
            ]
        }))
        .expect("minimal input")
    }

    #[test]
    fn parse_args_accepts_problem_out_and_collision_action() {
        let args = parse_args(
            "traffic_fel_reference",
            [
                "--problem=traffic.json".to_string(),
                "--out".to_string(),
                "out.json".to_string(),
                "--collision-action".to_string(),
                "warn".to_string(),
            ],
        )
        .expect("args");

        assert_eq!(args.problem, Some(PathBuf::from("traffic.json")));
        assert_eq!(args.out, Some(PathBuf::from("out.json")));
        assert_eq!(args.collision_action, Some("warn".to_string()));
    }

    #[test]
    fn network_infers_node_kinds_from_sources_and_sinks() {
        let input = minimal_input();
        let inferred_destination_sink_ids = inferred_destination_sinks_by_source(&input.trips);
        let network =
            network_from_raw(input.network, &inferred_destination_sink_ids).expect("network");

        assert_eq!(network.nodes[0].kind, TrafficNodeKind::Source);
        assert_eq!(network.nodes[1].kind, TrafficNodeKind::Sink);
        assert_eq!(
            network.sources[0].destination_sink_ids,
            Some(vec!["sink".to_string()])
        );
    }

    #[test]
    fn runs_minimal_shared_traffic_input() {
        let output = reference_output(minimal_input()).expect("reference output");
        let result = output.get("result").expect("result");

        assert_eq!(output.get("status").and_then(Value::as_str), Some("ok"));
        assert_eq!(
            result.get("generatedDemand").and_then(Value::as_u64),
            Some(1)
        );
        assert!(result
            .get("meanTravelTimeSec")
            .and_then(Value::as_f64)
            .is_some_and(f64::is_finite));
        assert!(result
            .get("meanSpeedMps")
            .and_then(Value::as_f64)
            .is_some_and(f64::is_finite));
    }

    #[test]
    fn shared_traffic_reference_ignores_external_env_and_runs_in_rust() {
        let _env_lock = TRAFFIC_FEL_REFERENCE_ENV_LOCK
            .lock()
            .expect("traffic FEL env lock");
        let _guards = traffic_reference_python_off_guards();

        let output = reference_output(minimal_input()).expect("reference output");
        let result = output.get("result").expect("result");

        assert_eq!(output.get("status").and_then(Value::as_str), Some("ok"));
        assert_eq!(output.get("backend").and_then(Value::as_str), Some("rust"));
        assert_eq!(
            output.get("simulator").and_then(Value::as_str),
            Some("rust:traffic-fel-reference")
        );
        assert_eq!(
            result.get("generatedDemand").and_then(Value::as_u64),
            Some(1)
        );
        assert!(result
            .get("notes")
            .and_then(Value::as_array)
            .is_some_and(|notes| notes.iter().any(|note| note
                .as_str()
                .is_some_and(|text| text.contains("crate-native smart traffic simulator")))));
    }

    #[test]
    fn force_external_python_env_still_runs_rust_traffic_reference() {
        let _env_lock = TRAFFIC_FEL_REFERENCE_ENV_LOCK
            .lock()
            .expect("traffic FEL env lock");
        let _python_bin_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-traffic-fel");
        let _python_guard = EnvVarGuard::set("PYTHON", "/definitely/not-python-for-traffic-fel");
        let _sumo_guard = EnvVarGuard::set("SUMO_HOME", "/definitely/not-sumo-for-traffic-fel");
        let _force_python_guard = EnvVarGuard::set("TRAFFIC_FEL_REFERENCE_FORCE_PYTHON", "1");
        let _force_sumo_guard = EnvVarGuard::set("SUMO_REFERENCE_FORCE_EXTERNAL", "1");
        let _global_force_guard = EnvVarGuard::set("ORES_EXTERNAL_REFERENCE_FORCE_PYTHON", "1");

        let output = reference_output(minimal_input()).expect("reference output");
        let result = output.get("result").expect("result");

        assert_eq!(output.get("status").and_then(Value::as_str), Some("ok"));
        assert_eq!(output.get("backend").and_then(Value::as_str), Some("rust"));
        assert_eq!(
            output.get("simulator").and_then(Value::as_str),
            Some("rust:traffic-fel-reference")
        );
        assert_eq!(
            result.get("generatedDemand").and_then(Value::as_u64),
            Some(1)
        );
        assert!(result
            .get("meanTravelTimeSec")
            .and_then(Value::as_f64)
            .is_some_and(f64::is_finite));
        assert!(!serde_json::to_string(&output)
            .expect("traffic reference output json")
            .contains("/definitely/not-python-for-traffic-fel"));
    }
}
