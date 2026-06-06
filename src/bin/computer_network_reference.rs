use std::env;
use std::fs;
use std::path::PathBuf;

use des_engine::des::general::computer_network::{
    build_bottleneck_computer_network_problem, build_default_computer_network_problem,
    run_computer_network_simulation, validate_computer_network_problem, ComputerNetworkProblem,
    NetworkFlowSpec, NetworkLinkSpec, NetworkNodeKind, NetworkNodeSpec, NetworkProtocol,
    NetworkRoutingMetric,
};
use des_engine::des::observability::logger::JsonValue;
use des_engine::des::runners::validate_computer_network::result_to_reference_json;
use serde_json::Value;

#[derive(Debug)]
struct CliError(String);

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Args {
    out: Option<PathBuf>,
    problem: Option<PathBuf>,
    builtin: Option<String>,
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--out PATH] [--problem PATH | --builtin small-enterprise|bottleneck-lab]"
    )
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
            "--out" => {
                parsed.out = Some(PathBuf::from(next_value(
                    program,
                    &mut args,
                    "--out",
                    inline_value,
                )?));
            }
            "--problem" => {
                parsed.problem = Some(PathBuf::from(next_value(
                    program,
                    &mut args,
                    "--problem",
                    inline_value,
                )?));
            }
            "--builtin" => {
                parsed.builtin = Some(next_value(program, &mut args, "--builtin", inline_value)?);
            }
            "-h" | "--help" => return Err(CliError(usage(program))),
            other => return Err(CliError(format!("unknown argument {other}; {}", usage(program)))),
        }
    }
    if parsed.problem.is_some() && parsed.builtin.is_some() {
        return Err(CliError(
            "--problem and --builtin are mutually exclusive".to_string(),
        ));
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

fn builtin_problem(name: Option<&str>) -> Result<ComputerNetworkProblem, CliError> {
    match name.unwrap_or("bottleneck-lab") {
        "small-enterprise" | "default" => Ok(build_default_computer_network_problem()),
        "bottleneck-lab" | "bottleneck" => Ok(build_bottleneck_computer_network_problem()),
        other => Err(CliError(format!("unknown computer-network builtin {other}"))),
    }
}

fn load_problem(args: &Args) -> Result<ComputerNetworkProblem, CliError> {
    if let Some(path) = &args.problem {
        let text =
            fs::read_to_string(path).map_err(|err| CliError(format!("read {}: {err}", path.display())))?;
        let json: Value = serde_json::from_str(&text)
            .map_err(|err| CliError(format!("parse {}: {err}", path.display())))?;
        return problem_from_json(&json);
    }
    builtin_problem(args.builtin.as_deref())
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a serde_json::Map<String, Value>, CliError> {
    value
        .as_object()
        .ok_or_else(|| CliError(format!("{label} must be an object")))
}

fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, CliError> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| CliError(format!("{key} must be an array")))
}

fn string_field(value: &Value, key: &str) -> Result<String, CliError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| CliError(format!("{key} must be a string")))
}

fn optional_string_field(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn f64_field(value: &Value, key: &str) -> Result<f64, CliError> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| CliError(format!("{key} must be numeric")))?;
    if number.is_finite() {
        Ok(number)
    } else {
        Err(CliError(format!("{key} must be finite")))
    }
}

fn optional_f64_field(value: &Value, key: &str) -> Result<Option<f64>, CliError> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => f64_field(value, key).map(Some),
    }
}

fn optional_bool_field(value: &Value, key: &str) -> Result<Option<bool>, CliError> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(raw) => raw
            .as_bool()
            .map(Some)
            .ok_or_else(|| CliError(format!("{key} must be boolean"))),
    }
}

fn optional_usize_field(value: &Value, key: &str) -> Result<Option<usize>, CliError> {
    match optional_u64_field(value, key)? {
        Some(raw) => usize::try_from(raw)
            .map(Some)
            .map_err(|_| CliError(format!("{key} is too large for usize"))),
        None => Ok(None),
    }
}

fn optional_u64_field(value: &Value, key: &str) -> Result<Option<u64>, CliError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    if let Some(number) = raw.as_u64() {
        return Ok(Some(number));
    }
    let number = raw
        .as_f64()
        .ok_or_else(|| CliError(format!("{key} must be a non-negative integer")))?;
    if number.is_finite() && number >= 0.0 && number.fract().abs() < 1e-12 {
        Ok(Some(number as u64))
    } else {
        Err(CliError(format!("{key} must be a non-negative integer")))
    }
}

fn optional_i64_field(value: &Value, key: &str) -> Result<Option<i64>, CliError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    if let Some(number) = raw.as_i64() {
        return Ok(Some(number));
    }
    let number = raw
        .as_f64()
        .ok_or_else(|| CliError(format!("{key} must be an integer")))?;
    if number.is_finite() && number.fract().abs() < 1e-12 {
        Ok(Some(number as i64))
    } else {
        Err(CliError(format!("{key} must be an integer")))
    }
}

fn parse_node_kind(value: &str) -> Result<NetworkNodeKind, CliError> {
    NetworkNodeKind::parse(&value.trim().to_ascii_lowercase())
        .ok_or_else(|| CliError(format!("unknown node kind {value}")))
}

fn parse_protocol(value: &str) -> Result<NetworkProtocol, CliError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "raw" => Ok(NetworkProtocol::Raw),
        "tcp" => Ok(NetworkProtocol::Tcp),
        "udp" => Ok(NetworkProtocol::Udp),
        "http" => Ok(NetworkProtocol::Http),
        other => Err(CliError(format!("unknown protocol {other}"))),
    }
}

fn parse_routing_metric(value: &str) -> Result<NetworkRoutingMetric, CliError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "latency" => Ok(NetworkRoutingMetric::Latency),
        "cost" => Ok(NetworkRoutingMetric::Cost),
        "hop" | "hops" => Ok(NetworkRoutingMetric::Hop),
        other => Err(CliError(format!("unknown routing metric {other}"))),
    }
}

fn problem_from_json(value: &Value) -> Result<ComputerNetworkProblem, CliError> {
    object(value, "problem")?;
    let nodes = array(value, "nodes")?
        .iter()
        .enumerate()
        .map(|(index, node)| {
            object(node, &format!("nodes[{index}]"))?;
            Ok(NetworkNodeSpec {
                id: string_field(node, "id")?,
                kind: parse_node_kind(&string_field(node, "kind")?)?,
                forwarding_rate_pps: optional_f64_field(node, "forwardingRatePps")?,
                queue_limit_packets: optional_usize_field(node, "queueLimitPackets")?,
            })
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    let links = array(value, "links")?
        .iter()
        .enumerate()
        .map(|(index, link)| {
            object(link, &format!("links[{index}]"))?;
            Ok(NetworkLinkSpec {
                id: string_field(link, "id")?,
                from: string_field(link, "from")?,
                to: string_field(link, "to")?,
                bandwidth_mbps: f64_field(link, "bandwidthMbps")?,
                latency_ms: f64_field(link, "latencyMs")?,
                cost_per_mb: optional_f64_field(link, "costPerMb")?,
                queue_limit_packets: optional_usize_field(link, "queueLimitPackets")?,
                bidirectional: optional_bool_field(link, "bidirectional")?,
            })
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    let flows = array(value, "flows")?
        .iter()
        .enumerate()
        .map(|(index, flow)| {
            object(flow, &format!("flows[{index}]"))?;
            Ok(NetworkFlowSpec {
                id: string_field(flow, "id")?,
                source: string_field(flow, "source")?,
                destination: string_field(flow, "destination")?,
                protocol: optional_string_field(flow, "protocol")
                    .as_deref()
                    .map(parse_protocol)
                    .transpose()?,
                rate_pps: f64_field(flow, "ratePps")?,
                packet_size_bytes: f64_field(flow, "packetSizeBytes")?,
                start_ms: optional_f64_field(flow, "startMs")?,
                end_ms: optional_f64_field(flow, "endMs")?,
                max_packets: optional_u64_field(flow, "maxPackets")?,
                ttl_hops: optional_i64_field(flow, "ttlHops")?,
            })
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    Ok(ComputerNetworkProblem {
        nodes,
        links,
        flows,
        duration_ms: f64_field(value, "durationMs")?,
        dt_ms: f64_field(value, "dtMs")?,
        routing_metric: optional_string_field(value, "routingMetric")
            .as_deref()
            .map(parse_routing_metric)
            .transpose()?,
        drain_after_sources_ms: optional_f64_field(value, "drainAfterSourcesMs")?,
        max_packets_in_system: optional_u64_field(value, "maxPacketsInSystem")?,
        sample_every_ms: optional_f64_field(value, "sampleEveryMs")?,
    })
}

fn run(args: &Args) -> Result<JsonValue, CliError> {
    let problem = load_problem(args)?;
    validate_computer_network_problem(&problem)
        .map_err(|err| CliError(format!("invalid computer-network problem: {err}")))?;
    let result = run_computer_network_simulation(&problem);
    Ok(JsonValue::Object(vec![
        ("status".to_string(), JsonValue::String("ok".to_string())),
        ("backend".to_string(), JsonValue::String("rust".to_string())),
        (
            "solver".to_string(),
            JsonValue::String("rust:computer-network".to_string()),
        ),
        ("result".to_string(), result_to_reference_json(&result)),
    ]))
}

fn write_output(path: &PathBuf, output: &JsonValue) -> Result<(), CliError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|err| CliError(format!("create {}: {err}", parent.display())))?;
    }
    fs::write(path, format!("{}\n", output.to_string_pretty(2)))
        .map_err(|err| CliError(format!("write {}: {err}", path.display())))
}

fn compact_stdout(args: &Args, output: &JsonValue) -> JsonValue {
    JsonValue::Object(vec![
        (
            "status".to_string(),
            output
                .get("status")
                .cloned()
                .unwrap_or_else(|| JsonValue::String("ok".to_string())),
        ),
        ("backend".to_string(), JsonValue::String("rust".to_string())),
        (
            "out".to_string(),
            args.out
                .as_ref()
                .map(|path| JsonValue::String(path.display().to_string()))
                .unwrap_or(JsonValue::Null),
        ),
    ])
}

fn error_json(message: String) -> JsonValue {
    JsonValue::Object(vec![
        ("status".to_string(), JsonValue::String("error".to_string())),
        ("backend".to_string(), JsonValue::String("rust".to_string())),
        ("message".to_string(), JsonValue::String(message)),
        ("result".to_string(), JsonValue::Object(Vec::new())),
    ])
}

fn main() {
    let raw_args = env::args().collect::<Vec<_>>();
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "computer_network_reference".to_string());
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
                println!("{output}");
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
    use des_engine::des::runners::validate_computer_network::problem_to_json;

    #[test]
    fn accepts_external_module_args() {
        let args = parse_args(
            "computer_network_reference",
            vec![
                "--out=/tmp/computer-network-reference.json".to_string(),
                "--builtin".to_string(),
                "small-enterprise".to_string(),
            ],
        )
        .expect("parse args");

        assert_eq!(
            args.out,
            Some(PathBuf::from("/tmp/computer-network-reference.json"))
        );
        assert_eq!(args.builtin.as_deref(), Some("small-enterprise"));
    }

    #[test]
    fn parses_validator_problem_json_shape() {
        let original = build_bottleneck_computer_network_problem();
        let parsed: Value =
            serde_json::from_str(&problem_to_json(&original).to_string()).expect("json");
        let decoded = problem_from_json(&parsed).expect("decode problem");

        assert_eq!(decoded.nodes.len(), original.nodes.len());
        assert_eq!(decoded.links.len(), original.links.len());
        assert_eq!(decoded.flows.len(), original.flows.len());
        assert_eq!(decoded.routing_metric, original.routing_metric);
    }

    #[test]
    fn emits_reference_json_shape() {
        let output = run(&Args {
            builtin: Some("bottleneck-lab".to_string()),
            ..Default::default()
        })
        .expect("run");
        let result = output.get("result").expect("result");

        assert_eq!(output.get("status").and_then(JsonValue::as_str), Some("ok"));
        assert_eq!(output.get("backend").and_then(JsonValue::as_str), Some("rust"));
        assert!(result
            .get("generatedPackets")
            .and_then(JsonValue::as_f64)
            .unwrap_or(0.0)
            > 0.0);
        assert!(result.get("flowStats").and_then(JsonValue::as_array).is_some());
        assert!(result.get("linkStats").and_then(JsonValue::as_array).is_some());
        assert!(result
            .get("bottlenecks")
            .and_then(JsonValue::as_array)
            .is_some());
    }
}
