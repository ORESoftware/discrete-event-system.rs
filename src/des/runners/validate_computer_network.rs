//! Port of `src/des/runners/validate-computer-network.ts`.
//!
//! Runs the computer-network DES in Rust and cross-checks the same problem with
//! a dependency-free Python reference, invoked through the sanctioned
//! external-program module system. The TS top-level `main()` becomes [`run`],
//! returning the process exit code.
//!
//! ## PORT NOTE
//!   * `import './external-modules'` (registration side-effect) →
//!     an explicit [`register_built_in_external_modules`] call in [`run`].
//!   * `JSON.stringify(problem, null, 2)` → [`problem_to_json`] (there is no
//!     `Serialize` derive on [`ComputerNetworkProblem`]; this helper mirrors the
//!     camelCase shape the Python reference consumes).
//!   * external `.result` is read back as a [`JsonValue`] (camelCase fields).
//!   * `process.exit(code)` → returned exit code.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use crate::des::general::computer_network::{
    build_default_computer_network_problem, run_computer_network_simulation,
    ComputerNetworkProblem, ComputerNetworkResult, NetworkFlowSpec, NetworkLinkSpec,
    NetworkNodeKind, NetworkNodeSpec, NetworkProtocol, NetworkRoutingMetric,
};
use crate::des::observability::logger::{parse_json, JsonValue};
use crate::des::runners::external_modules::{
    register_built_in_external_modules, COMPUTER_NETWORK_REFERENCE_ID,
};
use crate::des::runners::external_program::{
    repo_root_from_runner, run_external_module, ExternalModuleParams, ParamValue,
};

// -----------------------------------------------------------------------------
// Check accumulation (TS `CheckRow[]` + `check`/`close`/`sameCount`).
// -----------------------------------------------------------------------------

struct CheckRow {
    name: String,
    passed: bool,
    detail: Option<String>,
}

#[derive(Default)]
struct Checks {
    rows: Vec<CheckRow>,
}

impl Checks {
    fn check(&mut self, name: &str, passed: bool, detail: Option<String>) {
        let suffix = detail
            .as_ref()
            .map(|d| format!("  - {d}"))
            .unwrap_or_default();
        println!("  {}  {name}{suffix}", if passed { "PASS" } else { "FAIL" });
        self.rows.push(CheckRow {
            name: name.to_string(),
            passed,
            detail,
        });
    }

    fn close(&mut self, name: &str, actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        self.check(
            name,
            diff <= tol,
            Some(format!(
                "actual={} expected={} diff={} tol={}",
                fmt(actual),
                fmt(expected),
                exp3(diff),
                js_num(tol),
            )),
        );
    }

    fn same_count(&mut self, name: &str, actual: f64, expected: f64) {
        self.check(
            name,
            actual == expected,
            Some(format!(
                "actual={} expected={}",
                js_num(actual),
                js_num(expected)
            )),
        );
    }
}

/// `fmt(n)` — `Number.isFinite(n) ? n.toFixed(12).replace(/\.?0+$/, '') : String(n)`.
fn fmt(n: f64) -> String {
    if n.is_finite() {
        let s = format!("{n:.12}");
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        if trimmed.is_empty() || trimmed == "-" {
            "0".to_string()
        } else {
            trimmed.to_string()
        }
    } else if n.is_nan() {
        "NaN".to_string()
    } else if n > 0.0 {
        "Infinity".to_string()
    } else {
        "-Infinity".to_string()
    }
}

/// `${n}` for a JS number — integers print without a decimal point.
fn js_num(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else if n.is_nan() {
        "NaN".to_string()
    } else if n.is_infinite() {
        if n > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        }
    } else {
        format!("{n}")
    }
}

/// `n.toExponential(3)`.
fn exp3(n: f64) -> String {
    if !n.is_finite() {
        return js_num(n);
    }
    if n == 0.0 {
        return "0.000e+0".to_string();
    }
    let exp = n.abs().log10().floor() as i32;
    let mantissa = n / 10f64.powi(exp);
    let sign = if exp >= 0 { "+" } else { "-" };
    format!("{:.3}e{}{}", mantissa, sign, exp.abs())
}

// -----------------------------------------------------------------------------
// Problem → JSON (mirrors the TS `ComputerNetworkProblem` shape).
// -----------------------------------------------------------------------------

fn jn(v: f64) -> JsonValue {
    JsonValue::Number(v)
}

fn routing_metric_str(m: NetworkRoutingMetric) -> &'static str {
    match m {
        NetworkRoutingMetric::Latency => "latency",
        NetworkRoutingMetric::Cost => "cost",
        NetworkRoutingMetric::Hop => "hop",
    }
}

/// Serialize a [`ComputerNetworkProblem`] to the camelCase JSON the Python
/// reference reads. Optional fields are omitted when `None` (like
/// `JSON.stringify` dropping `undefined`).
pub fn problem_to_json(p: &ComputerNetworkProblem) -> JsonValue {
    let nodes = p
        .nodes
        .iter()
        .map(|n| {
            let mut o: Vec<(String, JsonValue)> = vec![
                ("id".to_string(), JsonValue::String(n.id.clone())),
                (
                    "kind".to_string(),
                    JsonValue::String(n.kind.as_str().to_string()),
                ),
            ];
            if let Some(r) = n.forwarding_rate_pps {
                o.push(("forwardingRatePps".to_string(), jn(r)));
            }
            if let Some(q) = n.queue_limit_packets {
                o.push(("queueLimitPackets".to_string(), jn(q as f64)));
            }
            JsonValue::Object(o)
        })
        .collect::<Vec<_>>();

    let links = p
        .links
        .iter()
        .map(|l| {
            let mut o: Vec<(String, JsonValue)> = vec![
                ("id".to_string(), JsonValue::String(l.id.clone())),
                ("from".to_string(), JsonValue::String(l.from.clone())),
                ("to".to_string(), JsonValue::String(l.to.clone())),
                ("bandwidthMbps".to_string(), jn(l.bandwidth_mbps)),
                ("latencyMs".to_string(), jn(l.latency_ms)),
            ];
            if let Some(c) = l.cost_per_mb {
                o.push(("costPerMb".to_string(), jn(c)));
            }
            if let Some(q) = l.queue_limit_packets {
                o.push(("queueLimitPackets".to_string(), jn(q as f64)));
            }
            if let Some(b) = l.bidirectional {
                o.push(("bidirectional".to_string(), JsonValue::Bool(b)));
            }
            JsonValue::Object(o)
        })
        .collect::<Vec<_>>();

    let flows = p
        .flows
        .iter()
        .map(|f| {
            let mut o: Vec<(String, JsonValue)> = vec![
                ("id".to_string(), JsonValue::String(f.id.clone())),
                ("source".to_string(), JsonValue::String(f.source.clone())),
                (
                    "destination".to_string(),
                    JsonValue::String(f.destination.clone()),
                ),
            ];
            if let Some(proto) = f.protocol {
                o.push((
                    "protocol".to_string(),
                    JsonValue::String(proto.as_str().to_string()),
                ));
            }
            o.push(("ratePps".to_string(), jn(f.rate_pps)));
            o.push(("packetSizeBytes".to_string(), jn(f.packet_size_bytes)));
            if let Some(s) = f.start_ms {
                o.push(("startMs".to_string(), jn(s)));
            }
            if let Some(e) = f.end_ms {
                o.push(("endMs".to_string(), jn(e)));
            }
            if let Some(m) = f.max_packets {
                o.push(("maxPackets".to_string(), jn(m as f64)));
            }
            if let Some(t) = f.ttl_hops {
                o.push(("ttlHops".to_string(), jn(t as f64)));
            }
            JsonValue::Object(o)
        })
        .collect::<Vec<_>>();

    let mut obj: Vec<(String, JsonValue)> = vec![
        ("nodes".to_string(), JsonValue::Array(nodes)),
        ("links".to_string(), JsonValue::Array(links)),
        ("flows".to_string(), JsonValue::Array(flows)),
        ("durationMs".to_string(), jn(p.duration_ms)),
        ("dtMs".to_string(), jn(p.dt_ms)),
    ];
    if let Some(m) = p.routing_metric {
        obj.push((
            "routingMetric".to_string(),
            JsonValue::String(routing_metric_str(m).to_string()),
        ));
    }
    if let Some(d) = p.drain_after_sources_ms {
        obj.push(("drainAfterSourcesMs".to_string(), jn(d)));
    }
    if let Some(m) = p.max_packets_in_system {
        obj.push(("maxPacketsInSystem".to_string(), jn(m as f64)));
    }
    if let Some(s) = p.sample_every_ms {
        obj.push(("sampleEveryMs".to_string(), jn(s)));
    }
    JsonValue::Object(obj)
}

// -----------------------------------------------------------------------------
// PORT NOTE — `buildBottleneckComputerNetworkProblem`.
//
// `build_default_computer_network_problem` exists in the Rust engine, but the
// bottleneck builder is **not yet ported** there. This is a faithful local copy
// of `computer-network.ts::buildBottleneckComputerNetworkProblem`; move it to
// `crate::des::general::computer_network` when that module gains it.
// -----------------------------------------------------------------------------

fn node(id: &str, kind: NetworkNodeKind, fwd: f64, queue: usize) -> NetworkNodeSpec {
    NetworkNodeSpec {
        id: id.to_string(),
        kind,
        forwarding_rate_pps: Some(fwd),
        queue_limit_packets: Some(queue),
    }
}

#[allow(clippy::too_many_arguments)]
fn link(
    id: &str,
    from: &str,
    to: &str,
    bw: f64,
    lat: f64,
    cost: f64,
    queue: usize,
) -> NetworkLinkSpec {
    NetworkLinkSpec {
        id: id.to_string(),
        from: from.to_string(),
        to: to.to_string(),
        bandwidth_mbps: bw,
        latency_ms: lat,
        cost_per_mb: Some(cost),
        queue_limit_packets: Some(queue),
        bidirectional: Some(true),
    }
}

fn flow(
    id: &str,
    source: &str,
    destination: &str,
    protocol: NetworkProtocol,
    rate_pps: f64,
    packet_size_bytes: f64,
    max_packets: u64,
) -> NetworkFlowSpec {
    NetworkFlowSpec {
        id: id.to_string(),
        source: source.to_string(),
        destination: destination.to_string(),
        protocol: Some(protocol),
        rate_pps,
        packet_size_bytes,
        start_ms: None,
        end_ms: None,
        max_packets: Some(max_packets),
        ttl_hops: None,
    }
}

fn build_bottleneck_computer_network_problem() -> ComputerNetworkProblem {
    ComputerNetworkProblem {
        nodes: vec![
            node("web-client", NetworkNodeKind::Host, 6000.0, 512),
            node("telemetry-client", NetworkNodeKind::Host, 6000.0, 512),
            node("edge", NetworkNodeKind::Switch, 12000.0, 1024),
            node("wan-router", NetworkNodeKind::Router, 9000.0, 1024),
            node("api-server", NetworkNodeKind::Host, 9000.0, 1024),
        ],
        links: vec![
            link("web-edge", "web-client", "edge", 100.0, 1.0, 0.001, 256),
            link(
                "telemetry-edge",
                "telemetry-client",
                "edge",
                100.0,
                1.0,
                0.001,
                256,
            ),
            link("edge-wan", "edge", "wan-router", 5.0, 25.0, 0.010, 96),
            link(
                "wan-api",
                "wan-router",
                "api-server",
                100.0,
                4.0,
                0.002,
                256,
            ),
        ],
        flows: vec![
            flow(
                "http-api",
                "web-client",
                "api-server",
                NetworkProtocol::Http,
                900.0,
                1100.0,
                1800,
            ),
            flow(
                "udp-telemetry",
                "telemetry-client",
                "api-server",
                NetworkProtocol::Udp,
                700.0,
                900.0,
                1400,
            ),
            flow(
                "tcp-bulk",
                "web-client",
                "api-server",
                NetworkProtocol::Tcp,
                350.0,
                1400.0,
                700,
            ),
        ],
        duration_ms: 2000.0,
        dt_ms: 1.0,
        routing_metric: Some(NetworkRoutingMetric::Latency),
        drain_after_sources_ms: Some(4000.0),
        max_packets_in_system: Some(10000),
        sample_every_ms: Some(100.0),
    }
}

// -----------------------------------------------------------------------------
// External invocation + JSON field helpers.
// -----------------------------------------------------------------------------

fn out_dir() -> PathBuf {
    repo_root_from_runner()
        .join("out")
        .join("external")
        .join("computer-network")
}

fn enum_num(v: &JsonValue, key: &str) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(f64::NAN)
}

fn write_problem(name: &str, problem: &ComputerNetworkProblem) -> Result<PathBuf, String> {
    let dir = out_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let problem_path = dir.join(format!("{name}-problem.json"));
    std::fs::write(&problem_path, problem_to_json(problem).to_string_pretty(2))
        .map_err(|e| e.to_string())?;
    Ok(problem_path)
}

/// `runExternal(name, problem)` — returns the parsed external `.result` object.
fn run_external(name: &str, problem: &ComputerNetworkProblem) -> Result<JsonValue, String> {
    let problem_path = write_problem(name, problem)?;
    let out = out_dir().join(format!("{name}-reference.json"));

    let mut params: ExternalModuleParams = HashMap::new();
    params.insert(
        "problem".to_string(),
        ParamValue::Str(problem_path.display().to_string()),
    );
    params.insert(
        "out".to_string(),
        ParamValue::Str(out.display().to_string()),
    );
    let ext = run_external_module(COMPUTER_NETWORK_REFERENCE_ID, &params)?;

    let arg_str = ext
        .args
        .iter()
        .map(|a| format!("{a:?}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!("  external command: {} {}", ext.command, arg_str);
    if !ext.stdout.trim().is_empty() {
        println!("{}", ext.stdout.trim());
    }
    if !ext.stderr.trim().is_empty() {
        eprintln!("{}", ext.stderr.trim());
    }
    if ext.status != Some(0) {
        return Err(format!(
            "external computer-network reference exited with status {:?}",
            ext.status
        ));
    }
    let text = std::fs::read_to_string(&out).map_err(|e| e.to_string())?;
    let parsed = parse_json(&text)?;
    Ok(parsed.get("result").cloned().unwrap_or(JsonValue::Null))
}

fn str_field(v: &JsonValue, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(str::to_string)
}

fn compare_scenario(
    checks: &mut Checks,
    name: &str,
    problem: &ComputerNetworkProblem,
) -> Result<(), String> {
    println!();
    println!("-- {name} --");
    let internal: ComputerNetworkResult = run_computer_network_simulation(problem);
    let external = run_external(name, problem)?;

    checks.same_count(
        &format!("{name}: generated packets"),
        internal.generated_packets,
        enum_num(&external, "generatedPackets"),
    );
    checks.same_count(
        &format!("{name}: delivered packets"),
        internal.delivered_packets,
        enum_num(&external, "deliveredPackets"),
    );
    checks.same_count(
        &format!("{name}: dropped packets"),
        internal.dropped_packets,
        enum_num(&external, "droppedPackets"),
    );
    checks.same_count(
        &format!("{name}: active packets"),
        internal.active_packets,
        enum_num(&external, "activePackets"),
    );
    checks.same_count(
        &format!("{name}: max active packets"),
        internal.max_active_packets,
        enum_num(&external, "maxActivePackets"),
    );
    checks.close(
        &format!("{name}: delivery ratio"),
        internal.delivery_ratio,
        enum_num(&external, "deliveryRatio"),
        1e-9,
    );
    checks.close(
        &format!("{name}: offered load Mbps"),
        internal.offered_load_mbps,
        enum_num(&external, "offeredLoadMbps"),
        1e-9,
    );
    checks.close(
        &format!("{name}: wire throughput Mbps"),
        internal.throughput_mbps,
        enum_num(&external, "throughputMbps"),
        1e-9,
    );
    checks.close(
        &format!("{name}: goodput Mbps"),
        internal.goodput_mbps,
        enum_num(&external, "goodputMbps"),
        1e-9,
    );
    checks.close(
        &format!("{name}: mean latency ms"),
        internal.mean_latency_ms,
        enum_num(&external, "meanLatencyMs"),
        1e-9,
    );
    checks.close(
        &format!("{name}: p95 latency ms"),
        internal.p95_latency_ms,
        enum_num(&external, "p95LatencyMs"),
        1e-9,
    );
    checks.close(
        &format!("{name}: total cost"),
        internal.total_cost,
        enum_num(&external, "totalCost"),
        1e-9,
    );
    checks.close(
        &format!("{name}: total simulated ms"),
        internal.total_simulated_ms,
        enum_num(&external, "totalSimulatedMs"),
        1e-9,
    );

    // Top bottleneck.
    let top_internal = internal.bottlenecks.first();
    let ext_bottlenecks = external.get("bottlenecks").and_then(|v| v.as_array());
    let top_external = ext_bottlenecks.and_then(|a| a.first());
    let int_kind = top_internal.map(|b| b.kind.clone());
    let int_id = top_internal.map(|b| b.id.clone());
    let int_reason = top_internal.map(|b| b.reason.clone());
    let ext_kind = top_external.and_then(|b| str_field(b, "kind"));
    let ext_id = top_external.and_then(|b| str_field(b, "id"));
    let ext_reason = top_external.and_then(|b| str_field(b, "reason"));
    let agree = int_kind == ext_kind && int_id == ext_id && int_reason == ext_reason;
    let int_desc = match top_internal {
        Some(b) => format!("{}:{} {}", b.kind, b.id, b.reason),
        None => "none".to_string(),
    };
    let ext_desc = match top_external {
        Some(b) => format!(
            "{}:{} {}",
            str_field(b, "kind").unwrap_or_default(),
            str_field(b, "id").unwrap_or_default(),
            str_field(b, "reason").unwrap_or_default()
        ),
        None => "none".to_string(),
    };
    checks.check(
        &format!("{name}: top bottleneck agrees"),
        agree,
        Some(format!("internal={int_desc} external={ext_desc}")),
    );

    // Flow stats.
    let ext_flows = index_by_id(external.get("flowStats"));
    for flow in &internal.flow_stats {
        let ref_flow = ext_flows.get(&flow.id);
        checks.check(
            &format!("{name}/{}: external flow present", flow.id),
            ref_flow.is_some(),
            None,
        );
        let Some(r) = ref_flow else { continue };
        checks.same_count(
            &format!("{name}/{}: generated", flow.id),
            flow.generated_packets,
            enum_num(r, "generatedPackets"),
        );
        checks.same_count(
            &format!("{name}/{}: delivered", flow.id),
            flow.delivered_packets,
            enum_num(r, "deliveredPackets"),
        );
        checks.same_count(
            &format!("{name}/{}: dropped", flow.id),
            flow.dropped_packets,
            enum_num(r, "droppedPackets"),
        );
        checks.close(
            &format!("{name}/{}: goodput", flow.id),
            flow.goodput_mbps,
            enum_num(r, "goodputMbps"),
            1e-9,
        );
        checks.close(
            &format!("{name}/{}: mean latency", flow.id),
            flow.mean_latency_ms,
            enum_num(r, "meanLatencyMs"),
            1e-9,
        );
    }

    // Link stats.
    let ext_links = index_by_id(external.get("linkStats"));
    for link in &internal.link_stats {
        let ref_link = ext_links.get(&link.id);
        checks.check(
            &format!("{name}/{}: external link present", link.id),
            ref_link.is_some(),
            None,
        );
        let Some(r) = ref_link else { continue };
        checks.same_count(
            &format!("{name}/{}: enqueued", link.id),
            link.enqueued_packets,
            enum_num(r, "enqueuedPackets"),
        );
        checks.same_count(
            &format!("{name}/{}: dropped", link.id),
            link.dropped_packets,
            enum_num(r, "droppedPackets"),
        );
        checks.close(
            &format!("{name}/{}: utilization", link.id),
            link.utilization,
            enum_num(r, "utilization"),
            1e-9,
        );
        checks.close(
            &format!("{name}/{}: mean queue delay", link.id),
            link.mean_queue_delay_ms,
            enum_num(r, "meanQueueDelayMs"),
            1e-9,
        );
    }

    // Invariant-violation lists agree (JSON.stringify equality on string arrays).
    let ext_violations: Vec<String> = external
        .get("invariantViolations")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    checks.check(
        &format!("{name}: invariant violation lists agree"),
        internal.invariant_violations == ext_violations,
        None,
    );
    Ok(())
}

/// `byId(xs)` over a JSON array of `{id, ...}` objects.
fn index_by_id(arr: Option<&JsonValue>) -> HashMap<String, JsonValue> {
    let mut map = HashMap::new();
    if let Some(JsonValue::Array(items)) = arr {
        for item in items {
            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                map.insert(id.to_string(), item.clone());
            }
        }
    }
    map
}

/// `main()` — returns the exit code (0 = all checks pass).
pub fn run() -> i32 {
    if let Err(e) = register_built_in_external_modules() {
        eprintln!("failed to register external modules: {e}");
        return 1;
    }

    println!("Computer-network DES: framework vs external Python reference");
    println!("===========================================================");

    let mut checks = Checks::default();
    if let Err(e) = compare_scenario(
        &mut checks,
        "small-enterprise",
        &build_default_computer_network_problem(),
    ) {
        eprintln!("{e}");
        return 1;
    }
    if let Err(e) = compare_scenario(
        &mut checks,
        "bottleneck-lab",
        &build_bottleneck_computer_network_problem(),
    ) {
        eprintln!("{e}");
        return 1;
    }

    println!();
    println!("========================================");
    let passed = checks.rows.iter().filter(|c| c.passed).count();
    println!(
        "validate-computer-network: {passed}/{} checks passed.",
        checks.rows.len()
    );
    if passed < checks.rows.len() {
        println!("FAILED:");
        for c in &checks.rows {
            if !c.passed {
                let detail = c
                    .detail
                    .as_ref()
                    .map(|d| format!(": {d}"))
                    .unwrap_or_default();
                println!("  - {}{detail}", c.name);
            }
        }
        return 1;
    }
    0
}
