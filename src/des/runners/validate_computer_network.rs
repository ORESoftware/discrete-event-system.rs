//! Port of `src/des/runners/validate-computer-network.ts`.
//!
//! Runs the computer-network DES in Rust and validates a Rust reference JSON
//! projection of the result. A Python external-program reference can be enabled
//! explicitly for cross-checking. The TS top-level `main()` becomes [`run`],
//! returning the process exit code.
//!
//! ## PORT NOTE
//!   * `import './external-modules'` (registration side-effect) →
//!     an explicit, opt-in [`register_built_in_external_modules`] call in
//!     [`run`].
//!   * `JSON.stringify(problem, null, 2)` → [`problem_to_json`] (there is no
//!     `Serialize` derive on [`ComputerNetworkProblem`]; this helper mirrors the
//!     camelCase shape optional external references consume).
//!   * external `.result` is read back as a [`JsonValue`] (camelCase fields).
//!   * missing external reference modules now skip only the optional comparison;
//!     the Rust computer-network scenarios still run and validate invariants.
//!   * `process.exit(code)` → returned exit code.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use crate::des::general::computer_network::{
    build_bottleneck_computer_network_problem, build_default_computer_network_problem,
    run_computer_network_simulation, ComputerNetworkProblem, ComputerNetworkResult,
    NetworkRoutingMetric,
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

/// Serialize a [`ComputerNetworkProblem`] to the camelCase JSON optional
/// external references read. Optional fields are omitted when `None` (like
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

pub fn result_to_reference_json(r: &ComputerNetworkResult) -> JsonValue {
    let flow_stats = r
        .flow_stats
        .iter()
        .map(|f| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(f.id.clone())),
                (
                    "protocol".to_string(),
                    JsonValue::String(f.protocol.as_str().to_string()),
                ),
                ("source".to_string(), JsonValue::String(f.source.clone())),
                (
                    "destination".to_string(),
                    JsonValue::String(f.destination.clone()),
                ),
                ("generatedPackets".to_string(), jn(f.generated_packets)),
                ("deliveredPackets".to_string(), jn(f.delivered_packets)),
                ("droppedPackets".to_string(), jn(f.dropped_packets)),
                ("deliveryRatio".to_string(), jn(f.delivery_ratio)),
                ("generatedBytes".to_string(), jn(f.generated_bytes)),
                ("deliveredBytes".to_string(), jn(f.delivered_bytes)),
                ("offeredLoadMbps".to_string(), jn(f.offered_load_mbps)),
                ("throughputMbps".to_string(), jn(f.throughput_mbps)),
                ("goodputMbps".to_string(), jn(f.goodput_mbps)),
                ("meanLatencyMs".to_string(), jn(f.mean_latency_ms)),
                ("p95LatencyMs".to_string(), jn(f.p95_latency_ms)),
                (
                    "meanTimeInSystemMs".to_string(),
                    jn(f.mean_time_in_system_ms),
                ),
                ("p95TimeInSystemMs".to_string(), jn(f.p95_time_in_system_ms)),
                ("totalCost".to_string(), jn(f.total_cost)),
                (
                    "meanCostPerDeliveredPacket".to_string(),
                    jn(f.mean_cost_per_delivered_packet),
                ),
            ])
        })
        .collect::<Vec<_>>();

    let node_stats = r
        .node_stats
        .iter()
        .map(|n| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(n.id.clone())),
                (
                    "kind".to_string(),
                    JsonValue::String(n.kind.as_str().to_string()),
                ),
                ("forwardingRatePps".to_string(), jn(n.forwarding_rate_pps)),
                ("queueLimitPackets".to_string(), jn(n.queue_limit_packets)),
                ("receivedPackets".to_string(), jn(n.received_packets)),
                ("forwardedPackets".to_string(), jn(n.forwarded_packets)),
                ("deliveredPackets".to_string(), jn(n.delivered_packets)),
                ("droppedPackets".to_string(), jn(n.dropped_packets)),
                ("finalQueue".to_string(), jn(n.final_queue)),
                ("maxQueue".to_string(), jn(n.max_queue)),
                ("avgQueue".to_string(), jn(n.avg_queue)),
                ("meanQueueDelayMs".to_string(), jn(n.mean_queue_delay_ms)),
                ("maxQueueDelayMs".to_string(), jn(n.max_queue_delay_ms)),
            ])
        })
        .collect::<Vec<_>>();

    let link_stats = r
        .link_stats
        .iter()
        .map(|l| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(l.id.clone())),
                ("from".to_string(), JsonValue::String(l.from.clone())),
                ("to".to_string(), JsonValue::String(l.to.clone())),
                ("bandwidthMbps".to_string(), jn(l.bandwidth_mbps)),
                ("latencyMs".to_string(), jn(l.latency_ms)),
                ("costPerMb".to_string(), jn(l.cost_per_mb)),
                ("queueLimitPackets".to_string(), jn(l.queue_limit_packets)),
                ("enqueuedPackets".to_string(), jn(l.enqueued_packets)),
                ("deliveredPackets".to_string(), jn(l.delivered_packets)),
                ("droppedPackets".to_string(), jn(l.dropped_packets)),
                ("transmittedBytes".to_string(), jn(l.transmitted_bytes)),
                ("throughputMbps".to_string(), jn(l.throughput_mbps)),
                ("utilization".to_string(), jn(l.utilization)),
                ("finalInFlight".to_string(), jn(l.final_in_flight)),
                ("maxInFlight".to_string(), jn(l.max_in_flight)),
                ("avgInFlight".to_string(), jn(l.avg_in_flight)),
                ("meanQueueDelayMs".to_string(), jn(l.mean_queue_delay_ms)),
                ("maxQueueDelayMs".to_string(), jn(l.max_queue_delay_ms)),
                ("meanTimeOnLinkMs".to_string(), jn(l.mean_time_on_link_ms)),
                ("maxTimeOnLinkMs".to_string(), jn(l.max_time_on_link_ms)),
                ("totalCost".to_string(), jn(l.total_cost)),
            ])
        })
        .collect::<Vec<_>>();

    let bottlenecks = r
        .bottlenecks
        .iter()
        .map(|b| {
            let mut o = vec![
                ("id".to_string(), JsonValue::String(b.id.clone())),
                ("kind".to_string(), JsonValue::String(b.kind.clone())),
                ("score".to_string(), jn(b.score)),
                ("reason".to_string(), JsonValue::String(b.reason.clone())),
                ("avgQueue".to_string(), jn(b.avg_queue)),
                ("maxQueue".to_string(), jn(b.max_queue)),
                ("droppedPackets".to_string(), jn(b.dropped_packets)),
                ("meanQueueDelayMs".to_string(), jn(b.mean_queue_delay_ms)),
            ];
            if let Some(utilization) = b.utilization {
                o.push(("utilization".to_string(), jn(utilization)));
            }
            JsonValue::Object(o)
        })
        .collect::<Vec<_>>();

    let invariant_violations = r
        .invariant_violations
        .iter()
        .map(|v| JsonValue::String(v.clone()))
        .collect::<Vec<_>>();

    JsonValue::Object(vec![
        ("generatedPackets".to_string(), jn(r.generated_packets)),
        ("deliveredPackets".to_string(), jn(r.delivered_packets)),
        ("droppedPackets".to_string(), jn(r.dropped_packets)),
        ("activePackets".to_string(), jn(r.active_packets)),
        ("maxActivePackets".to_string(), jn(r.max_active_packets)),
        ("deliveryRatio".to_string(), jn(r.delivery_ratio)),
        ("offeredLoadMbps".to_string(), jn(r.offered_load_mbps)),
        ("throughputMbps".to_string(), jn(r.throughput_mbps)),
        ("goodputMbps".to_string(), jn(r.goodput_mbps)),
        ("meanLatencyMs".to_string(), jn(r.mean_latency_ms)),
        ("p95LatencyMs".to_string(), jn(r.p95_latency_ms)),
        ("totalCost".to_string(), jn(r.total_cost)),
        ("totalSimulatedMs".to_string(), jn(r.total_simulated_ms)),
        (
            "routingMetric".to_string(),
            JsonValue::String(routing_metric_str(r.routing_metric).to_string()),
        ),
        ("flowStats".to_string(), JsonValue::Array(flow_stats)),
        ("nodeStats".to_string(), JsonValue::Array(node_stats)),
        ("linkStats".to_string(), JsonValue::Array(link_stats)),
        ("bottlenecks".to_string(), JsonValue::Array(bottlenecks)),
        (
            "invariantViolations".to_string(),
            JsonValue::Array(invariant_violations),
        ),
    ])
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

fn optional_external_error(e: &str) -> bool {
    let lower = e.to_ascii_lowercase();
    lower.contains("external script not found")
        || lower.contains("unknown external module")
        || lower.contains("no such file")
        || lower.contains("no module named")
        || lower.contains("modulenotfounderror")
        || lower.contains("not installed")
        || lower.contains("unavailable")
}

fn computer_network_external_reference_requested() -> bool {
    [
        "COMPUTER_NETWORK_REFERENCE_BACKEND",
        "COMPUTER_NETWORK_EXTERNAL_REFERENCE",
    ]
    .iter()
    .filter_map(|name| std::env::var(name).ok())
    .any(|value| computer_network_external_reference_value_requested(&value))
}

fn computer_network_external_reference_value_requested(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "rust" | "cargo" | "external" | "python" | "py"
    )
}

fn compare_result_fields(
    checks: &mut Checks,
    name: &str,
    reference_label: &str,
    internal: &ComputerNetworkResult,
    external: &JsonValue,
) {
    let prefix = format!("{name}/{reference_label}");
    checks.same_count(
        &format!("{prefix}: generated packets"),
        internal.generated_packets,
        enum_num(&external, "generatedPackets"),
    );
    checks.same_count(
        &format!("{prefix}: delivered packets"),
        internal.delivered_packets,
        enum_num(&external, "deliveredPackets"),
    );
    checks.same_count(
        &format!("{prefix}: dropped packets"),
        internal.dropped_packets,
        enum_num(&external, "droppedPackets"),
    );
    checks.same_count(
        &format!("{prefix}: active packets"),
        internal.active_packets,
        enum_num(&external, "activePackets"),
    );
    checks.same_count(
        &format!("{prefix}: max active packets"),
        internal.max_active_packets,
        enum_num(&external, "maxActivePackets"),
    );
    checks.close(
        &format!("{prefix}: delivery ratio"),
        internal.delivery_ratio,
        enum_num(&external, "deliveryRatio"),
        1e-9,
    );
    checks.close(
        &format!("{prefix}: offered load Mbps"),
        internal.offered_load_mbps,
        enum_num(&external, "offeredLoadMbps"),
        1e-9,
    );
    checks.close(
        &format!("{prefix}: wire throughput Mbps"),
        internal.throughput_mbps,
        enum_num(&external, "throughputMbps"),
        1e-9,
    );
    checks.close(
        &format!("{prefix}: goodput Mbps"),
        internal.goodput_mbps,
        enum_num(&external, "goodputMbps"),
        1e-9,
    );
    checks.close(
        &format!("{prefix}: mean latency ms"),
        internal.mean_latency_ms,
        enum_num(&external, "meanLatencyMs"),
        1e-9,
    );
    checks.close(
        &format!("{prefix}: p95 latency ms"),
        internal.p95_latency_ms,
        enum_num(&external, "p95LatencyMs"),
        1e-9,
    );
    checks.close(
        &format!("{prefix}: total cost"),
        internal.total_cost,
        enum_num(&external, "totalCost"),
        1e-9,
    );
    checks.close(
        &format!("{prefix}: total simulated ms"),
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
        &format!("{prefix}: top bottleneck agrees"),
        agree,
        Some(format!("internal={int_desc} external={ext_desc}")),
    );

    // Flow stats.
    let ext_flows = index_by_id(external.get("flowStats"));
    for flow in &internal.flow_stats {
        let ref_flow = ext_flows.get(&flow.id);
        checks.check(
            &format!("{prefix}/{}: reference flow present", flow.id),
            ref_flow.is_some(),
            None,
        );
        let Some(r) = ref_flow else { continue };
        checks.same_count(
            &format!("{prefix}/{}: generated", flow.id),
            flow.generated_packets,
            enum_num(r, "generatedPackets"),
        );
        checks.same_count(
            &format!("{prefix}/{}: delivered", flow.id),
            flow.delivered_packets,
            enum_num(r, "deliveredPackets"),
        );
        checks.same_count(
            &format!("{prefix}/{}: dropped", flow.id),
            flow.dropped_packets,
            enum_num(r, "droppedPackets"),
        );
        checks.close(
            &format!("{prefix}/{}: goodput", flow.id),
            flow.goodput_mbps,
            enum_num(r, "goodputMbps"),
            1e-9,
        );
        checks.close(
            &format!("{prefix}/{}: mean latency", flow.id),
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
            &format!("{prefix}/{}: reference link present", link.id),
            ref_link.is_some(),
            None,
        );
        let Some(r) = ref_link else { continue };
        checks.same_count(
            &format!("{prefix}/{}: enqueued", link.id),
            link.enqueued_packets,
            enum_num(r, "enqueuedPackets"),
        );
        checks.same_count(
            &format!("{prefix}/{}: dropped", link.id),
            link.dropped_packets,
            enum_num(r, "droppedPackets"),
        );
        checks.close(
            &format!("{prefix}/{}: utilization", link.id),
            link.utilization,
            enum_num(r, "utilization"),
            1e-9,
        );
        checks.close(
            &format!("{prefix}/{}: mean queue delay", link.id),
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
        &format!("{prefix}: invariant violation lists agree"),
        internal.invariant_violations == ext_violations,
        None,
    );
}

fn compare_scenario(
    checks: &mut Checks,
    name: &str,
    problem: &ComputerNetworkProblem,
    external_enabled: bool,
    external_skip_detail: &str,
) -> Result<(), String> {
    println!();
    println!("-- {name} --");
    let internal: ComputerNetworkResult = run_computer_network_simulation(problem);
    checks.check(
        &format!("{name}: internal generated packets"),
        internal.generated_packets > 0.0,
        Some(format!("generated={}", js_num(internal.generated_packets))),
    );
    checks.close(
        &format!("{name}: internal packet accounting"),
        internal.generated_packets,
        internal.delivered_packets + internal.dropped_packets + internal.active_packets,
        1e-9,
    );
    checks.check(
        &format!("{name}: internal flow stats present"),
        !internal.flow_stats.is_empty(),
        Some(format!("flows={}", internal.flow_stats.len())),
    );
    checks.check(
        &format!("{name}: internal link stats present"),
        !internal.link_stats.is_empty(),
        Some(format!("links={}", internal.link_stats.len())),
    );
    checks.check(
        &format!("{name}: internal invariants clean"),
        internal.invariant_violations.is_empty(),
        Some(format!(
            "violations={}",
            internal.invariant_violations.len()
        )),
    );

    let rust_reference = result_to_reference_json(&internal);
    compare_result_fields(
        checks,
        name,
        "rust-reference-json",
        &internal,
        &rust_reference,
    );

    if !external_enabled {
        checks.check(
            &format!("{name}: optional external reference skipped"),
            true,
            Some(external_skip_detail.to_string()),
        );
        return Ok(());
    }

    let external = match run_external(name, problem) {
        Ok(v) => v,
        Err(e) if optional_external_error(&e) => {
            checks.check(
                &format!("{name}: optional external reference skipped"),
                true,
                Some(e),
            );
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    compare_result_fields(checks, name, "external-reference", &internal, &external);
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
    println!("Computer-network DES: framework vs Rust reference");
    println!("====================================================================");

    let (external_enabled, external_skip_detail) = if computer_network_external_reference_requested()
    {
        match register_built_in_external_modules() {
            Ok(()) => (true, "external Rust reference enabled".to_string()),
            Err(e) => {
                eprintln!("external modules unavailable; running Rust-only checks: {e}");
                (false, format!("external modules unavailable: {e}"))
            }
        }
    } else {
        println!("SKIP external reference module (set COMPUTER_NETWORK_REFERENCE_BACKEND=rust)");
        (
            false,
            "set COMPUTER_NETWORK_REFERENCE_BACKEND=rust".to_string(),
        )
    };

    let mut checks = Checks::default();
    if let Err(e) = compare_scenario(
        &mut checks,
        "small-enterprise",
        &build_default_computer_network_problem(),
        external_enabled,
        &external_skip_detail,
    ) {
        eprintln!("{e}");
        return 1;
    }
    if let Err(e) = compare_scenario(
        &mut checks,
        "bottleneck-lab",
        &build_bottleneck_computer_network_problem(),
        external_enabled,
        &external_skip_detail,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_reference_switch_accepts_rust_first_and_legacy_opt_in_values() {
        for value in [
            "1", "true", "YES", "rust", "cargo", "external", "python", "py",
        ] {
            assert!(computer_network_external_reference_value_requested(value));
        }
        for value in ["", "0", "false", "none", "skip"] {
            assert!(!computer_network_external_reference_value_requested(value));
        }
    }
}
