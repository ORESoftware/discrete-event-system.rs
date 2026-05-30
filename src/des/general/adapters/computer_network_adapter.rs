//! Port of `src/des/general/adapters/computer-network-adapter.ts`
//! (module `des::general::adapters::computer_network_adapter`).
//!
//! JSON adapter for the packet-switched computer-network DES.
//!
//! ## Conversion notes
//!
//!   * `builtin: 'small-enterprise' | 'bottleneck-lab'` -> [`NetworkBuiltin`].
//!   * `problem?: ComputerNetworkProblem` reuses the engine
//!     [`ComputerNetworkProblem`] struct directly (the integrator deserialises
//!     JSON into it), so the nested node/link/flow param structs are not
//!     re-declared.
//!   * `params.problem ?? problemFromBuiltin(params.builtin)` ->
//!     `Option::unwrap_or_else`; `problemFromBuiltin` is a `match` defaulting to
//!     `small-enterprise`.
//!   * `fmt(x)` = `Number.isFinite(x) ? x.toFixed(3) : 'n/a'`.
//!   * The CSV emits one union schema across flow / link / node rows with blank
//!     columns, exactly as the TS does.
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet; the
//! adapter is exposed via [`adapter()`].
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/scenes/computer-network-scene`) is not ported, so `animate` is a
//! no-op here. The integrator should wire it once those modules exist.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::computer_network::{
    build_bottleneck_computer_network_problem, build_default_computer_network_problem,
    run_computer_network_simulation, ComputerNetworkProblem, ComputerNetworkResult,
    NetworkRoutingMetric,
};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};

/// `String(v)` for a JS number.
fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        let s = v.to_string();
        if s == "-0" { "0".to_string() } else { s }
    }
}

/// `fmt(x)` — `Number.isFinite(x) ? x.toFixed(3) : 'n/a'`.
fn fmt(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.3}")
    } else {
        "n/a".to_string()
    }
}

/// `builtin: 'small-enterprise' | 'bottleneck-lab'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkBuiltin {
    SmallEnterprise,
    BottleneckLab,
}

/// `interface ComputerNetworkParams`.
#[derive(Clone, Debug, Default)]
pub struct ComputerNetworkParams {
    pub builtin: Option<NetworkBuiltin>,
    pub problem: Option<ComputerNetworkProblem>,
}

fn routing_metric_str(m: NetworkRoutingMetric) -> &'static str {
    match m {
        NetworkRoutingMetric::Latency => "latency",
        NetworkRoutingMetric::Cost => "cost",
        NetworkRoutingMetric::Hop => "hop",
    }
}

/// `function problemFromBuiltin(builtin)`.
fn problem_from_builtin(builtin: Option<NetworkBuiltin>) -> ComputerNetworkProblem {
    match builtin.unwrap_or(NetworkBuiltin::SmallEnterprise) {
        NetworkBuiltin::BottleneckLab => build_bottleneck_computer_network_problem(),
        NetworkBuiltin::SmallEnterprise => build_default_computer_network_problem(),
    }
}

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default: None, description: None }
}

fn string_field() -> ParamSchema {
    ParamSchema::String { allowed: None, default: None, description: None }
}

fn str_enum_default(allowed: &[&str], default: &str) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: Some(default.to_string()),
        description: None,
    }
}

fn array(items: ParamSchema, min_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length: None, description: None }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>, description: Option<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: description.map(|s| s.to_string()),
    }
}

fn network_node_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("kind", ParamSchema::String {
                allowed: Some(vec!["host".to_string(), "router".to_string(), "switch".to_string()]),
                default: None,
                description: None,
            }),
            ("forwardingRatePps", num(Some(1e-9), None, None)),
            ("queueLimitPackets", num(Some(1.0), None, Some(true))),
        ],
        vec!["id", "kind"],
        None,
    )
}

fn network_link_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("from", string_field()),
            ("to", string_field()),
            ("bandwidthMbps", num(Some(1e-9), None, None)),
            ("latencyMs", num(Some(0.0), None, None)),
            ("costPerMb", num(Some(0.0), None, None)),
            ("queueLimitPackets", num(Some(1.0), None, Some(true))),
            ("bidirectional", ParamSchema::Boolean { default: Some(false), description: None }),
        ],
        vec!["id", "from", "to", "bandwidthMbps", "latencyMs"],
        None,
    )
}

fn network_flow_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("source", string_field()),
            ("destination", string_field()),
            ("protocol", str_enum_default(&["raw", "tcp", "udp", "http"], "raw")),
            ("ratePps", num(Some(0.0), None, None)),
            ("packetSizeBytes", num(Some(1.0), None, Some(true))),
            ("startMs", num(Some(0.0), None, None)),
            ("endMs", num(Some(0.0), None, None)),
            ("maxPackets", num(Some(0.0), None, Some(true))),
            ("ttlHops", num(Some(1.0), None, Some(true))),
        ],
        vec!["id", "source", "destination", "ratePps", "packetSizeBytes"],
        None,
    )
}

fn computer_network_problem_schema() -> ParamSchema {
    obj(
        vec![
            ("nodes", array(network_node_schema(), Some(2))),
            ("links", array(network_link_schema(), Some(1))),
            ("flows", array(network_flow_schema(), Some(1))),
            ("durationMs", num(Some(1e-9), None, None)),
            ("dtMs", num(Some(1e-9), None, None)),
            ("routingMetric", str_enum_default(&["latency", "cost", "hop"], "latency")),
            ("drainAfterSourcesMs", num(Some(0.0), None, None)),
            ("maxPacketsInSystem", num(Some(1.0), None, Some(true))),
            ("sampleEveryMs", num(Some(1e-9), None, None)),
        ],
        vec!["nodes", "links", "flows", "durationMs", "dtMs"],
        Some("Packet-switched topology with hosts/routers/switches, links, and traffic flows."),
    )
}

/// `const computerNetworkSchema`.
pub fn computer_network_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            (
                "builtin".to_string(),
                str_enum_default(&["small-enterprise", "bottleneck-lab"], "small-enterprise"),
            ),
            ("problem".to_string(), computer_network_problem_schema()),
        ],
        required: Some(vec![]),
        description: Some(
            "Computer-network DES with stationary host/router/switch/link entities and moving packet entities."
                .to_string(),
        ),
    }
}

/// `const adapter`.
pub struct ComputerNetworkAdapter;

/// Construct the adapter (see the module PORT NOTE on registration).
pub fn adapter() -> ComputerNetworkAdapter {
    ComputerNetworkAdapter
}

impl DESModelRegistration<ComputerNetworkParams, ComputerNetworkResult> for ComputerNetworkAdapter {
    fn id(&self) -> &str {
        "computer-network"
    }

    fn description(&self) -> &str {
        "Packet-switched computer-network DES from JSON topology, with latency, throughput, drops, and cost stats."
    }

    fn schema(&self) -> ParamSchema {
        computer_network_schema()
    }

    fn run(&self, params: ComputerNetworkParams, _runtime: &DESRuntimeConfig) -> ComputerNetworkResult {
        let builtin = params.builtin;
        let problem = params.problem.unwrap_or_else(|| problem_from_builtin(builtin));
        run_computer_network_simulation(&problem)
    }

    fn summarize(&self, result: &ComputerNetworkResult, _params: &ComputerNetworkParams) -> String {
        let top = result.bottlenecks.first();
        let top_str = match top {
            Some(b) => format!("{}:{} ({})", b.kind, b.id, b.reason),
            None => "none".to_string(),
        };
        let invariants = if result.invariant_violations.is_empty() {
            "ok".to_string()
        } else {
            format!("{} violations", result.invariant_violations.len())
        };
        [
            "COMPUTER-NETWORK DES".to_string(),
            "--------------------".to_string(),
            format!("  Routing metric:   {}", routing_metric_str(result.routing_metric)),
            format!("  Generated:        {}", js_number(result.generated_packets)),
            format!("  Delivered:        {}", js_number(result.delivered_packets)),
            format!("  Dropped:          {}", js_number(result.dropped_packets)),
            format!("  Active at stop:   {}", js_number(result.active_packets)),
            format!("  Max active:       {}", js_number(result.max_active_packets)),
            format!("  Delivery ratio:   {:.4}", result.delivery_ratio),
            format!("  Offered load:     {:.4} Mbps", result.offered_load_mbps),
            format!("  Wire throughput:  {:.4} Mbps", result.throughput_mbps),
            format!("  Goodput:          {:.4} Mbps", result.goodput_mbps),
            format!("  Mean latency:     {} ms", fmt(result.mean_latency_ms)),
            format!("  P95 latency:      {} ms", fmt(result.p95_latency_ms)),
            format!("  Top bottleneck:   {top_str}"),
            format!("  Total cost:       {:.6}", result.total_cost),
            format!("  Invariants:       {invariants}"),
        ]
        .join("\n")
    }

    fn write_csv(&self, result: &ComputerNetworkResult, csv_path: &str) {
        let mut lines = vec![
            "kind,id,from,to,generated,delivered,dropped,throughput_mbps,goodput_mbps,mean_latency_ms,p95_latency_ms,total_cost,utilization,avg_queue,max_queue,mean_queue_delay_ms"
                .to_string(),
        ];
        for f in &result.flow_stats {
            lines.push(csv_row([
                "flow".to_string(),
                f.id.clone(),
                f.source.clone(),
                f.destination.clone(),
                js_number(f.generated_packets),
                js_number(f.delivered_packets),
                js_number(f.dropped_packets),
                js_number(f.throughput_mbps),
                js_number(f.goodput_mbps),
                js_number(f.mean_latency_ms),
                js_number(f.p95_latency_ms),
                js_number(f.total_cost),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ]));
        }
        for l in &result.link_stats {
            lines.push(csv_row([
                "link".to_string(),
                l.id.clone(),
                l.from.clone(),
                l.to.clone(),
                js_number(l.enqueued_packets),
                js_number(l.delivered_packets),
                js_number(l.dropped_packets),
                js_number(l.throughput_mbps),
                String::new(),
                String::new(),
                String::new(),
                js_number(l.total_cost),
                js_number(l.utilization),
                js_number(l.avg_in_flight),
                js_number(l.max_in_flight),
                js_number(l.mean_queue_delay_ms),
            ]));
        }
        for n in &result.node_stats {
            lines.push(csv_row([
                "node".to_string(),
                n.id.clone(),
                String::new(),
                String::new(),
                js_number(n.received_packets),
                js_number(n.forwarded_packets + n.delivered_packets),
                js_number(n.dropped_packets),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                js_number(n.avg_queue),
                js_number(n.max_queue),
                js_number(n.mean_queue_delay_ms),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }

    fn animate(&self, _result: &ComputerNetworkResult, _params: &ComputerNetworkParams, _runtime: &DESRuntimeConfig) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }

    fn examples(&self) -> Vec<RegistrationExample<ComputerNetworkParams>> {
        vec![
            RegistrationExample {
                name: "small-enterprise".to_string(),
                spec: DESModelSpec {
                    schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                    model: "computer-network".to_string(),
                    description: Some(
                        "Small enterprise packet network with two clients, edge/core routers, and a server."
                            .to_string(),
                    ),
                    parameters: ComputerNetworkParams {
                        builtin: Some(NetworkBuiltin::SmallEnterprise),
                        problem: None,
                    },
                    runtime: None,
                    metadata: None,
                },
            },
            RegistrationExample {
                name: "bottleneck-lab".to_string(),
                spec: DESModelSpec {
                    schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                    model: "computer-network".to_string(),
                    description: Some(
                        "HTTP, UDP, and TCP flows over a narrow WAN link to expose traffic buildup and bottlenecks."
                            .to_string(),
                    ),
                    parameters: ComputerNetworkParams {
                        builtin: Some(NetworkBuiltin::BottleneckLab),
                        problem: None,
                    },
                    runtime: None,
                    metadata: None,
                },
            },
        ]
    }
}
