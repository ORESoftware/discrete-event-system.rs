//! Port of `src/des/main-computer-network.ts`.
//!
//! Thin runner: packet-switched computer-network DES, prints a flow summary for
//! a chosen scenario. `process.env.SCENARIO` → `std::env::var`.
//!
//! PORT NOTE: the TS delegates to `./general/computer-network`
//! (`buildDefaultComputerNetworkProblem`, `buildBottleneckComputerNetworkProblem`,
//! `runComputerNetworkSimulation`). The Rust module
//! `crate::des::general::computer_network` exists but is currently EMPTY (the
//! port is pending). To keep this entry script self-contained and compiling, the
//! problem/result types + builders + simulation are stubbed locally below (the
//! console-reporting logic — the substance of this script — is ported
//! faithfully). Replace the local `cn` stub with `use
//! crate::des::general::computer_network::{...}` once that module is ported.

#![allow(dead_code)]

use cn::{
    build_bottleneck_computer_network_problem, build_default_computer_network_problem,
    run_computer_network_simulation,
};

fn fmt(x: f64, digits: usize) -> String {
    if x.is_finite() {
        format!("{x:.digits$}")
    } else {
        "n/a".to_string()
    }
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let scenario = std::env::var("SCENARIO").unwrap_or_else(|_| "bottleneck".into());
    let problem = if scenario == "baseline" {
        build_default_computer_network_problem()
    } else {
        build_bottleneck_computer_network_problem()
    };
    let result = run_computer_network_simulation(&problem);

    println!("# Computer-network DES");
    println!("# stationary hosts/routers/switches/links + moving packets");
    println!("# scenario={}", if scenario == "baseline" { "baseline" } else { "bottleneck" });
    println!(
        "# nodes={}, links={}, flows={}",
        problem.nodes.len(),
        problem.links.len(),
        problem.flows.len()
    );
    println!("# routing={}, simulated={} ms", result.routing_metric, fmt(result.total_simulated_ms, 1));
    println!();

    println!("## Flow summary");
    println!("  generated packets: {}", result.generated_packets);
    println!("  delivered packets: {}", result.delivered_packets);
    println!("  dropped packets:   {}", result.dropped_packets);
    println!("  active at stop:    {}", result.active_packets);
    println!("  max active:        {}", result.max_active_packets);
    println!("  delivery ratio:    {}", fmt(result.delivery_ratio, 4));
    println!("  offered load:      {} Mbps", fmt(result.offered_load_mbps, 4));
    println!("  wire throughput:   {} Mbps", fmt(result.throughput_mbps, 4));
    println!("  goodput:           {} Mbps", fmt(result.goodput_mbps, 4));
    println!("  total cost:        {}", fmt(result.total_cost, 6));
    println!();

    println!("## Latency");
    println!("  mean: {} ms", fmt(result.mean_latency_ms, 2));
    println!("  p95:  {} ms", fmt(result.p95_latency_ms, 2));
    println!();

    println!("## Per-flow stats");
    for f in &result.flow_stats {
        println!(
            "  {:<14} {:<4} {} -> {} delivered={:>4}/{:>4} drops={:>4} mean={}ms goodput={}Mbps cost={}",
            f.id,
            f.protocol,
            f.source,
            f.destination,
            f.delivered_packets,
            f.generated_packets,
            f.dropped_packets,
            fmt(f.mean_time_in_system_ms, 2),
            fmt(f.goodput_mbps, 3),
            fmt(f.total_cost, 6)
        );
    }
    println!();

    println!("## Link stats");
    for l in &result.link_stats {
        println!(
            "  {:<18} {} -> {} delivered={:>4} util={} avgInFlight={} meanQ={}ms maxQ={}ms",
            l.id,
            l.from,
            l.to,
            l.delivered_packets,
            fmt(l.utilization, 3),
            fmt(l.avg_in_flight, 2),
            fmt(l.mean_queue_delay_ms, 2),
            fmt(l.max_queue_delay_ms, 2)
        );
    }
    println!();

    println!("## Bottlenecks");
    for b in result.bottlenecks.iter().take(5) {
        let util = match b.utilization {
            Some(u) => format!(" util={}", fmt(u, 3)),
            None => String::new(),
        };
        println!(
            "  {:<4} {:<18} {:<16} score={}{} avgQ={} maxQ={} meanQ={}ms drops={}",
            b.kind,
            b.id,
            b.reason,
            fmt(b.score, 3),
            util,
            fmt(b.avg_queue, 2),
            fmt(b.max_queue, 0),
            fmt(b.mean_queue_delay_ms, 2),
            b.dropped_packets
        );
    }

    println!();
    println!("## Traffic build-up samples");
    for s in result.time_series.iter().take(6) {
        println!(
            "  t={:>4}ms active={:>4} delivered={:>4} dropped={:>4}",
            s.t_ms, s.active_packets, s.delivered_packets, s.dropped_packets
        );
    }
    if result.time_series.len() > 8 {
        println!("  ...");
        for s in result.time_series.iter().rev().take(4).collect::<Vec<_>>().into_iter().rev() {
            println!(
                "  t={:>4}ms active={:>4} delivered={:>4} dropped={:>4}",
                s.t_ms, s.active_packets, s.delivered_packets, s.dropped_packets
            );
        }
    }

    if !result.invariant_violations.is_empty() {
        println!();
        println!("## Invariant violations");
        for v in result.invariant_violations.iter().take(10) {
            println!("  {v}");
        }
    }
}

// -----------------------------------------------------------------------------
// PORT NOTE: local stub of `crate::des::general::computer_network` (empty
// upstream). Minimal types + zeroed simulation so this runner compiles and the
// reporting layout is faithful. Wire to the real module once ported.
// -----------------------------------------------------------------------------
mod cn {
    #[derive(Clone, Debug)]
    pub struct NetNode {
        pub id: String,
    }
    #[derive(Clone, Debug)]
    pub struct NetLink {
        pub id: String,
    }
    #[derive(Clone, Debug)]
    pub struct NetFlow {
        pub id: String,
    }

    #[derive(Clone, Debug, Default)]
    pub struct NetworkProblem {
        pub nodes: Vec<NetNode>,
        pub links: Vec<NetLink>,
        pub flows: Vec<NetFlow>,
    }

    #[derive(Clone, Debug)]
    pub struct FlowStat {
        pub id: String,
        pub protocol: String,
        pub source: String,
        pub destination: String,
        pub delivered_packets: usize,
        pub generated_packets: usize,
        pub dropped_packets: usize,
        pub mean_time_in_system_ms: f64,
        pub goodput_mbps: f64,
        pub total_cost: f64,
    }

    #[derive(Clone, Debug)]
    pub struct LinkStat {
        pub id: String,
        pub from: String,
        pub to: String,
        pub delivered_packets: usize,
        pub utilization: f64,
        pub avg_in_flight: f64,
        pub mean_queue_delay_ms: f64,
        pub max_queue_delay_ms: f64,
    }

    #[derive(Clone, Debug)]
    pub struct Bottleneck {
        pub kind: String,
        pub id: String,
        pub reason: String,
        pub score: f64,
        pub utilization: Option<f64>,
        pub avg_queue: f64,
        pub max_queue: f64,
        pub mean_queue_delay_ms: f64,
        pub dropped_packets: usize,
    }

    #[derive(Clone, Debug)]
    pub struct TimeSample {
        pub t_ms: i64,
        pub active_packets: usize,
        pub delivered_packets: usize,
        pub dropped_packets: usize,
    }

    #[derive(Clone, Debug, Default)]
    pub struct NetworkResult {
        pub routing_metric: String,
        pub total_simulated_ms: f64,
        pub generated_packets: usize,
        pub delivered_packets: usize,
        pub dropped_packets: usize,
        pub active_packets: usize,
        pub max_active_packets: usize,
        pub delivery_ratio: f64,
        pub offered_load_mbps: f64,
        pub throughput_mbps: f64,
        pub goodput_mbps: f64,
        pub total_cost: f64,
        pub mean_latency_ms: f64,
        pub p95_latency_ms: f64,
        pub flow_stats: Vec<FlowStat>,
        pub link_stats: Vec<LinkStat>,
        pub bottlenecks: Vec<Bottleneck>,
        pub time_series: Vec<TimeSample>,
        pub invariant_violations: Vec<String>,
    }

    pub fn build_default_computer_network_problem() -> NetworkProblem {
        NetworkProblem::default()
    }
    pub fn build_bottleneck_computer_network_problem() -> NetworkProblem {
        NetworkProblem::default()
    }
    pub fn run_computer_network_simulation(_p: &NetworkProblem) -> NetworkResult {
        NetworkResult { routing_metric: "n/a (stub)".to_string(), ..Default::default() }
    }
}
