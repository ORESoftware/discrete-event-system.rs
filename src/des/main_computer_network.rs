//! Port of `src/des/main-computer-network.ts`.
//!
//! Thin runner: packet-switched computer-network DES, prints a flow summary for
//! a chosen scenario. `process.env.SCENARIO` → `std::env::var`.
//!
//! PORT NOTE: the TS delegates to `./general/computer-network`; this runner now
//! uses the real Rust `crate::des::general::computer_network` implementation.

#![allow(dead_code)]

use crate::des::general::computer_network::{
    build_bottleneck_computer_network_problem, build_default_computer_network_problem,
    run_computer_network_simulation, NetworkRoutingMetric,
};

fn fmt(x: f64, digits: usize) -> String {
    if x.is_finite() {
        format!("{x:.digits$}")
    } else {
        "n/a".to_string()
    }
}

fn routing_metric_label(metric: NetworkRoutingMetric) -> &'static str {
    match metric {
        NetworkRoutingMetric::Latency => "latency",
        NetworkRoutingMetric::Cost => "cost",
        NetworkRoutingMetric::Hop => "hop",
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
    println!(
        "# scenario={}",
        if scenario == "baseline" {
            "baseline"
        } else {
            "bottleneck"
        }
    );
    println!(
        "# nodes={}, links={}, flows={}",
        problem.nodes.len(),
        problem.links.len(),
        problem.flows.len()
    );
    println!(
        "# routing={}, simulated={} ms",
        routing_metric_label(result.routing_metric),
        fmt(result.total_simulated_ms, 1)
    );
    println!();

    println!("## Flow summary");
    println!("  generated packets: {}", result.generated_packets);
    println!("  delivered packets: {}", result.delivered_packets);
    println!("  dropped packets:   {}", result.dropped_packets);
    println!("  active at stop:    {}", result.active_packets);
    println!("  max active:        {}", result.max_active_packets);
    println!("  delivery ratio:    {}", fmt(result.delivery_ratio, 4));
    println!(
        "  offered load:      {} Mbps",
        fmt(result.offered_load_mbps, 4)
    );
    println!(
        "  wire throughput:   {} Mbps",
        fmt(result.throughput_mbps, 4)
    );
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
            f.protocol.as_str(),
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
        for s in result
            .time_series
            .iter()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
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
