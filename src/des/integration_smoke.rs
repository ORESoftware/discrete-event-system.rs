//! Cross-subsystem integration smoke tests.
//!
//! These exercise several independent engine subsystems end-to-end (network
//! simulation, graph algorithms, max-flow, and the Tier-2 decimal bookkeeping
//! policy) to guard against regressions that single-module unit tests miss.
//! They are compiled only under `cfg(test)`.

#[cfg(test)]
mod tests {
    use crate::des::general::computer_network::{
        build_bottleneck_computer_network_problem, run_computer_network_simulation,
        validate_computer_network_problem,
    };
    use crate::des::general::max_flow::{
        build_textbook_max_flow_problem, solve_max_flow, MaxFlowStatus,
    };
    use crate::des::general::shortest_path_des::{
        build_small_chain_graph, shortest_path_bellman_ford_des, shortest_path_dijkstra_des,
        BellmanFordOptions,
    };
    use crate::des::shared::precision::{bgn, to_f64, Decimal};

    /// The bottleneck computer-network scenario validates, runs, and conserves
    /// packets (generated == delivered + dropped + active).
    #[test]
    fn bottleneck_network_conserves_packets() {
        let p = build_bottleneck_computer_network_problem();
        assert!(validate_computer_network_problem(&p).is_ok());

        let result = run_computer_network_simulation(&p);
        let conserved =
            result.delivered_packets + result.dropped_packets + result.active_packets;
        assert!(
            (result.generated_packets - conserved).abs() < 0.5,
            "packet conservation violated: generated={} conserved={}",
            result.generated_packets,
            conserved
        );
        assert!(result.generated_packets > 0.0);
        assert_eq!(result.node_stats.len(), 5);
    }

    /// Dijkstra and Bellman–Ford DES solvers agree on the small-chain graph.
    #[test]
    fn dijkstra_and_bellman_ford_agree_on_chain() {
        let g = build_small_chain_graph();
        let bf = shortest_path_bellman_ford_des(&g, 0, BellmanFordOptions::default());
        let dj = shortest_path_dijkstra_des(&g, 0, BellmanFordOptions::default());
        assert_eq!(bf.distance, dj.distance);
        assert_eq!(bf.distance, vec![0.0, 1.0, 3.0, 5.0, 6.0]);
        assert!(!bf.has_negative_cycle_from_source);
    }

    /// The textbook max-flow network has the known optimal value 23, matching
    /// its min-cut capacity (max-flow / min-cut duality).
    #[test]
    fn textbook_max_flow_equals_min_cut() {
        let res = solve_max_flow(build_textbook_max_flow_problem());
        assert_eq!(res.status, MaxFlowStatus::Optimal);
        assert!((res.max_flow - 23.0).abs() < 1e-9, "max_flow={}", res.max_flow);
        assert!(
            (res.min_cut.capacity - res.max_flow).abs() < 1e-9,
            "cut={} flow={}",
            res.min_cut.capacity,
            res.max_flow
        );
    }

    /// Tier-2 policy: routing probabilities summed as exact decimals land on
    /// exactly 1 — the invariant the probability-decision entity relies on. The
    /// branch values (0.05 thirds of a 0.15 budget, etc.) are chosen so naive
    /// `f64` accumulation visibly drifts off 1.0, motivating the decimal guard.
    #[test]
    fn routing_probabilities_sum_to_one_exactly() {
        // 0.05 + 0.15 + 0.30 + 0.45 + 0.05 == 1.00.
        let probs = [0.05_f64, 0.15, 0.30, 0.45, 0.05];

        let mut exact = Decimal::ZERO;
        for &p in &probs {
            exact += bgn(p);
        }
        assert_eq!(exact, Decimal::ONE);
        assert_eq!(to_f64(exact), 1.0);
    }

    /// Decimal money bookkeeping: 1000 increments of $0.07 is exactly $70.00,
    /// not the $69.9999… an f64 register would drift to.
    #[test]
    fn decimal_money_bookkeeping_is_exact() {
        let increment = bgn(0.07);
        let mut balance = Decimal::ZERO;
        for _ in 0..1000 {
            balance += increment;
        }
        assert_eq!(balance, bgn(70.0));
        assert_eq!(to_f64(balance), 70.0);
    }
}
