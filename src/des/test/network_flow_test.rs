//! Port of src/des/test/network-flow-test.ts
//!
//! Tests for network-flow (animated max-flow DES), the modular max-flow/min-cut
//! solver, the modular traffic simulation + its max-flow bound, and the
//! stochastic-flow MDP interpretation, plus their hard precondition failures.
//!
//! PORT NOTE: groups [2] (continuous-time / smart traffic flow) and [3] (JSON
//! registry + output paths) depend on `smart-traffic-flow` and `des-registry`,
//! neither of which is ported yet, so they are deferred. Likewise the two
//! smart-traffic precondition cases [7.4]/[7.5] are deferred.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::max_flow::{
        build_textbook_max_flow_problem, solve_max_flow, MaxFlowEdge, MaxFlowProblem,
    };
    use crate::des::general::network_flow::{run_max_flow, FlowEdge, MaxFlowParams};
    use crate::des::general::stochastic_flow_mdp::{
        build_default_stochastic_flow_mdp_problem, solve_stochastic_flow_mdp, FlowMDPActionKind,
        SolveStochasticFlowMDPOptions,
    };
    use crate::des::general::traffic_flow::{
        build_default_traffic_problem, build_traffic_max_flow_problem, run_traffic_simulation,
    };

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-8 * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn edge(from: usize, to: usize, capacity: f64, name: &str) -> FlowEdge {
        FlowEdge {
            from,
            to,
            capacity,
            name: Some(name.to_string()),
        }
    }

    fn teaching_network() -> MaxFlowParams {
        MaxFlowParams {
            num_nodes: 6,
            source: 0,
            sink: 5,
            edges: vec![
                edge(0, 1, 16.0, "s-a"),
                edge(0, 2, 13.0, "s-b"),
                edge(1, 2, 10.0, "a-b"),
                edge(2, 1, 4.0, "b-a"),
                edge(1, 3, 12.0, "a-c"),
                edge(3, 2, 9.0, "c-b"),
                edge(2, 4, 14.0, "b-d"),
                edge(4, 3, 7.0, "d-c"),
                edge(3, 5, 20.0, "c-t"),
                edge(4, 5, 4.0, "d-t"),
            ],
            max_augmentations: None,
            node_coordinates: Some(vec![
                (90.0, 260.0),
                (260.0, 160.0),
                (260.0, 360.0),
                (520.0, 160.0),
                (520.0, 360.0),
                (760.0, 260.0),
            ]),
            node_names: Some(
                ["s", "a", "b", "c", "d", "t"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
        }
    }

    // [1] Animated/logged max-flow DES optimization
    #[test]
    fn animated_max_flow() {
        let r = run_max_flow(teaching_network(), None);
        assert!(close(r.max_flow, 23.0));
        assert!(close(r.min_cut.capacity, r.max_flow));
        assert!(r.validation.iter().all(|c| c.passed));
        assert!(!r.trace.is_empty());
        for i in 1..r.trace.len() {
            assert!(r.trace[i].value >= r.trace[i - 1].value - 1e-12);
        }
        assert!(r
            .edge_flows
            .iter()
            .all(|e| e.flow >= -1e-9 && e.flow <= e.capacity + 1e-9));

        // [1.7] rejects negative capacities
        let threw = std::panic::catch_unwind(|| {
            run_max_flow(
                MaxFlowParams {
                    num_nodes: 2,
                    source: 0,
                    sink: 1,
                    edges: vec![FlowEdge {
                        from: 0,
                        to: 1,
                        capacity: -1.0,
                        name: None,
                    }],
                    max_augmentations: None,
                    node_coordinates: None,
                    node_names: None,
                },
                None,
            )
        })
        .is_err();
        assert!(threw);
    }

    // [4] Modular max-flow/min-cut implementation
    #[test]
    fn modular_max_flow() {
        let r = solve_max_flow(build_textbook_max_flow_problem());
        assert!(close(r.max_flow, 23.0));
        assert!(close(r.min_cut.capacity, 23.0));
        assert!(!r.trace.is_empty());
        assert!(r.min_cut.source_side.contains(&r.source));
        assert!(r.min_cut.sink_side.contains(&r.sink));
        assert!(r
            .edge_flows
            .iter()
            .all(|e| e.flow >= -1e-9 && e.flow <= e.capacity + 1e-9));

        let p = MaxFlowProblem {
            num_nodes: 4,
            source: 0,
            sink: 3,
            edges: vec![
                MaxFlowEdge {
                    from: 0,
                    to: 1,
                    capacity: 4.0,
                    name: Some("s-a".into()),
                },
                MaxFlowEdge {
                    from: 0,
                    to: 2,
                    capacity: 3.0,
                    name: Some("s-b".into()),
                },
                MaxFlowEdge {
                    from: 1,
                    to: 2,
                    capacity: 1.0,
                    name: Some("a-b".into()),
                },
                MaxFlowEdge {
                    from: 1,
                    to: 3,
                    capacity: 2.0,
                    name: Some("a-t".into()),
                },
                MaxFlowEdge {
                    from: 2,
                    to: 3,
                    capacity: 4.0,
                    name: Some("b-t".into()),
                },
            ],
        };
        let bottleneck = solve_max_flow(p);
        assert!(close(bottleneck.max_flow, 6.0));
        assert!(close(bottleneck.min_cut.capacity, bottleneck.max_flow));
    }

    // [5] Modular traffic simulation and max-flow bound
    #[test]
    fn modular_traffic_simulation() {
        let p = build_default_traffic_problem();
        let r = run_traffic_simulation(&p);
        assert!(r.completed_cars > 0.0);
        assert!(r.generated_cars <= 240.0);
        assert!((r.generated_cars - (r.completed_cars + r.active_cars)).abs() < 1e-9);
        assert!(r.max_active_cars < 300.0);
        assert!(r.max_active_cars <= p.max_cars as f64);
        assert!(r.invariant_violations.is_empty());
        assert!(r.mean_travel_time_sec.is_finite());
        assert!(r.p95_travel_time_sec.is_finite());
        assert!(!r.time_series.is_empty());

        let mf = solve_max_flow(build_traffic_max_flow_problem(&p));
        assert!(close(r.max_flow_upper_bound_per_min, mf.max_flow));
        assert!(r.throughput_vs_max_flow <= 1.05);
        assert!(mf.max_flow > 0.0);
    }

    // [6] Stochastic-flow MDP interpretation
    #[test]
    fn stochastic_flow_mdp() {
        let p = build_default_stochastic_flow_mdp_problem();
        let horizon = p.horizon;
        let r = solve_stochastic_flow_mdp(
            p,
            SolveStochasticFlowMDPOptions {
                seed: Some(7),
                max_policy_rows: None,
            },
        );
        assert!(r.expected_reward.is_finite());
        assert!(r.num_states > 0);
        assert_eq!(r.initial_policy[0].action.kind, FlowMDPActionKind::Edge);
        assert!(r.expected_reward <= r.deterministic_max_flow + 1e-9);
        assert!(r.simulation.delivered <= r.deterministic_max_flow);
        assert_eq!(r.stage_history.len(), horizon + 1);

        // Deterministic variant: success_prob = 1, no costs/penalties.
        let mut p2 = build_default_stochastic_flow_mdp_problem();
        for e in p2.edges.iter_mut() {
            e.success_prob = 1.0;
            e.cost = Some(0.0);
        }
        p2.wait_penalty = Some(0.0);
        p2.failure_penalty = Some(0.0);
        let h2 = p2.horizon;
        let det = solve_stochastic_flow_mdp(
            p2,
            SolveStochasticFlowMDPOptions {
                seed: Some(1),
                max_policy_rows: None,
            },
        );
        assert!(close(det.expected_reward, det.deterministic_max_flow));
        assert!(close(det.simulation.delivered, det.deterministic_max_flow));
        assert_eq!(
            det.initial_policy
                .iter()
                .filter(|x| x.action.kind == FlowMDPActionKind::Edge)
                .count(),
            h2
        );
    }

    // [7] Hard precondition failures (the portable, non-smart-traffic subset)
    #[test]
    fn precondition_failures() {
        // [7.1] modular traffic rejects maxCars >= 300
        let threw_traffic = std::panic::catch_unwind(|| {
            let mut p = build_default_traffic_problem();
            p.max_cars = 300;
            run_traffic_simulation(&p)
        })
        .is_err();
        assert!(threw_traffic);

        // [7.2] modular max-flow rejects source == sink
        let threw_self = std::panic::catch_unwind(|| {
            solve_max_flow(MaxFlowProblem {
                num_nodes: 2,
                source: 0,
                sink: 0,
                edges: vec![MaxFlowEdge {
                    from: 0,
                    to: 1,
                    capacity: 1.0,
                    name: None,
                }],
            })
        })
        .is_err();
        assert!(threw_self);

        // [7.3] stochastic-flow MDP rejects invalid transition probability
        let threw_prob = std::panic::catch_unwind(|| {
            let mut p = build_default_stochastic_flow_mdp_problem();
            p.edges[0].success_prob = 1.5;
            solve_stochastic_flow_mdp(p, SolveStochasticFlowMDPOptions::default())
        })
        .is_err();
        assert!(threw_prob);
    }
}
