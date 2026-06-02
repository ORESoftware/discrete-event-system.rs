//! Tests for min-cost flow and its LP/external-solver cross-check.

#[cfg(test)]
mod tests {
    use crate::des::general::lp::{
        solve_lp_external, solve_lp_internal, ExternalSolverOptions, InternalSimplexOptions,
        LPStatus,
    };
    use crate::des::general::min_cost_flow::{
        min_cost_flow_to_lp, solve_min_cost_flow, MinCostFlowArc, MinCostFlowProblem,
        MinCostFlowStatus,
    };

    fn transportation_problem() -> MinCostFlowProblem {
        MinCostFlowProblem {
            num_nodes: 4,
            supplies: vec![5.0, 7.0, -6.0, -6.0],
            arcs: vec![
                MinCostFlowArc {
                    from: 0,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 5.0,
                    cost: 2.0,
                    name: Some("s0_d0".to_string()),
                },
                MinCostFlowArc {
                    from: 0,
                    to: 3,
                    lower_bound: 0.0,
                    capacity: 5.0,
                    cost: 4.0,
                    name: Some("s0_d1".to_string()),
                },
                MinCostFlowArc {
                    from: 1,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 6.0,
                    cost: 5.0,
                    name: Some("s1_d0".to_string()),
                },
                MinCostFlowArc {
                    from: 1,
                    to: 3,
                    lower_bound: 0.0,
                    capacity: 8.0,
                    cost: 1.0,
                    name: Some("s1_d1".to_string()),
                },
            ],
        }
    }

    #[test]
    fn min_cost_flow_matches_lp_and_external_bridge() {
        let problem = transportation_problem();
        let flow = solve_min_cost_flow(problem.clone());
        assert_eq!(flow.status, MinCostFlowStatus::Optimal);
        assert!((flow.total_cost - 21.0).abs() < 1e-9);

        let lp = min_cost_flow_to_lp(&problem);
        let internal = solve_lp_internal(&lp, &InternalSimplexOptions::default());
        assert_eq!(internal.status, LPStatus::Optimal);
        assert!((internal.objective - flow.total_cost).abs() < 1e-8);

        let external = solve_lp_external(
            &lp,
            &ExternalSolverOptions {
                method: Some("highs".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(external.status, LPStatus::Optimal, "{:?}", external.message);
        assert!((external.objective - flow.total_cost).abs() < 1e-8);
    }
}
