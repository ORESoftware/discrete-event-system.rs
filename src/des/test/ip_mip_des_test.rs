//! Port of src/des/test/ip-mip-des-test.ts
//!
//! Tests for the explicit station-graph IP/MIP solver (`general/ip-mip-des`):
//! station-graph knapsack, selectable LP backends, mixed integer/continuous
//! programs, cover cuts, auto technique selection, and precondition limits.
//! TS `try/catch` invalid-input cases map onto `#[should_panic]` tests.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::ip_mip_des::{
        build_binary_knapsack_ip, build_ipmip_solver_technique_plan, build_small_mixed_ip,
        solve_ipmip_with_des, ConcreteLpRelaxationAlgorithm, IPMIPProblem, IPMIPSolveOptions,
        IPMIPStatus, LpRelaxationAlgorithm,
    };
    use crate::des::general::lp::Sense;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn feasible(p: &IPMIPProblem, x: &[f64]) -> bool {
        if x.len() != p.c.len() {
            return false;
        }
        for j in 0..x.len() {
            if x[j] < -1e-7 {
                return false;
            }
            if let Some(ub) = &p.ub {
                if ub[j].is_finite() && x[j] > ub[j] + 1e-7 {
                    return false;
                }
            }
            if p.integer_vars[j] && (x[j] - x[j].round()).abs() > 1e-7 {
                return false;
            }
        }
        for i in 0..p.a.len() {
            let lhs: f64 = (0..x.len()).map(|j| p.a[i][j] * x[j]).sum();
            if lhs > p.b[i] + 1e-7 {
                return false;
            }
        }
        true
    }

    fn incremental() -> IPMIPSolveOptions {
        IPMIPSolveOptions {
            lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
            )),
            max_cut_rounds: Some(1),
            ..Default::default()
        }
    }

    // Group 1 — Station-graph knapsack with incremental LP backend.
    #[test]
    fn station_graph_knapsack() {
        let p =
            build_binary_knapsack_ip(vec![10.0, 40.0, 30.0, 50.0], vec![5.0, 4.0, 6.0, 3.0], 10.0);
        let r = solve_ipmip_with_des(p.clone(), incremental());
        assert_eq!(r.status, IPMIPStatus::Optimal);
        assert!(close(r.z, 90.0), "z={}", r.z);
        assert!(feasible(&p, &r.x));
        assert!(r
            .topology
            .iter()
            .any(|n| n.id == "ip-lp-relaxation" && n.role.contains("LP solver")));
        assert_eq!(r.execution_mode, "single-threaded");
        assert!(r
            .topology
            .iter()
            .any(|n| n.id == r.composite_station_id && n.role.contains("composite")));
        assert!(r.topology.iter().any(|n| n.id == "ip-node-decision"
            && n.parent_id.as_deref() == Some(r.composite_station_id.as_str())));
        assert!(
            r.token_stats.stateful > 0
                && r.token_stats.max_generation > 0
                && r.trace.iter().any(|e| e.node_token_id.is_some())
        );
        assert!(
            r.token_stats.by_kind.get("ip-node").copied().unwrap_or(0) > 0
                && r.token_stats
                    .by_kind
                    .get("ip-relaxation")
                    .copied()
                    .unwrap_or(0)
                    > 0
                && r.token_stats
                    .by_kind
                    .get("ip-candidate")
                    .copied()
                    .unwrap_or(0)
                    > 0
        );
        assert!(r.token_stats.state_transitions >= r.token_stats.stateful);
        assert!(r
            .trace
            .iter()
            .any(|e| e.lineage_root.as_deref() == Some("ip-node-0")
                && e.token_generation.unwrap_or(0) > 0));
        assert!(r.token_stats.stateless > 0);
        assert!(r.candidates_tried > 0, "candidates={}", r.candidates_tried);
        assert!(r.in_house_only && !r.uses_external_solvers);
        assert!(
            r.performance.elapsed_ms >= 0.0
                && r.performance.nodes_per_second >= 0.0
                && r.performance.tokens_created == r.token_stats.created
        );
    }

    // Group 2 — Selectable LP algorithms agree.
    #[test]
    fn selectable_lp_algorithms_agree() {
        let p =
            build_binary_knapsack_ip(vec![10.0, 40.0, 30.0, 50.0], vec![5.0, 4.0, 6.0, 3.0], 10.0);
        for alg in [
            ConcreteLpRelaxationAlgorithm::InternalSimplex,
            ConcreteLpRelaxationAlgorithm::InternalInteriorPoint,
            ConcreteLpRelaxationAlgorithm::DesSimplexDantzig,
            ConcreteLpRelaxationAlgorithm::DesSimplexBland,
        ] {
            let requested = LpRelaxationAlgorithm::Concrete(alg);
            let r = solve_ipmip_with_des(
                p.clone(),
                IPMIPSolveOptions {
                    lp_algorithm: Some(requested),
                    max_cut_rounds: Some(1),
                    ..Default::default()
                },
            );
            assert_eq!(r.status, IPMIPStatus::Optimal, "{:?}", alg);
            assert!(close(r.z, 90.0), "{:?}: z={}", alg, r.z);
            assert_eq!(r.lp_algorithm, requested);
        }
    }

    // Group 3 — Mixed integer/continuous program.
    #[test]
    fn mixed_integer_continuous() {
        let p = build_small_mixed_ip();
        let r = solve_ipmip_with_des(
            p.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(r.status, IPMIPStatus::Optimal);
        assert!(close(r.z, 13.0), "z={}", r.z);
        assert!(feasible(&p, &r.x));
        assert!(close(r.x[2], 10.0), "x2={}", r.x[2]);
    }

    // Group 4 — Cover cut strengthens binary relaxation.
    #[test]
    fn cover_cut_strengthens() {
        let p = build_binary_knapsack_ip(vec![10.0, 10.0, 10.0], vec![2.0, 2.0, 2.0], 3.0);
        let r_no_cuts = solve_ipmip_with_des(
            p.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(0),
                max_nodes: Some(50),
                ..Default::default()
            },
        );
        let r_cuts = solve_ipmip_with_des(
            p.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                max_nodes: Some(50),
                ..Default::default()
            },
        );
        assert_eq!(r_no_cuts.status, IPMIPStatus::Optimal);
        assert_eq!(r_cuts.status, IPMIPStatus::Optimal);
        assert!(close(r_cuts.z, 10.0), "z={}", r_cuts.z);
        assert!(r_cuts.cuts_added > 0, "cuts={}", r_cuts.cuts_added);
    }

    // Group 5 — Auto technique selection.
    #[test]
    fn auto_technique_selection() {
        let p =
            build_binary_knapsack_ip(vec![10.0, 40.0, 30.0, 50.0], vec![5.0, 4.0, 6.0, 3.0], 10.0);
        let r = solve_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Auto),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        assert_eq!(r.status, IPMIPStatus::Optimal);
        assert_eq!(r.lp_algorithm, LpRelaxationAlgorithm::Auto);
        assert!(
            r.lp_algorithm_usage
                .get(&ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual)
                .copied()
                .unwrap_or(0)
                > 0
        );
        assert!(
            r.technique_plan.features.all_binary
                && r.technique_plan.root_lp_algorithm
                    == ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual
        );
        assert!(r.in_house_only && !r.technique_plan.external_solvers_allowed);
    }

    fn dense_problem() -> IPMIPProblem {
        IPMIPProblem {
            sense: Sense::Max,
            c: (0..80).map(|i| 1.0 + (i % 5) as f64).collect(),
            a: (0..40)
                .map(|r| (0..80).map(|c| (((r + c) % 7) + 1) as f64).collect())
                .collect(),
            b: vec![500.0; 40],
            integer_vars: vec![false; 80],
            ub: Some(vec![10.0; 80]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        }
    }

    #[test]
    fn auto_plan_dense_root_in_house_or_external() {
        let dense = dense_problem();
        let in_house =
            build_ipmip_solver_technique_plan(&dense, LpRelaxationAlgorithm::Auto, false);
        assert!(
            !in_house.external_candidate
                && in_house.root_lp_algorithm
                    == ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual
        );
        let external = build_ipmip_solver_technique_plan(&dense, LpRelaxationAlgorithm::Auto, true);
        assert!(
            external.external_candidate
                && external.root_lp_algorithm == ConcreteLpRelaxationAlgorithm::ExternalHighsIpm
        );
    }

    #[test]
    #[should_panic]
    fn explicit_external_backend_requires_opt_in() {
        let dense = dense_problem();
        let _ = build_ipmip_solver_technique_plan(
            &dense,
            LpRelaxationAlgorithm::Concrete(ConcreteLpRelaxationAlgorithm::ExternalHighs),
            false,
        );
    }

    #[test]
    fn auto_plan_detects_separable_decomposition() {
        let separable = IPMIPProblem {
            sense: Sense::Max,
            c: vec![5.0, 4.0, 7.0, 6.0],
            a: vec![vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]],
            b: vec![1.0, 1.0],
            integer_vars: vec![true, true, true, true],
            ub: Some(vec![1.0, 1.0, 1.0, 1.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let plan =
            build_ipmip_solver_technique_plan(&separable, LpRelaxationAlgorithm::Auto, false);
        assert!(plan.decomposition_candidate && plan.features.constraint_variable_components == 2);
    }

    // Group 6 — Preconditions and limits.
    #[test]
    #[should_panic]
    fn rejects_malformed_a_row() {
        let bad = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 2.0],
            a: vec![vec![1.0]],
            b: vec![1.0],
            integer_vars: vec![true, true],
            ub: None,
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let _ = solve_ipmip_with_des(bad, IPMIPSolveOptions::default());
    }

    #[test]
    fn node_cap_respected() {
        let p = build_binary_knapsack_ip(
            (1..=12).map(|i| i as f64).collect(),
            (1..=12).map(|i| i as f64).collect(),
            20.0,
        );
        let r = solve_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                max_nodes: Some(1),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert!(r.status == IPMIPStatus::Optimal || r.status == IPMIPStatus::MaxNodes);
        assert!(r.nodes_explored <= 1, "nodes={}", r.nodes_explored);
    }
}
