//! Port of src/des/test/multistage-stochastic-test.ts
//!
//! Unit tests for multi-stage stochastic programming / SDDP
//! (`general/multistage-stochastic`) and the shared affine cut-pool base
//! (`des-base/cut-pool`): cut-pool envelopes, the per-stage inventory LP, the
//! exact extensive-form scenario tree, SDDP convergence to the exact tree, the
//! policy evaluator, and the problem-validation preconditions.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::des_base::cut_pool::{AffineCut, AffineCutPool, CutEnvelopeSense};
    use crate::des::general::lp::LPStatus;
    use crate::des::general::multistage_stochastic::{
        build_default_multi_stage_inventory_problem, evaluate_policy_exact,
        solve_exact_scenario_tree, solve_multi_stage_sddp, solve_stage_decision,
        validate_multi_stage_problem, DemandOutcome, SDDPOptions, SDDPStatus,
    };

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * f64::max(1.0, f64::max(a.abs(), b.abs()))
    }

    fn cut(alpha: f64, beta: Vec<f64>, source: Option<&str>) -> AffineCut {
        AffineCut { alpha, beta, source: source.map(|s| s.to_string()) }
    }

    // [1] AffineCutPool base utility.
    #[test]
    fn affine_cut_pool() {
        let mut upper = AffineCutPool::new(1, CutEnvelopeSense::Upper, &[]).unwrap();
        upper.add(cut(10.0, vec![0.0], Some("constant"))).unwrap();
        upper.add(cut(4.0, vec![1.0], Some("slope"))).unwrap();
        assert!(close(upper.evaluate(&[3.0]).unwrap(), 7.0, 1e-7));
        let active = upper.active_cut(&[3.0]).unwrap().unwrap();
        assert_eq!(active.source, Some("slope".to_string()));

        let lower = AffineCutPool::new(
            1,
            CutEnvelopeSense::Lower,
            &[cut(1.0, vec![1.0], None), cut(5.0, vec![-1.0], None)],
        )
        .unwrap();
        assert!(close(lower.evaluate(&[2.0]).unwrap(), 3.0, 1e-7));

        // Wrong beta dimension is rejected (a `cut.beta` precondition failure).
        let err = upper.add(cut(1.0, vec![1.0, 2.0], None));
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().to_lowercase().contains("cut.beta"));
    }

    // [2] Stage LP mechanics.
    #[test]
    fn stage_lp_mechanics() {
        let p = build_default_multi_stage_inventory_problem();
        let terminal = AffineCutPool::new(
            1,
            CutEnvelopeSense::Upper,
            &[cut(0.0, vec![p.salvage_value], None)],
        )
        .unwrap();
        let dec = solve_stage_decision(&p, p.horizon - 1, 4.0, 6.0, &terminal);
        assert_eq!(dec.status, LPStatus::Optimal);
        assert!(close(dec.next_inventory, 4.0 + dec.order - dec.sell, 1e-7), "next={}", dec.next_inventory);
        assert!(close(dec.sell + dec.stockout, 6.0, 1e-7), "sell+stockout={}", dec.sell + dec.stockout);
        assert!(
            dec.next_inventory >= -1e-8
                && dec.next_inventory <= p.capacity + 1e-8
                && dec.order >= -1e-8
                && dec.order <= p.max_order[p.horizon - 1] + 1e-8
        );
    }

    // [3] Exact extensive-form scenario tree.
    #[test]
    fn exact_scenario_tree() {
        let p = build_default_multi_stage_inventory_problem();
        let exact = solve_exact_scenario_tree(&p);
        assert_eq!(exact.status, "optimal");
        assert_eq!(exact.node_count, 30);
        assert!(exact.objective.is_finite() && exact.objective > 0.0, "z={}", exact.objective);
    }

    // [4] SDDP converges to exact tree on the default 4-stage problem.
    #[test]
    fn sddp_converges_to_exact() {
        let p = build_default_multi_stage_inventory_problem();
        let exact = solve_exact_scenario_tree(&p);
        let sddp = solve_multi_stage_sddp(
            p.clone(),
            SDDPOptions {
                max_iter: Some(80),
                tol: Some(1e-4),
                seed: Some(3),
                exact_objective: Some(exact.objective),
                evaluate_policy_every: Some(20),
                cut_grid_size: Some(21),
                ..Default::default()
            },
        );
        assert_eq!(sddp.status, SDDPStatus::Optimal, "status={:?}", sddp.status);
        assert!(sddp.upper_bound + 1e-5 >= exact.objective, "upper={} exact={}", sddp.upper_bound, exact.objective);
        assert!(
            (sddp.policy_value - exact.objective).abs() <= 1e-4,
            "policy={} exact={}",
            sddp.policy_value,
            exact.objective
        );
        assert!(sddp.cuts_per_stage[0..p.horizon].iter().all(|&n| n >= 2), "{:?}", sddp.cuts_per_stage);
        assert_eq!(sddp.cuts_per_stage[p.horizon], 1);
        assert_eq!(sddp.trace.len(), sddp.iterations);
    }

    // [5] Policy evaluator reproduces the SDDP-reported policy value.
    #[test]
    fn policy_evaluator_matches() {
        let p = build_default_multi_stage_inventory_problem();
        let exact = solve_exact_scenario_tree(&p);
        let sddp = solve_multi_stage_sddp(
            p.clone(),
            SDDPOptions {
                max_iter: Some(30),
                seed: Some(5),
                exact_objective: Some(exact.objective),
                ..Default::default()
            },
        );
        let pools: Vec<AffineCutPool> = sddp
            .cuts
            .iter()
            .map(|cuts| AffineCutPool::new(1, CutEnvelopeSense::Upper, cuts).unwrap())
            .collect();
        let policy_value = evaluate_policy_exact(&p, &pools);
        assert!(close(policy_value, sddp.policy_value, 1e-6), "{} vs {}", policy_value, sddp.policy_value);
    }

    #[test]
    #[should_panic(expected = "prob")]
    fn rejects_bad_probability_mass() {
        let p = build_default_multi_stage_inventory_problem();
        let mut bad = p.clone();
        bad.demands[0] = vec![
            DemandOutcome { demand: 1.0, prob: 0.2 },
            DemandOutcome { demand: 2.0, prob: 0.2 },
        ];
        validate_multi_stage_problem(&bad);
    }

    #[test]
    #[should_panic(expected = "initialInventory")]
    fn rejects_initial_inventory_above_capacity() {
        let p = build_default_multi_stage_inventory_problem();
        let mut bad = p.clone();
        bad.initial_inventory = p.capacity + 1.0;
        validate_multi_stage_problem(&bad);
    }
}
