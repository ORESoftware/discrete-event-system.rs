//! Port of src/des/test/stochastic-lp-test.ts
//!
//! Unit tests for the stochastic LP solver: subproblem dual extraction, SAA
//! monolithic solver, Benders decomposition, and the closed-form oracle.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::stochastic_lp::{
        build_production_scenarios, build_production_slp, solve_production_closed_form,
        solve_slp_benders, solve_slp_monolithic, solve_subproblem_with_duals, BendersOpts, Scenario,
        ScenarioMeta, SLPStatus, SubproblemStatus, UniformDemandSpec,
    };

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    // [1] solveSubproblemWithDuals — single product newsvendor recourse
    #[test]
    fn subproblem_single_product() {
        let r = solve_subproblem_with_duals(&[25.0], &[vec![1.0], vec![1.0]], &[60.0, 80.0]);
        assert_eq!(r.status, SubproblemStatus::Optimal);
        assert!(close(r.y[0], 60.0, 1e-7));
        assert!(close(r.obj, 1500.0, 1e-7));
        assert!(close(r.duals[0], 25.0, 1e-7));
        assert!(close(r.duals[1], 0.0, 1e-7));

        let r = solve_subproblem_with_duals(&[25.0], &[vec![1.0], vec![1.0]], &[90.0, 70.0]);
        assert!(close(r.y[0], 70.0, 1e-7));
        assert!(close(r.duals[0], 0.0, 1e-7));
        assert!(close(r.duals[1], 25.0, 1e-7));
    }

    // [2] solveSubproblemWithDuals — two-product, capacity-bound
    #[test]
    fn subproblem_two_product() {
        let w = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = solve_subproblem_with_duals(&[25.0, 28.0], &w, &[50.0, 40.0, 80.0, 60.0]);
        assert!(close(r.y[0], 50.0, 1e-7) && close(r.y[1], 40.0, 1e-7));
        assert!(close(r.obj, 2370.0, 1e-7));
        assert!(close(r.duals[0], 25.0, 1e-7) && close(r.duals[1], 28.0, 1e-7));
        assert!(close(r.duals[2], 0.0, 1e-7) && close(r.duals[3], 0.0, 1e-7));
    }

    // [3] Closed-form newsvendor agrees with derived formula
    #[test]
    fn closed_form_newsvendor() {
        let r = solve_production_closed_form(vec![10.0], vec![25.0], vec![(50.0, 100.0)]);
        assert!(close(r.x[0], 80.0, 1e-7));
        assert!(close(r.objective, 975.0, 1e-7));
    }

    // [4] Monolithic SAA on a 3-scenario discrete distribution
    #[test]
    fn monolithic_saa() {
        let slp = build_production_slp(vec![10.0, 12.0], vec![25.0, 28.0], None);
        let demands = [[40.0, 30.0], [60.0, 50.0], [80.0, 70.0]];
        let scenarios: Vec<Scenario> = demands
            .iter()
            .map(|d| Scenario {
                t: vec![vec![-1.0, 0.0], vec![0.0, -1.0], vec![0.0, 0.0], vec![0.0, 0.0]],
                h: vec![0.0, 0.0, d[0], d[1]],
                prob: Some(1.0 / 3.0),
                meta: Some(ScenarioMeta { d: d.to_vec() }),
            })
            .collect();
        let sol = solve_slp_monolithic(slp, scenarios);
        assert_eq!(sol.status, SLPStatus::Optimal);
        let x = &sol.x;
        assert!(x[0] >= -1e-9 && x[1] >= -1e-9);
        let direct = -10.0 * x[0] - 12.0 * x[1]
            + (1.0 / 3.0)
                * ((25.0 * x[0].min(40.0) + 28.0 * x[1].min(30.0))
                    + (25.0 * x[0].min(60.0) + 28.0 * x[1].min(50.0))
                    + (25.0 * x[0].min(80.0) + 28.0 * x[1].min(70.0)));
        assert!(close(sol.objective, direct, 1e-7));
    }

    // [5] Benders converges and matches monolithic on same scenarios
    #[test]
    fn benders_matches_monolithic() {
        let slp = build_production_slp(vec![10.0, 12.0], vec![25.0, 28.0], None);
        let sc =
            build_production_scenarios(UniformDemandSpec { ranges: vec![(50.0, 100.0), (40.0, 80.0)], seed: 7 }, 100);
        let mono = solve_slp_monolithic(slp.clone(), sc.clone());
        let bend = solve_slp_benders(slp, sc, BendersOpts { tol: Some(1e-9), ..Default::default() });
        assert_eq!(mono.status, SLPStatus::Optimal);
        assert_eq!(bend.status, SLPStatus::Optimal);
        assert!((mono.objective - bend.objective).abs() <= 1e-6);
        assert!(bend.iterations <= 50, "iters={}", bend.iterations);
        let trace = bend.benders_trace.as_ref().unwrap();
        let cut_count = trace.iter().filter(|t| t.cut_added.is_some()).count();
        assert_eq!(cut_count, bend.iterations - 1);
    }

    // [6] Benders convergence properties
    #[test]
    fn benders_convergence_properties() {
        let slp = build_production_slp(vec![10.0, 12.0], vec![25.0, 28.0], None);
        let sc =
            build_production_scenarios(UniformDemandSpec { ranges: vec![(50.0, 100.0), (40.0, 80.0)], seed: 11 }, 50);
        let bend =
            solve_slp_benders(slp, sc, BendersOpts { tol: Some(1e-9), max_iter: Some(200), ..Default::default() });
        let trace = bend.benders_trace.as_ref().unwrap();
        for i in 1..trace.len() {
            assert!(trace[i].upper_bound <= trace[i - 1].upper_bound + 1e-6);
        }
        let mut running_best = f64::NEG_INFINITY;
        for it in trace {
            let new_best = running_best.max(it.lower_bound);
            assert!(new_best >= running_best - 1e-9);
            running_best = new_best;
        }
        assert!(trace[trace.len() - 1].gap <= 1e-6);
    }

    // [7] As N grows, SAA optimum approaches closed-form
    #[test]
    fn saa_approaches_closed_form() {
        let c = vec![10.0, 12.0];
        let p = vec![25.0, 28.0];
        let ranges = vec![(50.0, 100.0), (40.0, 80.0)];
        let slp = build_production_slp(c.clone(), p.clone(), None);
        let cf = solve_production_closed_form(c, p, ranges.clone());
        let r = 8u32;
        let avg_bias = |n: usize| -> f64 {
            let mut acc = 0.0;
            for seed in 1..=r {
                let sc = build_production_scenarios(
                    UniformDemandSpec { ranges: ranges.clone(), seed: seed * 100 + n as u32 },
                    n,
                );
                let sol = solve_slp_benders(slp.clone(), sc, BendersOpts { tol: Some(1e-7), ..Default::default() });
                acc += sol.objective - cf.objective;
            }
            acc / r as f64
        };
        let bias20 = avg_bias(20);
        let bias2000 = avg_bias(2000);
        assert!(bias2000.abs() < bias20.abs(), "|{bias2000}| vs |{bias20}|");
    }

    // [8] Budget-constrained: budget binds and reduces objective
    #[test]
    fn budget_constraint_binds() {
        let c = vec![10.0, 12.0];
        let p = vec![25.0, 28.0];
        let ranges = vec![(50.0, 100.0), (40.0, 80.0)];
        let slp_unc = build_production_slp(c.clone(), p.clone(), None);
        let slp_budget = build_production_slp(c, p, Some(80.0));
        let sc = build_production_scenarios(UniformDemandSpec { ranges, seed: 31 }, 200);
        let unc = solve_slp_benders(slp_unc, sc.clone(), BendersOpts { tol: Some(1e-9), ..Default::default() });
        let bud = solve_slp_benders(slp_budget, sc, BendersOpts { tol: Some(1e-9), ..Default::default() });
        assert!(unc.x[0] + unc.x[1] >= 80.0 - 1e-7);
        assert!(bud.x[0] + bud.x[1] <= 80.0 + 1e-7);
        assert!(bud.objective <= unc.objective + 1e-7);
    }

    // [9] Subproblem duals: KKT optimality direct check
    #[test]
    fn subproblem_kkt_conditions() {
        let w = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        for trial in 0..5 {
            let t = trial as f64;
            let rhs = [40.0 + t * 5.0, 30.0 + t * 7.0, 80.0 - t * 3.0, 60.0 + t * 4.0];
            let r = solve_subproblem_with_duals(&[25.0, 28.0], &w, &rhs);
            assert_eq!(r.status, SubproblemStatus::Optimal);
            let mut comp = 0.0;
            for i in 0..4 {
                let lhs = if i % 2 == 0 { r.y[0] } else { r.y[1] };
                let slack = rhs[i] - lhs;
                comp += (r.duals[i] * slack).abs();
            }
            assert!(comp <= 1e-9, "trial={trial} comp={comp:e}");
            let lhs1 = r.duals[0] + r.duals[2];
            let lhs2 = r.duals[1] + r.duals[3];
            assert!(lhs1 + 1e-9 >= 25.0, "lhs1={lhs1}");
            assert!(lhs2 + 1e-9 >= 28.0, "lhs2={lhs2}");
        }
    }
}
