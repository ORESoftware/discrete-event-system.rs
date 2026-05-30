//! Port of src/des/test/milp-bnb-test.ts
//!
//! Unit tests for general/milp-bnb (mixed-integer branch-and-bound).

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::milp_bnb::{
        build_knapsack_milp, solve_milp, BranchRule, MILPProblem, MILPSolveOptions, MILPStatus,
        Sense,
    };

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 * 1.0_f64.max(a.abs()).max(b.abs())
    }

    // [1] Knapsack builder.
    #[test]
    fn knapsack_builder() {
        let k = build_knapsack_milp(vec![10.0, 20.0], vec![1.0, 2.0], 3.0);
        assert_eq!(k.sense, Sense::Max);
        assert_eq!(k.c[0], 10.0);
        assert_eq!(k.c[1], 20.0);
        assert_eq!(k.a.len(), 1);
        assert_eq!(k.a[0][0], 1.0);
        assert_eq!(k.a[0][1], 2.0);
        assert_eq!(k.b[0], 3.0);
        assert!(k.integer_vars.iter().all(|&b| b));
        assert!(k.ub.as_ref().unwrap().iter().all(|&b| b == 1.0));
    }

    // [2] Trivial knapsack — exact solution.
    #[test]
    fn trivial_knapsack_exact() {
        let r1 = solve_milp(
            &build_knapsack_milp(vec![5.0, 3.0], vec![1.0, 1.0], 2.0),
            MILPSolveOptions::default(),
        );
        assert_eq!(r1.status, MILPStatus::Optimal);
        assert!(close(r1.z, 8.0));
        assert!(close(r1.x[0], 1.0) && close(r1.x[1], 1.0));

        let r2 = solve_milp(
            &build_knapsack_milp(vec![5.0, 3.0], vec![1.0, 1.0], 1.0),
            MILPSolveOptions::default(),
        );
        assert!(close(r2.z, 5.0));
        assert!(close(r2.x[0], 1.0) && close(r2.x[1], 0.0));

        let r3 = solve_milp(
            &build_knapsack_milp(vec![5.0, 3.0], vec![1.0, 1.0], 0.0),
            MILPSolveOptions::default(),
        );
        assert!(close(r3.z, 0.0));
        assert!(close(r3.x[0], 0.0) && close(r3.x[1], 0.0));
    }

    // [3] Pure LP (no integer constraints) reduces to root.
    #[test]
    fn pure_lp_only_root() {
        let lp = MILPProblem {
            sense: Sense::Max,
            c: vec![3.0, 5.0],
            a: vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![3.0, 2.0]],
            b: vec![4.0, 12.0, 18.0],
            integer_vars: vec![false, false],
            ub: None,
            var_names: None,
            con_names: None,
        };
        let r = solve_milp(&lp, MILPSolveOptions::default());
        assert!(close(r.z, 36.0));
        assert!(close(r.x[0], 2.0) && close(r.x[1], 6.0));
        assert_eq!(r.nodes_explored, 1);
        assert!(r.gap < 1e-9);
    }

    // [4] Mixed integer/continuous.
    #[test]
    fn mixed_integer_continuous() {
        let milp = MILPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0, 1.0],
            a: vec![vec![1.0, 1.0, 0.0]],
            b: vec![3.0],
            integer_vars: vec![true, true, false],
            ub: Some(vec![10.0, 10.0, 10.0]),
            var_names: None,
            con_names: None,
        };
        let r = solve_milp(&milp, MILPSolveOptions::default());
        assert_eq!(r.status, MILPStatus::Optimal);
        assert!(r.x[0] + r.x[1] <= 3.0 + 1e-6);
        assert!((r.x[0] - r.x[0].round()).abs() < 1e-6 && (r.x[1] - r.x[1].round()).abs() < 1e-6);
        assert!(close(r.z, 13.0));
    }

    // [5] Bounding properties.
    #[test]
    fn bounding_properties() {
        let milp = MILPProblem {
            sense: Sense::Max,
            c: vec![10.0, 6.0, 4.0],
            a: vec![
                vec![1.0, 1.0, 1.0],
                vec![10.0, 4.0, 5.0],
                vec![2.0, 2.0, 6.0],
            ],
            b: vec![100.0, 600.0, 300.0],
            integer_vars: vec![true, true, true],
            ub: Some(vec![f64::INFINITY, f64::INFINITY, f64::INFINITY]),
            var_names: None,
            con_names: None,
        };
        let r = solve_milp(
            &milp,
            MILPSolveOptions {
                max_nodes: Some(5000),
                ..Default::default()
            },
        );
        assert_eq!(r.status, MILPStatus::Optimal);
        assert!(
            r.best_bound >= r.z - 1e-6,
            "bestBound={}, z={}",
            r.best_bound,
            r.z
        );
        assert!(r.gap >= 0.0);
    }

    // [6] Feasibility constraints satisfied.
    #[test]
    fn feasibility_satisfied() {
        let milp = build_knapsack_milp(
            vec![8.0, 12.0, 15.0, 22.0, 7.0],
            vec![3.0, 5.0, 6.0, 9.0, 4.0],
            12.0,
        );
        let r = solve_milp(&milp, MILPSolveOptions::default());
        assert_eq!(r.status, MILPStatus::Optimal);
        assert!(r.x.iter().all(|&v| v >= -1e-9));
        assert!(r.x.iter().all(|&v| v <= 1.0 + 1e-9));
        let w = [3.0, 5.0, 6.0, 9.0, 4.0];
        let used: f64 = (0..r.x.len()).map(|i| w[i] * r.x[i]).sum();
        assert!(used <= 12.0 + 1e-9, "used = {:.3}", used);
        assert!(r.x.iter().all(|&v| (v - v.round()).abs() < 1e-6));
    }

    // [7] Branch-rule choice changes node count but not optimum.
    #[test]
    fn branch_rule_same_optimum() {
        let milp = build_knapsack_milp(
            vec![15.0, 17.0, 8.0, 9.0, 12.0, 5.0, 30.0, 25.0],
            vec![10.0, 12.0, 5.0, 7.0, 8.0, 3.0, 15.0, 13.0],
            35.0,
        );
        let r_most = solve_milp(
            &milp,
            MILPSolveOptions {
                branch_rule: Some(BranchRule::MostFractional),
                ..Default::default()
            },
        );
        let r_first = solve_milp(
            &milp,
            MILPSolveOptions {
                branch_rule: Some(BranchRule::FirstFractional),
                ..Default::default()
            },
        );
        assert!(close(r_most.z, r_first.z));
    }

    // [8] Trace recording.
    #[test]
    fn trace_recording() {
        let r = solve_milp(
            &build_knapsack_milp(vec![10.0, 40.0, 30.0, 50.0], vec![5.0, 4.0, 6.0, 3.0], 10.0),
            MILPSolveOptions::default(),
        );
        assert_eq!(r.trace.len(), r.nodes_explored);
        assert_eq!(r.trace[0].node_id, 0);
        assert_eq!(r.trace[0].depth, 0);
        assert!(r.trace[0].branch_var.is_none());
        assert!(r.trace.iter().any(|e| e.incumbent_updated));
        let branched: Vec<_> = r.trace.iter().filter(|e| !e.pruned).collect();
        assert!(branched.iter().all(|e| !e.fractional.is_empty()));
    }

    // [9] maxNodes early termination.
    #[test]
    fn max_nodes_early_termination() {
        let milp = build_knapsack_milp(
            (0..30).map(|i| 1.0 + i as f64).collect(),
            (0..30).map(|i| 1.0 + i as f64).collect(),
            100.0,
        );
        let r = solve_milp(
            &milp,
            MILPSolveOptions {
                max_nodes: Some(5),
                ..Default::default()
            },
        );
        assert!(r.nodes_explored <= 5);
        assert!(r.status == MILPStatus::Optimal || r.status == MILPStatus::MaxNodes);
    }
}
