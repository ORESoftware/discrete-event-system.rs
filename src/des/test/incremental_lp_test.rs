//! Port of src/des/test/incremental-lp-test.ts
//!
//! Unit tests for the incremental / warm-startable LP solver. The TypeScript
//! manual `check()` tally becomes `#[test]` functions with `assert!`.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::incremental_lp::{
        IncrementalLP, IncrementalLPInit, IncrementalPivotRule, Sense as IncSense, SolverStatus,
    };
    use crate::des::general::lp::{
        solve_lp_internal, InternalSimplexOptions, LPProblem, Sense as LpSense,
    };
    use crate::des::shared::capabilities::{RandomSource, SeededRandom};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn close_tol(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn arr_close(a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(&x, &y)| close_tol(x, y, 1e-7))
    }

    fn base_init() -> IncrementalLPInit {
        IncrementalLPInit {
            sense: IncSense::Max,
            c: vec![3.0, 5.0],
            a: vec![vec![2.0, 1.0], vec![1.0, 3.0]],
            b: vec![100.0, 90.0],
            var_names: None,
            con_names: None,
        }
    }

    fn solve_internal_max(c: Vec<f64>, a_ub: Vec<Vec<f64>>, b_ub: Vec<f64>) -> LPProblem {
        LPProblem {
            sense: LpSense::Max,
            c,
            a_ub: Some(a_ub),
            b_ub: Some(b_ub),
            ..Default::default()
        }
    }

    // [1] Constructor + initial pivot.
    #[test]
    fn constructor_and_initial_pivot() {
        let mut inc = IncrementalLP::new(base_init());
        assert_eq!(inc.num_struct, 2);
        assert_eq!(inc.basis.len(), 2);
        assert_eq!(inc.basis[0], 2);
        assert_eq!(inc.basis[1], 3);
        assert_eq!(inc.status, SolverStatus::Primal);
        assert!(arr_close(&inc.get_x(), &[0.0, 0.0]));
        inc.solve_to_optimum(1000);
        assert_eq!(inc.status, SolverStatus::Optimal);
        assert!(arr_close(&inc.get_x(), &[42.0, 16.0]));
        assert!(close(inc.get_z(), 206.0));
    }

    // [2] Add constraint, dual simplex restart.
    #[test]
    fn add_constraint_dual_restart() {
        let mut inc = IncrementalLP::new(base_init());
        inc.solve_to_optimum(1000);
        inc.apply_add_constraint(&[1.0, 0.0], 30.0, None);
        assert!(inc.status == SolverStatus::Optimal || inc.status == SolverStatus::Dual);
        inc.solve_to_optimum(1000);
        assert_eq!(inc.status, SolverStatus::Optimal);
        assert!(inc.get_x()[0] <= 30.0 + 1e-9);
        inc.apply_add_constraint(&[0.0, 1.0], 10.0, None);
        inc.solve_to_optimum(1000);
        assert!(inc.get_x()[1] <= 10.0 + 1e-9);
    }

    // [3] Remove constraint.
    #[test]
    fn remove_constraint() {
        let mut inc = IncrementalLP::new(IncrementalLPInit {
            sense: IncSense::Max,
            c: vec![3.0, 5.0],
            a: vec![vec![2.0, 1.0], vec![1.0, 3.0], vec![1.0, 0.0]],
            b: vec![100.0, 90.0, 30.0],
            var_names: None,
            con_names: None,
        });
        inc.solve_to_optimum(1000);
        inc.apply_remove_constraint(2);
        inc.solve_to_optimum(1000);
        let stat = solve_lp_internal(
            &solve_internal_max(
                vec![3.0, 5.0],
                vec![vec![2.0, 1.0], vec![1.0, 3.0]],
                vec![100.0, 90.0],
            ),
            &InternalSimplexOptions::default(),
        );
        assert!(close(inc.get_z(), stat.objective));
        assert!(arr_close(&inc.get_x(), &stat.x));
        assert_eq!(inc.tab.len() - 1, 2);
    }

    // [4] Change objective.
    #[test]
    fn change_objective() {
        let mut inc = IncrementalLP::new(IncrementalLPInit {
            sense: IncSense::Max,
            c: vec![1.0, 1.0],
            a: vec![vec![2.0, 1.0], vec![1.0, 3.0]],
            b: vec![100.0, 90.0],
            var_names: None,
            con_names: None,
        });
        inc.solve_to_optimum(1000);
        let z1 = inc.get_z();
        inc.apply_change_objective(&[10.0, 1.0]);
        inc.solve_to_optimum(1000);
        assert!(inc.get_z() > z1);
        let stat = solve_lp_internal(
            &solve_internal_max(
                vec![10.0, 1.0],
                vec![vec![2.0, 1.0], vec![1.0, 3.0]],
                vec![100.0, 90.0],
            ),
            &InternalSimplexOptions::default(),
        );
        assert!(close(inc.get_z(), stat.objective));
        assert!(arr_close(&inc.get_x(), &stat.x));
    }

    // [5] Add variable.
    #[test]
    fn add_variable() {
        let mut inc = IncrementalLP::new(base_init());
        inc.solve_to_optimum(1000);
        inc.apply_add_variable(&[1.0, 1.0], 100.0, None);
        inc.solve_to_optimum(1000);
        assert_eq!(inc.num_struct, 3);
        assert!(inc.basis.contains(&2));
        let stat = solve_lp_internal(
            &solve_internal_max(
                vec![3.0, 5.0, 100.0],
                vec![vec![2.0, 1.0, 1.0], vec![1.0, 3.0, 1.0]],
                vec![100.0, 90.0],
            ),
            &InternalSimplexOptions::default(),
        );
        assert!(arr_close(&inc.get_x(), &stat.x));
        assert!(close(inc.get_z(), stat.objective));
    }

    // [6] Remove variable (non-basic).
    #[test]
    fn remove_variable_non_basic() {
        let mut inc = IncrementalLP::new(IncrementalLPInit {
            sense: IncSense::Max,
            c: vec![3.0, 5.0, -10.0],
            a: vec![vec![2.0, 1.0, 1.0], vec![1.0, 3.0, 1.0]],
            b: vec![100.0, 90.0],
            var_names: None,
            con_names: None,
        });
        inc.solve_to_optimum(1000);
        assert!(!inc.basis.contains(&2));
        let z_before = inc.get_z();
        inc.apply_remove_variable(2);
        assert_eq!(inc.num_struct, 2);
        assert!(close(inc.get_z(), z_before));
        inc.solve_to_optimum(1000);
        let stat = solve_lp_internal(
            &solve_internal_max(
                vec![3.0, 5.0],
                vec![vec![2.0, 1.0], vec![1.0, 3.0]],
                vec![100.0, 90.0],
            ),
            &InternalSimplexOptions::default(),
        );
        assert!(close(inc.get_z(), stat.objective));
    }

    // [7] Remove variable (basic).
    #[test]
    fn remove_variable_basic() {
        let mut inc = IncrementalLP::new(base_init());
        inc.solve_to_optimum(1000);
        assert!(inc.basis.contains(&0));
        inc.apply_remove_variable(0);
        assert_eq!(inc.num_struct, 1);
        inc.solve_to_optimum(1000);
        let stat = solve_lp_internal(
            &solve_internal_max(vec![5.0], vec![vec![1.0], vec![3.0]], vec![100.0, 90.0]),
            &InternalSimplexOptions::default(),
        );
        assert!(close(inc.get_z(), stat.objective));
        assert!(arr_close(&inc.get_x(), &stat.x));
    }

    // [8] Idempotence: same modification twice should not destabilise.
    #[test]
    fn idempotent_objective_change() {
        let mut inc = IncrementalLP::new(base_init());
        inc.solve_to_optimum(1000);
        inc.apply_change_objective(&[3.0, 5.0]);
        inc.solve_to_optimum(1000);
        inc.apply_change_objective(&[3.0, 5.0]);
        inc.solve_to_optimum(1000);
        assert!(close(inc.get_z(), 206.0));
    }

    // [9] Detect unboundedness after constraint removal.
    #[test]
    fn unbounded_after_removal() {
        let mut inc = IncrementalLP::new(IncrementalLPInit {
            sense: IncSense::Max,
            c: vec![1.0, 1.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![10.0],
            var_names: None,
            con_names: None,
        });
        inc.solve_to_optimum(1000);
        assert_eq!(inc.status, SolverStatus::Optimal);
        inc.apply_remove_constraint(0);
        inc.solve_to_optimum(1000);
        assert_eq!(inc.status, SolverStatus::Unbounded);
    }

    // [10] Snapshot integrity.
    #[test]
    fn snapshot_integrity() {
        let mut inc = IncrementalLP::new(base_init());
        inc.solve_to_optimum(1000);
        let snap = inc.snapshot(None, None);
        assert_eq!(snap.num_struct, 2);
        assert_eq!(snap.num_constraints, 2);
        assert!(close(snap.z, inc.get_z()));
        assert!(snap.reduced_costs.iter().all(|&r| r >= -1e-9));
        assert!(snap.rhs.iter().all(|&v| v >= -1e-9));
        assert!(snap.is_optimal);
        assert_eq!(snap.mode, SolverStatus::Optimal);
    }

    // [11] Many modifications agree with static solver throughout.
    #[test]
    fn random_lps_agree_with_static() {
        // The TS `rng(seed)` is mulberry32, identical to `SeededRandom`.
        let mut r = SeededRandom::new(424242);
        let mut ok_count = 0;
        let mut total = 0;
        for _ in 0..12 {
            let n = 2 + (r.next_float() * 3.0).floor() as usize;
            let m = 2 + (r.next_float() * 3.0).floor() as usize;
            let c: Vec<f64> = (0..n)
                .map(|_| 1.0 + (r.next_float() * 9.0).floor())
                .collect();
            let a: Vec<Vec<f64>> = (0..m)
                .map(|_| {
                    (0..n)
                        .map(|_| 1.0 + (r.next_float() * 5.0).floor())
                        .collect()
                })
                .collect();
            let b: Vec<f64> = (0..m)
                .map(|_| 30.0 + (r.next_float() * 40.0).floor())
                .collect();
            let mut inc = IncrementalLP::new(IncrementalLPInit {
                sense: IncSense::Max,
                c: c.clone(),
                a: a.clone(),
                b: b.clone(),
                var_names: None,
                con_names: None,
            });
            inc.solve_to_optimum(1000);
            let stat = solve_lp_internal(
                &solve_internal_max(c, a, b),
                &InternalSimplexOptions::default(),
            );
            total += 1;
            if close_tol(inc.get_z(), stat.objective, 1e-7) {
                ok_count += 1;
            }
        }
        assert_eq!(ok_count, total, "{ok_count}/{total}");
    }

    #[test]
    fn bland_rule_solves_degenerate_cycling_example() {
        // Beale's classic cycling tableau: Dantzig's rule can revisit bases at
        // the origin, while Bland's lowest-index rule must terminate.
        let c = vec![10.0, -57.0, -9.0, -24.0];
        let a = vec![
            vec![0.5, -5.5, -2.5, 9.0],
            vec![0.5, -1.5, -0.5, 1.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ];
        let b = vec![0.0, 0.0, 1.0];
        let mut inc = IncrementalLP::new(IncrementalLPInit {
            sense: IncSense::Max,
            c: c.clone(),
            a: a.clone(),
            b: b.clone(),
            var_names: None,
            con_names: None,
        });
        inc.set_pivot_rule(IncrementalPivotRule::Bland);
        inc.solve_to_optimum(100);

        let stat = solve_lp_internal(
            &solve_internal_max(c, a, b),
            &InternalSimplexOptions::default(),
        );
        assert_eq!(inc.status, SolverStatus::Optimal);
        assert!(close_tol(inc.get_z(), stat.objective, 1e-7));
        assert!(arr_close(&inc.get_x(), &stat.x));
    }
}
