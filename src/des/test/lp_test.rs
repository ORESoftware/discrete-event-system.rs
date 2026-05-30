//! Port of src/des/test/lp-test.ts
//!
//! Unit tests for the LP infrastructure: the in-process simplex, the
//! DES-engine simplex, and the MDP-as-LP transformation. The TypeScript file
//! spanned `lp`, `lp-des`, `des-lp-bridge` and `value-iteration`; the Rust
//! port keeps every assertion, grouped into `#[test]` functions.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::des_lp_bridge::{
        build_mdp_lp, solve_mdp_as_lp, MdpAsLpOptions,
    };
    use crate::des::general::lp::{
        lp_to_string, solve_lp_internal, InternalSimplexOptions, LPProblem, LPStatus, Sense,
    };
    use crate::des::general::lp_des::{solve_lp_via_des, DESSimplexOptions, PivotRule};
    use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn approx7(a: f64, b: f64) -> bool {
        approx(a, b, 1e-7)
    }

    fn max_abs(u: &[f64], v: &[f64]) -> f64 {
        let mut m = 0.0;
        for i in 0..u.len() {
            m = f64::max(m, (u[i] - v[i]).abs());
        }
        m
    }

    fn iopts() -> InternalSimplexOptions {
        InternalSimplexOptions::default()
    }

    // =========================================================================
    // Unit: in-process simplex, hand-checkable LPs.
    // =========================================================================

    #[test]
    fn internal_simplex_trivial_max() {
        // max x  s.t. x ≤ 5  → x* = 5.
        let lp = LPProblem {
            sense: Sense::Max,
            c: vec![1.0],
            a_ub: Some(vec![vec![1.0]]),
            b_ub: Some(vec![5.0]),
            ..Default::default()
        };
        let r = solve_lp_internal(&lp, &iopts());
        assert_eq!(r.status, LPStatus::Optimal);
        assert!(approx7(r.x[0], 5.0), "x={}", r.x[0]);
    }

    #[test]
    fn internal_simplex_two_var_vertex() {
        // max 3x+2y s.t. x+y≤4, x+3y≤6 → (4, 0), obj 12.
        let lp = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 3.0]]),
            b_ub: Some(vec![4.0, 6.0]),
            ..Default::default()
        };
        let r = solve_lp_internal(&lp, &iopts());
        assert_eq!(r.status, LPStatus::Optimal);
        assert!(approx7(r.x[0], 4.0));
        assert!(approx7(r.x[1], 0.0));
        assert!(approx7(r.objective, 12.0));
    }

    #[test]
    fn internal_simplex_equality() {
        // x + y = 1, max x → x*=1, y*=0.
        let lp = LPProblem {
            sense: Sense::Max,
            c: vec![1.0, 0.0],
            a_eq: Some(vec![vec![1.0, 1.0]]),
            b_eq: Some(vec![1.0]),
            ..Default::default()
        };
        let r = solve_lp_internal(&lp, &iopts());
        assert_eq!(r.status, LPStatus::Optimal);
        assert!(approx7(r.x[0], 1.0));
        assert!(approx7(r.x[1], 0.0));
    }

    #[test]
    fn internal_simplex_infeasible() {
        // x ≥ 5 AND x ≤ 3 (as -x ≤ -5 and x ≤ 3).
        let lp = LPProblem {
            sense: Sense::Max,
            c: vec![1.0],
            a_ub: Some(vec![vec![-1.0], vec![1.0]]),
            b_ub: Some(vec![-5.0, 3.0]),
            ..Default::default()
        };
        let r = solve_lp_internal(&lp, &iopts());
        assert_eq!(r.status, LPStatus::Infeasible);
    }

    #[test]
    fn internal_simplex_unbounded() {
        // max x s.t. -x ≤ 1 (x ≥ -1, no upper bound).
        let lp = LPProblem {
            sense: Sense::Max,
            c: vec![1.0],
            a_ub: Some(vec![vec![-1.0]]),
            b_ub: Some(vec![1.0]),
            ..Default::default()
        };
        let r = solve_lp_internal(&lp, &iopts());
        assert_eq!(r.status, LPStatus::Unbounded);
    }

    #[test]
    fn internal_simplex_min_form() {
        // min x  s.t. x ≥ 2 (as -x ≤ -2) → x*=2.
        let lp = LPProblem {
            sense: Sense::Min,
            c: vec![1.0],
            a_ub: Some(vec![vec![-1.0]]),
            b_ub: Some(vec![-2.0]),
            ..Default::default()
        };
        let r = solve_lp_internal(&lp, &iopts());
        assert_eq!(r.status, LPStatus::Optimal);
        assert!(approx7(r.x[0], 2.0));
    }

    // =========================================================================
    // Unit: lp_to_string pretty-printer.
    // =========================================================================

    #[test]
    fn pretty_printer_contains_expected_terms() {
        let lp = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 3.0]]),
            b_ub: Some(vec![4.0, 6.0]),
            var_names: Some(vec!["x".to_string(), "y".to_string()]),
            ..Default::default()
        };
        let s = lp_to_string(&lp);
        assert!(s.contains("max"));
        assert!(s.contains("x + y ≤ 4"));
        assert!(s.contains("x + 3y ≤ 6"));
    }

    // =========================================================================
    // Unit: DES-engine simplex equivalence with in-process simplex.
    // =========================================================================

    #[test]
    fn des_simplex_matches_internal_simplex() {
        struct Case {
            name: &'static str,
            lp: LPProblem,
        }
        let cases = vec![
            Case {
                name: "simple max",
                lp: LPProblem {
                    sense: Sense::Max,
                    c: vec![3.0, 2.0],
                    a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 3.0]]),
                    b_ub: Some(vec![4.0, 6.0]),
                    ..Default::default()
                },
            },
            Case {
                name: "redundant constraints",
                lp: LPProblem {
                    sense: Sense::Max,
                    c: vec![1.0, 1.0],
                    a_ub: Some(vec![vec![1.0, 1.0], vec![2.0, 2.0]]),
                    b_ub: Some(vec![3.0, 6.0]),
                    ..Default::default()
                },
            },
            Case {
                name: "min with phase-1",
                lp: LPProblem {
                    sense: Sense::Min,
                    c: vec![1.0, 1.0],
                    a_ub: Some(vec![vec![-1.0, -1.0]]),
                    b_ub: Some(vec![-1.0]),
                    ..Default::default()
                },
            },
            Case {
                name: "equality constraints",
                lp: LPProblem {
                    sense: Sense::Max,
                    c: vec![1.0, 1.0],
                    a_eq: Some(vec![vec![1.0, 1.0]]),
                    b_eq: Some(vec![3.0]),
                    ..Default::default()
                },
            },
        ];
        for case in &cases {
            let internal = solve_lp_internal(&case.lp, &iopts());
            let des_d = solve_lp_via_des(
                &case.lp,
                &DESSimplexOptions {
                    pivot_rule: Some(PivotRule::Dantzig),
                    ..Default::default()
                },
            );
            let des_b = solve_lp_via_des(
                &case.lp,
                &DESSimplexOptions {
                    pivot_rule: Some(PivotRule::Bland),
                    ..Default::default()
                },
            );
            assert_eq!(des_d.status, internal.status, "case '{}'", case.name);
            assert_eq!(des_b.status, internal.status, "case '{}'", case.name);
            if internal.status == LPStatus::Optimal {
                assert!(
                    approx(des_d.objective, internal.objective, 1e-9),
                    "case '{}' dantzig",
                    case.name
                );
                assert!(
                    approx(des_b.objective, internal.objective, 1e-9),
                    "case '{}' bland",
                    case.name
                );
            }
        }
    }

    // =========================================================================
    // Unit: DES simplex pivot trace is monotone in obj (phase 2).
    // =========================================================================

    #[test]
    fn des_simplex_phase2_trace_is_monotone() {
        let lp = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 3.0], vec![2.0, 1.0]]),
            b_ub: Some(vec![4.0, 6.0, 7.0]),
            ..Default::default()
        };
        let r = solve_lp_via_des(
            &lp,
            &DESSimplexOptions {
                pivot_rule: Some(PivotRule::Dantzig),
                ..Default::default()
            },
        );
        assert!(!r.trace.pivot_history.is_empty(), "pivot trace exists");
        let mut monotone = true;
        let mut prev = f64::NEG_INFINITY;
        for p in &r.trace.pivot_history {
            if p.phase != 2 {
                continue;
            }
            if p.obj < prev - 1e-9 {
                monotone = false;
            }
            prev = p.obj;
        }
        assert!(monotone, "phase-2 objective trace is non-decreasing");
        assert!(approx7(r.objective, prev), "final obj equals optimum");
    }

    // =========================================================================
    // Unit: MDP-as-LP build_mdp_lp shape check.
    // =========================================================================

    fn mdp_3state_2action() -> MDPSpec {
        MDPSpec {
            num_states: 3,
            num_actions: Box::new(|_s| 2),
            outcomes: Box::new(|s, a| {
                if s == 2 {
                    return vec![Outcome {
                        prob: 1.0,
                        reward: 0.0,
                        next_state: 2,
                    }];
                }
                let target = if a == 1 { (s + 1).min(2) } else { s.saturating_sub(1) };
                let reward = if target == 2 { 1.0 } else { 0.0 };
                vec![Outcome {
                    prob: 1.0,
                    reward,
                    next_state: target,
                }]
            }),
            is_terminal: Some(Box::new(|s| s == 2)),
            terminal_reward: Some(Box::new(|_s| 0.0)),
            state_label: None,
            action_label: None,
        }
    }

    #[test]
    fn build_mdp_lp_shape() {
        let lp = build_mdp_lp(&mdp_3state_2action(), 0.9, None);
        assert_eq!(lp.c.len(), 3, "LP has 3 variables (one per state)");
        assert_eq!(lp.sense, Sense::Min, "LP minimises uniform sum of V");
        assert!(
            lp.c.iter().all(|&v| approx7(v, 1.0 / 3.0)),
            "μ_s = 1/N for uniform stationary measure"
        );
        // Two ≤ rows per non-terminal state (×2 actions) + 2 rows terminal pin = 6.
        let num_constraints = lp.a_ub.as_ref().map(|m| m.len()).unwrap_or(0);
        assert_eq!(num_constraints, 2 * 2 + 2 * 1, "got {num_constraints}");
    }

    // =========================================================================
    // Unit: solve_mdp_as_lp ≡ value_iteration on a tiny MDP.
    // =========================================================================

    fn mdp_4state_line() -> MDPSpec {
        MDPSpec {
            num_states: 4,
            num_actions: Box::new(|_s| 2),
            outcomes: Box::new(|s, a| {
                if s == 3 {
                    return vec![Outcome {
                        prob: 1.0,
                        reward: 0.0,
                        next_state: 3,
                    }];
                }
                let intended = if a == 1 { (s + 1).min(3) } else { s.saturating_sub(1) };
                let slip = if a == 1 { s.saturating_sub(1) } else { (s + 1).min(3) };
                let r = |sp: usize| if sp == 3 { 1.0 } else { 0.0 };
                vec![
                    Outcome {
                        prob: 0.8,
                        reward: r(intended),
                        next_state: intended,
                    },
                    Outcome {
                        prob: 0.2,
                        reward: r(slip),
                        next_state: slip,
                    },
                ]
            }),
            is_terminal: Some(Box::new(|s| s == 3)),
            terminal_reward: Some(Box::new(|_s| 0.0)),
            state_label: None,
            action_label: None,
        }
    }

    #[test]
    fn solve_mdp_as_lp_matches_value_iteration() {
        // Force the in-process simplex so the test needs no python/scipy.
        std::env::set_var("LP_SOLVER", "internal");
        let vi = value_iteration(
            mdp_4state_line(),
            VIOptions {
                gamma: 0.9,
                tol: 1e-12,
                max_iter: 100_000,
                ..Default::default()
            },
        );
        let lp = solve_mdp_as_lp(&mdp_4state_line(), 0.9, &MdpAsLpOptions::default())
            .expect("LP should solve to optimality");
        assert!(
            max_abs(&lp.v, &vi.v) < 1e-7,
            "V*_LP ≡ V*_VI (max|Δ|={})",
            max_abs(&lp.v, &vi.v)
        );
        for s in 0..4 {
            if s != 3 {
                assert_eq!(lp.policy[s], vi.policy[s], "π* mismatch at state {s}");
            }
        }
    }
}
