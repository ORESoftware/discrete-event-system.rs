//! Port of src/des/test/argmax-tiebreak-test.ts
//!
//! Verifies that the random tie-breaking in `arg_max_with_tie_break`,
//! `scan_arg_max_tie_break`, and the algorithms that use them distributes
//! uniformly across the tied set (rather than always picking action 0). The
//! `mulberry32` PRNG maps onto `SeededRandom`; the `-1` "no winner" sentinel
//! becomes `None`.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;

    use crate::des::general::des_base::argmax::{
        all_arg_max_ties, arg_max_with_tie_break, choose_random_tied, scan_arg_max_tie_break,
        ARGMAX_EPS_DEFAULT,
    };
    use crate::des::general::des_base::finite_horizon_dp::{
        DPOutcome, DpOptions, DpState, FiniteHorizonDPStation,
    };
    use crate::des::general::des_base::station::{DESStation, StationCore};
    use crate::des::general::mcts::{mcts, ApplyResult, MCTSEnv, MCTSOptions};
    use crate::des::general::milp_bnb::{
        solve_milp, MILPProblem, MILPSolveOptions, MILPStatus, Sense,
    };
    use crate::des::general::prng::mulberry32;
    use crate::des::general::qlearning_des::{QLearningAgent, QLearningOptions};
    use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};
    use crate::des::shared::capabilities::{RandomSource, SeededRandom};

    /// Wrap a seeded mulberry32 into the `Box<dyn Fn() -> f64>` that `VIOptions`
    /// expects (interior mutability so the closure can be `Fn`).
    fn boxed_rng(seed: u32) -> Box<dyn Fn() -> f64> {
        let cell = RefCell::new(mulberry32(seed));
        Box::new(move || cell.borrow_mut().next_float())
    }

    // =========================================================================
    // PURE UTILITY: arg_max_with_tie_break.
    // =========================================================================

    #[test]
    fn arg_max_empty_and_singleton() {
        assert_eq!(
            arg_max_with_tie_break(&[], &mut mulberry32(1), ARGMAX_EPS_DEFAULT),
            None
        );
        assert_eq!(
            arg_max_with_tie_break(&[42.0], &mut mulberry32(1), ARGMAX_EPS_DEFAULT),
            Some(0)
        );
    }

    #[test]
    fn arg_max_unique_winner_always_returned() {
        let mut rng = mulberry32(1);
        for _ in 0..200 {
            assert_eq!(
                arg_max_with_tie_break(&[1.0, 2.0, 3.0, 2.0, 1.0], &mut rng, ARGMAX_EPS_DEFAULT),
                Some(2)
            );
        }
    }

    #[test]
    fn arg_max_five_way_tie_is_uniform() {
        let mut counts = [0usize; 5];
        let trials = 5000;
        let mut tie_rng = mulberry32(42);
        for _ in 0..trials {
            let idx = arg_max_with_tie_break(
                &[7.0, 7.0, 7.0, 7.0, 7.0],
                &mut tie_rng,
                ARGMAX_EPS_DEFAULT,
            )
            .unwrap();
            counts[idx] += 1;
        }
        let expected = trials as f64 / 5.0;
        let sigma = (trials as f64 * (1.0 / 5.0) * (4.0 / 5.0)).sqrt();
        for &c in &counts {
            assert!(
                (c as f64 - expected).abs() <= 4.0 * sigma,
                "counts={:?} expected≈{} σ≈{}",
                counts,
                expected,
                sigma
            );
        }
        assert!(
            counts.iter().all(|&c| c > 0),
            "every index hit at least once"
        );
    }

    #[test]
    fn arg_max_eps_tolerance_treats_near_equal_as_tied() {
        let mut counts2 = [0usize; 3];
        for t in 0..1000u32 {
            let idx = arg_max_with_tie_break(
                &[1.0, 1.0 + 1e-15, 1.0 - 1e-15],
                &mut mulberry32(t + 1),
                ARGMAX_EPS_DEFAULT,
            )
            .unwrap();
            counts2[idx] += 1;
        }
        assert!(counts2.iter().all(|&c| c > 100), "counts={:?}", counts2);
    }

    // =========================================================================
    // scan_arg_max_tie_break.
    // =========================================================================

    #[test]
    fn scan_excludes_neg_infinity() {
        let idx = scan_arg_max_tie_break(
            4,
            |a| if a == 1 { f64::NEG_INFINITY } else { 7.0 },
            &mut mulberry32(1),
            ARGMAX_EPS_DEFAULT,
        );
        assert_ne!(idx, Some(1));
    }

    #[test]
    fn scan_all_neg_infinity_returns_none() {
        let r1 = scan_arg_max_tie_break(
            3,
            |_a| f64::NEG_INFINITY,
            &mut mulberry32(1),
            ARGMAX_EPS_DEFAULT,
        );
        assert_eq!(r1, None);
    }

    #[test]
    fn scan_uniform_over_four_way_tie() {
        let mut counts = [0usize; 4];
        for t in 0..2000u32 {
            let i = scan_arg_max_tie_break(
                4,
                |_a| 1.0,
                &mut mulberry32(t * 31 + 17),
                ARGMAX_EPS_DEFAULT,
            )
            .unwrap();
            counts[i] += 1;
        }
        assert!(
            counts.iter().all(|&c| c > 350 && c < 650),
            "counts={:?}",
            counts
        );
    }

    // =========================================================================
    // all_arg_max_ties / choose_random_tied.
    // =========================================================================

    #[test]
    fn all_ties_and_choose_random_tied() {
        assert_eq!(
            all_arg_max_ties(&[1.0, 3.0, 3.0, 2.0, 3.0], ARGMAX_EPS_DEFAULT),
            vec![1, 2, 4]
        );
        assert_eq!(all_arg_max_ties(&[5.0], ARGMAX_EPS_DEFAULT), vec![0]);
        assert_eq!(
            all_arg_max_ties(&[], ARGMAX_EPS_DEFAULT),
            Vec::<usize>::new()
        );
        assert_eq!(choose_random_tied::<i32>(&[], &mut mulberry32(1)), None);
        assert_eq!(choose_random_tied(&[42], &mut mulberry32(1)), Some(&42));
    }

    // =========================================================================
    // VALUE ITERATION ON SYMMETRIC MDP.
    // =========================================================================

    fn symmetric_spec() -> MDPSpec {
        MDPSpec {
            num_states: 5,
            num_actions: Box::new(|_s| 4),
            outcomes: Box::new(|s, _a| {
                // Every action from state 0 is identical (→ state 4, reward 1);
                // all other transitions lead to state 4 with reward 0.
                if s == 0 {
                    vec![Outcome {
                        prob: 1.0,
                        reward: 1.0,
                        next_state: 4,
                    }]
                } else {
                    vec![Outcome {
                        prob: 1.0,
                        reward: 0.0,
                        next_state: 4,
                    }]
                }
            }),
            is_terminal: Some(Box::new(|s| s == 4)),
            terminal_reward: None,
            state_label: None,
            action_label: None,
        }
    }

    #[test]
    fn symmetric_mdp_deterministic_argmax_picks_action_zero() {
        let det = value_iteration(
            symmetric_spec(),
            VIOptions {
                random_tie_break: false,
                gamma: 0.9,
                ..Default::default()
            },
        );
        assert_eq!(det.policy[0], 0);
    }

    #[test]
    fn symmetric_mdp_random_tie_break_visits_multiple_actions() {
        let mut seen: HashSet<i32> = HashSet::new();
        for seed in 1..=20u32 {
            let r = value_iteration(
                symmetric_spec(),
                VIOptions {
                    random_tie_break: true,
                    gamma: 0.9,
                    rng: boxed_rng(seed),
                    ..Default::default()
                },
            );
            seen.insert(r.policy[0]);
        }
        assert!(seen.len() >= 2, "seen actions: {:?}", seen);
    }

    #[test]
    fn symmetric_mdp_value_function_invariant_under_tie_break() {
        let det = value_iteration(
            symmetric_spec(),
            VIOptions {
                random_tie_break: false,
                gamma: 0.9,
                ..Default::default()
            },
        );
        let rnd = value_iteration(
            symmetric_spec(),
            VIOptions {
                random_tie_break: true,
                gamma: 0.9,
                rng: boxed_rng(123),
                ..Default::default()
            },
        );
        for s in 0..5 {
            assert!(
                (rnd.v[s] - det.v[s]).abs() <= 1e-9,
                "V* mismatch at state {s}"
            );
        }
    }

    // =========================================================================
    // Q-LEARNING GREEDY POLICY ON FRESH AGENT.
    // =========================================================================

    #[test]
    fn fresh_qlearning_agent_greedy_policy_visits_multiple_actions() {
        let mut seen_action: HashSet<usize> = HashSet::new();
        for seed in 1..=30u32 {
            let mut agent = QLearningAgent::new(
                "q",
                QLearningOptions {
                    alpha: 0.1,
                    gamma: 0.95,
                    epsilon: 0.0,
                    epsilon_min: None,
                    epsilon_decay: None,
                    num_states: 3,
                    num_actions: 5,
                    q_init: None,
                },
                Box::new(mulberry32(seed)),
            );
            let pol = agent.greedy_policy();
            seen_action.insert(pol[0]);
        }
        assert!(seen_action.len() >= 3, "seen: {:?}", seen_action);
    }

    // =========================================================================
    // MCTS ON DEGENERATE ENVIRONMENT (all actions identical).
    // =========================================================================

    /// 1-step env: every action gives reward 1 then terminal.
    struct DegenerateEnv;

    impl MCTSEnv<i64> for DegenerateEnv {
        fn num_actions(&self, _s: &i64) -> usize {
            4
        }
        fn apply_action(&self, _s: &i64, _a: usize) -> ApplyResult<i64> {
            ApplyResult {
                next: -1,
                reward: 1.0,
                done: true,
            }
        }
        fn is_terminal(&self, s: &i64) -> bool {
            *s == -1
        }
        fn rollout_depth(&self) -> usize {
            1
        }
        fn gamma(&self) -> f64 {
            1.0
        }
    }

    #[test]
    fn mcts_identical_reward_env_varies_across_seeds() {
        let mut acted: HashSet<usize> = HashSet::new();
        for seed in 1..=30u32 {
            let r = mcts(
                Box::new(DegenerateEnv),
                0_i64,
                MCTSOptions {
                    iterations: 8,
                    ..Default::default()
                },
                mulberry32(seed),
            );
            acted.insert(r.action);
        }
        assert!(acted.len() >= 2, "seen: {:?}", acted);
    }

    // =========================================================================
    // MILP B&B: branch_seed varies the search tree.
    // =========================================================================

    #[test]
    fn milp_symmetric_optimum_invariant_across_branch_seeds() {
        // max x1+x2+x3+x4 s.t. x_j ≤ 1.5, x_j ∈ ℤ → all four LP-relax to 1.5.
        let p = MILPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0, 1.0, 1.0],
            a: vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
            b: vec![1.5, 1.5, 1.5, 1.5],
            integer_vars: vec![true, true, true, true],
            ub: Some(vec![1.5, 1.5, 1.5, 1.5]),
            var_names: None,
            con_names: None,
        };
        let mut z_set: HashSet<i64> = HashSet::new();
        for seed in 1..=10u32 {
            let sol = solve_milp(
                &p,
                MILPSolveOptions {
                    branch_seed: Some(seed),
                    ..Default::default()
                },
            );
            if sol.status == MILPStatus::Optimal && !sol.x.is_empty() {
                z_set.insert((sol.z * 1e6).round() as i64);
            }
        }
        assert_eq!(z_set.len(), 1, "z values: {:?}", z_set);
    }

    // =========================================================================
    // FINITE-HORIZON DP: V_t invariant, π_t varies.
    // =========================================================================

    /// Simple 2-state, 3-action, 5-stage DP where all actions are equally good.
    struct SymDP {
        core: StationCore,
        state: DpState,
    }

    impl SymDP {
        fn new(random_tie_break: bool, rng: Box<dyn RandomSource>) -> Self {
            let mut dp = SymDP {
                core: StationCore::new("sym-dp"),
                state: DpState::new(DpOptions {
                    random_tie_break,
                    rng: Some(rng),
                    ..Default::default()
                }),
            };
            dp.bootstrap();
            dp
        }
    }

    impl DESStation for SymDP {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn run_time_step(&mut self) {
            self.run_dp_step();
        }
        fn has_work(&self) -> bool {
            self.dp_has_work()
        }
    }

    impl FiniteHorizonDPStation for SymDP {
        fn dp_state(&self) -> &DpState {
            &self.state
        }
        fn dp_state_mut(&mut self) -> &mut DpState {
            &mut self.state
        }
        fn horizon(&self) -> usize {
            5
        }
        fn num_states(&self) -> usize {
            2
        }
        fn num_actions(&self, _state: usize, _stage: usize) -> usize {
            3
        }
        fn transitions(&self, state: usize, _action: usize, _stage: usize) -> Vec<DPOutcome> {
            vec![DPOutcome {
                prob: 1.0,
                reward: 1.0,
                next_state: state,
            }]
        }
    }

    fn drive(dp: &mut SymDP) {
        while !dp.is_finished() {
            dp.run_dp_step();
        }
    }

    #[test]
    fn finite_horizon_dp_value_invariant_policy_varies() {
        let mut det = SymDP::new(false, Box::new(SeededRandom::new(1)));
        drive(&mut det);
        let mut rnd = SymDP::new(true, Box::new(mulberry32(7)));
        drive(&mut rnd);

        // V_t unchanged by random tie-break.
        for t in 0..=5 {
            for s in 0..2 {
                assert!(
                    (det.dp_state().v[t][s] - rnd.dp_state().v[t][s]).abs() <= 1e-12,
                    "V mismatch at t={t}, s={s}"
                );
            }
        }

        // Deterministic always picks action 0.
        for t in 0..5 {
            for s in 0..2 {
                assert_eq!(det.get_action(t, s), Some(0), "det π at t={t}, s={s}");
            }
        }

        // Random tie-break: ≥ 2 distinct actions across seeds.
        let mut seen_actions: HashSet<usize> = HashSet::new();
        for seed in 1..=20u32 {
            let mut r = SymDP::new(true, Box::new(mulberry32(seed)));
            drive(&mut r);
            for t in 0..5 {
                for s in 0..2 {
                    if let Some(a) = r.get_action(t, s) {
                        seen_actions.insert(a);
                    }
                }
            }
        }
        assert!(seen_actions.len() >= 2, "seen: {:?}", seen_actions);
    }
}
