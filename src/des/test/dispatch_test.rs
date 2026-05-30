//! Port of src/des/test/dispatch-test.ts
//!
//! Unit tests for the multi-class dispatch combo (`general/dispatch`): the DES
//! simulator invariants, heuristic policies, the fluid LP relaxation (with
//! cross-solver agreement against `lp` and `lp-des`), the MDP-VI policy, and an
//! MCTS sanity check. `mulberry32` maps onto `SeededRandom`.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::dispatch::{
        build_dispatch_fluid_lp, policy_fluid_lp, policy_mdp_vi, policy_round_robin, policy_sect,
        policy_shortest_queue, simulate_dispatch, DispatchPolicy, DispatchProblem, DispatchState,
        MdpViPolicyOptions,
    };
    use crate::des::general::lp::{solve_lp_internal, InternalSimplexOptions, LPStatus};
    use crate::des::general::lp_des::{solve_lp_via_des, DESSimplexOptions};
    use crate::des::general::mcts::{mcts, ApplyResult, MCTSEnv, MCTSOptions};
    use crate::des::general::prng::mulberry32;

    fn problem_2x2() -> DispatchProblem {
        DispatchProblem {
            m: 2,
            k: 2,
            arrival_rate: 1.6,
            class_prob: vec![0.6, 0.4],
            service_rate: vec![vec![2.0, 0.8], vec![0.8, 2.0]],
        }
    }

    fn problem_3x3() -> DispatchProblem {
        DispatchProblem {
            m: 3,
            k: 3,
            arrival_rate: 2.4,
            class_prob: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            service_rate: vec![vec![1.6, 0.9, 0.7], vec![0.7, 1.6, 0.9], vec![0.9, 0.7, 1.6]],
        }
    }

    fn state(m: usize, k: usize, q: Vec<i64>) -> DispatchState {
        DispatchState {
            m,
            k,
            q,
            idle_until: vec![0.0; m],
            in_service: vec![-1; m],
            now: 0.0,
        }
    }

    // Group 1 — DES simulator basic invariants.
    #[test]
    fn simulator_basic_invariants() {
        let mut pol = policy_round_robin();
        let r = simulate_dispatch(&problem_2x2(), &mut pol, 1000, 42, 0);
        assert!(r.completed_jobs > 0);
        assert_eq!(r.per_machine_jobs.iter().sum::<i64>(), 1000);
        assert!(r.mean_sojourn.is_finite() && r.mean_sojourn > 0.0);
        assert!(r.per_machine_utilisation.iter().all(|&u| (0.0..=1.0).contains(&u)));
    }

    // Group 2 — Round-robin distributes evenly.
    #[test]
    fn round_robin_distributes_evenly() {
        let mut pol = policy_round_robin();
        let r = simulate_dispatch(&problem_3x3(), &mut pol, 3000, 7, 0);
        assert!(
            r.per_machine_jobs.iter().all(|&c| (c - 1000).abs() <= 1),
            "counts = {:?}",
            r.per_machine_jobs
        );
    }

    // Group 3 — SECT chooses the class-aligned machine when queues are empty.
    #[test]
    fn sect_class_alignment_and_overflow() {
        let mut policy = policy_sect(&problem_2x2());
        assert_eq!(policy.pick(&state(2, 2, vec![0, 0]), 0), 0);
        assert_eq!(policy.pick(&state(2, 2, vec![0, 0]), 1), 1);
        // q=[5,0]: (5+1)/2.0 = 3.0 > (0+1)/0.8 = 1.25 → overflow to machine 1.
        assert_eq!(policy.pick(&state(2, 2, vec![5, 0]), 0), 1);
    }

    // Group 4 — Shortest-queue picks the lower queue index on ties.
    #[test]
    fn shortest_queue_tie_break() {
        let mut policy = policy_shortest_queue();
        assert_eq!(policy.pick(&state(3, 1, vec![3, 1, 5]), 0), 1);
        assert_eq!(policy.pick(&state(3, 1, vec![0, 0, 2]), 0), 0);
    }

    // Group 5 — Fluid LP shape and feasibility.
    #[test]
    fn fluid_lp_shape() {
        let lp = build_dispatch_fluid_lp(&problem_3x3());
        assert_eq!(lp.c.len(), 10);
        assert_eq!(lp.a_eq.as_ref().map(|m| m.len()).unwrap_or(0), 3);
        assert_eq!(lp.a_ub.as_ref().map(|m| m.len()).unwrap_or(0), 3);
        for i in 0..9 {
            assert_eq!(lp.c[i], 0.0);
        }
        assert_eq!(lp.c[9], 1.0);
    }

    // Group 6 — Fluid LP cross-solver agreement.
    #[test]
    fn fluid_lp_cross_solver_agreement() {
        let lp = build_dispatch_fluid_lp(&problem_3x3());
        let a = solve_lp_internal(&lp, &InternalSimplexOptions::default());
        let b = solve_lp_via_des(&lp, &DESSimplexOptions::default());
        assert_eq!(a.status, LPStatus::Optimal);
        assert_eq!(b.status, LPStatus::Optimal);
        assert!((a.objective - b.objective).abs() <= 1e-9);
        let dx = a
            .x
            .iter()
            .zip(b.x.iter())
            .map(|(v, w)| (v - w).abs())
            .fold(0.0_f64, f64::max);
        assert!(dx < 1e-6, "max |Δx| = {dx}");
    }

    // Group 7 — Fluid LP policy: x* sums to 1 per class, ≥ 0.
    #[test]
    fn fluid_lp_policy_shape() {
        let p = problem_3x3();
        let r = policy_fluid_lp(&p, 0);
        for c in 0..p.k {
            let sum: f64 = r.x[c].iter().sum();
            assert!((sum - 1.0).abs() <= 1e-6, "class {c}: Σ = {sum}");
            assert!(r.x[c].iter().all(|&v| v >= -1e-9));
        }
        assert!(r.bottleneck_load > 0.0 && r.bottleneck_load <= 1.0, "t* = {}", r.bottleneck_load);
    }

    // Group 8 — MDP-VI policy returns a legal action at every state.
    #[test]
    fn mdp_vi_legal_actions() {
        let mut r = policy_mdp_vi(
            &problem_2x2(),
            MdpViPolicyOptions {
                q_max: Some(3),
                gamma: Some(0.95),
                rollouts_per_sa: Some(30),
                seed: Some(1),
                ..Default::default()
            },
        );
        assert_eq!(r.num_states, 4usize.pow(2) * 2);
        for q1 in 0..=3 {
            for q2 in 0..=3 {
                for c in 0..2 {
                    let a = r.policy.pick(&state(2, 2, vec![q1, q2]), c);
                    assert!(a == 0 || a == 1, "got {a}");
                }
            }
        }
    }

    // Group 9 — MCTS finds the obvious optimal action.
    struct TwoActionEnv;
    impl MCTSEnv<i64> for TwoActionEnv {
        fn num_actions(&self, _s: &i64) -> usize {
            2
        }
        fn apply_action(&self, s: &i64, a: usize) -> ApplyResult<i64> {
            ApplyResult {
                next: s + 1,
                reward: if a == 0 { 1.0 } else { 10.0 },
                done: s + 1 >= 1,
            }
        }
        fn is_terminal(&self, s: &i64) -> bool {
            *s >= 1
        }
        fn rollout_depth(&self) -> usize {
            0
        }
        fn gamma(&self) -> f64 {
            1.0
        }
    }

    #[test]
    fn mcts_picks_higher_reward_action() {
        let result = mcts(
            Box::new(TwoActionEnv),
            0_i64,
            MCTSOptions { iterations: 50, ..Default::default() },
            mulberry32(7),
        );
        assert_eq!(result.action, 1, "got {}", result.action);
    }

    // Group 10 — Reproducibility: same seed ⇒ same result.
    #[test]
    fn reproducible_same_seed() {
        let mut pa = policy_sect(&problem_2x2());
        let a = simulate_dispatch(&problem_2x2(), &mut pa, 500, 1234, 0);
        let mut pb = policy_sect(&problem_2x2());
        let b = simulate_dispatch(&problem_2x2(), &mut pb, 500, 1234, 0);
        assert_eq!(a.mean_sojourn, b.mean_sojourn);
        assert_eq!(a.per_machine_jobs, b.per_machine_jobs);
    }

    // Group 11 — MDP-VI reward-model alignment.
    #[test]
    fn mdp_vi_reward_alignment() {
        let mut r = policy_mdp_vi(
            &problem_2x2(),
            MdpViPolicyOptions {
                q_max: Some(4),
                gamma: Some(0.95),
                rollouts_per_sa: Some(100),
                seed: Some(2),
                ..Default::default()
            },
        );
        assert_eq!(r.policy.pick(&state(2, 2, vec![0, 0]), 0), 0);
        assert_eq!(r.policy.pick(&state(2, 2, vec![0, 0]), 1), 1);
    }
}
