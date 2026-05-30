//! Port of src/des/test/optimization-as-des-test.ts
//!
//! Tests the four concrete "algorithm-as-DES" leaves — simulated annealing, hill
//! climbing, genetic algorithm (TSP) and the RL agents (Q-learning, PPO) — plus
//! the shared episode-accounting helpers.
//!
//! PORT NOTE: the TS groups that subclass `DESStation`, `SingleStateOptimizer`,
//! `PopulationOptimizer` and `EnvironmentStation` to probe template-method /
//! channel / runner machinery (groups 1, 2, the source-seeding parts of 3 & 4,
//! and the `EnvironmentStation` token-semantics part of 7) rely on TS subclass
//! overrides of protected hooks. Those have no direct Rust analogue here and are
//! deferred; the algorithm leaves and accounting helpers are ported faithfully.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::des_base::environment::{PureEnvironment, StepResult};
    use crate::des::general::des_base::episode_accounting::{
        EpisodeAccounting, VectorEpisodeAccounting,
    };
    use crate::des::general::ga_des::{run_tsp_ga_des, TSPGAOptions};
    use crate::des::general::genetic_tsp::{
        build_pentagon_tsp, build_random_tsp, held_karp_exact, is_permutation, tour_length,
        InitMode,
    };
    use crate::des::general::ppo_des::{run_ppo_des, RunPPOOptions};
    use crate::des::general::qlearning_des::{run_qlearning_des, RunQLearningOptions};
    use crate::des::general::rl_environments::{
        eval_policy, Corridor, Environment, EvalPolicyOptions, GridWorld, GridWorldOptions,
    };
    use crate::des::general::sa_des::{
        run_tsp_hill_climber_des, run_tsp_sa_des, CoolingSchedule, TSPSAOptions,
    };
    use crate::des::shared::capabilities::SeededRandom;

    // Bridges `Environment` to the `PureEnvironment<usize, usize>` the tabular
    // Q-learning / PPO runners require.
    struct PureEnvAdapter<E: Environment> {
        env: E,
    }

    impl<E: Environment> PureEnvironment<usize, usize> for PureEnvAdapter<E> {
        fn num_states(&self) -> usize {
            self.env.num_states()
        }
        fn num_actions(&self) -> usize {
            self.env.num_actions()
        }
        fn reset(&mut self) -> usize {
            self.env.reset()
        }
        fn step(&mut self, state: usize, action: usize) -> StepResult<usize> {
            let o = self.env.step(state, action);
            StepResult {
                next_state: o.next_state,
                reward: o.reward,
                done: o.done,
            }
        }
    }

    fn pentagon_cooling() -> CoolingSchedule {
        CoolingSchedule::Geometric {
            t0: 50.0,
            alpha: 0.998,
            t_min: None,
        }
    }

    fn sa_options() -> TSPSAOptions {
        TSPSAOptions {
            cooling: pentagon_cooling(),
            max_iterations: 2000,
            seed: 1,
            init: None,
            moves: None,
            penalty_per_violation: None,
            trace_stride: None,
            stall_limit: None,
        }
    }

    // SingleStateOptimizer leaf: SA on a pentagon TSP reaches the exact optimum.
    #[test]
    fn sa_pentagon_optimum() {
        let inst = build_pentagon_tsp(5, 50.0);
        let n = inst.n;
        let opt = tour_length(&inst, &[0, 1, 2, 3, 4]);
        let sa = run_tsp_sa_des(inst, sa_options(), None);
        assert!((sa.best_cost - opt).abs() < 1e-9);
        for i in 1..sa.best_history.len() {
            assert!(sa.best_history[i] <= sa.best_history[i - 1] + 1e-12);
        }
        assert!(is_permutation(&sa.best_tour, n));
    }

    // Hill-climber leaf: reaches the pentagon optimum and accepts only strict
    // improvements (accepted == improved).
    #[test]
    fn hill_climber_pentagon() {
        let inst = build_pentagon_tsp(5, 50.0);
        let opt = tour_length(&inst, &[0, 1, 2, 3, 4]);
        let hc = run_tsp_hill_climber_des(inst, sa_options(), None);
        assert!((hc.best_cost - opt).abs() < 1e-9);
        assert_eq!(hc.accepted_count, hc.improve_count);
    }

    // PopulationOptimizer leaf: GA finds a near-optimal tour on n = 8.
    #[test]
    fn ga_near_optimal() {
        let inst = build_random_tsp(8, 17, None);
        let n = inst.n;
        let exact = held_karp_exact(&inst);
        let ga = run_tsp_ga_des(
            inst,
            TSPGAOptions {
                pop_size: 40,
                num_generations: 80,
                tournament_size: None,
                crossover_prob: None,
                mutation_prob: None,
                elitism: Some(2),
                seed: 1,
                init: Some(InitMode::NearestNeighbor),
                penalty_per_violation: None,
            },
            None,
        );
        assert!(ga.best_length <= exact.length * 1.05);
        for i in 1..ga.best_history.len() {
            assert!(ga.best_history[i] <= ga.best_history[i - 1] + 1e-12);
        }
        assert!(is_permutation(&ga.best_tour, n));
        assert_eq!(ga.best_history.len(), ga.generations + 1);
    }

    // RLAgent leaf: Q-learning on GridWorld matches the Bellman-optimal value.
    #[test]
    fn qlearning_gridworld() {
        let env = GridWorld::new(GridWorldOptions::default());
        let opt = env.optimal_v(0.95, 1e-9, 5000);
        let ql = run_qlearning_des(
            Box::new(PureEnvAdapter {
                env: GridWorld::new(GridWorldOptions::default()),
            }),
            RunQLearningOptions {
                num_episodes: 500.0,
                alpha: 0.3,
                gamma: 0.95,
                epsilon: 0.8,
                epsilon_min: Some(0.05),
                epsilon_decay: Some(0.99),
                max_steps_per_episode: Some(50),
                seed: Some(1),
                des_options: None,
            },
        );
        let max_q0 = ql.q[0].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((max_q0 - opt.v[0]).abs() < 0.01);

        let mut rng = SeededRandom::new(1);
        let eval_q = eval_policy(
            &env,
            |s, _rng| ql.policy[s],
            &mut rng,
            EvalPolicyOptions {
                num_episodes: 100,
                max_steps_per_episode: 100,
                gamma: 0.95,
            },
        );
        assert_eq!(eval_q.success_rate, 1.0);
        assert_eq!(ql.total_episodes, 500);
        assert_eq!(ql.reward_history.len(), 500);
    }

    // PolicyGradientAgent leaf: PPO on Corridor(8) approaches the optimal value.
    #[test]
    fn ppo_corridor() {
        let env = Corridor::new(8, 0);
        let opt = env.optimal_v(0.95, 1e-9, 5000);
        let ppo = run_ppo_des(
            Box::new(PureEnvAdapter {
                env: Corridor::new(8, 0),
            }),
            RunPPOOptions {
                total_steps: 8000,
                rollout_len: 64,
                num_epochs: 6,
                mini_batch_size: 16,
                policy_lr: 0.05,
                value_lr: 0.1,
                gamma: 0.95,
                lambda: 0.95,
                clip_eps: 0.2,
                entropy_coef: Some(0.01),
                normalise_advantage: None,
                max_steps_per_episode: Some(30),
                seed: Some(1),
                des_options: None,
            },
        );
        assert!((ppo.v[0] - opt.v[0]).abs() < 0.05);
        assert_eq!(ppo.policy[0], 1);

        let mut rng = SeededRandom::new(1);
        let eval_p = eval_policy(
            &env,
            |s, _rng| ppo.policy[s],
            &mut rng,
            EvalPolicyOptions {
                num_episodes: 50,
                max_steps_per_episode: 30,
                gamma: 0.95,
            },
        );
        assert_eq!(eval_p.success_rate, 1.0);
        assert!(ppo.total_updates >= 100);
        assert!(ppo.total_steps >= 8000);
    }

    // Shared episode-accounting helpers (scalar and vector).
    #[test]
    fn episode_accounting_helpers() {
        let mut scalar = EpisodeAccounting::new();
        scalar.record_step(2.0);
        scalar.record_step(-0.5);
        let done = scalar.finish_episode();
        assert!(done.reward == 1.5 && done.length == 2.0);
        assert!(scalar.reward_history[0] == 1.5);
        assert!(scalar.length_history[0] == 2.0);
        assert_eq!(scalar.total_steps, 2);

        let mut vector = VectorEpisodeAccounting::new(2);
        vector.record_step(&[1.0, -1.0]);
        vector.record_step(&[0.5, 2.0]);
        let v_done = vector.finish_episode(2.0);
        assert_eq!(v_done.rewards, vec![1.5, 1.0]);
        assert_eq!(vector.reward_history, vec![vec![1.5, 1.0]]);
        assert_eq!(vector.length_history[0], 2.0);
        assert_eq!(vector.total_steps, 2);
    }
}
