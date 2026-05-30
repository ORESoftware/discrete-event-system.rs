//! Port of `src/des/main-optimization-as-des.ts`.
//!
//! Runs all four "algorithm-as-DES" families (SA, hill-climb, GA, Q-learning,
//! PPO) on small comparable problems and prints comparison tables. All share the
//! same `run_iterative_des` runner and channel mechanics; only the algorithmic
//! hooks differ.
//!
//! Conversion notes:
//!   - every algorithm is seeded for reproducibility.
//!   - top-level studies → [`run`].
//!   - delegates to `general::{sa_des, ga_des, qlearning_des, ppo_des,
//!     genetic_tsp, rl_environments}`.

use crate::des::general::des_base::environment::{PureEnvironment, StepResult};
use crate::des::general::ga_des::{run_tsp_ga_des, TSPGAOptions};
use crate::des::general::genetic_tsp::{
    build_pentagon_tsp, build_random_tsp, held_karp_exact, is_permutation, tour_length, InitMode,
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

// PORT NOTE: the `rl_environments` envs implement the pure `Environment` trait
// but the algorithm-as-DES runners require `des_base::PureEnvironment<usize,
// usize>`. This thin local adapter bridges the two. Remove once the crate
// exposes a canonical adapter.
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
        StepResult { next_state: o.next_state, reward: o.reward, done: o.done }
    }
}

fn fmt(x: f64, n: usize) -> String {
    format!("{:.*}", n, x)
}

fn pct(x: f64) -> String {
    format!("{:.2}%", 100.0 * x)
}

// -----------------------------------------------------------------------------
// STUDY 1 — TSP n=5 pentagon (exact optimum known)
// -----------------------------------------------------------------------------

fn tsp_pentagon_study() {
    println!();
    println!("=== STUDY 1 ─ Pentagon TSP (n=5, exact = perimeter) ─ algorithm comparison");
    let inst = build_pentagon_tsp(5, 50.0);
    let n = inst.n;
    let opt = tour_length(&inst, &[0, 1, 2, 3, 4]);

    let sa = run_tsp_sa_des(
        inst.clone(),
        TSPSAOptions {
            cooling: CoolingSchedule::Geometric { t0: 50.0, alpha: 0.998, t_min: None },
            max_iterations: 3000,
            seed: 1,
            init: None,
            moves: None,
            penalty_per_violation: None,
            trace_stride: None,
            stall_limit: None,
        },
        None,
    );
    let hc = run_tsp_hill_climber_des(
        inst.clone(),
        TSPSAOptions {
            cooling: CoolingSchedule::Geometric { t0: 50.0, alpha: 0.998, t_min: None },
            max_iterations: 3000,
            seed: 1,
            init: None,
            moves: None,
            penalty_per_violation: None,
            trace_stride: None,
            stall_limit: None,
        },
        None,
    );
    let ga = run_tsp_ga_des(
        inst.clone(),
        TSPGAOptions {
            pop_size: 30,
            num_generations: 80,
            tournament_size: None,
            crossover_prob: None,
            mutation_prob: None,
            elitism: None,
            seed: 1,
            init: None,
            penalty_per_violation: None,
        },
        None,
    );

    println!("  algo                length        gap          ticks  hooks invoked");
    println!(
        "  SA                  {}  {}     {:>5}  {} iter, {} acc, {} impr",
        fmt(sa.best_cost, 4),
        pct(sa.best_cost / opt - 1.0),
        sa.ticks,
        sa.iterations,
        sa.accepted_count,
        sa.improve_count
    );
    println!(
        "  HC                  {}  {}     {:>5}  {} iter, {} acc",
        fmt(hc.best_cost, 4),
        pct(hc.best_cost / opt - 1.0),
        hc.ticks,
        hc.iterations,
        hc.accepted_count
    );
    println!(
        "  GA                  {}  {}     {:>5}  gens={}",
        fmt(ga.best_length, 4),
        pct(ga.best_length / opt - 1.0),
        ga.ticks,
        ga.generations
    );
    println!("  optimum             {}", fmt(opt, 4));
    if !is_permutation(&sa.best_tour, n) {
        panic!("SA tour invalid");
    }
    if !is_permutation(&ga.best_tour, n) {
        panic!("GA tour invalid");
    }
}

// -----------------------------------------------------------------------------
// STUDY 2 — TSP n=12 random vs Held-Karp
// -----------------------------------------------------------------------------

fn tsp_random12_study() {
    println!();
    println!("=== STUDY 2 ─ Random TSP (n=12) ─ SA / HC / GA vs Held-Karp");
    let inst = build_random_tsp(12, 17, None);
    let exact = held_karp_exact(&inst);

    let sa = run_tsp_sa_des(
        inst.clone(),
        TSPSAOptions {
            cooling: CoolingSchedule::Geometric { t0: 100.0, alpha: 0.998, t_min: None },
            max_iterations: 5000,
            seed: 1,
            init: None,
            moves: None,
            penalty_per_violation: None,
            trace_stride: None,
            stall_limit: None,
        },
        None,
    );
    let hc = run_tsp_hill_climber_des(
        inst.clone(),
        TSPSAOptions {
            cooling: CoolingSchedule::Geometric { t0: 100.0, alpha: 0.998, t_min: None },
            max_iterations: 5000,
            seed: 1,
            init: None,
            moves: None,
            penalty_per_violation: None,
            trace_stride: None,
            stall_limit: None,
        },
        None,
    );
    let ga = run_tsp_ga_des(
        inst.clone(),
        TSPGAOptions {
            pop_size: 50,
            num_generations: 200,
            tournament_size: None,
            crossover_prob: None,
            mutation_prob: None,
            elitism: None,
            seed: 1,
            init: Some(InitMode::NearestNeighbor),
            penalty_per_violation: None,
        },
        None,
    );

    println!("  algo                length        gap          ticks");
    println!(
        "  SA                  {}  {}     {}",
        fmt(sa.best_cost, 4),
        pct(sa.best_cost / exact.length - 1.0),
        sa.ticks
    );
    println!(
        "  HC                  {}  {}     {}",
        fmt(hc.best_cost, 4),
        pct(hc.best_cost / exact.length - 1.0),
        hc.ticks
    );
    println!(
        "  GA                  {}  {}     {}",
        fmt(ga.best_length, 4),
        pct(ga.best_length / exact.length - 1.0),
        ga.ticks
    );
    println!("  Held-Karp (exact)   {}", fmt(exact.length, 4));
}

// -----------------------------------------------------------------------------
// STUDY 3 — GridWorld via Q-learning
// -----------------------------------------------------------------------------

fn grid_world_study() {
    println!();
    println!("=== STUDY 3 ─ 4x4 GridWorld ─ Q-learning vs Bellman-optimal V*");
    let env = GridWorld::new(GridWorldOptions::default());
    let opt = env.optimal_v(0.95, 1e-9, 5000);
    let ql = run_qlearning_des(
        Box::new(PureEnvAdapter { env: GridWorld::new(GridWorldOptions::default()) }),
        RunQLearningOptions {
            num_episodes: 600.0,
            alpha: 0.3,
            gamma: 0.95,
            epsilon: 0.8,
            epsilon_decay: Some(0.995),
            epsilon_min: Some(0.05),
            max_steps_per_episode: Some(50),
            seed: Some(1),
            des_options: None,
        },
    );
    let mut rng = SeededRandom::new(1);
    let eval_q = eval_policy(
        &env,
        |s, _rng| ql.policy[s],
        &mut rng,
        EvalPolicyOptions { num_episodes: 200, max_steps_per_episode: 100, gamma: 0.95 },
    );
    println!("  state    optimal V*    learned max_a Q[s,a]    optimal a*    learned a*");
    for s in 0..env.num_states {
        let v = ql.q[s].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  {:>5}    {:>10}    {:>20}    {:>10}    {:>10}",
            s,
            fmt(opt.v[s], 3),
            fmt(v, 3),
            opt.pi[s],
            ql.policy[s]
        );
    }
    println!(
        "  episodes={}  steps={}  ticks={}",
        ql.total_episodes, ql.total_steps, ql.total_ticks
    );
    println!(
        "  greedy success={}  meanReturn={}  optimalReturn={}",
        pct(eval_q.success_rate),
        fmt(eval_q.mean_return, 3),
        fmt(opt.v[0], 3)
    );
}

// -----------------------------------------------------------------------------
// STUDY 4 — Corridor via PPO
// -----------------------------------------------------------------------------

fn corridor_study() {
    println!();
    println!("=== STUDY 4 ─ Corridor(8) ─ PPO vs Bellman-optimal V*");
    let env = Corridor::new(8, 0);
    let opt = env.optimal_v(0.95, 1e-9, 5000);
    let ppo = run_ppo_des(
        Box::new(PureEnvAdapter { env: Corridor::new(8, 0) }),
        RunPPOOptions {
            total_steps: 10_000,
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
    let mut rng = SeededRandom::new(1);
    let eval_p = eval_policy(
        &env,
        |s, _rng| ppo.policy[s],
        &mut rng,
        EvalPolicyOptions { num_episodes: 200, max_steps_per_episode: 30, gamma: 0.95 },
    );
    println!("  state    optimal V*    PPO V_φ(s)    optimal a*    PPO a*");
    for s in 0..env.num_states {
        println!(
            "  {:>5}    {:>10}    {:>10}    {:>10}    {:>7}",
            s,
            fmt(opt.v[s], 3),
            fmt(ppo.v[s], 3),
            opt.pi[s],
            ppo.policy[s]
        );
    }
    println!(
        "  episodes={}  steps={}  updates={}  ticks={}",
        ppo.total_episodes, ppo.total_steps, ppo.total_updates, ppo.total_ticks
    );
    println!(
        "  greedy success={}  meanReturn={}  optimalReturn={}",
        pct(eval_p.success_rate),
        fmt(eval_p.mean_return, 3),
        fmt(opt.v[0], 3)
    );
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    println!("=== optimization-as-DES — SA, HC, GA, Q-learning, PPO ─ all on the same engine");
    tsp_pentagon_study();
    tsp_random12_study();
    grid_world_study();
    corridor_study();
    println!();
    println!("All five algorithms are concrete LEAVES of the four algorithm-family");
    println!("base classes (SingleStateOptimizer, PopulationOptimizer, RLAgentStation,");
    println!("PolicyGradientAgent) and share the same runIterativeDES runner.");
}
