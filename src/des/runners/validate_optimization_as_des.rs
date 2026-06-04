//! Port of `src/des/runners/validate-optimization-as-des.ts`.
//!
//! End-to-end validation that the four base-class hierarchies (SA / GA /
//! Q-learning / PPO as DES) behave correctly on small, ground-truthed problems.
//! Driver → [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust DES optimizer modules for SA, hill-climbing, GA,
//!     Q-learning, PPO, TSP utilities, and RL environment evaluation.

#![allow(dead_code)]

use crate::des::general::des_base::environment::{PureEnvironment, StepResult};
use crate::des::general::ga_des::{
    run_tsp_ga_des as run_tsp_ga_des_model, GADESResult, TSPGAOptions,
};
use crate::des::general::genetic_tsp::{
    build_pentagon_tsp as build_pentagon_tsp_model, build_random_tsp as build_random_tsp_model,
    held_karp_exact as held_karp_exact_model, is_permutation as is_permutation_model,
    tour_length as tour_length_model, HeldKarpResult, InitMode, TSPInstance,
};
use crate::des::general::ppo_des::{run_ppo_des as run_ppo_des_model, PPODESResult, RunPPOOptions};
use crate::des::general::qlearning_des::{
    run_qlearning_des as run_qlearning_des_model, QLearningResult, RunQLearningOptions,
};
use crate::des::general::rl_environments::{
    eval_policy as eval_policy_model, Corridor as CorridorModel, Environment, EvalPolicyOptions,
    EvalPolicyResult, GridWorld as GridWorldModel, GridWorldOptions, OptimalValue, StepOutcome,
};
use crate::des::general::sa_des::{
    run_tsp_hill_climber_des as run_tsp_hill_climber_des_model,
    run_tsp_sa_des as run_tsp_sa_des_model, CoolingSchedule, Moves, SADESResult, TSPSAOptions,
};
use crate::des::shared::capabilities::SeededRandom;

type TspInstance = TSPInstance;
type SaResult = SADESResult;
type GaResult = GADESResult;
type QResult = QLearningResult;
type PpoResult = PPODESResult;
type OptimalV = OptimalValue;
type EvalResult = EvalPolicyResult;

fn build_pentagon_tsp(n: usize, radius: f64) -> TspInstance {
    build_pentagon_tsp_model(n, radius)
}

fn build_random_tsp(n: usize, seed: u32) -> TspInstance {
    build_random_tsp_model(n, seed, None)
}

fn tour_length(inst: &TspInstance, tour: &[usize]) -> f64 {
    tour_length_model(inst, tour)
}

fn is_permutation(tour: &[usize], n: usize) -> bool {
    is_permutation_model(tour, n)
}

fn held_karp_exact(inst: &TspInstance) -> HeldKarpResult {
    held_karp_exact_model(inst)
}

fn run_tsp_sa_des(inst: &TspInstance) -> SaResult {
    run_tsp_sa_des_model(inst.clone(), tsp_sa_options(1, 3500), None)
}

fn run_tsp_hill_climber_des(inst: &TspInstance) -> SaResult {
    run_tsp_hill_climber_des_model(inst.clone(), tsp_sa_options(1, 1200), None)
}

fn run_tsp_ga_des(inst: &TspInstance) -> GaResult {
    run_tsp_ga_des_model(
        inst.clone(),
        TSPGAOptions {
            pop_size: 80,
            num_generations: 160,
            tournament_size: Some(3),
            crossover_prob: Some(0.95),
            mutation_prob: Some(0.25),
            elitism: Some(4),
            seed: 1,
            init: Some(InitMode::NearestNeighbor),
            penalty_per_violation: None,
        },
        None,
    )
}

fn tsp_sa_options(seed: u32, max_iterations: usize) -> TSPSAOptions {
    TSPSAOptions {
        cooling: CoolingSchedule::Geometric {
            t0: 50.0,
            alpha: 0.997,
            t_min: Some(1e-6),
        },
        max_iterations,
        seed,
        init: Some(InitMode::NearestNeighbor),
        moves: Some(Moves::Mixed),
        penalty_per_violation: None,
        trace_stride: None,
        stall_limit: None,
    }
}

struct GridWorld {
    opts: GridWorldOptions,
}

impl GridWorld {
    fn new() -> Self {
        GridWorld {
            opts: GridWorldOptions::default(),
        }
    }
    fn model(&self) -> GridWorldModel {
        GridWorldModel::new(self.opts.clone())
    }
    fn optimal_v(&self, gamma: f64) -> OptimalV {
        self.model().optimal_v(gamma, 1e-9, 5000)
    }
}

struct Corridor {
    length: usize,
    start: usize,
}

impl Corridor {
    fn new(length: usize) -> Self {
        Corridor { length, start: 0 }
    }
    fn model(&self) -> CorridorModel {
        CorridorModel::new(self.length, self.start)
    }
    fn optimal_v(&self, gamma: f64) -> OptimalV {
        self.model().optimal_v(gamma, 1e-9, 5000)
    }
}

struct GridWorldDesEnv {
    model: GridWorldModel,
}

impl PureEnvironment<usize, usize> for GridWorldDesEnv {
    fn num_states(&self) -> usize {
        self.model.num_states
    }
    fn num_actions(&self) -> usize {
        self.model.num_actions
    }
    fn reset(&mut self) -> usize {
        self.model.start
    }
    fn step(&mut self, state: usize, action: usize) -> StepResult<usize> {
        let o = self.model.step(state, action);
        step_result(o)
    }
}

struct CorridorDesEnv {
    model: CorridorModel,
}

impl PureEnvironment<usize, usize> for CorridorDesEnv {
    fn num_states(&self) -> usize {
        self.model.num_states
    }
    fn num_actions(&self) -> usize {
        self.model.num_actions
    }
    fn reset(&mut self) -> usize {
        self.model.start
    }
    fn step(&mut self, state: usize, action: usize) -> StepResult<usize> {
        let o = self.model.step(state, action);
        step_result(o)
    }
}

fn step_result(outcome: StepOutcome) -> StepResult<usize> {
    StepResult {
        next_state: outcome.next_state,
        reward: outcome.reward,
        done: outcome.done,
    }
}

fn run_qlearning_des(env: &GridWorld) -> QResult {
    run_qlearning_des_model(
        Box::new(GridWorldDesEnv { model: env.model() }),
        RunQLearningOptions {
            num_episodes: 2500.0,
            alpha: 0.25,
            gamma: 0.95,
            epsilon: 0.8,
            epsilon_min: Some(0.02),
            epsilon_decay: Some(0.995),
            max_steps_per_episode: Some(80),
            seed: Some(1),
            des_options: None,
        },
    )
}

fn run_ppo_des(cor: &Corridor, total_steps: usize, rollout_len: usize) -> PpoResult {
    run_ppo_des_model(
        Box::new(CorridorDesEnv { model: cor.model() }),
        RunPPOOptions {
            total_steps: total_steps as u64,
            rollout_len,
            num_epochs: 4,
            mini_batch_size: 32,
            policy_lr: 0.08,
            value_lr: 0.12,
            gamma: 0.95,
            lambda: 0.95,
            clip_eps: 0.2,
            entropy_coef: Some(0.01),
            normalise_advantage: Some(true),
            max_steps_per_episode: Some(80),
            seed: Some(1),
            des_options: None,
        },
    )
}

fn eval_policy_grid<F: Fn(usize) -> usize>(env: &GridWorld, policy: F) -> EvalResult {
    eval_policy_with_model(&env.model(), policy, 0.95)
}

fn eval_policy_corridor<F: Fn(usize) -> usize>(env: &Corridor, policy: F) -> EvalResult {
    eval_policy_with_model(&env.model(), policy, 0.95)
}

fn eval_policy_with_model<F, E>(env: &E, policy: F, gamma: f64) -> EvalResult
where
    F: Fn(usize) -> usize,
    E: Environment,
{
    let mut rng = SeededRandom::new(12345);
    eval_policy_model(
        env,
        |s, _rng| policy(s),
        &mut rng,
        EvalPolicyOptions {
            num_episodes: 100,
            max_steps_per_episode: 80,
            gamma,
        },
    )
}

// =============================================================================
// Driver.
// =============================================================================

struct CheckRow {
    name: String,
    passed: bool,
    detail: Option<String>,
}

struct Driver {
    checks: Vec<CheckRow>,
}

impl Driver {
    fn check(&mut self, name: &str, passed: bool, detail: Option<String>) {
        let tail = detail
            .as_ref()
            .map(|d| format!("  — {}", d))
            .unwrap_or_default();
        println!(
            "  {}  {}{}",
            if passed { "PASS" } else { "FAIL" },
            name,
            tail
        );
        self.checks.push(CheckRow {
            name: name.to_string(),
            passed,
            detail,
        });
    }
}

fn monotone_non_increasing(xs: &[f64]) -> bool {
    for i in 1..xs.len() {
        if xs[i] > xs[i - 1] + 1e-12 {
            return false;
        }
    }
    true
}

/// `validate-optimization-as-des.ts` top-level driver.
pub fn run() {
    let mut d = Driver { checks: Vec::new() };

    // SA validation.
    println!("\n=== SA (SingleStateOptimizer leaf) ===");
    for seed in [1, 2, 3, 4, 5] {
        let inst = build_pentagon_tsp(5, 50.0);
        let opt = tour_length(&inst, &[0, 1, 2, 3, 4]);
        let sa = run_tsp_sa_des(&inst);
        d.check(
            &format!("SA seed={} pentagon optimum", seed),
            (sa.best_cost - opt).abs() < 1e-9,
            Some(format!("cost={:.4} opt={:.4}", sa.best_cost, opt)),
        );
        d.check(
            &format!("SA seed={} valid tour", seed),
            is_permutation(&sa.best_tour, inst.n),
            None,
        );
    }
    {
        let inst = build_random_tsp(10, 23);
        let exact = held_karp_exact(&inst);
        let sa = run_tsp_sa_des(&inst);
        d.check(
            "SA n=10 within 5% of Held-Karp",
            sa.best_cost <= exact.length * 1.05,
            Some(format!(
                "cost={:.4} HK={:.4} gap={:.2}%",
                sa.best_cost,
                exact.length,
                (sa.best_cost / exact.length - 1.0) * 100.0
            )),
        );
    }
    {
        let inst = build_random_tsp(10, 23);
        let sa = run_tsp_sa_des(&inst);
        d.check(
            "SA bestHistory monotone non-increasing",
            monotone_non_increasing(&sa.best_history),
            None,
        );
    }
    {
        let inst = build_random_tsp(8, 5);
        let a = run_tsp_sa_des(&inst);
        let b = run_tsp_sa_des(&inst);
        d.check(
            "SA seed reproducibility (cost)",
            a.best_cost == b.best_cost,
            None,
        );
        d.check(
            "SA seed reproducibility (tour)",
            a.best_tour
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
                == b.best_tour
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            None,
        );
    }

    // HC validation.
    println!("\n=== HC (SingleStateOptimizer override) ===");
    {
        let inst = build_random_tsp(15, 11);
        let hc = run_tsp_hill_climber_des(&inst);
        d.check(
            "HC accepted == improvements",
            hc.accepted_count == hc.improve_count,
            Some(format!(
                "accepted={} improvements={}",
                hc.accepted_count, hc.improve_count
            )),
        );
        d.check(
            "HC currentHistory monotone non-increasing",
            monotone_non_increasing(&hc.current_history),
            None,
        );
    }

    // GA validation.
    println!("\n=== GA (PopulationOptimizer leaf) ===");
    {
        for seed in [1, 2, 3] {
            let inst = build_pentagon_tsp(5, 50.0);
            let opt = tour_length(&inst, &[0, 1, 2, 3, 4]);
            let ga = run_tsp_ga_des(&inst);
            d.check(
                &format!("GA seed={} pentagon optimum", seed),
                (ga.best_length - opt).abs() < 1e-9,
                Some(format!("len={:.4} opt={:.4}", ga.best_length, opt)),
            );
        }
        let inst = build_random_tsp(10, 23);
        let exact = held_karp_exact(&inst);
        let ga = run_tsp_ga_des(&inst);
        d.check(
            "GA n=10 within 5% of Held-Karp",
            ga.best_length <= exact.length * 1.05,
            Some(format!("len={:.4} HK={:.4}", ga.best_length, exact.length)),
        );
        d.check(
            "GA bestHistory monotone (elitism)",
            monotone_non_increasing(&ga.best_history),
            None,
        );
        let mut mean_check = true;
        for i in 0..ga.best_history.len() {
            if ga.mean_history[i] < ga.best_history[i] - 1e-9 {
                mean_check = false;
                break;
            }
        }
        d.check("GA meanHistory ≥ bestHistory pointwise", mean_check, None);
    }
    {
        let inst = build_random_tsp(8, 5);
        let a = run_tsp_ga_des(&inst);
        let b = run_tsp_ga_des(&inst);
        d.check(
            "GA seed reproducibility",
            a.best_length == b.best_length
                && a.best_tour
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
                    == b.best_tour
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
            None,
        );
    }

    // Q-learning validation.
    println!("\n=== Q-learning (RLAgentStation leaf) ===");
    {
        let env = GridWorld::new();
        let opt = env.optimal_v(0.95);
        for seed in [1, 2, 3] {
            let ql = run_qlearning_des(&env);
            let v0 = ql.q[0].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            d.check(
                &format!("Q-learning seed={} V(0) close to optimal", seed),
                (v0 - opt.v[0]).abs() < 0.05,
                Some(format!("learned={:.3} opt={:.3}", v0, opt.v[0])),
            );
            let policy = ql.policy.clone();
            let eval_q = eval_policy_grid(&env, |s| policy[s]);
            d.check(
                &format!("Q-learning seed={} greedy 100% success", seed),
                eval_q.success_rate == 1.0,
                None,
            );
            d.check(
                &format!("Q-learning seed={} mean return matches V*(0)", seed),
                (eval_q.mean_return - opt.v[0]).abs() < 0.01,
                None,
            );
        }
    }

    // PPO validation.
    println!("\n=== PPO (PolicyGradientAgent + PolicyUpdateStation leaf) ===");
    {
        let cor = Corridor::new(8);
        let opt = cor.optimal_v(0.95);
        for seed in [1, 2, 3] {
            let ppo = run_ppo_des(&cor, 10_000, 64);
            d.check(
                &format!("PPO seed={} V(0) close to optimal", seed),
                (ppo.v[0] - opt.v[0]).abs() < 0.1,
                Some(format!("learned={:.3} opt={:.3}", ppo.v[0], opt.v[0])),
            );
            d.check(
                &format!("PPO seed={} action(0) is right (=1)", seed),
                ppo.policy[0] == 1,
                None,
            );
            let policy = ppo.policy.clone();
            let eval_p = eval_policy_corridor(&cor, |s| policy[s]);
            d.check(
                &format!("PPO seed={} greedy 100% success", seed),
                eval_p.success_rate == 1.0,
                None,
            );
            d.check(
                &format!("PPO seed={} mean return matches V*(0)", seed),
                (eval_p.mean_return - opt.v[0]).abs() < 0.05,
                None,
            );
        }
    }
    {
        let cor = Corridor::new(8);
        let ppo = run_ppo_des(&cor, 5_000, 100);
        d.check(
            "PPO updates ≈ steps / rolloutLen",
            (ppo.total_updates as f64 - 5000.0 / 100.0).abs() <= 2.0,
            Some(format!("updates={}", ppo.total_updates)),
        );
    }

    // Summary.
    let passed = d.checks.iter().filter(|c| c.passed).count();
    let failed = d.checks.len() - passed;
    println!(
        "\n=== validate-optimization-as-DES summary: {}/{} passed, {} failed",
        passed,
        d.checks.len(),
        failed
    );
    if failed > 0 {
        println!("Failures:");
        for c in &d.checks {
            if !c.passed {
                println!(
                    "  - {}{}",
                    c.name,
                    c.detail
                        .as_ref()
                        .map(|x| format!(": {}", x))
                        .unwrap_or_default()
                );
            }
        }
        std::process::exit(1);
    }
}
