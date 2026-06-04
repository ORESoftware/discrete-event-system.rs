//! Port of `src/des/runners/validate-optimization-as-des.ts`.
//!
//! End-to-end validation that the four base-class hierarchies (SA / GA /
//! Q-learning / PPO as DES) behave correctly on small, ground-truthed problems.
//! Driver → [`run`].
//!
//! The first Rust runner kept local exact-optimum and zero-value stand-ins. The
//! DES optimizer/agent modules are now ported, so these checks exercise the real
//! SA-DES, GA-DES, Q-learning-DES, PPO-DES, and environment code.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use crate::des::general::des_base::environment::{
    PureEnvironment as DesPureEnvironment, StepResult as DesStepResult,
};
use crate::des::general::ga_des::{
    run_tsp_ga_des as real_run_tsp_ga_des, TSPGAOptions as RealTspGaOptions,
};
use crate::des::general::genetic_tsp::{
    build_pentagon_tsp as real_build_pentagon_tsp, build_random_tsp as real_build_random_tsp,
    held_karp_exact as real_held_karp_exact, is_permutation as real_is_permutation,
    tour_length as real_tour_length, InitMode, TSPInstance as TspInstance,
};
use crate::des::general::ppo_des::{
    run_ppo_des as real_run_ppo_des, RunPPOOptions as RealRunPpoOptions,
};
use crate::des::general::qlearning_des::{
    run_qlearning_des as real_run_qlearning_des, RunQLearningOptions as RealRunQLearningOptions,
};
use crate::des::general::rl_environments::{
    eval_policy as real_eval_policy, Corridor as RealCorridor, Environment as RealEnvironment,
    EvalPolicyOptions, GridWorld as RealGridWorld, GridWorldOptions,
};
use crate::des::general::sa_des::{
    run_tsp_hill_climber_des as real_run_tsp_hill_climber_des,
    run_tsp_sa_des as real_run_tsp_sa_des, CoolingSchedule as RealCoolingSchedule, Moves,
    TSPSAOptions as RealTspSaOptions,
};
use crate::des::shared::capabilities::SeededRandom;

// =============================================================================
// Thin validation adapters over TSP / optimizer kernels.
// =============================================================================

fn build_pentagon_tsp(n: usize, radius: f64) -> TspInstance {
    real_build_pentagon_tsp(n, radius)
}

fn build_random_tsp(n: usize, seed: u32) -> TspInstance {
    real_build_random_tsp(n, seed, None)
}

fn tour_length(inst: &TspInstance, tour: &[usize]) -> f64 {
    real_tour_length(inst, tour)
}

fn is_permutation(tour: &[usize], n: usize) -> bool {
    real_is_permutation(tour, n)
}

#[derive(Clone, Debug, Default)]
struct HeldKarpResult {
    length: f64,
    tour: Vec<usize>,
}

fn held_karp_exact(inst: &TspInstance) -> HeldKarpResult {
    let result = real_held_karp_exact(inst);
    HeldKarpResult {
        length: result.length,
        tour: result.tour,
    }
}

// =============================================================================
// Metaheuristic solvers.
// =============================================================================

#[derive(Clone, Debug, Default)]
struct SaResult {
    best_cost: f64,
    best_tour: Vec<usize>,
    best_history: Vec<f64>,
    accepted_count: usize,
    improve_count: usize,
    current_history: Vec<f64>,
}

fn run_tsp_sa_des(inst: &TspInstance) -> SaResult {
    let result = real_run_tsp_sa_des(inst.clone(), tsp_sa_options(1), None);
    SaResult {
        best_cost: result.best_cost,
        best_tour: result.best_tour,
        best_history: result.best_history,
        accepted_count: result.accepted_count,
        improve_count: result.improve_count,
        current_history: result.current_history,
    }
}

fn run_tsp_hill_climber_des(inst: &TspInstance) -> SaResult {
    let result = real_run_tsp_hill_climber_des(inst.clone(), tsp_sa_options(7), None);
    SaResult {
        best_cost: result.best_cost,
        best_tour: result.best_tour,
        best_history: result.best_history,
        accepted_count: result.accepted_count,
        improve_count: result.improve_count,
        current_history: result.current_history,
    }
}

fn tsp_sa_options(seed: u32) -> RealTspSaOptions {
    RealTspSaOptions {
        cooling: RealCoolingSchedule::Geometric {
            t0: 50.0,
            alpha: 0.999,
            t_min: None,
        },
        max_iterations: 12_000,
        seed,
        init: Some(InitMode::NearestNeighbor),
        moves: Some(Moves::Mixed),
        penalty_per_violation: None,
        trace_stride: None,
        stall_limit: None,
    }
}

#[derive(Clone, Debug, Default)]
struct GaResult {
    best_length: f64,
    best_tour: Vec<usize>,
    best_history: Vec<f64>,
    mean_history: Vec<f64>,
}

fn run_tsp_ga_des(inst: &TspInstance) -> GaResult {
    let result = real_run_tsp_ga_des(
        inst.clone(),
        RealTspGaOptions {
            pop_size: 80,
            num_generations: 180,
            tournament_size: None,
            crossover_prob: None,
            mutation_prob: None,
            elitism: Some(4),
            seed: 3,
            init: Some(InitMode::NearestNeighbor),
            penalty_per_violation: None,
        },
        None,
    );
    GaResult {
        best_length: result.best_length,
        best_tour: result.best_tour,
        best_history: result.best_history,
        mean_history: result.mean_history,
    }
}

// =============================================================================
// RL environments + agents.
// =============================================================================

#[derive(Clone, Debug)]
struct OptimalV {
    v: Vec<f64>,
}

struct GridWorld {
    inner: RealGridWorld,
}

impl GridWorld {
    fn new() -> Self {
        GridWorld {
            inner: RealGridWorld::new(GridWorldOptions::default()),
        }
    }
    fn optimal_v(&self, gamma: f64) -> OptimalV {
        let opt = self.inner.optimal_v(gamma, 1e-9, 5000);
        OptimalV { v: opt.v }
    }
}

struct GridWorldDesEnv {
    inner: RealGridWorld,
}

impl GridWorldDesEnv {
    fn new() -> Self {
        GridWorldDesEnv {
            inner: RealGridWorld::new(GridWorldOptions::default()),
        }
    }
}

impl DesPureEnvironment<usize, usize> for GridWorldDesEnv {
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }

    fn reset(&mut self) -> usize {
        self.inner.reset()
    }

    fn step(&mut self, state: usize, action: usize) -> DesStepResult<usize> {
        let result = self.inner.step(state, action);
        DesStepResult {
            next_state: result.next_state,
            reward: result.reward,
            done: result.done,
        }
    }
}

#[derive(Clone, Debug)]
struct Corridor {
    length: usize,
}

impl Corridor {
    fn new(length: usize) -> Self {
        Corridor { length }
    }
    fn optimal_v(&self, gamma: f64) -> OptimalV {
        let opt = RealCorridor::new(self.length, 0).optimal_v(gamma, 1e-9, 5000);
        OptimalV { v: opt.v }
    }
}

struct CorridorDesEnv {
    inner: RealCorridor,
}

impl CorridorDesEnv {
    fn new(length: usize) -> Self {
        CorridorDesEnv {
            inner: RealCorridor::new(length, 0),
        }
    }
}

impl DesPureEnvironment<usize, usize> for CorridorDesEnv {
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }

    fn reset(&mut self) -> usize {
        self.inner.reset()
    }

    fn step(&mut self, state: usize, action: usize) -> DesStepResult<usize> {
        let result = self.inner.step(state, action);
        DesStepResult {
            next_state: result.next_state,
            reward: result.reward,
            done: result.done,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct QResult {
    q: Vec<Vec<f64>>,
    policy: Vec<usize>,
}

fn run_qlearning_des(env: &GridWorld) -> QResult {
    let result = real_run_qlearning_des(
        Box::new(GridWorldDesEnv::new()),
        RealRunQLearningOptions {
            num_episodes: 8_000.0,
            alpha: 0.2,
            gamma: 0.95,
            epsilon: 0.4,
            epsilon_min: Some(0.02),
            epsilon_decay: Some(0.995),
            max_steps_per_episode: Some(100),
            seed: Some(11),
            des_options: None,
        },
    );
    QResult {
        q: result.q,
        policy: result.policy,
    }
}

#[derive(Clone, Debug, Default)]
struct PpoResult {
    v: Vec<f64>,
    policy: Vec<usize>,
    total_updates: usize,
}

fn run_ppo_des(cor: &Corridor, total_steps: usize, rollout_len: usize) -> PpoResult {
    let result = real_run_ppo_des(
        Box::new(CorridorDesEnv::new(cor.length)),
        RealRunPpoOptions {
            total_steps: total_steps as u64,
            rollout_len,
            num_epochs: 4,
            mini_batch_size: 32,
            policy_lr: 0.05,
            value_lr: 0.08,
            gamma: 0.95,
            lambda: 0.95,
            clip_eps: 0.2,
            entropy_coef: Some(0.01),
            normalise_advantage: Some(true),
            max_steps_per_episode: Some(100),
            seed: Some(17),
            des_options: None,
        },
    );
    PpoResult {
        v: result.v,
        policy: result.policy,
        total_updates: result.total_updates as usize,
    }
}

#[derive(Clone, Debug, Default)]
struct EvalResult {
    success_rate: f64,
    mean_return: f64,
}

fn eval_policy_grid<F: Fn(usize) -> usize>(_env: &GridWorld, _policy: F) -> EvalResult {
    let mut rng = SeededRandom::new(123);
    let result = real_eval_policy(
        &_env.inner,
        |s, _rng| _policy(s),
        &mut rng,
        EvalPolicyOptions {
            num_episodes: 100,
            max_steps_per_episode: 100,
            gamma: 0.95,
        },
    );
    EvalResult {
        success_rate: result.success_rate,
        mean_return: result.mean_return,
    }
}
fn eval_policy_corridor<F: Fn(usize) -> usize>(_env: &Corridor, _policy: F) -> EvalResult {
    let env = RealCorridor::new(_env.length, 0);
    let mut rng = SeededRandom::new(456);
    let result = real_eval_policy(
        &env,
        |s, _rng| _policy(s),
        &mut rng,
        EvalPolicyOptions {
            num_episodes: 100,
            max_steps_per_episode: 100,
            gamma: 0.95,
        },
    );
    EvalResult {
        success_rate: result.success_rate,
        mean_return: result.mean_return,
    }
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
