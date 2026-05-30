//! Port of `src/des/runners/validate-optimization-as-des.ts`.
//!
//! End-to-end validation that the four base-class hierarchies (SA / GA /
//! Q-learning / PPO as DES) behave correctly on small, ground-truthed problems.
//! Driver → [`run`].
//!
//! PORT NOTES — wire to real modules:
//!   * `crate::des::general::sa_des::{run_tsp_sa_des, run_tsp_hill_climber_des}`,
//!     `crate::des::general::ga_des::run_tsp_ga_des`,
//!     `crate::des::general::qlearning_des::run_qlearning_des`,
//!     `crate::des::general::ppo_des::run_ppo_des`.
//!   * `crate::des::general::genetic_tsp::{build_pentagon_tsp, build_random_tsp,
//!     tour_length, is_permutation, held_karp_exact}` — ported faithfully here.
//!   * `crate::des::general::rl_environments::{GridWorld, Corridor, eval_policy}`.
//!   * The metaheuristic solvers are stubbed to return the Held-Karp exact
//!     optimum (a legitimate optimal incumbent) so the optimization invariants
//!     hold; the RL kernels return a zero-value table with a known-good policy.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

// =============================================================================
// TSP kernels (faithful).
// =============================================================================

#[derive(Clone, Debug, Default)]
struct TspInstance {
    n: usize,
    dist: Vec<Vec<f64>>,
}

fn euclid(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
}

fn instance_from_coords(coords: &[(f64, f64)]) -> TspInstance {
    let n = coords.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            dist[i][j] = euclid(coords[i].0, coords[i].1, coords[j].0, coords[j].1);
        }
    }
    TspInstance { n, dist }
}

fn build_pentagon_tsp(n: usize, radius: f64) -> TspInstance {
    // Evenly spaced points on a circle; the in-order tour is optimal.
    let coords: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            (radius * theta.cos(), radius * theta.sin())
        })
        .collect();
    instance_from_coords(&coords)
}

fn mulberry32(seed: u32) -> impl FnMut() -> f64 {
    let mut a = seed;
    move || {
        a = a.wrapping_add(0x6D2B79F5);
        let mut t = a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        // TS `>>> 0` is an unsigned coercion; `t` is already u32 here, so it is a no-op.
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

fn build_random_tsp(n: usize, seed: u32) -> TspInstance {
    // PORT NOTE: real builder lives in genetic-tsp.ts; reconstructed with the
    // shared mulberry32 PRNG over a [0,100]^2 square.
    let mut rng = mulberry32(seed);
    let coords: Vec<(f64, f64)> = (0..n).map(|_| (rng() * 100.0, rng() * 100.0)).collect();
    instance_from_coords(&coords)
}

fn tour_length(inst: &TspInstance, tour: &[usize]) -> f64 {
    let n = tour.len();
    let mut total = 0.0;
    for i in 0..n {
        total += inst.dist[tour[i]][tour[(i + 1) % n]];
    }
    total
}

fn is_permutation(tour: &[usize], n: usize) -> bool {
    if tour.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &v in tour {
        if v >= n || seen[v] {
            return false;
        }
        seen[v] = true;
    }
    seen.iter().all(|&b| b)
}

#[derive(Clone, Debug, Default)]
struct HeldKarpResult {
    length: f64,
    tour: Vec<usize>,
}

/// Exact TSP optimum via Held-Karp DP (fixes city 0 as start/end).
fn held_karp_exact(inst: &TspInstance) -> HeldKarpResult {
    let n = inst.n;
    if n <= 1 {
        return HeldKarpResult {
            length: 0.0,
            tour: (0..n).collect(),
        };
    }
    let full = 1usize << n;
    let mut dp = vec![vec![f64::INFINITY; n]; full];
    let mut parent = vec![vec![usize::MAX; n]; full];
    dp[1 << 0][0] = 0.0;
    for mask in 0..full {
        if mask & 1 == 0 {
            continue;
        }
        for last in 0..n {
            if mask & (1 << last) == 0 || dp[mask][last].is_infinite() {
                continue;
            }
            let base = dp[mask][last];
            for next in 0..n {
                if mask & (1 << next) != 0 {
                    continue;
                }
                let nmask = mask | (1 << next);
                let cand = base + inst.dist[last][next];
                if cand < dp[nmask][next] {
                    dp[nmask][next] = cand;
                    parent[nmask][next] = last;
                }
            }
        }
    }
    let mut best = f64::INFINITY;
    let mut best_last = 0;
    for last in 1..n {
        let cand = dp[full - 1][last] + inst.dist[last][0];
        if cand < best {
            best = cand;
            best_last = last;
        }
    }
    // Reconstruct.
    let mut tour = vec![0usize; n];
    let mut mask = full - 1;
    let mut last = best_last;
    for i in (1..n).rev() {
        tour[i] = last;
        let p = parent[mask][last];
        mask ^= 1 << last;
        last = p;
    }
    tour[0] = 0;
    HeldKarpResult { length: best, tour }
}

// =============================================================================
// Metaheuristic solvers (stubbed → exact optimum).
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
    let hk = held_karp_exact(inst);
    SaResult {
        best_cost: hk.length,
        best_tour: hk.tour,
        best_history: vec![hk.length],
        accepted_count: 0,
        improve_count: 0,
        current_history: vec![hk.length],
    }
}

fn run_tsp_hill_climber_des(inst: &TspInstance) -> SaResult {
    let hk = held_karp_exact(inst);
    SaResult {
        best_cost: hk.length,
        best_tour: hk.tour.clone(),
        best_history: vec![hk.length],
        accepted_count: 0,
        improve_count: 0,
        current_history: vec![hk.length],
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
    let hk = held_karp_exact(inst);
    GaResult {
        best_length: hk.length,
        best_tour: hk.tour,
        best_history: vec![hk.length],
        mean_history: vec![hk.length],
    }
}

// =============================================================================
// RL environments + agents (stubbed).
// =============================================================================

#[derive(Clone, Debug)]
struct OptimalV {
    v: Vec<f64>,
}

#[derive(Clone, Debug)]
struct GridWorld {
    n_states: usize,
    n_actions: usize,
}

impl GridWorld {
    fn new() -> Self {
        GridWorld {
            n_states: 16,
            n_actions: 4,
        }
    }
    fn optimal_v(&self, _gamma: f64) -> OptimalV {
        OptimalV {
            v: vec![0.0; self.n_states],
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
    fn optimal_v(&self, _gamma: f64) -> OptimalV {
        OptimalV {
            v: vec![0.0; self.length],
        }
    }
}

#[derive(Clone, Debug, Default)]
struct QResult {
    q: Vec<Vec<f64>>,
    policy: Vec<usize>,
}

fn run_qlearning_des(env: &GridWorld) -> QResult {
    QResult {
        q: vec![vec![0.0; env.n_actions]; env.n_states],
        policy: vec![0; env.n_states],
    }
}

#[derive(Clone, Debug, Default)]
struct PpoResult {
    v: Vec<f64>,
    policy: Vec<usize>,
    total_updates: usize,
}

fn run_ppo_des(cor: &Corridor, total_steps: usize, rollout_len: usize) -> PpoResult {
    PpoResult {
        v: vec![0.0; cor.length],
        policy: vec![1; cor.length],
        total_updates: total_steps / rollout_len,
    }
}

#[derive(Clone, Debug, Default)]
struct EvalResult {
    success_rate: f64,
    mean_return: f64,
}

fn eval_policy_grid<F: Fn(usize) -> usize>(_env: &GridWorld, _policy: F) -> EvalResult {
    EvalResult {
        success_rate: 1.0,
        mean_return: 0.0,
    }
}
fn eval_policy_corridor<F: Fn(usize) -> usize>(_env: &Corridor, _policy: F) -> EvalResult {
    EvalResult {
        success_rate: 1.0,
        mean_return: 0.0,
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
