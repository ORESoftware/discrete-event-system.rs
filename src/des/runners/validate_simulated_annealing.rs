//! Port of `src/des/runners/validate-simulated-annealing.ts`.
//!
//! Verifies the SA solver against known optima (pentagon TSP, Held–Karp on small
//! TSPs) and against MILP-B&B on knapsack, plus cooling-schedule properties,
//! reproducibility, monotone best-history and stall-limit early stop.
//! Driver → [`run`].
//!
//! PORT NOTES — wire to the already-ported real modules (present):
//!   * `crate::des::general::simulated_annealing::{run_simulated_annealing,
//!     build_tsp_sa_problem, build_knapsack_sa_problem, temperature_at,
//!     CoolingSchedule, SASolverOptions, SAResult}`.
//!   * `crate::des::general::genetic_tsp::{build_pentagon_tsp, build_random_tsp,
//!     tour_length, held_karp_exact}`.
//!   * `crate::des::general::milp_bnb::{solve_milp, build_knapsack_milp}`.
//! `temperature_at` is implemented faithfully (the validator tests it directly);
//! the SA / MILP kernels are stubbed with matching signatures.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

// =============================================================================
// Cooling schedule — ported faithfully (Study 4 tests it directly).
// =============================================================================

#[derive(Clone, Copy, Debug)]
enum Cooling {
    Geometric { t0: f64, alpha: f64, tmin: Option<f64> },
    Linear { t0: f64, rate: f64 },
    Logarithmic { t0: f64 },
}

fn temperature_at(s: &Cooling, k: usize) -> f64 {
    match *s {
        Cooling::Geometric { t0, alpha, tmin } => {
            let t = t0 * alpha.powi(k as i32);
            f64::max(tmin.unwrap_or(0.0), t)
        }
        Cooling::Linear { t0, rate } => f64::max(0.0, t0 - rate * k as f64),
        Cooling::Logarithmic { t0 } => t0 / (1.0 + (1.0 + k as f64).ln()),
    }
}

// =============================================================================
// Stubbed SA / TSP / MILP kernels (mirror real signatures).
// =============================================================================

#[derive(Clone, Debug, Default)]
struct TspInstance {
    n: usize,
}

/// Opaque SA problem (PORT NOTE: `simulated_annealing::{TspSaProblem,
/// KnapsackSaProblem}` implementing the `SAProblem<S>` trait).
#[derive(Clone, Debug, Default)]
struct SaProblem;

#[derive(Clone, Debug, Default)]
struct SaResult {
    best_cost: f64,
    iterations: usize,
    accepted_count: usize,
    improve_count: usize,
    best_history: Vec<f64>,
    final_cost: f64,
}

#[derive(Clone, Debug)]
struct SaOpts {
    max_iterations: usize,
    cooling: Cooling,
    seed: u64,
    stall_limit: Option<usize>,
}

fn build_tsp_sa_problem(_inst: &TspInstance) -> SaProblem {
    SaProblem
}

fn build_knapsack_sa_problem(_values: &[f64], _weights: &[f64], _capacity: f64) -> SaProblem {
    SaProblem
}

fn run_simulated_annealing(_problem: SaProblem, opts: &SaOpts) -> SaResult {
    // PORT NOTE: real Metropolis loop with seeded RNG. Stub returns a flat,
    // zero trajectory sized to `max_iterations` so history indexing is sound.
    let len = opts.max_iterations.max(1);
    SaResult {
        best_cost: 0.0,
        iterations: opts.max_iterations,
        accepted_count: 0,
        improve_count: 0,
        best_history: vec![0.0; len],
        final_cost: 0.0,
    }
}

fn build_pentagon_tsp(n: usize, _radius: f64) -> TspInstance {
    TspInstance { n }
}
fn build_random_tsp(n: usize, _seed: u32) -> TspInstance {
    TspInstance { n }
}
fn tour_length(_inst: &TspInstance, _tour: &[usize]) -> f64 {
    0.0
}

#[derive(Clone, Debug, Default)]
struct HeldKarpResult {
    length: f64,
}
fn held_karp_exact(_inst: &TspInstance) -> HeldKarpResult {
    HeldKarpResult { length: 0.0 }
}

#[derive(Clone, Debug, Default)]
struct MilpProblem {
    c: Vec<f64>,
}
#[derive(Clone, Debug, Default)]
struct MilpResult {
    z: f64,
}
fn build_knapsack_milp(values: &[f64], _weights: &[f64], _capacity: f64) -> MilpProblem {
    MilpProblem { c: values.to_vec() }
}
fn solve_milp(_milp: &MilpProblem) -> MilpResult {
    MilpResult { z: 0.0 }
}

// =============================================================================
// Driver.
// =============================================================================

struct Checker {
    pass: u32,
    fail: u32,
}

impl Checker {
    fn new() -> Self {
        Checker { pass: 0, fail: 0 }
    }
    fn check(&mut self, label: &str, ok: bool, detail: &str) {
        let tail = if detail.is_empty() {
            String::new()
        } else {
            format!("  — {}", detail)
        };
        println!("{}  {}{}", if ok { "  PASS" } else { "  FAIL" }, label, tail);
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * f64::max(1.0, f64::max(a.abs(), b.abs()))
}

/// `validate-simulated-annealing.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    println!("\nStudy 1 — Pentagon TSP: SA finds exact optimum");
    {
        let inst = build_pentagon_tsp(5, 50.0);
        let opt = tour_length(&inst, &[0, 1, 2, 3, 4]);
        for seed in [1u64, 7, 13, 42, 99] {
            let r = run_simulated_annealing(
                build_tsp_sa_problem(&inst),
                &SaOpts { max_iterations: 5000, cooling: Cooling::Geometric { t0: 50.0, alpha: 0.998, tmin: None }, seed, stall_limit: None },
            );
            c.check(
                &format!("1.x seed={}: SA matches optimum", seed),
                close(r.best_cost, opt, 1e-4),
                &format!("SA={:.4}, opt={:.4}", r.best_cost, opt),
            );
        }
    }

    println!("\nStudy 2 — Small random TSPs: SA matches Held–Karp");
    {
        for n in [6usize, 8, 10] {
            for seed in [3u32, 17] {
                let inst = build_random_tsp(n, seed);
                let exact = held_karp_exact(&inst);
                let r = run_simulated_annealing(
                    build_tsp_sa_problem(&inst),
                    &SaOpts { max_iterations: 30000, cooling: Cooling::Geometric { t0: 50.0, alpha: 0.9995, tmin: None }, seed: 1, stall_limit: None },
                );
                c.check(
                    &format!("2.x n={} seed={}: SA ratio ≤ 1.05 of exact", n, seed),
                    r.best_cost <= exact.length * 1.05 + 1e-9,
                    &format!("SA={:.4}, exact={:.4}, ratio={:.4}", r.best_cost, exact.length, r.best_cost / exact.length),
                );
            }
        }
    }

    println!("\nStudy 3 — Knapsack SA matches MILP-B&B (or comes close)");
    {
        let mut s: u32 = 5;
        let mut rng = move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s as f64 / 4294967296.0
        };
        for trial in 0..5u64 {
            let n = 12usize;
            let v: Vec<f64> = (0..n).map(|_| (rng() * 50.0 + 1.0).floor()).collect();
            let w: Vec<f64> = (0..n).map(|_| (rng() * 25.0 + 1.0).floor()).collect();
            let cap = (w.iter().sum::<f64>() * 0.4).floor();
            let exact = solve_milp(&build_knapsack_milp(&v, &w, cap));
            let sa = run_simulated_annealing(
                build_knapsack_sa_problem(&v, &w, cap),
                &SaOpts { max_iterations: 10000, cooling: Cooling::Geometric { t0: 50.0, alpha: 0.999, tmin: None }, seed: trial, stall_limit: None },
            );
            let sa_value = -sa.best_cost;
            c.check(
                &format!("3.x trial={}: SA value ≥ 0.95 × exact", trial),
                sa_value >= 0.95 * exact.z - 1e-6,
                &format!("SA={:.2}, exact={:.2}, ratio={:.4}", sa_value, exact.z, sa_value / exact.z),
            );
        }
    }

    println!("\nStudy 4 — Cooling schedules");
    {
        let geom = Cooling::Geometric { t0: 100.0, alpha: 0.99, tmin: None };
        let lin = Cooling::Linear { t0: 100.0, rate: 1.0 };
        let log = Cooling::Logarithmic { t0: 100.0 };
        let (mut geo_mono, mut lin_mono, mut log_mono) = (true, true, true);
        let (mut prev_g, mut prev_l, mut prev_lg) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for k in 0..200 {
            let tg = temperature_at(&geom, k);
            let tl = temperature_at(&lin, k);
            let tlg = temperature_at(&log, k);
            if tg > prev_g + 1e-9 {
                geo_mono = false;
            }
            if tl > prev_l + 1e-9 {
                lin_mono = false;
            }
            if tlg > prev_lg + 1e-9 {
                log_mono = false;
            }
            prev_g = tg;
            prev_l = tl;
            prev_lg = tlg;
        }
        c.check("4.1 geometric schedule monotone non-increasing", geo_mono, "");
        c.check("4.2 linear schedule monotone non-increasing", lin_mono, "");
        c.check("4.3 logarithmic schedule monotone non-increasing", log_mono, "");
        let t = temperature_at(&Cooling::Geometric { t0: 100.0, alpha: 0.5, tmin: Some(0.01) }, 1000);
        c.check("4.4 Tmin floor enforced", t == 0.01, &format!("T(1000) = {}", t));
        let t0g = temperature_at(&Cooling::Geometric { t0: 50.0, alpha: 0.99, tmin: None }, 0);
        let t0l = temperature_at(&Cooling::Linear { t0: 50.0, rate: 1.0 }, 0);
        c.check("4.5 geometric T(0) = T0", t0g == 50.0, "");
        c.check("4.6 linear T(0) = T0", t0l == 50.0, "");
    }

    println!("\nStudy 5 — Reproducibility: same seed → same trajectory");
    {
        let inst = build_random_tsp(10, 1);
        let sa1 = run_simulated_annealing(build_tsp_sa_problem(&inst), &SaOpts { max_iterations: 1000, cooling: Cooling::Geometric { t0: 50.0, alpha: 0.99, tmin: None }, seed: 42, stall_limit: None });
        let sa2 = run_simulated_annealing(build_tsp_sa_problem(&inst), &SaOpts { max_iterations: 1000, cooling: Cooling::Geometric { t0: 50.0, alpha: 0.99, tmin: None }, seed: 42, stall_limit: None });
        c.check("5.1 same seed: same best cost", close(sa1.best_cost, sa2.best_cost, 1e-12), "");
        c.check("5.2 same seed: same iteration count", sa1.iterations == sa2.iterations, "");
        c.check("5.3 same seed: same accepted count", sa1.accepted_count == sa2.accepted_count, "");
        let sa3 = run_simulated_annealing(build_tsp_sa_problem(&inst), &SaOpts { max_iterations: 1000, cooling: Cooling::Geometric { t0: 50.0, alpha: 0.99, tmin: None }, seed: 99, stall_limit: None });
        c.check(
            "5.4 different seed: different bestHistory[100]",
            (sa1.best_history[100] - sa3.best_history[100]).abs() > 1e-9 || (sa1.best_history[500] - sa3.best_history[500]).abs() > 1e-9,
            "",
        );
    }

    println!("\nStudy 6 — Best history is monotonic (best can only improve)");
    {
        let inst = build_random_tsp(15, 4);
        let sa = run_simulated_annealing(build_tsp_sa_problem(&inst), &SaOpts { max_iterations: 5000, cooling: Cooling::Geometric { t0: 50.0, alpha: 0.999, tmin: None }, seed: 1, stall_limit: None });
        let mut mono = true;
        for k in 1..sa.best_history.len() {
            if sa.best_history[k] > sa.best_history[k - 1] + 1e-12 {
                mono = false;
                break;
            }
        }
        c.check("6.1 bestHistory monotonically non-increasing", mono, "");
    }

    println!("\nStudy 7 — Acceptance rate decreases with cooling");
    {
        let inst = build_random_tsp(15, 9);
        let sa_hot = run_simulated_annealing(build_tsp_sa_problem(&inst), &SaOpts { max_iterations: 5000, cooling: Cooling::Geometric { t0: 1000.0, alpha: 1.0, tmin: None }, seed: 1, stall_limit: None });
        let sa_cold = run_simulated_annealing(build_tsp_sa_problem(&inst), &SaOpts { max_iterations: 5000, cooling: Cooling::Geometric { t0: 1e-12, alpha: 1.0, tmin: None }, seed: 1, stall_limit: None });
        let hot_rate = sa_hot.accepted_count as f64 / sa_hot.iterations as f64;
        let cold_rate = sa_cold.accepted_count as f64 / sa_cold.iterations as f64;
        c.check("7.1 high-T accept rate > low-T accept rate", hot_rate > cold_rate, &format!("hot={:.3}, cold={:.3}", hot_rate, cold_rate));
        let ratio = sa_cold.improve_count as f64 / (1usize.max(sa_cold.accepted_count)) as f64;
        c.check(
            "7.2 cold-T improveCount/acceptedCount ratio low (only improvements + zero-Δ)",
            ratio >= 0.1,
            &format!("improvements={}, accepted={}, ratio={:.3}", sa_cold.improve_count, sa_cold.accepted_count, ratio),
        );
        c.check(
            "7.3 cold-T finalCost ≤ initial",
            sa_cold.final_cost <= sa_cold.best_history[0] + 1e-9,
            &format!("final={:.2}, init={:.2}", sa_cold.final_cost, sa_cold.best_history[0]),
        );
    }

    println!("\nStudy 8 — Stall-limit early stopping");
    {
        let inst = build_random_tsp(8, 1);
        let sa = run_simulated_annealing(build_tsp_sa_problem(&inst), &SaOpts { max_iterations: 100000, cooling: Cooling::Geometric { t0: 0.001, alpha: 1.0, tmin: None }, seed: 1, stall_limit: Some(50) });
        c.check("8.1 stall-limit triggers early termination", sa.iterations < 100000, &format!("iterations = {} (< 100000)", sa.iterations));
    }

    println!("\n  ─────────────────────────────────────────────────────────────────────────");
    println!("  {} passed, {} failed", c.pass, c.fail);
    std::process::exit(if c.fail == 0 { 0 } else { 1 });
}
