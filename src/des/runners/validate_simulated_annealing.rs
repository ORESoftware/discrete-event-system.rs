//! Port of `src/des/runners/validate-simulated-annealing.ts`.
//!
//! Verifies the SA solver against known optima (pentagon TSP, Held–Karp on small
//! TSPs) and against MILP-B&B on knapsack, plus cooling-schedule properties,
//! reproducibility, monotone best-history and stall-limit early stop.
//! Driver → [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust simulated-annealing, TSP, and MILP-B&B modules.

#![allow(dead_code)]

use std::rc::Rc;

use crate::des::general::genetic_tsp::{
    build_pentagon_tsp as build_pentagon_tsp_model, build_random_tsp as build_random_tsp_model,
    held_karp_exact as held_karp_exact_model, tour_length as tour_length_model, HeldKarpResult,
    TSPInstance, Tour,
};
use crate::des::general::milp_bnb::{
    build_knapsack_milp as build_knapsack_milp_model, solve_milp as solve_milp_model, MILPProblem,
    MILPSolution, MILPSolveOptions,
};
use crate::des::general::simulated_annealing::{
    build_knapsack_sa_problem as build_knapsack_sa_problem_model,
    build_tsp_sa_problem as build_tsp_sa_problem_model,
    run_simulated_annealing as run_simulated_annealing_model,
    temperature_at as temperature_at_model, CoolingSchedule, KnapsackInstance, KnapsackSaProblem,
    SASolverOptions, TSPSAProblemOptions, TspSaProblem,
};

type Cooling = CoolingSchedule;
type TspInstance = TSPInstance;
type MilpProblem = MILPProblem;
type MilpResult = MILPSolution;

enum SaProblem {
    Tsp(TspSaProblem),
    Knapsack(KnapsackSaProblem),
}

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
    SaProblem::Tsp(build_tsp_sa_problem_model(
        _inst.clone(),
        TSPSAProblemOptions::default(),
    ))
}

fn build_knapsack_sa_problem(values: &[f64], weights: &[f64], capacity: f64) -> SaProblem {
    SaProblem::Knapsack(build_knapsack_sa_problem_model(
        KnapsackInstance {
            values: values.to_vec(),
            weights: weights.to_vec(),
            capacity,
        },
        1e6,
    ))
}

fn sa_options(opts: &SaOpts) -> SASolverOptions {
    SASolverOptions {
        max_iterations: opts.max_iterations,
        cooling: opts.cooling,
        seed: Some(opts.seed as u32),
        stall_limit: opts.stall_limit,
        verbose: None,
        record_trace: None,
        trace_stride: None,
    }
}

fn run_simulated_annealing(problem: SaProblem, opts: &SaOpts) -> SaResult {
    match problem {
        SaProblem::Tsp(problem) => {
            let result = run_simulated_annealing_model::<Tour>(Rc::new(problem), sa_options(opts));
            SaResult {
                best_cost: result.best_cost,
                iterations: result.iterations,
                accepted_count: result.accepted_count,
                improve_count: result.improve_count,
                best_history: result.best_history,
                final_cost: result.final_cost,
            }
        }
        SaProblem::Knapsack(problem) => {
            let result =
                run_simulated_annealing_model::<Vec<f64>>(Rc::new(problem), sa_options(opts));
            SaResult {
                best_cost: result.best_cost,
                iterations: result.iterations,
                accepted_count: result.accepted_count,
                improve_count: result.improve_count,
                best_history: result.best_history,
                final_cost: result.final_cost,
            }
        }
    }
}

fn build_pentagon_tsp(n: usize, radius: f64) -> TspInstance {
    build_pentagon_tsp_model(n, radius)
}
fn build_random_tsp(n: usize, seed: u32) -> TspInstance {
    build_random_tsp_model(n, seed, None)
}
fn tour_length(inst: &TspInstance, tour: &[usize]) -> f64 {
    tour_length_model(inst, tour)
}
fn held_karp_exact(inst: &TspInstance) -> HeldKarpResult {
    held_karp_exact_model(inst)
}

fn temperature_at(s: &Cooling, k: usize) -> f64 {
    temperature_at_model(s, k)
}

fn build_knapsack_milp(values: &[f64], weights: &[f64], capacity: f64) -> MilpProblem {
    build_knapsack_milp_model(values.to_vec(), weights.to_vec(), capacity)
}
fn solve_milp(milp: &MilpProblem) -> MilpResult {
    solve_milp_model(milp, MILPSolveOptions::default())
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
        println!(
            "{}  {}{}",
            if ok { "  PASS" } else { "  FAIL" },
            label,
            tail
        );
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
                &SaOpts {
                    max_iterations: 5000,
                    cooling: Cooling::Geometric {
                        t0: 50.0,
                        alpha: 0.998,
                        t_min: None,
                    },
                    seed,
                    stall_limit: None,
                },
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
                    &SaOpts {
                        max_iterations: 30000,
                        cooling: Cooling::Geometric {
                            t0: 50.0,
                            alpha: 0.9995,
                            t_min: None,
                        },
                        seed: 1,
                        stall_limit: None,
                    },
                );
                c.check(
                    &format!("2.x n={} seed={}: SA ratio ≤ 1.05 of exact", n, seed),
                    r.best_cost <= exact.length * 1.05 + 1e-9,
                    &format!(
                        "SA={:.4}, exact={:.4}, ratio={:.4}",
                        r.best_cost,
                        exact.length,
                        r.best_cost / exact.length
                    ),
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
                &SaOpts {
                    max_iterations: 10000,
                    cooling: Cooling::Geometric {
                        t0: 50.0,
                        alpha: 0.999,
                        t_min: None,
                    },
                    seed: trial,
                    stall_limit: None,
                },
            );
            let sa_value = -sa.best_cost;
            c.check(
                &format!("3.x trial={}: SA value ≥ 0.95 × exact", trial),
                sa_value >= 0.95 * exact.z - 1e-6,
                &format!(
                    "SA={:.2}, exact={:.2}, ratio={:.4}",
                    sa_value,
                    exact.z,
                    sa_value / exact.z
                ),
            );
        }
    }

    println!("\nStudy 4 — Cooling schedules");
    {
        let geom = Cooling::Geometric {
            t0: 100.0,
            alpha: 0.99,
            t_min: None,
        };
        let lin = Cooling::Linear {
            t0: 100.0,
            rate: 1.0,
            t_min: None,
        };
        let log = Cooling::Logarithmic {
            t0: 100.0,
            t_min: None,
        };
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
        c.check(
            "4.1 geometric schedule monotone non-increasing",
            geo_mono,
            "",
        );
        c.check("4.2 linear schedule monotone non-increasing", lin_mono, "");
        c.check(
            "4.3 logarithmic schedule monotone non-increasing",
            log_mono,
            "",
        );
        let t = temperature_at(
            &Cooling::Geometric {
                t0: 100.0,
                alpha: 0.5,
                t_min: Some(0.01),
            },
            1000,
        );
        c.check(
            "4.4 Tmin floor enforced",
            t == 0.01,
            &format!("T(1000) = {}", t),
        );
        let t0g = temperature_at(
            &Cooling::Geometric {
                t0: 50.0,
                alpha: 0.99,
                t_min: None,
            },
            0,
        );
        let t0l = temperature_at(
            &Cooling::Linear {
                t0: 50.0,
                rate: 1.0,
                t_min: None,
            },
            0,
        );
        c.check("4.5 geometric T(0) = T0", t0g == 50.0, "");
        c.check("4.6 linear T(0) = T0", t0l == 50.0, "");
    }

    println!("\nStudy 5 — Reproducibility: same seed → same trajectory");
    {
        let inst = build_random_tsp(10, 1);
        let sa1 = run_simulated_annealing(
            build_tsp_sa_problem(&inst),
            &SaOpts {
                max_iterations: 1000,
                cooling: Cooling::Geometric {
                    t0: 50.0,
                    alpha: 0.99,
                    t_min: None,
                },
                seed: 42,
                stall_limit: None,
            },
        );
        let sa2 = run_simulated_annealing(
            build_tsp_sa_problem(&inst),
            &SaOpts {
                max_iterations: 1000,
                cooling: Cooling::Geometric {
                    t0: 50.0,
                    alpha: 0.99,
                    t_min: None,
                },
                seed: 42,
                stall_limit: None,
            },
        );
        c.check(
            "5.1 same seed: same best cost",
            close(sa1.best_cost, sa2.best_cost, 1e-12),
            "",
        );
        c.check(
            "5.2 same seed: same iteration count",
            sa1.iterations == sa2.iterations,
            "",
        );
        c.check(
            "5.3 same seed: same accepted count",
            sa1.accepted_count == sa2.accepted_count,
            "",
        );
        let sa3 = run_simulated_annealing(
            build_tsp_sa_problem(&inst),
            &SaOpts {
                max_iterations: 1000,
                cooling: Cooling::Geometric {
                    t0: 50.0,
                    alpha: 0.99,
                    t_min: None,
                },
                seed: 99,
                stall_limit: None,
            },
        );
        c.check(
            "5.4 different seed: different bestHistory[100]",
            (sa1.best_history[100] - sa3.best_history[100]).abs() > 1e-9
                || (sa1.best_history[500] - sa3.best_history[500]).abs() > 1e-9,
            "",
        );
    }

    println!("\nStudy 6 — Best history is monotonic (best can only improve)");
    {
        let inst = build_random_tsp(15, 4);
        let sa = run_simulated_annealing(
            build_tsp_sa_problem(&inst),
            &SaOpts {
                max_iterations: 5000,
                cooling: Cooling::Geometric {
                    t0: 50.0,
                    alpha: 0.999,
                    t_min: None,
                },
                seed: 1,
                stall_limit: None,
            },
        );
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
        let sa_hot = run_simulated_annealing(
            build_tsp_sa_problem(&inst),
            &SaOpts {
                max_iterations: 5000,
                cooling: Cooling::Geometric {
                    t0: 1000.0,
                    alpha: 1.0,
                    t_min: None,
                },
                seed: 1,
                stall_limit: None,
            },
        );
        let sa_cold = run_simulated_annealing(
            build_tsp_sa_problem(&inst),
            &SaOpts {
                max_iterations: 5000,
                cooling: Cooling::Geometric {
                    t0: 1e-12,
                    alpha: 1.0,
                    t_min: None,
                },
                seed: 1,
                stall_limit: None,
            },
        );
        let hot_rate = sa_hot.accepted_count as f64 / sa_hot.iterations as f64;
        let cold_rate = sa_cold.accepted_count as f64 / sa_cold.iterations as f64;
        c.check(
            "7.1 high-T accept rate > low-T accept rate",
            hot_rate > cold_rate,
            &format!("hot={:.3}, cold={:.3}", hot_rate, cold_rate),
        );
        let ratio = sa_cold.improve_count as f64 / (1usize.max(sa_cold.accepted_count)) as f64;
        c.check(
            "7.2 cold-T improveCount/acceptedCount ratio low (only improvements + zero-Δ)",
            ratio >= 0.1,
            &format!(
                "improvements={}, accepted={}, ratio={:.3}",
                sa_cold.improve_count, sa_cold.accepted_count, ratio
            ),
        );
        c.check(
            "7.3 cold-T finalCost ≤ initial",
            sa_cold.final_cost <= sa_cold.best_history[0] + 1e-9,
            &format!(
                "final={:.2}, init={:.2}",
                sa_cold.final_cost, sa_cold.best_history[0]
            ),
        );
    }

    println!("\nStudy 8 — Stall-limit early stopping");
    {
        let inst = build_random_tsp(8, 1);
        let sa = run_simulated_annealing(
            build_tsp_sa_problem(&inst),
            &SaOpts {
                max_iterations: 100000,
                cooling: Cooling::Geometric {
                    t0: 0.001,
                    alpha: 1.0,
                    t_min: None,
                },
                seed: 1,
                stall_limit: Some(50),
            },
        );
        c.check(
            "8.1 stall-limit triggers early termination",
            sa.iterations < 100000,
            &format!("iterations = {} (< 100000)", sa.iterations),
        );
    }

    println!("\n  ─────────────────────────────────────────────────────────────────────────");
    println!("  {} passed, {} failed", c.pass, c.fail);
    std::process::exit(if c.fail == 0 { 0 } else { 1 });
}
