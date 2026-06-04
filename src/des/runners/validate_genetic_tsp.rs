//! Port of `src/des/runners/validate-genetic-tsp.ts`.
//!
//! Verifies the GA-TSP solver against known-optimal instances (pentagon,
//! Held–Karp) and its constraint-handling policies (cut / penalize / repair).
//! Top-level driver → [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust GA-TSP module for instance construction, DES-style GA
//!     evolution, exact Held-Karp checks, precedence handling, and 1-tree bounds.

#![allow(dead_code)]

use crate::des::general::genetic_tsp::{
    build_pentagon_tsp as build_pentagon_tsp_model, build_random_tsp as build_random_tsp_model,
    check_precedence as check_precedence_model, held_karp_exact as held_karp_exact_model,
    is_permutation as is_permutation_model, one_tree_lower_bound as one_tree_lower_bound_model,
    run_genetic_tsp as run_genetic_tsp_model, tour_length as tour_length_model, Feasibility,
    GASolverOptions, GASolverResult, HeldKarpResult, InitMode, TSPInstance,
};

type TspInstance = TSPInstance;
type GaResult = GASolverResult;

#[derive(Clone, Debug, Default)]
struct GaOptions {
    population_size: usize,
    num_generations: usize,
    seed: u64,
    feasibility: Option<&'static str>,
    retry_limit: Option<usize>,
    penalty_per_violation: Option<f64>,
    elitism: Option<usize>,
    init: Option<&'static str>,
}

fn build_pentagon_tsp(n: usize, radius: f64) -> TspInstance {
    build_pentagon_tsp_model(n, radius)
}

fn build_random_tsp(n: usize, seed: u32, precedence: Option<Vec<(usize, usize)>>) -> TspInstance {
    build_random_tsp_model(n, seed, precedence)
}

fn parse_feasibility(value: Option<&str>) -> Option<Feasibility> {
    match value {
        Some("cut") => Some(Feasibility::Cut),
        Some("penalize") => Some(Feasibility::Penalize),
        Some("repair") => Some(Feasibility::Repair),
        Some(other) => panic!("unknown GA feasibility mode: {other}"),
        None => None,
    }
}

fn parse_init_mode(value: Option<&str>) -> Option<InitMode> {
    match value {
        Some("random") => Some(InitMode::Random),
        Some("nearest-neighbor") => Some(InitMode::NearestNeighbor),
        Some(other) => panic!("unknown GA init mode: {other}"),
        None => None,
    }
}

fn run_genetic_tsp(instance: &TspInstance, options: &GaOptions) -> GaResult {
    run_genetic_tsp_model(
        instance.clone(),
        GASolverOptions {
            population_size: Some(options.population_size),
            num_generations: Some(options.num_generations),
            seed: Some(options.seed as u32),
            feasibility: parse_feasibility(options.feasibility),
            retry_limit: options.retry_limit,
            penalty_per_violation: options.penalty_per_violation,
            elitism: options.elitism,
            init: parse_init_mode(options.init),
            ..Default::default()
        },
    )
}

fn is_permutation(tour: &[usize], n: usize) -> bool {
    is_permutation_model(tour, n)
}

fn tour_length(instance: &TspInstance, tour: &[usize]) -> f64 {
    tour_length_model(instance, tour)
}

fn check_precedence(instance: &TspInstance, tour: &[usize]) -> Option<(usize, usize)> {
    check_precedence_model(instance, tour)
}

fn held_karp_exact(instance: &TspInstance) -> HeldKarpResult {
    held_karp_exact_model(instance)
}

fn one_tree_lower_bound(instance: &TspInstance) -> f64 {
    one_tree_lower_bound_model(instance)
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

/// `validate-genetic-tsp.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    println!("\nStudy 1 — Pentagon: GA reaches the analytical optimum");
    {
        let n = 5usize;
        let r_radius = 50.0;
        let inst = build_pentagon_tsp(n, r_radius);
        let optimal = n as f64 * 2.0 * r_radius * (std::f64::consts::PI / n as f64).sin();
        println!("    analytical optimum = {:.6}", optimal);
        let r = run_genetic_tsp(
            &inst,
            &GaOptions {
                population_size: 60,
                num_generations: 100,
                seed: 1,
                ..Default::default()
            },
        );
        println!("    GA best length     = {:.6}", r.best_length);
        c.check(
            "GA finds an optimal pentagon tour (within 1e-9)",
            (r.best_length - optimal).abs() < 1e-9,
            &format!("Δ = {:.2e}", r.best_length - optimal),
        );
        c.check(
            "best tour is a valid permutation",
            is_permutation(&r.best_tour, n),
            "",
        );
    }

    println!("\nStudy 2 — Small random instance: GA matches Held–Karp exact");
    {
        for seed_tsp in [3u32, 17, 99] {
            let inst = build_random_tsp(10, seed_tsp, None);
            let exact = held_karp_exact(&inst);
            let r = run_genetic_tsp(
                &inst,
                &GaOptions {
                    population_size: 80,
                    num_generations: 200,
                    seed: (seed_tsp + 1) as u64,
                    ..Default::default()
                },
            );
            println!(
                "    seed {}: HK = {:.3}, GA = {:.3}",
                seed_tsp, exact.length, r.best_length
            );
            c.check(
                &format!("seed={}: GA within 0.5% of Held–Karp optimum", seed_tsp),
                r.best_length <= exact.length * 1.005,
                &format!(
                    "gap = {:.3}%",
                    (r.best_length - exact.length) / exact.length * 100.0
                ),
            );
        }
    }

    println!("\nStudy 3 — 1-tree lower bound is a valid bound");
    {
        for n in [8usize, 12, 15] {
            let inst = build_random_tsp(n, n as u32, None);
            let lb = one_tree_lower_bound(&inst);
            let r = run_genetic_tsp(
                &inst,
                &GaOptions {
                    population_size: 60,
                    num_generations: 100,
                    seed: (n + 100) as u64,
                    ..Default::default()
                },
            );
            println!(
                "    n={}: 1-tree lb = {:.2}, GA best = {:.2}",
                n, lb, r.best_length
            );
            c.check(
                &format!("n={}: 1-tree lower bound ≤ GA best", n),
                lb <= r.best_length + 1e-9,
                &format!("lb={:.3}, ga={:.3}", lb, r.best_length),
            );
        }
    }

    println!("\nStudy 4 — Precedence constraints: all branches respected");
    {
        let mut inst = build_random_tsp(15, 42, None);
        inst.precedence = Some(vec![(0, 14), (1, 13), (2, 12), (3, 11)]);
        let r = run_genetic_tsp(
            &inst,
            &GaOptions {
                population_size: 80,
                num_generations: 150,
                seed: 42,
                feasibility: Some("cut"),
                retry_limit: Some(12),
                ..Default::default()
            },
        );
        c.check(
            "best tour is a valid permutation",
            is_permutation(&r.best_tour, 15),
            "",
        );
        c.check(
            "best tour respects all 4 precedence pairs (no violations remain)",
            check_precedence(&inst, &r.best_tour).is_none(),
            &format!("violation: {:?}", check_precedence(&inst, &r.best_tour)),
        );
        println!(
            "    feasible kids evaluated = {}, infeasible kids cut = {}",
            r.total_feasible_evaluated, r.total_infeasible_cut
        );
        c.check(
            "at least some children were infeasible (i.e. branch-cutting active)",
            r.total_infeasible_cut > 0,
            &format!("cut count = {}", r.total_infeasible_cut),
        );
    }

    println!("\nStudy 5 — Constraint policies converge differently");
    {
        let mut inst = build_random_tsp(16, 11, None);
        inst.precedence = Some(vec![(0, 15), (2, 13), (4, 11), (6, 9)]);
        let cut = run_genetic_tsp(
            &inst,
            &GaOptions {
                population_size: 80,
                num_generations: 200,
                seed: 11,
                feasibility: Some("cut"),
                retry_limit: Some(12),
                ..Default::default()
            },
        );
        let pen = run_genetic_tsp(
            &inst,
            &GaOptions {
                population_size: 80,
                num_generations: 200,
                seed: 11,
                feasibility: Some("penalize"),
                penalty_per_violation: Some(1e6),
                ..Default::default()
            },
        );
        let repair = run_genetic_tsp(
            &inst,
            &GaOptions {
                population_size: 80,
                num_generations: 200,
                seed: 11,
                feasibility: Some("repair"),
                ..Default::default()
            },
        );
        println!(
            "    cut     best length = {:.3}, infeasible cut = {}",
            cut.best_length, cut.total_infeasible_cut
        );
        println!(
            "    penalize best length = {:.3} (might include +∞ if infeasible)",
            pen.best_length
        );
        println!("    repair  best length = {:.3}", repair.best_length);
        c.check(
            "cut policy: best tour is feasible",
            check_precedence(&inst, &cut.best_tour).is_none(),
            "",
        );
        c.check(
            "repair policy: best tour is feasible",
            check_precedence(&inst, &repair.best_tour).is_none(),
            "",
        );
        c.check(
            "cut policy actually cut some offspring",
            cut.total_infeasible_cut > 0,
            "",
        );
    }

    println!("\nStudy 6 — Convergence: best tour length is monotone non-increasing");
    {
        let inst = build_random_tsp(20, 7, None);
        let r = run_genetic_tsp(
            &inst,
            &GaOptions {
                population_size: 80,
                num_generations: 200,
                seed: 7,
                elitism: Some(4),
                ..Default::default()
            },
        );
        let mut monotone = true;
        for g in 1..r.per_generation_best.len() {
            if r.per_generation_best[g] > r.per_generation_best[g - 1] + 1e-9 {
                monotone = false;
                break;
            }
        }
        let head = r
            .per_generation_best
            .iter()
            .take(15)
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
            .join(" ");
        c.check(
            "elitism guarantees best-so-far is monotone non-increasing",
            monotone,
            &format!("first {} = {}", 15.min(r.per_generation_best.len()), head),
        );
        let last = *r.per_generation_best.last().unwrap();
        let first = r.per_generation_best[0];
        c.check(
            "GA improves over the initial population",
            last < first,
            &format!("{:.2} → {:.2}", first, last),
        );
    }

    println!(
        "\n{} checks: {} passed, {} failed",
        c.pass + c.fail,
        c.pass,
        c.fail
    );
    if c.fail > 0 {
        std::process::exit(1);
    }
}
