//! Port of `src/des/main-genetic-tsp.ts`.
//!
//! CLI: solve TSP with a genetic algorithm modelled as a DES, with optional
//! precedence constraints (the branch-cutting pathway).
//!
//! Delegates to `crate::des::general::genetic_tsp`. `process.env.*` →
//! `std::env::var`; GA randomness is reproducible via the option `seed`.
//!
//! PORT NOTE: `ANIMATE=1` rendering (FrameRecorder + genetic-tsp scene) is
//! omitted — no animation engine is ported.

#![allow(dead_code)]

use std::time::Instant;

use crate::des::general::genetic_tsp::{
    build_pentagon_tsp, build_random_tsp, check_precedence, held_karp_exact, is_permutation,
    one_tree_lower_bound, run_genetic_tsp, Feasibility, GASolverOptions, InitMode, TSPInstance,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn stringify_precedence(pp: &Option<Vec<(usize, usize)>>) -> String {
    match pp {
        Some(p) => format!(
            "[{}]",
            p.iter()
                .map(|(a, b)| format!("[{a},{b}]"))
                .collect::<Vec<_>>()
                .join(",")
        ),
        None => "null".to_string(),
    }
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let n = env_usize("N_CITIES", 25);
    let seed = env_usize("SEED", 7) as u32;
    let generations = env_usize("GENERATIONS", 200);
    let pop_size = env_usize("POP", 100);
    let use_precedence = std::env::var("PRECEDENCE").as_deref() == Ok("1");
    let feasibility = match std::env::var("FEASIBILITY")
        .unwrap_or_else(|_| "cut".into())
        .as_str()
    {
        "penalize" => Feasibility::Penalize,
        "repair" => Feasibility::Repair,
        _ => Feasibility::Cut,
    };
    let animate = std::env::var("ANIMATE").as_deref() == Ok("1");
    let instance_kind = std::env::var("INSTANCE").unwrap_or_else(|_| "random".into());

    let mut instance: TSPInstance = if instance_kind == "pentagon" {
        build_pentagon_tsp(n, 50.0)
    } else {
        build_random_tsp(n, seed, None)
    };
    if use_precedence {
        let mut pp: Vec<(usize, usize)> = Vec::new();
        for i in 0..4.min(n / 3) {
            pp.push((i, n - 1 - i));
        }
        instance.precedence = Some(pp);
    }

    println!("# Genetic-TSP solver (GA inside the DES engine)");
    println!("# n={n} cities, instance={instance_kind}, seed={seed}");
    println!("# population={pop_size}, generations={generations}");
    if use_precedence {
        println!(
            "# precedence pairs: {}, feasibility={}",
            stringify_precedence(&instance.precedence),
            match feasibility {
                Feasibility::Cut => "cut",
                Feasibility::Penalize => "penalize",
                Feasibility::Repair => "repair",
            }
        );
    } else {
        println!("# no precedence constraints");
    }
    println!();
    println!(
        "# Lower bound (1-tree relaxation): {:.3}",
        one_tree_lower_bound(&instance)
    );
    if n <= 14 && !use_precedence {
        print!("# Computing exact Held–Karp optimum ... ");
        let t0 = Instant::now();
        let hk = held_karp_exact(&instance);
        println!(
            "length = {:.3} in {}ms",
            hk.length,
            t0.elapsed().as_millis()
        );
    } else {
        println!("# (Held–Karp skipped: n > 14 or precedence active)");
    }
    println!();

    print!("# Running GA ... ");
    let result = run_genetic_tsp(
        instance.clone(),
        GASolverOptions {
            population_size: Some(pop_size),
            num_generations: Some(generations),
            seed: Some(seed + 1000),
            feasibility: Some(feasibility),
            elitism: Some(4),
            init: Some(InitMode::NearestNeighbor),
            ..Default::default()
        },
    );
    println!("done in {}ms", result.elapsed_ms as i64);
    println!();

    println!("# Best tour length found  = {:.3}", result.best_length);
    println!(
        "# Best tour valid permutation? {}",
        is_permutation(&result.best_tour, instance.n)
    );
    println!(
        "# Best tour feasible (precedence)? {}",
        check_precedence(&instance, &result.best_tour).is_none()
    );
    println!(
        "# Total feasible children evaluated  = {}",
        result.total_feasible_evaluated
    );
    println!(
        "# Total infeasible children cut      = {}",
        result.total_infeasible_cut
    );
    println!();
    println!(
        "# Tour: {} → {}",
        result
            .best_tour
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" → "),
        result.best_tour[0]
    );
    println!();

    let step = (generations / 10).max(1);
    println!("# Convergence (sampled):");
    println!("#   gen     best        mean");
    let mut g = 0;
    while g < generations {
        println!(
            "#   {:>4}  {:>8}  {:>8}",
            g,
            format!("{:.3}", result.per_generation_best[g]),
            format!("{:.3}", result.per_generation_mean[g])
        );
        g += step;
    }
    println!();

    if animate {
        println!("# (animation omitted in Rust port — see PORT NOTE)");
    }
}
