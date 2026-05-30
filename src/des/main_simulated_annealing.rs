//! Port of `src/des/main-simulated-annealing.ts`.
//!
//! CLI driver for Simulated Annealing: TSP at several sizes, a cooling-schedule
//! comparison, and 0/1 knapsack SA vs MILP-B&B.
//!
//! Conversion notes:
//!   - SA acceptance routes through the injected RNG inside
//!     `general::simulated_annealing` (seeded, deterministic).
//!   - `CoolingSchedule` union → enum; problems are passed as
//!     `Rc<dyn SAProblem<S>>`.
//!   - TS reuses one deterministic instance for multiple consuming calls; here
//!     the instance is rebuilt with the same seed (identical) where ownership
//!     would otherwise be moved twice.

use std::rc::Rc;
use std::time::Instant;

use crate::des::general::genetic_tsp::{
    build_pentagon_tsp, build_random_tsp, held_karp_exact, run_genetic_tsp, tour_length,
    GASolverOptions, InitMode, Tour,
};
use crate::des::general::milp_bnb::{build_knapsack_milp, solve_milp, MILPSolveOptions};
use crate::des::general::simulated_annealing::{
    build_knapsack_sa_problem, build_tsp_sa_problem, run_simulated_annealing, CoolingSchedule,
    KnapsackInstance, SAProblem, SASolverOptions, TSPSAProblemOptions,
};

fn header(s: &str) {
    println!();
    println!("{}", "═".repeat(96));
    println!("  {}", s);
    println!("{}", "═".repeat(96));
}

fn sa_opts(max_iterations: usize, cooling: CoolingSchedule, seed: u32) -> SASolverOptions {
    SASolverOptions {
        max_iterations,
        cooling,
        seed: Some(seed),
        stall_limit: None,
        verbose: None,
        record_trace: None,
        trace_stride: None,
    }
}

fn default_tsp_sa_opts() -> TSPSAProblemOptions {
    TSPSAProblemOptions { penalty_per_violation: None, init: None, moves: None }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    header("STUDY 1 — Pentagon TSP (n=5): SA finds the exact optimum");
    {
        let inst = build_pentagon_tsp(5, 50.0);
        let optimum = tour_length(&inst, &[0, 1, 2, 3, 4]);
        let sa_p: Rc<dyn SAProblem<Tour>> = Rc::new(build_tsp_sa_problem(inst, default_tsp_sa_opts()));
        let r = run_simulated_annealing(
            sa_p,
            sa_opts(2000, CoolingSchedule::Geometric { t0: 50.0, alpha: 0.998, t_min: None }, 1),
        );
        println!("  optimum (perimeter) = {:.4}", optimum);
        println!("  SA best             = {:.4}    ratio = {:.6}", r.best_cost, r.best_cost / optimum);
        println!(
            "  iters = {}, accepted = {}, improvements = {}",
            r.iterations, r.accepted_count, r.improve_count
        );
    }

    header("STUDY 2 — n=12 random TSP: SA vs Held–Karp (exact) vs GA");
    {
        let inst = build_random_tsp(12, 17, None);
        let t0 = Instant::now();
        let exact = held_karp_exact(&inst);
        let dt_exact = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let sa_p: Rc<dyn SAProblem<Tour>> = Rc::new(build_tsp_sa_problem(inst, default_tsp_sa_opts()));
        let sa = run_simulated_annealing(
            sa_p,
            sa_opts(20000, CoolingSchedule::Geometric { t0: 50.0, alpha: 0.9995, t_min: None }, 1),
        );
        let dt_sa = t1.elapsed().as_millis();
        let t2 = Instant::now();
        let ga = run_genetic_tsp(
            build_random_tsp(12, 17, None),
            GASolverOptions {
                population_size: Some(80),
                num_generations: Some(200),
                seed: Some(1),
                init: Some(InitMode::NearestNeighbor),
                ..Default::default()
            },
        );
        let dt_ga = t2.elapsed().as_millis();
        println!("  Held–Karp (exact)    z = {:.4}    wall = {} ms", exact.length, dt_exact);
        println!(
            "  SA       z = {:.4}  ratio = {:.6}    wall = {} ms ({} iters)",
            sa.best_cost,
            sa.best_cost / exact.length,
            dt_sa,
            sa.iterations
        );
        println!(
            "  GA       z = {:.4}  ratio = {:.6}    wall = {} ms ({} generations)",
            ga.best_length,
            ga.best_length / exact.length,
            dt_ga,
            ga.generations
        );
    }

    header("STUDY 3 — n=30 random TSP: SA vs GA, equal compute");
    {
        let t0 = Instant::now();
        let sa_p: Rc<dyn SAProblem<Tour>> =
            Rc::new(build_tsp_sa_problem(build_random_tsp(30, 99, None), default_tsp_sa_opts()));
        let sa = run_simulated_annealing(
            sa_p,
            sa_opts(100000, CoolingSchedule::Geometric { t0: 200.0, alpha: 0.99995, t_min: None }, 1),
        );
        let dt_sa = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let ga = run_genetic_tsp(
            build_random_tsp(30, 99, None),
            GASolverOptions {
                population_size: Some(200),
                num_generations: Some(500),
                seed: Some(1),
                init: Some(InitMode::NearestNeighbor),
                ..Default::default()
            },
        );
        let dt_ga = t1.elapsed().as_millis();
        println!("  SA   z = {:.4}    wall = {} ms ({} iters)", sa.best_cost, dt_sa, sa.iterations);
        println!("  GA   z = {:.4}    wall = {} ms ({} generations)", ga.best_length, dt_ga, ga.generations);
        let winner = if sa.best_cost < ga.best_length { "SA" } else { "GA" };
        println!("  winner: {}    margin = {:.4}", winner, (sa.best_cost - ga.best_length).abs());
    }

    header("STUDY 4 — Cooling schedules on the same TSP");
    {
        // n = 20 > 14, so the Held–Karp exact reference is skipped (TS `null`).
        let schedules: Vec<(&str, CoolingSchedule, usize)> = vec![
            ("geometric  α=0.999, T0=50", CoolingSchedule::Geometric { t0: 50.0, alpha: 0.999, t_min: None }, 30000),
            ("geometric  α=0.9995, T0=200", CoolingSchedule::Geometric { t0: 200.0, alpha: 0.9995, t_min: None }, 30000),
            ("logarithmic T0=200", CoolingSchedule::Logarithmic { t0: 200.0, t_min: None }, 30000),
            ("linear     rate=0.005", CoolingSchedule::Linear { t0: 100.0, rate: 0.005, t_min: None }, 20000),
            ("exp-restart α=0.99, p=2000", CoolingSchedule::ExpRestart { t0: 50.0, alpha: 0.99, period: 2000, t_min: None }, 30000),
        ];
        println!("  {:<28}{:>10}{:>8}{:>10}", "schedule", "best", "iters", "wall(ms)");
        for (name, sched, iters) in schedules {
            let t0 = Instant::now();
            let sa_p: Rc<dyn SAProblem<Tour>> =
                Rc::new(build_tsp_sa_problem(build_random_tsp(20, 5, None), default_tsp_sa_opts()));
            let r = run_simulated_annealing(sa_p, sa_opts(iters, sched, 7));
            let dt = t0.elapsed().as_millis();
            println!(
                "  {:<28}{:>10}{:>8}{:>10}",
                name,
                format!("{:.2}", r.best_cost),
                r.iterations,
                dt
            );
        }
    }

    header("STUDY 5 — 0/1 knapsack: SA heuristic vs MILP-B&B exact");
    {
        let mut s: u32 = 1234;
        let mut rng = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s as f64 / 4_294_967_296.0
        };
        let n = 15usize;
        let v: Vec<f64> = (0..n).map(|_| (rng() * 50.0 + 1.0).floor()).collect();
        let w: Vec<f64> = (0..n).map(|_| (rng() * 25.0 + 1.0).floor()).collect();
        let cap = (w.iter().sum::<f64>() * 0.4).floor();
        println!("  n={}, capacity={}", n, num_str(cap));
        let t0 = Instant::now();
        let exact = solve_milp(&build_knapsack_milp(v.clone(), w.clone(), cap), MILPSolveOptions::default());
        let dt_exact = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let knap = KnapsackInstance { values: v, weights: w, capacity: cap };
        let sa_p: Rc<dyn SAProblem<Vec<f64>>> = Rc::new(build_knapsack_sa_problem(knap, 1e6));
        let sa = run_simulated_annealing(
            sa_p,
            sa_opts(5000, CoolingSchedule::Geometric { t0: 30.0, alpha: 0.999, t_min: None }, 11),
        );
        let dt_sa = t1.elapsed().as_millis();
        println!(
            "  MILP-B&B (exact):  z = {:.2}    wall = {} ms    nodes = {}",
            exact.z, dt_exact, exact.nodes_explored
        );
        println!(
            "  SA (heuristic):    z = {:.2}    wall = {} ms    iters = {}",
            -sa.best_cost, dt_sa, sa.iterations
        );
        let ratio = (-sa.best_cost) / exact.z;
        println!("  SA / exact = {:.6}    (1.0 = found exact optimum)", ratio);
    }
}

/// JS `String(x)` for a number: integer-valued floats print bare.
fn num_str(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}
