//! Port of `src/des/runners/steady-state.ts`.
//!
//! Mathematical verification driver: the closed-form steady state from the
//! difference equations vs four numerical kernels (difference, ODE-RK4,
//! Gillespie SSA, FEL-individual) run as an **open system**
//! (`sourceCap = +∞`, `phase1Days = +∞`). The TS top-level `main()` becomes
//! [`run`].
//!
//! ## PORT NOTE
//!
//!   * `process.env.{N,HORIZON}` → `std::env::var`.
//!   * `Number.POSITIVE_INFINITY` → `f64::INFINITY`.
//!   * stochastic seeds `0xA0000+i` / `0xB0000+i` kept verbatim.
//!   * `console.log` → `println!`; `ReturnType<typeof runDifferenceOnce>` →
//!     `RunResult`. The `diffRuns` map keys on `f64::to_bits` since `f64` is not
//!     `Hash`/`Eq`.

#![allow(dead_code)]

use std::collections::HashMap;

use super::difference_runner::{analytical_steady_state, max_stable_step, run_difference_once};
use super::fel_runner::run_fel_once;
use super::gillespie_runner::run_gillespie_once;
use super::ode_runner::run_ode_once;
use super::stats::{mean, stddev};
use super::types::{
    default_config, RunOpts, RunResult, ServiceDiscipline, SimConfig, COMPARTMENT_ORDER,
};

fn fmt(n: f64, d: usize) -> String {
    if n.is_finite() {
        format!("{n:.d$}")
    } else {
        "DIVERGED".to_string()
    }
}

fn pad_end(s: &str, width: usize) -> String {
    if s.chars().count() >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - s.chars().count()))
    }
}

fn pad_start(s: &str, width: usize) -> String {
    if s.chars().count() >= width {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(width - s.chars().count()))
    }
}

fn pop(map: &HashMap<String, f64>, c: &str) -> f64 {
    map.get(c).copied().unwrap_or(0.0)
}

/// `main()` — run the steady-state verification.
pub fn run() {
    let n_reps: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let horizon: f64 = std::env::var("HORIZON")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000.0);

    let cfg_open = SimConfig {
        source_cap: f64::INFINITY,
        phase1_days: f64::INFINITY,
        horizon_days: horizon,
        step_size: 0.05,
        ..default_config()
    };

    println!("steady-state verification: open system (lambda const), horizon={horizon}d");
    println!();

    // --- Closed-form analytical -------------------------------------------
    let a = analytical_steady_state(&cfg_open);
    let lifespan = (1.0 / a.q) * (3.0 * 0.3 + 0.4 * 0.3 + 0.6 * (0.3 + 0.2 * (0.3 + 0.2)) + 2.0);
    println!("=== closed-form analytical steady state ===");
    println!(
        "  arrival rate                lambda = 1/mu_arr      = {}/day",
        fmt(a.lambda, 4)
    );
    println!(
        "  per-S-pass death fraction   q = (1-p_a)*p_h*p_d    = {}",
        fmt(a.q, 5)
    );
    println!(
        "  S throughput                f_S = lambda/q         = {}/day",
        fmt(a.f_s, 3)
    );
    println!(
        "  total alive at steady state Sum N*_alive           = {}",
        fmt(a.total_alive - pop(&a.populations, "D"), 3)
    );
    println!(
        "  mean lifespan               1/q * cycle_time       ≈ {} days",
        fmt(lifespan, 1)
    );
    println!(
        "  max stable forward-Euler dt 2 * min(mu_c)          = {} days",
        fmt(max_stable_step(&cfg_open), 3)
    );
    println!();

    // --- Forward-Euler stability + convergence demo -----------------------
    println!("=== forward-Euler difference equation: stability + convergence demo ===");
    println!("  (compares N(T) to analytical N* across a range of dt)");
    let dts = [0.5_f64, 0.39, 0.1, 0.05, 0.01];
    let mut diff_runs: HashMap<u64, RunResult> = HashMap::new();
    for &dt in &dts {
        let cfg = SimConfig {
            step_size: dt,
            ..cfg_open.clone()
        };
        diff_runs.insert(dt.to_bits(), run_difference_once(&cfg, &RunOpts::default()));
    }
    let mut header = pad_end("compartment", 14) + &pad_start("analytical", 12);
    for d in dts {
        header += &pad_start(&format!("dt={d}"), 12);
    }
    println!("{header}");
    for c in COMPARTMENT_ORDER {
        let mut line =
            pad_end(&format!("<{c}>"), 14) + &pad_start(&fmt(pop(&a.populations, c), 3), 12);
        for d in dts {
            let v = pop(&diff_runs[&d.to_bits()].final_populations, c);
            line += &pad_start(&fmt(v, 3), 12);
        }
        println!("{line}");
    }
    let dt_ref = 0.01_f64;
    let mut max_err = f64::NEG_INFINITY;
    for c in COMPARTMENT_ORDER {
        let e = (pop(&diff_runs[&dt_ref.to_bits()].final_populations, c) - pop(&a.populations, c))
            .abs();
        if e > max_err {
            max_err = e;
        }
    }
    println!(
        "  max |diff(dt={dt_ref}) - analytical| over compartments: {}",
        fmt(max_err, 6)
    );
    if !max_err.is_finite() || max_err > 1e-2 {
        eprintln!(
            "[steady-state] difference-equation final state disagrees with closed form at dt={dt_ref}: max|Δ|={} — kernel and analytics may be inconsistent.",
            fmt(max_err, 6)
        );
    }
    println!(
        "  dt=0.5 > maxStableStep={} -> DIVERGED, as predicted by stability analysis.",
        fmt(max_stable_step(&cfg_open), 2)
    );
    println!();

    // --- Run all kernels in open-system mode at horizon T -----------------
    let ode = run_ode_once(
        &SimConfig {
            horizon_days: horizon,
            ..cfg_open.clone()
        },
        &RunOpts::default(),
    );
    if COMPARTMENT_ORDER
        .iter()
        .any(|c| !pop(&ode.final_populations, c).is_finite())
    {
        eprintln!(
            "[steady-state] ODE RK4 produced non-finite final populations at horizon={horizon} (stepSize={}) — integration likely diverged.",
            cfg_open.step_size
        );
    }
    let mut ssa_runs: Vec<RunResult> = Vec::new();
    let mut fel_runs: Vec<RunResult> = Vec::new();
    for i in 0..n_reps {
        ssa_runs.push(run_gillespie_once(
            &cfg_open,
            &RunOpts {
                seed: Some(0xA0000 + i as u64),
                ..Default::default()
            },
        ));
        fel_runs.push(run_fel_once(
            &cfg_open,
            &RunOpts {
                seed: Some(0xB0000 + i as u64),
                service: Some(ServiceDiscipline::Individual),
                ..Default::default()
            },
        ));
    }

    let fel0 = &fel_runs[0];
    let ssa0 = &ssa_runs[0];
    println!("kernel timings (single rep):");
    println!("  ODE RK4         : {} ms", ode.elapsed_ms);
    let ssa_walls: Vec<f64> = ssa_runs.iter().map(|r| r.elapsed_ms as f64).collect();
    let fel_walls: Vec<f64> = fel_runs.iter().map(|r| r.elapsed_ms as f64).collect();
    println!(
        "  Gillespie SSA   : {} ms (mean={} ms across N={n_reps})",
        ssa0.elapsed_ms,
        format!("{:.0}", mean(&ssa_walls))
    );
    println!(
        "  FEL-individual  : {} ms (mean={} ms across N={n_reps})",
        fel0.elapsed_ms,
        format!("{:.0}", mean(&fel_walls))
    );
    println!();

    // --- Column A: fixed-point estimates ----------------------------------
    println!("=== fixed-point estimates of N*_c (should all agree) ===");
    println!(
        "{}{}{}{}{}{}",
        pad_end("compartment", 14),
        pad_start("analytical", 13),
        pad_start("diff N(T)", 13),
        pad_start("ODE N(T)", 13),
        pad_start("Gillespie <N(T)>", 20),
        pad_start("FEL-ind <N(T)>", 20)
    );
    for c in COMPARTMENT_ORDER {
        let ana = pop(&a.populations, c);
        let dif = pop(&diff_runs[&0.05_f64.to_bits()].final_populations, c);
        let ode_f = pop(&ode.final_populations, c);
        let ssa_f: Vec<f64> = ssa_runs
            .iter()
            .map(|r| pop(&r.final_populations, c))
            .collect();
        let fel_f: Vec<f64> = fel_runs
            .iter()
            .map(|r| pop(&r.final_populations, c))
            .collect();
        println!(
            "{}{}{}{}{}{}",
            pad_end(&format!("<{c}>"), 14),
            pad_start(&fmt(ana, 3), 13),
            pad_start(&fmt(dif, 3), 13),
            pad_start(&fmt(ode_f, 3), 13),
            pad_start(
                &format!("{} ± {}", fmt(mean(&ssa_f), 1), fmt(stddev(&ssa_f), 1)),
                20
            ),
            pad_start(
                &format!("{} ± {}", fmt(mean(&fel_f), 1), fmt(stddev(&fel_f), 1)),
                20
            )
        );
    }
    println!("  notes:");
    println!("   - \"diff N(T)\" and \"ODE N(T)\" are deterministic, exact at large T.");
    println!("   - \"<N(T)>\" is the average across N reps of the snapshot at t=T;");
    println!("     unbiased for N*_c but with sqrt(N*_c) Poisson-like variance per rep.");
    println!();

    // --- Column B: time-averaged populations ------------------------------
    println!("=== time-averaged populations <N_c>_[0,T] (deterministic vs stochastic) ===");
    println!(
        "{}{}{}{}{}",
        pad_end("compartment", 14),
        pad_start("analytical N*", 15),
        pad_start("ODE <N>_t", 13),
        pad_start("Gillespie <N>_t", 20),
        pad_start("FEL-ind <N>_t", 20)
    );
    for c in COMPARTMENT_ORDER {
        let ana = pop(&a.populations, c);
        let ode_t = pop(&ode.time_avg_populations, c);
        let ssa_t: Vec<f64> = ssa_runs
            .iter()
            .map(|r| pop(&r.time_avg_populations, c))
            .collect();
        let fel_t: Vec<f64> = fel_runs
            .iter()
            .map(|r| pop(&r.time_avg_populations, c))
            .collect();
        println!(
            "{}{}{}{}{}",
            pad_end(&format!("<{c}>"), 14),
            pad_start(&fmt(ana, 3), 15),
            pad_start(&fmt(ode_t, 3), 13),
            pad_start(
                &format!("{} ± {}", fmt(mean(&ssa_t), 3), fmt(stddev(&ssa_t), 3)),
                20
            ),
            pad_start(
                &format!("{} ± {}", fmt(mean(&fel_t), 3), fmt(stddev(&fel_t), 3)),
                20
            )
        );
    }
    println!("  notes:");
    println!("   - <N>_t < N*  for finite T, by an amount = transient deficit / T.");
    println!("   - ODE <N>_t is the integral of the trajectory; stochastic <N>_t");
    println!("     should match it, NOT the fixed point. They do.");
    println!();

    // --- Sanity checks ----------------------------------------------------
    println!("=== sanity: total alive populations and cumulative deaths ===");
    let all_ana = a.total_alive - pop(&a.populations, "D");
    let ode_alive: f64 = COMPARTMENT_ORDER
        .iter()
        .map(|c| pop(&ode.final_populations, c))
        .sum();
    let ssa_alive: Vec<f64> = ssa_runs
        .iter()
        .map(|r| {
            COMPARTMENT_ORDER
                .iter()
                .map(|c| pop(&r.final_populations, c))
                .sum()
        })
        .collect();
    let fel_alive: Vec<f64> = fel_runs
        .iter()
        .map(|r| {
            COMPARTMENT_ORDER
                .iter()
                .map(|c| pop(&r.final_populations, c))
                .sum()
        })
        .collect();
    println!("  analytical Sum N*_alive    : {}", fmt(all_ana, 3));
    println!("  ODE     N(T) Sum alive     : {}", fmt(ode_alive, 3));
    println!(
        "  Gillespie    <N(T)> alive  : {} ± {}",
        fmt(mean(&ssa_alive), 3),
        fmt(stddev(&ssa_alive), 3)
    );
    println!(
        "  FEL-ind      <N(T)> alive  : {} ± {}",
        fmt(mean(&fel_alive), 3),
        fmt(stddev(&fel_alive), 3)
    );

    println!();
    let exp_deaths = horizon * a.lambda;
    println!("cumulative deaths over [0, T] (steady-state rate = lambda = 1/day):");
    println!("  expected horizon * lambda  : {}", fmt(exp_deaths, 1));
    println!(
        "  ODE D(T)                   : {}",
        fmt(ode.totals.absorbed, 1)
    );
    let ssa_abs: Vec<f64> = ssa_runs.iter().map(|r| r.totals.absorbed).collect();
    let fel_abs: Vec<f64> = fel_runs.iter().map(|r| r.totals.absorbed).collect();
    println!(
        "  Gillespie absorbed         : {} ± {}",
        fmt(mean(&ssa_abs), 1),
        fmt(stddev(&ssa_abs), 1)
    );
    println!(
        "  FEL-ind   absorbed         : {} ± {}",
        fmt(mean(&fel_abs), 1),
        fmt(stddev(&fel_abs), 1)
    );
    println!("  (these are biased low by transient deficit; ratio actual/expected -> 1 as T -> infinity)");
}
