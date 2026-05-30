//! Port of `src/des/main-lp-factory.ts`.
//!
//! Simulation-optimisation: solve a factory scheduling LP (max weekly profit
//! subject to per-machine time budgets), then stress the nominal plan with a
//! DES (log-normal processing times, machine breakdowns, finite buffers) and
//! measure the realised vs. nominal gap.
//!
//! ## Rust shape
//!   * `LPProblem`/`LPSolution`, `solve_lp_then_simulate` are reused from
//!     `crate::des::general::{lp, des_lp_bridge}`. The solver is selected by the
//!     `LP_SOLVER` env var inside `solve_lp` (matching the TS default
//!     `scipy:highs`, internal fallback).
//!   * The minimal in-file DES (`simulate_factory`) is ported 1:1 with `f64`
//!     numeric state and `mulberry32` (`crate::des::general::prng`) for the
//!     log-normal / breakdown sampling.

#![allow(dead_code)]

use std::f64::consts::PI;

use crate::des::general::des_lp_bridge::solve_lp_then_simulate;
use crate::des::general::lp::{lp_to_string, LPProblem, LPSolution, LpSolverOptions, Sense};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

/// Products × machines problem data.
#[derive(Clone, Debug)]
pub struct FactoryProblem {
    pub products: Vec<String>,
    pub machines: Vec<String>,
    /// Mean processing time machine m × product p, in minutes. `tau[m][p]`.
    pub tau: Vec<Vec<f64>>,
    /// Weekly capacity per machine (minutes).
    pub capacity: Vec<f64>,
    /// Profit per finished unit.
    pub profit: Vec<f64>,
}

/// The canonical 3-product / 4-machine instance.
pub fn factory() -> FactoryProblem {
    FactoryProblem {
        products: vec!["Widget-A".into(), "Widget-B".into(), "Widget-C".into()],
        machines: vec![
            "Lathe".into(),
            "Mill".into(),
            "Drill".into(),
            "Press".into(),
        ],
        tau: vec![
            vec![3.0, 5.0, 2.5], // Lathe
            vec![2.5, 1.5, 4.0], // Mill
            vec![1.0, 2.0, 1.5], // Drill
            vec![4.0, 3.0, 2.0], // Press
        ],
        profit: vec![40.0, 30.0, 50.0],
        capacity: vec![2400.0, 2400.0, 2400.0, 2400.0],
    }
}

/// Build the factory LP (optionally shrinking the RHS by `robust_factor`).
pub fn build_factory_lp(prob: &FactoryProblem, robust_factor: f64) -> LPProblem {
    let m = prob.machines.len();
    let mut a_ub: Vec<Vec<f64>> = Vec::new();
    let mut b_ub: Vec<f64> = Vec::new();
    for mi in 0..m {
        a_ub.push(prob.tau[mi].clone());
        b_ub.push(prob.capacity[mi] * robust_factor);
    }
    LPProblem {
        sense: Sense::Max,
        c: prob.profit.clone(),
        a_ub: Some(a_ub),
        b_ub: Some(b_ub),
        a_eq: None,
        b_eq: None,
        lb: None,
        ub: None,
        var_names: Some(prob.products.clone()),
        con_names: Some(
            prob.machines
                .iter()
                .map(|m| format!("{m} capacity"))
                .collect(),
        ),
    }
}

/// Realised metrics from one DES replication.
#[derive(Clone, Debug)]
pub struct SimResult {
    pub realised_throughput: Vec<f64>,
    pub realised_revenue: f64,
    pub utilisation: Vec<f64>,
    pub breakdowns: Vec<f64>,
    pub wall_clock_min: f64,
}

/// DES disturbance parameters.
#[derive(Clone, Copy, Debug)]
pub struct SimParams {
    pub proc_cv: f64,
    pub break_prob_per_min: f64,
    pub break_duration_min: f64,
    pub total_min: i64,
    pub seed: u32,
}

/// Log-normal sample with given mean and coefficient of variation (Box-Muller).
fn lognormal_sample(mean: f64, cv: f64, rng: &mut SeededRandom) -> f64 {
    if cv <= 0.0 {
        return mean;
    }
    let sigma2 = (1.0 + cv * cv).ln();
    let sigma = sigma2.sqrt();
    let mu = mean.ln() - 0.5 * sigma2;
    let u1 = 1e-12_f64.max(rng.next_float());
    let u2 = rng.next_float();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    0.1_f64.max((mu + sigma * z).exp())
}

/// Flow units through the machines in series, with stochastic times +
/// breakdowns + finite buffers.
pub fn simulate_factory(prob: &FactoryProblem, plan: &[f64], params: SimParams) -> SimResult {
    let m = prob.machines.len();
    let p = prob.products.len();
    let mut rng = mulberry32(params.seed);

    // Round-robin job schedule capped at the planned total.
    let mut sched: Vec<usize> = Vec::new();
    let mut remaining: Vec<i64> = plan.iter().map(|x| x.floor() as i64).collect();
    while remaining.iter().any(|&r| r > 0) {
        for pi in 0..p {
            if remaining[pi] > 0 {
                sched.push(pi);
                remaining[pi] -= 1;
            }
        }
    }

    // Per-machine state.
    let mut slot: Vec<Option<(usize, i64)>> = vec![None; m]; // (productId, remaining)
    let mut buffer: Vec<Vec<usize>> = vec![Vec::new(); m + 1];
    buffer[0] = sched;
    let mut down_until: Vec<i64> = vec![-1; m];
    let mut breakdowns: Vec<f64> = vec![0.0; m];
    let mut utilisation: Vec<f64> = vec![0.0; m];

    for t in 0..params.total_min {
        // Step machines last → first.
        for mi in (0..m).rev() {
            if t >= down_until[mi]
                && slot[mi].is_some()
                && rng.next_float() < params.break_prob_per_min
            {
                breakdowns[mi] += 1.0;
                let dur = (-(1e-12_f64.max(rng.next_float())).ln() * params.break_duration_min)
                    .round() as i64;
                down_until[mi] = t + dur.max(1);
            }
            let is_down = t < down_until[mi];
            if let Some((product_id, rem)) = slot[mi] {
                if !is_down {
                    let new_rem = rem - 1;
                    utilisation[mi] += 1.0;
                    if new_rem <= 0 {
                        buffer[mi + 1].push(product_id);
                        slot[mi] = None;
                    } else {
                        slot[mi] = Some((product_id, new_rem));
                    }
                }
            }
            if slot[mi].is_none() && !buffer[mi].is_empty() {
                let product_id = buffer[mi].remove(0);
                let mean_t = prob.tau[mi][product_id];
                let proc_t = lognormal_sample(mean_t, params.proc_cv, &mut rng);
                slot[mi] = Some((product_id, proc_t.ceil() as i64));
            }
        }
    }

    let mut realised_throughput = vec![0.0; p];
    for &pid in &buffer[m] {
        realised_throughput[pid] += 1.0;
    }
    let mut realised_revenue = 0.0;
    for pi in 0..p {
        realised_revenue += prob.profit[pi] * realised_throughput[pi];
    }

    SimResult {
        realised_throughput,
        realised_revenue,
        utilisation: utilisation
            .iter()
            .map(|u| u / params.total_min as f64)
            .collect(),
        breakdowns,
        wall_clock_min: params.total_min as f64,
    }
}

/// Welch t statistic for nominal-vs-realised comparison.
#[derive(Clone, Copy, Debug)]
pub struct WelchT {
    pub t: f64,
    pub mean_a: f64,
    pub mean_b: f64,
    pub sd_a: f64,
    pub sd_b: f64,
}

pub fn welch_t(a: &[f64], b: &[f64]) -> WelchT {
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let variance = |xs: &[f64], mu: f64| {
        xs.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0).max(1.0)
    };
    let ma = mean(a);
    let mb = mean(b);
    let va = variance(a, ma);
    let vb = variance(b, mb);
    let t = (ma - mb) / (va / a.len() as f64 + vb / b.len() as f64).sqrt();
    WelchT {
        t,
        mean_a: ma,
        mean_b: mb,
        sd_a: va.sqrt(),
        sd_b: vb.sqrt(),
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Run the N replications used in `main` / the robustness sweep.
fn run_reps(
    prob: &FactoryProblem,
    plan: &LPSolution,
    n_reps: usize,
    params_base: SimParams,
) -> Vec<SimResult> {
    (0..n_reps)
        .map(|r| {
            simulate_factory(
                prob,
                &plan.x,
                SimParams {
                    seed: 1000 + r as u32,
                    ..params_base
                },
            )
        })
        .collect()
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let robust_factor = env_f64("ROBUST", 1.0);
    let proc_cv = env_f64("PROC_CV", 0.25);
    let break_prob = env_f64("BREAK_PROB", 0.002);
    let break_duration = env_f64("BREAK_DUR", 30.0);
    let total_min = env_f64("TOTAL_MIN", 2400.0) as i64;
    let n_reps = std::env::var("N_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30usize);
    let solver_label = std::env::var("LP_SOLVER").unwrap_or_else(|_| "scipy:highs".to_string());

    let prob = factory();
    let params_base = SimParams {
        proc_cv,
        break_prob_per_min: break_prob,
        break_duration_min: break_duration,
        total_min,
        seed: 1000,
    };

    let lp = build_factory_lp(&prob, robust_factor);
    println!("# Factory scheduling LP + DES bridge");
    println!(
        "# solver={solver_label}  robustFactor={robust_factor}  procCV={proc_cv}  breakProb={break_prob}/min"
    );
    println!();

    let result = solve_lp_then_simulate(
        &lp,
        |plan: &LPSolution| run_reps(&prob, plan, n_reps, params_base),
        &LpSolverOptions::default(),
    )
    .expect("LP solved to optimality");

    let plan = &result.plan;
    let reps = &result.realised;
    println!(
        "# LP solver:    {}    iters={}    elapsed={}ms",
        plan.solver,
        plan.iters
            .map(|i| i.to_string())
            .unwrap_or_else(|| "-".into()),
        plan.elapsed_ms
    );
    println!(
        "# LP plan x = [ {} ]",
        plan.x
            .iter()
            .map(|v| format!("{v:.2}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("# LP NOMINAL revenue = ${:.2}", plan.objective);
    if let Some(dual) = &plan.dual_ub {
        if !dual.is_empty() {
            println!("# Shadow prices on capacity (machine $/min):");
            for mi in 0..prob.machines.len() {
                println!(
                    "#   {:<8} : ${:.4}/min",
                    prob.machines[mi],
                    dual.get(mi).copied().unwrap_or(0.0)
                );
            }
        }
    }
    if let Some(rc) = &plan.reduced_costs {
        if !rc.is_empty() {
            println!(
                "# Reduced costs (binding ⇒ x = 0): {}",
                rc.iter()
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    println!();

    let revenues: Vec<f64> = reps.iter().map(|r| r.realised_revenue).collect();
    let mean_rev = revenues.iter().sum::<f64>() / revenues.len() as f64;
    let sd_rev = (revenues.iter().map(|v| (v - mean_rev).powi(2)).sum::<f64>()
        / (revenues.len() as f64 - 1.0).max(1.0))
    .sqrt();
    let gap = (plan.objective - mean_rev) / plan.objective * 100.0;

    println!("# DES realised over {n_reps} reps (1-week sim @ {total_min} min):");
    println!("#   mean revenue = ${mean_rev:.2}    sd = ${sd_rev:.2}");
    let p = prob.products.len();
    let m = prob.machines.len();
    let tput: Vec<String> = (0..p)
        .map(|pi| {
            format!(
                "{:.1}",
                reps.iter().map(|r| r.realised_throughput[pi]).sum::<f64>() / reps.len() as f64
            )
        })
        .collect();
    println!("#   throughput   = [ {} ]", tput.join(", "));
    let util: Vec<String> = (0..m)
        .map(|mi| {
            format!(
                "{:.1}%",
                100.0 * reps.iter().map(|r| r.utilisation[mi]).sum::<f64>() / reps.len() as f64
            )
        })
        .collect();
    println!("#   utilisation  = [ {} ]", util.join(", "));
    let brk: Vec<String> = (0..m)
        .map(|mi| {
            format!(
                "{:.2}",
                reps.iter().map(|r| r.breakdowns[mi]).sum::<f64>() / reps.len() as f64
            )
        })
        .collect();
    println!("#   breakdowns   = [ {} ]", brk.join(", "));
    println!();
    println!(
        "# Plan-vs-realised gap: ${:.2} ({:.1}% of nominal)",
        plan.objective - mean_rev,
        gap
    );
    println!("#   ↑ this is the cost of believing a deterministic LP in a stochastic factory");
    println!();

    // Robustness sweep.
    if std::env::var("SWEEP").as_deref() == Ok("1") {
        println!("# === Robustness sweep: shrink LP RHS by various factors ===");
        println!("#   robust    LP nominal      mean realised      realised sd      net gain over plan-as-is");
        let baseline_rev = mean_rev;
        for rf in [1.00, 0.95, 0.90, 0.85, 0.80, 0.75] {
            let lp2 = build_factory_lp(&prob, rf);
            let sub = solve_lp_then_simulate(
                &lp2,
                |plan: &LPSolution| run_reps(&prob, plan, n_reps, params_base),
                &LpSolverOptions::default(),
            )
            .expect("LP solved");
            let rev: Vec<f64> = sub.realised.iter().map(|r| r.realised_revenue).collect();
            let mr = rev.iter().sum::<f64>() / rev.len() as f64;
            let sr = (rev.iter().map(|v| (v - mr).powi(2)).sum::<f64>()
                / (rev.len() as f64 - 1.0).max(1.0))
            .sqrt();
            let delta = mr - baseline_rev;
            println!(
                "#   {:.2}      ${:>9}      ${:>9}      ${:>7}      {}${:.2}",
                rf,
                format!("{:.2}", sub.plan.objective),
                format!("{mr:.2}"),
                format!("{sr:.2}"),
                if delta >= 0.0 { "+" } else { "" },
                delta
            );
        }
        println!();
        println!("#   Lower robust factor = lower nominal but possibly higher realised because");
        println!("#   the plan no longer overcommits machines that breakdown / vary.");
    }

    let _ = lp_to_string; // referenced for API parity; LP printing not used here.
}
