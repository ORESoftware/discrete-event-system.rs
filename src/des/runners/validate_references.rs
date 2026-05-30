//! Port of `src/des/runners/validate-references.ts`.
//!
//! Compares four independent reference kernels (FEL-individual, Gillespie SSA,
//! ODE-RK4, PerIndividual) on the SEIR-with-hospitalization model via Welch
//! t-tests on time-averaged populations plus one deterministic ODE run.
//! Top-level `main()` → [`run`].
//!
//! PORT NOTES — wire to the already-ported sibling runner modules (present in
//! this same `runners/` directory; need `mod.rs` wiring which is out of scope):
//!   * `super::types::{default_config, RunResult, COMPARTMENT_ORDER}`.
//!   * `super::fel_runner::run_fel_once`, `super::per_individual_runner::run_per_individual_once`,
//!     `super::gillespie_runner::run_gillespie_once`, `super::ode_runner::run_ode_once`.
//!   * `super::stats::{mean, stddev, welch}`.
//! `mean`/`stddev`/`welch`/`erf` are reproduced faithfully here; the four kernels
//! are stubbed with matching signatures (they return empty maps → printed zeros).

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::collections::HashMap;
use std::time::Instant;

// =============================================================================
// Stats helpers (faithful copies of `stats.rs`).
// =============================================================================

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn sample_variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
}

fn stddev(xs: &[f64]) -> f64 {
    sample_variance(xs).sqrt()
}

#[derive(Clone, Copy, Debug)]
struct WelchResult {
    t: f64,
    p_value_two_sided: f64,
    reject95: bool,
    reject99: bool,
}

fn welch(a: &[f64], b: &[f64]) -> WelchResult {
    let m_a = mean(a);
    let m_b = mean(b);
    let v_a = sample_variance(a);
    let v_b = sample_variance(b);
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    let se_sq = v_a / n_a + v_b / n_b;
    let t = if se_sq > 0.0 { (m_a - m_b) / se_sq.sqrt() } else { 0.0 };
    let p = if se_sq > 0.0 { 2.0 * (1.0 - normal_cdf(t.abs())) } else { 1.0 };
    WelchResult { t, p_value_two_sided: p, reject95: t.abs() > 1.96, reject99: t.abs() > 2.58 }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let (a1, a2, a3, a4, a5, p) = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911);
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

// =============================================================================
// Config + RunResult (faithful subset of `types.rs`).
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct Probabilities {
    asymptomatic_share: f64,
    hospitalization_given_symptom: f64,
    case_fatality_given_hospital: f64,
}

#[derive(Clone, Debug)]
struct SimConfig {
    step_size: f64,
    horizon_days: f64,
    phase1_days: f64,
    source_cap: f64,
    probabilities: Probabilities,
}

fn default_config() -> SimConfig {
    SimConfig {
        step_size: 1.0,
        horizon_days: 1200.0,
        phase1_days: 800.0,
        source_cap: 500.0,
        probabilities: Probabilities {
            asymptomatic_share: 0.40,
            hospitalization_given_symptom: 0.20,
            case_fatality_given_hospital: 0.12,
        },
    }
}

const COMPARTMENT_ORDER: [&str; 7] = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R"];

#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    created: f64,
    absorbed: f64,
}

#[derive(Clone, Debug, Default)]
struct RunResult {
    totals: Totals,
    split_probs: HashMap<String, HashMap<String, f64>>,
    time_avg_populations: HashMap<String, f64>,
    elapsed_ms: u128,
}

#[derive(Clone, Copy, Debug, Default)]
struct RunOpts {
    seed: u64,
    service_individual: bool,
}

fn run_per_individual_once(_cfg: &SimConfig, _opts: RunOpts) -> RunResult {
    RunResult::default()
}
fn run_fel_once(_cfg: &SimConfig, _opts: RunOpts) -> RunResult {
    RunResult::default()
}
fn run_gillespie_once(_cfg: &SimConfig, _opts: RunOpts) -> RunResult {
    RunResult::default()
}
fn run_ode_once(_cfg: &SimConfig) -> RunResult {
    RunResult::default()
}

// =============================================================================
// Formatting helpers.
// =============================================================================

fn fmt(n: f64, d: usize) -> String {
    if n.is_finite() {
        format!("{:.*}", d, n)
    } else {
        format!("{}", n)
    }
}

fn pad_start(s: &str, w: usize) -> String {
    if s.chars().count() >= w {
        s.to_string()
    } else {
        format!("{}{}", " ".repeat(w - s.chars().count()), s)
    }
}

fn pad_end(s: &str, w: usize) -> String {
    if s.chars().count() >= w {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(w - s.chars().count()))
    }
}

fn collect_split(rs: &[RunResult], from: &str, to: &str) -> Vec<f64> {
    rs.iter()
        .map(|r| r.split_probs.get(from).and_then(|m| m.get(to)).copied().unwrap_or(0.0))
        .collect()
}

fn collect_pop(rs: &[RunResult], c: &str) -> Vec<f64> {
    rs.iter().map(|r| r.time_avg_populations.get(c).copied().unwrap_or(0.0)).collect()
}

fn kernel_stats<F: Fn(&RunResult) -> f64>(rs: &[RunResult], extractor: F) -> String {
    let xs: Vec<f64> = rs.iter().map(|r| extractor(r)).collect();
    pad_start(&format!("{} ± {}", fmt(mean(&xs), 4), fmt(stddev(&xs), 4)), 20)
}

/// `validate-references.ts` `main()`.
pub fn run() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let pi_stepsize: f64 = std::env::var("STEPSIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(0.05);

    let mut cfg = default_config();
    cfg.step_size = pi_stepsize;
    println!("validate-references.ts: N={} reps per stochastic kernel; PI stepSize={}d", n, pi_stepsize);
    println!("  horizon={}d   phase1={}d   sourceCap={}", cfg.horizon_days, cfg.phase1_days, cfg.source_cap);
    println!();

    let mut pi_runs: Vec<RunResult> = Vec::new();
    let mut fel_runs: Vec<RunResult> = Vec::new();
    let mut ssa_runs: Vec<RunResult> = Vec::new();

    let t0 = Instant::now();
    let default_cfg = default_config();
    for i in 0..n {
        pi_runs.push(run_per_individual_once(&cfg, RunOpts { seed: 0xC0000 + i as u64, service_individual: false }));
        fel_runs.push(run_fel_once(&default_cfg, RunOpts { seed: 0xD0000 + i as u64, service_individual: true }));
        ssa_runs.push(run_gillespie_once(&default_cfg, RunOpts { seed: 0xE0000 + i as u64, service_individual: false }));
    }
    let ode = run_ode_once(&default_cfg);
    let elapsed = t0.elapsed().as_millis();

    println!("total wall: {} ms", elapsed);
    println!("  per-individual mean wall:  {} ms / rep", fmt(mean(&pi_runs.iter().map(|r| r.elapsed_ms as f64).collect::<Vec<_>>()), 1));
    println!("  fel-individual mean wall:  {} ms / rep", fmt(mean(&fel_runs.iter().map(|r| r.elapsed_ms as f64).collect::<Vec<_>>()), 1));
    println!("  gillespie SSA  mean wall:  {} ms / rep", fmt(mean(&ssa_runs.iter().map(|r| r.elapsed_ms as f64).collect::<Vec<_>>()), 1));
    println!("  ODE RK4  wall:             {} ms (deterministic)", ode.elapsed_ms);

    println!();
    println!("=== empirical branching probabilities ===");
    let header_cols = ["expected", "PerIndividual", "FEL-individual", "Gillespie SSA", "ODE"];
    println!("              {}", header_cols.iter().map(|s| pad_start(s, 20)).collect::<Vec<_>>().join(""));
    let splits: [(&str, &str, f64); 6] = [
        ("I-P", "I-A", cfg.probabilities.asymptomatic_share),
        ("I-P", "I-S", 1.0 - cfg.probabilities.asymptomatic_share),
        ("I-S", "R", 1.0 - cfg.probabilities.hospitalization_given_symptom),
        ("I-S", "I-H", cfg.probabilities.hospitalization_given_symptom),
        ("I-H", "R", 1.0 - cfg.probabilities.case_fatality_given_hospital),
        ("I-H", "D", cfg.probabilities.case_fatality_given_hospital),
    ];
    for (from, to, expected) in splits {
        let fel = collect_split(&fel_runs, from, to);
        let pi = collect_split(&pi_runs, from, to);
        let ssa = collect_split(&ssa_runs, from, to);
        let ode_val = ode.split_probs.get(from).and_then(|m| m.get(to)).copied().unwrap_or(0.0);
        println!(
            "{}{}{}{}{}{}",
            pad_end(&format!("{} -> {}", from, to), 14),
            pad_start(&fmt(expected, 4), 20),
            pad_start(&format!("{} ± {}", fmt(mean(&pi), 4), fmt(stddev(&pi), 4)), 20),
            pad_start(&format!("{} ± {}", fmt(mean(&fel), 4), fmt(stddev(&fel), 4)), 20),
            pad_start(&format!("{} ± {}", fmt(mean(&ssa), 4), fmt(stddev(&ssa), 4)), 20),
            pad_start(&fmt(ode_val, 4), 20),
        );
    }

    println!();
    println!("=== time-averaged compartment populations ===");
    let pop_cols = ["PerIndividual", "FEL-individual", "Gillespie SSA", "ODE"];
    println!("              {}", pop_cols.iter().map(|s| pad_start(s, 20)).collect::<Vec<_>>().join(""));
    for c in COMPARTMENT_ORDER {
        let pi = collect_pop(&pi_runs, c);
        let fel = collect_pop(&fel_runs, c);
        let ssa = collect_pop(&ssa_runs, c);
        println!(
            "{}{}{}{}{}",
            pad_end(&format!("<{}>", c), 14),
            pad_start(&format!("{} ± {}", fmt(mean(&pi), 3), fmt(stddev(&pi), 3)), 20),
            pad_start(&format!("{} ± {}", fmt(mean(&fel), 3), fmt(stddev(&fel), 3)), 20),
            pad_start(&format!("{} ± {}", fmt(mean(&ssa), 3), fmt(stddev(&ssa), 3)), 20),
            pad_start(&fmt(ode.time_avg_populations.get(c).copied().unwrap_or(f64::NAN), 3), 20),
        );
    }

    println!();
    println!("=== pairwise Welch t-tests on time-averaged populations ===");
    let pairs: [(&str, &Vec<RunResult>, &Vec<RunResult>); 3] = [
        ("PI vs FEL-ind ", &pi_runs, &fel_runs),
        ("PI vs Gillesp ", &pi_runs, &ssa_runs),
        ("FEL vs Gilles ", &fel_runs, &ssa_runs),
    ];
    println!(
        "compartment    {}",
        pairs.iter().map(|p| pad_start(&format!("{}  t (p)", p.0), 30)).collect::<Vec<_>>().join("")
    );
    for c in COMPARTMENT_ORDER {
        let cells: Vec<String> = pairs
            .iter()
            .map(|(_, a, b)| {
                let w = welch(&collect_pop(a, c), &collect_pop(b, c));
                let verdict = if w.reject99 { "  NO99 " } else if w.reject95 { "  no95 " } else { "  yes  " };
                pad_start(&format!("{} (p={}) {}", pad_start(&fmt(w.t, 2), 7), fmt(w.p_value_two_sided, 3), verdict), 30)
            })
            .collect();
        println!("{}{}", pad_end(&format!("<{}>", c), 14), cells.join(""));
    }

    println!();
    println!("=== totals ===");
    println!(
        "{}{}{}{}{}",
        pad_end("created ", 14),
        kernel_stats(&pi_runs, |r| r.totals.created),
        kernel_stats(&fel_runs, |r| r.totals.created),
        kernel_stats(&ssa_runs, |r| r.totals.created),
        pad_start(&fmt(ode.totals.created, 1), 20),
    );
    println!(
        "{}{}{}{}{}",
        pad_end("absorbed (D)", 14),
        kernel_stats(&pi_runs, |r| r.totals.absorbed),
        kernel_stats(&fel_runs, |r| r.totals.absorbed),
        kernel_stats(&ssa_runs, |r| r.totals.absorbed),
        pad_start(&fmt(ode.totals.absorbed, 1), 20),
    );
}
