//! Port of `src/des/runners/validate-with-externals.ts`.
//!
//! Side-by-side comparison of every in-repo SEIR kernel (PerIndividual, FEL,
//! Gillespie, ODE) plus external JSON drops, with pairwise Welch t-tests against
//! FEL-individual. This driver only reads JSON files; it never invokes an
//! interpreter. Driver → [`run`].
//!
//! PORT NOTES — wire to real modules (all present in `src/des/runners/`):
//!   * `crate::des::runners::types::{DEFAULT_CONFIG, RunResult, COMPARTMENT_ORDER}`.
//!   * `crate::des::runners::{fel_runner::run_fel_once, per_individual_runner::run_per_individual_once,
//!     gillespie_runner::run_gillespie_once, ode_runner::run_ode_once}`.
//!   * `crate::des::runners::stats::{mean, stddev, welch}` (ported faithfully here).
//!   * Reading `out/external/<tool>/<seed>.json` needs `serde_json` (absent) →
//!     `load_external` returns `[]`, so the NOTE branch fires (mirrors the TS
//!     behaviour when no external JSONs are present).

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::collections::HashMap;
use std::path::PathBuf;

// =============================================================================
// Local mirrors of `types.ts` (subset used by this driver) + stats helpers.
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
    probabilities: Probabilities,
}

fn default_config() -> SimConfig {
    SimConfig {
        step_size: 1.0,
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
    kernel: String,
    seed: u64,
    totals: Totals,
    final_populations: HashMap<String, f64>,
    transition_counts: HashMap<String, HashMap<String, f64>>,
    split_probs: HashMap<String, HashMap<String, f64>>,
    time_avg_populations: HashMap<String, f64>,
    peak_populations: HashMap<String, f64>,
    elapsed_ms: f64,
}

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

fn erf(x: f64) -> f64 {
    let (a1, a2, a3, a4, a5, p) = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
        0.3275911,
    );
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}
fn welch(a: &[f64], b: &[f64]) -> WelchResult {
    let (m_a, m_b) = (mean(a), mean(b));
    let (v_a, v_b) = (sample_variance(a), sample_variance(b));
    let se_sq = v_a / a.len() as f64 + v_b / b.len() as f64;
    let t = if se_sq > 0.0 {
        (m_a - m_b) / se_sq.sqrt()
    } else {
        0.0
    };
    let p = if se_sq > 0.0 {
        2.0 * (1.0 - normal_cdf(t.abs()))
    } else {
        1.0
    };
    WelchResult {
        t,
        p_value_two_sided: p,
        reject95: t.abs() > 1.96,
        reject99: t.abs() > 2.58,
    }
}

// =============================================================================
// Stubbed kernels + external loader.
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
struct RunOpts {
    seed: u64,
    service_individual: bool,
}

fn run_per_individual_once(_cfg: &SimConfig, opts: RunOpts) -> RunResult {
    RunResult {
        kernel: "per-individual".to_string(),
        seed: opts.seed,
        ..Default::default()
    }
}
fn run_fel_once(_cfg: &SimConfig, opts: RunOpts) -> RunResult {
    RunResult {
        kernel: "fel".to_string(),
        seed: opts.seed,
        ..Default::default()
    }
}
fn run_gillespie_once(_cfg: &SimConfig, opts: RunOpts) -> RunResult {
    RunResult {
        kernel: "gillespie".to_string(),
        seed: opts.seed,
        ..Default::default()
    }
}
fn run_ode_once(_cfg: &SimConfig) -> RunResult {
    RunResult {
        kernel: "ode".to_string(),
        ..Default::default()
    }
}

fn load_external(_tool_dir: &PathBuf) -> Vec<RunResult> {
    // PORT NOTE: read every *.json, JSON.parse, keep only SEIR-shaped objects.
    // Needs serde_json (absent) → returns empty.
    Vec::new()
}

fn collect_split(rs: &[RunResult], from: &str, to: &str) -> Vec<f64> {
    rs.iter()
        .map(|r| {
            r.split_probs
                .get(from)
                .and_then(|m| m.get(to))
                .copied()
                .unwrap_or(0.0)
        })
        .collect()
}
fn collect_pop(rs: &[RunResult], c: &str) -> Vec<f64> {
    rs.iter()
        .map(|r| r.time_avg_populations.get(c).copied().unwrap_or(0.0))
        .collect()
}

fn fmt(n: f64, d: usize) -> String {
    if n.is_finite() {
        format!("{:.*}", d, n)
    } else {
        format!("{}", n)
    }
}
fn pad_end(s: &str, w: usize) -> String {
    if s.len() >= w {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(w - s.len()))
    }
}
fn pad_start(s: &str, w: usize) -> String {
    if s.len() >= w {
        s.to_string()
    } else {
        format!("{}{}", " ".repeat(w - s.len()), s)
    }
}

enum Column {
    Runs(String, Vec<RunResult>),
    Single(String, RunResult),
}

impl Column {
    fn name(&self) -> &str {
        match self {
            Column::Runs(n, _) => n,
            Column::Single(n, _) => n,
        }
    }
}

/// `validate-with-externals.ts` `main`.
pub fn run() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let pi_stepsize: f64 = std::env::var("STEPSIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05);
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let external_dir = root.join("out").join("external");

    let mut cfg = default_config();
    cfg.step_size = pi_stepsize;
    println!(
        "validate-with-externals: PI stepSize={}d   N={} reps (in-repo) | external runs read from {}",
        pi_stepsize,
        n,
        external_dir.display()
    );
    println!();

    let mut pi_runs = Vec::new();
    let mut fel_runs = Vec::new();
    let mut ssa_runs = Vec::new();
    let t0 = std::time::Instant::now();
    for i in 0..n {
        pi_runs.push(run_per_individual_once(
            &cfg,
            RunOpts {
                seed: 0xC0000 + i as u64,
                service_individual: false,
            },
        ));
        fel_runs.push(run_fel_once(
            &default_config(),
            RunOpts {
                seed: 0xD0000 + i as u64,
                service_individual: true,
            },
        ));
        ssa_runs.push(run_gillespie_once(
            &default_config(),
            RunOpts {
                seed: 0xE0000 + i as u64,
                service_individual: false,
            },
        ));
    }
    let ode = run_ode_once(&default_config());
    let in_repo_ms = t0.elapsed().as_millis();

    // Discover external tool runs.
    let mut external_dirs: Vec<PathBuf> = Vec::new();
    if external_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&external_dir) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    external_dirs.push(e.path());
                }
            }
        }
    }
    let mut externals: Vec<(String, Vec<RunResult>)> = Vec::new();
    for tool in &external_dirs {
        let runs = load_external(tool);
        if !runs.is_empty() {
            externals.push((
                tool.file_name().unwrap().to_string_lossy().to_string(),
                runs,
            ));
        }
    }
    if externals.is_empty() {
        println!(
            "NOTE: no external JSONs found under {}",
            external_dir.display()
        );
        println!("      run `bash external-references/run-all.sh` first to populate them.");
        println!();
    }

    let mut columns: Vec<Column> = vec![
        Column::Runs("PerIndividual".to_string(), pi_runs.clone()),
        Column::Runs("FEL-individual".to_string(), fel_runs.clone()),
        Column::Runs("Gillespie SSA".to_string(), ssa_runs.clone()),
        Column::Single("ODE (det)".to_string(), ode.clone()),
    ];
    for (name, runs) in &externals {
        columns.push(Column::Runs(name.clone(), runs.clone()));
    }

    println!("in-repo wall: {} ms", in_repo_ms);
    for col in &columns {
        match col {
            Column::Runs(name, runs) => {
                println!(
                    "  {} N={}  mean wall={} ms / rep",
                    pad_end(name, 18),
                    pad_start(&runs.len().to_string(), 3),
                    fmt(
                        mean(&runs.iter().map(|r| r.elapsed_ms).collect::<Vec<_>>()),
                        1
                    )
                );
            }
            Column::Single(name, single) => {
                println!(
                    "  {} (deterministic)  wall={} ms",
                    pad_end(name, 18),
                    single.elapsed_ms
                );
            }
        }
    }
    println!();

    let splits: Vec<(&str, &str, f64)> = vec![
        ("I-P", "I-A", cfg.probabilities.asymptomatic_share),
        ("I-P", "I-S", 1.0 - cfg.probabilities.asymptomatic_share),
        (
            "I-S",
            "R",
            1.0 - cfg.probabilities.hospitalization_given_symptom,
        ),
        (
            "I-S",
            "I-H",
            cfg.probabilities.hospitalization_given_symptom,
        ),
        (
            "I-H",
            "R",
            1.0 - cfg.probabilities.case_fatality_given_hospital,
        ),
        ("I-H", "D", cfg.probabilities.case_fatality_given_hospital),
    ];

    let col_width = 22;
    println!("=== empirical branching probabilities ===");
    println!(
        "{}{}{}",
        pad_end("transition", 14),
        pad_start("expected", 10),
        columns
            .iter()
            .map(|c| pad_start(c.name(), col_width))
            .collect::<String>()
    );
    for (from, to, expected) in &splits {
        let cells: String = columns
            .iter()
            .map(|col| match col {
                Column::Runs(_, runs) => {
                    let xs = collect_split(runs, from, to);
                    pad_start(
                        &format!("{} ± {}", fmt(mean(&xs), 4), fmt(stddev(&xs), 4)),
                        col_width,
                    )
                }
                Column::Single(_, single) => {
                    let v = single
                        .split_probs
                        .get(*from)
                        .and_then(|m| m.get(*to))
                        .copied()
                        .unwrap_or(0.0);
                    pad_start(&fmt(v, 4), col_width)
                }
            })
            .collect();
        println!(
            "{}{}{}",
            pad_end(&format!("{} -> {}", from, to), 14),
            pad_start(&fmt(*expected, 4), 10),
            cells
        );
    }

    println!();
    println!("=== time-averaged compartment populations ===");
    println!(
        "{}{}",
        pad_end("compartment", 14),
        columns
            .iter()
            .map(|c| pad_start(c.name(), col_width))
            .collect::<String>()
    );
    for c in COMPARTMENT_ORDER {
        let cells: String = columns
            .iter()
            .map(|col| match col {
                Column::Runs(_, runs) => {
                    let xs = collect_pop(runs, c);
                    pad_start(
                        &format!("{} ± {}", fmt(mean(&xs), 3), fmt(stddev(&xs), 3)),
                        col_width,
                    )
                }
                Column::Single(_, single) => {
                    let v = single.time_avg_populations.get(c).copied().unwrap_or(0.0);
                    pad_start(&fmt(v, 3), col_width)
                }
            })
            .collect();
        println!("{}{}", pad_end(&format!("<{}>", c), 14), cells);
    }

    if !externals.is_empty() {
        println!();
        println!("=== pairwise Welch t-tests vs FEL-individual (populations) ===");
        let ref_runs = &fel_runs;
        let mut others: Vec<(String, &Vec<RunResult>)> = vec![
            ("PerIndividual ".to_string(), &pi_runs),
            ("Gillespie SSA ".to_string(), &ssa_runs),
        ];
        for (name, runs) in &externals {
            others.push((pad_end(name, 14), runs));
        }
        println!(
            "compartment    {}",
            others
                .iter()
                .map(|o| pad_start(&format!("{}  t (p)", o.0), 28))
                .collect::<String>()
        );
        for c in COMPARTMENT_ORDER {
            let cells: String = others
                .iter()
                .map(|(_, rs)| {
                    let w = welch(&collect_pop(rs, c), &collect_pop(ref_runs, c));
                    let verdict = if w.reject99 {
                        "NO99 "
                    } else if w.reject95 {
                        "no95 "
                    } else {
                        " yes "
                    };
                    pad_start(
                        &format!(
                            "{} (p={}) {}",
                            pad_start(&fmt(w.t, 2), 6),
                            fmt(w.p_value_two_sided, 3),
                            verdict
                        ),
                        28,
                    )
                })
                .collect();
            println!("{}{}", pad_end(&format!("<{}>", c), 14), cells);
        }
    }

    println!();
    println!("=== totals ===");
    println!(
        "{}{}",
        pad_end("metric", 14),
        columns
            .iter()
            .map(|c| pad_start(c.name(), col_width))
            .collect::<String>()
    );
    let extractors: [(&str, fn(&RunResult) -> f64); 2] = [
        ("created   ", |r: &RunResult| r.totals.created),
        ("absorbed D", |r: &RunResult| r.totals.absorbed),
    ];
    for (label, extract) in extractors {
        let cells: String = columns
            .iter()
            .map(|col| match col {
                Column::Runs(_, runs) => {
                    let xs: Vec<f64> = runs.iter().map(|r| extract(r)).collect();
                    pad_start(
                        &format!("{} ± {}", fmt(mean(&xs), 1), fmt(stddev(&xs), 1)),
                        col_width,
                    )
                }
                Column::Single(_, single) => pad_start(&fmt(extract(single), 1), col_width),
            })
            .collect();
        println!("{}{}", pad_end(label, 14), cells);
    }
}
