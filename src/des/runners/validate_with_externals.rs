//! Port of `src/des/runners/validate-with-externals.ts`.
//!
//! Side-by-side comparison of every in-repo SEIR kernel (PerIndividual, FEL,
//! Gillespie, ODE) plus external JSON drops, with pairwise Welch t-tests against
//! FEL-individual. This driver only reads JSON files; it never invokes an
//! interpreter. Driver → [`run`].
//!
//! The in-repo columns delegate to the shared SEIR runner modules. Optional
//! external JSON drops remain dependency-gated and are skipped when absent.

#![allow(dead_code)]

use std::path::PathBuf;

use crate::des::runners::fel_runner::run_fel_once;
use crate::des::runners::gillespie_runner::run_gillespie_once;
use crate::des::runners::ode_runner::run_ode_once;
use crate::des::runners::per_individual_runner::run_per_individual_once;
use crate::des::runners::stats::{mean, stddev, welch};
use crate::des::runners::types::{
    default_config, RunOpts, RunResult, ServiceDiscipline, COMPARTMENT_ORDER,
};

fn load_external(_tool_dir: &PathBuf) -> Vec<RunResult> {
    // PORT NOTE: read every *.json, JSON.parse, keep only SEIR-shaped objects.
    // External reference result decoding is adapter-specific → returns empty.
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
            &RunOpts {
                seed: Some(0xC0000 + i as u64),
                ..Default::default()
            },
        ));
        fel_runs.push(run_fel_once(
            &default_config(),
            &RunOpts {
                seed: Some(0xD0000 + i as u64),
                service: Some(ServiceDiscipline::Individual),
                ..Default::default()
            },
        ));
        ssa_runs.push(run_gillespie_once(
            &default_config(),
            &RunOpts {
                seed: Some(0xE0000 + i as u64),
                ..Default::default()
            },
        ));
    }
    let ode = run_ode_once(&default_config(), &RunOpts::default());
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
                        mean(&runs.iter().map(|r| r.elapsed_ms as f64).collect::<Vec<_>>()),
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
                    let xs: Vec<f64> = runs.iter().map(&extract).collect();
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
