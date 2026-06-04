//! Port of `src/des/runners/validate-with-externals.ts`.
//!
//! Side-by-side comparison of every in-repo SEIR kernel (PerIndividual, FEL,
//! Gillespie, ODE) plus external JSON drops, with pairwise Welch t-tests against
//! FEL-individual. This driver only reads JSON files; it never invokes an
//! interpreter. Driver -> [`run`].
//!
//! The early Rust runner used zero-output local mirrors and skipped external
//! JSON parsing. The shared runner modules and serde boundary are now available,
//! so the in-repo columns use the real kernels and external directories are
//! scanned for SEIR-shaped JSON payloads.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use super::fel_runner::run_fel_once;
use super::gillespie_runner::run_gillespie_once;
use super::ode_runner::run_ode_once;
use super::per_individual_runner::run_per_individual_once;
use super::stats::{mean, stddev, welch};
use super::types::{
    default_config, Kernel, RunOpts, RunResult, ServiceDiscipline, Totals, COMPARTMENT_ORDER,
};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct ExternalTotals {
    created: f64,
    absorbed: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct ExternalRunResult {
    kernel: Option<String>,
    seed: Option<u64>,
    totals: ExternalTotals,
    #[serde(alias = "final_populations")]
    final_populations: HashMap<String, f64>,
    #[serde(alias = "transition_counts")]
    transition_counts: HashMap<String, HashMap<String, f64>>,
    #[serde(alias = "split_probs")]
    split_probs: HashMap<String, HashMap<String, f64>>,
    #[serde(alias = "time_avg_populations")]
    time_avg_populations: HashMap<String, f64>,
    #[serde(alias = "peak_populations")]
    peak_populations: HashMap<String, f64>,
    #[serde(alias = "elapsed_ms")]
    elapsed_ms: Option<f64>,
}

fn kernel_from_external(value: Option<&str>) -> Kernel {
    match value.unwrap_or("").to_ascii_lowercase().as_str() {
        "fel" | "fel-individual" => Kernel::Fel,
        "per-individual" | "perindividual" => Kernel::PerIndividual,
        "gillespie" | "gillespie-ssa" | "ssa" => Kernel::Gillespie,
        "ode" | "ode-rk4" => Kernel::Ode,
        "difference" => Kernel::Difference,
        _ => Kernel::Framework,
    }
}

fn has_seir_shape(run: &ExternalRunResult) -> bool {
    COMPARTMENT_ORDER.iter().any(|c| {
        run.time_avg_populations.contains_key(*c) || run.final_populations.contains_key(*c)
    }) || ["I-P", "I-S", "I-H"]
        .iter()
        .any(|from| run.split_probs.contains_key(*from))
}

fn external_run_to_result(run: ExternalRunResult) -> Option<RunResult> {
    if !has_seir_shape(&run) {
        return None;
    }

    let elapsed_ms = run
        .elapsed_ms
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(|v| v.round() as u128)
        .unwrap_or_default();

    Some(RunResult {
        kernel: kernel_from_external(run.kernel.as_deref()),
        config: default_config(),
        seed: run.seed.unwrap_or_default(),
        totals: Totals {
            created: run.totals.created,
            absorbed: run.totals.absorbed,
        },
        final_populations: run.final_populations,
        transition_counts: run.transition_counts,
        split_probs: run.split_probs,
        time_avg_populations: run.time_avg_populations,
        peak_populations: run.peak_populations,
        elapsed_ms,
    })
}

fn external_runs_from_value(value: Value) -> Vec<RunResult> {
    match value {
        Value::Array(items) => items
            .into_iter()
            .filter_map(|item| serde_json::from_value::<ExternalRunResult>(item).ok())
            .filter_map(external_run_to_result)
            .collect(),
        Value::Object(_) => serde_json::from_value::<ExternalRunResult>(value)
            .ok()
            .and_then(external_run_to_result)
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

fn load_external(tool_dir: &Path) -> Vec<RunResult> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(tool_dir) {
        Ok(entries) => entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect(),
        Err(_) => return Vec::new(),
    };
    files.sort();

    let mut runs = Vec::new();
    for path in files {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        runs.extend(external_runs_from_value(value));
    }
    runs.sort_by_key(|run| run.seed);
    runs
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

fn seeded_opts(seed: u64) -> RunOpts {
    RunOpts {
        seed: Some(seed),
        ..Default::default()
    }
}

fn fel_individual_opts(seed: u64) -> RunOpts {
    RunOpts {
        seed: Some(seed),
        service: Some(ServiceDiscipline::Individual),
        ..Default::default()
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
    let default_cfg = default_config();
    for i in 0..n {
        pi_runs.push(run_per_individual_once(
            &cfg,
            &seeded_opts(0xC0000 + i as u64),
        ));
        fel_runs.push(run_fel_once(
            &default_cfg,
            &fel_individual_opts(0xD0000 + i as u64),
        ));
        ssa_runs.push(run_gillespie_once(
            &default_cfg,
            &seeded_opts(0xE0000 + i as u64),
        ));
    }
    let ode = run_ode_once(&default_cfg, &RunOpts::default());
    let in_repo_ms = t0.elapsed().as_millis();

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
    external_dirs.sort();

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
                        &format!("{} +- {}", fmt(mean(&xs), 4), fmt(stddev(&xs), 4)),
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
                        &format!("{} +- {}", fmt(mean(&xs), 3), fmt(stddev(&xs), 3)),
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
                        &format!("{} +- {}", fmt(mean(&xs), 1), fmt(stddev(&xs), 1)),
                        col_width,
                    )
                }
                Column::Single(_, single) => pad_start(&fmt(extract(single), 1), col_width),
            })
            .collect();
        println!("{}{}", pad_end(label, 14), cells);
    }
}
