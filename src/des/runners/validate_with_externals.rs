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

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::des::observability::logger::{parse_json, JsonValue};
use crate::des::runners::fel_runner::run_fel_once;
use crate::des::runners::gillespie_runner::run_gillespie_once;
use crate::des::runners::ode_runner::run_ode_once;
use crate::des::runners::per_individual_runner::run_per_individual_once;
use crate::des::runners::stats::{mean, stddev, welch};
use crate::des::runners::types::{
    default_config, Kernel, Probabilities, RunOpts, RunResult, ServiceDiscipline, SimConfig,
    Totals, COMPARTMENT_ORDER,
};

fn get_any<'a>(value: &'a JsonValue, names: &[&str]) -> Option<&'a JsonValue> {
    names.iter().find_map(|name| value.get(name))
}

fn number_field(value: &JsonValue, names: &[&str]) -> Option<f64> {
    get_any(value, names).and_then(|v| v.as_f64())
}

fn string_field<'a>(value: &'a JsonValue, names: &[&str]) -> Option<&'a str> {
    get_any(value, names).and_then(|v| v.as_str())
}

fn number_pair(value: &JsonValue) -> Option<(f64, f64)> {
    let items = value.as_array()?;
    let a = items.first()?.as_f64()?;
    let b = items.get(1)?.as_f64()?;
    Some((a, b))
}

fn number_map(value: &JsonValue) -> Option<HashMap<String, f64>> {
    let entries = value.as_object()?;
    Some(
        entries
            .iter()
            .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
            .collect(),
    )
}

fn nested_number_map(value: &JsonValue) -> Option<HashMap<String, HashMap<String, f64>>> {
    let entries = value.as_object()?;
    Some(
        entries
            .iter()
            .filter_map(|(k, v)| number_map(v).map(|row| (k.clone(), row)))
            .collect(),
    )
}

fn parse_kernel(value: &JsonValue) -> Kernel {
    match string_field(value, &["kernel"]).unwrap_or_default() {
        "framework" => Kernel::Framework,
        "fel" => Kernel::Fel,
        "per-individual" | "perIndividual" => Kernel::PerIndividual,
        "gillespie" => Kernel::Gillespie,
        "ode" => Kernel::Ode,
        _ => Kernel::Difference,
    }
}

fn parse_probabilities(value: &JsonValue, defaults: Probabilities) -> Probabilities {
    Probabilities {
        asymptomatic_share: number_field(value, &["asymptomaticShare", "asymptomatic_share"])
            .unwrap_or(defaults.asymptomatic_share),
        hospitalization_given_symptom: number_field(
            value,
            &[
                "hospitalizationGivenSymptom",
                "hospitalization_given_symptom",
            ],
        )
        .unwrap_or(defaults.hospitalization_given_symptom),
        case_fatality_given_hospital: number_field(
            value,
            &["caseFatalityGivenHospital", "case_fatality_given_hospital"],
        )
        .unwrap_or(defaults.case_fatality_given_hospital),
    }
}

fn parse_config(value: Option<&JsonValue>) -> SimConfig {
    let mut cfg = default_config();
    let Some(value) = value else {
        return cfg;
    };
    cfg.step_size = number_field(value, &["stepSize", "step_size"]).unwrap_or(cfg.step_size);
    cfg.horizon_days =
        number_field(value, &["horizonDays", "horizon_days"]).unwrap_or(cfg.horizon_days);
    cfg.phase1_days =
        number_field(value, &["phase1Days", "phase1_days"]).unwrap_or(cfg.phase1_days);
    cfg.source_cap = number_field(value, &["sourceCap", "source_cap"]).unwrap_or(cfg.source_cap);
    if let Some(pair) =
        get_any(value, &["arrivalsInterarrival", "arrivals_interarrival"]).and_then(number_pair)
    {
        cfg.arrivals_interarrival = pair;
    }
    if let Some(probabilities) = get_any(value, &["probabilities"]) {
        cfg.probabilities = parse_probabilities(probabilities, cfg.probabilities);
    }
    if let Some(entries) = get_any(value, &["residence"]).and_then(|v| v.as_object()) {
        let mut residence = HashMap::new();
        for (key, raw) in entries {
            if let Some(pair) = number_pair(raw) {
                residence.insert(key.clone(), pair);
            }
        }
        if !residence.is_empty() {
            cfg.residence = residence;
        }
    }
    cfg
}

fn parse_run_result(value: &JsonValue) -> Option<RunResult> {
    let totals = get_any(value, &["totals"])?;
    let split_probs = get_any(value, &["splitProbs", "split_probs"]).and_then(nested_number_map)?;
    let time_avg_populations =
        get_any(value, &["timeAvgPopulations", "time_avg_populations"]).and_then(number_map)?;
    if split_probs.is_empty() || time_avg_populations.is_empty() {
        return None;
    }
    let created = number_field(totals, &["created"])?;
    let absorbed = number_field(totals, &["absorbed"])?;
    let seed = number_field(value, &["seed"]).unwrap_or(0.0);
    let elapsed_ms = number_field(value, &["elapsedMs", "elapsed_ms"]).unwrap_or(0.0);
    Some(RunResult {
        kernel: parse_kernel(value),
        config: parse_config(get_any(value, &["config"])),
        seed: seed.max(0.0).round() as u64,
        totals: Totals { created, absorbed },
        final_populations: get_any(value, &["finalPopulations", "final_populations"])
            .and_then(number_map)
            .unwrap_or_default(),
        transition_counts: get_any(value, &["transitionCounts", "transition_counts"])
            .and_then(nested_number_map)
            .unwrap_or_default(),
        split_probs,
        time_avg_populations,
        peak_populations: get_any(value, &["peakPopulations", "peak_populations"])
            .and_then(number_map)
            .unwrap_or_default(),
        elapsed_ms: elapsed_ms.max(0.0).round() as u128,
    })
}

fn append_run_results(value: &JsonValue, out: &mut Vec<RunResult>) {
    if let Some(run) = parse_run_result(value) {
        out.push(run);
        return;
    }
    if let Some(items) = value.as_array() {
        for item in items {
            append_run_results(item, out);
        }
        return;
    }
    for key in ["runs", "results"] {
        if let Some(items) = value.get(key).and_then(|v| v.as_array()) {
            for item in items {
                append_run_results(item, out);
            }
        }
    }
    if let Some(result) = value.get("result") {
        append_run_results(result, out);
    }
}

fn collect_json_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, files);
        } else if path.extension().and_then(|s| s.to_str()) == Some("json") {
            files.push(path);
        }
    }
}

fn load_external(tool_dir: &PathBuf) -> Vec<RunResult> {
    let mut files = Vec::new();
    collect_json_files(tool_dir, &mut files);
    files.sort();

    let mut runs = Vec::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        match parse_json(&text) {
            Ok(value) => append_run_results(&value, &mut runs),
            Err(err) => eprintln!(
                "[validate-with-externals] ignoring malformed JSON {}: {}",
                path.display(),
                err
            ),
        }
    }
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
            "NOTE: no SEIR-shaped external JSON runs found under {}",
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
