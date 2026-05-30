//! Port of `src/des/runners/replicate.ts`.
//!
//! Runs N independent replications of the `{framework, FEL}` kernels on the same
//! model and Welch-t-tests every empirical metric (split probabilities,
//! time-averaged populations, totals). The TS top-level `main()` driver becomes
//! [`run`].
//!
//! ## PORT NOTE
//!
//!   * `process.env.N` → `std::env::var("N")`.
//!   * fixed seeds `0x10000`/`0x20000` are kept verbatim.
//!   * `console.log` → `println!`.
//!   * `fs`/`path` + `JSON.stringify(.., null, 2)` → `std::fs` +
//!     [`crate::des::runners::shared::results_to_json`] /
//!     [`config_to_json`](crate::des::runners::shared::config_to_json) rendered
//!     with `JsonValue::to_string_pretty(2)`. The artifact is written to
//!     `./out/replicate-results.json` relative to the working directory (the TS
//!     resolves `../../../out` from `__dirname`, i.e. the repo root).

#![allow(dead_code)]

use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::des::observability::logger::JsonValue;

use super::fel_runner::run_fel_once;
use super::framework_runner::run_framework_once;
use super::shared::{config_to_json, results_to_json};
use super::stats::{mean, stddev, welch};
use super::types::{default_config, RunOpts, RunResult, COMPARTMENT_ORDER};

const BASE_SEED_FW: u64 = 0x10000;
const BASE_SEED_FEL: u64 = 0x20000;

fn fmt(n: f64, d: usize) -> String {
    if n.is_finite() {
        format!("{n:.d$}")
    } else {
        n.to_string()
    }
}

fn pad_end(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - s.len()))
    }
}

fn pad_start(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(width - s.len()))
    }
}

fn collect_split(results: &[RunResult], from: &str, to: &str) -> Vec<f64> {
    results
        .iter()
        .map(|r| {
            r.split_probs
                .get(from)
                .and_then(|m| m.get(to))
                .copied()
                .unwrap_or(0.0)
        })
        .collect()
}

fn collect_pop(results: &[RunResult], compartment: &str) -> Vec<f64> {
    results
        .iter()
        .map(|r| r.time_avg_populations.get(compartment).copied().unwrap_or(0.0))
        .collect()
}

/// `main()` — run the replication study and print the report.
pub fn run() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let config = default_config();
    println!("replicate.ts: {n} replications per kernel");
    println!(
        "  config:   stepSize={}d horizon={}d cap={}",
        config.step_size, config.horizon_days, config.source_cap
    );

    let mut fw_results: Vec<RunResult> = Vec::new();
    let mut fel_results: Vec<RunResult> = Vec::new();

    let t0 = Instant::now();
    for i in 0..n {
        fw_results.push(run_framework_once(
            &config,
            &RunOpts { seed: Some(BASE_SEED_FW + i as u64), ..Default::default() },
        ));
        fel_results.push(run_fel_once(
            &config,
            &RunOpts { seed: Some(BASE_SEED_FEL + i as u64), ..Default::default() },
        ));
    }
    let elapsed = t0.elapsed().as_millis();
    println!("  total:    {elapsed} ms");
    let fw_walls: Vec<f64> = fw_results.iter().map(|r| r.elapsed_ms as f64).collect();
    let fel_walls: Vec<f64> = fel_results.iter().map(|r| r.elapsed_ms as f64).collect();
    println!("  framework mean wall: {} ms", fmt(mean(&fw_walls), 1));
    println!("  fel       mean wall: {} ms", fmt(mean(&fel_walls), 1));
    println!();

    let p = &config.probabilities;
    let splits: [(&str, &str, f64); 6] = [
        ("I-P", "I-A", p.asymptomatic_share),
        ("I-P", "I-S", 1.0 - p.asymptomatic_share),
        ("I-S", "R", 1.0 - p.hospitalization_given_symptom),
        ("I-S", "I-H", p.hospitalization_given_symptom),
        ("I-H", "R", 1.0 - p.case_fatality_given_hospital),
        ("I-H", "D", p.case_fatality_given_hospital),
    ];

    println!("=== empirical branching probabilities (N={n} reps each) ===");
    println!(
        "{}{}{}{}{}{}{}",
        pad_end("transition", 15),
        pad_start("expected", 10),
        pad_start("  framework: mean ± sd", 28),
        pad_start("  fel: mean ± sd", 22),
        pad_start("  Welch t", 12),
        pad_start("  p (2-sided)", 15),
        pad_start("  agree?", 10),
    );
    for (from, to, expected) in splits {
        let fw = collect_split(&fw_results, from, to);
        let fl = collect_split(&fel_results, from, to);
        let w = welch(&fw, &fl);
        let verdict = if w.reject95 { "NO (95%)" } else { "yes" };
        println!(
            "{}{}      {} ± {}    {} ± {}   t={}   p={}      {}",
            pad_end(&format!("{from} -> {to}"), 15),
            pad_start(&fmt(expected, 4), 10),
            fmt(w.mean_a, 4),
            fmt(stddev(&fw), 4),
            fmt(w.mean_b, 4),
            fmt(stddev(&fl), 4),
            pad_start(&fmt(w.t, 2), 6),
            pad_start(&fmt(w.p_value_two_sided, 4), 8),
            verdict,
        );
    }

    println!();
    println!("=== time-averaged compartment populations (N={n} reps each) ===");
    println!(
        "{}{}{}{}{}{}",
        pad_end("compartment", 15),
        pad_start("  framework: mean ± sd", 28),
        pad_start("  fel: mean ± sd", 22),
        pad_start("  Welch t", 12),
        pad_start("  p (2-sided)", 15),
        pad_start("  agree?", 10),
    );
    for c in COMPARTMENT_ORDER {
        let fw = collect_pop(&fw_results, c);
        let fl = collect_pop(&fel_results, c);
        let w = welch(&fw, &fl);
        let verdict = if w.reject99 {
            "NO (99%)"
        } else if w.reject95 {
            "NO (95%)"
        } else {
            "yes"
        };
        println!(
            "{}      {} ± {}    {} ± {}   t={}   p={}      {}",
            pad_end(&format!("<{c}>"), 15),
            fmt(w.mean_a, 3),
            fmt(stddev(&fw), 3),
            fmt(w.mean_b, 3),
            fmt(stddev(&fl), 3),
            pad_start(&fmt(w.t, 2), 6),
            pad_start(&fmt(w.p_value_two_sided, 4), 8),
            verdict,
        );
    }

    println!();
    println!("=== totals (created, absorbed-deaths) ===");
    let extractors: [(&str, fn(&RunResult) -> f64); 2] = [
        ("created  ", |r: &RunResult| r.totals.created),
        ("absorbed ", |r: &RunResult| r.totals.absorbed),
    ];
    for (label, extract) in extractors {
        let fw: Vec<f64> = fw_results.iter().map(extract).collect();
        let fl: Vec<f64> = fel_results.iter().map(extract).collect();
        let w = welch(&fw, &fl);
        let verdict = if w.reject95 { "NO (95%)" } else { "yes" };
        println!(
            "{}   framework={} ± {}   fel={} ± {}   t={}   p={}   {}",
            label,
            fmt(w.mean_a, 1),
            fmt(stddev(&fw), 1),
            fmt(w.mean_b, 1),
            fmt(stddev(&fl), 1),
            fmt(w.t, 2),
            fmt(w.p_value_two_sided, 4),
            verdict,
        );
    }

    let out_dir = Path::new("out");
    if let Err(e) = fs::create_dir_all(out_dir) {
        eprintln!("[replicate] could not create out dir: {e}");
        return;
    }
    let out_path = out_dir.join("replicate-results.json");
    let payload = JsonValue::Object(vec![
        ("n".to_string(), JsonValue::Number(n as f64)),
        ("config".to_string(), config_to_json(&config)),
        ("framework".to_string(), results_to_json(&fw_results)),
        ("fel".to_string(), results_to_json(&fel_results)),
    ]);
    if let Err(e) = fs::write(&out_path, payload.to_string_pretty(2)) {
        eprintln!("[replicate] could not write artifact: {e}");
        return;
    }
    println!("\nartifacts written:\n  {}", out_path.display());
}
