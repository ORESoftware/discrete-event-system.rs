//! Port of `src/des/runners/per-individual-vs-fel.ts`.
//!
//! Verification driver: confirms the framework's `PerIndividualProcessor`
//! (single queue + per-entity exit clocks) converges to the classical FEL
//! reference, via Welch t-tests on every metric, plus a convergence sweep. The
//! TS top-level `main()` becomes [`run`].
//!
//! ## PORT NOTE
//!
//!   * `process.env.N` → `std::env::var`.
//!   * seeds `0x60000+i` / `0x70000+i` / `0x80000+i` / `0xA0000+i` /
//!     `0xB0000+i+round(ss*1000)` kept verbatim.
//!   * `console.log` → `println!`.

#![allow(dead_code)]

use std::time::Instant;

use super::fel_runner::run_fel_once;
use super::framework_runner::run_framework_once;
use super::per_individual_runner::run_per_individual_once;
use super::stats::{mean, stddev, welch};
use super::types::{default_config, RunOpts, RunResult, ServiceDiscipline, SimConfig, COMPARTMENT_ORDER};

fn fmt(n: f64, d: usize) -> String {
    if n.is_finite() {
        format!("{n:.d$}")
    } else {
        n.to_string()
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

fn collect_split(rs: &[RunResult], from: &str, to: &str) -> Vec<f64> {
    rs.iter()
        .map(|r| r.split_probs.get(from).and_then(|m| m.get(to)).copied().unwrap_or(0.0))
        .collect()
}

fn collect_pop(rs: &[RunResult], c: &str) -> Vec<f64> {
    rs.iter().map(|r| r.time_avg_populations.get(c).copied().unwrap_or(0.0)).collect()
}

struct Kernel<'a> {
    name: &'a str,
    runs: &'a [RunResult],
}

fn report_table(label: &str, kernels: &[Kernel], cfg: &SimConfig) {
    println!("\n=== {label} ===");
    let mut header = pad_end("metric", 18);
    for k in kernels {
        header += &pad_start(&format!("{}: mean ± sd", k.name), 28);
    }
    header += &format!(
        "   Welch t ({} vs {})   p (2s)   agree?",
        kernels[0].name, kernels[1].name
    );
    println!("{header}");

    let p = &cfg.probabilities;
    let splits: [(&str, &str); 6] = [
        ("I-P", "I-A"),
        ("I-P", "I-S"),
        ("I-S", "R"),
        ("I-S", "I-H"),
        ("I-H", "R"),
        ("I-H", "D"),
    ];
    let _ = p; // expected values are not printed in this table (parity with TS)
    for (from, to) in splits {
        let mut cells = String::new();
        for k in kernels {
            let xs = collect_split(k.runs, from, to);
            cells += &pad_start(&format!("{} ± {}", fmt(mean(&xs), 4), fmt(stddev(&xs), 4)), 28);
        }
        let w = welch(&collect_split(kernels[0].runs, from, to), &collect_split(kernels[1].runs, from, to));
        let verdict = if w.reject95 { "NO (95%)" } else { "yes" };
        println!(
            "{}{cells}      t={}   p={}   {verdict}",
            pad_end(&format!("{from} -> {to}"), 18),
            pad_start(&fmt(w.t, 2), 6),
            pad_start(&fmt(w.p_value_two_sided, 3), 6)
        );
    }

    for c in COMPARTMENT_ORDER {
        let mut cells = String::new();
        for k in kernels {
            let xs = collect_pop(k.runs, c);
            cells += &pad_start(&format!("{} ± {}", fmt(mean(&xs), 3), fmt(stddev(&xs), 3)), 28);
        }
        let w = welch(&collect_pop(kernels[0].runs, c), &collect_pop(kernels[1].runs, c));
        let verdict = if w.reject99 {
            "NO (99%)"
        } else if w.reject95 {
            "NO (95%)"
        } else {
            "yes"
        };
        println!(
            "{}{cells}      t={}   p={}   {verdict}",
            pad_end(&format!("<{c}>"), 18),
            pad_start(&fmt(w.t, 2), 6),
            pad_start(&fmt(w.p_value_two_sided, 3), 6)
        );
    }
}

/// `main()` — run the per-individual vs FEL convergence study.
pub fn run() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let pi_config = SimConfig { step_size: 0.1, ..default_config() };
    let default_cfg = default_config();

    println!(
        "per-individual-vs-fel: {n} reps each kernel; per-ind stepSize={}d",
        pi_config.step_size
    );

    let mut pi_runs: Vec<RunResult> = Vec::new();
    let mut fel_runs: Vec<RunResult> = Vec::new();
    let mut fw_runs: Vec<RunResult> = Vec::new();

    let t0 = Instant::now();
    for i in 0..n {
        pi_runs.push(run_per_individual_once(
            &pi_config,
            &RunOpts { seed: Some(0x60000 + i as u64), ..Default::default() },
        ));
        fel_runs.push(run_fel_once(
            &default_cfg,
            &RunOpts {
                seed: Some(0x70000 + i as u64),
                service: Some(ServiceDiscipline::Individual),
                ..Default::default()
            },
        ));
        fw_runs.push(run_framework_once(
            &pi_config,
            &RunOpts { seed: Some(0x80000 + i as u64), ..Default::default() },
        ));
    }
    let elapsed = t0.elapsed().as_millis();

    println!("total wall {elapsed} ms");
    let pi_walls: Vec<f64> = pi_runs.iter().map(|r| r.elapsed_ms as f64).collect();
    let fel_walls: Vec<f64> = fel_runs.iter().map(|r| r.elapsed_ms as f64).collect();
    let fw_walls: Vec<f64> = fw_runs.iter().map(|r| r.elapsed_ms as f64).collect();
    println!(
        "mean per-rep wall:  per-individual={} ms   fel={} ms   original framework (stepSize={})={} ms",
        fmt(mean(&pi_walls), 1),
        fmt(mean(&fel_walls), 1),
        pi_config.step_size,
        fmt(mean(&fw_walls), 1)
    );

    report_table(
        &format!(
            "per-individual processor (stepSize={}) VS classical FEL",
            pi_config.step_size
        ),
        &[
            Kernel { name: "per-individual", runs: &pi_runs },
            Kernel { name: "fel", runs: &fel_runs },
        ],
        &pi_config,
    );

    report_table(
        &format!(
            "per-individual processor VS three-queue framework (both stepSize={})",
            pi_config.step_size
        ),
        &[
            Kernel { name: "per-individual", runs: &pi_runs },
            Kernel { name: "three-queue", runs: &fw_runs },
        ],
        &pi_config,
    );

    println!();
    println!("=== summary ===");
    println!("Branching probabilities and slow-compartment populations (S, E, R) agree");
    println!("between the per-individual processor and the M/M/inf FEL kernel within");
    println!("Welch-t at 95%, confirming the new station type implements correct CTMC");
    println!("semantics. Fast-compartment means show a small residual fixed-step bias");
    println!("that decays with stepSize -> 0; a quick convergence sweep follows.");
    println!();

    // ---- PI -> FEL convergence demo --------------------------------------
    let step_sweep = [0.5_f64, 0.1, 0.05, 0.02];
    let n_conv = 5usize;
    println!("=== PI -> FEL convergence sweep (N={n_conv} reps each, M/M/inf FEL fixed) ===");
    let mut header = pad_end("compartment", 13) + &pad_start("fel mean", 12);
    for s in step_sweep {
        header += &pad_start(&format!("pi (ss={s}) ratio"), 20);
    }
    println!("{header}");
    let mut fel_conv_runs: Vec<RunResult> = Vec::new();
    for i in 0..n_conv {
        fel_conv_runs.push(run_fel_once(
            &default_cfg,
            &RunOpts {
                seed: Some(0xA0000 + i as u64),
                service: Some(ServiceDiscipline::Individual),
                ..Default::default()
            },
        ));
    }
    let pi_conv_runs: Vec<Vec<RunResult>> = step_sweep
        .iter()
        .map(|&ss| {
            let cfg = SimConfig { step_size: ss, ..default_cfg.clone() };
            let mut reps: Vec<RunResult> = Vec::new();
            for i in 0..n_conv {
                let seed = 0xB0000 + i as u64 + (ss * 1000.0).round() as u64;
                reps.push(run_per_individual_once(&cfg, &RunOpts { seed: Some(seed), ..Default::default() }));
            }
            reps
        })
        .collect();
    for c in COMPARTMENT_ORDER {
        let fel_mean = mean(&collect_pop(&fel_conv_runs, c));
        let mut line = pad_end(&format!("<{c}>"), 13) + &pad_start(&fmt(fel_mean, 3), 12);
        for rs in &pi_conv_runs {
            let ratio = mean(&collect_pop(rs, c)) / fel_mean.max(1e-9);
            line += &pad_start(&fmt(ratio, 3), 20);
        }
        println!("{line}");
    }
}
