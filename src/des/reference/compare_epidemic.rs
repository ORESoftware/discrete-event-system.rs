//! Port of `src/des/reference/compare-epidemic.ts`.
//!
//! Side-by-side comparison of the framework run (no-FEL, station-driven) versus
//! the classical Future-Event-List reference. Reads both JSONL event logs and
//! reports metrics side-by-side with absolute and relative differences plus a
//! rough Poisson sqrt(N) Monte-Carlo tolerance band.
//!
//! The TypeScript file is an entry script (shebang + `main()` reading
//! `process.argv` + `run()` at EOF). Per the migration rules the logic lives in
//! [`run`], which takes the two log paths explicitly; no `fn main` is added.
//! Comparing two single runs is noisy — the tolerance band ("OK"/"WIDE") is a
//! sufficiency check for "did the framework get the model right?", not a formal
//! statistical test.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::observability::logger::{read_events, JsonValue};

fn fmt(n: f64, digits: usize) -> String {
    if n.is_finite() {
        format!("{n:.digits$}")
    } else {
        js_num(n)
    }
}

fn js_num(n: f64) -> String {
    if n.is_nan() {
        "NaN".to_string()
    } else if n == f64::INFINITY {
        "Infinity".to_string()
    } else if n == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        format!("{n}")
    }
}

fn event_kind(e: &JsonValue) -> &str {
    e.get("kind").and_then(|v| v.as_str()).unwrap_or("")
}

fn jstr<'a>(e: &'a JsonValue, key: &str) -> &'a str {
    e.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

struct RunSummary {
    label: String,
    events: Vec<JsonValue>,
    start: JsonValue,
    end: JsonValue,
    transitions: Vec<JsonValue>,
    ticks: Vec<JsonValue>,
    totals_by_destination: HashMap<String, u64>,
    splits_by_from: HashMap<String, HashMap<String, u64>>,
    time_avg_populations: HashMap<String, f64>,
    peak_populations: HashMap<String, f64>,
}

const COMPARTMENTS: [&str; 7] = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R"];

fn summarize(label: &str, file: &str) -> RunSummary {
    let events = read_events(file).unwrap_or_else(|e| panic!("{e}"));
    let start = events
        .iter()
        .find(|e| event_kind(e) == "sim_start")
        .cloned()
        .expect("no sim_start event found");
    let end = events
        .iter()
        .find(|e| event_kind(e) == "sim_end")
        .cloned()
        .expect("no sim_end event found");
    let transitions: Vec<JsonValue> = events
        .iter()
        .filter(|e| event_kind(e) == "transition")
        .cloned()
        .collect();
    let ticks: Vec<JsonValue> = events
        .iter()
        .filter(|e| event_kind(e) == "tick")
        .cloned()
        .collect();

    let mut totals_by_destination: HashMap<String, u64> = HashMap::new();
    let mut splits_by_from: HashMap<String, HashMap<String, u64>> = HashMap::new();
    for t in &transitions {
        let to = jstr(t, "to").to_string();
        let from = jstr(t, "from").to_string();
        *totals_by_destination.entry(to.clone()).or_insert(0) += 1;
        *splits_by_from
            .entry(from)
            .or_default()
            .entry(to)
            .or_insert(0) += 1;
    }

    let mut time_avg: HashMap<String, f64> = HashMap::new();
    let mut peak: HashMap<String, f64> = HashMap::new();
    for c in COMPARTMENTS {
        let mut sum = 0.0_f64;
        let mut pk = 0.0_f64;
        for tk in &ticks {
            let v = tk
                .get("populations")
                .and_then(|p| p.get(c))
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            sum += v;
            if v > pk {
                pk = v;
            }
        }
        time_avg.insert(
            c.to_string(),
            if !ticks.is_empty() {
                sum / ticks.len() as f64
            } else {
                0.0
            },
        );
        peak.insert(c.to_string(), pk);
    }

    RunSummary {
        label: label.to_string(),
        events,
        start,
        end,
        transitions,
        ticks,
        totals_by_destination,
        splits_by_from,
        time_avg_populations: time_avg,
        peak_populations: peak,
    }
}

fn compare_row(label: &str, a: f64, b: f64, digits: usize) {
    let diff = b - a;
    let rel = if a == 0.0 {
        if b == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        diff / a
    };
    let sigma = a.max(1.0).sqrt(); // Poisson tolerance for counts
    let within = diff.abs() <= 2.5 * sigma;
    let verdict = if within { "OK" } else { "WIDE" };
    println!(
        "  {:<38} framework={:>10}  fel={:>10}  diff={:>8}  rel={:>7}  {}",
        label,
        fmt(a, digits),
        fmt(b, digits),
        fmt(diff, digits),
        fmt(rel, 3),
        verdict
    );
}

fn split_total(splits: &HashMap<String, HashMap<String, u64>>, from: &str) -> u64 {
    splits.get(from).map(|r| r.values().sum()).unwrap_or(0)
}

fn split_count(splits: &HashMap<String, HashMap<String, u64>>, from: &str, to: &str) -> u64 {
    splits
        .get(from)
        .and_then(|r| r.get(to))
        .copied()
        .unwrap_or(0)
}

/// Default framework / FEL log paths (repository-root `out/`).
pub fn default_framework_log() -> std::path::PathBuf {
    std::path::Path::new("out").join("epidemic-events.jsonl")
}

pub fn default_fel_log() -> std::path::PathBuf {
    std::path::Path::new("out").join("epidemic-events-fel.jsonl")
}

/// Compare the framework log against the FEL-reference log and print the report.
pub fn run(fw_file: &str, fel_file: &str) {
    println!("================================================================");
    println!("framework (no-FEL) vs. classical FEL reference");
    println!("  framework log: {fw_file}");
    println!("  fel log:       {fel_file}");
    println!("================================================================");

    let fw = summarize("framework", fw_file);
    let fel = summarize("fel-ref", fel_file);

    // ------ totals ----------------------------------------------------------
    println!();
    println!("--- totals (count-based, Poisson tolerance ~2.5 sigma) ---");
    compare_row(
        "entities created (source -> S)",
        *fw.totals_by_destination.get("S").unwrap_or(&0) as f64,
        *fel.totals_by_destination.get("S").unwrap_or(&0) as f64,
        0,
    );
    compare_row(
        "S-visits (anything -> S)",
        fw.transitions
            .iter()
            .filter(|t| jstr(t, "to") == "S")
            .count() as f64,
        fel.transitions
            .iter()
            .filter(|t| jstr(t, "to") == "S")
            .count() as f64,
        0,
    );
    compare_row(
        "cumulative deaths (sink absorbs)",
        fw.end
            .pointer(&["totals", "absorbed"])
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        fel.end
            .pointer(&["totals", "absorbed"])
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        0,
    );
    compare_row(
        "total transitions logged",
        fw.transitions.len() as f64,
        fel.transitions.len() as f64,
        0,
    );

    // ------ branching probabilities -----------------------------------------
    println!();
    println!("--- empirical branching probabilities ---");
    let asym = fw
        .start
        .pointer(&["config", "probabilities", "asymptomaticShare"])
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let hosp = fw
        .start
        .pointer(&["config", "probabilities", "hospitalizationGivenSymptom"])
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let cfr = fw
        .start
        .pointer(&["config", "probabilities", "caseFatalityGivenHospital"])
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let expected_splits: Vec<(&str, Vec<(&str, f64)>)> = vec![
        ("I-P", vec![("I-A", asym), ("I-S", 1.0 - asym)]),
        ("I-S", vec![("R", 1.0 - hosp), ("I-H", hosp)]),
        ("I-H", vec![("R", 1.0 - cfr), ("D", cfr)]),
    ];
    for (from, exp) in &expected_splits {
        for (to, p) in exp {
            let fw_tot = split_total(&fw.splits_by_from, from);
            let fl_tot = split_total(&fel.splits_by_from, from);
            let fw_hat = if fw_tot > 0 {
                split_count(&fw.splits_by_from, from, to) as f64 / fw_tot as f64
            } else {
                0.0
            };
            let fl_hat = if fl_tot > 0 {
                split_count(&fel.splits_by_from, from, to) as f64 / fl_tot as f64
            } else {
                0.0
            };
            println!(
                "  {:<15}  expected={}  framework={} (n={:>4})  fel={} (n={:>4})",
                format!("{from} -> {to}"),
                fmt(*p, 3),
                fmt(fw_hat, 3),
                fw_tot,
                fmt(fl_hat, 3),
                fl_tot
            );
        }
    }

    // ------ time-averaged populations ---------------------------------------
    println!();
    println!("--- time-averaged compartment populations (over all ticks) ---");
    for c in COMPARTMENTS {
        compare_row(
            &format!("<{c}>(t)"),
            *fw.time_avg_populations.get(c).unwrap_or(&0.0),
            *fel.time_avg_populations.get(c).unwrap_or(&0.0),
            2,
        );
    }

    // ------ peak populations ------------------------------------------------
    println!();
    println!("--- peak compartment populations ---");
    for c in COMPARTMENTS {
        compare_row(
            &format!("max {c}(t)"),
            *fw.peak_populations.get(c).unwrap_or(&0.0),
            *fel.peak_populations.get(c).unwrap_or(&0.0),
            0,
        );
    }

    println!();
    println!("================================================================");
    println!("verdicts: \"OK\" = within Poisson 2.5 sigma; \"WIDE\" = larger gap");
    println!("(single-replicate comparison; for tighter bounds, run N reps)");
    println!("================================================================");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("des_compare_{}_{}", std::process::id(), name));
        p
    }

    fn minimal_log() -> &'static str {
        concat!(
            r#"{"kind":"sim_start","config":{"probabilities":{"asymptomaticShare":0.4,"hospitalizationGivenSymptom":0.2,"caseFatalityGivenHospital":0.12}}}"#,
            "\n",
            r#"{"kind":"tick","t":1,"populations":{"S":2,"E":1}}"#,
            "\n",
            r#"{"kind":"transition","t":1,"entity":"f0","from":"__source__","to":"S"}"#,
            "\n",
            r#"{"kind":"sim_end","t":1,"elapsedMs":1,"totals":{"created":1,"absorbed":0}}"#,
            "\n",
        )
    }

    #[test]
    fn summarize_extracts_metrics() {
        let path = temp_path("fw.jsonl");
        let p = path.to_str().unwrap();
        fs::write(p, minimal_log()).unwrap();
        let s = summarize("framework", p);
        assert_eq!(s.label, "framework");
        assert_eq!(s.transitions.len(), 1);
        assert_eq!(*s.totals_by_destination.get("S").unwrap_or(&0), 1);
        assert_eq!(s.events.len(), 4);
        // <S>(t) = 2 over a single tick.
        assert_eq!(*s.time_avg_populations.get("S").unwrap_or(&-1.0), 2.0);
        assert_eq!(*s.peak_populations.get("E").unwrap_or(&-1.0), 1.0);
        let _ = fs::remove_file(p);
    }

    #[test]
    fn run_compares_two_logs_without_panicking() {
        let fw = temp_path("run_fw.jsonl");
        let fel = temp_path("run_fel.jsonl");
        let fwp = fw.to_str().unwrap();
        let felp = fel.to_str().unwrap();
        fs::write(fwp, minimal_log()).unwrap();
        fs::write(felp, minimal_log()).unwrap();
        run(fwp, felp);
        let _ = fs::remove_file(fwp);
        let _ = fs::remove_file(felp);
    }
}
