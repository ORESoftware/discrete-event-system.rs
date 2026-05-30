//! Port of `src/des/runners/shared.ts`.
//!
//! Shared transition-counter + record/aggregation helpers for the kernel
//! runners.
//!
//! ## Rust shape
//!
//!   * `class TransitionCounter` → [`TransitionCounter`] struct + impl.
//!   * `interface TransitionTables` → [`TransitionTables`] struct.
//!   * `Map<string, Map<string, number>>` → `HashMap<String, HashMap<String, f64>>`
//!     (iteration order is N/A in Rust, which is fine: the tables are built by
//!     iterating the fixed `MATRIX_ROWS`/`MATRIX_COLS`).
//!   * `?? 0` map-lookup fallbacks → `.get(..).copied().unwrap_or(0.0)`.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::observability::logger::JsonValue;

use super::types::{
    compartment_groups, Kernel, Probabilities, RunResult, SimConfig, COMPARTMENT_ORDER,
};

pub const TRANSITION_MATRIX_ROWS: [&str; 9] =
    ["__source__", "S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D"];
pub const TRANSITION_MATRIX_COLS: [&str; 9] =
    ["S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D", "main-sink"];

/// `Map<string, Map<string, number>>`.
pub type TransitionCountMap = HashMap<String, HashMap<String, f64>>;

/// Built count + split tables (`interface TransitionTables`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TransitionTables {
    pub counts: HashMap<String, HashMap<String, f64>>,
    pub splits: HashMap<String, HashMap<String, f64>>,
}

/// Records `(from -> to)` transition counts.
#[derive(Clone, Debug, Default)]
pub struct TransitionCounter {
    counts_by_from: TransitionCountMap,
}

impl TransitionCounter {
    pub fn new() -> Self {
        TransitionCounter {
            counts_by_from: HashMap::new(),
        }
    }

    /// `record(from, to)` — increment the `(from -> to)` cell by one.
    pub fn record(&mut self, from: &str, to: &str) {
        let row = self.counts_by_from.entry(from.to_string()).or_default();
        *row.entry(to.to_string()).or_insert(0.0) += 1.0;
    }

    /// Build the count/split tables with the default rows/cols.
    pub fn tables(&self) -> TransitionTables {
        self.tables_with(&TRANSITION_MATRIX_ROWS, &TRANSITION_MATRIX_COLS)
    }

    /// Build the count/split tables with explicit rows/cols.
    pub fn tables_with(&self, rows: &[&str], cols: &[&str]) -> TransitionTables {
        build_transition_tables(&self.counts_by_from, rows, cols)
    }
}

/// `buildTransitionTables` — row-normalise the count map into a split matrix.
pub fn build_transition_tables(
    transition_count: &TransitionCountMap,
    rows: &[&str],
    cols: &[&str],
) -> TransitionTables {
    let mut counts: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let mut splits: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for &r in rows {
        let mut counts_row: HashMap<String, f64> = HashMap::new();
        let mut splits_row: HashMap<String, f64> = HashMap::new();
        let row = transition_count.get(r);
        let mut total = 0.0;
        for &c in cols {
            let v = row.and_then(|m| m.get(c)).copied().unwrap_or(0.0);
            counts_row.insert(c.to_string(), v);
            total += v;
        }
        for &c in cols {
            let v = counts_row.get(c).copied().unwrap_or(0.0);
            splits_row.insert(c.to_string(), if total > 0.0 { v / total } else { 0.0 });
        }
        counts.insert(r.to_string(), counts_row);
        splits.insert(r.to_string(), splits_row);
    }
    TransitionTables { counts, splits }
}

/// `analyticalTransitionTables` — the exact branching matrix from the model
/// probabilities.
pub fn analytical_transition_tables(probabilities: &Probabilities) -> TransitionTables {
    let p = probabilities;
    let sparse: Vec<(&str, Vec<(&str, f64)>)> = vec![
        ("__source__", vec![("S", 1.0)]),
        ("S", vec![("E", 1.0)]),
        ("E", vec![("I-P", 1.0)]),
        (
            "I-P",
            vec![
                ("I-A", p.asymptomatic_share),
                ("I-S", 1.0 - p.asymptomatic_share),
            ],
        ),
        ("I-A", vec![("R", 1.0)]),
        (
            "I-S",
            vec![
                ("R", 1.0 - p.hospitalization_given_symptom),
                ("I-H", p.hospitalization_given_symptom),
            ],
        ),
        (
            "I-H",
            vec![
                ("R", 1.0 - p.case_fatality_given_hospital),
                ("D", p.case_fatality_given_hospital),
            ],
        ),
        ("R", vec![("S", 1.0)]),
        ("D", vec![("main-sink", 1.0)]),
    ];

    let mut map: TransitionCountMap = HashMap::new();
    for (from, row) in sparse {
        let inner: HashMap<String, f64> =
            row.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        map.insert(from.to_string(), inner);
    }
    build_transition_tables(&map, &TRANSITION_MATRIX_ROWS, &TRANSITION_MATRIX_COLS)
}

/// `zeroCompartmentRecord` — `{S: 0, E: 0, …}` over `COMPARTMENT_ORDER`.
pub fn zero_compartment_record() -> HashMap<String, f64> {
    COMPARTMENT_ORDER
        .iter()
        .map(|c| (c.to_string(), 0.0))
        .collect()
}

/// `compartmentPopulations` — fold per-station populations into compartments.
pub fn compartment_populations(
    population_of_station: impl Fn(&str) -> f64,
) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();
    for c in COMPARTMENT_ORDER {
        let sum: f64 = compartment_groups(c)
            .iter()
            .map(|sid| population_of_station(sid))
            .sum();
        out.insert(c.to_string(), sum);
    }
    out
}

/// `averageRecord` — divide each compartment sum by `max(1, denominator)`.
pub fn average_record(sums: &HashMap<String, f64>, denominator: f64) -> HashMap<String, f64> {
    let safe_denominator = denominator.max(1.0);
    let mut out: HashMap<String, f64> = HashMap::new();
    for c in COMPARTMENT_ORDER {
        out.insert(
            c.to_string(),
            sums.get(c).copied().unwrap_or(0.0) / safe_denominator,
        );
    }
    out
}

/// `updatePeaks` — elementwise running max into `peak`.
pub fn update_peaks(peak: &mut HashMap<String, f64>, values: &HashMap<String, f64>) {
    for c in COMPARTMENT_ORDER {
        let v = values.get(c).copied().unwrap_or(0.0);
        let entry = peak.entry(c.to_string()).or_insert(0.0);
        if v > *entry {
            *entry = v;
        }
    }
}

/// `meanResidence(config, id)` — midpoint of the `[a, b]` residence interval.
pub fn mean_residence(config: &SimConfig, id: &str) -> f64 {
    let (a, b) = config.residence[id];
    (a + b) / 2.0
}

// =============================================================================
// PORT NOTE — JSON serialization helpers (Rust-only addition).
//
// The TS drivers (`replicate`, `steady-state`, `stepsize-sweep`,
// `per-individual-vs-fel`) dump `RunResult`/config artifacts with
// `JSON.stringify(...)`. Rust has no structural reflection, so these helpers
// turn the result types into `JsonValue` (then `.to_string()`). They are not a
// 1:1 of any single TS symbol; they exist so each driver's `fs.writeFileSync`
// has a faithful payload without depending on `serde`.
// =============================================================================

fn jn(v: f64) -> JsonValue {
    JsonValue::Number(v)
}
fn js(v: &str) -> JsonValue {
    JsonValue::String(v.to_string())
}

/// `HashMap<String, f64>` → JSON object with keys sorted for stable output.
fn map_to_json(m: &HashMap<String, f64>) -> JsonValue {
    let mut keys: Vec<&String> = m.keys().collect();
    keys.sort();
    JsonValue::Object(keys.into_iter().map(|k| (k.clone(), jn(m[k]))).collect())
}

/// `HashMap<String, HashMap<String, f64>>` → nested JSON object (sorted keys).
fn nested_map_to_json(m: &HashMap<String, HashMap<String, f64>>) -> JsonValue {
    let mut keys: Vec<&String> = m.keys().collect();
    keys.sort();
    JsonValue::Object(
        keys.into_iter()
            .map(|k| (k.clone(), map_to_json(&m[k])))
            .collect(),
    )
}

/// Serialize the model branching probabilities.
pub fn probabilities_to_json(p: &Probabilities) -> JsonValue {
    JsonValue::Object(vec![
        ("asymptomaticShare".to_string(), jn(p.asymptomatic_share)),
        (
            "hospitalizationGivenSymptom".to_string(),
            jn(p.hospitalization_given_symptom),
        ),
        (
            "caseFatalityGivenHospital".to_string(),
            jn(p.case_fatality_given_hospital),
        ),
    ])
}

/// Serialize a [`SimConfig`] (mirrors the TS `config` literal shape).
pub fn config_to_json(c: &SimConfig) -> JsonValue {
    JsonValue::Object(vec![
        ("stepSize".to_string(), jn(c.step_size)),
        ("horizonDays".to_string(), jn(c.horizon_days)),
        ("phase1Days".to_string(), jn(c.phase1_days)),
        ("sourceCap".to_string(), jn(c.source_cap)),
        (
            "arrivalsInterarrival".to_string(),
            JsonValue::Array(vec![
                jn(c.arrivals_interarrival.0),
                jn(c.arrivals_interarrival.1),
            ]),
        ),
        ("residence".to_string(), {
            let mut keys: Vec<&String> = c.residence.keys().collect();
            keys.sort();
            JsonValue::Object(
                keys.into_iter()
                    .map(|k| {
                        let (a, b) = c.residence[k];
                        (k.clone(), JsonValue::Array(vec![jn(a), jn(b)]))
                    })
                    .collect(),
            )
        }),
        (
            "probabilities".to_string(),
            probabilities_to_json(&c.probabilities),
        ),
    ])
}

/// Serialize a [`RunResult`] (mirrors the TS `RunResult` object shape).
pub fn result_to_json(r: &RunResult) -> JsonValue {
    JsonValue::Object(vec![
        ("kernel".to_string(), js(r.kernel.as_str())),
        ("config".to_string(), config_to_json(&r.config)),
        ("seed".to_string(), jn(r.seed as f64)),
        (
            "totals".to_string(),
            JsonValue::Object(vec![
                ("created".to_string(), jn(r.totals.created)),
                ("absorbed".to_string(), jn(r.totals.absorbed)),
            ]),
        ),
        (
            "finalPopulations".to_string(),
            map_to_json(&r.final_populations),
        ),
        (
            "transitionCounts".to_string(),
            nested_map_to_json(&r.transition_counts),
        ),
        ("splitProbs".to_string(), nested_map_to_json(&r.split_probs)),
        (
            "timeAvgPopulations".to_string(),
            map_to_json(&r.time_avg_populations),
        ),
        (
            "peakPopulations".to_string(),
            map_to_json(&r.peak_populations),
        ),
        ("elapsedMs".to_string(), jn(r.elapsed_ms as f64)),
    ])
}

/// Serialize a slice of results to a JSON array.
pub fn results_to_json(results: &[RunResult]) -> JsonValue {
    JsonValue::Array(results.iter().map(result_to_json).collect())
}

/// Wire-string for a [`Kernel`] (re-export convenience for drivers).
pub fn kernel_label(k: Kernel) -> &'static str {
    k.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::runners::types::default_config;

    #[test]
    fn counter_records_and_splits() {
        let mut tc = TransitionCounter::new();
        tc.record("I-P", "I-A");
        tc.record("I-P", "I-A");
        tc.record("I-P", "I-S");
        let tables = tc.tables();
        let ip = &tables.splits["I-P"];
        assert!((ip["I-A"] - 2.0 / 3.0).abs() < 1e-12);
        assert!((ip["I-S"] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn analytical_tables_match_probabilities() {
        let cfg = default_config();
        let t = analytical_transition_tables(&cfg.probabilities);
        assert!((t.splits["I-P"]["I-A"] - 0.40).abs() < 1e-12);
        assert!((t.splits["I-H"]["D"] - 0.12).abs() < 1e-12);
    }

    #[test]
    fn mean_residence_midpoint() {
        let cfg = default_config();
        assert!((mean_residence(&cfg, "R") - 2.0).abs() < 1e-12);
    }
}
