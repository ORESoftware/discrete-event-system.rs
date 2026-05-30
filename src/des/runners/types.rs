//! Port of `src/des/runners/types.ts`.
//!
//! Shared config/result types for the SEIR kernel runners (framework, FEL
//! reference, per-individual-clock framework variant, Gillespie, ODE,
//! difference). This is a pure data module: no I/O, no RNG.
//!
//! ## Rust shape (faithful translation)
//!
//!   * `type Kernel = 'framework' | ...`  → [`Kernel`] enum with `as_str()`
//!     returning the kebab-case wire string.
//!   * `interface SimConfig` / `RunOpts` / `RunResult` → structs.
//!   * `[number, number]` interarrival/residence tuples → `(f64, f64)`.
//!   * `Record<string, number>` population maps → `HashMap<String, f64>`.
//!   * `Record<string, [number, number]>` residence → `HashMap<String, (f64, f64)>`.
//!   * `buildSuccessors` returns an ORDERED list (`Vec<(String, Vec<Successor>)>`)
//!     because the FEL kernel iterates the keys to schedule initial events and
//!     the RNG draw order depends on that iteration order.

#![allow(dead_code)]

use std::collections::HashMap;

/// Which simulation kernel produced a [`RunResult`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Framework,
    Fel,
    PerIndividual,
    Gillespie,
    Ode,
    Difference,
}

impl Kernel {
    /// Wire string (matches the TS string-literal union, kebab-case).
    pub fn as_str(self) -> &'static str {
        match self {
            Kernel::Framework => "framework",
            Kernel::Fel => "fel",
            Kernel::PerIndividual => "per-individual",
            Kernel::Gillespie => "gillespie",
            Kernel::Ode => "ode",
            Kernel::Difference => "difference",
        }
    }
}

/// Branching probabilities for the SEIR-with-hospitalization model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probabilities {
    pub asymptomatic_share: f64,
    pub hospitalization_given_symptom: f64,
    pub case_fatality_given_hospital: f64,
}

/// Configuration shared by all kernels.
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// Days per discrete step. Only the framework kernel uses it.
    pub step_size: f64,
    /// Total simulation horizon in days.
    pub horizon_days: f64,
    /// Days during which the source is active. After this we drain.
    pub phase1_days: f64,
    /// How many entities the source emits before quiescing.
    pub source_cap: f64,
    /// Inter-arrival uniform `[a, b]` (days) for the source.
    pub arrivals_interarrival: (f64, f64),
    /// Per-station service-clock uniform `[a, b]` (days) at every station.
    pub residence: HashMap<String, (f64, f64)>,
    pub probabilities: Probabilities,
}

/// FEL service discipline (`opts.service`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceDiscipline {
    /// Single per-station service clock (M/M/1). Default.
    Fifo,
    /// Per-entity exit clock at arrival (M/M/inf).
    Individual,
}

impl ServiceDiscipline {
    pub fn as_str(self) -> &'static str {
        match self {
            ServiceDiscipline::Fifo => "fifo",
            ServiceDiscipline::Individual => "individual",
        }
    }
}

/// Per-run options.
#[derive(Clone, Debug, Default)]
pub struct RunOpts {
    /// Seed for the seedable PRNG; deterministic when supplied.
    pub seed: Option<u64>,
    /// If true, dump JSONL events to `log_path`.
    pub log_events: bool,
    /// Where to write the JSONL log if `log_events` is true.
    pub log_path: Option<String>,
    /// Sample populations only every N days (default 1).
    pub sample_every_days: Option<f64>,
    /// FEL service discipline.
    pub service: Option<ServiceDiscipline>,
}

/// Totals reported by every kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Totals {
    pub created: f64,
    pub absorbed: f64,
}

/// The result bundle returned by every kernel runner.
#[derive(Clone, Debug)]
pub struct RunResult {
    pub kernel: Kernel,
    pub config: SimConfig,
    pub seed: u64,
    pub totals: Totals,
    pub final_populations: HashMap<String, f64>,
    /// Counts of (from -> to) transitions, decision nodes flattened away.
    pub transition_counts: HashMap<String, HashMap<String, f64>>,
    /// Empirical row-stochastic transition matrix.
    pub split_probs: HashMap<String, HashMap<String, f64>>,
    /// Mean over per-day samples.
    pub time_avg_populations: HashMap<String, f64>,
    pub peak_populations: HashMap<String, f64>,
    pub elapsed_ms: u128,
}

/// `COMPARTMENT_ORDER`.
pub const COMPARTMENT_ORDER: [&str; 7] = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R"];

/// `COMPARTMENT_GROUPS[c]` — the station ids that roll up into compartment `c`.
pub fn compartment_groups(c: &str) -> &'static [&'static str] {
    match c {
        "S" => &["S"],
        "E" => &["E"],
        "I-P" => &["I-P", "I-P-Decision"],
        "I-A" => &["I-A"],
        "I-S" => &["I-S", "I-S-Decision"],
        "I-H" => &["I-H", "I-H-Decision"],
        "R" => &["R"],
        _ => &[],
    }
}

/// `DEFAULT_RESIDENCE` — uniform `[a, b]` residence intervals (days).
pub fn default_residence() -> HashMap<String, (f64, f64)> {
    let entries: [(&str, (f64, f64)); 11] = [
        ("S", (0.20, 0.40)),
        ("E", (0.20, 0.40)),
        ("I-P", (0.20, 0.40)),
        ("I-A", (0.20, 0.40)),
        ("I-S", (0.20, 0.40)),
        ("I-H", (0.20, 0.40)),
        ("R", (1.50, 2.50)),
        ("D", (0.10, 0.30)),
        ("I-P-Decision", (0.05, 0.15)),
        ("I-S-Decision", (0.05, 0.15)),
        ("I-H-Decision", (0.05, 0.15)),
    ];
    entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

/// `DEFAULT_CONFIG`.
pub fn default_config() -> SimConfig {
    SimConfig {
        step_size: 1.0,
        horizon_days: 1200.0,
        phase1_days: 800.0,
        source_cap: 500.0,
        arrivals_interarrival: (0.7, 1.3),
        residence: default_residence(),
        probabilities: Probabilities {
            asymptomatic_share: 0.40,
            hospitalization_given_symptom: 0.20,
            case_fatality_given_hospital: 0.12,
        },
    }
}

/// `EDGES` — directed transitions (decision nodes still explicit here).
pub const EDGES: [(&str, &str); 15] = [
    ("main-source", "S"),
    ("S", "E"),
    ("E", "I-P"),
    ("I-P", "I-P-Decision"),
    ("I-P-Decision", "I-A"),
    ("I-P-Decision", "I-S"),
    ("I-A", "R"),
    ("I-S", "I-S-Decision"),
    ("I-S-Decision", "R"),
    ("I-S-Decision", "I-H"),
    ("I-H", "I-H-Decision"),
    ("I-H-Decision", "R"),
    ("I-H-Decision", "D"),
    ("D", "main-sink"),
    ("R", "S"),
];

/// One successor branch (`{prob, to}` in the TS object literal).
#[derive(Clone, Debug, PartialEq)]
pub struct Successor {
    pub prob: f64,
    pub to: String,
}

impl Successor {
    fn new(prob: f64, to: &str) -> Self {
        Successor {
            prob,
            to: to.to_string(),
        }
    }
}

/// Successor map (used by FEL and per-individual kernels). Mirrors `EDGES` but
/// folds branching probabilities into the from-station whose successors are the
/// decision node. Returned as an ORDERED `Vec` to preserve JS key-iteration
/// order (the FEL kernel's initial-event scheduling depends on it).
pub fn build_successors(p: &Probabilities) -> Vec<(String, Vec<Successor>)> {
    let asym = p.asymptomatic_share;
    let hosp = p.hospitalization_given_symptom;
    let cfr = p.case_fatality_given_hospital;
    vec![
        ("main-source".to_string(), vec![Successor::new(1.0, "S")]),
        ("S".to_string(), vec![Successor::new(1.0, "E")]),
        ("E".to_string(), vec![Successor::new(1.0, "I-P")]),
        ("I-P".to_string(), vec![Successor::new(1.0, "I-P-Decision")]),
        (
            "I-P-Decision".to_string(),
            vec![
                Successor::new(asym, "I-A"),
                Successor::new(1.0 - asym, "I-S"),
            ],
        ),
        ("I-A".to_string(), vec![Successor::new(1.0, "R")]),
        ("I-S".to_string(), vec![Successor::new(1.0, "I-S-Decision")]),
        (
            "I-S-Decision".to_string(),
            vec![Successor::new(1.0 - hosp, "R"), Successor::new(hosp, "I-H")],
        ),
        ("I-H".to_string(), vec![Successor::new(1.0, "I-H-Decision")]),
        (
            "I-H-Decision".to_string(),
            vec![Successor::new(1.0 - cfr, "R"), Successor::new(cfr, "D")],
        ),
        ("R".to_string(), vec![Successor::new(1.0, "S")]),
        ("D".to_string(), vec![Successor::new(1.0, "main-sink")]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let c = default_config();
        assert_eq!(c.step_size, 1.0);
        assert_eq!(c.source_cap, 500.0);
        assert_eq!(c.residence["R"], (1.50, 2.50));
        assert_eq!(c.probabilities.asymptomatic_share, 0.40);
    }

    #[test]
    fn successors_preserve_order_and_sum_to_one() {
        let s = build_successors(&default_config().probabilities);
        assert_eq!(s[0].0, "main-source");
        let ipd = &s.iter().find(|(k, _)| k == "I-P-Decision").unwrap().1;
        let sum: f64 = ipd.iter().map(|b| b.prob).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kernel_wire_strings() {
        assert_eq!(Kernel::PerIndividual.as_str(), "per-individual");
        assert_eq!(Kernel::Fel.as_str(), "fel");
    }
}
