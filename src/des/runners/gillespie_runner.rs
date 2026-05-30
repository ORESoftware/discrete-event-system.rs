//! Port of `src/des/runners/gillespie-runner.ts`.
//!
//! Gillespie Stochastic Simulation Algorithm (direct method) compartment-level
//! kernel: no entity objects, no event list, just `N_c` counts and per-reaction
//! propensities.
//!
//! The TS `interface Reaction` carried `propensity`/`fire` closures over
//! module-level mutable bindings, which Rust does not allow. Per the migration
//! note this becomes a fixed reaction table (`REACTIONS`, index-stable) plus
//! `propensity(idx)` / `fire(idx)` methods on a [`Sim`] state struct matched on
//! the index — preserving the exact ordering the reaction-selection loop relies
//! on. RNG is injected (`with_seed`) rather than read from a global.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Instant;

use crate::des::observability::logger::{JsonValue, JsonlLogger, LogLevel};
use crate::des::shared::capabilities::{Clock, RandomSource, SystemClock};
use crate::des::general::prng::with_seed;

use super::shared::{
    average_record, mean_residence, update_peaks, zero_compartment_record, TransitionCounter,
};
use super::types::{Kernel, RunOpts, RunResult, SimConfig, Totals, COMPARTMENT_ORDER};

fn js(v: &str) -> JsonValue {
    JsonValue::String(v.to_string())
}
fn jn(v: f64) -> JsonValue {
    JsonValue::Number(v)
}
fn jb(v: bool) -> JsonValue {
    JsonValue::Bool(v)
}
fn jobj(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// Compartment counts (`N`). No `D` — deaths are absorbed immediately.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    s: f64,
    e: f64,
    i_p: f64,
    i_a: f64,
    i_s: f64,
    i_h: f64,
    r: f64,
}

impl Counts {
    fn compartment(&self, c: &str) -> f64 {
        match c {
            "S" => self.s,
            "E" => self.e,
            "I-P" => self.i_p,
            "I-A" => self.i_a,
            "I-S" => self.i_s,
            "I-H" => self.i_h,
            "R" => self.r,
            _ => 0.0,
        }
    }
}

struct Mu {
    arrival: f64,
    s: f64,
    e: f64,
    i_p: f64,
    i_a: f64,
    i_s: f64,
    i_h: f64,
    r: f64,
}

/// `(id, from, to)` for each reaction. Order is significant: the
/// reaction-selection loop walks this in order.
const REACTIONS: [(&str, &str, &str); 11] = [
    ("src", "__source__", "S"),
    ("S->E", "S", "E"),
    ("E->I-P", "E", "I-P"),
    ("I-P->I-A", "I-P", "I-A"),
    ("I-P->I-S", "I-P", "I-S"),
    ("I-A->R", "I-A", "R"),
    ("I-S->R", "I-S", "R"),
    ("I-S->I-H", "I-S", "I-H"),
    ("I-H->R", "I-H", "R"),
    ("I-H->D", "I-H", "D"),
    ("R->S", "R", "S"),
];

struct Sim<'a> {
    config: &'a SimConfig,
    mu: Mu,
    n: Counts,
    source_created: f64,
    absorbed: f64,
    phase2: bool,
    transitions: TransitionCounter,
}

impl<'a> Sim<'a> {
    fn propensity(&self, idx: usize) -> f64 {
        let p = &self.config.probabilities;
        let mu = &self.mu;
        let n = &self.n;
        match idx {
            0
                if self.source_created < self.config.source_cap && !self.phase2 => {
                    1.0 / mu.arrival
                }
            1 => n.s / mu.s,
            2 => n.e / mu.e,
            3 => n.i_p * p.asymptomatic_share / mu.i_p,
            4 => n.i_p * (1.0 - p.asymptomatic_share) / mu.i_p,
            5 => n.i_a / mu.i_a,
            6 => n.i_s * (1.0 - p.hospitalization_given_symptom) / mu.i_s,
            7 => n.i_s * p.hospitalization_given_symptom / mu.i_s,
            8 => n.i_h * (1.0 - p.case_fatality_given_hospital) / mu.i_h,
            9 => n.i_h * p.case_fatality_given_hospital / mu.i_h,
            10 => n.r / mu.r,
            _ => 0.0,
        }
    }

    fn fire(&mut self, idx: usize) {
        match idx {
            0 => {
                self.n.s += 1.0;
                self.source_created += 1.0;
                self.transitions.record("__source__", "S");
            }
            1 => {
                self.n.s -= 1.0;
                self.n.e += 1.0;
                self.transitions.record("S", "E");
            }
            2 => {
                self.n.e -= 1.0;
                self.n.i_p += 1.0;
                self.transitions.record("E", "I-P");
            }
            3 => {
                self.n.i_p -= 1.0;
                self.n.i_a += 1.0;
                self.transitions.record("I-P", "I-A");
            }
            4 => {
                self.n.i_p -= 1.0;
                self.n.i_s += 1.0;
                self.transitions.record("I-P", "I-S");
            }
            5 => {
                self.n.i_a -= 1.0;
                self.n.r += 1.0;
                self.transitions.record("I-A", "R");
            }
            6 => {
                self.n.i_s -= 1.0;
                self.n.r += 1.0;
                self.transitions.record("I-S", "R");
            }
            7 => {
                self.n.i_s -= 1.0;
                self.n.i_h += 1.0;
                self.transitions.record("I-S", "I-H");
            }
            8 => {
                self.n.i_h -= 1.0;
                self.n.r += 1.0;
                self.transitions.record("I-H", "R");
            }
            9 => {
                self.n.i_h -= 1.0;
                self.absorbed += 1.0;
                // Mirror the FEL/framework topology: I-H -> D, then D -> main-sink.
                self.transitions.record("I-H", "D");
                self.transitions.record("D", "main-sink");
            }
            10 => {
                self.n.r -= 1.0;
                self.n.s += 1.0;
                self.transitions.record("R", "S");
            }
            _ => {}
        }
    }
}

/// `runGillespieOnce` — seed the RNG then run the direct-method SSA.
pub fn run_gillespie_once(config: &SimConfig, opts: &RunOpts) -> RunResult {
    let seed = opts.seed.unwrap_or_else(|| SystemClock.now_ms() as u64);
    with_seed(seed as u32, |rng| run_gillespie_inner(config, opts, seed, rng))
}

fn run_gillespie_inner(
    config: &SimConfig,
    opts: &RunOpts,
    seed: u64,
    rng: &mut dyn RandomSource,
) -> RunResult {
    let sample_every = opts.sample_every_days.unwrap_or(1.0);
    let mut logger = if opts.log_events {
        opts.log_path
            .as_ref()
            .map(|path| JsonlLogger::new(path, LogLevel::Info))
    } else {
        None
    };

    let mu = Mu {
        arrival: (config.arrivals_interarrival.0 + config.arrivals_interarrival.1) / 2.0,
        s: mean_residence(config, "S"),
        e: mean_residence(config, "E"),
        i_p: mean_residence(config, "I-P"),
        i_a: mean_residence(config, "I-A"),
        i_s: mean_residence(config, "I-S"),
        i_h: mean_residence(config, "I-H"),
        r: mean_residence(config, "R"),
    };

    if let Some(logger) = logger.as_mut() {
        let seed_val = match opts.seed {
            Some(seed) => jn(seed as f64),
            None => jn(seed as f64),
        };
        logger.log(jobj(vec![
            ("kind", js("sim_start")),
            (
                "config",
                jobj(vec![
                    ("kernel", js("gillespie-ssa")),
                    ("seed", seed_val),
                    ("tPhase1", jn(config.phase1_days)),
                    ("tMax", jn(config.horizon_days)),
                    ("sourceCap", jn(config.source_cap)),
                ]),
            ),
        ]));
    }

    let mut sim = Sim {
        config,
        mu,
        n: Counts::default(),
        source_created: 0.0,
        absorbed: 0.0,
        phase2: false,
        transitions: TransitionCounter::new(),
    };

    let mut pop_sums = zero_compartment_record();
    let mut peak = zero_compartment_record();
    let mut next_sample_at = sample_every;
    let mut samples = 0.0_f64;
    let mut t = 0.0_f64;

    let started_at = Instant::now();

    while t < config.horizon_days {
        if !sim.phase2 && t >= config.phase1_days {
            sim.phase2 = true;
            if let Some(logger) = logger.as_mut() {
                logger.log(jobj(vec![
                    ("kind", js("phase_change")),
                    ("t", jn(t.floor())),
                    ("phase", js("drain")),
                ]));
            }
        }

        let props: Vec<f64> = (0..REACTIONS.len()).map(|i| sim.propensity(i)).collect();
        let total: f64 = props.iter().sum();

        if total <= 0.0 {
            // No reactions enabled. Skip ahead to the horizon.
            let dt = config.horizon_days - t;
            for c in COMPARTMENT_ORDER {
                *pop_sums.get_mut(c).unwrap() += sim.n.compartment(c) * dt;
            }
            while next_sample_at <= config.horizon_days {
                sample_at(&sim, &mut peak, &mut samples, &mut logger, next_sample_at);
                next_sample_at += sample_every;
            }
            // TS advanced `t` to the horizon here; in Rust `t` is not read after
            // the loop, so the assignment is omitted to avoid a dead store.
            break;
        }

        let dt = -(rng.next_float().ln()) / total;
        for c in COMPARTMENT_ORDER {
            *pop_sums.get_mut(c).unwrap() += sim.n.compartment(c) * dt;
        }
        while next_sample_at <= t + dt && next_sample_at <= config.horizon_days {
            sample_at(&sim, &mut peak, &mut samples, &mut logger, next_sample_at);
            next_sample_at += sample_every;
        }
        t += dt;
        if t > config.horizon_days {
            break;
        }

        let u = rng.next_float() * total;
        let mut cum = 0.0;
        let mut fired = REACTIONS.len() - 1;
        for (i, &prop) in props.iter().enumerate() {
            cum += prop;
            if u < cum {
                fired = i;
                break;
            }
        }
        sim.fire(fired);
        if let Some(logger) = logger.as_mut() {
            let (_, from, to) = REACTIONS[fired];
            logger.log(jobj(vec![
                ("kind", js("transition")),
                ("t", jn(t)),
                ("from", js(from)),
                ("to", js(to)),
            ]));
        }
    }

    let elapsed = started_at.elapsed().as_millis();
    let mut final_populations: HashMap<String, f64> = HashMap::new();
    for c in COMPARTMENT_ORDER {
        final_populations.insert(c.to_string(), sim.n.compartment(c));
    }

    let tables = sim.transitions.tables();
    let time_avg = average_record(&pop_sums, config.horizon_days);

    if let Some(logger) = logger.as_mut() {
        let final_pop_json: Vec<(String, JsonValue)> = COMPARTMENT_ORDER
            .iter()
            .map(|c| (c.to_string(), jn(sim.n.compartment(c))))
            .collect();
        logger.log(jobj(vec![
            ("kind", js("sim_end")),
            ("t", jn(config.horizon_days)),
            ("elapsedMs", jn(elapsed as f64)),
            (
                "totals",
                JsonValue::Object(vec![
                    ("created".to_string(), jn(sim.source_created)),
                    ("absorbed".to_string(), jn(sim.absorbed)),
                    ("finalPopulations".to_string(), JsonValue::Object(final_pop_json)),
                ]),
            ),
        ]));
        logger.close();
    }

    let _ = samples;
    RunResult {
        kernel: Kernel::Gillespie,
        config: config.clone(),
        seed,
        totals: Totals { created: sim.source_created, absorbed: sim.absorbed },
        final_populations,
        transition_counts: tables.counts,
        split_probs: tables.splits,
        time_avg_populations: time_avg,
        peak_populations: peak,
        elapsed_ms: elapsed,
    }
}

fn sample_at(
    sim: &Sim,
    peak: &mut HashMap<String, f64>,
    samples: &mut f64,
    logger: &mut Option<JsonlLogger>,
    t_now: f64,
) {
    let mut total_alive = 0.0;
    let mut pops: HashMap<String, f64> = HashMap::new();
    for c in COMPARTMENT_ORDER {
        let v = sim.n.compartment(c);
        pops.insert(c.to_string(), v);
        total_alive += v;
    }
    update_peaks(peak, &pops);
    *samples += 1.0;
    if let Some(logger) = logger.as_mut() {
        let populations: Vec<(String, JsonValue)> = COMPARTMENT_ORDER
            .iter()
            .map(|c| (c.to_string(), jn(sim.n.compartment(c))))
            .collect();
        logger.log(jobj(vec![
            ("kind", js("tick")),
            ("t", jn(t_now)),
            ("populations", JsonValue::Object(populations)),
            ("cumD", jn(sim.absorbed)),
            ("alive", jn(total_alive)),
            ("sourcesActive", jb(!sim.phase2)),
        ]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::runners::types::{default_config, SimConfig};

    #[test]
    fn gillespie_kernel_runs_deterministically() {
        let cfg = SimConfig { horizon_days: 100.0, ..default_config() };
        let a = run_gillespie_once(&cfg, &RunOpts { seed: Some(42), ..Default::default() });
        let b = run_gillespie_once(&cfg, &RunOpts { seed: Some(42), ..Default::default() });
        assert_eq!(a.kernel, Kernel::Gillespie);
        assert_eq!(a.totals.created, b.totals.created);
        assert_eq!(a.totals.absorbed, b.totals.absorbed);
    }
}
