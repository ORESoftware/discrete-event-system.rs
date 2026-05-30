//! Port of `src/des/runners/per-individual-runner.ts`.
//!
//! `runPerIndividualOnce` — the same SEIR-with-hospitalization graph as the
//! framework runner, but every processor station is a `PerIndividualProcessor`
//! (M/M/∞) instead of the three-queue `EntityProcessor` (M/M/1). Each entity
//! gets an **independent** residence-time draw at `takeItem` time, which is how
//! a CTMC kernel behaves; with a small `stepSize` it converges to the FEL
//! reference. The fixed-step run loop is otherwise identical to the framework
//! runner.
//!
//! ## PORT NOTE — local entity-graph stub
//!
//! Same situation as `framework_runner`: the `entity-processing`
//! (`PerIndividualProcessor`), `entity-source`, `entity-sink`,
//! `entity-decision`, `abstract`, `random-variables/rv`, and
//! `observers/program-observer` modules are **not yet ported** to Rust. This
//! file ships the smallest self-contained engine reproducing the per-individual
//! semantics:
//!
//!   * A processor station holds a `Vec<(entity, remaining_time)>` of in-flight
//!     individuals; `doTimeStep` decrements every clock and routes the finished
//!     ones out (true M/M/∞).
//!   * The residence draw `a + rng()*(b - a)` happens in [`PerIndividualSim::deliver`]
//!     exactly when an entity is taken into a processor — mirroring the TS
//!     `drawDuration` closure firing at `takeItem`.
//!   * `stationPopulation` = `items.len()` for processors, decision-queue length
//!     for decisions, `0` for source/sink (matches the TS reflection on
//!     `e.items` / `e.queue`).
//!   * Decision nodes route their queue instantly by probability (their tiny
//!     residence is dropped — a documented approximation; fold in a per-entity
//!     decision clock once `entity-decision` is ported for exact occupancy).
//!   * `Math.random()`/`withSeed` → injected `SeededRandom`; `bgn(stepSize)` →
//!     `f64` `dt` (wire `Decimal` after the real `entity-processing` port).
//!   * `(global as any).turnOffSources` → the `sources_off` field; the
//!     `console.log` muting is unnecessary in Rust.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::des::general::general::fisher_yates_shuffle;
use crate::des::general::prng::with_seed;
use crate::des::observability::logger::{JsonValue, JsonlLogger, LogLevel};
use crate::des::shared::capabilities::{Clock, RandomSource, SystemClock};

use super::shared::{
    average_record, compartment_populations, update_peaks, zero_compartment_record,
    TransitionCounter,
};
use super::types::{Kernel, RunOpts, RunResult, SimConfig, Totals, COMPARTMENT_ORDER, EDGES};

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
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

enum Kind {
    Source { a: f64, b: f64, cap: f64 },
    Processor { a: f64, b: f64 },
    Decision { probs: Vec<f64> },
    Sink,
}

struct Station {
    id: String,
    kind: Kind,
    outs: Vec<usize>,
    /// In-flight individuals `(entity, remaining_service_time)` (M/M/∞).
    items: Vec<(u64, f64)>,
    /// Decision queue.
    pending: VecDeque<u64>,
    source_clock: f64,
    emitted: f64,
    created: f64,
    destroyed: f64,
}

impl Station {
    fn population(&self) -> f64 {
        match self.kind {
            Kind::Processor { .. } => self.items.len() as f64,
            Kind::Decision { .. } => self.pending.len() as f64,
            _ => 0.0,
        }
    }
}

struct PerIndividualSim {
    stations: Vec<Station>,
    index_of: HashMap<String, usize>,
    last_processor: HashMap<u64, String>,
    transitions: TransitionCounter,
    next_id: u64,
    sources_off: bool,
}

impl PerIndividualSim {
    fn new(config: &SimConfig) -> Self {
        let order: [&str; 13] = [
            "main-source",
            "S",
            "E",
            "I-P",
            "I-P-Decision",
            "I-A",
            "I-S",
            "I-S-Decision",
            "I-H",
            "I-H-Decision",
            "R",
            "D",
            "main-sink",
        ];
        let p = &config.probabilities;
        let mut stations: Vec<Station> = Vec::new();
        let mut index_of: HashMap<String, usize> = HashMap::new();
        for (i, id) in order.iter().enumerate() {
            index_of.insert(id.to_string(), i);
            let kind = match *id {
                "main-source" => Kind::Source {
                    a: config.arrivals_interarrival.0,
                    b: config.arrivals_interarrival.1,
                    cap: config.source_cap,
                },
                "main-sink" => Kind::Sink,
                "I-P-Decision" => Kind::Decision {
                    probs: vec![p.asymptomatic_share, 1.0 - p.asymptomatic_share],
                },
                "I-S-Decision" => Kind::Decision {
                    probs: vec![
                        1.0 - p.hospitalization_given_symptom,
                        p.hospitalization_given_symptom,
                    ],
                },
                "I-H-Decision" => Kind::Decision {
                    probs: vec![
                        1.0 - p.case_fatality_given_hospital,
                        p.case_fatality_given_hospital,
                    ],
                },
                other => {
                    let (a, b) = config.residence[other];
                    Kind::Processor { a, b }
                }
            };
            stations.push(Station {
                id: id.to_string(),
                kind,
                outs: Vec::new(),
                items: Vec::new(),
                pending: VecDeque::new(),
                source_clock: 0.0,
                emitted: 0.0,
                created: 0.0,
                destroyed: 0.0,
            });
        }
        for (src, tgt) in EDGES {
            let si = index_of[src];
            let ti = index_of[tgt];
            stations[si].outs.push(ti);
        }
        PerIndividualSim {
            stations,
            index_of,
            last_processor: HashMap::new(),
            transitions: TransitionCounter::new(),
            next_id: 0,
            sources_off: false,
        }
    }

    fn population_of(&self, id: &str) -> f64 {
        match self.index_of.get(id) {
            Some(&i) => self.stations[i].population(),
            None => 0.0,
        }
    }

    fn step_station(
        &mut self,
        idx: usize,
        dt: f64,
        rng: &mut dyn RandomSource,
    ) -> Vec<(usize, u64)> {
        let mut routes: Vec<(usize, u64)> = Vec::new();
        let out0 = self.stations[idx].outs.first().copied();
        let outs = self.stations[idx].outs.clone();
        enum Local {
            Source { a: f64, b: f64, cap: f64 },
            Processor,
            Decision { probs: Vec<f64> },
            Sink,
        }
        let local = match &self.stations[idx].kind {
            Kind::Source { a, b, cap } => Local::Source {
                a: *a,
                b: *b,
                cap: *cap,
            },
            Kind::Processor { .. } => Local::Processor,
            Kind::Decision { probs } => Local::Decision {
                probs: probs.clone(),
            },
            Kind::Sink => Local::Sink,
        };
        match local {
            Local::Source { a, b, cap } => {
                let sources_off = self.sources_off;
                self.stations[idx].source_clock -= dt;
                while self.stations[idx].source_clock <= 0.0 {
                    if !sources_off && self.stations[idx].emitted < cap {
                        let id = self.next_id;
                        self.next_id += 1;
                        self.stations[idx].created += 1.0;
                        self.stations[idx].emitted += 1.0;
                        if let Some(t) = out0 {
                            routes.push((t, id));
                        }
                    }
                    self.stations[idx].source_clock += a + rng.next_float() * (b - a);
                    if sources_off {
                        break;
                    }
                }
            }
            Local::Processor => {
                // Decrement every in-flight clock; route the finished ones out.
                let mut survivors: Vec<(u64, f64)> = Vec::new();
                let items = std::mem::take(&mut self.stations[idx].items);
                for (entity, rem) in items {
                    let rem = rem - dt;
                    if rem <= 0.0 {
                        if let Some(t) = out0 {
                            routes.push((t, entity));
                        }
                    } else {
                        survivors.push((entity, rem));
                    }
                }
                self.stations[idx].items = survivors;
            }
            Local::Decision { probs } => {
                let pending: Vec<u64> = self.stations[idx].pending.drain(..).collect();
                for entity in pending {
                    let r = rng.next_float();
                    let mut cum = 0.0;
                    let mut chosen = outs.len().saturating_sub(1);
                    for (i, p) in probs.iter().enumerate() {
                        cum += *p;
                        if r < cum {
                            chosen = i;
                            break;
                        }
                    }
                    if let Some(&t) = outs.get(chosen) {
                        routes.push((t, entity));
                    }
                }
            }
            Local::Sink => {}
        }
        routes
    }

    /// `takeItem` hook: record the transition and (for a processor) draw the
    /// individual's residence time at entry.
    fn deliver(
        &mut self,
        target: usize,
        entity: u64,
        t: f64,
        rng: &mut dyn RandomSource,
        logger: &mut Option<JsonlLogger>,
    ) {
        let target_id = self.stations[target].id.clone();
        let (a, b, is_proc, is_decision, is_sink) = match &self.stations[target].kind {
            Kind::Processor { a, b } => (*a, *b, true, false, false),
            Kind::Decision { .. } => (0.0, 0.0, false, true, false),
            Kind::Sink => (0.0, 0.0, false, false, true),
            Kind::Source { .. } => (0.0, 0.0, false, false, false),
        };
        if is_decision {
            self.stations[target].pending.push_back(entity);
            return;
        }
        let prev = self
            .last_processor
            .get(&entity)
            .cloned()
            .unwrap_or_else(|| "__source__".to_string());
        self.transitions.record(&prev, &target_id);
        if let Some(logger) = logger.as_mut() {
            logger.log(jobj(vec![
                ("kind", js("transition")),
                ("t", jn(t)),
                ("entity", js(&format!("e{entity}"))),
                ("from", js(&prev)),
                ("to", js(&target_id)),
            ]));
        }
        if is_sink {
            self.stations[target].destroyed += 1.0;
        } else if is_proc {
            let dur = a + rng.next_float() * (b - a);
            self.last_processor.insert(entity, target_id);
            self.stations[target].items.push((entity, dur));
        }
    }
}

/// `runPerIndividualOnce` — seed the RNG then run the per-individual kernel.
pub fn run_per_individual_once(config: &SimConfig, opts: &RunOpts) -> RunResult {
    let seed = opts.seed.unwrap_or_else(|| SystemClock.now_ms() as u64);
    with_seed(seed as u32, |rng| run_inner(config, opts, seed, rng))
}

fn run_inner(
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

    let step_size = config.step_size;
    let phase1_steps = (config.phase1_days / config.step_size).round() as i64;
    let phase2_steps =
        ((config.horizon_days - config.phase1_days) / config.step_size).round() as i64;
    let steps_per_sample = (sample_every / config.step_size).round().max(1.0) as i64;

    let mut sim = PerIndividualSim::new(config);
    let source_idx = sim.index_of["main-source"];
    let sink_idx = sim.index_of["main-sink"];

    if let Some(logger) = logger.as_mut() {
        let seed_val = match opts.seed {
            Some(s) => jn(s as f64),
            None => jn(seed as f64),
        };
        logger.log(jobj(vec![
            ("kind", js("sim_start")),
            (
                "config",
                jobj(vec![
                    ("kernel", js("per-individual")),
                    ("seed", seed_val),
                    ("stepSize", jn(config.step_size)),
                    ("phase1Steps", jn(phase1_steps as f64)),
                    ("phase2Steps", jn(phase2_steps as f64)),
                    ("sourceCap", jn(config.source_cap)),
                ]),
            ),
        ]));
    }

    let mut pop_sums = zero_compartment_record();
    let mut peak = zero_compartment_record();
    let mut samples = 0.0_f64;

    let n = sim.stations.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut current_day = 0.0_f64;

    let started_at = Instant::now();

    for i in 0..phase1_steps {
        current_day = (i + 1) as f64 * config.step_size;
        run_tick(
            &mut sim,
            &mut order,
            step_size,
            current_day,
            rng,
            &mut logger,
        );
        if (i + 1) % steps_per_sample == 0 {
            sample_now(
                &sim,
                sink_idx,
                &mut pop_sums,
                &mut peak,
                &mut samples,
                &mut logger,
                current_day,
            );
        }
    }
    sim.sources_off = true;
    if let Some(logger) = logger.as_mut() {
        logger.log(jobj(vec![
            ("kind", js("phase_change")),
            ("t", jn(current_day)),
            ("phase", js("drain")),
        ]));
    }
    for i in 0..phase2_steps {
        current_day = (phase1_steps + i + 1) as f64 * config.step_size;
        run_tick(
            &mut sim,
            &mut order,
            step_size,
            current_day,
            rng,
            &mut logger,
        );
        if (phase1_steps + i + 1) % steps_per_sample == 0 {
            sample_now(
                &sim,
                sink_idx,
                &mut pop_sums,
                &mut peak,
                &mut samples,
                &mut logger,
                current_day,
            );
        }
    }

    let elapsed = started_at.elapsed().as_millis();

    let created = sim.stations[source_idx].created;
    let absorbed = sim.stations[sink_idx].destroyed;
    let final_populations = compartment_populations(|id| sim.population_of(id));

    let tables = sim.transitions.tables();
    let time_avg = average_record(&pop_sums, samples);

    if let Some(logger) = logger.as_mut() {
        logger.log(jobj(vec![
            ("kind", js("sim_end")),
            ("t", jn(current_day)),
            ("elapsedMs", jn(elapsed as f64)),
        ]));
        logger.close();
    }

    RunResult {
        kernel: Kernel::PerIndividual,
        config: config.clone(),
        seed,
        totals: Totals { created, absorbed },
        final_populations,
        transition_counts: tables.counts,
        split_probs: tables.splits,
        time_avg_populations: time_avg,
        peak_populations: peak,
        elapsed_ms: elapsed,
    }
}

fn run_tick(
    sim: &mut PerIndividualSim,
    order: &mut [usize],
    step_size: f64,
    t: f64,
    rng: &mut dyn RandomSource,
    logger: &mut Option<JsonlLogger>,
) {
    fisher_yates_shuffle(order, rng);
    for idx in 0..order.len() {
        let station = order[idx];
        let routes = sim.step_station(station, step_size, rng);
        for (target, entity) in routes {
            sim.deliver(target, entity, t, rng, logger);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_now(
    sim: &PerIndividualSim,
    sink_idx: usize,
    pop_sums: &mut HashMap<String, f64>,
    peak: &mut HashMap<String, f64>,
    samples: &mut f64,
    logger: &mut Option<JsonlLogger>,
    t: f64,
) {
    let populations = compartment_populations(|id| sim.population_of(id));
    let mut total_alive = 0.0;
    for c in COMPARTMENT_ORDER {
        let v = populations.get(c).copied().unwrap_or(0.0);
        *pop_sums.get_mut(c).unwrap() += v;
        total_alive += v;
    }
    update_peaks(peak, &populations);
    let cum_d = sim.stations[sink_idx].destroyed;
    *samples += 1.0;
    if let Some(logger) = logger.as_mut() {
        let pops_json: Vec<(String, JsonValue)> = COMPARTMENT_ORDER
            .iter()
            .map(|c| {
                (
                    c.to_string(),
                    jn(populations.get(*c).copied().unwrap_or(0.0)),
                )
            })
            .collect();
        logger.log(jobj(vec![
            ("kind", js("tick")),
            ("t", jn(t)),
            ("populations", JsonValue::Object(pops_json)),
            ("cumD", jn(cum_d)),
            ("alive", jn(total_alive)),
            ("sourcesActive", jb(!sim.sources_off)),
        ]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::runners::types::{default_config, SimConfig};

    #[test]
    fn per_individual_kernel_runs_deterministically() {
        let cfg = SimConfig {
            horizon_days: 200.0,
            phase1_days: 120.0,
            ..default_config()
        };
        let a = run_per_individual_once(
            &cfg,
            &RunOpts {
                seed: Some(7),
                ..Default::default()
            },
        );
        let b = run_per_individual_once(
            &cfg,
            &RunOpts {
                seed: Some(7),
                ..Default::default()
            },
        );
        assert_eq!(a.kernel, Kernel::PerIndividual);
        assert_eq!(a.totals.created, b.totals.created);
        assert!(a.totals.created > 0.0);
    }
}
