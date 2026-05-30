//! Port of `src/des/runners/framework-runner.ts`.
//!
//! The framework SEIR-with-hospitalization kernel as a callable function
//! (`runFrameworkOnce`). The TS wires the live demo's entity graph
//! (`EntitySource` → three-queue `EntityProcessor` stations →
//! `ProbabilityDecisionEntity` → `EntitySink`) and runs a fixed-step loop.
//!
//! ## PORT NOTE — local entity-graph stub
//!
//! The entity layer this runner drives (`entity-source`, **`entity-processing`**
//! / `EntityProcessor`, `entity-sink`, `entity-decision`, `abstract`,
//! `random-variables/rv`, `observers/program-observer`) is **not yet present**
//! in the Rust tree (`entity-processing` in particular has no `.rs` port). Per
//! the migration brief this file ships the *smallest self-contained local stub*
//! that reproduces the framework's observable semantics so the file compiles
//! and so the driver scripts (`replicate`, `stepsize-sweep`,
//! `per-individual-vs-fel`) can call it:
//!
//!   * Stations are an index-addressed graph ([`Station`] / [`Kind`]); the
//!     three-queue `EntityProcessor` collapses to one M/M/1 service clock per
//!     station (the documented framework semantic), decisions route by
//!     probability, the sink absorbs.
//!   * The TS `takeItem` monkey-patch that recorded transitions becomes a
//!     central [`FrameworkSim::deliver`] hook (records `prev -> station` exactly
//!     when an entity is taken into a processor/sink, decisions flattened).
//!   * `Math.random()` / `withSeed` → injected `SeededRandom`.
//!   * `bgn(stepSize)` (the exact `Decimal` DES clock) is approximated by an
//!     `f64` `dt`; wire `crate::des::shared::precision::Decimal` once the real
//!     `entity-processing` port lands for exact-clock parity.
//!   * `(global as any).turnOffSources` → the `sources_off` field;
//!     `console.log` suppression is unnecessary in Rust and dropped.
//!
//! When the real entity hierarchy is ported, replace this module's `Station`
//! engine with the genuine `EntitySource`/`EntityProcessor`/… structs.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::des::observability::logger::{JsonValue, JsonlLogger, LogLevel};
use crate::des::shared::capabilities::{Clock, RandomSource, SystemClock};
use crate::des::general::general::fisher_yates_shuffle;
use crate::des::general::prng::with_seed;

use super::shared::{
    average_record, compartment_populations, update_peaks, zero_compartment_record,
    TransitionCounter,
};
use super::types::{
    Kernel, RunOpts, RunResult, SimConfig, Totals, COMPARTMENT_ORDER, EDGES,
};

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

/// Station behaviour (`EntitySource` / `EntityProcessor` /
/// `ProbabilityDecisionEntity` / `EntitySink`).
enum Kind {
    Source { a: f64, b: f64, turn_off_after: f64 },
    Processor { a: f64, b: f64 },
    Decision { probs: Vec<f64> },
    Sink,
}

struct Station {
    id: String,
    kind: Kind,
    /// Out-connection target indices, in `addOutConnection` order.
    outs: Vec<usize>,
    /// Input queue (entity ids).
    input: VecDeque<u64>,
    /// M/M/1 service slot: `(entity, remaining_service_time)`.
    serving: Option<(u64, f64)>,
    /// Source clock + emission count.
    source_clock: f64,
    emitted: f64,
    created: f64,
    destroyed: f64,
}

impl Station {
    /// `stationPopulation(id)`: input + in-service.
    fn population(&self) -> f64 {
        let p = self.input.len() as f64;
        p + if self.serving.is_some() { 1.0 } else { 0.0 }
    }
}

struct FrameworkSim {
    stations: Vec<Station>,
    index_of: HashMap<String, usize>,
    last_processor: HashMap<u64, String>,
    transitions: TransitionCounter,
    next_id: u64,
    sources_off: bool,
}

impl FrameworkSim {
    fn new(config: &SimConfig) -> Self {
        // programEntities insertion order (matters for shuffle parity).
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
                    turn_off_after: config.source_cap,
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
                input: VecDeque::new(),
                serving: None,
                source_clock: 0.0,
                emitted: 0.0,
                created: 0.0,
                destroyed: 0.0,
            });
        }
        // Wire EDGES (addOutConnection order).
        for (src, tgt) in EDGES {
            let si = index_of[src];
            let ti = index_of[tgt];
            stations[si].outs.push(ti);
        }
        FrameworkSim {
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

    /// Advance one station by `dt`, returning routing actions `(target, entity)`.
    fn step_station(&mut self, idx: usize, dt: f64, rng: &mut dyn RandomSource) -> Vec<(usize, u64)> {
        let mut routes: Vec<(usize, u64)> = Vec::new();
        let out0 = self.stations[idx].outs.first().copied();
        let outs = self.stations[idx].outs.clone();
        // Read the kind params into owned locals so no borrow of `self.stations`
        // is held while we mutate it below.
        enum Local {
            Source { a: f64, b: f64, cap: f64 },
            Processor { a: f64, b: f64 },
            Decision { probs: Vec<f64> },
            Sink,
        }
        let local = match &self.stations[idx].kind {
            Kind::Source { a, b, turn_off_after } => Local::Source { a: *a, b: *b, cap: *turn_off_after },
            Kind::Processor { a, b } => Local::Processor { a: *a, b: *b },
            Kind::Decision { probs } => Local::Decision { probs: probs.clone() },
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
            Local::Processor { a, b } => {
                if self.stations[idx].serving.is_none() {
                    if let Some(head) = self.stations[idx].input.pop_front() {
                        self.stations[idx].serving = Some((head, a + rng.next_float() * (b - a)));
                    }
                }
                if let Some((entity, rem)) = self.stations[idx].serving {
                    let rem = rem - dt;
                    if rem <= 0.0 {
                        self.stations[idx].serving = None;
                        if let Some(t) = out0 {
                            routes.push((t, entity));
                        }
                    } else {
                        self.stations[idx].serving = Some((entity, rem));
                    }
                }
            }
            Local::Decision { probs } => {
                // Route the whole queue instantly by probability.
                let pending: Vec<u64> = self.stations[idx].input.drain(..).collect();
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

    /// Take an entity into `target` (the `takeItem` hook): record the transition
    /// when entering a processor/sink (decisions flattened).
    fn deliver(
        &mut self,
        target: usize,
        entity: u64,
        t: f64,
        logger: &mut Option<JsonlLogger>,
    ) {
        let target_id = self.stations[target].id.clone();
        let is_decision = matches!(self.stations[target].kind, Kind::Decision { .. });
        let is_sink = matches!(self.stations[target].kind, Kind::Sink);
        if is_decision {
            self.stations[target].input.push_back(entity);
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
        } else {
            self.last_processor.insert(entity, target_id);
            self.stations[target].input.push_back(entity);
        }
    }
}

/// `runFrameworkOnce` — seed the RNG then run the framework kernel.
pub fn run_framework_once(config: &SimConfig, opts: &RunOpts) -> RunResult {
    let seed = opts.seed.unwrap_or_else(|| SystemClock.now_ms() as u64);
    with_seed(seed as u32, |rng| run_framework_inner(config, opts, seed, rng))
}

fn run_framework_inner(
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
    let phase2_steps = ((config.horizon_days - config.phase1_days) / config.step_size).round() as i64;
    let steps_per_sample = (sample_every / config.step_size).round().max(1.0) as i64;

    if !(phase1_steps as f64).is_finite() || !(phase2_steps as f64).is_finite() {
        eprintln!("[framework-runner] non-finite phase step counts — open-system config will not terminate.");
    }
    if config.step_size <= 0.0 {
        eprintln!("[framework-runner] stepSize={} is not positive.", config.step_size);
    }

    let mut sim = FrameworkSim::new(config);
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
                    ("kernel", js("framework")),
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
        run_tick(&mut sim, &mut order, step_size, current_day, rng, &mut logger);
        if (i + 1) % steps_per_sample == 0 {
            sample_now(&sim, sink_idx, &mut pop_sums, &mut peak, &mut samples, &mut logger, current_day);
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
        run_tick(&mut sim, &mut order, step_size, current_day, rng, &mut logger);
        if (phase1_steps + i + 1) % steps_per_sample == 0 {
            sample_now(&sim, sink_idx, &mut pop_sums, &mut peak, &mut samples, &mut logger, current_day);
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
        kernel: Kernel::Framework,
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

/// One fixed-step tick: shuffle the station order then advance + route each.
fn run_tick(
    sim: &mut FrameworkSim,
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
            sim.deliver(target, entity, t, logger);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_now(
    sim: &FrameworkSim,
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
            .map(|c| (c.to_string(), jn(populations.get(*c).copied().unwrap_or(0.0))))
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
    fn framework_kernel_runs_deterministically() {
        let cfg = SimConfig { horizon_days: 200.0, phase1_days: 120.0, ..default_config() };
        let a = run_framework_once(&cfg, &RunOpts { seed: Some(0x10000), ..Default::default() });
        let b = run_framework_once(&cfg, &RunOpts { seed: Some(0x10000), ..Default::default() });
        assert_eq!(a.kernel, Kernel::Framework);
        assert_eq!(a.totals.created, b.totals.created);
        assert!(a.totals.created > 0.0);
    }
}
