//! Port of `src/des/runners/fel-runner.ts`.
//!
//! FEL (Future-Event-List) reference kernel. Supports two service disciplines
//! (`'fifo'` → M/M/1, `'individual'` → M/M/inf) switched via `opts.service`,
//! sharing the rest of the kernel (event loop, sampling, logging, transition
//! counting).
//!
//! ## Rust shape
//!
//!   * `opts.service: 'fifo' | 'individual'` → matched [`ServiceDiscipline`].
//!   * `Math.random()` (drawUniform) + `withSeed` → injected `RandomSource`.
//!   * the recursive `arrive` closure over mutable bindings → a `&mut self`
//!     method on [`Fel`] (decision nodes recurse into `self.arrive(..)`).
//!   * the sorted-array FEL (`splice`) → a `Vec<FelEvent>` with a stable
//!     linear insert (`insert after all events with time <= e.time`), so ties
//!     stay FIFO and the RNG draw order matches the TS.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::des::observability::logger::{JsonValue, JsonlLogger, LogLevel};
use crate::des::shared::capabilities::{Clock, RandomSource, SystemClock};
use crate::des::general::prng::with_seed;

use super::shared::{
    average_record, compartment_populations, update_peaks, zero_compartment_record,
    TransitionCounter,
};
use super::types::{
    build_successors, Kernel, RunOpts, RunResult, ServiceDiscipline, SimConfig, Successor, Totals,
    COMPARTMENT_ORDER,
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

fn draw_uniform(a: f64, b: f64, rng: &mut dyn RandomSource) -> f64 {
    a + rng.next_float() * (b - a)
}

fn succ_lookup<'a>(succs: &'a [(String, Vec<Successor>)], from: &str) -> &'a [Successor] {
    succs
        .iter()
        .find(|(k, _)| k == from)
        .map(|(_, v)| v.as_slice())
        .unwrap_or(&[])
}

fn draw_successor(succs: &[Successor], from: &str, rng: &mut dyn RandomSource) -> String {
    if succs.len() == 1 {
        return succs[0].to.clone();
    }
    let r = rng.next_float();
    let mut cum = 0.0;
    for sb in succs {
        cum += sb.prob;
        if r < cum {
            return sb.to.clone();
        }
    }
    if (cum - 1.0).abs() > 1e-6 {
        eprintln!(
            "[fel-runner] successor probabilities from \"{from}\" sum to {cum} (\u{2260} 1); draw r={r} fell through, defaulting to last successor \"{}\".",
            succs.last().map(|s| s.to.as_str()).unwrap_or("")
        );
    }
    succs.last().map(|s| s.to.clone()).unwrap_or_default()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FelKind {
    Source,
    Service,
    Exit,
}

struct FelEvent {
    time: f64,
    kind: FelKind,
    /// kind=service or kind=exit.
    station: Option<String>,
    /// kind=exit: the specific entity.
    entity: Option<String>,
}

fn insert_event(fel: &mut Vec<FelEvent>, e: FelEvent) {
    let mut i = 0;
    while i < fel.len() && fel[i].time <= e.time {
        i += 1;
    }
    fel.insert(i, e);
}

struct Fel<'a> {
    config: &'a SimConfig,
    successors: Vec<(String, Vec<Successor>)>,
    service: ServiceDiscipline,
    queues: HashMap<String, VecDeque<String>>,
    population: HashMap<String, f64>,
    transitions: TransitionCounter,
    absorbed: f64,
    last_station: HashMap<String, String>,
    fel: Vec<FelEvent>,
}

impl<'a> Fel<'a> {
    fn arrive(
        &mut self,
        t: f64,
        entity_id: &str,
        from_station: &str,
        to_station: &str,
        rng: &mut dyn RandomSource,
        logger: &mut Option<JsonlLogger>,
    ) {
        if to_station == "main-sink" {
            self.transitions.record(from_station, "main-sink");
            if let Some(logger) = logger.as_mut() {
                logger.log(jobj(vec![
                    ("kind", js("transition")),
                    ("t", jn(t)),
                    ("entity", js(entity_id)),
                    ("from", js(from_station)),
                    ("to", js("main-sink")),
                ]));
            }
            self.absorbed += 1.0;
            return;
        }
        if to_station.ends_with("-Decision") {
            // Decisions are instantaneous: pick a successor immediately and recurse.
            let dest = draw_successor(succ_lookup(&self.successors, to_station), to_station, rng);
            self.arrive(t, entity_id, from_station, &dest, rng, logger);
            return;
        }
        self.transitions.record(from_station, to_station);
        if let Some(logger) = logger.as_mut() {
            logger.log(jobj(vec![
                ("kind", js("transition")),
                ("t", jn(t)),
                ("entity", js(entity_id)),
                ("from", js(from_station)),
                ("to", js(to_station)),
            ]));
        }
        *self.population.entry(to_station.to_string()).or_insert(0.0) += 1.0;
        if self.service == ServiceDiscipline::Fifo {
            self.queues
                .entry(to_station.to_string())
                .or_default()
                .push_back(entity_id.to_string());
            self.last_station.insert(entity_id.to_string(), to_station.to_string());
        } else {
            let (a, b) = match self.config.residence.get(to_station) {
                Some(r) => *r,
                None => {
                    eprintln!(
                        "[fel-runner] no residence interval configured for station \"{to_station}\"; per-individual exit event cannot be scheduled."
                    );
                    (0.0, 0.0)
                }
            };
            insert_event(
                &mut self.fel,
                FelEvent {
                    time: t + draw_uniform(a, b, rng),
                    kind: FelKind::Exit,
                    station: Some(to_station.to_string()),
                    entity: Some(entity_id.to_string()),
                },
            );
        }
    }
}

/// `runFelOnce` — seed the RNG then run the FEL kernel.
pub fn run_fel_once(config: &SimConfig, opts: &RunOpts) -> RunResult {
    let seed = opts.seed.unwrap_or_else(|| SystemClock.now_ms() as u64);
    with_seed(seed as u32, |rng| run_fel_inner(config, opts, seed, rng))
}

fn run_fel_inner(
    config: &SimConfig,
    opts: &RunOpts,
    seed: u64,
    rng: &mut dyn RandomSource,
) -> RunResult {
    let sample_every = opts.sample_every_days.unwrap_or(1.0);
    let service = opts.service.unwrap_or(ServiceDiscipline::Fifo);
    let mut logger = if opts.log_events {
        opts.log_path
            .as_ref()
            .map(|path| JsonlLogger::new(path, LogLevel::Info))
    } else {
        None
    };

    let successors = build_successors(&config.probabilities);

    let mut queues: HashMap<String, VecDeque<String>> = HashMap::new();
    let mut population: HashMap<String, f64> = HashMap::new();
    for (s, _) in &successors {
        queues.insert(s.clone(), VecDeque::new());
        population.insert(s.clone(), 0.0);
    }
    population.insert("main-sink".to_string(), 0.0);

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
                    ("kernel", js(&format!("fel-{}", service.as_str()))),
                    ("seed", seed_val),
                    ("service", js(service.as_str())),
                    ("tPhase1", jn(config.phase1_days)),
                    ("tMax", jn(config.horizon_days)),
                    ("sourceCap", jn(config.source_cap)),
                ]),
            ),
        ]));
    }

    let mut sim = Fel {
        config,
        successors,
        service,
        queues,
        population,
        transitions: TransitionCounter::new(),
        absorbed: 0.0,
        last_station: HashMap::new(),
        fel: Vec::new(),
    };

    let mut source_created = 0.0_f64;
    let mut next_entity_id: u64 = 0;
    let mut phase2 = false;

    let mut pop_sums = zero_compartment_record();
    let mut peak = zero_compartment_record();
    let mut next_sample_at = sample_every;
    let mut samples = 0.0_f64;

    // Initial events.
    insert_event(
        &mut sim.fel,
        FelEvent {
            time: draw_uniform(
                config.arrivals_interarrival.0,
                config.arrivals_interarrival.1,
                rng,
            ),
            kind: FelKind::Source,
            station: None,
            entity: None,
        },
    );
    if service == ServiceDiscipline::Fifo {
        let keys: Vec<String> = sim.successors.iter().map(|(k, _)| k.clone()).collect();
        for s in keys {
            if s == "main-source" {
                continue;
            }
            let (a, b) = config.residence[&s];
            insert_event(
                &mut sim.fel,
                FelEvent {
                    time: draw_uniform(a, b, rng),
                    kind: FelKind::Service,
                    station: Some(s),
                    entity: None,
                },
            );
        }
    }

    let started_at = Instant::now();

    loop {
        let e = if sim.fel.is_empty() {
            break;
        } else {
            sim.fel.remove(0)
        };
        if e.time > config.horizon_days {
            break;
        }

        while next_sample_at <= e.time.floor() && next_sample_at <= config.horizon_days {
            sample_at(&sim, &mut pop_sums, &mut peak, &mut samples, &mut logger, phase2, next_sample_at);
            next_sample_at += sample_every;
        }

        if !phase2 && e.time >= config.phase1_days {
            phase2 = true;
            if let Some(logger) = logger.as_mut() {
                logger.log(jobj(vec![
                    ("kind", js("phase_change")),
                    ("t", jn(e.time.floor())),
                    ("phase", js("drain")),
                ]));
            }
        }

        match e.kind {
            FelKind::Source => {
                if source_created < config.source_cap && !phase2 {
                    let id = format!("f{next_entity_id}");
                    next_entity_id += 1;
                    sim.arrive(e.time, &id, "__source__", "S", rng, &mut logger);
                    source_created += 1.0;
                }
                let nt = e.time
                    + draw_uniform(
                        config.arrivals_interarrival.0,
                        config.arrivals_interarrival.1,
                        rng,
                    );
                insert_event(
                    &mut sim.fel,
                    FelEvent { time: nt, kind: FelKind::Source, station: None, entity: None },
                );
            }
            FelKind::Service => {
                let station = e.station.clone().unwrap();
                let head = sim.queues.get_mut(&station).and_then(|q| q.pop_front());
                if let Some(head) = head {
                    *sim.population.get_mut(&station).unwrap() -= 1.0;
                    let dest = draw_successor(succ_lookup(&sim.successors, &station), &station, rng);
                    sim.arrive(e.time, &head, &station, &dest, rng, &mut logger);
                }
                let (a, b) = config.residence[&station];
                let nt = e.time + draw_uniform(a, b, rng);
                insert_event(
                    &mut sim.fel,
                    FelEvent {
                        time: nt,
                        kind: FelKind::Service,
                        station: Some(station),
                        entity: None,
                    },
                );
            }
            FelKind::Exit => {
                let station = e.station.clone().unwrap();
                let entity = e.entity.clone().unwrap();
                *sim.population.get_mut(&station).unwrap() -= 1.0;
                let dest = draw_successor(succ_lookup(&sim.successors, &station), &station, rng);
                sim.arrive(e.time, &entity, &station, &dest, rng, &mut logger);
            }
        }
    }
    while next_sample_at <= config.horizon_days {
        sample_at(&sim, &mut pop_sums, &mut peak, &mut samples, &mut logger, phase2, next_sample_at);
        next_sample_at += sample_every;
    }

    let elapsed = started_at.elapsed().as_millis();
    let final_populations = compartment_populations(|sid| sim.population.get(sid).copied().unwrap_or(0.0));

    let tables = sim.transitions.tables();
    let time_avg = average_record(&pop_sums, samples);

    if let Some(logger) = logger.as_mut() {
        let final_pop_json: Vec<(String, JsonValue)> = COMPARTMENT_ORDER
            .iter()
            .map(|c| (c.to_string(), jn(final_populations.get(*c).copied().unwrap_or(0.0))))
            .collect();
        logger.log(jobj(vec![
            ("kind", js("sim_end")),
            ("t", jn(config.horizon_days)),
            ("elapsedMs", jn(elapsed as f64)),
            (
                "totals",
                JsonValue::Object(vec![
                    ("created".to_string(), jn(source_created)),
                    ("absorbed".to_string(), jn(sim.absorbed)),
                    ("finalPopulations".to_string(), JsonValue::Object(final_pop_json)),
                ]),
            ),
        ]));
        logger.close();
    }

    RunResult {
        kernel: Kernel::Fel,
        config: config.clone(),
        seed,
        totals: Totals { created: source_created, absorbed: sim.absorbed },
        final_populations,
        transition_counts: tables.counts,
        split_probs: tables.splits,
        time_avg_populations: time_avg,
        peak_populations: peak,
        elapsed_ms: elapsed,
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_at(
    sim: &Fel,
    pop_sums: &mut HashMap<String, f64>,
    peak: &mut HashMap<String, f64>,
    samples: &mut f64,
    logger: &mut Option<JsonlLogger>,
    phase2: bool,
    t: f64,
) {
    let populations = compartment_populations(|sid| sim.population.get(sid).copied().unwrap_or(0.0));
    let mut total_alive = 0.0;
    for c in COMPARTMENT_ORDER {
        let v = populations.get(c).copied().unwrap_or(0.0);
        total_alive += v;
        *pop_sums.get_mut(c).unwrap() += v;
    }
    update_peaks(peak, &populations);
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
            ("cumD", jn(sim.absorbed)),
            ("alive", jn(total_alive)),
            ("sourcesActive", jb(!phase2)),
        ]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::runners::types::{default_config, SimConfig};

    #[test]
    fn fel_fifo_runs_deterministically() {
        let cfg = SimConfig { horizon_days: 200.0, phase1_days: 120.0, ..default_config() };
        let a = run_fel_once(&cfg, &RunOpts { seed: Some(7), ..Default::default() });
        let b = run_fel_once(&cfg, &RunOpts { seed: Some(7), ..Default::default() });
        assert_eq!(a.kernel, Kernel::Fel);
        assert_eq!(a.totals.created, b.totals.created);
        assert_eq!(a.totals.absorbed, b.totals.absorbed);
    }

    #[test]
    fn fel_individual_service_runs() {
        let cfg = SimConfig { horizon_days: 200.0, phase1_days: 120.0, ..default_config() };
        let r = run_fel_once(
            &cfg,
            &RunOpts { seed: Some(11), service: Some(ServiceDiscipline::Individual), ..Default::default() },
        );
        assert!(r.totals.created > 0.0);
    }
}
