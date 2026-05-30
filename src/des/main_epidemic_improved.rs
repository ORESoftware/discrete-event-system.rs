//! Port of `src/des/main-epidemic-improved.ts`.
//!
//! SEIR-with-hospitalization epidemic over the entity graph; samples compartment
//! populations to CSV and prints an empirical transition matrix.
//!
//! Reuses the real entity modules (source / processor / probability-decision /
//! sink) wired through a [`Stn`] enum, exactly as `main_epidemic`. `bgn` →
//! `crate::des::shared::precision::bgn`; `fs`/`path` → `std::fs`; the JSONL event
//! log uses `crate::des::observability::logger::JsonlLogger`.
//!
//! PORT NOTES:
//!   * TS instruments each station by REPLACING `takeItem` at runtime to record
//!     transitions. Rust has no method monkeypatching, so transitions are
//!     reconstructed by a per-step membership scan: when a tracked entity first
//!     appears in a new processor station, a `prev -> cur` transition is
//!     recorded (decisions are transparent); when a tracked entity vanishes from
//!     every station it is recorded as `prev -> main-sink`. Timing differs
//!     slightly from the arrival-time hook but yields the same SEIR-level matrix.
//!   * `console.log = () => {}` muting during the run cannot be replicated (no
//!     global `println!` override); the entities' own per-step logs remain.
//!   * `(global as any).turnOffSources` has no analog — only the per-source
//!     `turn_off_after_count` guard applies.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::des::entity_decision::probability_decision::{Branch, ProbabilityDecisionEntity};
use crate::des::entity_processing::processing::EntityProcessor;
use crate::des::entity_routing::output_routing_policy::OutputRoutingPolicy;
use crate::des::entity_sink::sink::EntitySink;
use crate::des::entity_source::source::EntitySource;
use crate::des::general::general::fisher_yates_shuffle;
use crate::des::observability::logger::{JsonValue, JsonlLogger, LogLevel};
use crate::des::observers::program_observer::ProgramObserver;
use crate::des::r#abstract::interfaces::{HasInput, HasManyOutputConnections, HasOutput};
use crate::des::r#abstract::r#abstract::{Entity, EntityObserver};
use crate::des::random_variables::rv::{RandomVariable, UniformRandomVariable};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::precision::{bgn, to_f64, Decimal};

const PHASE_1_STEPS: usize = 800;
const PHASE_2_STEPS: usize = 400;
const TURN_OFF_AFTER_COUNT: i64 = 500;
const ARRIVALS_INTERARRIVAL: [f64; 2] = [0.7, 1.3];

const COMPARTMENT_ORDER: [&str; 7] = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R"];
const PROCESSOR_IDS: [&str; 8] = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D"];
const DECISION_IDS: [&str; 3] = ["I-P-Decision", "I-S-Decision", "I-H-Decision"];

fn residence(id: &str) -> (f64, f64) {
    match id {
        "R" => (1.50, 2.50),
        "D" => (0.10, 0.30),
        "I-P-Decision" | "I-S-Decision" | "I-H-Decision" => (0.05, 0.15),
        _ => (0.20, 0.40), // S, E, I-P, I-A, I-S, I-H
    }
}

/// Decision-node membership rolls into its upstream compartment.
fn compartment_groups(c: &str) -> Vec<&'static str> {
    match c {
        "I-P" => vec!["I-P", "I-P-Decision"],
        "I-S" => vec!["I-S", "I-S-Decision"],
        "I-H" => vec!["I-H", "I-H-Decision"],
        "S" => vec!["S"],
        "E" => vec!["E"],
        "I-A" => vec!["I-A"],
        "R" => vec!["R"],
        _ => vec![],
    }
}

struct SeedGen(u32);
impl SeedGen {
    fn next(&mut self) -> Box<dyn RandomSource> {
        self.0 = self.0.wrapping_add(1);
        Box::new(SeededRandom::new(self.0))
    }
}

fn uni(id: &str, seeds: &mut SeedGen) -> Box<dyn RandomVariable> {
    let (a, b) = residence(id);
    Box::new(UniformRandomVariable::new(bgn(a), bgn(b), seeds.next()))
}

#[derive(Clone)]
enum Stn {
    Source(Rc<RefCell<EntitySource>>),
    Proc(Rc<RefCell<EntityProcessor>>),
    Decision(Rc<RefCell<ProbabilityDecisionEntity>>),
    Sink(Rc<RefCell<EntitySink>>),
}

impl Stn {
    fn as_has_input(&self) -> Rc<RefCell<dyn HasInput>> {
        match self {
            Stn::Proc(p) => p.clone(),
            Stn::Decision(d) => d.clone(),
            Stn::Sink(s) => s.clone(),
            Stn::Source(_) => panic!("source has no input"),
        }
    }
    fn as_has_many_out(&self) -> Rc<RefCell<dyn HasManyOutputConnections>> {
        match self {
            Stn::Source(s) => s.clone(),
            Stn::Proc(p) => p.clone(),
            Stn::Decision(d) => d.clone(),
            Stn::Sink(_) => panic!("sink has no output"),
        }
    }
    fn add_out_connection(&self, target: Rc<RefCell<dyn HasInput>>) {
        match self {
            Stn::Source(s) => { s.borrow_mut().add_out_connection(target); }
            Stn::Proc(p) => { p.borrow_mut().add_out_connection(target); }
            Stn::Decision(d) => { d.borrow_mut().add_out_connection(target); }
            Stn::Sink(_) => {}
        }
    }
    fn add_in_connection(&self, source: Rc<RefCell<dyn HasManyOutputConnections>>) {
        match self {
            Stn::Proc(p) => { p.borrow_mut().add_in_connection(source); }
            Stn::Decision(d) => { d.borrow_mut().add_in_connection(source); }
            Stn::Sink(s) => { s.borrow_mut().add_in_connection(source); }
            Stn::Source(_) => {}
        }
    }
    fn setup(&self) {
        match self {
            Stn::Source(s) => {
                HasOutput::do_setup_after_output_conn(&mut *s.borrow_mut());
                HasOutput::do_setup_after_input_conn(&mut *s.borrow_mut());
            }
            Stn::Proc(p) => {
                HasOutput::do_setup_after_output_conn(&mut *p.borrow_mut());
                HasOutput::do_setup_after_input_conn(&mut *p.borrow_mut());
            }
            Stn::Decision(d) => {
                HasOutput::do_setup_after_output_conn(&mut *d.borrow_mut());
                HasOutput::do_setup_after_input_conn(&mut *d.borrow_mut());
            }
            Stn::Sink(s) => {
                HasInput::do_setup_after_output_conn(&mut *s.borrow_mut());
                HasInput::do_setup_after_input_conn(&mut *s.borrow_mut());
            }
        }
    }
    fn do_time_step(&self, step: Decimal) {
        match self {
            Stn::Source(s) => s.borrow_mut().do_time_step(step),
            Stn::Proc(p) => p.borrow_mut().do_time_step(step),
            Stn::Decision(d) => d.borrow_mut().do_time_step(step),
            Stn::Sink(s) => s.borrow_mut().do_time_step(step),
        }
    }
}

fn jnum(n: f64) -> JsonValue {
    JsonValue::Number(n)
}
fn jstr(s: &str) -> JsonValue {
    JsonValue::String(s.to_string())
}
fn jobj(v: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(v.into_iter().map(|(k, val)| (k.to_string(), val)).collect())
}

/// Entry point (TS top-level `run()` closure + invocation).
pub fn run() {
    let step_size = bgn(1.0);
    let obs_concrete = Rc::new(RefCell::new(ProgramObserver::new()));
    let obs: Rc<RefCell<dyn EntityObserver>> = obs_concrete.clone();
    let mut seeds = SeedGen(1);

    // Branch probabilities.
    let asymptomatic_share = bgn(0.40);
    let hospitalization_given_symptom = bgn(0.20);
    let case_fatality_given_hospital = bgn(0.12);

    // Concrete handles (also used for population sampling / membership scans).
    let source = Rc::new(RefCell::new(EntitySource::new(
        "main-source".to_string(),
        Box::new(UniformRandomVariable::new(
            bgn(ARRIVALS_INTERARRIVAL[0]),
            bgn(ARRIVALS_INTERARRIVAL[1]),
            seeds.next(),
        )),
        TURN_OFF_AFTER_COUNT,
    )));
    source.borrow_mut().subscribe(obs.clone());

    let mut procs: HashMap<String, Rc<RefCell<EntityProcessor>>> = HashMap::new();
    for id in PROCESSOR_IDS {
        let p = Rc::new(RefCell::new(EntityProcessor::new(
            id.to_string(),
            uni(id, &mut seeds),
            OutputRoutingPolicy::default(),
        )));
        p.borrow_mut().subscribe(obs.clone());
        procs.insert(id.to_string(), p);
    }

    let mut decisions: HashMap<String, Rc<RefCell<ProbabilityDecisionEntity>>> = HashMap::new();
    let decision_branches: [(&str, [Decimal; 2]); 3] = [
        ("I-P-Decision", [asymptomatic_share, bgn(1.0) - asymptomatic_share]),
        ("I-S-Decision", [bgn(1.0) - hospitalization_given_symptom, hospitalization_given_symptom]),
        ("I-H-Decision", [bgn(1.0) - case_fatality_given_hospital, case_fatality_given_hospital]),
    ];
    for (id, probs) in decision_branches {
        let d = Rc::new(RefCell::new(ProbabilityDecisionEntity::new(
            id.to_string(),
            vec![Branch { index: 0, prob: probs[0] }, Branch { index: 1, prob: probs[1] }],
            uni(id, &mut seeds),
            seeds.next(),
        )));
        d.borrow_mut().subscribe(obs.clone());
        decisions.insert(id.to_string(), d);
    }

    let sink = Rc::new(RefCell::new(EntitySink::new("main-sink".to_string())));
    sink.borrow_mut().subscribe(obs.clone());

    // Ordered station table (matches the TS Map literal order).
    let order: Vec<&str> = vec![
        "main-source", "S", "E", "I-P", "I-P-Decision", "I-A", "I-S", "I-S-Decision", "I-H",
        "I-H-Decision", "R", "D", "main-sink",
    ];
    let node_of = |id: &str| -> Stn {
        match id {
            "main-source" => Stn::Source(source.clone()),
            "main-sink" => Stn::Sink(sink.clone()),
            _ if procs.contains_key(id) => Stn::Proc(procs[id].clone()),
            _ => Stn::Decision(decisions[id].clone()),
        }
    };

    let edges: [(&str, &str); 15] = [
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
    for (source_id, target_id) in edges {
        let s = node_of(source_id);
        let t = node_of(target_id);
        s.add_out_connection(t.as_has_input());
        t.add_in_connection(s.as_has_many_out());
    }
    for id in order.iter().copied() {
        node_of(id).setup();
    }

    // --- Observability logger -------------------------------------------------
    let _ = std::fs::create_dir_all("out");
    let event_log_path = "out/epidemic-events.jsonl";
    let mut logger = JsonlLogger::new(event_log_path, LogLevel::Info);

    let edges_json: Vec<JsonValue> =
        edges.iter().map(|(a, b)| JsonValue::Array(vec![jstr(a), jstr(b)])).collect();
    logger.log(jobj(vec![
        ("kind", jstr("sim_start")),
        (
            "config",
            jobj(vec![
                ("stepSize", jnum(to_f64(step_size))),
                ("phase1Steps", jnum(PHASE_1_STEPS as f64)),
                ("phase2Steps", jnum(PHASE_2_STEPS as f64)),
                ("sourceCap", jnum(TURN_OFF_AFTER_COUNT as f64)),
                (
                    "arrivalsInterarrival",
                    JsonValue::Array(vec![jnum(ARRIVALS_INTERARRIVAL[0]), jnum(ARRIVALS_INTERARRIVAL[1])]),
                ),
                (
                    "probabilities",
                    jobj(vec![
                        ("asymptomaticShare", jnum(to_f64(asymptomatic_share))),
                        ("hospitalizationGivenSymptom", jnum(to_f64(hospitalization_given_symptom))),
                        ("caseFatalityGivenHospital", jnum(to_f64(case_fatality_given_hospital))),
                    ]),
                ),
                ("edges", JsonValue::Array(edges_json)),
            ]),
        ),
    ]));

    // --- Sampling + transition reconstruction state ---------------------------
    let station_population = |id: &str| -> usize {
        if let Some(p) = procs.get(id) {
            let p = p.borrow();
            p.base.queue.len() + p.processing_queue.len() + p.out_queue.len()
        } else if let Some(d) = decisions.get(id) {
            d.borrow().queue.len()
        } else {
            0
        }
    };
    let compartment_population =
        |c: &str| -> usize { compartment_groups(c).into_iter().map(|sid| station_population(sid)).sum() };
    let cumulative_deaths = || -> i64 { sink.borrow().destroyed_count };

    let mut trajectory: Vec<HashMap<String, f64>> = Vec::new();
    let mut transition_count: HashMap<String, HashMap<String, u64>> = HashMap::new();
    let mut last_processor: HashMap<String, String> = HashMap::new();
    let mut absorbed: HashSet<String> = HashSet::new();

    let program_list: Vec<Stn> = order.iter().copied().map(|id| node_of(id)).collect();
    let mut list = program_list.clone();
    let mut rng = SeededRandom::new(0x00BEEF02);

    let sample = |trajectory: &mut Vec<HashMap<String, f64>>,
                  logger: &mut JsonlLogger,
                  t: usize,
                  sources_active: bool| {
        let mut row: HashMap<String, f64> = HashMap::new();
        row.insert("t".into(), t as f64);
        let mut total_alive = 0usize;
        let mut populations: Vec<(&str, JsonValue)> = Vec::new();
        for c in COMPARTMENT_ORDER {
            let n = compartment_population(c);
            row.insert(c.to_string(), n as f64);
            populations.push((c, jnum(n as f64)));
            total_alive += n;
        }
        let d_cum = cumulative_deaths();
        row.insert("D_cum".into(), d_cum as f64);
        row.insert("alive".into(), total_alive as f64);
        trajectory.push(row);
        logger.log(jobj(vec![
            ("kind", jstr("tick")),
            ("t", jnum(t as f64)),
            ("populations", JsonValue::Object(populations.into_iter().map(|(k, v)| (k.to_string(), v)).collect())),
            ("cumD", jnum(d_cum as f64)),
            ("alive", jnum(total_alive as f64)),
            ("sourcesActive", JsonValue::Bool(sources_active)),
        ]));
    };

    // Membership scan reconstructing transitions (see PORT NOTE).
    let scan = |transition_count: &mut HashMap<String, HashMap<String, u64>>,
                last_processor: &mut HashMap<String, String>,
                absorbed: &mut HashSet<String>| {
        let record = |tc: &mut HashMap<String, HashMap<String, u64>>, from: &str, to: &str| {
            *tc.entry(from.to_string()).or_default().entry(to.to_string()).or_insert(0) += 1;
        };
        let mut present_proc: HashMap<String, String> = HashMap::new();
        let mut present_any: HashSet<String> = HashSet::new();
        for pid in PROCESSOR_IDS {
            let p = procs[pid].borrow();
            for q in [&p.base.queue, &p.processing_queue, &p.out_queue] {
                for e in q.iter() {
                    let eid = e.borrow().id();
                    present_proc.insert(eid.clone(), pid.to_string());
                    present_any.insert(eid);
                }
            }
        }
        for did in DECISION_IDS {
            let d = decisions[did].borrow();
            for e in d.queue.iter() {
                present_any.insert(e.borrow().id());
            }
        }
        for (eid, cur) in &present_proc {
            if last_processor.get(eid) != Some(cur) {
                let prev = last_processor.get(eid).cloned().unwrap_or_else(|| "__source__".to_string());
                record(transition_count, &prev, cur);
                last_processor.insert(eid.clone(), cur.clone());
            }
        }
        let tracked: Vec<String> = last_processor.keys().cloned().collect();
        for eid in tracked {
            if !present_any.contains(&eid) && !absorbed.contains(&eid) {
                let prev = last_processor.get(&eid).cloned().unwrap();
                record(transition_count, &prev, "main-sink");
                absorbed.insert(eid);
            }
        }
    };

    let started_at = std::time::Instant::now();
    let mut current_step;

    for i in 0..PHASE_1_STEPS {
        current_step = i + 1;
        fisher_yates_shuffle(&mut list, &mut rng);
        for v in &list {
            v.do_time_step(step_size);
        }
        scan(&mut transition_count, &mut last_processor, &mut absorbed);
        sample(&mut trajectory, &mut logger, current_step, true);
    }

    // PORT NOTE: turnOffSources global omitted; source self-quiesces at the cap.
    current_step = PHASE_1_STEPS;
    logger.log(jobj(vec![
        ("kind", jstr("phase_change")),
        ("t", jnum(current_step as f64)),
        ("phase", jstr("drain")),
    ]));

    for i in 0..PHASE_2_STEPS {
        current_step = PHASE_1_STEPS + i + 1;
        fisher_yates_shuffle(&mut list, &mut rng);
        for v in &list {
            v.do_time_step(step_size);
        }
        scan(&mut transition_count, &mut last_processor, &mut absorbed);
        sample(&mut trajectory, &mut logger, current_step, false);
    }

    let elapsed = started_at.elapsed().as_millis();

    // --- Reports --------------------------------------------------------------
    println!();
    println!("=== epidemic simulator (improved) ==========================");
    println!(
        "steps run: {} ({} arriving, {} draining)",
        PHASE_1_STEPS + PHASE_2_STEPS,
        PHASE_1_STEPS,
        PHASE_2_STEPS
    );
    println!("wall time: {elapsed} ms");
    println!("total entities created: {}", source.borrow().created_count);
    println!("cumulative deaths absorbed by sink: {}", cumulative_deaths());
    println!();

    let final_row = trajectory.last().cloned().unwrap_or_default();
    println!("--- final compartment populations ---");
    for c in COMPARTMENT_ORDER {
        println!("  {:<4}: {}", c, final_row.get(c).copied().unwrap_or(0.0) as i64);
    }
    println!("  D (cum): {}", final_row.get("D_cum").copied().unwrap_or(0.0) as i64);
    println!();

    let matrix_rows = ["__source__", "S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D"];
    let matrix_cols = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D", "main-sink"];
    let count = |r: &str, c: &str| -> u64 { transition_count.get(r).and_then(|row| row.get(c)).copied().unwrap_or(0) };
    let row_sum = |r: &str| -> u64 { transition_count.get(r).map(|row| row.values().sum()).unwrap_or(0) };

    let mut header = format!("{:<11}", "from \\ to");
    for c in matrix_cols {
        header.push_str(&format!("{c:<11}"));
    }
    header.push_str(&format!("{:<11}", "sum"));

    println!("--- empirical transition counts ---");
    println!("{header}");
    for r in matrix_rows {
        let mut line = format!("{r:<11}");
        for c in matrix_cols {
            line.push_str(&format!("{:<11}", count(r, c)));
        }
        line.push_str(&format!("{:<11}", row_sum(r)));
        println!("{line}");
    }
    println!();

    println!("--- empirical transition probabilities (row-stochastic) ---");
    println!("{header}");
    for r in matrix_rows {
        let total = row_sum(r);
        let mut line = format!("{r:<11}");
        for c in matrix_cols {
            let v = count(r, c);
            let cell = if total > 0 {
                let p = v as f64 / total as f64;
                if p == 0.0 { ".".to_string() } else { format!("{p:.3}") }
            } else {
                ".".to_string()
            };
            line.push_str(&format!("{cell:<11}"));
        }
        line.push_str(&format!("{:<11}", if total > 0 { "1.000" } else { "." }));
        println!("{line}");
    }
    println!();

    // --- Persist artifacts ----------------------------------------------------
    let csv_path = "out/epidemic-trajectory.csv";
    let mut cols: Vec<String> = vec!["t".to_string()];
    cols.extend(COMPARTMENT_ORDER.iter().map(|s| s.to_string()));
    cols.push("D_cum".to_string());
    cols.push("alive".to_string());
    let mut csv = cols.join(",");
    csv.push('\n');
    for r in &trajectory {
        let line: Vec<String> =
            cols.iter().map(|c| (r.get(c).copied().unwrap_or(0.0) as i64).to_string()).collect();
        csv.push_str(&line.join(","));
        csv.push('\n');
    }
    let _ = std::fs::write(csv_path, csv);

    let matrix_path = "out/epidemic-transition-matrix.json";
    let mut serial: Vec<(String, JsonValue)> = Vec::new();
    for r in matrix_rows {
        let total = row_sum(r);
        let mut row_entries: Vec<(String, JsonValue)> = Vec::new();
        for c in matrix_cols {
            let v = count(r, c);
            let p = if total > 0 { (v as f64 / total as f64 * 1e6).round() / 1e6 } else { 0.0 };
            row_entries.push((c.to_string(), jnum(p)));
        }
        serial.push((r.to_string(), JsonValue::Object(row_entries)));
    }
    let _ = std::fs::write(matrix_path, JsonValue::Object(serial).to_string_pretty(2));

    logger.log(jobj(vec![
        ("kind", jstr("sim_end")),
        ("t", jnum(current_step as f64)),
        ("elapsedMs", jnum(elapsed as f64)),
        (
            "totals",
            jobj(vec![
                ("created", jnum(source.borrow().created_count as f64)),
                ("absorbed", jnum(cumulative_deaths() as f64)),
                (
                    "finalPopulations",
                    JsonValue::Object(
                        COMPARTMENT_ORDER
                            .iter()
                            .map(|c| (c.to_string(), jnum(final_row.get(*c).copied().unwrap_or(0.0))))
                            .collect(),
                    ),
                ),
            ]),
        ),
    ]));
    logger.close();

    println!("artifacts written:");
    println!("  {csv_path}");
    println!("  {matrix_path}");
    let kind_counts = logger
        .get_kind_counts()
        .into_iter()
        .map(|(k, n)| format!("{k}={n}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("  {}  ({} events: {})", event_log_path, logger.get_event_count(), kind_counts);
    println!("============================================================");
}
