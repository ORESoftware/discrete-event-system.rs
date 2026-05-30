//! Port of `src/des/reference/main-epidemic-fel.ts`.
//!
//! Classical Future-Event-List (FEL) reference implementation of the SEIR-with-
//! hospitalization epidemic model. Where the framework runs every station's
//! local logic on a fixed time step, a classical DES kernel maintains a
//! priority queue of future events, pops the next-soonest, advances the clock
//! to it, and runs only the affected handler. Different scheduling algorithm,
//! same model — so the statistics should match modulo Monte-Carlo noise, which
//! makes this both a sanity check and a comparison target for the validator.
//!
//! The framework semantic this replicates: a station's random variable controls
//! one global service clock for that station (NOT a per-individual residence
//! draw); when the clock fires, the head of that station's queue (if any) is
//! routed downstream and the clock is rescheduled.
//!
//! Migration notes: the TS file is an entry script (`run()` at EOF) that writes
//! CSV/JSON/JSONL artifacts; the logic lives in [`run`], which takes an output
//! directory and an injected `RandomSource` (so it matches the framework's
//! seeded runs) instead of reaching for `Math.random()`. No `fn main` is added.
//! The FEL is a stable sorted `Vec` (insert after all events with time <= t,
//! pop from the front) rather than a `BinaryHeap`, deliberately, so the
//! tie-order — and therefore the exact sequence of RNG draws — matches the TS.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::des::observability::logger::{JsonValue, JsonlLogger, LogLevel};
use crate::des::shared::capabilities::RandomSource;

// --- Config (intentionally identical to main-epidemic-improved.ts) ----------

const RESIDENCE: [(&str, (f64, f64)); 11] = [
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
const ARRIVALS_INTERARRIVAL: (f64, f64) = (0.7, 1.3);
const TURN_OFF_AFTER_COUNT: u64 = 500;

const PROBS_ASYMPTOMATIC_SHARE: f64 = 0.40;
const PROBS_HOSPITALIZATION_GIVEN_SYMPTOM: f64 = 0.20;
const PROBS_CASE_FATALITY_GIVEN_HOSPITAL: f64 = 0.12;

const T_PHASE_1: i64 = 800; // sources active
const T_MAX: i64 = 1200; // total horizon

const COMPARTMENT_ORDER: [&str; 7] = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R"];

const EDGES: [(&str, &str); 15] = [
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

/// One successor branch. Probabilities within a row sum to 1.
#[derive(Clone, Copy)]
struct Branch {
    prob: f64,
    to: &'static str,
}

/// Graph as a successor map (ordered to preserve key-iteration order, which the
/// initial event scheduling — and thus RNG draw order — depends on).
fn successors_table() -> Vec<(&'static str, Vec<Branch>)> {
    vec![
        ("main-source", vec![Branch { prob: 1.0, to: "S" }]),
        ("S", vec![Branch { prob: 1.0, to: "E" }]),
        (
            "E",
            vec![Branch {
                prob: 1.0,
                to: "I-P",
            }],
        ),
        (
            "I-P",
            vec![Branch {
                prob: 1.0,
                to: "I-P-Decision",
            }],
        ),
        (
            "I-P-Decision",
            vec![
                Branch {
                    prob: PROBS_ASYMPTOMATIC_SHARE,
                    to: "I-A",
                },
                Branch {
                    prob: 1.0 - PROBS_ASYMPTOMATIC_SHARE,
                    to: "I-S",
                },
            ],
        ),
        ("I-A", vec![Branch { prob: 1.0, to: "R" }]),
        (
            "I-S",
            vec![Branch {
                prob: 1.0,
                to: "I-S-Decision",
            }],
        ),
        (
            "I-S-Decision",
            vec![
                Branch {
                    prob: 1.0 - PROBS_HOSPITALIZATION_GIVEN_SYMPTOM,
                    to: "R",
                },
                Branch {
                    prob: PROBS_HOSPITALIZATION_GIVEN_SYMPTOM,
                    to: "I-H",
                },
            ],
        ),
        (
            "I-H",
            vec![Branch {
                prob: 1.0,
                to: "I-H-Decision",
            }],
        ),
        (
            "I-H-Decision",
            vec![
                Branch {
                    prob: 1.0 - PROBS_CASE_FATALITY_GIVEN_HOSPITAL,
                    to: "R",
                },
                Branch {
                    prob: PROBS_CASE_FATALITY_GIVEN_HOSPITAL,
                    to: "D",
                },
            ],
        ),
        ("R", vec![Branch { prob: 1.0, to: "S" }]),
        (
            "D",
            vec![Branch {
                prob: 1.0,
                to: "main-sink",
            }],
        ),
    ]
}

/// `COMPARTMENT_GROUPS[c]` — the station ids that roll up into compartment `c`.
fn compartment_groups(c: &str) -> &'static [&'static str] {
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

fn residence_of(station: &str) -> (f64, f64) {
    for entry in RESIDENCE {
        if entry.0 == station {
            return entry.1;
        }
    }
    panic!("no residence interval for station '{station}'");
}

// --- Random helpers ---------------------------------------------------------

fn draw_uniform(a: f64, b: f64, rng: &mut dyn RandomSource) -> f64 {
    a + rng.next_float() * (b - a)
}

fn draw_successor(
    from: &str,
    table: &[(&'static str, Vec<Branch>)],
    rng: &mut dyn RandomSource,
) -> &'static str {
    let succs = match table.iter().find(|entry| entry.0 == from) {
        Some(e) => &e.1,
        None => panic!("no successors for station '{from}'"),
    };
    if succs.len() == 1 {
        return succs[0].to;
    }
    let r = rng.next_float();
    let mut cum = 0.0;
    for sb in succs {
        cum += sb.prob;
        if r < cum {
            return sb.to;
        }
    }
    succs[succs.len() - 1].to
}

// --- Future event list (stable sorted-array PQ; fine for ~12 stations) ------

struct FelEvent {
    time: f64,
    station: &'static str,
}

fn insert_event(fel: &mut Vec<FelEvent>, e: FelEvent) {
    // Linear scan is O(n) but n <= number-of-stations (== 12) here; insert
    // after all events with time <= e.time so equal-time ties stay FIFO.
    let mut i = 0;
    while i < fel.len() && fel[i].time <= e.time {
        i += 1;
    }
    fel.insert(i, e);
}

fn pop_event(fel: &mut Vec<FelEvent>) -> Option<FelEvent> {
    if fel.is_empty() {
        None
    } else {
        Some(fel.remove(0))
    }
}

// --- JSON construction helpers ----------------------------------------------

fn s(v: &str) -> JsonValue {
    JsonValue::String(v.to_string())
}

fn num(v: f64) -> JsonValue {
    JsonValue::Number(v)
}

fn arr(items: Vec<JsonValue>) -> JsonValue {
    JsonValue::Array(items)
}

fn obj(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

/// JS `String(number)`: shortest round-trip for finite values, with `Infinity`
/// / `-Infinity` / `NaN` spelled out as JS would.
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

/// `+(x).toFixed(d)` — round to `d` decimals, then back to a number.
fn to_fixed_num(x: f64, d: i32) -> f64 {
    let f = 10f64.powi(d);
    (x * f).round() / f
}

fn residence_json() -> JsonValue {
    let entries: Vec<(String, JsonValue)> = RESIDENCE
        .iter()
        .map(|entry| {
            let (a, b) = entry.1;
            (entry.0.to_string(), JsonValue::Array(vec![num(a), num(b)]))
        })
        .collect();
    JsonValue::Object(entries)
}

fn edges_json() -> JsonValue {
    JsonValue::Array(
        EDGES
            .iter()
            .map(|(a, b)| JsonValue::Array(vec![s(a), s(b)]))
            .collect(),
    )
}

// --- Event handlers (own their mutable state explicitly; the TS used closures
//     over module-level mutable bindings, which Rust does not allow) ----------

fn record_transition(
    tc: &mut HashMap<&'static str, HashMap<&'static str, u64>>,
    from: &'static str,
    to: &'static str,
) {
    *tc.entry(from).or_default().entry(to).or_insert(0) += 1;
}

#[allow(clippy::too_many_arguments)]
fn arrive(
    t: f64,
    entity_id: &str,
    station: &'static str,
    queues: &mut HashMap<&'static str, VecDeque<String>>,
    last_station: &mut HashMap<String, &'static str>,
    transition_count: &mut HashMap<&'static str, HashMap<&'static str, u64>>,
    absorbed: &mut u64,
    logger: &mut JsonlLogger,
) {
    if station == "main-sink" {
        let prev = *last_station.get(entity_id).unwrap_or(&"__source__");
        record_transition(transition_count, prev, "main-sink");
        logger.log(obj(vec![
            ("kind", s("transition")),
            ("t", num(t)),
            ("entity", s(entity_id)),
            ("from", s(prev)),
            ("to", s("main-sink")),
        ]));
        *absorbed += 1;
        return;
    }
    if !station.ends_with("-Decision") {
        let prev = *last_station.get(entity_id).unwrap_or(&"__source__");
        record_transition(transition_count, prev, station);
        logger.log(obj(vec![
            ("kind", s("transition")),
            ("t", num(t)),
            ("entity", s(entity_id)),
            ("from", s(prev)),
            ("to", s(station)),
        ]));
        last_station.insert(entity_id.to_string(), station);
    }
    queues
        .get_mut(station)
        .expect("queue for station exists")
        .push_back(entity_id.to_string());
}

fn sample_at(
    t: i64,
    queues: &HashMap<&'static str, VecDeque<String>>,
    absorbed: u64,
    phase2: bool,
    trajectory: &mut Vec<HashMap<&'static str, f64>>,
    logger: &mut JsonlLogger,
) {
    let mut row: HashMap<&'static str, f64> = HashMap::new();
    row.insert("t", t as f64);
    let mut total_alive = 0.0_f64;
    let mut populations: Vec<(String, JsonValue)> = Vec::new();
    for c in COMPARTMENT_ORDER {
        let count: usize = compartment_groups(c)
            .iter()
            .map(|sid| queues.get(sid).map(|q| q.len()).unwrap_or(0))
            .sum();
        row.insert(c, count as f64);
        populations.push((c.to_string(), JsonValue::Number(count as f64)));
        total_alive += count as f64;
    }
    row.insert("D_cum", absorbed as f64);
    row.insert("alive", total_alive);
    trajectory.push(row);
    logger.log(JsonValue::Object(vec![
        ("kind".to_string(), JsonValue::String("tick".to_string())),
        ("t".to_string(), JsonValue::Number(t as f64)),
        ("populations".to_string(), JsonValue::Object(populations)),
        ("cumD".to_string(), JsonValue::Number(absorbed as f64)),
        ("alive".to_string(), JsonValue::Number(total_alive)),
        ("sourcesActive".to_string(), JsonValue::Bool(!phase2)),
    ]));
}

/// Default output directory (repository-root `out/`).
pub fn default_out_dir() -> std::path::PathBuf {
    std::path::PathBuf::from("out")
}

/// Run the FEL reference simulation, writing artifacts into `out_dir` and using
/// the injected `rng` for all stochastic choices.
pub fn run(out_dir: &Path, rng: &mut dyn RandomSource) {
    fs::create_dir_all(out_dir).unwrap_or_else(|e| panic!("cannot create out dir: {e}"));
    let event_log_path = out_dir.join("epidemic-events-fel.jsonl");
    let mut logger = JsonlLogger::new(
        event_log_path.to_str().expect("utf8 event-log path"),
        LogLevel::Info,
    );

    logger.log(obj(vec![
        ("kind", s("sim_start")),
        (
            "config",
            obj(vec![
                ("kernel", s("fel-reference")),
                ("tPhase1", num(T_PHASE_1 as f64)),
                ("tMax", num(T_MAX as f64)),
                ("sourceCap", num(TURN_OFF_AFTER_COUNT as f64)),
                (
                    "arrivalsInterarrival",
                    arr(vec![
                        num(ARRIVALS_INTERARRIVAL.0),
                        num(ARRIVALS_INTERARRIVAL.1),
                    ]),
                ),
                ("residence", residence_json()),
                (
                    "probabilities",
                    obj(vec![
                        ("asymptomaticShare", num(PROBS_ASYMPTOMATIC_SHARE)),
                        (
                            "hospitalizationGivenSymptom",
                            num(PROBS_HOSPITALIZATION_GIVEN_SYMPTOM),
                        ),
                        (
                            "caseFatalityGivenHospital",
                            num(PROBS_CASE_FATALITY_GIVEN_HOSPITAL),
                        ),
                    ]),
                ),
                ("edges", edges_json()),
            ]),
        ),
    ]));

    let succs_table = successors_table();

    // Per-station FIFO queue of entity IDs.
    let mut queues: HashMap<&'static str, VecDeque<String>> = HashMap::new();
    for entry in &succs_table {
        queues.insert(entry.0, VecDeque::new());
    }
    queues.insert("main-sink", VecDeque::new()); // keeps lookups uniform

    // Last non-decision station an entity was at, for transition logging.
    let mut last_station: HashMap<String, &'static str> = HashMap::new();

    let mut source_created: u64 = 0;
    let mut absorbed: u64 = 0;
    let mut next_entity_id: u64 = 0;
    let mut phase2 = false;

    // Trajectory snapshots at integer times t = 1..T_MAX.
    let mut trajectory: Vec<HashMap<&'static str, f64>> = Vec::new();
    let mut next_sample_at: i64 = 1;

    let mut transition_count: HashMap<&'static str, HashMap<&'static str, u64>> = HashMap::new();

    let mut fel: Vec<FelEvent> = Vec::new();

    // Schedule one initial service event per station (and the source).
    for entry in &succs_table {
        let station = entry.0;
        let (a, b) = if station == "main-source" {
            ARRIVALS_INTERARRIVAL
        } else {
            residence_of(station)
        };
        let time = draw_uniform(a, b, rng);
        insert_event(&mut fel, FelEvent { time, station });
    }

    let started_at = Instant::now();

    loop {
        let e = match pop_event(&mut fel) {
            Some(e) => e,
            None => break,
        };
        if e.time > T_MAX as f64 {
            break;
        }

        // Sample any integer ticks we passed since the last event.
        let floor_t = e.time.floor() as i64;
        while next_sample_at <= floor_t && next_sample_at <= T_MAX {
            sample_at(
                next_sample_at,
                &queues,
                absorbed,
                phase2,
                &mut trajectory,
                &mut logger,
            );
            next_sample_at += 1;
        }

        // Phase change: turn off source past T_PHASE_1.
        if !phase2 && e.time >= T_PHASE_1 as f64 {
            phase2 = true;
            logger.log(obj(vec![
                ("kind", s("phase_change")),
                ("t", num(e.time.floor())),
                ("phase", s("drain")),
            ]));
        }

        if e.station == "main-source" {
            if source_created < TURN_OFF_AFTER_COUNT && !phase2 {
                let id = format!("f{next_entity_id}");
                next_entity_id += 1;
                arrive(
                    e.time,
                    &id,
                    "S",
                    &mut queues,
                    &mut last_station,
                    &mut transition_count,
                    &mut absorbed,
                    &mut logger,
                );
                source_created += 1;
            }
            let (a, b) = ARRIVALS_INTERARRIVAL;
            let nt = e.time + draw_uniform(a, b, rng);
            insert_event(
                &mut fel,
                FelEvent {
                    time: nt,
                    station: "main-source",
                },
            );
        } else {
            // Service one entity at this station's queue head, if any.
            let head = queues.get_mut(e.station).and_then(|q| q.pop_front());
            if let Some(head) = head {
                let dest = draw_successor(e.station, &succs_table, rng);
                arrive(
                    e.time,
                    &head,
                    dest,
                    &mut queues,
                    &mut last_station,
                    &mut transition_count,
                    &mut absorbed,
                    &mut logger,
                );
            }
            let (a, b) = residence_of(e.station);
            let nt = e.time + draw_uniform(a, b, rng);
            insert_event(
                &mut fel,
                FelEvent {
                    time: nt,
                    station: e.station,
                },
            );
        }
    }

    // Flush any remaining ticks up to T_MAX.
    while next_sample_at <= T_MAX {
        sample_at(
            next_sample_at,
            &queues,
            absorbed,
            phase2,
            &mut trajectory,
            &mut logger,
        );
        next_sample_at += 1;
    }

    let elapsed = started_at.elapsed().as_millis();

    // ---- Reports -----------------------------------------------------------
    println!();
    println!("=== epidemic simulator (FEL reference) =====================");
    println!(
        "horizon: {} ({} arriving, {} draining)",
        T_MAX,
        T_PHASE_1,
        T_MAX - T_PHASE_1
    );
    println!("wall time: {elapsed} ms");
    println!("total entities created: {source_created}");
    println!("cumulative deaths absorbed by sink: {absorbed}");
    println!();

    let final_row = trajectory.last().cloned().unwrap_or_default();
    println!("--- final compartment populations ---");
    for c in COMPARTMENT_ORDER {
        println!("  {:<4}: {}", c, js_num(*final_row.get(c).unwrap_or(&0.0)));
    }
    println!(
        "  D (cum): {}",
        js_num(*final_row.get("D_cum").unwrap_or(&0.0))
    );
    println!();

    // ---- Persist artifacts --------------------------------------------------
    let csv_path = out_dir.join("epidemic-trajectory-fel.csv");
    let mut cols: Vec<&'static str> = vec!["t"];
    cols.extend_from_slice(&COMPARTMENT_ORDER);
    cols.push("D_cum");
    cols.push("alive");
    let mut csv = String::new();
    csv.push_str(&cols.join(","));
    csv.push('\n');
    for r in &trajectory {
        let line: Vec<String> = cols
            .iter()
            .map(|c| js_num(*r.get(*c).unwrap_or(&0.0)))
            .collect();
        csv.push_str(&line.join(","));
        csv.push('\n');
    }
    fs::write(&csv_path, csv).unwrap_or_else(|e| panic!("cannot write csv: {e}"));

    let matrix_path = out_dir.join("epidemic-transition-matrix-fel.json");
    let matrix_rows = ["__source__", "S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D"];
    let matrix_cols = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D", "main-sink"];
    let mut serial: Vec<(String, JsonValue)> = Vec::new();
    for r in matrix_rows {
        let row = transition_count.get(r);
        let mut total = 0u64;
        if let Some(rr) = row {
            for v in rr.values() {
                total += *v;
            }
        }
        let mut inner: Vec<(String, JsonValue)> = Vec::new();
        for c in matrix_cols {
            let v = row.and_then(|rr| rr.get(c)).copied().unwrap_or(0);
            let val = if total > 0 {
                to_fixed_num(v as f64 / total as f64, 6)
            } else {
                0.0
            };
            inner.push((c.to_string(), JsonValue::Number(val)));
        }
        serial.push((r.to_string(), JsonValue::Object(inner)));
    }
    let serial_matrix = JsonValue::Object(serial);
    fs::write(&matrix_path, serial_matrix.to_string_pretty(2))
        .unwrap_or_else(|e| panic!("cannot write matrix: {e}"));

    let final_populations: Vec<(String, JsonValue)> = COMPARTMENT_ORDER
        .iter()
        .map(|c| {
            (
                c.to_string(),
                JsonValue::Number(*final_row.get(*c).unwrap_or(&0.0)),
            )
        })
        .collect();
    let transition_counts: Vec<(String, JsonValue)> = matrix_rows
        .iter()
        .map(|r| {
            let inner: Vec<(String, JsonValue)> = transition_count
                .get(*r)
                .map(|rr| {
                    rr.iter()
                        .map(|(k, v)| (k.to_string(), JsonValue::Number(*v as f64)))
                        .collect()
                })
                .unwrap_or_default();
            (r.to_string(), JsonValue::Object(inner))
        })
        .collect();

    logger.log(obj(vec![
        ("kind", s("sim_end")),
        ("t", num(T_MAX as f64)),
        ("elapsedMs", num(elapsed as f64)),
        (
            "totals",
            JsonValue::Object(vec![
                (
                    "created".to_string(),
                    JsonValue::Number(source_created as f64),
                ),
                ("absorbed".to_string(), JsonValue::Number(absorbed as f64)),
                (
                    "finalPopulations".to_string(),
                    JsonValue::Object(final_populations),
                ),
                (
                    "transitionCounts".to_string(),
                    JsonValue::Object(transition_counts),
                ),
            ]),
        ),
    ]));
    logger.close();

    println!("artifacts written:");
    println!("  {}", csv_path.display());
    println!("  {}", matrix_path.display());
    println!("  {}", event_log_path.display());
    println!("============================================================");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::observability::logger::read_events;
    use crate::des::shared::capabilities::SeededRandom;

    #[test]
    fn run_produces_consistent_artifacts() {
        let dir = std::env::temp_dir().join(format!("des_fel_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut rng = SeededRandom::new(12345);
        run(&dir, &mut rng);

        let log = dir.join("epidemic-events-fel.jsonl");
        let csv = dir.join("epidemic-trajectory-fel.csv");
        let matrix = dir.join("epidemic-transition-matrix-fel.json");
        assert!(log.exists(), "event log written");
        assert!(csv.exists(), "csv written");
        assert!(matrix.exists(), "matrix written");

        let events = read_events(log.to_str().unwrap()).expect("read events");
        assert!(events
            .iter()
            .any(|e| e.get("kind").and_then(|v| v.as_str()) == Some("sim_start")));
        let end = events
            .iter()
            .find(|e| e.get("kind").and_then(|v| v.as_str()) == Some("sim_end"))
            .expect("sim_end present");

        // The matrix JSON round-trips through the parser.
        let mtext = fs::read_to_string(&matrix).unwrap();
        assert!(crate::des::observability::logger::parse_json(&mtext).is_ok());

        // Mass: source emissions == createdCount, and within the configured cap.
        let created = end
            .pointer(&["totals", "created"])
            .and_then(|v| v.as_f64())
            .unwrap_or(-1.0);
        assert!(
            created >= 0.0 && created <= TURN_OFF_AFTER_COUNT as f64,
            "created={created}"
        );
        let source_emissions = events
            .iter()
            .filter(|e| {
                e.get("kind").and_then(|v| v.as_str()) == Some("transition")
                    && e.get("from").and_then(|v| v.as_str()) == Some("__source__")
            })
            .count() as f64;
        assert_eq!(
            source_emissions, created,
            "source emissions match createdCount"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
