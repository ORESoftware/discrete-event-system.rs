//! Port of `src/des/runners/validate-elevator.ts`.
//!
//! Compares the framework elevator-sim aggregates (`out/elevator-framework.json`)
//! against the SimPy continuous-time reference
//! (`out/external/elevator/simpy.json`) when those artifacts exist. If they are
//! missing, the runner generates the framework result directly from the real Rust
//! `main_elevator` engine and performs Rust-only invariant checks.
//! Top-level `main()` → [`run`].
//!
//! PORT NOTES:
//!   * `process.exit(code)` → explicit `std::process::exit` at the end of the
//!     validation branch.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::des::main_elevator::{
    build_schedule, run_elevator, Aggregates as RustAggregates,
    ElevatorConfig as RustElevatorConfig, Person as RustPerson,
};

// =============================================================================
// Typed views of the two JSON files. The framework writer emits camelCase keys
// (`nFloors`, `meanWait`, `fromFloor`, …); `serde(default)` keeps parsing
// tolerant of a reference file that omits fields.
// =============================================================================

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct ElevatorConfig {
    n_floors: i64,
    n_elevators: i64,
    capacity: i64,
    floor_travel_time: f64,
    service_time: f64,
    arrival_rate: f64,
    sim_t: f64,
    step_size: f64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct Aggregates {
    n: f64,
    n_served: f64,
    mean_wait: f64,
    mean_travel: f64,
    mean_total: f64,
    p95_wait: f64,
    p95_total: f64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct Person {
    id: i64,
    from_floor: i64,
    to_floor: i64,
    arrival_time: f64,
    board_time: f64,
    exit_time: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct FrameworkJson {
    config: ElevatorConfig,
    aggregates: Aggregates,
    people: Vec<Person>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct SimPyJson {
    aggregates: Aggregates,
    people: Vec<Person>,
}

fn load_json_opt<T: serde::de::DeserializeOwned>(p: &Path) -> Option<T> {
    if !p.exists() {
        eprintln!("[validate-elevator] missing {}", p.display());
        return None;
    }
    let text = match std::fs::read_to_string(p) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("[validate-elevator] read error {}: {e}", p.display());
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("[validate-elevator] parse error {}: {e}", p.display());
            None
        }
    }
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn local_config_from_rust(c: &RustElevatorConfig) -> ElevatorConfig {
    ElevatorConfig {
        n_floors: c.n_floors,
        n_elevators: c.n_elevators as i64,
        capacity: c.capacity as i64,
        floor_travel_time: c.floor_travel_time,
        service_time: c.service_time,
        arrival_rate: c.arrival_rate,
        sim_t: c.sim_t,
        step_size: c.step_size,
    }
}

fn local_aggregates_from_rust(a: &RustAggregates) -> Aggregates {
    Aggregates {
        n: a.n as f64,
        n_served: a.n_served as f64,
        mean_wait: a.mean_wait,
        mean_travel: a.mean_travel,
        mean_total: a.mean_total,
        p95_wait: a.p95_wait,
        p95_total: a.p95_total,
    }
}

fn local_person_from_rust(p: &RustPerson) -> Person {
    Person {
        id: p.id,
        from_floor: p.from_floor,
        to_floor: p.to_floor,
        arrival_time: p.arrival_time,
        board_time: p.board_time,
        exit_time: p.exit_time,
    }
}

fn generated_framework_json() -> FrameworkJson {
    let cfg = RustElevatorConfig {
        n_floors: 4,
        n_elevators: 3,
        capacity: 8,
        floor_travel_time: 4.0,
        service_time: 3.0,
        arrival_rate: 0.2,
        sim_t: 1800.0,
        step_size: 0.5,
        seed: 1,
        dispatch_mode: "uncoordinated".to_string(),
    };
    let schedule = build_schedule(&cfg);
    let result = run_elevator(cfg, schedule);
    FrameworkJson {
        config: local_config_from_rust(&result.config),
        aggregates: local_aggregates_from_rust(&result.aggregates),
        people: result.people.iter().map(local_person_from_rust).collect(),
    }
}

struct Matched {
    id: i64,
    from_floor: i64,
    to_floor: i64,
    board_diff: f64,
    exit_diff: f64,
}

/// `validate-elevator.ts` `main()`.
pub fn run() {
    let ts_path = root().join("out").join("elevator-framework.json");
    let py_path = root()
        .join("out")
        .join("external")
        .join("elevator")
        .join("simpy.json");

    let ts: FrameworkJson = match load_json_opt(&ts_path) {
        Some(v) => v,
        None => {
            eprintln!("[validate-elevator] generating framework result from Rust engine");
            generated_framework_json()
        }
    };
    let py: Option<SimPyJson> = load_json_opt(&py_path);

    println!("Elevator: framework (fixed-step DES) vs optional SimPy reference");
    println!("=====================================================================");
    println!(
        "  {} floors, {} elevators, capacity {}",
        ts.config.n_floors, ts.config.n_elevators, ts.config.capacity
    );
    println!(
        "  travel={}s/floor, service={}s, λ={}/s, simT={}s",
        ts.config.floor_travel_time,
        ts.config.service_time,
        ts.config.arrival_rate,
        ts.config.sim_t
    );
    println!("  framework dt = {}s", ts.config.step_size);
    println!();

    let ts_agg = &ts.aggregates;
    if py.is_none() {
        let finite_metrics = [
            ts_agg.mean_wait,
            ts_agg.mean_travel,
            ts_agg.mean_total,
            ts_agg.p95_wait,
            ts_agg.p95_total,
        ]
        .iter()
        .all(|v| v.is_finite());
        let internal_ok = ts_agg.n > 0.0
            && ts_agg.n_served > 0.0
            && ts_agg.n_served <= ts_agg.n
            && !ts.people.is_empty()
            && finite_metrics;

        println!("  SimPy reference unavailable; Rust-only checks:");
        println!("    generated people: {:.0}", ts_agg.n);
        println!("    served people:    {:.0}", ts_agg.n_served);
        println!("    people trace:     {}", ts.people.len());
        println!(
            "    finite metrics:   {}",
            if finite_metrics { "yes" } else { "NO" }
        );
        println!("{}", if internal_ok { "  PASS" } else { "  FAIL" });
        std::process::exit(if internal_ok { 0 } else { 1 });
    }
    let py = py.expect("checked above");
    let py_agg = &py.aggregates;

    println!(
        "  {:<14} {:>12} {:>12} {:>10} {:>10}",
        "metric", "framework", "SimPy", "Δ", "Δ / dt"
    );
    let rows: [(&str, f64, f64); 7] = [
        ("n", ts_agg.n, py_agg.n),
        ("nServed", ts_agg.n_served, py_agg.n_served),
        ("meanWait", ts_agg.mean_wait, py_agg.mean_wait),
        ("meanTravel", ts_agg.mean_travel, py_agg.mean_travel),
        ("meanTotal", ts_agg.mean_total, py_agg.mean_total),
        ("p95Wait", ts_agg.p95_wait, py_agg.p95_wait),
        ("p95Total", ts_agg.p95_total, py_agg.p95_total),
    ];
    for (name, a, b) in rows {
        let d = a - b;
        let d_n = d / ts.config.step_size;
        println!(
            "  {:<14} {:>12} {:>12} {:>10} {:>10}",
            name,
            format!("{:.2}", a),
            format!("{:.2}", b),
            format!("{:.2}", d),
            format!("{:.2}", d_n)
        );
    }

    // Per-person comparison matched by id.
    let mut py_by_id: HashMap<i64, Person> = HashMap::new();
    for p in &py.people {
        py_by_id.insert(p.id, *p);
    }

    let mut matched: Vec<Matched> = Vec::new();
    let mut unmatched = 0usize;
    for a in &ts.people {
        match py_by_id.get(&a.id) {
            None => {
                unmatched += 1;
                continue;
            }
            Some(b) => matched.push(Matched {
                id: a.id,
                from_floor: a.from_floor,
                to_floor: a.to_floor,
                board_diff: a.board_time - b.board_time,
                exit_diff: a.exit_time - b.exit_time,
            }),
        }
    }

    let mut max_board_diff = 0.0_f64;
    let mut max_exit_diff = 0.0_f64;
    let mut sum_abs_board = 0.0_f64;
    let mut sum_abs_exit = 0.0_f64;
    for m in &matched {
        sum_abs_board += m.board_diff.abs();
        sum_abs_exit += m.exit_diff.abs();
        if m.board_diff.abs() > max_board_diff {
            max_board_diff = m.board_diff.abs();
        }
        if m.exit_diff.abs() > max_exit_diff {
            max_exit_diff = m.exit_diff.abs();
        }
    }
    let denom = matched.len().max(1) as f64;
    let mean_abs_board = sum_abs_board / denom;
    let mean_abs_exit = sum_abs_exit / denom;

    println!();
    println!(
        "  Per-person time differences (over {} matched persons):",
        matched.len()
    );
    println!(
        "    mean |board_ts - board_simpy| = {:.3} s   (~{:.2} × dt)",
        mean_abs_board,
        mean_abs_board / ts.config.step_size
    );
    println!(
        "    mean |exit_ts  - exit_simpy|  = {:.3} s   (~{:.2} × dt)",
        mean_abs_exit,
        mean_abs_exit / ts.config.step_size
    );
    println!(
        "    max  |board diff|             = {:.3} s",
        max_board_diff
    );
    println!("    max  |exit  diff|             = {:.3} s", max_exit_diff);
    if unmatched > 0 {
        println!("  WARN: {} unmatched persons", unmatched);
    }

    // Acceptance: aggregate metrics within 10% of SimPy.
    let agg_ok = (ts_agg.mean_wait - py_agg.mean_wait).abs() < 0.10 * (py_agg.mean_wait + 1.0)
        && (ts_agg.mean_travel - py_agg.mean_travel).abs() < 0.10 * (py_agg.mean_travel + 1.0)
        && (ts_agg.mean_total - py_agg.mean_total).abs() < 0.10 * (py_agg.mean_total + 1.0);

    println!();
    println!(
        "  aggregate Δ within 10% of SimPy: {}",
        if agg_ok { "yes" } else { "NO" }
    );
    println!("{}", if agg_ok { "  PASS" } else { "  FAIL" });
    std::process::exit(if agg_ok { 0 } else { 1 });
}
