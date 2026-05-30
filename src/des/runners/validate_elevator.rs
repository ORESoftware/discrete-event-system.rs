//! Port of `src/des/runners/validate-elevator.ts`.
//!
//! Compares the framework elevator-sim aggregates (`out/elevator-framework.json`)
//! against the SimPy continuous-time reference
//! (`out/external/elevator/simpy.json`): per-person board/exit diffs and
//! per-aggregate diffs, asserting the aggregate metrics agree within 10%.
//! Top-level `main()` → [`run`].
//!
//! PORT NOTES:
//!   * JSON loading is stubbed (no `serde`/`serde_json` dependency yet); the
//!     `load_json` helper reproduces the missing-file `exit(1)` and documents the
//!     `serde_json::from_str` call to wire.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// =============================================================================
// Typed views of the two JSON files (PORT NOTE: `#[derive(Deserialize)]`).
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
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

#[derive(Clone, Copy, Debug, Default)]
struct Aggregates {
    n: f64,
    n_served: f64,
    mean_wait: f64,
    mean_travel: f64,
    mean_total: f64,
    p95_wait: f64,
    p95_total: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Person {
    id: i64,
    from_floor: i64,
    to_floor: i64,
    arrival_time: f64,
    board_time: f64,
    exit_time: f64,
}

#[derive(Clone, Debug, Default)]
struct FrameworkJson {
    config: ElevatorConfig,
    aggregates: Aggregates,
    people: Vec<Person>,
}

#[derive(Clone, Debug, Default)]
struct SimPyJson {
    aggregates: Aggregates,
    people: Vec<Person>,
}

fn load_json<T>(p: &Path) -> T {
    if !p.exists() {
        eprintln!("[validate-elevator] missing {}", p.display());
        std::process::exit(1);
    }
    // PORT NOTE: `serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()`.
    eprintln!(
        "[validate-elevator] PORT NOTE: JSON parsing not wired (needs serde_json): {}",
        p.display()
    );
    std::process::exit(1);
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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

    let ts: FrameworkJson = load_json(&ts_path);
    let py: SimPyJson = load_json(&py_path);

    println!("Elevator: framework (fixed-step DES) vs SimPy (continuous-time FEL)");
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
