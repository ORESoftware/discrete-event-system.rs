//! Port of `src/des/runners/compare-elevator-dispatch.ts`.
//!
//! Sweeps seeds and arrival rates to quantify coordinated vs uncoordinated
//! elevator dispatch, printing a table + per-rate summary and writing a JSON
//! dump. The TS top-level `main()` becomes [`run`].
//!
//! ## PORT NOTE — local elevator-sim stub
//!
//! This driver imports `ElevatorConfig`/`buildSchedule`/`runElevator` from
//! `../main-elevator`, which is **not yet ported** to Rust (no `main_elevator.rs`
//! exists — only `validate_elevator.rs`, which reads pre-baked JSON). Per the
//! migration brief this file ships the *smallest self-contained* elevator engine
//! that reproduces the comparison's observable behaviour (coordinated dispatch
//! lowers mean/p95 wait by claiming each hall call exactly once; uncoordinated
//! dispatch lets multiple cars chase the same call and eat redundant service
//! stops). It is **not** a numeric match for `main-elevator`; replace
//! [`build_schedule`] / [`run_elevator`] with the real port when it lands.
//!
//! Other notes:
//!   * `process.env.{SEEDS,LAMBDAS,SIM_T}` → `std::env::var` + split/parse.
//!   * `Math.random`/seeding → `with_seed`.
//!   * `fs`/`path` + `JSON.stringify(.., null, 2)` → `std::fs` + `JsonValue`.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::des::general::prng::with_seed;
use crate::des::observability::logger::JsonValue;
use crate::des::shared::capabilities::RandomSource;

/// `ElevatorConfig` minus `dispatchMode` (the TS `Omit<…, 'dispatchMode'>`).
#[derive(Clone, Copy, Debug)]
struct ElevatorConfig {
    n_floors: i64,
    n_elevators: i64,
    capacity: i64,
    floor_travel_time: f64,
    service_time: f64,
    arrival_rate: f64,
    sim_t: f64,
    step_size: f64,
    seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DispatchMode {
    Uncoordinated,
    Coordinated,
}

/// Aggregated wait/total statistics (`runElevator(...).aggregates`).
#[derive(Clone, Copy, Debug, Default)]
struct Aggregates {
    mean_wait: f64,
    p95_wait: f64,
    mean_total: f64,
}

#[derive(Clone, Copy, Debug)]
struct Passenger {
    arrival: f64,
    from: i64,
    to: i64,
    board: Option<f64>,
    exit: Option<f64>,
}

struct TrialAggregate {
    seed: u64,
    lambda: f64,
    uncoord: Aggregates,
    coord: Aggregates,
}

fn parse_list(env: Option<String>, def: &[f64]) -> Vec<f64> {
    match env {
        None => def.to_vec(),
        Some(s) => s
            .split(',')
            .filter_map(|x| x.trim().parse::<f64>().ok())
            .filter(|x| x.is_finite())
            .collect(),
    }
}

fn pct(a: f64, b: f64) -> String {
    if b == 0.0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", (a / b - 1.0) * 100.0)
    }
}

fn pad_start(s: &str, width: usize) -> String {
    if s.chars().count() >= width {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(width - s.chars().count()))
    }
}

// =============================================================================
// PORT NOTE: minimal stand-in elevator engine (see module docs).
// =============================================================================

/// `buildSchedule(cfg)` — Poisson hall-call arrivals over `[0, simT]`.
fn build_schedule(cfg: &ElevatorConfig) -> Vec<Passenger> {
    with_seed(cfg.seed as u32, |rng| {
        let mut passengers: Vec<Passenger> = Vec::new();
        let mut t = 0.0_f64;
        let n = cfg.n_floors.max(2);
        loop {
            let u = rng.next_float().max(1e-12);
            t += -(1.0 - u).ln() / cfg.arrival_rate.max(1e-9);
            if t >= cfg.sim_t {
                break;
            }
            let from = (rng.next_float() * n as f64).floor() as i64;
            let mut to = (rng.next_float() * n as f64).floor() as i64;
            if to == from {
                to = (to + 1) % n;
            }
            passengers.push(Passenger {
                arrival: t,
                from,
                to,
                board: None,
                exit: None,
            });
        }
        passengers
    })
}

struct Elevator {
    pos: f64,
    dwell: f64,
    target: Option<i64>,
    onboard: Vec<usize>,
}

/// `runElevator(cfg, schedule)` → wait/total aggregates.
fn run_elevator(cfg: &ElevatorConfig, mode: DispatchMode, schedule: &[Passenger]) -> Aggregates {
    let mut passengers: Vec<Passenger> = schedule.to_vec();
    let dt = cfg.step_size.max(1e-3);
    let mut elevators: Vec<Elevator> = (0..cfg.n_elevators.max(1))
        .map(|_| Elevator {
            pos: 0.0,
            dwell: 0.0,
            target: None,
            onboard: Vec::new(),
        })
        .collect();
    // Index of next arrival not yet released into the waiting pool.
    let mut next_arrival = 0usize;
    let mut waiting: Vec<usize> = Vec::new();
    // Coordinated mode: which elevator claimed each waiting passenger.
    let mut claimed_by: Vec<Option<usize>> = vec![None; passengers.len()];

    let eps = 0.001_f64;
    let mut t = 0.0_f64;
    let arrival_order: Vec<usize> = {
        let mut idx: Vec<usize> = (0..passengers.len()).collect();
        idx.sort_by(|&a, &b| {
            passengers[a]
                .arrival
                .partial_cmp(&passengers[b].arrival)
                .unwrap()
        });
        idx
    };

    while t <= cfg.sim_t + dt {
        // Release arrivals.
        while next_arrival < arrival_order.len() {
            let pid = arrival_order[next_arrival];
            if passengers[pid].arrival <= t {
                waiting.push(pid);
                next_arrival += 1;
            } else {
                break;
            }
        }

        // Coordinated dispatch: claim each unclaimed waiting call for the
        // nearest elevator with spare capacity.
        if mode == DispatchMode::Coordinated {
            for &pid in &waiting {
                if claimed_by[pid].is_some() {
                    continue;
                }
                let from = passengers[pid].from as f64;
                let mut best: Option<(usize, f64)> = None;
                for (ei, e) in elevators.iter().enumerate() {
                    let load = e.onboard.len() as i64
                        + claimed_by.iter().filter(|c| **c == Some(ei)).count() as i64;
                    if load >= cfg.capacity {
                        continue;
                    }
                    let d = (e.pos - from).abs();
                    if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                        best = Some((ei, d));
                    }
                }
                if let Some((ei, _)) = best {
                    claimed_by[pid] = Some(ei);
                }
            }
        }

        // Per-elevator move + service.
        for ei in 0..elevators.len() {
            if elevators[ei].dwell > 0.0 {
                elevators[ei].dwell -= dt;
                continue;
            }
            // Choose a target if none.
            if elevators[ei].target.is_none() {
                elevators[ei].target =
                    choose_target(ei, mode, &elevators, &waiting, &claimed_by, &passengers);
            }
            let Some(target) = elevators[ei].target else {
                continue;
            };
            // Move toward target.
            let speed = dt / cfg.floor_travel_time.max(1e-9);
            let diff = target as f64 - elevators[ei].pos;
            if diff.abs() <= speed + eps {
                elevators[ei].pos = target as f64;
                service_stop(
                    ei,
                    cfg,
                    mode,
                    t,
                    &mut elevators,
                    &mut waiting,
                    &mut claimed_by,
                    &mut passengers,
                );
                elevators[ei].dwell = cfg.service_time;
                elevators[ei].target = None;
            } else {
                elevators[ei].pos += speed * diff.signum();
            }
        }

        t += dt;
    }

    aggregate(&passengers)
}

fn choose_target(
    ei: usize,
    mode: DispatchMode,
    elevators: &[Elevator],
    waiting: &[usize],
    claimed_by: &[Option<usize>],
    passengers: &[Passenger],
) -> Option<i64> {
    let pos = elevators[ei].pos;
    let mut best: Option<(i64, f64)> = None;
    let mut consider = |floor: i64| {
        let d = (floor as f64 - pos).abs();
        if best.map(|(_, bd)| d < bd).unwrap_or(true) {
            best = Some((floor, d));
        }
    };
    // Onboard destinations always count.
    for &pid in &elevators[ei].onboard {
        consider(passengers[pid].to);
    }
    // Pickups.
    for &pid in waiting {
        let eligible = match mode {
            DispatchMode::Coordinated => claimed_by[pid] == Some(ei),
            DispatchMode::Uncoordinated => true,
        };
        if eligible {
            consider(passengers[pid].from);
        }
    }
    best.map(|(f, _)| f)
}

#[allow(clippy::too_many_arguments)]
fn service_stop(
    ei: usize,
    cfg: &ElevatorConfig,
    mode: DispatchMode,
    t: f64,
    elevators: &mut [Elevator],
    waiting: &mut Vec<usize>,
    claimed_by: &mut [Option<usize>],
    passengers: &mut [Passenger],
) {
    let floor = elevators[ei].pos.round() as i64;
    // Drop off onboard whose destination is this floor.
    let mut still: Vec<usize> = Vec::new();
    for &pid in &elevators[ei].onboard {
        if passengers[pid].to == floor {
            passengers[pid].exit = Some(t);
        } else {
            still.push(pid);
        }
    }
    elevators[ei].onboard = still;
    // Board waiting passengers at this floor (respecting capacity + claims).
    let mut remaining: Vec<usize> = Vec::new();
    for &pid in waiting.iter() {
        let here = passengers[pid].from == floor;
        let mine = match mode {
            DispatchMode::Coordinated => claimed_by[pid] == Some(ei),
            DispatchMode::Uncoordinated => true,
        };
        if here && mine && (elevators[ei].onboard.len() as i64) < cfg.capacity {
            passengers[pid].board = Some(t);
            elevators[ei].onboard.push(pid);
            claimed_by[pid] = None;
        } else {
            remaining.push(pid);
        }
    }
    *waiting = remaining;
}

fn aggregate(passengers: &[Passenger]) -> Aggregates {
    let mut waits: Vec<f64> = Vec::new();
    let mut totals: Vec<f64> = Vec::new();
    for p in passengers {
        if let Some(board) = p.board {
            waits.push(board - p.arrival);
        }
        if let (Some(_), Some(exit)) = (p.board, p.exit) {
            totals.push(exit - p.arrival);
        }
    }
    Aggregates {
        mean_wait: mean(&waits),
        p95_wait: percentile(&waits, 0.95),
        mean_total: mean(&totals),
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn percentile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((q * (v.len() as f64 - 1.0)).round() as usize).min(v.len() - 1);
    v[idx]
}

fn jn(v: f64) -> JsonValue {
    JsonValue::Number(v)
}

fn agg_json(a: &Aggregates) -> JsonValue {
    JsonValue::Object(vec![
        ("meanWait".to_string(), jn(a.mean_wait)),
        ("p95Wait".to_string(), jn(a.p95_wait)),
        ("meanTotal".to_string(), jn(a.mean_total)),
    ])
}

/// `main()` — run the dispatch sweep.
pub fn run() {
    let seeds = parse_list(std::env::var("SEEDS").ok(), &[1.0, 2.0, 3.0, 4.0, 5.0])
        .into_iter()
        .map(|x| x as u64)
        .collect::<Vec<_>>();
    let lambdas = parse_list(std::env::var("LAMBDAS").ok(), &[0.2, 0.4]);
    let sim_t: f64 = std::env::var("SIM_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1800.0);

    let mut trials: Vec<TrialAggregate> = Vec::new();
    println!("# elevator dispatch comparison");
    println!(
        "#   seeds = {}",
        seeds
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "#   λ     = {} arrivals/s",
        lambdas
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("#   simT  = {sim_t}s");
    println!();
    println!("  λ     seed   meanWait u→c     p95Wait u→c     meanTotal u→c    Δmean   Δp95");
    println!(
        "  ────  ────  ─────────────────  ─────────────────  ──────────────────  ──────  ──────"
    );

    for &lambda in &lambdas {
        for &seed in &seeds {
            let base = ElevatorConfig {
                n_floors: 4,
                n_elevators: 3,
                capacity: 8,
                floor_travel_time: 4.0,
                service_time: 3.0,
                arrival_rate: lambda,
                sim_t,
                step_size: 0.5,
                seed,
            };
            let schedule = build_schedule(&base);
            let u = run_elevator(&base, DispatchMode::Uncoordinated, &schedule);
            let c = run_elevator(&base, DispatchMode::Coordinated, &schedule);
            trials.push(TrialAggregate {
                seed,
                lambda,
                uncoord: u,
                coord: c,
            });

            let mw = format!(
                "{} → {}",
                pad_start(&format!("{:.2}", u.mean_wait), 5),
                pad_start(&format!("{:.2}", c.mean_wait), 5)
            );
            let pw = format!(
                "{} → {}",
                pad_start(&format!("{:.1}", u.p95_wait), 5),
                pad_start(&format!("{:.1}", c.p95_wait), 5)
            );
            let mt = format!(
                "{} → {}",
                pad_start(&format!("{:.2}", u.mean_total), 6),
                pad_start(&format!("{:.2}", c.mean_total), 6)
            );
            let dm = pad_start(&pct(c.mean_wait, u.mean_wait), 7);
            let dp = pad_start(&pct(c.p95_wait, u.p95_wait), 7);
            println!(
                "  {:.2}  {}  {mw}   {pw}   {mt}  {dm} {dp}",
                lambda,
                pad_start(&seed.to_string(), 4)
            );
        }
    }

    // Aggregate by lambda (insertion order preserved via Vec of keys).
    let mut by_lambda: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    let mut lambda_order: Vec<u64> = Vec::new();
    for (i, tr) in trials.iter().enumerate() {
        let key = tr.lambda.to_bits();
        if !by_lambda.contains_key(&key) {
            lambda_order.push(key);
        }
        by_lambda.entry(key).or_default().push(i);
    }
    println!();
    println!("  Per-rate summary (mean over seeds):");
    for key in &lambda_order {
        let idxs = &by_lambda[key];
        let lambda = f64::from_bits(*key);
        let mean_of = |sel: &dyn Fn(&TrialAggregate) -> f64| -> f64 {
            idxs.iter().map(|&i| sel(&trials[i])).sum::<f64>() / idxs.len() as f64
        };
        let u_mw = mean_of(&|t| t.uncoord.mean_wait);
        let c_mw = mean_of(&|t| t.coord.mean_wait);
        let u_p95 = mean_of(&|t| t.uncoord.p95_wait);
        let c_p95 = mean_of(&|t| t.coord.p95_wait);
        let u_mt = mean_of(&|t| t.uncoord.mean_total);
        let c_mt = mean_of(&|t| t.coord.mean_total);
        println!(
            "    λ={:.2}: meanWait {:.2}→{:.2} ({}), p95Wait {:.1}→{:.1} ({}), meanTotal {:.2}→{:.2} ({})",
            lambda, u_mw, c_mw, pct(c_mw, u_mw), u_p95, c_p95, pct(c_p95, u_p95), u_mt, c_mt, pct(c_mt, u_mt)
        );
    }

    let out_dir = Path::new("out");
    if let Err(e) = fs::create_dir_all(out_dir) {
        eprintln!("# could not create out dir: {e}");
        return;
    }
    let out_path = out_dir.join("elevator-dispatch-sweep.json");
    let trials_json = JsonValue::Array(
        trials
            .iter()
            .map(|t| {
                JsonValue::Object(vec![
                    ("seed".to_string(), jn(t.seed as f64)),
                    ("lambda".to_string(), jn(t.lambda)),
                    ("uncoord".to_string(), agg_json(&t.uncoord)),
                    ("coord".to_string(), agg_json(&t.coord)),
                ])
            })
            .collect(),
    );
    let payload = JsonValue::Object(vec![
        (
            "seeds".to_string(),
            JsonValue::Array(seeds.iter().map(|s| jn(*s as f64)).collect()),
        ),
        (
            "lambdas".to_string(),
            JsonValue::Array(lambdas.iter().map(|l| jn(*l)).collect()),
        ),
        ("simT".to_string(), jn(sim_t)),
        ("trials".to_string(), trials_json),
    ]);
    if let Err(e) = fs::write(&out_path, payload.to_string_pretty(2)) {
        eprintln!("# could not write artifact: {e}");
        return;
    }
    println!();
    println!("# wrote {}", out_path.display());
}
