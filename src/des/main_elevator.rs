//! Port of `src/des/main-elevator.ts`.
//!
//! 3-elevator, 4-floor building: passenger-arrival simulation layered with a
//! SCAN/LOOK dispatch policy (uncoordinated / coordinated / coordinated-pickup).
//! Defines the model AND runs it.
//!
//! `mulberry32`/`withSeed` → `crate::des::general::prng`; `process.env.*` →
//! `std::env::var`; `fs` → `std::fs`; the JSON artifacts are built with
//! `crate::des::observability::logger::JsonValue` (no `serde`).
//!
//! PORT NOTES:
//!   * `TimeSteppedStation` base is inlined: `Building` owns every entity and
//!     orchestrates the per-tick logic directly (avoids `Rc<RefCell>` graph
//!     wiring); elevator stepping is split into free functions so floors, the
//!     coordinator, and the active car are borrowed disjointly.
//!   * the "cosmetic" Fisher–Yates shuffle of the passive (floor/sink) stations
//!     has no functional effect (the TS comment says so) and is omitted.
//!   * the optional `ANIMATE=1` branch needs
//!     `animation/scenes/elevator-scene` (not ported); animation is stubbed.

#![allow(dead_code)]

use crate::des::general::prng::{mulberry32, with_seed};
use crate::des::observability::logger::JsonValue;
use crate::des::shared::capabilities::RandomSource;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir2 {
    Up,
    Down,
}
impl Dir2 {
    fn label(self) -> &'static str {
        match self {
            Dir2::Up => "up",
            Dir2::Down => "down",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Edir {
    Up,
    Down,
    Idle,
}
impl Edir {
    fn to_dir2_or_up(self) -> Dir2 {
        match self {
            Edir::Down => Dir2::Down,
            _ => Dir2::Up,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ElevatorState {
    Idle,
    Moving,
    Serving,
}

#[derive(Clone)]
struct Person {
    id: i64,
    from_floor: i64,
    to_floor: i64,
    arrival_time: f64,
    board_time: f64,
    exit_time: f64,
}
impl Person {
    fn new(id: i64, from_floor: i64, to_floor: i64) -> Self {
        Person {
            id,
            from_floor,
            to_floor,
            arrival_time: -1.0,
            board_time: -1.0,
            exit_time: -1.0,
        }
    }
}

#[derive(Clone, Copy)]
struct ScheduledArrival {
    t: f64,
    from_floor: i64,
    to_floor: i64,
}

struct PersonSource {
    schedule: Vec<ScheduledArrival>,
    idx: usize,
    next_id: i64,
}

struct Floor {
    floor_number: i64,
    up_queue: Vec<Person>,
    down_queue: Vec<Person>,
    exited_here: Vec<Person>,
}
impl Floor {
    fn add_person(&mut self, p: Person) {
        if p.to_floor > p.from_floor {
            self.up_queue.push(p);
        } else {
            self.down_queue.push(p);
        }
    }
    fn has_call(&self, dir: Dir2) -> bool {
        match dir {
            Dir2::Up => !self.up_queue.is_empty(),
            Dir2::Down => !self.down_queue.is_empty(),
        }
    }
    fn take_from_queue(&mut self, dir: Dir2, cap: usize) -> Vec<Person> {
        let q = match dir {
            Dir2::Up => &mut self.up_queue,
            Dir2::Down => &mut self.down_queue,
        };
        let n = cap.min(q.len());
        q.drain(0..n).collect()
    }
}

#[derive(Default)]
struct Coordinator {
    claimed_by: HashMap<String, usize>,
}
impl Coordinator {
    fn reset(&mut self) {
        self.claimed_by.clear();
    }
    fn key(floor: i64, dir: Dir2) -> String {
        format!("{}-{}", floor, dir.label())
    }
    fn is_claimed_by_other(&self, floor: i64, dir: Dir2, my_idx: usize) -> bool {
        match self.claimed_by.get(&Coordinator::key(floor, dir)) {
            Some(owner) => *owner != my_idx,
            None => false,
        }
    }
    fn claim(&mut self, floor: i64, dir: Dir2, by_idx: usize) {
        self.claimed_by
            .entry(Coordinator::key(floor, dir))
            .or_insert(by_idx);
    }
    fn seed_from_active(&mut self, elevators: &[Elevator]) {
        self.reset();
        for e in elevators {
            if (e.state == ElevatorState::Moving || e.state == ElevatorState::Serving)
                && (e.direction == Edir::Up || e.direction == Edir::Down)
            {
                self.claim(
                    e.target_floor.round() as i64,
                    e.direction.to_dir2_or_up(),
                    e.idx,
                );
            }
        }
    }
}

struct Elevator {
    idx: usize,
    capacity: usize,
    floor_travel_time: f64,
    service_time: f64,
    state: ElevatorState,
    current_floor: f64,
    target_floor: f64,
    direction: Edir,
    passengers: Vec<Person>,
    service_remaining: f64,
    opportunistic_pickups: bool,
}
impl Elevator {
    fn speed(&self) -> f64 {
        1.0 / self.floor_travel_time
    }
    fn is_full(&self) -> bool {
        self.passengers.len() >= self.capacity
    }
}

fn from_passenger_here(passengers: &[Person], floor: i64) -> bool {
    passengers.iter().any(|p| p.to_floor == floor)
}

#[derive(Default)]
struct ExitSink {
    collected: Vec<Person>,
}

#[derive(Clone)]
struct ElevatorConfig {
    n_floors: i64,
    n_elevators: usize,
    capacity: usize,
    floor_travel_time: f64,
    service_time: f64,
    arrival_rate: f64,
    sim_t: f64,
    step_size: f64,
    seed: u32,
    dispatch_mode: String,
}

#[derive(Clone)]
struct Aggregates {
    n: usize,
    n_served: usize,
    mean_wait: f64,
    mean_travel: f64,
    mean_total: f64,
    p95_wait: f64,
    p95_total: f64,
}

struct ElevatorResult {
    config: ElevatorConfig,
    schedule: Vec<ScheduledArrival>,
    people: Vec<Person>,
    aggregates: Aggregates,
}

/// `buildSchedule(cfg)` — deterministic Poisson arrival schedule.
fn build_schedule(cfg: &ElevatorConfig) -> Vec<ScheduledArrival> {
    let seed = cfg.seed;
    with_seed(seed, |_g| {
        let mut rng = mulberry32(seed);
        let mut out = Vec::new();
        let mut t = 0.0;
        loop {
            let u = 1.0 - rng.next_float();
            t += -u.ln() / cfg.arrival_rate;
            if t > cfg.sim_t {
                break;
            }
            let from_floor = 1 + (rng.next_float() * cfg.n_floors as f64).floor() as i64;
            let mut to_floor;
            loop {
                to_floor = 1 + (rng.next_float() * cfg.n_floors as f64).floor() as i64;
                if to_floor != from_floor {
                    break;
                }
            }
            out.push(ScheduledArrival {
                t,
                from_floor,
                to_floor,
            });
        }
        out
    })
}

struct Building {
    source: PersonSource,
    floors: Vec<Floor>,
    elevators: Vec<Elevator>,
    sink: ExitSink,
    coordinator: Option<Coordinator>,
    config: ElevatorConfig,
}

impl Building {
    fn new(cfg: ElevatorConfig, schedule: Vec<ScheduledArrival>) -> Self {
        let floors: Vec<Floor> = (1..=cfg.n_floors)
            .map(|i| Floor {
                floor_number: i,
                up_queue: Vec::new(),
                down_queue: Vec::new(),
                exited_here: Vec::new(),
            })
            .collect();
        let source = PersonSource {
            schedule,
            idx: 0,
            next_id: 0,
        };
        let dispatch_mode = if cfg.dispatch_mode.is_empty() {
            "uncoordinated".to_string()
        } else {
            cfg.dispatch_mode.clone()
        };
        let coordinated = dispatch_mode != "uncoordinated";
        let elevators: Vec<Elevator> = (0..cfg.n_elevators)
            .map(|k| {
                let start = 1 + (k as i64 % cfg.n_floors);
                Elevator {
                    idx: k,
                    capacity: cfg.capacity,
                    floor_travel_time: cfg.floor_travel_time,
                    service_time: cfg.service_time,
                    state: ElevatorState::Idle,
                    current_floor: start as f64,
                    target_floor: start as f64,
                    direction: Edir::Idle,
                    passengers: Vec::new(),
                    service_remaining: 0.0,
                    opportunistic_pickups: coordinated && dispatch_mode == "coordinated-pickup",
                }
            })
            .collect();
        let coordinator = if coordinated {
            Some(Coordinator::default())
        } else {
            None
        };
        Building {
            source,
            floors,
            elevators,
            sink: ExitSink::default(),
            coordinator,
            config: cfg,
        }
    }

    fn tick_once(&mut self, t: i64) {
        let dt = self.config.step_size;
        // 1. Source emits newly-arrived persons onto floors.
        let now = t as f64 * dt;
        while self.source.idx < self.source.schedule.len()
            && self.source.schedule[self.source.idx].t <= now
        {
            let a = self.source.schedule[self.source.idx];
            self.source.idx += 1;
            let mut p = Person::new(self.source.next_id, a.from_floor, a.to_floor);
            self.source.next_id += 1;
            p.arrival_time = a.t;
            self.floors[(a.from_floor - 1) as usize].add_person(p);
        }
        // 2. Coordinator snapshot of active trajectories.
        if let Some(c) = self.coordinator.as_mut() {
            c.seed_from_active(&self.elevators);
        }
        // 3. Passive stations: floors drain deboarded people to the sink.
        for f in &mut self.floors {
            if !f.exited_here.is_empty() {
                for p in f.exited_here.drain(..) {
                    self.sink.collected.push(p);
                }
            }
        }
        // 4. Elevators run in index order (deterministic coordinated picks).
        for k in 0..self.elevators.len() {
            elevator_run_time_step(
                &mut self.elevators[k],
                &mut self.floors,
                self.coordinator.as_mut(),
                dt,
                t,
            );
        }
    }

    fn is_complete(&self, schedule_len: usize) -> bool {
        self.sink.collected.len() == schedule_len
    }
}

/// SCAN/LOOK next-target selection (`pickNextTarget`).
fn pick_next_target(
    e: &Elevator,
    floors: &[Floor],
    coord: Option<&Coordinator>,
) -> Option<(f64, Dir2)> {
    let cur = e.current_floor;

    let try_dir = |dir: Dir2| -> Option<(f64, Dir2)> {
        let better = |a: f64, b: f64| if dir == Dir2::Up { a < b } else { a > b };
        let mut best = -1.0f64;
        let mut from_passenger = false;
        for p in &e.passengers {
            let ahead = match dir {
                Dir2::Up => p.to_floor as f64 > cur,
                Dir2::Down => (p.to_floor as f64) < cur,
            };
            if ahead && (best < 0.0 || better(p.to_floor as f64, best)) {
                best = p.to_floor as f64;
                from_passenger = true;
            }
        }
        for f in floors {
            let ahead = match dir {
                Dir2::Up => f.floor_number as f64 > cur,
                Dir2::Down => (f.floor_number as f64) < cur,
            };
            if ahead && f.has_call(dir) {
                if let Some(c) = coord {
                    if c.is_claimed_by_other(f.floor_number, dir, e.idx)
                        && !from_passenger_here(&e.passengers, f.floor_number)
                    {
                        continue;
                    }
                }
                if best < 0.0 || better(f.floor_number as f64, best) {
                    best = f.floor_number as f64;
                    from_passenger = false;
                }
            }
        }
        let _ = from_passenger;
        if best > 0.0 {
            Some((best, dir))
        } else {
            None
        }
    };

    match e.direction {
        Edir::Up => {
            if let Some(t) = try_dir(Dir2::Up) {
                return Some(t);
            }
            if let Some(u) = try_dir(Dir2::Down) {
                return Some(u);
            }
        }
        Edir::Down => {
            if let Some(t) = try_dir(Dir2::Down) {
                return Some(t);
            }
            if let Some(u) = try_dir(Dir2::Up) {
                return Some(u);
            }
        }
        Edir::Idle => {}
    }

    let mut best_dist = f64::INFINITY;
    let mut best_floor = -1.0f64;
    let mut best_dir = Dir2::Up;
    for f in floors {
        let d = (f.floor_number as f64 - cur).abs();
        let up_claimed = coord
            .map(|c| c.is_claimed_by_other(f.floor_number, Dir2::Up, e.idx))
            .unwrap_or(false);
        let down_claimed = coord
            .map(|c| c.is_claimed_by_other(f.floor_number, Dir2::Down, e.idx))
            .unwrap_or(false);
        if f.has_call(Dir2::Up) && !up_claimed && d < best_dist {
            best_dist = d;
            best_floor = f.floor_number as f64;
            best_dir = Dir2::Up;
        }
        if f.has_call(Dir2::Down) && !down_claimed && d < best_dist {
            best_dist = d;
            best_floor = f.floor_number as f64;
            best_dir = Dir2::Down;
        }
    }
    if best_floor > 0.0 {
        Some((best_floor, best_dir))
    } else {
        None
    }
}

fn opportunistic_pit_stop(
    e: &Elevator,
    floors: &[Floor],
    coord: Option<&Coordinator>,
    new_floor: f64,
) -> Option<f64> {
    if !e.opportunistic_pickups || e.is_full() {
        return None;
    }
    let dir = match e.direction {
        Edir::Up => Dir2::Up,
        Edir::Down => Dir2::Down,
        Edir::Idle => return None,
    };
    let eps = 1e-9;
    let mut best = -1.0f64;
    for f in floors {
        let big_f = f.floor_number as f64;
        if (big_f - e.current_floor).abs() < eps {
            continue;
        }
        match dir {
            Dir2::Up => {
                if !(big_f > e.current_floor + eps && big_f <= new_floor + eps) {
                    continue;
                }
            }
            Dir2::Down => {
                if !(big_f < e.current_floor - eps && big_f >= new_floor - eps) {
                    continue;
                }
            }
        }
        if !f.has_call(dir) {
            continue;
        }
        if big_f == e.target_floor {
            continue;
        }
        if let Some(c) = coord {
            if c.is_claimed_by_other(f.floor_number, dir, e.idx) {
                continue;
            }
        }
        if best < 0.0 || (dir == Dir2::Up && big_f < best) || (dir == Dir2::Down && big_f > best) {
            best = big_f;
        }
    }
    if best > 0.0 {
        Some(best)
    } else {
        None
    }
}

fn elevator_run_time_step(
    e: &mut Elevator,
    floors: &mut [Floor],
    mut coord: Option<&mut Coordinator>,
    step_size: f64,
    t: i64,
) {
    let now = t as f64 * step_size;

    if e.state == ElevatorState::Idle {
        if let Some((floor, dir)) = pick_next_target(e, floors, coord.as_deref()) {
            e.target_floor = floor;
            e.direction = match dir {
                Dir2::Up => Edir::Up,
                Dir2::Down => Edir::Down,
            };
            e.state = ElevatorState::Moving;
            if let Some(c) = coord.as_deref_mut() {
                c.claim(floor.round() as i64, dir, e.idx);
            }
        }
    }

    if e.state == ElevatorState::Moving {
        let sign = if e.target_floor > e.current_floor {
            1.0
        } else {
            -1.0
        };
        let remaining = e.target_floor - e.current_floor;
        let delta = e.speed() * step_size * sign;
        let new_floor = e.current_floor + delta;
        let pit = opportunistic_pit_stop(e, floors, coord.as_deref(), new_floor);
        if let Some(pit) = pit {
            e.current_floor = pit;
            e.state = ElevatorState::Serving;
            e.service_remaining = e.service_time;
            if let Some(c) = coord.as_deref_mut() {
                c.claim(pit.round() as i64, e.direction.to_dir2_or_up(), e.idx);
            }
        } else if delta.abs() >= remaining.abs() - 1e-12 {
            e.current_floor = e.target_floor;
            e.state = ElevatorState::Serving;
            e.service_remaining = e.service_time;
        } else {
            e.current_floor = new_floor;
        }
    }

    if e.state == ElevatorState::Serving {
        let floor_idx = (e.current_floor.round() as i64 - 1) as usize;
        if e.service_remaining == e.service_time {
            let floor_number = floors[floor_idx].floor_number;
            let mut keep = Vec::new();
            let mut deboard = Vec::new();
            for p in e.passengers.drain(..) {
                if p.to_floor == floor_number {
                    deboard.push(p);
                } else {
                    keep.push(p);
                }
            }
            e.passengers = keep;
            for mut p in deboard {
                p.exit_time = now;
                floors[floor_idx].exited_here.push(p);
            }
            let dir = e.direction.to_dir2_or_up();
            let slots = e.capacity - e.passengers.len();
            let boarding = floors[floor_idx].take_from_queue(dir, slots);
            for mut p in boarding {
                p.board_time = now;
                e.passengers.push(p);
            }
        }
        e.service_remaining -= step_size;
        if e.service_remaining <= 0.0 {
            match pick_next_target(e, floors, coord.as_deref()) {
                None => {
                    e.state = ElevatorState::Idle;
                    e.direction = Edir::Idle;
                }
                Some((floor, dir)) => {
                    e.target_floor = floor;
                    e.direction = match dir {
                        Dir2::Up => Edir::Up,
                        Dir2::Down => Edir::Down,
                    };
                    e.state = ElevatorState::Moving;
                    if let Some(c) = coord {
                        c.claim(floor.round() as i64, dir, e.idx);
                    }
                }
            }
        }
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}
fn p95(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[(0.95 * (sorted.len() - 1) as f64).floor() as usize]
}

fn run_elevator(cfg: ElevatorConfig, schedule: Vec<ScheduledArrival>) -> ElevatorResult {
    let schedule_len = schedule.len();
    let mut b = Building::new(cfg.clone(), schedule.clone());
    let n = (cfg.sim_t / cfg.step_size).round() as i64;
    for t in 0..n {
        b.tick_once(t);
    }
    let mut extra = 0;
    while extra < n && !b.is_complete(schedule_len) {
        b.tick_once(n + extra);
        extra += 1;
    }
    let served: Vec<Person> = b
        .sink
        .collected
        .iter()
        .filter(|p| p.exit_time > 0.0)
        .cloned()
        .collect();
    let waits: Vec<f64> = served
        .iter()
        .map(|p| p.board_time - p.arrival_time)
        .collect();
    let travels: Vec<f64> = served.iter().map(|p| p.exit_time - p.board_time).collect();
    let totals: Vec<f64> = served
        .iter()
        .map(|p| p.exit_time - p.arrival_time)
        .collect();
    ElevatorResult {
        config: cfg,
        schedule,
        people: served.clone(),
        aggregates: Aggregates {
            n: schedule_len,
            n_served: served.len(),
            mean_wait: mean(&waits),
            mean_travel: mean(&travels),
            mean_total: mean(&totals),
            p95_wait: p95(&waits),
            p95_total: p95(&totals),
        },
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
fn config_json(c: &ElevatorConfig) -> JsonValue {
    jobj(vec![
        ("nFloors", jnum(c.n_floors as f64)),
        ("nElevators", jnum(c.n_elevators as f64)),
        ("capacity", jnum(c.capacity as f64)),
        ("floorTravelTime", jnum(c.floor_travel_time)),
        ("serviceTime", jnum(c.service_time)),
        ("arrivalRate", jnum(c.arrival_rate)),
        ("simT", jnum(c.sim_t)),
        ("stepSize", jnum(c.step_size)),
        ("seed", jnum(c.seed as f64)),
        ("dispatchMode", jstr(&c.dispatch_mode)),
    ])
}
fn aggregates_json(a: &Aggregates) -> JsonValue {
    jobj(vec![
        ("n", jnum(a.n as f64)),
        ("nServed", jnum(a.n_served as f64)),
        ("meanWait", jnum(a.mean_wait)),
        ("meanTravel", jnum(a.mean_travel)),
        ("meanTotal", jnum(a.mean_total)),
        ("p95Wait", jnum(a.p95_wait)),
        ("p95Total", jnum(a.p95_total)),
    ])
}
fn people_json(people: &[Person]) -> JsonValue {
    JsonValue::Array(
        people
            .iter()
            .map(|p| {
                jobj(vec![
                    ("id", jnum(p.id as f64)),
                    ("fromFloor", jnum(p.from_floor as f64)),
                    ("toFloor", jnum(p.to_floor as f64)),
                    ("arrivalTime", jnum(p.arrival_time)),
                    ("boardTime", jnum(p.board_time)),
                    ("exitTime", jnum(p.exit_time)),
                ])
            })
            .collect(),
    )
}
fn schedule_json(schedule: &[ScheduledArrival]) -> JsonValue {
    JsonValue::Array(
        schedule
            .iter()
            .map(|a| {
                jobj(vec![
                    ("t", jnum(a.t)),
                    ("fromFloor", jnum(a.from_floor as f64)),
                    ("toFloor", jnum(a.to_floor as f64)),
                ])
            })
            .collect(),
    )
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let base = ElevatorConfig {
        n_floors: env_f64("FLOORS", 4.0) as i64,
        n_elevators: env_f64("ELEVATORS", 3.0) as usize,
        capacity: env_f64("CAPACITY", 8.0) as usize,
        floor_travel_time: env_f64("TRAVEL_T", 4.0),
        service_time: env_f64("SERVICE_T", 3.0),
        arrival_rate: env_f64("LAMBDA", 0.2),
        sim_t: env_f64("SIM_T", 1800.0),
        step_size: env_f64("STEPSIZE", 0.5),
        seed: env_f64("SEED", 1.0) as u32,
        dispatch_mode: String::new(),
    };
    println!("# Elevator simulation");
    println!(
        "#   {} floors, {} elevators, capacity {}",
        base.n_floors, base.n_elevators, base.capacity
    );
    println!(
        "#   travel={}s/floor, service={}s, λ={}/s",
        base.floor_travel_time, base.service_time, base.arrival_rate
    );
    println!(
        "#   simT={}s, dt={}s, seed={}",
        base.sim_t, base.step_size, base.seed
    );

    let schedule = build_schedule(&base);
    println!("#   schedule has {} arrivals", schedule.len());

    let modes = ["uncoordinated", "coordinated", "coordinated-pickup"];
    let runs: Vec<String> = match std::env::var("DISPATCH") {
        Ok(d) => vec![d],
        Err(_) => modes.iter().map(|s| s.to_string()).collect(),
    };
    if std::env::var("ANIMATE").as_deref() == Ok("1") {
        // PORT NOTE: elevator animation scene not ported; see header.
        println!("# (ANIMATE=1 requested but elevator scene not ported — animation skipped)");
    }

    let mut results: Vec<ElevatorResult> = Vec::new();
    for mode in &runs {
        let mut cfg = base.clone();
        cfg.dispatch_mode = mode.clone();
        let t0 = std::time::Instant::now();
        let result = run_elevator(cfg, schedule.clone());
        let ms = t0.elapsed().as_millis();
        let a = result.aggregates.clone();
        println!();
        println!("# dispatchMode = {:<20} ({ms} ms)", mode);
        println!("#   served {}/{} people", a.n_served, a.n);
        println!(
            "#   meanWait   = {:.2} s     p95Wait  = {:.2} s",
            a.mean_wait, a.p95_wait
        );
        println!("#   meanTravel = {:.2} s", a.mean_travel);
        println!(
            "#   meanTotal  = {:.2} s     p95Total = {:.2} s",
            a.mean_total, a.p95_total
        );
        results.push(result);
    }

    if results.len() >= 2 {
        println!();
        println!("# pairwise improvements (vs uncoordinated baseline, results[0]):");
        let baseline = results[0].aggregates.clone();
        for r in results.iter().skip(1) {
            let a = &r.aggregates;
            let d_wait = (a.mean_wait / baseline.mean_wait - 1.0) * 100.0;
            let d_p95 = (a.p95_wait / baseline.p95_wait - 1.0) * 100.0;
            let d_total = (a.mean_total / baseline.mean_total - 1.0) * 100.0;
            println!(
                "#   {:<20} meanWait {:.1}% , p95Wait {:.1}% , meanTotal {:.1}%",
                r.config.dispatch_mode, d_wait, d_p95, d_total
            );
        }
    }

    let _ = std::fs::create_dir_all("out");
    let out_path = "out/elevator-framework.json";
    let framework = jobj(vec![
        ("config", config_json(&results[0].config)),
        ("schedule", schedule_json(&results[0].schedule)),
        ("people", people_json(&results[0].people)),
        ("aggregates", aggregates_json(&results[0].aggregates)),
    ]);
    let _ = std::fs::write(out_path, framework.to_string());
    println!("# wrote {out_path}");

    if results.len() >= 2 {
        let cmp_path = "out/elevator-dispatch-comparison.json";
        let runs_json = JsonValue::Array(
            results
                .iter()
                .map(|r| {
                    jobj(vec![
                        ("dispatchMode", jstr(&r.config.dispatch_mode)),
                        ("aggregates", aggregates_json(&r.aggregates)),
                        ("people", people_json(&r.people)),
                    ])
                })
                .collect(),
        );
        let cmp = jobj(vec![
            ("schedule", schedule_json(&schedule)),
            ("runs", runs_json),
        ]);
        let _ = std::fs::write(cmp_path, cmp.to_string());
        println!("# wrote {cmp_path}");
    }
}
