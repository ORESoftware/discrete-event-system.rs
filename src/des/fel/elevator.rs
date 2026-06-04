//! A **Future Event List** elevator simulation — additive, next-event style.
//!
//! This is a brand-new FEL model of a three-shaft elevator bank serving a building;
//! it does **not** touch the existing time-stepped elevator (`crate::des::
//! main_elevator` / `main_elevator_highrise`). Events drive everything:
//!
//! * **passenger arrival** at a floor (Poisson stream) with a random destination,
//! * **car step** (one shaft's car finishes traversing one floor),
//! * **doors close** (after a dwell during which passengers board/alight).
//!
//! Each shaft runs a **LOOK** (collective-control) policy: keep moving in the
//! current direction while claimed calls/destinations lie ahead, reverse when
//! there are none, and idle when the bank is empty (the FEL simply skips forward
//! to the next arrival — no idle ticking).
//!
//! Alongside the simulator this module ships canonical **MDP** and **POMDP**
//! formulations of the elevator-dispatch *decision* problem ([`elevator_mdp_spec`],
//! [`elevator_pomdp_spec`]) as `des/mdp/v1` / `des/pomdp/v1` specs that run on the
//! existing first-class model citizens (`crate::des::model`).

use std::collections::VecDeque;

use serde_json::{json, Value};

use super::engine::Engine;

// --- tiny self-contained SplitMix64 RNG (deterministic, sim-local) ---------
struct Rng {
    s: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { s: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (9_007_199_254_740_992.0 + 1.0)
    }
    fn exp(&mut self, rate: f64) -> f64 {
        -self.unit().ln() / rate
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir {
    Up,
    Down,
    Idle,
}
impl Dir {
    fn label(self) -> &'static str {
        match self {
            Dir::Up => "up",
            Dir::Down => "down",
            Dir::Idle => "idle",
        }
    }
}

struct Passenger {
    dest: usize,
    arrived: f64,
}

/// Per-shaft mutable car state.
#[derive(Clone)]
struct CarState {
    floor: usize,
    dir: Dir,
    doors_open: bool,
    active: bool,
    onboard: Vec<usize>,
}

impl CarState {
    fn new() -> Self {
        CarState {
            floor: 0,
            dir: Dir::Idle,
            doors_open: false,
            active: false,
            onboard: Vec::new(),
        }
    }
}

/// Mutable FEL elevator world.
struct ElevWorld {
    rng: Rng,
    floors: usize,
    capacity: usize,
    travel: f64,
    dwell: f64,
    arrival_rate: f64,
    cars: Vec<CarState>,
    waiting: Vec<VecDeque<Passenger>>,
    served: u64,
    boarded: u64,
    arrivals: u64,
    total_wait: f64,
    frames: Vec<Value>,
}

impl ElevWorld {
    fn new(
        seed: u64,
        floors: usize,
        shafts: usize,
        capacity: usize,
        travel: f64,
        dwell: f64,
        rate: f64,
    ) -> Self {
        ElevWorld {
            rng: Rng::new(seed),
            floors,
            capacity,
            travel,
            dwell,
            arrival_rate: rate,
            cars: (0..shafts).map(|_| CarState::new()).collect(),
            waiting: (0..floors).map(|_| VecDeque::new()).collect(),
            served: 0,
            boarded: 0,
            arrivals: 0,
            total_wait: 0.0,
            frames: Vec::new(),
        }
    }
    fn waiting_counts(&self) -> Vec<usize> {
        self.waiting.iter().map(|q| q.len()).collect()
    }

    fn total_in_cars(&self) -> usize {
        self.cars.iter().map(|car| car.onboard.len()).sum()
    }

    fn active_cars(&self) -> usize {
        self.cars
            .iter()
            .filter(|car| car.active || car.doors_open)
            .count()
    }

    fn fleet_label(&self) -> String {
        let active = self.active_cars();
        if active == 0 {
            "idle".to_string()
        } else {
            format!("{active} active")
        }
    }

    fn car_frames(&self) -> Vec<Value> {
        self.cars
            .iter()
            .enumerate()
            .map(|(i, car)| {
                json!({
                    "id": i,
                    "floor": car.floor,
                    "dir": car.dir.label(),
                    "doors": car.doors_open,
                    "active": car.active,
                    "inCar": car.onboard.len(),
                })
            })
            .collect()
    }

    fn look_distance(&self, car: &CarState, floor: usize) -> f64 {
        if car.floor == floor {
            return 0.0;
        }
        let top = self.floors.saturating_sub(1);
        match car.dir {
            Dir::Idle => car.floor.abs_diff(floor) as f64,
            Dir::Up if floor >= car.floor => (floor - car.floor) as f64,
            Dir::Up => (top - car.floor + top - floor) as f64,
            Dir::Down if floor <= car.floor => (car.floor - floor) as f64,
            Dir::Down => (car.floor + floor) as f64,
        }
    }

    fn hall_owner(&self, floor: usize) -> Option<usize> {
        if self.waiting[floor].is_empty() {
            return None;
        }
        self.cars
            .iter()
            .enumerate()
            .filter(|(_, car)| car.onboard.len() < self.capacity)
            .min_by(|(a_id, a), (b_id, b)| {
                let score_a = self.look_distance(a, floor)
                    + a.onboard.len() as f64 * 0.35
                    + if a.active { 0.15 } else { 0.0 };
                let score_b = self.look_distance(b, floor)
                    + b.onboard.len() as f64 * 0.35
                    + if b.active { 0.15 } else { 0.0 };
                score_a.total_cmp(&score_b).then_with(|| a_id.cmp(b_id))
            })
            .map(|(id, _)| id)
    }

    /// Floors with a pending reason for this shaft to visit: its in-car
    /// drop-offs plus hall calls assigned to it by the shared dispatcher.
    fn targets_for(&self, car_id: usize) -> Vec<usize> {
        let car = &self.cars[car_id];
        (0..self.floors)
            .filter(|&f| car.onboard.contains(&f) || self.hall_owner(f) == Some(car_id))
            .collect()
    }
}

fn record(eng: &mut Engine<ElevWorld>, kind: &str) {
    let w = &eng.world;
    let first = &w.cars[0];
    let frame = json!({
        "t": eng.now(),
        "car": first.floor,
        "dir": w.fleet_label(),
        "doors": w.cars.iter().any(|car| car.doors_open),
        "cars": w.car_frames(),
        "wait": w.waiting_counts(),
        "inCar": w.total_in_cars(),
        "served": w.served,
        "events": eng.events_processed(),
        "kind": kind,
    });
    eng.world.frames.push(frame);
}

fn passenger_arrival(eng: &mut Engine<ElevWorld>) {
    let rate = eng.world.arrival_rate;
    let ia = eng.world.rng.exp(rate);
    eng.schedule_after(ia, passenger_arrival);

    let floors = eng.world.floors;
    let now = eng.now();
    let origin = eng.world.rng.below(floors);
    let mut dest = eng.world.rng.below(floors);
    if dest == origin {
        dest = (dest + 1) % floors;
    }
    eng.world.waiting[origin].push_back(Passenger { dest, arrived: now });
    eng.world.arrivals += 1;
    record(eng, "arrival");

    wake_idle_cars(eng);
}

/// The car has finished traversing one floor in its current direction.
fn car_step(eng: &mut Engine<ElevWorld>, car_id: usize) {
    if car_id >= eng.world.cars.len() {
        return;
    }
    let floors = eng.world.floors;
    let f = eng.world.cars[car_id].floor as i64;
    let nf = match eng.world.cars[car_id].dir {
        Dir::Up => f + 1,
        Dir::Down => f - 1,
        Dir::Idle => f,
    }
    .clamp(0, floors as i64 - 1) as usize;
    eng.world.cars[car_id].floor = nf;
    record(eng, &format!("move:{car_id}"));
    on_arrive(eng, car_id);
}

/// Decide, on arriving at `car_floor`, whether to stop (open doors) or roll on.
fn on_arrive(eng: &mut Engine<ElevWorld>, car_id: usize) {
    let f = eng.world.cars[car_id].floor;
    let need_alight = eng.world.cars[car_id].onboard.contains(&f);
    let need_board = !eng.world.waiting[f].is_empty()
        && eng.world.cars[car_id].onboard.len() < eng.world.capacity
        && eng.world.hall_owner(f) == Some(car_id);
    if need_alight || need_board {
        eng.world.cars[car_id].doors_open = true;
        record(eng, &format!("doors_open:{car_id}"));
        let dwell = eng.world.dwell;
        eng.schedule_after(dwell, move |eng| doors_close(eng, car_id));
    } else {
        decide_step(eng, car_id);
    }
}

/// Doors close: passengers alight (reached destination) and board (FIFO, up to
/// capacity); then re-decide the next move.
fn doors_close(eng: &mut Engine<ElevWorld>, car_id: usize) {
    if car_id >= eng.world.cars.len() {
        return;
    }
    let f = eng.world.cars[car_id].floor;
    let now = eng.now();
    let before = eng.world.cars[car_id].onboard.len();
    eng.world.cars[car_id].onboard.retain(|&d| d != f);
    eng.world.served += (before - eng.world.cars[car_id].onboard.len()) as u64;
    while !eng.world.waiting[f].is_empty()
        && eng.world.cars[car_id].onboard.len() < eng.world.capacity
    {
        let p = eng.world.waiting[f].pop_front().expect("nonempty");
        eng.world.total_wait += now - p.arrived;
        eng.world.boarded += 1;
        eng.world.cars[car_id].onboard.push(p.dest);
    }
    eng.world.cars[car_id].doors_open = false;
    record(eng, &format!("doors_close:{car_id}"));
    decide_step(eng, car_id);
    wake_idle_cars(eng);
}

/// LOOK dispatch: continue in the current direction while targets lie ahead,
/// else reverse, else idle (and stop scheduling — the FEL skips to the next
/// arrival).
fn decide_step(eng: &mut Engine<ElevWorld>, car_id: usize) {
    if car_id >= eng.world.cars.len() {
        return;
    }
    let targets = eng.world.targets_for(car_id);
    if targets.is_empty() {
        eng.world.cars[car_id].dir = Dir::Idle;
        eng.world.cars[car_id].active = false;
        record(eng, &format!("idle:{car_id}"));
        return;
    }
    let f = eng.world.cars[car_id].floor;
    // A new claimed arrival (or drop-off) at the current floor that this shaft
    // can service now.
    let serviceable_here = eng.world.cars[car_id].onboard.contains(&f)
        || (!eng.world.waiting[f].is_empty()
            && eng.world.cars[car_id].onboard.len() < eng.world.capacity
            && eng.world.hall_owner(f) == Some(car_id));
    if targets.contains(&f) && serviceable_here {
        on_arrive(eng, car_id);
        return;
    }
    let up_ahead = targets.iter().any(|&t| t > f);
    let down_ahead = targets.iter().any(|&t| t < f);
    let dir = match eng.world.cars[car_id].dir {
        Dir::Up if up_ahead => Dir::Up,
        Dir::Up if down_ahead => Dir::Down,
        Dir::Down if down_ahead => Dir::Down,
        Dir::Down if up_ahead => Dir::Up,
        _ => {
            if up_ahead {
                Dir::Up
            } else if down_ahead {
                Dir::Down
            } else {
                Dir::Idle
            }
        }
    };
    if dir == Dir::Idle {
        // Only target is the current floor but it is not serviceable (e.g. car
        // full); idle to avoid spinning.
        eng.world.cars[car_id].dir = Dir::Idle;
        eng.world.cars[car_id].active = false;
        record(eng, &format!("idle:{car_id}"));
        return;
    }
    eng.world.cars[car_id].dir = dir;
    eng.world.cars[car_id].active = true;
    let travel = eng.world.travel;
    eng.schedule_after(travel, move |eng| car_step(eng, car_id));
}

fn wake_idle_cars(eng: &mut Engine<ElevWorld>) {
    let shafts = eng.world.cars.len();
    for car_id in 0..shafts {
        if !eng.world.cars[car_id].active && !eng.world.cars[car_id].doors_open {
            decide_step(eng, car_id);
        }
    }
}

/// Configuration for a FEL elevator run.
pub struct ElevatorConfig {
    pub floors: usize,
    pub shafts: usize,
    pub capacity: usize,
    pub travel: f64,
    pub dwell: f64,
    pub arrival_rate: f64,
    pub horizon: f64,
    pub seed: u64,
}

impl Default for ElevatorConfig {
    fn default() -> Self {
        ElevatorConfig {
            floors: 6,
            shafts: 3,
            capacity: 8,
            travel: 1.5,
            dwell: 2.5,
            arrival_rate: 0.55,
            horizon: 120.0,
            seed: 0xE1E4_A705_u64,
        }
    }
}

/// Run the FEL elevator and return `{ meta, frames }` ready for the animation
/// renderer.
pub fn run_fel_elevator(cfg: &ElevatorConfig) -> Value {
    let floors = cfg.floors.max(2);
    let shafts = cfg.shafts.max(1);
    let capacity = cfg.capacity.max(1);
    let travel = cfg.travel.max(0.0);
    let dwell = cfg.dwell.max(0.0);
    let arrival_rate = cfg.arrival_rate.max(f64::MIN_POSITIVE);
    let horizon = cfg.horizon.max(0.0);
    let mut eng = Engine::new(ElevWorld::new(
        cfg.seed,
        floors,
        shafts,
        capacity,
        travel,
        dwell,
        arrival_rate,
    ));
    record(&mut eng, "start");
    let first = eng.world.rng.exp(arrival_rate);
    eng.schedule_after(first, passenger_arrival);
    eng.run_until(horizon);

    let events = eng.events_processed();
    let w = &eng.world;
    let mean_wait = if w.boarded > 0 {
        w.total_wait / w.boarded as f64
    } else {
        0.0
    };
    // Final hold frame.
    let first = &w.cars[0];
    let final_frame = json!({
        "t": horizon,
        "car": first.floor,
        "dir": w.fleet_label(),
        "doors": w.cars.iter().any(|car| car.doors_open),
        "cars": w.car_frames(),
        "wait": w.waiting_counts(),
        "inCar": w.total_in_cars(),
        "served": w.served,
        "events": events, "kind": "end",
    });
    let mut frames = eng.world.frames;
    frames.push(final_frame);

    json!({
        "meta": {
            "floors": floors,
            "shafts": shafts,
            "capacity": capacity,
            "travel": travel,
            "dwell": dwell,
            "arrivalRate": arrival_rate,
            "horizon": horizon,
            "events": events,
            "arrivals": eng.world.arrivals,
            "served": eng.world.served,
            "meanWait": mean_wait,
        },
        "frames": frames,
    })
}

// ===========================================================================
// MDP / POMDP elevator-dispatch models (canonical specs for the model registry)
// ===========================================================================

/// A fully-observed elevator-dispatch **MDP** (`des/mdp/v1`).
///
/// State = `(carFloor, callIndex)` where `callIndex` in `0..floors` is a pending
/// hall call at that floor and `floors` means "no call" — `floors·(floors+1)`
/// states. Actions: `up`, `down`, `serve`. Serving the pending floor pays `+10`
/// and (stochastically) admits the next call; moving/idling while a passenger
/// waits costs time. Value iteration recovers the obvious "drive to the call and
/// serve it" policy.
pub fn elevator_mdp_spec() -> Value {
    let floors = 3usize;
    let none = floors; // "no call" sentinel
    let calls = floors + 1; // call indices 0..floors plus "none"
    let num_states = floors * calls;
    let p_new = 0.6; // a new call appears soon after one is served
    let p_arrive = 0.3; // calls trickle in while the car is idle
    let enc = |f: usize, c: usize| f * calls + c;

    let mut transitions: Vec<Vec<Vec<Value>>> = Vec::with_capacity(num_states);
    let mut state_labels: Vec<String> = Vec::with_capacity(num_states);

    for f in 0..floors {
        for c in 0..calls {
            let call = if c == none {
                "idle".to_string()
            } else {
                format!("call@{c}")
            };
            state_labels.push(format!("f{f}·{call}"));

            let mut actions: Vec<Vec<Value>> = Vec::with_capacity(3);
            for a in 0..3 {
                let f2 = match a {
                    0 => (f + 1).min(floors - 1),
                    1 => f.saturating_sub(1),
                    _ => f,
                };
                let pending = c != none;
                let mut outs: Vec<Value> = Vec::new();
                if a == 2 && pending && c == f {
                    // Served the call at the current floor.
                    let each = p_new / floors as f64;
                    for j in 0..floors {
                        outs.push(json!({"prob": each, "reward": 10.0, "next": enc(f2, j)}));
                    }
                    outs.push(json!({"prob": 1.0 - p_new, "reward": 10.0, "next": enc(f2, none)}));
                } else if pending {
                    // Call still waiting: pay time; serving the wrong floor wastes a cycle.
                    let reward = if a == 2 { -2.0 } else { -1.0 };
                    outs.push(json!({"prob": 1.0, "reward": reward, "next": enc(f2, c)}));
                } else {
                    // No call: exogenous arrivals trickle in.
                    let reward = if a == 2 { -1.0 } else { -0.3 };
                    let each = p_arrive / floors as f64;
                    for j in 0..floors {
                        outs.push(json!({"prob": each, "reward": reward, "next": enc(f2, j)}));
                    }
                    outs.push(
                        json!({"prob": 1.0 - p_arrive, "reward": reward, "next": enc(f2, none)}),
                    );
                }
                actions.push(outs);
            }
            transitions.push(actions);
        }
    }

    json!({
        "$schema": "des/mdp/v1",
        "numStates": num_states,
        "discount": 0.9,
        "transitions": transitions,
        "terminal": [],
        "stateLabels": state_labels,
        "actionLabels": ["up", "down", "serve"],
    })
}

/// A partially-observed elevator-dispatch **POMDP** (`des/pomdp/v1`).
///
/// Hidden demand at a service floor is `empty` / `waiting` / `crowded`. The
/// controller only sees a **noisy hall-call button** (`quiet` / `call`) — it
/// cannot tell `waiting` from `crowded`, and the button false-triggers/misses.
/// Actions: `hold` (let demand build, paying a waiting cost) or `dispatch`
/// (serve — big payoff if demand exists, a wasted trip if not). Belief tracking
/// over the three hidden states drives the dispatch decision.
pub fn elevator_pomdp_spec() -> Value {
    json!({
        "$schema": "des/pomdp/v1",
        "numStates": 3,
        "numActions": 2,
        "numObservations": 2,
        "discount": 0.95,
        "transition": [
            [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.6, 0.4], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ],
        "observation": [
            [[0.85, 0.15], [0.85, 0.15]],
            [[0.30, 0.70], [0.30, 0.70]],
            [[0.10, 0.90], [0.10, 0.90]]
        ],
        "reward": [
            [0.0, -3.0],
            [-1.0, 8.0],
            [-3.0, 15.0]
        ],
        "initialBelief": [1.0, 0.0, 0.0],
        "stateLabels": ["empty", "waiting", "crowded"],
        "actionLabels": ["hold", "dispatch"],
        "observationLabels": ["quiet", "call"]
    })
}

/// Render a FEL-elevator `{ meta, frames }` data object to a self-contained
/// animated HTML page (vertical shafts, boarding/alighting, live charts).
pub fn render_elevator_html(data: &Value) -> String {
    let data_str = serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string());
    HTML_TEMPLATE.replace("__DES_DATA__", &data_str)
}

const HTML_TEMPLATE: &str = r####"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FEL elevator simulation</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;background:#0b1021;color:#e6edf3}
main{max-width:1080px;margin:0 auto;padding:22px 20px 60px}
h1{font-size:1.4rem;margin:0 0 4px}
.sub{color:#9aa4b2;margin:0 0 14px;font-size:.9rem;line-height:1.5;max-width:82ch}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}
.chip{font-size:.78rem;border:1px solid #2b3344;border-radius:6px;padding:3px 9px;color:#c9d4e3;background:#0f1422}
.chip b{color:#fff}
.stage{display:grid;grid-template-columns:340px 1fr;gap:16px}
.panel{border:1px solid #21262d;border-radius:12px;background:#0f1422;padding:14px}
.stats{display:flex;gap:16px;flex-wrap:wrap;font-size:.82rem;color:#9aa4b2;margin:0 0 10px}
.stats b{color:#e6edf3;font-variant-numeric:tabular-nums}
.clock{color:#fff;font-weight:600}
svg{display:block;width:100%}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:16px;border-top:1px solid #21262d;padding-top:14px}
button{font:inherit;font-size:.85rem;cursor:pointer;border-radius:8px;padding:7px 12px;border:1px solid #2b3344;background:#161b22;color:#e6edf3}
button:hover{border-color:#3b82f6}
button.primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
.controls label{font-size:.8rem;color:#9aa4b2;display:inline-flex;align-items:center;gap:8px}
#scrub{flex:1;min-width:160px}
.legend{display:flex;gap:16px;align-items:center;font-size:.78rem;color:#9aa4b2;margin:14px 0 4px}
.legend .sw{width:14px;height:3px;border-radius:2px;display:inline-block}
</style>
</head>
<body>
<main>
<h1>FEL elevator &mdash; next-event simulation</h1>
<p class="sub">Three elevator shafts serving one building under a shared <b>LOOK</b> (collective-control) policy, simulated on the engine's <b>future-event-list</b> scheduler: the clock jumps from one event to the next (passenger arrival, car-step across a floor, doors close) and skips all idle time. Watch passengers queue on each landing (left dots), then get claimed by the shaft that can naturally reach them.</p>
<div class="chips" id="chips"></div>

<div class="stage">
  <div class="panel">
    <div class="stats">
      <span>clock <b class="clock" id="clock">0.0s</b></span>
      <span>events <b id="events">0</b></span>
      <span>fleet <b id="dir">idle</b></span>
    </div>
    <svg id="bld" viewBox="0 0 360 420"></svg>
  </div>
  <div class="panel">
    <div class="stats">
      <span>in&nbsp;car <b id="incar">0</b></span>
      <span>waiting <b id="waiting">0</b></span>
      <span>served <b id="served">0</b></span>
    </div>
    <div class="legend">
      <span><span class="sw" style="background:#f59e0b"></span>waiting (total)</span>
      <span><span class="sw" style="background:#38bdf8"></span>in car</span>
    </div>
    <svg id="chart" viewBox="0 0 700 360"></svg>
  </div>
</div>

<div class="controls">
  <button class="primary" id="play">&#9654; Play</button>
  <button id="stepb">&#9664;| step</button>
  <button id="stepf">|&#9654; step</button>
  <label>speed <input type="range" id="speed" min="0.5" max="16" step="0.5" value="5"> <span id="speedv">5&times;</span></label>
  <label>t <input type="range" id="scrub" min="0" max="1000" step="1" value="0"></label>
</div>
</main>

<script>
const DATA = __DES_DATA__;
const M = DATA.meta, F = M.floors, T = M.horizon, FR = DATA.frames;

(function(){
  const c=document.getElementById('chips');
  const shafts = M.shafts || 1;
  const it=[['floors',F],['shafts',shafts],['capacity/shaft',M.capacity],['arrival &lambda;',M.arrivalRate+' /s'],
    ['travel',M.travel+' s/floor'],['dwell',M.dwell+' s'],['horizon',T+' s'],
    ['FEL events',M.events.toLocaleString()],['arrivals',M.arrivals],['served',M.served],
    ['mean wait',M.meanWait.toFixed(1)+' s']];
  c.innerHTML=it.map(function(p){return '<span class="chip">'+p[0]+' <b>'+p[1]+'</b></span>';}).join('');
})();

function frameAt(t){
  let lo=0,hi=FR.length-1,ans=0;
  while(lo<=hi){const m=(lo+hi)>>1; if(FR[m].t<=t){ans=m;lo=m+1;} else hi=m-1;}
  return FR[ans];
}

// ---- building renderer ----
const BW=360, BH=420, MTOP=18, MBOT=22, SHX=148, SHW=42, SHG=8;
const lane = (BH-MTOP-MBOT)/F;
function floorY(f){ return MTOP + (F-1-f)*lane; }  // floor 0 at the bottom
function carsOf(fr){
  return fr.cars || [{id:0,floor:fr.car,dir:fr.dir,doors:fr.doors,active:fr.dir!=='idle',inCar:fr.inCar}];
}
function drawBuilding(fr){
  let s='';
  const cars=carsOf(fr);
  // floor lanes + landings
  for(let f=0;f<F;f++){
    const y=floorY(f);
    s+='<line x1="20" y1="'+(y+lane)+'" x2="'+(BW-14)+'" y2="'+(y+lane)+'" stroke="#1b2230"/>';
    s+='<text x="22" y="'+(y+lane/2+4)+'" fill="#6b7689" font-size="11">F'+f+'</text>';
    // waiting dots on the landing (left of the shaft)
    const w=fr.wait[f];
    const shown=Math.min(w,6);
    for(let k=0;k<shown;k++){
      s+='<circle cx="'+(SHX-14-k*11)+'" cy="'+(y+lane/2)+'" r="4.2" fill="#f59e0b" opacity="'+(0.5+0.5*(1-k/6))+'"/>';
    }
    if(w>6){ s+='<text x="'+(SHX-14-6*11)+'" y="'+(y+lane/2+4)+'" fill="#f59e0b" font-size="10" text-anchor="end">+'+(w-6)+'</text>'; }
  }
  for(let i=0;i<cars.length;i++){
    const sx=SHX+i*(SHW+SHG);
    s+='<rect x="'+sx+'" y="'+MTOP+'" width="'+SHW+'" height="'+(BH-MTOP-MBOT)+'" fill="#0c1120" stroke="#283142"/>';
    s+='<text x="'+(sx+SHW/2)+'" y="'+(BH-5)+'" fill="#6b7689" font-size="10" text-anchor="middle">S'+(i+1)+'</text>';
  }
  for(let i=0;i<cars.length;i++){
    const car=cars[i], sx=SHX+i*(SHW+SHG), cy=floorY(car.floor);
    const ch=Math.max(18,lane-8);
    const open=car.doors;
    const carColor = open? '#15803d' : (car.active? '#1f6feb' : '#334155');
    s+='<rect x="'+(sx+5)+'" y="'+(cy+4)+'" width="'+(SHW-10)+'" height="'+ch+'" rx="5" fill="'+carColor+'" stroke="#3b4757" stroke-width="1.5"/>';
    if(open){
      const midx=sx+SHW/2;
      s+='<line x1="'+midx+'" y1="'+(cy+5)+'" x2="'+midx+'" y2="'+(cy+3+ch)+'" stroke="#0c1120" stroke-width="5"/>';
    }
    s+='<text x="'+(sx+SHW/2)+'" y="'+(cy+4+ch/2+5)+'" fill="#fff" font-size="12" font-weight="700" text-anchor="middle">'+car.inCar+'</text>';
    const arrow = car.dir==='up'? '\u25b2' : (car.dir==='down'? '\u25bc' : '\u25cf');
    s+='<text x="'+(sx+SHW/2)+'" y="'+(cy+3)+'" fill="#9ecbff" font-size="10" text-anchor="middle">'+arrow+'</text>';
  }
  document.getElementById('bld').innerHTML=s;
}

// ---- chart: waiting + in-car over time ----
const CW=700, CH=360, ML=34, MR=12, MT=10, MB=24;
const PW=CW-ML-MR, PH=CH-MT-MB;
let maxY=2;
for(const f of FR){ let tot=f.wait.reduce(function(a,b){return a+b;},0); maxY=Math.max(maxY, tot, f.inCar); }
maxY=Math.ceil(maxY+1);
function cx(t){return ML+PW*(t/T);}
function cyv(v){return MT+PH*(1-v/maxY);}
function path(key){
  let d='',px=null,py=null;
  for(const f of FR){
    const v = key==='wait'? f.wait.reduce(function(a,b){return a+b;},0) : f.inCar;
    const X=cx(f.t),Y=cyv(v);
    if(px===null){d='M'+X+' '+Y;} else {d+=' L'+X+' '+py+' L'+X+' '+Y;}
    px=X;py=Y;
  }
  return d;
}
const waitPath=path('wait'), carPath=path('inCar');
function drawChart(t){
  let s='';
  s+='<line x1="'+ML+'" y1="'+cyv(0)+'" x2="'+(CW-MR)+'" y2="'+cyv(0)+'" stroke="#30363d"/>';
  for(let g=0;g<=maxY;g+=Math.max(1,Math.round(maxY/5))){
    s+='<line x1="'+ML+'" y1="'+cyv(g)+'" x2="'+(CW-MR)+'" y2="'+cyv(g)+'" stroke="#1b2230"/>';
    s+='<text x="'+(ML-6)+'" y="'+(cyv(g)+4)+'" fill="#6b7689" font-size="10" text-anchor="end">'+g+'</text>';
  }
  for(let g=0;g<=T;g+=20){ s+='<text x="'+cx(g)+'" y="'+(CH-7)+'" fill="#6b7689" font-size="10" text-anchor="middle">'+g+'s</text>'; }
  s+='<path d="'+waitPath+'" fill="none" stroke="#f59e0b" stroke-width="1.5"/>';
  s+='<path d="'+carPath+'" fill="none" stroke="#38bdf8" stroke-width="1.5"/>';
  const fx=cx(t);
  s+='<line x1="'+fx+'" y1="'+MT+'" x2="'+fx+'" y2="'+cyv(0)+'" stroke="#e6edf3" stroke-width="1" opacity="0.5"/>';
  document.getElementById('chart').innerHTML=s;
}

// ---- playback ----
let tPlay=0,playing=false,last=null;
const scrub=document.getElementById('scrub');
function render(){
  const fr=frameAt(tPlay);
  drawBuilding(fr); drawChart(tPlay);
  document.getElementById('clock').textContent=tPlay.toFixed(1)+'s';
  document.getElementById('events').textContent=fr.events.toLocaleString();
  const active=carsOf(fr).filter(function(c){return c.active || c.doors;}).length;
  document.getElementById('dir').textContent=active===0?'idle':active+' active';
  document.getElementById('incar').textContent=fr.inCar;
  document.getElementById('waiting').textContent=fr.wait.reduce(function(a,b){return a+b;},0);
  document.getElementById('served').textContent=fr.served;
  scrub.value=Math.round(1000*tPlay/T);
}
function tick(ts){
  if(!playing){last=null;return;}
  if(last===null)last=ts;
  const dtr=(ts-last)/1000; last=ts;
  tPlay+=dtr*parseFloat(document.getElementById('speed').value);
  if(tPlay>=T){tPlay=T;playing=false;document.getElementById('play').innerHTML='&#9654; Play';}
  render();
  if(playing)requestAnimationFrame(tick);
}
document.getElementById('play').onclick=function(){
  if(tPlay>=T)tPlay=0;
  playing=!playing;
  this.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';
  if(playing)requestAnimationFrame(tick);
};
document.getElementById('speed').oninput=function(){document.getElementById('speedv').textContent=this.value+'\u00d7';};
scrub.oninput=function(){playing=false;document.getElementById('play').innerHTML='&#9654; Play';tPlay=T*this.value/1000;render();};
function stepTo(dir){
  playing=false;document.getElementById('play').innerHTML='&#9654; Play';
  let idx=0; for(let i=0;i<FR.length;i++){if(FR[i].t<=tPlay+1e-9)idx=i;else break;}
  idx=Math.min(FR.length-1,Math.max(0,idx+dir)); tPlay=FR[idx].t; render();
}
document.getElementById('stepf').onclick=function(){stepTo(1);};
document.getElementById('stepb').onclick=function(){stepTo(-1);};
render();
</script>
</body>
</html>
"####;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::model::with_builtins;

    #[test]
    fn fel_elevator_serves_passengers_and_records_frames() {
        let data = run_fel_elevator(&ElevatorConfig {
            horizon: 60.0,
            ..Default::default()
        });
        let served = data["meta"]["served"].as_u64().unwrap();
        let frames = data["frames"].as_array().unwrap();
        let floors = data["meta"]["floors"].as_u64().unwrap() as usize;
        let shafts = data["meta"]["shafts"].as_u64().unwrap() as usize;
        assert_eq!(shafts, 3, "default FEL elevator should use three shafts");
        assert!(served > 0, "expected some passengers served");
        assert!(frames.len() > 10, "expected a frame stream");
        // No car may ever be outside the building.
        for f in frames {
            let cars = f["cars"].as_array().expect("frame includes shaft states");
            assert_eq!(cars.len(), shafts, "frame should carry every shaft");
            for car in cars {
                let floor = car["floor"].as_u64().unwrap() as usize;
                assert!(floor < floors, "car floor out of range: {floor}");
            }
        }
    }

    #[test]
    fn elevator_mdp_spec_validates_and_solves() {
        let spec = elevator_mdp_spec();
        let art = with_builtins()
            .run("mdp", &spec)
            .expect("elevator MDP solves");
        assert_eq!(art.kind, "mdp");
        assert!(!art.frames.is_empty(), "MDP produced rollout frames");
    }

    #[test]
    fn elevator_pomdp_spec_validates_and_solves() {
        let spec = elevator_pomdp_spec();
        let art = with_builtins()
            .run("pomdp", &spec)
            .expect("elevator POMDP solves");
        assert_eq!(art.kind, "pomdp");
        assert!(!art.frames.is_empty(), "POMDP produced belief frames");
    }
}
