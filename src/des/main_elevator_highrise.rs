//! Port of `src/des/main-elevator-highrise.ts`.
//!
//! 50-floor, 6-shaft elevator model exploring dispatch-policy tradeoffs across
//! decision authorities. Pre-solves MDP dispatch tunings with value iteration
//! (`crate::des::general::value_iteration`), runs each policy/authority pair,
//! prints diagnostics, and writes a JSON results artifact.
//!
//! `mulberry32`/`withSeed` → `crate::des::general::prng`; `process.env` →
//! `std::env`; `fs`/`path` → `std::fs`; JSON → `observability::logger::JsonValue`.
//!
//! PORT NOTES:
//!   * passengers are shared, mutable objects in the TS source (the same object
//!     lives in `people`, a floor queue, then a car). That identity is preserved
//!     with `Rc<RefCell<HighrisePassenger>>`.
//!   * `SmartMovable`/`TimeSteppedStation` bases are inlined; per-car stepping is
//!     split into free functions so `floors`, `completed`, `mdp_decision_log`,
//!     and the active car are borrowed as disjoint `self` fields.
//!   * the HTML animation (`animation/types`, `animation/html-player`, frames,
//!     shapes, charts, `buildHTMLSet`) is NOT ported: frame/series recording is
//!     omitted and a placeholder `.html` is written. The full data artifact
//!     (`elevator-highrise-results.json`) is produced faithfully.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::des::general::prng::{mulberry32, with_seed};
use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};
use crate::des::observability::logger::JsonValue;
use crate::des::shared::capabilities::RandomSource;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Enums (TS string unions).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum HighrisePolicy {
    FewestStops,
    LowestTotalTime,
    EnergyEfficient,
    CenterPreposition,
    ZonedService,
    MdpCallOnly,
    MdpTuned,
}
impl HighrisePolicy {
    fn slug(self) -> &'static str {
        match self {
            HighrisePolicy::FewestStops => "fewest-stops",
            HighrisePolicy::LowestTotalTime => "lowest-total-time",
            HighrisePolicy::EnergyEfficient => "energy-efficient",
            HighrisePolicy::CenterPreposition => "center-preposition",
            HighrisePolicy::ZonedService => "zoned-service",
            HighrisePolicy::MdpCallOnly => "mdp-call-only",
            HighrisePolicy::MdpTuned => "mdp-tuned",
        }
    }
    fn from_slug(s: &str) -> Option<Self> {
        Some(match s {
            "fewest-stops" => HighrisePolicy::FewestStops,
            "lowest-total-time" => HighrisePolicy::LowestTotalTime,
            "energy-efficient" => HighrisePolicy::EnergyEfficient,
            "center-preposition" => HighrisePolicy::CenterPreposition,
            "zoned-service" => HighrisePolicy::ZonedService,
            "mdp-call-only" => HighrisePolicy::MdpCallOnly,
            "mdp-tuned" => HighrisePolicy::MdpTuned,
            _ => return None,
        })
    }
    fn label(self) -> &'static str {
        match self {
            HighrisePolicy::FewestStops => "Fewest stops",
            HighrisePolicy::LowestTotalTime => "Lowest total per-person time",
            HighrisePolicy::EnergyEfficient => "Energy efficient",
            HighrisePolicy::CenterPreposition => "Center preposition",
            HighrisePolicy::ZonedService => "Zoned / even-odd service",
            HighrisePolicy::MdpCallOnly => "MDP no queue info",
            HighrisePolicy::MdpTuned => "MDP destination dispatch",
        }
    }
    fn summary(self) -> &'static str {
        match self {
            HighrisePolicy::FewestStops => "Batches riders by destination and avoids intermediate pickups once occupied.",
            HighrisePolicy::LowestTotalTime => "Dispatches the best nearby car and accepts useful same-direction pickups.",
            HighrisePolicy::EnergyEfficient => "Penalizes travel, starts, and stop churn; prefers existing motion and batching.",
            HighrisePolicy::CenterPreposition => "Uses lowest-time dispatch, then parks idle cars near the building center.",
            HighrisePolicy::ZonedService => "Constrains shafts to all, low, mid, high, even, and odd service patterns.",
            HighrisePolicy::MdpCallOnly => "Uses value iteration with only binary hall-call, age, direction, and car-distance observations.",
            HighrisePolicy::MdpTuned => "Uses value iteration with destination-dispatch counts and destination-group estimates.",
        }
    }
}

pub const HIGHRISE_POLICIES: [HighrisePolicy; 7] = [
    HighrisePolicy::FewestStops,
    HighrisePolicy::LowestTotalTime,
    HighrisePolicy::EnergyEfficient,
    HighrisePolicy::CenterPreposition,
    HighrisePolicy::ZonedService,
    HighrisePolicy::MdpCallOnly,
    HighrisePolicy::MdpTuned,
];

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecisionAuthority {
    Central,
    Decentralized,
    Hybrid,
}
impl DecisionAuthority {
    fn slug(self) -> &'static str {
        match self {
            DecisionAuthority::Central => "central",
            DecisionAuthority::Decentralized => "decentralized",
            DecisionAuthority::Hybrid => "hybrid",
        }
    }
    fn from_slug(s: &str) -> Option<Self> {
        Some(match s {
            "central" => DecisionAuthority::Central,
            "decentralized" => DecisionAuthority::Decentralized,
            "hybrid" => DecisionAuthority::Hybrid,
            _ => return None,
        })
    }
    fn label(self) -> &'static str {
        match self {
            DecisionAuthority::Central => "Central brain",
            DecisionAuthority::Decentralized => "Smart movables",
            DecisionAuthority::Hybrid => "Hybrid",
        }
    }
    fn summary(self) -> &'static str {
        match self {
            DecisionAuthority::Central => "One global controller claims requests and coordinates shafts.",
            DecisionAuthority::Decentralized => "Each elevator chooses from its local sensor view; duplicate claims are allowed.",
            DecisionAuthority::Hybrid => "The controller handles urgent calls while idle cars make local decisions.",
        }
    }
}

pub const DECISION_AUTHORITIES: [DecisionAuthority; 3] =
    [DecisionAuthority::Central, DecisionAuthority::Decentralized, DecisionAuthority::Hybrid];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MDPObservability {
    CallOnly,
    DestinationDispatch,
}
impl MDPObservability {
    fn slug(self) -> &'static str {
        match self {
            MDPObservability::CallOnly => "call-only",
            MDPObservability::DestinationDispatch => "destination-dispatch",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CarState {
    Idle,
    Moving,
    Serving,
    Prepositioning,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TargetReason {
    Pickup,
    Dropoff,
    Home,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DecisionSource {
    Central,
    Local,
    Hybrid,
    None,
}
impl DecisionSource {
    fn slug(self) -> &'static str {
        match self {
            DecisionSource::Central => "central",
            DecisionSource::Local => "local",
            DecisionSource::Hybrid => "hybrid",
            DecisionSource::None => "none",
        }
    }
}

// ---------------------------------------------------------------------------
// Config / result records.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct HighriseElevatorConfig {
    pub n_floors: i32,
    pub n_elevators: i32,
    pub capacity: i32,
    pub floor_travel_time: f64,
    pub service_time: f64,
    pub arrival_rate: f64,
    pub sim_t: f64,
    pub drain_t: f64,
    pub step_size: f64,
    pub seed: u32,
    pub local_sensor_radius: f64,
    pub urgent_wait_threshold: f64,
}

#[derive(Clone, Copy)]
struct ScheduledArrival {
    t: f64,
    from_floor: i32,
    to_floor: i32,
}

#[derive(Clone)]
struct HighrisePassengerSnapshot {
    id: i64,
    from_floor: i32,
    to_floor: i32,
    arrival_time: f64,
    board_time: f64,
    exit_time: f64,
}

#[derive(Clone)]
struct HighriseAggregates {
    n: usize,
    n_served: usize,
    mean_wait: f64,
    mean_travel: f64,
    mean_total: f64,
    p95_wait: f64,
    p95_total: f64,
    total_stops: f64,
    total_distance_floors: f64,
    total_energy: f64,
    timed_out: usize,
}

#[derive(Clone)]
struct HighriseElevatorResult {
    policy: HighrisePolicy,
    authority: DecisionAuthority,
    config: HighriseElevatorConfig,
    schedule: Vec<ScheduledArrival>,
    people: Vec<HighrisePassengerSnapshot>,
    aggregates: HighriseAggregates,
    mdp_tuning: Option<MDPDispatchTuningSummary>,
    mdp_run: Option<MDPRunDiagnostics>,
    marginal_vs_lowest_time: Option<MarginalComparison>,
}

#[derive(Clone)]
struct DispatchScoreWeights {
    distance: f64,
    trip: f64,
    queue: f64,
    wait: f64,
    same_direction: f64,
    destination_group: f64,
}

struct PickupFeatures {
    distance: f64,
    oldest_wait: f64,
    queue_len: f64,
    trip: f64,
    same_side: f64,
    max_group: f64,
}

#[derive(Clone)]
struct MDPActionProfile {
    label: &'static str,
    weights: DispatchScoreWeights,
}

#[derive(Clone)]
struct MDPDispatchTuning {
    observability: MDPObservability,
    num_states: usize,
    actions: Vec<MDPActionProfile>,
    policy: Vec<i32>,
    gamma: f64,
    iterations: usize,
    final_delta: f64,
    learned_weights: DispatchScoreWeights,
    state_labels: Vec<String>,
    action_labels: Vec<String>,
}

#[derive(Clone)]
struct StatePolicyRow {
    state: String,
    action: String,
}

#[derive(Clone)]
struct MDPDispatchTuningSummary {
    observability: MDPObservability,
    num_states: usize,
    gamma: f64,
    iterations: usize,
    final_delta: f64,
    learned_weights: DispatchScoreWeights,
    state_policy: Vec<StatePolicyRow>,
}

#[derive(Clone)]
struct ActionCount {
    action: String,
    count: usize,
    share: f64,
}
#[derive(Clone)]
struct TopState {
    state: String,
    action: String,
    count: usize,
}
#[derive(Clone)]
struct MarginalBin {
    bin: String,
    count: usize,
    dominant_action: String,
    share: f64,
}
#[derive(Clone)]
struct MarginalRow {
    variable: String,
    bins: Vec<MarginalBin>,
}
#[derive(Clone)]
struct MDPRunDiagnostics {
    total_decisions: usize,
    action_counts: Vec<ActionCount>,
    top_states: Vec<TopState>,
    marginals: Vec<MarginalRow>,
}

#[derive(Clone)]
struct MarginalComparison {
    baseline_policy: HighrisePolicy,
    baseline_authority: DecisionAuthority,
    mean_wait_delta: f64,
    mean_total_delta: f64,
    stops_delta: f64,
    energy_delta: f64,
}

struct MDPDecisionLogEntry {
    state_id: usize,
    state: String,
    action: String,
    bins: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// Entities.
// ---------------------------------------------------------------------------

struct HighrisePassenger {
    id: i64,
    from_floor: i32,
    to_floor: i32,
    arrival_time: f64,
    board_time: f64,
    exit_time: f64,
}
impl HighrisePassenger {
    fn new(id: i64, from_floor: i32, to_floor: i32, arrival_time: f64) -> Self {
        HighrisePassenger { id, from_floor, to_floor, arrival_time, board_time: -1.0, exit_time: -1.0 }
    }
    fn direction(&self) -> i32 {
        sign((self.to_floor - self.from_floor) as f64)
    }
}

type PaxRef = Rc<RefCell<HighrisePassenger>>;

struct ElevatorCar {
    idx: i32,
    capacity: i32,
    current_floor: f64,
    target_floor: Option<f64>,
    target_reason: Option<TargetReason>,
    pickup_direction: i32,
    direction: i32,
    state: CarState,
    passengers: Vec<PaxRef>,
    service_remaining: f64,
    decision_source: DecisionSource,
    stops: f64,
    starts: f64,
    distance_floors: f64,
    energy: f64,
    allowed_floors: Option<HashSet<i32>>,
}
impl ElevatorCar {
    fn new(idx: i32, start_floor: f64, capacity: i32) -> Self {
        ElevatorCar {
            idx,
            capacity,
            current_floor: start_floor,
            target_floor: None,
            target_reason: None,
            pickup_direction: 0,
            direction: 0,
            state: CarState::Idle,
            passengers: Vec::new(),
            service_remaining: 0.0,
            decision_source: DecisionSource::None,
            stops: 0.0,
            starts: 0.0,
            distance_floors: 0.0,
            energy: 0.0,
            allowed_floors: None,
        }
    }
    fn id(&self) -> String {
        format!("E{}", self.idx)
    }
    fn spare_capacity(&self) -> i32 {
        self.capacity - self.passengers.len() as i32
    }
    fn is_full(&self) -> bool {
        self.spare_capacity() <= 0
    }
    fn set_target(&mut self, floor: f64, reason: TargetReason, pickup_dir: i32, n_floors: i32) {
        let floor = clamp(floor.round(), 0.0, (n_floors - 1) as f64);
        if self.target_floor != Some(floor) && (self.current_floor - floor).abs() > 1e-9 {
            self.starts += 1.0;
            self.energy += 1.5;
        }
        self.target_floor = Some(floor);
        self.target_reason = Some(reason);
        self.pickup_direction = pickup_dir;
    }
}

#[derive(Default)]
struct FloorQueues {
    up: Vec<PaxRef>,
    down: Vec<PaxRef>,
}

struct HighriseBuilding {
    config: HighriseElevatorConfig,
    policy: HighrisePolicy,
    schedule: Vec<ScheduledArrival>,
    authority: DecisionAuthority,
    mdp_tuning: Option<MDPDispatchTuning>,
    floors: Vec<FloorQueues>,
    elevators: Vec<ElevatorCar>,
    completed: Vec<PaxRef>,
    people: Vec<PaxRef>,
    mdp_decision_log: Vec<MDPDecisionLogEntry>,
    next_arrival_index: usize,
}

impl HighriseBuilding {
    fn new(
        config: HighriseElevatorConfig,
        policy: HighrisePolicy,
        schedule: Vec<ScheduledArrival>,
        authority: DecisionAuthority,
        mdp_tuning: Option<MDPDispatchTuning>,
    ) -> Self {
        let floors: Vec<FloorQueues> = (0..config.n_floors).map(|_| FloorQueues::default()).collect();
        let elevators: Vec<ElevatorCar> = (0..config.n_elevators)
            .map(|i| {
                let start = ((config.n_floors - 1) as f64 * (i + 1) as f64 / (config.n_elevators + 1) as f64).round();
                let mut car = ElevatorCar::new(i, start, config.capacity);
                if policy == HighrisePolicy::ZonedService {
                    car.allowed_floors = Some(allowed_floors_for(i, config.n_floors));
                }
                car
            })
            .collect();
        HighriseBuilding {
            config,
            policy,
            schedule,
            authority,
            mdp_tuning,
            floors,
            elevators,
            completed: Vec::new(),
            people: Vec::new(),
            mdp_decision_log: Vec::new(),
            next_arrival_index: 0,
        }
    }

    fn run_time_step(&mut self, tick: i64) {
        let now = tick as f64 * self.config.step_size;
        self.emit_arrivals(now);
        self.advance_cars(now);
        self.make_dispatch_decisions(now);
    }

    fn pending_passenger_count(&self) -> usize {
        self.floors.iter().map(|f| f.up.len() + f.down.len()).sum()
    }
    fn in_car_count(&self) -> usize {
        self.elevators.iter().map(|e| e.passengers.len()).sum()
    }
    fn all_arrivals_emitted(&self) -> bool {
        self.next_arrival_index >= self.schedule.len()
    }
    fn is_drained(&self) -> bool {
        self.all_arrivals_emitted() && self.pending_passenger_count() == 0 && self.in_car_count() == 0
    }
    fn total_energy(&self) -> f64 {
        self.elevators.iter().map(|e| e.energy).sum()
    }
    fn total_distance(&self) -> f64 {
        self.elevators.iter().map(|e| e.distance_floors).sum()
    }
    fn total_stops(&self) -> f64 {
        self.elevators.iter().map(|e| e.stops).sum()
    }

    fn emit_arrivals(&mut self, now: f64) {
        while self.next_arrival_index < self.schedule.len() && self.schedule[self.next_arrival_index].t <= now {
            let a = self.schedule[self.next_arrival_index];
            self.next_arrival_index += 1;
            let id = self.people.len() as i64;
            let p = Rc::new(RefCell::new(HighrisePassenger::new(id, a.from_floor, a.to_floor, a.t)));
            self.people.push(p.clone());
            let dir = p.borrow().direction();
            if dir > 0 {
                self.floors[a.from_floor as usize].up.push(p);
            } else {
                self.floors[a.from_floor as usize].down.push(p);
            }
        }
    }

    fn advance_cars(&mut self, now: f64) {
        let step = self.config.step_size;
        let travel = self.config.floor_travel_time;
        for k in 0..self.elevators.len() {
            {
                let car = &mut self.elevators[k];
                if car.service_remaining > 0.0 {
                    car.service_remaining = (car.service_remaining - step).max(0.0);
                    if car.service_remaining > 0.0 {
                        continue;
                    }
                    car.state = CarState::Idle;
                    car.target_floor = None;
                    car.target_reason = None;
                    car.pickup_direction = 0;
                }
                if car.target_floor.is_none() {
                    continue;
                }
            }

            let delta = self.elevators[k].target_floor.unwrap() - self.elevators[k].current_floor;
            if delta.abs() < 1e-9 {
                service_floor(&mut self.elevators[k], &mut self.floors, &mut self.completed, &self.config, self.policy, now);
                continue;
            }

            let dir = sign(delta);
            {
                let car = &mut self.elevators[k];
                if car.direction != dir {
                    car.starts += 1.0;
                    car.energy += 1.5;
                }
                car.direction = dir;
                car.state = if car.target_reason == Some(TargetReason::Home) {
                    CarState::Prepositioning
                } else {
                    CarState::Moving
                };
                let step_floors = step / travel;
                let move_amt = delta.abs().min(step_floors);
                car.current_floor += dir as f64 * move_amt;
                car.distance_floors += move_amt;
                car.energy += move_amt * (1.0 + 0.055 * car.passengers.len() as f64);
            }

            let reached = (self.elevators[k].target_floor.unwrap() - self.elevators[k].current_floor).abs() < 1e-9;
            if reached {
                self.elevators[k].current_floor = self.elevators[k].target_floor.unwrap();
                service_floor(&mut self.elevators[k], &mut self.floors, &mut self.completed, &self.config, self.policy, now);
            }
        }
    }

    fn make_dispatch_decisions(&mut self, now: f64) {
        match self.authority {
            DecisionAuthority::Central => self.assign_central_cars(now, false, None),
            DecisionAuthority::Decentralized => {
                let mut claimed = HashSet::new();
                self.assign_autonomous_cars(now, &mut claimed, DecisionSource::Local);
            }
            DecisionAuthority::Hybrid => {
                self.assign_central_cars(now, true, Some(DecisionSource::Hybrid));
                let mut claimed = HashSet::new();
                self.assign_autonomous_cars(now, &mut claimed, DecisionSource::Hybrid);
            }
        }
    }

    fn assign_central_cars(&mut self, now: f64, urgent_only: bool, source: Option<DecisionSource>) {
        let n_floors = self.config.n_floors;
        let mut claimed: HashSet<String> = HashSet::new();
        for car in &self.elevators {
            if car.target_reason == Some(TargetReason::Pickup) {
                if let Some(tf) = car.target_floor {
                    claimed.insert(request_key(tf, car.pickup_direction));
                }
            }
        }

        for k in 0..self.elevators.len() {
            if self.elevators[k].service_remaining > 0.0 {
                continue;
            }
            if !self.elevators[k].passengers.is_empty() {
                let dest = choose_dropoff(&self.elevators[k], self.policy);
                if let Some(d) = dest {
                    self.elevators[k].set_target(d, TargetReason::Dropoff, 0, n_floors);
                }
                self.elevators[k].decision_source = source.unwrap_or(DecisionSource::Central);
                continue;
            }
            {
                let car = &self.elevators[k];
                if car.target_floor.is_some()
                    && car.target_reason != Some(TargetReason::Home)
                    && (car.current_floor - car.target_floor.unwrap()).abs() > 1e-9
                {
                    continue;
                }
            }

            let pickup = choose_pickup(
                &self.elevators[k],
                &self.floors,
                &self.config,
                self.policy,
                self.mdp_tuning.as_ref(),
                &mut self.mdp_decision_log,
                now,
                &claimed,
                None,
                urgent_only,
            );
            if let Some(pk) = pickup {
                claimed.insert(request_key(pk.floor as f64, pk.dir));
                self.elevators[k].set_target(pk.floor as f64, TargetReason::Pickup, pk.dir, n_floors);
                self.elevators[k].decision_source = source.unwrap_or(DecisionSource::Central);
                continue;
            }

            if urgent_only {
                continue;
            }
            let home = home_floor(&self.elevators[k], &self.config, self.policy);
            let cur = self.elevators[k].current_floor;
            if let Some(h) = home {
                if (h - cur).abs() > 0.1 {
                    self.elevators[k].set_target(h, TargetReason::Home, 0, n_floors);
                    self.elevators[k].decision_source = source.unwrap_or(DecisionSource::Central);
                    continue;
                }
            }
            let car = &mut self.elevators[k];
            car.target_floor = None;
            car.target_reason = None;
            car.pickup_direction = 0;
            car.state = CarState::Idle;
            car.direction = 0;
            car.decision_source = DecisionSource::None;
        }
    }

    fn assign_autonomous_cars(&mut self, now: f64, claimed: &mut HashSet<String>, source: DecisionSource) {
        let n_floors = self.config.n_floors;
        for k in 0..self.elevators.len() {
            if self.elevators[k].service_remaining > 0.0 {
                continue;
            }
            if !self.elevators[k].passengers.is_empty() {
                let dest = choose_dropoff(&self.elevators[k], self.policy);
                if let Some(d) = dest {
                    self.elevators[k].set_target(d, TargetReason::Dropoff, 0, n_floors);
                }
                self.elevators[k].decision_source = source;
                continue;
            }
            {
                let car = &self.elevators[k];
                if car.target_floor.is_some()
                    && car.target_reason != Some(TargetReason::Home)
                    && (car.current_floor - car.target_floor.unwrap()).abs() > 1e-9
                {
                    continue;
                }
            }

            let local = choose_pickup(
                &self.elevators[k],
                &self.floors,
                &self.config,
                self.policy,
                self.mdp_tuning.as_ref(),
                &mut self.mdp_decision_log,
                now,
                claimed,
                Some(self.config.local_sensor_radius),
                false,
            );
            let pickup = match local {
                Some(p) => Some(p),
                None => choose_pickup(
                    &self.elevators[k],
                    &self.floors,
                    &self.config,
                    self.policy,
                    self.mdp_tuning.as_ref(),
                    &mut self.mdp_decision_log,
                    now,
                    claimed,
                    None,
                    true,
                ),
            };
            if let Some(pk) = pickup {
                if source != DecisionSource::Local {
                    claimed.insert(request_key(pk.floor as f64, pk.dir));
                }
                self.elevators[k].set_target(pk.floor as f64, TargetReason::Pickup, pk.dir, n_floors);
                self.elevators[k].decision_source = source;
                continue;
            }

            let home = home_floor(&self.elevators[k], &self.config, self.policy);
            let cur = self.elevators[k].current_floor;
            if let Some(h) = home {
                if (h - cur).abs() > 0.1 {
                    self.elevators[k].set_target(h, TargetReason::Home, 0, n_floors);
                    self.elevators[k].decision_source = source;
                    continue;
                }
            }
            if self.elevators[k].target_reason == Some(TargetReason::Home) {
                self.elevators[k].decision_source = source;
            } else {
                let car = &mut self.elevators[k];
                car.target_floor = None;
                car.target_reason = None;
                car.pickup_direction = 0;
                car.state = CarState::Idle;
                car.direction = 0;
                car.decision_source = DecisionSource::None;
            }
        }
    }
}

struct PickupResult {
    floor: i32,
    dir: i32,
}

fn service_floor(
    car: &mut ElevatorCar,
    floors: &mut [FloorQueues],
    completed: &mut Vec<PaxRef>,
    config: &HighriseElevatorConfig,
    policy: HighrisePolicy,
    now: f64,
) {
    let floor = car.current_floor.round() as i32;
    let mut changed = false;

    let mut remaining = Vec::new();
    let mut deboard = Vec::new();
    for p in car.passengers.drain(..) {
        if p.borrow().to_floor == floor {
            deboard.push(p);
        } else {
            remaining.push(p);
        }
    }
    car.passengers = remaining;
    for p in deboard {
        p.borrow_mut().exit_time = now;
        completed.push(p);
        changed = true;
    }

    if car.target_reason != Some(TargetReason::Home) {
        let boarded = board_passengers(car, floors, config, policy, floor, now);
        changed = changed || boarded > 0;
    }

    if changed || car.target_reason == Some(TargetReason::Pickup) || car.target_reason == Some(TargetReason::Dropoff) {
        car.stops += 1.0;
        car.energy += 0.8;
        car.service_remaining = config.service_time;
        car.state = CarState::Serving;
        car.direction = 0;
        return;
    }

    car.target_floor = None;
    car.target_reason = None;
    car.pickup_direction = 0;
    car.state = CarState::Idle;
    car.direction = 0;
}

fn board_passengers(
    car: &mut ElevatorCar,
    floors: &mut [FloorQueues],
    _config: &HighriseElevatorConfig,
    policy: HighrisePolicy,
    floor: i32,
    now: f64,
) -> usize {
    if car.is_full() {
        return 0;
    }
    let fidx = floor as usize;
    let dir = boarding_direction(car, floors, fidx, policy);
    if dir == 0 {
        return 0;
    }
    let queue: &[PaxRef] = if dir > 0 { &floors[fidx].up } else { &floors[fidx].down };
    if queue.is_empty() {
        return 0;
    }

    let direct_dest = if policy == HighrisePolicy::FewestStops {
        dominant_destination(queue, car)
    } else {
        None
    };
    let mut chosen: Vec<PaxRef> = Vec::new();
    let mut keep: Vec<PaxRef> = Vec::new();
    for p in queue.iter() {
        if car.spare_capacity() - chosen.len() as i32 <= 0 {
            keep.push(p.clone());
            continue;
        }
        if !can_car_serve_passenger(car, &p.borrow()) {
            keep.push(p.clone());
            continue;
        }
        if policy == HighrisePolicy::FewestStops {
            if let Some(dd) = direct_dest {
                if p.borrow().to_floor != dd {
                    keep.push(p.clone());
                    continue;
                }
            }
        }
        if policy == HighrisePolicy::EnergyEfficient && !car.passengers.is_empty() && would_add_reverse_stop(car, &p.borrow())
        {
            keep.push(p.clone());
            continue;
        }
        chosen.push(p.clone());
    }
    if dir > 0 {
        floors[fidx].up = keep;
    } else {
        floors[fidx].down = keep;
    }

    let n = chosen.len();
    for p in &chosen {
        p.borrow_mut().board_time = now;
        car.passengers.push(p.clone());
    }
    car.energy += n as f64 * 0.08;
    n
}

fn boarding_direction(car: &ElevatorCar, floors: &[FloorQueues], floor: usize, policy: HighrisePolicy) -> i32 {
    if policy == HighrisePolicy::FewestStops && !car.passengers.is_empty() {
        return 0;
    }
    if !car.passengers.is_empty() {
        return match choose_dropoff(car, policy) {
            None => 0,
            Some(next) => sign(next - floor as f64),
        };
    }
    if car.pickup_direction != 0 {
        return car.pickup_direction;
    }
    let up = floors[floor].up.len();
    let down = floors[floor].down.len();
    if up == 0 && down == 0 {
        return 0;
    }
    if up == 0 {
        return -1;
    }
    if down == 0 {
        return 1;
    }
    let oldest_up = floors[floor].up.first().map(|p| p.borrow().arrival_time).unwrap_or(f64::INFINITY);
    let oldest_down = floors[floor].down.first().map(|p| p.borrow().arrival_time).unwrap_or(f64::INFINITY);
    if oldest_up <= oldest_down {
        1
    } else {
        -1
    }
}

fn choose_pickup(
    car: &ElevatorCar,
    floors: &[FloorQueues],
    config: &HighriseElevatorConfig,
    policy: HighrisePolicy,
    mdp_tuning: Option<&MDPDispatchTuning>,
    log: &mut Vec<MDPDecisionLogEntry>,
    now: f64,
    claimed: &HashSet<String>,
    local_radius: Option<f64>,
    urgent_only: bool,
) -> Option<PickupResult> {
    let mut best: Option<(i32, i32, f64, PickupFeatures)> = None;
    for floor in 0..floors.len() {
        for dir in [1i32, -1] {
            let queue: &[PaxRef] = if dir > 0 { &floors[floor].up } else { &floors[floor].down };
            let eligible: Vec<&PaxRef> = queue.iter().filter(|p| can_car_serve_passenger(car, &p.borrow())).collect();
            if eligible.is_empty() {
                continue;
            }
            let key = request_key(floor as f64, dir);
            if claimed.contains(&key) {
                continue;
            }
            let distance = (car.current_floor - floor as f64).abs();
            if let Some(r) = local_radius {
                if distance > r {
                    continue;
                }
            }
            let oldest_wait = (now - eligible[0].borrow().arrival_time).max(0.0);
            if urgent_only && oldest_wait < config.urgent_wait_threshold {
                continue;
            }
            let queue_len = eligible.len() as f64;
            let trip = average_trip_floors_refs(&eligible);
            let same_side = if sign(floor as f64 - car.current_floor) == dir { 1.0 } else { 0.0 };
            let max_group = largest_destination_group_refs(&eligible) as f64;
            let features = PickupFeatures { distance, oldest_wait, queue_len, trip, same_side, max_group };
            let score = score_pickup(&features, policy, mdp_tuning);
            let take = match &best {
                None => true,
                Some((_, _, bscore, _)) => score < *bscore,
            };
            if take {
                best = Some((floor as i32, dir, score, features));
            }
        }
    }
    if let Some((floor, dir, _, features)) = best {
        record_mdp_decision(&features, policy, mdp_tuning, log);
        return Some(PickupResult { floor, dir });
    }
    None
}

fn score_pickup(features: &PickupFeatures, policy: HighrisePolicy, mdp_tuning: Option<&MDPDispatchTuning>) -> f64 {
    let w = weights_for(features, policy, mdp_tuning);
    features.distance * w.distance + features.trip * w.trip
        - features.queue_len * w.queue
        - features.oldest_wait * w.wait
        - features.same_side * w.same_direction
        - features.max_group * w.destination_group
}

fn weights_for(features: &PickupFeatures, policy: HighrisePolicy, mdp_tuning: Option<&MDPDispatchTuning>) -> DispatchScoreWeights {
    if let Some(decision) = mdp_decision_for(features, policy, mdp_tuning) {
        let tuning = mdp_tuning.unwrap();
        return tuning
            .actions
            .get(decision.1)
            .map(|a| a.weights.clone())
            .unwrap_or_else(|| tuning.learned_weights.clone());
    }
    policy_score_weights(policy)
}

fn mdp_decision_for(
    features: &PickupFeatures,
    policy: HighrisePolicy,
    mdp_tuning: Option<&MDPDispatchTuning>,
) -> Option<(usize, usize, String)> {
    if !is_mdp_policy(policy) {
        return None;
    }
    let tuning = mdp_tuning?;
    let state_id = encode_mdp_dispatch_state(features, tuning.observability);
    let action_idx = tuning.policy[state_id].max(0) as usize;
    let action = tuning.action_labels.get(action_idx).cloned().unwrap_or_else(|| format!("a{action_idx}"));
    Some((state_id, action_idx, action))
}

fn record_mdp_decision(
    features: &PickupFeatures,
    policy: HighrisePolicy,
    mdp_tuning: Option<&MDPDispatchTuning>,
    log: &mut Vec<MDPDecisionLogEntry>,
) {
    let decision = match mdp_decision_for(features, policy, mdp_tuning) {
        Some(d) => d,
        None => return,
    };
    let tuning = match mdp_tuning {
        Some(t) => t,
        None => return,
    };
    let (state_id, _action_idx, action) = decision;
    log.push(MDPDecisionLogEntry {
        state_id,
        state: tuning.state_labels.get(state_id).cloned().unwrap_or_else(|| format!("s{state_id}")),
        action,
        bins: mdp_bin_labels(&decode_mdp_dispatch_state(state_id, tuning.observability)),
    });
}

fn home_floor(car: &ElevatorCar, config: &HighriseElevatorConfig, policy: HighrisePolicy) -> Option<f64> {
    if policy == HighrisePolicy::CenterPreposition || is_mdp_policy(policy) {
        let center = (config.n_floors - 1) as f64 / 2.0;
        let spacing = 3.0;
        return Some(clamp(
            (center + (car.idx as f64 - (config.n_elevators - 1) as f64 / 2.0) * spacing).round(),
            0.0,
            (config.n_floors - 1) as f64,
        ));
    }
    if policy == HighrisePolicy::ZonedService {
        if let Some(allowed) = &car.allowed_floors {
            let mut floors: Vec<i32> = allowed.iter().copied().collect();
            floors.sort_unstable();
            return Some(*floors.get(floors.len() / 2).unwrap_or(&0) as f64);
        }
    }
    None
}

fn can_car_serve_passenger(car: &ElevatorCar, p: &HighrisePassenger) -> bool {
    match &car.allowed_floors {
        None => true,
        Some(allowed) => allowed.contains(&p.from_floor) && allowed.contains(&p.to_floor),
    }
}

// ---------------------------------------------------------------------------
// Schedule + run orchestration.
// ---------------------------------------------------------------------------

fn build_highrise_schedule(cfg: &HighriseElevatorConfig) -> Vec<ScheduledArrival> {
    let seed = cfg.seed;
    with_seed(seed, |_g| {
        let mut rng = mulberry32(seed);
        let mut out = Vec::new();
        let mut t = 0.0;
        loop {
            t += -((1e-9_f64).max(1.0 - rng.next_float())).ln() / cfg.arrival_rate;
            if t > cfg.sim_t {
                break;
            }
            let r = rng.next_float();
            let (from_floor, to_floor);
            if r < 0.55 {
                from_floor = 0;
                to_floor = 1 + (rng.next_float() * (cfg.n_floors - 1) as f64).floor() as i32;
            } else if r < 0.80 {
                from_floor = 1 + (rng.next_float() * (cfg.n_floors - 1) as f64).floor() as i32;
                to_floor = 0;
            } else {
                from_floor = 1 + (rng.next_float() * (cfg.n_floors - 1) as f64).floor() as i32;
                let mut tf;
                loop {
                    tf = 1 + (rng.next_float() * (cfg.n_floors - 1) as f64).floor() as i32;
                    if tf != from_floor {
                        break;
                    }
                }
                to_floor = tf;
            }
            out.push(ScheduledArrival { t, from_floor, to_floor });
        }
        out
    })
}

struct HighriseRunOptions {
    authority: DecisionAuthority,
    mdp_tuning: Option<MDPDispatchTuning>,
}

fn run_highrise_elevators(
    cfg: &HighriseElevatorConfig,
    policy: HighrisePolicy,
    schedule: &[ScheduledArrival],
    opts: HighriseRunOptions,
) -> HighriseElevatorResult {
    let mut building = HighriseBuilding::new(cfg.clone(), policy, schedule.to_vec(), opts.authority, opts.mdp_tuning.clone());
    // PORT NOTE: frame/series recording omitted (animation not ported).
    let max_ticks = ((cfg.sim_t + cfg.drain_t) / cfg.step_size).round() as i64;
    for tick in 0..=max_ticks {
        building.run_time_step(tick);
        if tick as f64 * cfg.step_size >= cfg.sim_t && building.is_drained() {
            break;
        }
    }
    make_result(policy, opts.authority, cfg, schedule, &building, opts.mdp_tuning.as_ref())
}

fn make_result(
    policy: HighrisePolicy,
    authority: DecisionAuthority,
    config: &HighriseElevatorConfig,
    schedule: &[ScheduledArrival],
    building: &HighriseBuilding,
    mdp_tuning: Option<&MDPDispatchTuning>,
) -> HighriseElevatorResult {
    let served: Vec<PaxRef> = building.people.iter().filter(|p| p.borrow().exit_time >= 0.0).cloned().collect();
    let waits: Vec<f64> = served.iter().map(|p| p.borrow().board_time - p.borrow().arrival_time).collect();
    let travels: Vec<f64> = served.iter().map(|p| p.borrow().exit_time - p.borrow().board_time).collect();
    let totals: Vec<f64> = served.iter().map(|p| p.borrow().exit_time - p.borrow().arrival_time).collect();
    let people: Vec<HighrisePassengerSnapshot> = building
        .people
        .iter()
        .map(|p| {
            let b = p.borrow();
            HighrisePassengerSnapshot {
                id: b.id,
                from_floor: b.from_floor,
                to_floor: b.to_floor,
                arrival_time: b.arrival_time,
                board_time: b.board_time,
                exit_time: b.exit_time,
            }
        })
        .collect();
    HighriseElevatorResult {
        policy,
        authority,
        config: config.clone(),
        schedule: schedule.to_vec(),
        aggregates: HighriseAggregates {
            n: building.people.len(),
            n_served: served.len(),
            mean_wait: mean(&waits),
            mean_travel: mean(&travels),
            mean_total: mean(&totals),
            p95_wait: percentile(&waits, 0.95),
            p95_total: percentile(&totals, 0.95),
            total_stops: building.total_stops(),
            total_distance_floors: building.total_distance(),
            total_energy: building.total_energy(),
            timed_out: building.people.len() - served.len(),
        },
        people,
        mdp_tuning: mdp_tuning.map(summarize_mdp_tuning),
        mdp_run: if building.mdp_decision_log.is_empty() {
            None
        } else {
            Some(summarize_mdp_run(&building.mdp_decision_log))
        },
        marginal_vs_lowest_time: None,
    }
}

// ---------------------------------------------------------------------------
// Policy weight tables.
// ---------------------------------------------------------------------------

fn dsw(distance: f64, trip: f64, queue: f64, wait: f64, same_direction: f64, destination_group: f64) -> DispatchScoreWeights {
    DispatchScoreWeights { distance, trip, queue, wait, same_direction, destination_group }
}

fn policy_score_weights(policy: HighrisePolicy) -> DispatchScoreWeights {
    match policy {
        HighrisePolicy::FewestStops => dsw(2.0, 0.15, 0.15, 0.025, 0.1, 3.0),
        HighrisePolicy::LowestTotalTime => dsw(1.25, 0.2, 1.1, 0.08, 0.25, 0.2),
        HighrisePolicy::EnergyEfficient => dsw(2.2, 0.35, 0.45, 0.035, 0.8, 0.6),
        HighrisePolicy::CenterPreposition => dsw(1.15, 0.18, 0.9, 0.065, 0.3, 0.2),
        HighrisePolicy::ZonedService => dsw(1.35, 0.15, 0.8, 0.05, 0.4, 0.25),
        HighrisePolicy::MdpCallOnly => dsw(1.25, 0.2, 1.1, 0.08, 0.25, 0.2),
        HighrisePolicy::MdpTuned => dsw(1.25, 0.2, 1.1, 0.08, 0.25, 0.2),
    }
}

// ---------------------------------------------------------------------------
// MDP dispatch tuning.
// ---------------------------------------------------------------------------

fn mdp_distance_bins() -> [f64; 4] {
    [2.0, 8.0, 18.0, f64::INFINITY]
}
fn mdp_queue_bins() -> [f64; 3] {
    [1.0, 4.0, f64::INFINITY]
}
fn mdp_wait_bins() -> [f64; 3] {
    [15.0, 45.0, f64::INFINITY]
}
fn mdp_trip_bins() -> [f64; 3] {
    [8.0, 22.0, f64::INFINITY]
}
fn mdp_batch_bins() -> [f64; 3] {
    [1.0, 3.0, f64::INFINITY]
}
const CALL_ONLY_EXPECTED_TRIP: f64 = 18.0;

fn mdp_action_profiles() -> Vec<MDPActionProfile> {
    vec![
        MDPActionProfile { label: "direct-batch", weights: dsw(1.8, 0.12, 0.3, 0.035, 0.2, 2.8) },
        MDPActionProfile { label: "latency", weights: dsw(1.05, 0.18, 1.35, 0.105, 0.25, 0.3) },
        MDPActionProfile { label: "energy", weights: dsw(2.45, 0.42, 0.55, 0.04, 1.1, 0.65) },
        MDPActionProfile { label: "balanced", weights: dsw(1.35, 0.2, 0.95, 0.07, 0.45, 0.45) },
        MDPActionProfile { label: "oldest-first", weights: dsw(0.95, 0.16, 0.75, 0.16, 0.2, 0.15) },
    ]
}

#[derive(Clone, Default)]
struct MDPDispatchStateBins {
    distance_bin: usize,
    wait_bin: usize,
    same_side: usize,
    queue_bin: Option<usize>,
    trip_bin: Option<usize>,
    batch_bin: Option<usize>,
}

fn is_mdp_policy(policy: HighrisePolicy) -> bool {
    policy == HighrisePolicy::MdpCallOnly || policy == HighrisePolicy::MdpTuned
}

fn observability_for_policy(policy: HighrisePolicy) -> MDPObservability {
    if policy == HighrisePolicy::MdpCallOnly {
        MDPObservability::CallOnly
    } else {
        MDPObservability::DestinationDispatch
    }
}

fn mdp_num_states(observability: MDPObservability) -> usize {
    let mut n = mdp_distance_bins().len() * mdp_wait_bins().len() * 2;
    if observability == MDPObservability::DestinationDispatch {
        n *= mdp_queue_bins().len() * mdp_trip_bins().len() * mdp_batch_bins().len();
    }
    n
}

fn env_f64_opt(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn optimize_highrise_dispatch_mdp(observability: MDPObservability) -> MDPDispatchTuning {
    let num_states = mdp_num_states(observability);
    let state_labels: Vec<String> =
        (0..num_states).map(|s| label_mdp_dispatch_state(&decode_mdp_dispatch_state(s, observability))).collect();
    let profiles = mdp_action_profiles();
    let action_labels: Vec<String> = profiles.iter().map(|a| a.label.to_string()).collect();

    let gamma = env_f64_opt("MDP_GAMMA", 0.92);
    let n_actions = profiles.len();
    let spec = MDPSpec {
        num_states,
        num_actions: Box::new(move |_s| n_actions),
        outcomes: Box::new(move |s, a| abstract_dispatch_outcomes(s, a, observability)),
        is_terminal: None,
        terminal_reward: None,
        state_label: {
            let labels = state_labels.clone();
            Some(Box::new(move |s| labels.get(s).cloned().unwrap_or_default()))
        },
        action_label: {
            let labels = action_labels.clone();
            Some(Box::new(move |a| labels.get(a).cloned().unwrap_or_default()))
        },
    };
    let opts = VIOptions {
        gamma,
        tol: env_f64_opt("MDP_TOL", 1e-8),
        max_iter: env_f64_opt("MDP_MAX_ITER", 10000.0) as usize,
        random_tie_break: false,
        ..Default::default()
    };
    let vi = value_iteration(spec, opts);
    let learned_weights = average_mdp_weights(&vi.policy, observability);
    MDPDispatchTuning {
        observability,
        num_states,
        actions: profiles,
        policy: vi.policy,
        gamma: vi.gamma,
        iterations: vi.iterations,
        final_delta: vi.final_delta,
        learned_weights,
        state_labels,
        action_labels,
    }
}

fn abstract_dispatch_outcomes(s: usize, action_idx: usize, observability: MDPObservability) -> Vec<Outcome> {
    let st = decode_mdp_dispatch_state(s, observability);
    let profiles = mdp_action_profiles();
    let action = &profiles[action_idx];
    let d = [1.0, 5.0, 13.0, 28.0][st.distance_bin];
    let q = if observability == MDPObservability::DestinationDispatch {
        [1.0, 3.0, 7.0][st.queue_bin.unwrap_or(0)]
    } else {
        1.0
    };
    let wait = [8.0, 30.0, 75.0][st.wait_bin];
    let trip = if observability == MDPObservability::DestinationDispatch {
        [4.0, 15.0, 31.0][st.trip_bin.unwrap_or(0)]
    } else {
        CALL_ONLY_EXPECTED_TRIP
    };
    let batch = if observability == MDPObservability::DestinationDispatch {
        [1.0, 2.0, 5.0][st.batch_bin.unwrap_or(0)]
    } else {
        1.0
    };
    let same = st.same_side as f64;

    let w = &action.weights;
    let direct_demand = if observability == MDPObservability::DestinationDispatch
        && (st.batch_bin.unwrap_or(0) >= 1 || (st.queue_bin.unwrap_or(0) >= 1 && st.trip_bin.unwrap_or(0) >= 1))
    {
        1.0
    } else {
        0.0
    };
    let urgency = st.wait_bin as f64 / 2.0;
    let energy_risk = st.distance_bin as f64 / 3.0 + (if same > 0.0 { -0.25 } else { 0.25 });
    let time_cost = d * 1.35 + trip * 0.65 + wait * 0.85 - q * 3.5 - batch * 1.5;
    let stop_cost = (2.2 - w.destination_group * direct_demand - st.batch_bin.unwrap_or(0) as f64 * 0.35).max(0.5);
    let energy_cost = d * (1.0 + 0.24 * w.distance) + trip * 0.07 - same * w.same_direction;
    let hidden_queue_penalty = if observability == MDPObservability::CallOnly {
        (urgency - w.wait * 7.0).max(0.0) + (0.8 - w.distance * 0.18 - w.same_direction * 0.25).max(0.0)
    } else {
        0.0
    };
    let mismatch = (urgency - w.wait * 8.0).max(0.0)
        + (energy_risk - w.same_direction * 0.45).max(0.0)
        + (direct_demand + st.batch_bin.unwrap_or(0) as f64 * 0.35 - w.destination_group * 0.28).max(0.0)
        + hidden_queue_penalty;
    let reward = -(time_cost + energy_cost * 1.8 + stop_cost * 6.0 + mismatch * 12.0);

    let fit = clamp(
        0.52
            + (if w.wait > 0.09 && st.wait_bin >= 1 { 0.12 } else { 0.0 })
            + (if w.destination_group > 1.5 && direct_demand > 0.0 { 0.12 } else { 0.0 })
            + (if w.destination_group > 1.5 && st.batch_bin.unwrap_or(0) >= 1 { 0.08 } else { 0.0 })
            + (if w.same_direction > 0.7 && same > 0.0 { 0.10 } else { 0.0 })
            + (if w.distance > 2.0 && st.distance_bin <= 1 { 0.08 } else { 0.0 })
            - mismatch * 0.05,
        0.2,
        0.88,
    );

    let mut improved = MDPDispatchStateBins {
        distance_bin: st.distance_bin.saturating_sub(if w.distance > 1.7 { 1 } else { 0 }),
        wait_bin: st.wait_bin.saturating_sub(if w.wait > 0.08 { 1 } else { 0 }),
        same_side: same as usize,
        ..Default::default()
    };
    let mut degraded = MDPDispatchStateBins {
        distance_bin: (st.distance_bin + if w.distance < 1.2 { 1 } else { 0 }).min(mdp_distance_bins().len() - 1),
        wait_bin: (st.wait_bin + if w.wait < 0.07 { 1 } else { 0 }).min(mdp_wait_bins().len() - 1),
        same_side: if same > 0.0 { 1 } else { 0 },
        ..Default::default()
    };
    if observability == MDPObservability::DestinationDispatch {
        improved.queue_bin = Some(st.queue_bin.unwrap_or(0).saturating_sub(if w.queue > 0.8 || w.destination_group > 1.5 { 1 } else { 0 }));
        improved.trip_bin = Some(st.trip_bin.unwrap_or(0).saturating_sub(if w.destination_group > 1.5 { 1 } else { 0 }));
        improved.batch_bin = Some(st.batch_bin.unwrap_or(0).saturating_sub(if w.destination_group > 1.5 { 1 } else { 0 }));
        degraded.queue_bin = Some((st.queue_bin.unwrap_or(0) + if w.queue < 0.75 { 1 } else { 0 }).min(mdp_queue_bins().len() - 1));
        degraded.trip_bin = Some(st.trip_bin.unwrap_or(0));
        degraded.batch_bin = Some(st.batch_bin.unwrap_or(0));
    }
    vec![
        Outcome { prob: fit, reward, next_state: encode_mdp_dispatch_bins(&improved, observability) },
        Outcome { prob: 1.0 - fit, reward: reward - 5.0 - wait * 0.05, next_state: encode_mdp_dispatch_bins(&degraded, observability) },
    ]
}

fn encode_mdp_dispatch_state(features: &PickupFeatures, observability: MDPObservability) -> usize {
    if observability == MDPObservability::CallOnly {
        return encode_mdp_dispatch_bins(
            &MDPDispatchStateBins {
                distance_bin: bin_index(features.distance, &mdp_distance_bins()),
                wait_bin: bin_index(features.oldest_wait, &mdp_wait_bins()),
                same_side: if features.same_side > 0.0 { 1 } else { 0 },
                ..Default::default()
            },
            observability,
        );
    }
    encode_mdp_dispatch_bins(
        &MDPDispatchStateBins {
            distance_bin: bin_index(features.distance, &mdp_distance_bins()),
            queue_bin: Some(bin_index(features.queue_len, &mdp_queue_bins())),
            wait_bin: bin_index(features.oldest_wait, &mdp_wait_bins()),
            trip_bin: Some(bin_index(features.trip, &mdp_trip_bins())),
            batch_bin: Some(bin_index(features.max_group, &mdp_batch_bins())),
            same_side: if features.same_side > 0.0 { 1 } else { 0 },
        },
        observability,
    )
}

fn encode_mdp_dispatch_bins(st: &MDPDispatchStateBins, observability: MDPObservability) -> usize {
    let mut idx = st.distance_bin;
    if observability == MDPObservability::DestinationDispatch {
        idx = idx * mdp_queue_bins().len() + st.queue_bin.unwrap_or(0);
    }
    idx = idx * mdp_wait_bins().len() + st.wait_bin;
    if observability == MDPObservability::DestinationDispatch {
        idx = idx * mdp_trip_bins().len() + st.trip_bin.unwrap_or(0);
        idx = idx * mdp_batch_bins().len() + st.batch_bin.unwrap_or(0);
    }
    idx * 2 + st.same_side
}

fn decode_mdp_dispatch_state(s: usize, observability: MDPObservability) -> MDPDispatchStateBins {
    let mut s = s;
    let same_side = s % 2;
    s /= 2;
    let mut batch_bin = None;
    let mut trip_bin = None;
    let mut queue_bin = None;
    if observability == MDPObservability::DestinationDispatch {
        batch_bin = Some(s % mdp_batch_bins().len());
        s /= mdp_batch_bins().len();
        trip_bin = Some(s % mdp_trip_bins().len());
        s /= mdp_trip_bins().len();
    }
    let wait_bin = s % mdp_wait_bins().len();
    s /= mdp_wait_bins().len();
    if observability == MDPObservability::DestinationDispatch {
        queue_bin = Some(s % mdp_queue_bins().len());
        s /= mdp_queue_bins().len();
    }
    let distance_bin = s;
    MDPDispatchStateBins { distance_bin, queue_bin, wait_bin, trip_bin, batch_bin, same_side }
}

fn label_mdp_dispatch_state(st: &MDPDispatchStateBins) -> String {
    let mut parts = vec![format!("d{}", st.distance_bin)];
    if let Some(q) = st.queue_bin {
        parts.push(format!("q{q}"));
    }
    parts.push(format!("w{}", st.wait_bin));
    if let Some(t) = st.trip_bin {
        parts.push(format!("trip{t}"));
    }
    if let Some(b) = st.batch_bin {
        parts.push(format!("batch{b}"));
    }
    parts.push(if st.same_side != 0 { "same".to_string() } else { "reverse".to_string() });
    parts.join("/")
}

fn mdp_bin_labels(st: &MDPDispatchStateBins) -> Vec<(String, String)> {
    let mut out = vec![
        ("distance".to_string(), format!("d{}", st.distance_bin)),
        ("wait".to_string(), format!("w{}", st.wait_bin)),
        ("direction".to_string(), if st.same_side != 0 { "same".to_string() } else { "reverse".to_string() }),
    ];
    if let Some(q) = st.queue_bin {
        out.push(("queue".to_string(), format!("q{q}")));
    }
    if let Some(t) = st.trip_bin {
        out.push(("trip".to_string(), format!("trip{t}")));
    }
    if let Some(b) = st.batch_bin {
        out.push(("batch".to_string(), format!("batch{b}")));
    }
    out
}

fn bin_index(x: f64, thresholds: &[f64]) -> usize {
    for (i, &t) in thresholds.iter().enumerate() {
        if x <= t {
            return i;
        }
    }
    thresholds.len() - 1
}

fn average_mdp_weights(policy: &[i32], observability: MDPObservability) -> DispatchScoreWeights {
    let profiles = mdp_action_profiles();
    let mut out = dsw(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut total = 0.0;
    for (s, &a) in policy.iter().enumerate() {
        let st = decode_mdp_dispatch_state(s, observability);
        let importance = 1.0
            + st.queue_bin.unwrap_or(0) as f64
            + st.wait_bin as f64
            + st.batch_bin.unwrap_or(0) as f64 * 0.7
            + st.trip_bin.unwrap_or(0) as f64 * 0.35;
        let weights = &profiles[a.max(0) as usize].weights;
        out.distance += weights.distance * importance;
        out.trip += weights.trip * importance;
        out.queue += weights.queue * importance;
        out.wait += weights.wait * importance;
        out.same_direction += weights.same_direction * importance;
        out.destination_group += weights.destination_group * importance;
        total += importance;
    }
    if total != 0.0 {
        out.distance /= total;
        out.trip /= total;
        out.queue /= total;
        out.wait /= total;
        out.same_direction /= total;
        out.destination_group /= total;
    }
    out
}

fn summarize_mdp_tuning(tuning: &MDPDispatchTuning) -> MDPDispatchTuningSummary {
    let candidates = [0i64, 1, 5, 17, 43, 87, 129, 173, tuning.num_states as i64 - 1];
    let mut interesting: Vec<usize> = Vec::new();
    for &s in &candidates {
        if s >= 0 && (s as usize) < tuning.num_states && !interesting.contains(&(s as usize)) {
            interesting.push(s as usize);
        }
    }
    MDPDispatchTuningSummary {
        observability: tuning.observability,
        num_states: tuning.num_states,
        gamma: tuning.gamma,
        iterations: tuning.iterations,
        final_delta: tuning.final_delta,
        learned_weights: tuning.learned_weights.clone(),
        state_policy: interesting
            .into_iter()
            .map(|s| StatePolicyRow {
                state: tuning.state_labels[s].clone(),
                action: tuning.action_labels[tuning.policy[s].max(0) as usize].clone(),
            })
            .collect(),
    }
}

fn summarize_mdp_run(log: &[MDPDecisionLogEntry]) -> MDPRunDiagnostics {
    let total = log.len();
    let mut action_counts: Vec<(String, usize)> = Vec::new();
    let mut state_counts: Vec<(String, TopState)> = Vec::new();
    // marginal: variable -> bin -> action -> count, preserving insertion order.
    let mut marginal: Vec<(String, Vec<(String, Vec<(String, usize)>)>)> = Vec::new();

    for row in log {
        bump(&mut action_counts, &row.action);
        let state_key = format!("{}|{}", row.state, row.action);
        match state_counts.iter_mut().find(|(k, _)| *k == state_key) {
            Some((_, ts)) => ts.count += 1,
            None => state_counts.push((state_key, TopState { state: row.state.clone(), action: row.action.clone(), count: 1 })),
        }
        for (variable, bin) in &row.bins {
            let by_bin = match marginal.iter_mut().find(|(v, _)| v == variable) {
                Some((_, b)) => b,
                None => {
                    marginal.push((variable.clone(), Vec::new()));
                    &mut marginal.last_mut().unwrap().1
                }
            };
            let by_action = match by_bin.iter_mut().find(|(b, _)| b == bin) {
                Some((_, a)) => a,
                None => {
                    by_bin.push((bin.clone(), Vec::new()));
                    &mut by_bin.last_mut().unwrap().1
                }
            };
            bump(by_action, &row.action);
        }
    }

    let mut action_count_rows: Vec<ActionCount> = action_counts
        .into_iter()
        .map(|(action, count)| ActionCount { action, count, share: if total > 0 { count as f64 / total as f64 } else { 0.0 } })
        .collect();
    action_count_rows.sort_by(|a, b| b.count.cmp(&a.count));

    let mut top_states: Vec<TopState> = state_counts.into_iter().map(|(_, ts)| ts).collect();
    top_states.sort_by(|a, b| b.count.cmp(&a.count));
    top_states.truncate(8);

    let mut marginals: Vec<MarginalRow> = marginal
        .into_iter()
        .map(|(variable, by_bin)| {
            let mut bins: Vec<MarginalBin> = by_bin
                .into_iter()
                .map(|(bin, by_action)| {
                    let count: usize = by_action.iter().map(|(_, n)| *n).sum();
                    let mut dominant_action = String::new();
                    let mut dominant_count: i64 = -1;
                    for (action, n) in &by_action {
                        if *n as i64 > dominant_count {
                            dominant_action = action.clone();
                            dominant_count = *n as i64;
                        }
                    }
                    MarginalBin {
                        bin,
                        count,
                        dominant_action,
                        share: if count > 0 { dominant_count as f64 / count as f64 } else { 0.0 },
                    }
                })
                .collect();
            bins.sort_by(|a, b| a.bin.cmp(&b.bin));
            MarginalRow { variable, bins }
        })
        .collect();
    marginals.sort_by(|a, b| a.variable.cmp(&b.variable));

    MDPRunDiagnostics { total_decisions: total, action_counts: action_count_rows, top_states, marginals }
}

fn bump(counts: &mut Vec<(String, usize)>, key: &str) {
    match counts.iter_mut().find(|(k, _)| k == key) {
        Some((_, c)) => *c += 1,
        None => counts.push((key.to_string(), 1)),
    }
}

fn compare_to_baseline(result: &HighriseElevatorResult, baseline: &HighriseElevatorResult) -> MarginalComparison {
    MarginalComparison {
        baseline_policy: baseline.policy,
        baseline_authority: baseline.authority,
        mean_wait_delta: result.aggregates.mean_wait - baseline.aggregates.mean_wait,
        mean_total_delta: result.aggregates.mean_total - baseline.aggregates.mean_total,
        stops_delta: result.aggregates.total_stops - baseline.aggregates.total_stops,
        energy_delta: result.aggregates.total_energy - baseline.aggregates.total_energy,
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers (console + variant summaries).
// ---------------------------------------------------------------------------

fn variant_summary(result: &HighriseElevatorResult) -> String {
    let a = &result.aggregates;
    let mut out = format!(
        "{} {} Mean total {:.1}s, stops {}, energy {:.1}.",
        result.policy.summary(),
        result.authority.summary(),
        a.mean_total,
        a.total_stops,
        a.total_energy
    );
    if let (Some(tuning), Some(run)) = (&result.mdp_tuning, &result.mdp_run) {
        let w = &tuning.learned_weights;
        let marginal_name = if tuning.observability == MDPObservability::DestinationDispatch { "batch" } else { "wait" };
        out += &format!(
            " MDP is pre-solved by value iteration ({} states, {} sweeps, observability={}), then this run exercised {} learned pickup decisions. Observed actions: {}. {} marginal: {}. Learned weights favor destination grouping={:.2}, distance={:.2}, wait={:.2}.",
            tuning.num_states,
            tuning.iterations,
            tuning.observability.slug(),
            run.total_decisions,
            format_action_shares(run, 3),
            marginal_name,
            format_marginal(run, marginal_name),
            w.destination_group,
            w.distance,
            w.wait
        );
    }
    if let Some(m) = &result.marginal_vs_lowest_time {
        out += &format!(
            " Marginal vs {} / {}: mean total {}, wait {}, stops {}, energy {}.",
            m.baseline_policy.label(),
            m.baseline_authority.label(),
            format_signed(m.mean_total_delta, "s"),
            format_signed(m.mean_wait_delta, "s"),
            format_signed(m.stops_delta, ""),
            format_signed(m.energy_delta, "")
        );
    }
    out
}

fn format_action_shares(run: &MDPRunDiagnostics, max_items: usize) -> String {
    if run.action_counts.is_empty() {
        return "none".to_string();
    }
    run.action_counts
        .iter()
        .take(max_items)
        .map(|r| format!("{} {}/{} ({:.0}%)", r.action, r.count, run.total_decisions, 100.0 * r.share))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_marginal(run: &MDPRunDiagnostics, variable: &str) -> String {
    let row = run.marginals.iter().find(|m| m.variable == variable);
    match row {
        None => "none".to_string(),
        Some(row) if row.bins.is_empty() => "none".to_string(),
        Some(row) => row
            .bins
            .iter()
            .map(|b| format!("{}->{} {:.0}%", b.bin, b.dominant_action, 100.0 * b.share))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

fn format_signed(x: f64, suffix: &str) -> String {
    let sign = if x > 0.0 { "+" } else { "" };
    format!("{sign}{:.1}{suffix}", x)
}

// ---------------------------------------------------------------------------
// Dropoff / boarding helpers.
// ---------------------------------------------------------------------------

fn choose_dropoff(car: &ElevatorCar, policy: HighrisePolicy) -> Option<f64> {
    if car.passengers.is_empty() {
        return None;
    }
    if policy == HighrisePolicy::FewestStops {
        return Some(car.passengers[0].borrow().to_floor as f64);
    }
    let current = car.current_floor;
    let dir = if car.direction != 0 {
        car.direction
    } else {
        sign(car.passengers[0].borrow().to_floor as f64 - current)
    };
    let mut ahead: Vec<f64> = car
        .passengers
        .iter()
        .map(|p| p.borrow().to_floor as f64)
        .filter(|&f| dir == 0 || sign(f - current) == dir || (f - current).abs() < 1e-9)
        .collect();
    ahead.sort_by(|a, b| (a - current).abs().partial_cmp(&(b - current).abs()).unwrap());
    if let Some(&first) = ahead.first() {
        return Some(first);
    }
    let mut all: Vec<f64> = car.passengers.iter().map(|p| p.borrow().to_floor as f64).collect();
    all.sort_by(|a, b| (a - current).abs().partial_cmp(&(b - current).abs()).unwrap());
    all.first().copied()
}

fn allowed_floors_for(idx: i32, n_floors: i32) -> HashSet<i32> {
    let mut floors = HashSet::new();
    let add_range = |a: i32, b: i32, floors: &mut HashSet<i32>| {
        for f in a.max(0)..=b.min(n_floors - 1) {
            floors.insert(f);
        }
    };
    if idx == 0 {
        add_range(0, n_floors - 1, &mut floors);
    } else if idx == 1 {
        add_range(0, 20, &mut floors);
    } else if idx == 2 {
        floors.insert(0);
        add_range(15, 35, &mut floors);
    } else if idx == 3 {
        floors.insert(0);
        add_range(30, n_floors - 1, &mut floors);
    } else if idx == 4 {
        let mut f = 0;
        while f < n_floors {
            floors.insert(f);
            f += 2;
        }
    } else if idx == 5 {
        floors.insert(0);
        let mut f = 1;
        while f < n_floors {
            floors.insert(f);
            f += 2;
        }
    } else {
        add_range(0, n_floors - 1, &mut floors);
    }
    floors
}

fn dominant_destination(queue: &[PaxRef], car: &ElevatorCar) -> Option<i32> {
    let mut counts: Vec<(i32, i64)> = Vec::new();
    for p in queue {
        let f = p.borrow().to_floor;
        match counts.iter_mut().find(|(k, _)| *k == f) {
            Some((_, c)) => *c += 1,
            None => counts.push((f, 1)),
        }
    }
    let mut best: Option<i32> = None;
    let mut best_count: i64 = -1;
    for (floor, count) in counts {
        let take = count > best_count
            || (count == best_count
                && best.is_some()
                && (floor as f64 - car.current_floor).abs() < (best.unwrap() as f64 - car.current_floor).abs());
        if take {
            best = Some(floor);
            best_count = count;
        }
    }
    best
}

fn largest_destination_group_refs(queue: &[&PaxRef]) -> i64 {
    let mut counts: Vec<(i32, i64)> = Vec::new();
    let mut best = 0;
    for p in queue {
        let f = p.borrow().to_floor;
        let n = match counts.iter_mut().find(|(k, _)| *k == f) {
            Some((_, c)) => {
                *c += 1;
                *c
            }
            None => {
                counts.push((f, 1));
                1
            }
        };
        best = best.max(n);
    }
    best
}

fn average_trip_floors_refs(queue: &[&PaxRef]) -> f64 {
    if queue.is_empty() {
        return 0.0;
    }
    queue.iter().map(|p| (p.borrow().to_floor - p.borrow().from_floor).abs() as f64).sum::<f64>() / queue.len() as f64
}

fn would_add_reverse_stop(car: &ElevatorCar, p: &HighrisePassenger) -> bool {
    if car.direction == 0 {
        return false;
    }
    sign(p.to_floor as f64 - car.current_floor) != car.direction
}

fn sign(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    lo.max(hi.min(x))
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn percentile(xs: &[f64], p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[((sorted.len() - 1) as f64 * p).floor() as usize]
}

fn request_key(floor: f64, dir: i32) -> String {
    format!("{}:{}", floor as i64, dir)
}

// ---------------------------------------------------------------------------
// JSON serialization (results artifact).
// ---------------------------------------------------------------------------

fn jnum(n: f64) -> JsonValue {
    JsonValue::Number(n)
}
fn jstr(s: &str) -> JsonValue {
    JsonValue::String(s.to_string())
}
fn jobj(v: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(v.into_iter().map(|(k, val)| (k.to_string(), val)).collect())
}

fn weights_json(w: &DispatchScoreWeights) -> JsonValue {
    jobj(vec![
        ("distance", jnum(w.distance)),
        ("trip", jnum(w.trip)),
        ("queue", jnum(w.queue)),
        ("wait", jnum(w.wait)),
        ("sameDirection", jnum(w.same_direction)),
        ("destinationGroup", jnum(w.destination_group)),
    ])
}

fn config_json(c: &HighriseElevatorConfig) -> JsonValue {
    jobj(vec![
        ("nFloors", jnum(c.n_floors as f64)),
        ("nElevators", jnum(c.n_elevators as f64)),
        ("capacity", jnum(c.capacity as f64)),
        ("floorTravelTime", jnum(c.floor_travel_time)),
        ("serviceTime", jnum(c.service_time)),
        ("arrivalRate", jnum(c.arrival_rate)),
        ("simT", jnum(c.sim_t)),
        ("drainT", jnum(c.drain_t)),
        ("stepSize", jnum(c.step_size)),
        ("seed", jnum(c.seed as f64)),
        ("localSensorRadius", jnum(c.local_sensor_radius)),
        ("urgentWaitThreshold", jnum(c.urgent_wait_threshold)),
    ])
}

fn schedule_json(schedule: &[ScheduledArrival]) -> JsonValue {
    JsonValue::Array(
        schedule
            .iter()
            .map(|a| jobj(vec![("t", jnum(a.t)), ("fromFloor", jnum(a.from_floor as f64)), ("toFloor", jnum(a.to_floor as f64))]))
            .collect(),
    )
}

fn aggregates_json(a: &HighriseAggregates) -> JsonValue {
    jobj(vec![
        ("n", jnum(a.n as f64)),
        ("nServed", jnum(a.n_served as f64)),
        ("meanWait", jnum(a.mean_wait)),
        ("meanTravel", jnum(a.mean_travel)),
        ("meanTotal", jnum(a.mean_total)),
        ("p95Wait", jnum(a.p95_wait)),
        ("p95Total", jnum(a.p95_total)),
        ("totalStops", jnum(a.total_stops)),
        ("totalDistanceFloors", jnum(a.total_distance_floors)),
        ("totalEnergy", jnum(a.total_energy)),
        ("timedOut", jnum(a.timed_out as f64)),
    ])
}

fn tuning_summary_json(t: &MDPDispatchTuningSummary) -> JsonValue {
    jobj(vec![
        ("observability", jstr(t.observability.slug())),
        ("numStates", jnum(t.num_states as f64)),
        ("gamma", jnum(t.gamma)),
        ("iterations", jnum(t.iterations as f64)),
        ("finalDelta", jnum(t.final_delta)),
        ("learnedWeights", weights_json(&t.learned_weights)),
        (
            "statePolicy",
            JsonValue::Array(
                t.state_policy
                    .iter()
                    .map(|r| jobj(vec![("state", jstr(&r.state)), ("action", jstr(&r.action))]))
                    .collect(),
            ),
        ),
    ])
}

fn run_diag_json(r: &MDPRunDiagnostics) -> JsonValue {
    jobj(vec![
        ("totalDecisions", jnum(r.total_decisions as f64)),
        (
            "actionCounts",
            JsonValue::Array(
                r.action_counts
                    .iter()
                    .map(|c| jobj(vec![("action", jstr(&c.action)), ("count", jnum(c.count as f64)), ("share", jnum(c.share))]))
                    .collect(),
            ),
        ),
        (
            "topStates",
            JsonValue::Array(
                r.top_states
                    .iter()
                    .map(|s| jobj(vec![("state", jstr(&s.state)), ("action", jstr(&s.action)), ("count", jnum(s.count as f64))]))
                    .collect(),
            ),
        ),
        (
            "marginals",
            JsonValue::Array(
                r.marginals
                    .iter()
                    .map(|m| {
                        jobj(vec![
                            ("variable", jstr(&m.variable)),
                            (
                                "bins",
                                JsonValue::Array(
                                    m.bins
                                        .iter()
                                        .map(|b| {
                                            jobj(vec![
                                                ("bin", jstr(&b.bin)),
                                                ("count", jnum(b.count as f64)),
                                                ("dominantAction", jstr(&b.dominant_action)),
                                                ("share", jnum(b.share)),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

fn marginal_json(m: &MarginalComparison) -> JsonValue {
    jobj(vec![
        ("baselinePolicy", jstr(m.baseline_policy.slug())),
        ("baselineAuthority", jstr(m.baseline_authority.slug())),
        ("meanWaitDelta", jnum(m.mean_wait_delta)),
        ("meanTotalDelta", jnum(m.mean_total_delta)),
        ("stopsDelta", jnum(m.stops_delta)),
        ("energyDelta", jnum(m.energy_delta)),
    ])
}

fn result_json(r: &HighriseElevatorResult) -> JsonValue {
    let mut fields = vec![
        ("policy", jstr(r.policy.slug())),
        ("authority", jstr(r.authority.slug())),
        ("config", config_json(&r.config)),
        ("schedule", schedule_json(&r.schedule)),
        (
            "people",
            JsonValue::Array(
                r.people
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
            ),
        ),
        ("aggregates", aggregates_json(&r.aggregates)),
    ];
    if let Some(t) = &r.mdp_tuning {
        fields.push(("mdpTuning", tuning_summary_json(t)));
    }
    if let Some(run) = &r.mdp_run {
        fields.push(("mdpRun", run_diag_json(run)));
    }
    if let Some(m) = &r.marginal_vs_lowest_time {
        fields.push(("marginalVsLowestTime", marginal_json(m)));
    }
    jobj(fields)
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

fn default_config() -> HighriseElevatorConfig {
    HighriseElevatorConfig {
        n_floors: env_f64_opt("FLOORS", 50.0) as i32,
        n_elevators: env_f64_opt("ELEVATORS", 6.0) as i32,
        capacity: env_f64_opt("CAPACITY", 12.0) as i32,
        floor_travel_time: env_f64_opt("TRAVEL_T", 1.35),
        service_time: env_f64_opt("SERVICE_T", 3.0),
        arrival_rate: env_f64_opt("LAMBDA", 0.22),
        sim_t: env_f64_opt("SIM_T", 360.0),
        drain_t: env_f64_opt("DRAIN_T", 300.0),
        step_size: env_f64_opt("STEPSIZE", 0.1),
        seed: env_f64_opt("SEED", 11.0) as u32,
        local_sensor_radius: env_f64_opt("LOCAL_SENSOR_RADIUS", 12.0),
        urgent_wait_threshold: env_f64_opt("URGENT_WAIT", 45.0),
    }
}

pub fn run() {
    let cfg = default_config();
    let schedule = build_highrise_schedule(&cfg);

    let policies: Vec<HighrisePolicy> = match std::env::var("POLICIES") {
        Ok(s) => s.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()).filter_map(HighrisePolicy::from_slug).collect(),
        Err(_) => HIGHRISE_POLICIES.to_vec(),
    };
    let authorities: Vec<DecisionAuthority> = match std::env::var("AUTHORITIES") {
        Ok(s) => s.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()).filter_map(DecisionAuthority::from_slug).collect(),
        Err(_) => DECISION_AUTHORITIES.to_vec(),
    };
    let record_every = env_f64_opt("RECORD_EVERY", (1.0_f64).max((env_f64_opt("ANIM_DT", 2.0) / cfg.step_size).round())) as i64;

    let mut mdp_tunings: Vec<(HighrisePolicy, MDPDispatchTuning)> = Vec::new();
    for &policy in &policies {
        if is_mdp_policy(policy) && !mdp_tunings.iter().any(|(p, _)| *p == policy) {
            mdp_tunings.push((policy, optimize_highrise_dispatch_mdp(observability_for_policy(policy))));
        }
    }

    println!("# High-rise elevator simulation");
    println!("#   {} floors, {} shafts, capacity {}", cfg.n_floors, cfg.n_elevators, cfg.capacity);
    println!(
        "#   dt={}s, recordEvery={} ticks, local sensor radius={} floors",
        cfg.step_size, record_every, cfg.local_sensor_radius
    );
    println!(
        "#   {} scheduled arrivals, source active {}s, drain {}s",
        schedule.len(),
        cfg.sim_t,
        cfg.drain_t
    );
    for (policy, tuning) in &mdp_tunings {
        let w = &tuning.learned_weights;
        println!(
            "#   {} VI: {} abstract states, {} actions, {} sweeps, max|ΔV|={:e}",
            policy.label(),
            tuning.num_states,
            tuning.actions.len(),
            tuning.iterations,
            tuning.final_delta
        );
        println!(
            "#   MDP learned weights: distance={:.3}, trip={:.3}, queue={:.3}, wait={:.3}, sameDir={:.3}, destGroup={:.3}",
            w.distance, w.trip, w.queue, w.wait, w.same_direction, w.destination_group
        );
    }

    let mut results: Vec<HighriseElevatorResult> = Vec::new();
    for &authority in &authorities {
        for &policy in &policies {
            let tuning = mdp_tunings.iter().find(|(p, _)| *p == policy).map(|(_, t)| t.clone());
            let mut result = run_highrise_elevators(&cfg, policy, &schedule, HighriseRunOptions { authority, mdp_tuning: tuning });
            if is_mdp_policy(policy) {
                if let Some(baseline) = results
                    .iter()
                    .find(|r| r.authority == authority && r.policy == HighrisePolicy::LowestTotalTime)
                {
                    result.marginal_vs_lowest_time = Some(compare_to_baseline(&result, baseline));
                }
            }
            let a = &result.aggregates;
            println!();
            println!("# {} / {}", policy.label(), authority.label());
            println!(
                "#   served {}/{}, mean wait {:.1}s, mean total {:.1}s, p95 total {:.1}s",
                a.n_served, a.n, a.mean_wait, a.mean_total, a.p95_total
            );
            println!(
                "#   stops {}, distance {:.1} floors, energy index {:.1}",
                a.total_stops, a.total_distance_floors, a.total_energy
            );
            if let Some(run) = &result.mdp_run {
                let marginal_name = match &result.mdp_tuning {
                    Some(t) if t.observability == MDPObservability::DestinationDispatch => "batch",
                    _ => "wait",
                };
                println!("#   MDP observed actions: {}", format_action_shares(run, 3));
                println!("#   MDP {} marginal: {}", marginal_name, format_marginal(run, marginal_name));
            }
            // variantSummary feeds the animation; computed for parity but unused here.
            let _ = variant_summary(&result);
            results.push(result);
        }
    }

    let _ = std::fs::create_dir_all("out");
    // PORT NOTE: HTML animation player not ported; write a placeholder artifact.
    let html_path = "out/elevator-highrise.html";
    let _ = std::fs::write(
        html_path,
        "<!doctype html><meta charset=\"utf-8\"><title>High-rise elevator dispatch policies</title>\
         <p>PORT NOTE: animation/html-player not ported in the Rust migration. \
         See out/elevator-highrise-results.json for the full data artifact.</p>",
    );

    let tunings_json = JsonValue::Object(
        mdp_tunings
            .iter()
            .map(|(policy, tuning)| (policy.slug().to_string(), tuning_summary_json(&summarize_mdp_tuning(tuning))))
            .collect(),
    );
    let json = jobj(vec![
        ("schedule", schedule_json(&schedule)),
        ("mdpTunings", tunings_json),
        ("results", JsonValue::Array(results.iter().map(result_json).collect())),
    ]);
    let json_path = "out/elevator-highrise-results.json";
    let _ = std::fs::write(json_path, json.to_string_pretty(2));

    println!();
    println!("# wrote {html_path}");
    println!("# wrote {json_path}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_deterministic() {
        let cfg = HighriseElevatorConfig {
            n_floors: 20,
            n_elevators: 3,
            capacity: 8,
            floor_travel_time: 1.0,
            service_time: 2.0,
            arrival_rate: 0.3,
            sim_t: 60.0,
            drain_t: 30.0,
            step_size: 0.5,
            seed: 7,
            local_sensor_radius: 10.0,
            urgent_wait_threshold: 30.0,
        };
        let a = build_highrise_schedule(&cfg);
        let b = build_highrise_schedule(&cfg);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.t - y.t).abs() < 1e-12);
            assert_eq!(x.from_floor, y.from_floor);
            assert_eq!(x.to_floor, y.to_floor);
        }
    }

    #[test]
    fn mdp_state_encode_decode_roundtrips() {
        for obs in [MDPObservability::CallOnly, MDPObservability::DestinationDispatch] {
            for s in 0..mdp_num_states(obs) {
                let st = decode_mdp_dispatch_state(s, obs);
                assert_eq!(encode_mdp_dispatch_bins(&st, obs), s);
            }
        }
    }

    #[test]
    fn bin_index_thresholds() {
        let bins = mdp_wait_bins();
        assert_eq!(bin_index(5.0, &bins), 0);
        assert_eq!(bin_index(30.0, &bins), 1);
        assert_eq!(bin_index(1000.0, &bins), 2);
    }
}
