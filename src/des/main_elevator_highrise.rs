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
//!     shapes, charts, `buildHTMLSet`) is ported: `run_highrise_elevators`
//!     records per-tick frames + a system-trajectory chart into an
//!     [`Animation`], and `run` collects one variant per policy/authority pair
//!     into a single self-contained `out/elevator-highrise.html` via
//!     [`build_html_set`]. The full data artifact
//!     (`elevator-highrise-results.json`) is produced alongside it.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::des::animation::html_player::{
    build_html_set, build_html_set_external, AnimationSetOptions, AnimationVariant,
    ExternalAnimationVariant,
};
use crate::des::animation::types::{
    js_num, to_fixed, Anchor, Animation, ChartSeries, ChartSpec, CircleShape, FontWeight, Frame,
    FrameParts, LineShape, RectShape, Shape, TextShape,
};
use crate::des::general::pomdp::{mdp_value_iteration, MDPVIOptions, POMDPSpec};
use crate::des::general::prng::{mulberry32, with_seed};
use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};
use crate::des::observability::logger::JsonValue;
use crate::des::shared::capabilities::RandomSource;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
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
    PomdpBelief,
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
            HighrisePolicy::PomdpBelief => "pomdp-belief",
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
            "pomdp-belief" => HighrisePolicy::PomdpBelief,
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
            HighrisePolicy::PomdpBelief => "POMDP belief dispatch",
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
            HighrisePolicy::PomdpBelief => "Tracks a belief over hidden demand classes from call-only observations, then dispatches by expected value.",
        }
    }
}

pub const HIGHRISE_POLICIES: [HighrisePolicy; 8] = [
    HighrisePolicy::FewestStops,
    HighrisePolicy::LowestTotalTime,
    HighrisePolicy::EnergyEfficient,
    HighrisePolicy::CenterPreposition,
    HighrisePolicy::ZonedService,
    HighrisePolicy::MdpCallOnly,
    HighrisePolicy::MdpTuned,
    HighrisePolicy::PomdpBelief,
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
            DecisionAuthority::Central => {
                "One global controller claims requests and coordinates shafts."
            }
            DecisionAuthority::Decentralized => {
                "Each elevator chooses from its local sensor view; duplicate claims are allowed."
            }
            DecisionAuthority::Hybrid => {
                "The controller handles urgent calls while idle cars make local decisions."
            }
        }
    }
}

pub const DECISION_AUTHORITIES: [DecisionAuthority; 3] = [
    DecisionAuthority::Central,
    DecisionAuthority::Decentralized,
    DecisionAuthority::Hybrid,
];

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
impl CarState {
    fn label(self) -> &'static str {
        match self {
            CarState::Idle => "idle",
            CarState::Moving => "moving",
            CarState::Serving => "serving",
            CarState::Prepositioning => "prepositioning",
        }
    }
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

#[derive(Clone)]
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum DispatchDecisionModel {
    Mdp,
    Pomdp,
}
impl DispatchDecisionModel {
    fn slug(self) -> &'static str {
        match self {
            DispatchDecisionModel::Mdp => "mdp",
            DispatchDecisionModel::Pomdp => "pomdp",
        }
    }
}

#[derive(Clone, Copy)]
struct HiddenDemandFeatures {
    queue_len: f64,
    trip: f64,
    destination_group: f64,
}

#[derive(Clone)]
struct POMDPDispatchTuning {
    hidden_state_labels: Vec<String>,
    observation_labels: Vec<String>,
    initial_belief: Vec<f64>,
    observation_likelihood: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
    hidden_features: Vec<HiddenDemandFeatures>,
}

#[derive(Clone)]
struct MDPDispatchTuning {
    model: DispatchDecisionModel,
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
    pomdp: Option<POMDPDispatchTuning>,
}

#[derive(Clone)]
struct StatePolicyRow {
    state: String,
    action: String,
}

#[derive(Clone)]
struct MDPDispatchTuningSummary {
    model: DispatchDecisionModel,
    observability: MDPObservability,
    num_states: usize,
    hidden_states: Option<usize>,
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
    mean_belief_entropy: Option<f64>,
    dominant_hidden_states: Vec<ActionCount>,
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
    belief: Option<Vec<(String, f64)>>,
    belief_entropy: Option<f64>,
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
        HighrisePassenger {
            id,
            from_floor,
            to_floor,
            arrival_time,
            board_time: -1.0,
            exit_time: -1.0,
        }
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
        let floors: Vec<FloorQueues> = (0..config.n_floors)
            .map(|_| FloorQueues::default())
            .collect();
        let elevators: Vec<ElevatorCar> = (0..config.n_elevators)
            .map(|i| {
                let start = ((config.n_floors - 1) as f64 * (i + 1) as f64
                    / (config.n_elevators + 1) as f64)
                    .round();
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
        self.all_arrivals_emitted()
            && self.pending_passenger_count() == 0
            && self.in_car_count() == 0
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
        while self.next_arrival_index < self.schedule.len()
            && self.schedule[self.next_arrival_index].t <= now
        {
            let a = self.schedule[self.next_arrival_index];
            self.next_arrival_index += 1;
            let id = self.people.len() as i64;
            let p = Rc::new(RefCell::new(HighrisePassenger::new(
                id,
                a.from_floor,
                a.to_floor,
                a.t,
            )));
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
                service_floor(
                    &mut self.elevators[k],
                    &mut self.floors,
                    &mut self.completed,
                    &self.config,
                    self.policy,
                    now,
                );
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

            let reached =
                (self.elevators[k].target_floor.unwrap() - self.elevators[k].current_floor).abs()
                    < 1e-9;
            if reached {
                self.elevators[k].current_floor = self.elevators[k].target_floor.unwrap();
                service_floor(
                    &mut self.elevators[k],
                    &mut self.floors,
                    &mut self.completed,
                    &self.config,
                    self.policy,
                    now,
                );
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
                self.elevators[k].set_target(
                    pk.floor as f64,
                    TargetReason::Pickup,
                    pk.dir,
                    n_floors,
                );
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

    fn assign_autonomous_cars(
        &mut self,
        now: f64,
        claimed: &mut HashSet<String>,
        source: DecisionSource,
    ) {
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
                self.elevators[k].set_target(
                    pk.floor as f64,
                    TargetReason::Pickup,
                    pk.dir,
                    n_floors,
                );
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

    if changed
        || car.target_reason == Some(TargetReason::Pickup)
        || car.target_reason == Some(TargetReason::Dropoff)
    {
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
    let queue: &[PaxRef] = if dir > 0 {
        &floors[fidx].up
    } else {
        &floors[fidx].down
    };
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
        if policy == HighrisePolicy::EnergyEfficient
            && !car.passengers.is_empty()
            && would_add_reverse_stop(car, &p.borrow())
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

fn boarding_direction(
    car: &ElevatorCar,
    floors: &[FloorQueues],
    floor: usize,
    policy: HighrisePolicy,
) -> i32 {
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
    let oldest_up = floors[floor]
        .up
        .first()
        .map(|p| p.borrow().arrival_time)
        .unwrap_or(f64::INFINITY);
    let oldest_down = floors[floor]
        .down
        .first()
        .map(|p| p.borrow().arrival_time)
        .unwrap_or(f64::INFINITY);
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
            let queue: &[PaxRef] = if dir > 0 {
                &floors[floor].up
            } else {
                &floors[floor].down
            };
            let eligible: Vec<&PaxRef> = queue
                .iter()
                .filter(|p| can_car_serve_passenger(car, &p.borrow()))
                .collect();
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
            let same_side = if sign(floor as f64 - car.current_floor) == dir {
                1.0
            } else {
                0.0
            };
            let max_group = largest_destination_group_refs(&eligible) as f64;
            let features = PickupFeatures {
                distance,
                oldest_wait,
                queue_len,
                trip,
                same_side,
                max_group,
            };
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

fn score_pickup(
    features: &PickupFeatures,
    policy: HighrisePolicy,
    mdp_tuning: Option<&MDPDispatchTuning>,
) -> f64 {
    let scored = features_for_dispatch_score(features, policy, mdp_tuning);
    let w = weights_for(features, policy, mdp_tuning);
    scored.distance * w.distance + scored.trip * w.trip
        - scored.queue_len * w.queue
        - scored.oldest_wait * w.wait
        - scored.same_side * w.same_direction
        - scored.max_group * w.destination_group
}

fn weights_for(
    features: &PickupFeatures,
    policy: HighrisePolicy,
    mdp_tuning: Option<&MDPDispatchTuning>,
) -> DispatchScoreWeights {
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
    if !is_decision_process_policy(policy) {
        return None;
    }
    let tuning = mdp_tuning?;
    let state_id = encode_mdp_dispatch_state(features, tuning.observability);
    let action_idx = if tuning.model == DispatchDecisionModel::Pomdp {
        tuning
            .pomdp
            .as_ref()
            .map(|p| pomdp_action_for_observation(p, state_id))
            .unwrap_or_else(|| tuning.policy[state_id].max(0) as usize)
    } else {
        tuning.policy[state_id].max(0) as usize
    };
    let action = tuning
        .action_labels
        .get(action_idx)
        .cloned()
        .unwrap_or_else(|| format!("a{action_idx}"));
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
    let (belief, belief_entropy) = match &tuning.pomdp {
        Some(pomdp) => {
            let b = pomdp_belief_for_observation(pomdp, state_id);
            let rows = pomdp
                .hidden_state_labels
                .iter()
                .cloned()
                .zip(b.iter().copied())
                .collect::<Vec<_>>();
            (Some(rows), Some(entropy_bits(&b)))
        }
        None => (None, None),
    };
    log.push(MDPDecisionLogEntry {
        state_id,
        state: tuning
            .state_labels
            .get(state_id)
            .cloned()
            .unwrap_or_else(|| format!("s{state_id}")),
        action,
        bins: mdp_bin_labels(&decode_mdp_dispatch_state(state_id, tuning.observability)),
        belief,
        belief_entropy,
    });
}

fn home_floor(
    car: &ElevatorCar,
    config: &HighriseElevatorConfig,
    policy: HighrisePolicy,
) -> Option<f64> {
    if policy == HighrisePolicy::CenterPreposition || is_decision_process_policy(policy) {
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
            out.push(ScheduledArrival {
                t,
                from_floor,
                to_floor,
            });
        }
        out
    })
}

struct HighriseRunOptions {
    authority: DecisionAuthority,
    mdp_tuning: Option<MDPDispatchTuning>,
    /// Record one animation frame every Nth tick (>= 1).
    record_every_ticks: i64,
}

fn run_highrise_elevators(
    cfg: &HighriseElevatorConfig,
    policy: HighrisePolicy,
    schedule: &[ScheduledArrival],
    opts: HighriseRunOptions,
) -> (HighriseElevatorResult, Animation) {
    let mut building = HighriseBuilding::new(
        cfg.clone(),
        policy,
        schedule.to_vec(),
        opts.authority,
        opts.mdp_tuning.clone(),
    );
    let record_every = opts.record_every_ticks.max(1);
    let mut frames: Vec<Frame> = Vec::new();
    let mut series = HighriseSeries::default();

    let max_ticks = ((cfg.sim_t + cfg.drain_t) / cfg.step_size).round() as i64;
    for tick in 0..=max_ticks {
        building.run_time_step(tick);
        let t = tick as f64 * cfg.step_size;
        // Frames are collected offline (never rendered during the loop), so the
        // player can scrub / reverse / change speed purely over the index.
        if tick % record_every == 0 {
            let parts = build_highrise_frame(t, tick as f64, &building);
            frames.push(parts.into_frame(t, tick as f64));
            series.t.push(t);
            series
                .waiting
                .push(building.pending_passenger_count() as f64);
            series.in_car.push(building.in_car_count() as f64);
            series.served.push(building.completed.len() as f64);
            series.energy.push(building.total_energy());
        }
        if tick as f64 * cfg.step_size >= cfg.sim_t && building.is_drained() {
            break;
        }
    }

    let result = make_result(
        policy,
        opts.authority,
        cfg,
        schedule,
        &building,
        opts.mdp_tuning.as_ref(),
    );
    let animation = Animation {
        width: STAGE_W,
        height: STAGE_H,
        fps: 18.0,
        title: Some("High-rise elevator dispatch policies".to_string()),
        subtitle: Some(format!(
            "{} floors, {} shafts, cap={}, dt={}s, {} arrivals",
            cfg.n_floors,
            cfg.n_elevators,
            cfg.capacity,
            js_num(cfg.step_size),
            schedule.len()
        )),
        frames,
        charts: Some(vec![build_highrise_chart(&series)]),
        background: Some("#ffffff".to_string()),
    };
    (result, animation)
}

fn make_result(
    policy: HighrisePolicy,
    authority: DecisionAuthority,
    config: &HighriseElevatorConfig,
    schedule: &[ScheduledArrival],
    building: &HighriseBuilding,
    mdp_tuning: Option<&MDPDispatchTuning>,
) -> HighriseElevatorResult {
    let served: Vec<PaxRef> = building
        .people
        .iter()
        .filter(|p| p.borrow().exit_time >= 0.0)
        .cloned()
        .collect();
    let waits: Vec<f64> = served
        .iter()
        .map(|p| p.borrow().board_time - p.borrow().arrival_time)
        .collect();
    let travels: Vec<f64> = served
        .iter()
        .map(|p| p.borrow().exit_time - p.borrow().board_time)
        .collect();
    let totals: Vec<f64> = served
        .iter()
        .map(|p| p.borrow().exit_time - p.borrow().arrival_time)
        .collect();
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
// Animation (port of `buildHighriseFrame` / `drawMetrics` / `buildHighriseChart`).
// ---------------------------------------------------------------------------

const STAGE_W: f64 = 1200.0;
const STAGE_H: f64 = 760.0;
const BUILD_X: f64 = 78.0;
const BUILD_Y: f64 = 44.0;
const BUILD_W: f64 = 760.0;
const BUILD_H: f64 = 560.0;
const METRIC_X: f64 = 870.0;
const METRIC_Y: f64 = 44.0;
const METRIC_W: f64 = 290.0;
const METRIC_H: f64 = 560.0;

/// Parallel time-series collected during a run, fed to [`build_highrise_chart`].
#[derive(Default)]
struct HighriseSeries {
    t: Vec<f64>,
    waiting: Vec<f64>,
    in_car: Vec<f64>,
    served: Vec<f64>,
    energy: Vec<f64>,
}

/// Pixel y of a (possibly fractional) floor; floor 0 at the bottom.
fn floor_y(floor: f64, cfg: &HighriseElevatorConfig) -> f64 {
    let span = ((cfg.n_floors - 1) as f64).max(1.0);
    BUILD_Y + BUILD_H - (floor / span) * BUILD_H
}

fn car_color(car: &ElevatorCar) -> &'static str {
    if car.state == CarState::Serving {
        return "#f59e0b";
    }
    if car.state == CarState::Prepositioning {
        return "#7c3aed";
    }
    if car.direction > 0 {
        return "#16a34a";
    }
    if car.direction < 0 {
        return "#2563eb";
    }
    "#9ca3af"
}

fn build_highrise_frame(t: f64, tick: f64, b: &HighriseBuilding) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let cfg = &b.config;
    let n_floors = cfg.n_floors;
    let floor_h = BUILD_H / n_floors as f64;
    let shaft_w = BUILD_W / cfg.n_elevators as f64;
    let car_w = 42.0_f64.min(shaft_w * 0.42);
    let car_h = 7.0_f64.max(floor_h * 0.82);

    shapes.push(Shape::Rect(RectShape {
        x: BUILD_X,
        y: BUILD_Y,
        w: BUILD_W,
        h: BUILD_H,
        fill: "#fff".to_string(),
        stroke: Some("#c8c8c8".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));

    for f in 0..n_floors {
        let y = floor_y(f as f64, cfg);
        let major = f % 5 == 0 || f == n_floors - 1;
        if major {
            shapes.push(Shape::Line(LineShape {
                x1: BUILD_X,
                y1: y,
                x2: BUILD_X + BUILD_W,
                y2: y,
                stroke: (if f == 0 { "#a3a3a3" } else { "#e3e3e3" }).to_string(),
                stroke_width: Some(if f == 0 { 1.2 } else { 1.0 }),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: BUILD_X - 10.0,
                y: y + 4.0,
                text: f.to_string(),
                font_size: Some(10.0),
                fill: Some("#555".to_string()),
                anchor: Some(Anchor::End),
                ..Default::default()
            }));
        }
        let queues = &b.floors[f as usize];
        let waiting = queues.up.len() + queues.down.len();
        if waiting > 0 {
            let y_mid = y - floor_h / 2.0;
            let up_w = 80.0_f64.min(queues.up.len() as f64 * 4.0);
            let down_w = 80.0_f64.min(queues.down.len() as f64 * 4.0);
            if up_w > 0.0 {
                shapes.push(Shape::Rect(RectShape {
                    x: BUILD_X + 8.0,
                    y: y_mid - 4.0,
                    w: up_w,
                    h: 4.0,
                    fill: "#16a34a".to_string(),
                    rx: Some(1.0),
                    ..Default::default()
                }));
            }
            if down_w > 0.0 {
                shapes.push(Shape::Rect(RectShape {
                    x: BUILD_X + 8.0,
                    y: y_mid + 1.0,
                    w: down_w,
                    h: 4.0,
                    fill: "#2563eb".to_string(),
                    rx: Some(1.0),
                    ..Default::default()
                }));
            }
            shapes.push(Shape::Text(TextShape {
                x: BUILD_X + 94.0,
                y: y_mid + 4.0,
                text: waiting.to_string(),
                font_size: Some(8.0),
                fill: Some("#444".to_string()),
                anchor: Some(Anchor::Start),
                ..Default::default()
            }));
        }
    }

    for k in 0..cfg.n_elevators {
        let sx = BUILD_X + k as f64 * shaft_w + shaft_w / 2.0;
        shapes.push(Shape::Line(LineShape {
            x1: sx,
            y1: BUILD_Y,
            x2: sx,
            y2: BUILD_Y + BUILD_H,
            stroke: "#ededed".to_string(),
            stroke_width: Some(1.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: sx,
            y: BUILD_Y + BUILD_H + 18.0,
            text: format!("E{k}"),
            font_size: Some(10.0),
            fill: Some("#555".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }

    for car in &b.elevators {
        let sx = BUILD_X + car.idx as f64 * shaft_w + shaft_w / 2.0;
        let y = floor_y(car.current_floor, cfg) - car_h / 2.0;
        let fill = car_color(car);
        shapes.push(Shape::Rect(RectShape {
            x: sx - car_w / 2.0,
            y,
            w: car_w,
            h: car_h,
            fill: fill.to_string(),
            stroke: Some("#222".to_string()),
            stroke_width: Some(0.7),
            rx: Some(2.0),
            title: Some(format!(
                "{} {} F{} pax={}/{}",
                car.id(),
                car.state.label(),
                to_fixed(car.current_floor, 1),
                car.passengers.len(),
                car.capacity
            )),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: sx,
            y: y + car_h / 2.0 + 3.0,
            text: car.passengers.len().to_string(),
            font_size: Some(9.0),
            fill: Some("#fff".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        if let Some(target) = car.target_floor {
            if car.target_reason != Some(TargetReason::Home) {
                let ty = floor_y(target, cfg);
                shapes.push(Shape::Line(LineShape {
                    x1: sx,
                    y1: y + car_h / 2.0,
                    x2: sx,
                    y2: ty,
                    stroke: "#777".to_string(),
                    stroke_width: Some(0.7),
                    dasharray: Some("2,3".to_string()),
                    opacity: Some(0.75),
                    ..Default::default()
                }));
                shapes.push(Shape::Circle(CircleShape {
                    x: sx,
                    y: ty,
                    r: 2.4,
                    fill: "#777".to_string(),
                    ..Default::default()
                }));
            }
        }
    }

    draw_metrics(&mut shapes, b, t, tick);
    FrameParts::with_caption(
        shapes,
        format!(
            "policy={}  authority={}  t={}s  waiting={}  in-car={}  served={}",
            b.policy.label(),
            b.authority.label(),
            to_fixed(t, 1),
            b.pending_passenger_count(),
            b.in_car_count(),
            b.completed.len()
        ),
    )
}

fn draw_metrics(shapes: &mut Vec<Shape>, b: &HighriseBuilding, t: f64, tick: f64) {
    let result = make_result(
        b.policy,
        b.authority,
        &b.config,
        &b.schedule,
        b,
        b.mdp_tuning.as_ref(),
    );
    let a = &result.aggregates;
    shapes.push(Shape::Rect(RectShape {
        x: METRIC_X,
        y: METRIC_Y,
        w: METRIC_W,
        h: METRIC_H,
        fill: "#fbfbfb".to_string(),
        stroke: Some("#ddd".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: METRIC_X + 14.0,
        y: METRIC_Y + 24.0,
        text: b.policy.label().to_string(),
        font_size: Some(15.0),
        fill: Some("#111".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: METRIC_X + 14.0,
        y: METRIC_Y + 44.0,
        text: format!(
            "{}  tick {}  t={}s",
            b.authority.label(),
            js_num(tick),
            to_fixed(t, 1)
        ),
        font_size: Some(11.0),
        fill: Some("#555".to_string()),
        ..Default::default()
    }));

    let rows: [(&str, String); 9] = [
        ("waiting", b.pending_passenger_count().to_string()),
        ("in cars", b.in_car_count().to_string()),
        ("served", format!("{}/{}", a.n_served, a.n)),
        ("mean wait", format!("{}s", to_fixed(a.mean_wait, 1))),
        ("mean total", format!("{}s", to_fixed(a.mean_total, 1))),
        ("p95 total", format!("{}s", to_fixed(a.p95_total, 1))),
        ("stops", js_num(a.total_stops.round())),
        (
            "distance",
            format!("{} floors", to_fixed(a.total_distance_floors, 1)),
        ),
        ("energy index", to_fixed(a.total_energy, 1)),
    ];
    for (i, (label, value)) in rows.iter().enumerate() {
        let y = METRIC_Y + 76.0 + i as f64 * 24.0;
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 14.0,
            y,
            text: label.to_string(),
            font_size: Some(11.0),
            fill: Some("#666".to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + METRIC_W - 14.0,
            y,
            text: value.clone(),
            font_size: Some(12.0),
            fill: Some("#222".to_string()),
            anchor: Some(Anchor::End),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }

    let y0 = METRIC_Y + 320.0;
    shapes.push(Shape::Text(TextShape {
        x: METRIC_X + 14.0,
        y: y0,
        text: "Shafts".to_string(),
        font_size: Some(12.0),
        fill: Some("#333".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    for car in &b.elevators {
        let y = y0 + 22.0 + car.idx as f64 * 28.0;
        shapes.push(Shape::Rect(RectShape {
            x: METRIC_X + 14.0,
            y: y - 10.0,
            w: 12.0,
            h: 12.0,
            fill: car_color(car).to_string(),
            rx: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 34.0,
            y,
            text: car.id(),
            font_size: Some(11.0),
            fill: Some("#222".to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 66.0,
            y,
            text: format!(
                "F{} {}/{}",
                to_fixed(car.current_floor, 1),
                car.passengers.len(),
                car.capacity
            ),
            font_size: Some(11.0),
            fill: Some("#444".to_string()),
            ..Default::default()
        }));
        let status = match car.target_floor {
            None => format!("{} {}", car.state.label(), car.decision_source.slug()),
            Some(tf) => format!(
                "{} ->F{} {}",
                car.state.label(),
                js_num(tf),
                car.decision_source.slug()
            ),
        };
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 150.0,
            y,
            text: status,
            font_size: Some(10.0),
            fill: Some("#666".to_string()),
            ..Default::default()
        }));
    }
}

fn build_highrise_chart(series: &HighriseSeries) -> ChartSpec {
    ChartSpec {
        x: BUILD_X,
        y: 635.0,
        w: BUILD_W,
        h: 95.0,
        title: Some("System trajectory".to_string()),
        y_min: Some(0.0),
        y_max: None,
        y_label: None,
        series: vec![
            ChartSeries {
                label: "waiting".to_string(),
                color: "#dc2626".to_string(),
                t: series.t.clone(),
                y: series.waiting.clone(),
            },
            ChartSeries {
                label: "in cars".to_string(),
                color: "#2563eb".to_string(),
                t: series.t.clone(),
                y: series.in_car.clone(),
            },
            ChartSeries {
                label: "served".to_string(),
                color: "#16a34a".to_string(),
                t: series.t.clone(),
                y: series.served.clone(),
            },
            ChartSeries {
                label: "energy/10".to_string(),
                color: "#7c3aed".to_string(),
                t: series.t.clone(),
                y: series.energy.iter().map(|v| v / 10.0).collect(),
            },
        ],
        cursor: None,
    }
}

// ---------------------------------------------------------------------------
// Policy weight tables.
// ---------------------------------------------------------------------------

fn dsw(
    distance: f64,
    trip: f64,
    queue: f64,
    wait: f64,
    same_direction: f64,
    destination_group: f64,
) -> DispatchScoreWeights {
    DispatchScoreWeights {
        distance,
        trip,
        queue,
        wait,
        same_direction,
        destination_group,
    }
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
        HighrisePolicy::PomdpBelief => dsw(1.25, 0.2, 1.1, 0.08, 0.25, 0.2),
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
        MDPActionProfile {
            label: "direct-batch",
            weights: dsw(1.8, 0.12, 0.3, 0.035, 0.2, 2.8),
        },
        MDPActionProfile {
            label: "latency",
            weights: dsw(1.05, 0.18, 1.35, 0.105, 0.25, 0.3),
        },
        MDPActionProfile {
            label: "energy",
            weights: dsw(2.45, 0.42, 0.55, 0.04, 1.1, 0.65),
        },
        MDPActionProfile {
            label: "balanced",
            weights: dsw(1.35, 0.2, 0.95, 0.07, 0.45, 0.45),
        },
        MDPActionProfile {
            label: "oldest-first",
            weights: dsw(0.95, 0.16, 0.75, 0.16, 0.2, 0.15),
        },
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

fn is_decision_process_policy(policy: HighrisePolicy) -> bool {
    is_mdp_policy(policy) || policy == HighrisePolicy::PomdpBelief
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
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn optimize_highrise_dispatch_mdp(observability: MDPObservability) -> MDPDispatchTuning {
    let num_states = mdp_num_states(observability);
    let state_labels: Vec<String> = (0..num_states)
        .map(|s| label_mdp_dispatch_state(&decode_mdp_dispatch_state(s, observability)))
        .collect();
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
            Some(Box::new(move |s| {
                labels.get(s).cloned().unwrap_or_default()
            }))
        },
        action_label: {
            let labels = action_labels.clone();
            Some(Box::new(move |a| {
                labels.get(a).cloned().unwrap_or_default()
            }))
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
        model: DispatchDecisionModel::Mdp,
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
        pomdp: None,
    }
}

fn optimize_highrise_dispatch_pomdp() -> MDPDispatchTuning {
    let observability = MDPObservability::CallOnly;
    let num_observations = mdp_num_states(observability);
    let observation_labels: Vec<String> = (0..num_observations)
        .map(|s| label_mdp_dispatch_state(&decode_mdp_dispatch_state(s, observability)))
        .collect();
    let profiles = mdp_action_profiles();
    let action_labels: Vec<String> = profiles.iter().map(|a| a.label.to_string()).collect();
    let hidden_state_labels = pomdp_hidden_state_labels();
    let hidden_features = pomdp_hidden_features();
    let initial_belief = vec![0.34, 0.24, 0.27, 0.15];
    let observation_likelihood = pomdp_observation_likelihood(num_observations);
    let transition = pomdp_hidden_transition(&profiles);
    let reward = pomdp_hidden_reward(&profiles);
    let gamma = env_f64_opt("POMDP_GAMMA", env_f64_opt("MDP_GAMMA", 0.92));

    let transition_rc = Rc::new(transition);
    let observation_rc = Rc::new(observation_likelihood.clone());
    let reward_rc = Rc::new(reward);
    let spec = POMDPSpec {
        states: (0..hidden_state_labels.len()).collect(),
        actions: (0..profiles.len()).collect(),
        observations: (0..num_observations).collect(),
        transition: {
            let t = transition_rc.clone();
            Box::new(move |s: usize, a: usize| t[s][a].clone())
        },
        observation: {
            let o = observation_rc.clone();
            Box::new(move |sp: usize, _a: usize| o[sp].clone())
        },
        reward: {
            let r = reward_rc.clone();
            Box::new(move |s: usize, a: usize| r[s][a])
        },
        discount: gamma,
        initial_belief: Some(initial_belief.clone()),
        is_terminal: None,
    };
    let vi = mdp_value_iteration(
        &spec,
        &MDPVIOptions {
            tol: env_f64_opt("POMDP_TOL", env_f64_opt("MDP_TOL", 1e-8)),
            max_iter: env_f64_opt("POMDP_MAX_ITER", env_f64_opt("MDP_MAX_ITER", 10000.0)) as usize,
        },
    );

    let pomdp = POMDPDispatchTuning {
        hidden_state_labels,
        observation_labels: observation_labels.clone(),
        initial_belief,
        observation_likelihood,
        q: vi.q,
        hidden_features,
    };
    let policy: Vec<i32> = (0..num_observations)
        .map(|obs| pomdp_action_for_observation(&pomdp, obs) as i32)
        .collect();
    let learned_weights = average_mdp_weights(&policy, observability);

    MDPDispatchTuning {
        model: DispatchDecisionModel::Pomdp,
        observability,
        num_states: num_observations,
        actions: profiles,
        policy,
        gamma,
        iterations: vi.iterations,
        final_delta: vi.final_delta,
        learned_weights,
        state_labels: observation_labels,
        action_labels,
        pomdp: Some(pomdp),
    }
}

fn pomdp_hidden_state_labels() -> Vec<String> {
    vec![
        "light-local".to_string(),
        "urgent-aging".to_string(),
        "dense-batch".to_string(),
        "long-haul-batch".to_string(),
    ]
}

fn pomdp_hidden_features() -> Vec<HiddenDemandFeatures> {
    vec![
        HiddenDemandFeatures {
            queue_len: 1.0,
            trip: 7.0,
            destination_group: 1.0,
        },
        HiddenDemandFeatures {
            queue_len: 2.0,
            trip: 14.0,
            destination_group: 1.0,
        },
        HiddenDemandFeatures {
            queue_len: 5.0,
            trip: 18.0,
            destination_group: 3.0,
        },
        HiddenDemandFeatures {
            queue_len: 6.0,
            trip: 32.0,
            destination_group: 4.0,
        },
    ]
}

fn pomdp_observation_likelihood(num_observations: usize) -> Vec<Vec<f64>> {
    let distance = [
        [0.40, 0.35, 0.18, 0.07],
        [0.25, 0.35, 0.28, 0.12],
        [0.45, 0.35, 0.15, 0.05],
        [0.08, 0.22, 0.42, 0.28],
    ];
    let wait = [
        [0.70, 0.23, 0.07],
        [0.12, 0.42, 0.46],
        [0.25, 0.45, 0.30],
        [0.18, 0.38, 0.44],
    ];
    let side = [[0.45, 0.55], [0.50, 0.50], [0.35, 0.65], [0.55, 0.45]];
    let mut out = vec![vec![0.0; num_observations]; 4];
    for hidden in 0..4 {
        let mut z = 0.0;
        for obs in 0..num_observations {
            let st = decode_mdp_dispatch_state(obs, MDPObservability::CallOnly);
            let p = distance[hidden][st.distance_bin]
                * wait[hidden][st.wait_bin]
                * side[hidden][st.same_side];
            out[hidden][obs] = p;
            z += p;
        }
        if z > 0.0 {
            for obs in 0..num_observations {
                out[hidden][obs] /= z;
            }
        }
    }
    out
}

fn pomdp_hidden_transition(profiles: &[MDPActionProfile]) -> Vec<Vec<Vec<f64>>> {
    let mut t = vec![vec![vec![0.0; 4]; profiles.len()]; 4];
    for (a, profile) in profiles.iter().enumerate() {
        for s in 0..4 {
            t[s][a] = match profile.label {
                "direct-batch" => match s {
                    0 => vec![0.64, 0.14, 0.17, 0.05],
                    1 => vec![0.28, 0.38, 0.26, 0.08],
                    2 => vec![0.22, 0.18, 0.50, 0.10],
                    _ => vec![0.12, 0.18, 0.42, 0.28],
                },
                "latency" | "oldest-first" => match s {
                    0 => vec![0.58, 0.27, 0.12, 0.03],
                    1 => vec![0.38, 0.44, 0.14, 0.04],
                    2 => vec![0.24, 0.34, 0.32, 0.10],
                    _ => vec![0.12, 0.30, 0.36, 0.22],
                },
                "energy" => match s {
                    0 => vec![0.70, 0.20, 0.08, 0.02],
                    1 => vec![0.14, 0.52, 0.24, 0.10],
                    2 => vec![0.10, 0.25, 0.46, 0.19],
                    _ => vec![0.05, 0.18, 0.36, 0.41],
                },
                _ => match s {
                    0 => vec![0.62, 0.23, 0.12, 0.03],
                    1 => vec![0.30, 0.44, 0.20, 0.06],
                    2 => vec![0.18, 0.28, 0.42, 0.12],
                    _ => vec![0.08, 0.24, 0.42, 0.26],
                },
            };
        }
    }
    t
}

fn pomdp_hidden_reward(profiles: &[MDPActionProfile]) -> Vec<Vec<f64>> {
    let mut reward = vec![vec![0.0; profiles.len()]; 4];
    for (a, profile) in profiles.iter().enumerate() {
        for hidden in 0..4 {
            reward[hidden][a] = match (hidden, profile.label) {
                (0, "energy") => 5.0,
                (0, "balanced") => 2.0,
                (0, "latency") => 0.5,
                (0, "oldest-first") => 0.0,
                (0, "direct-batch") => -1.5,
                (1, "oldest-first") => 6.0,
                (1, "latency") => 5.5,
                (1, "balanced") => 2.0,
                (1, "direct-batch") => -0.5,
                (1, "energy") => -2.0,
                (2, "direct-batch") => 7.0,
                (2, "balanced") => 3.0,
                (2, "latency") => 2.0,
                (2, "oldest-first") => 1.0,
                (2, "energy") => -1.0,
                (3, "direct-batch") => 5.0,
                (3, "balanced") => 4.0,
                (3, "energy") => 1.0,
                (3, "latency") => 0.5,
                (3, "oldest-first") => 0.0,
                _ => 0.0,
            };
        }
    }
    reward
}

fn abstract_dispatch_outcomes(
    s: usize,
    action_idx: usize,
    observability: MDPObservability,
) -> Vec<Outcome> {
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
        && (st.batch_bin.unwrap_or(0) >= 1
            || (st.queue_bin.unwrap_or(0) >= 1 && st.trip_bin.unwrap_or(0) >= 1))
    {
        1.0
    } else {
        0.0
    };
    let urgency = st.wait_bin as f64 / 2.0;
    let energy_risk = st.distance_bin as f64 / 3.0 + (if same > 0.0 { -0.25 } else { 0.25 });
    let time_cost = d * 1.35 + trip * 0.65 + wait * 0.85 - q * 3.5 - batch * 1.5;
    let stop_cost =
        (2.2 - w.destination_group * direct_demand - st.batch_bin.unwrap_or(0) as f64 * 0.35)
            .max(0.5);
    let energy_cost = d * (1.0 + 0.24 * w.distance) + trip * 0.07 - same * w.same_direction;
    let hidden_queue_penalty = if observability == MDPObservability::CallOnly {
        (urgency - w.wait * 7.0).max(0.0)
            + (0.8 - w.distance * 0.18 - w.same_direction * 0.25).max(0.0)
    } else {
        0.0
    };
    let mismatch = (urgency - w.wait * 8.0).max(0.0)
        + (energy_risk - w.same_direction * 0.45).max(0.0)
        + (direct_demand + st.batch_bin.unwrap_or(0) as f64 * 0.35 - w.destination_group * 0.28)
            .max(0.0)
        + hidden_queue_penalty;
    let reward = -(time_cost + energy_cost * 1.8 + stop_cost * 6.0 + mismatch * 12.0);

    let fit = clamp(
        0.52 + (if w.wait > 0.09 && st.wait_bin >= 1 {
            0.12
        } else {
            0.0
        }) + (if w.destination_group > 1.5 && direct_demand > 0.0 {
            0.12
        } else {
            0.0
        }) + (if w.destination_group > 1.5 && st.batch_bin.unwrap_or(0) >= 1 {
            0.08
        } else {
            0.0
        }) + (if w.same_direction > 0.7 && same > 0.0 {
            0.10
        } else {
            0.0
        }) + (if w.distance > 2.0 && st.distance_bin <= 1 {
            0.08
        } else {
            0.0
        }) - mismatch * 0.05,
        0.2,
        0.88,
    );

    let mut improved = MDPDispatchStateBins {
        distance_bin: st
            .distance_bin
            .saturating_sub(if w.distance > 1.7 { 1 } else { 0 }),
        wait_bin: st
            .wait_bin
            .saturating_sub(if w.wait > 0.08 { 1 } else { 0 }),
        same_side: same as usize,
        ..Default::default()
    };
    let mut degraded = MDPDispatchStateBins {
        distance_bin: (st.distance_bin + if w.distance < 1.2 { 1 } else { 0 })
            .min(mdp_distance_bins().len() - 1),
        wait_bin: (st.wait_bin + if w.wait < 0.07 { 1 } else { 0 }).min(mdp_wait_bins().len() - 1),
        same_side: if same > 0.0 { 1 } else { 0 },
        ..Default::default()
    };
    if observability == MDPObservability::DestinationDispatch {
        improved.queue_bin = Some(st.queue_bin.unwrap_or(0).saturating_sub(
            if w.queue > 0.8 || w.destination_group > 1.5 {
                1
            } else {
                0
            },
        ));
        improved.trip_bin = Some(
            st.trip_bin
                .unwrap_or(0)
                .saturating_sub(if w.destination_group > 1.5 { 1 } else { 0 }),
        );
        improved.batch_bin = Some(
            st.batch_bin
                .unwrap_or(0)
                .saturating_sub(if w.destination_group > 1.5 { 1 } else { 0 }),
        );
        degraded.queue_bin = Some(
            (st.queue_bin.unwrap_or(0) + if w.queue < 0.75 { 1 } else { 0 })
                .min(mdp_queue_bins().len() - 1),
        );
        degraded.trip_bin = Some(st.trip_bin.unwrap_or(0));
        degraded.batch_bin = Some(st.batch_bin.unwrap_or(0));
    }
    vec![
        Outcome {
            prob: fit,
            reward,
            next_state: encode_mdp_dispatch_bins(&improved, observability),
        },
        Outcome {
            prob: 1.0 - fit,
            reward: reward - 5.0 - wait * 0.05,
            next_state: encode_mdp_dispatch_bins(&degraded, observability),
        },
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
    MDPDispatchStateBins {
        distance_bin,
        queue_bin,
        wait_bin,
        trip_bin,
        batch_bin,
        same_side,
    }
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
    parts.push(if st.same_side != 0 {
        "same".to_string()
    } else {
        "reverse".to_string()
    });
    parts.join("/")
}

fn mdp_bin_labels(st: &MDPDispatchStateBins) -> Vec<(String, String)> {
    let mut out = vec![
        ("distance".to_string(), format!("d{}", st.distance_bin)),
        ("wait".to_string(), format!("w{}", st.wait_bin)),
        (
            "direction".to_string(),
            if st.same_side != 0 {
                "same".to_string()
            } else {
                "reverse".to_string()
            },
        ),
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

fn features_for_dispatch_score(
    features: &PickupFeatures,
    policy: HighrisePolicy,
    mdp_tuning: Option<&MDPDispatchTuning>,
) -> PickupFeatures {
    let Some(tuning) = mdp_tuning else {
        return features.clone();
    };
    if !is_decision_process_policy(policy) {
        return features.clone();
    }
    if let Some(pomdp) = &tuning.pomdp {
        let obs = encode_mdp_dispatch_state(features, MDPObservability::CallOnly);
        let belief = pomdp_belief_for_observation(pomdp, obs);
        let hidden = pomdp_expected_hidden_features(pomdp, &belief);
        return PickupFeatures {
            queue_len: hidden.queue_len,
            trip: hidden.trip,
            max_group: hidden.destination_group,
            ..features.clone()
        };
    }
    if tuning.observability == MDPObservability::CallOnly {
        return PickupFeatures {
            queue_len: 1.0,
            trip: CALL_ONLY_EXPECTED_TRIP,
            max_group: 1.0,
            ..features.clone()
        };
    }
    features.clone()
}

fn pomdp_belief_for_observation(pomdp: &POMDPDispatchTuning, obs_id: usize) -> Vec<f64> {
    let mut b = vec![0.0; pomdp.initial_belief.len()];
    let mut z = 0.0;
    for s in 0..b.len() {
        let likelihood = pomdp
            .observation_likelihood
            .get(s)
            .and_then(|row| row.get(obs_id))
            .copied()
            .unwrap_or(0.0);
        b[s] = pomdp.initial_belief[s] * likelihood;
        z += b[s];
    }
    if z <= 0.0 || !z.is_finite() {
        return pomdp.initial_belief.clone();
    }
    for p in &mut b {
        *p /= z;
    }
    b
}

fn pomdp_expected_hidden_features(
    pomdp: &POMDPDispatchTuning,
    belief: &[f64],
) -> HiddenDemandFeatures {
    let mut out = HiddenDemandFeatures {
        queue_len: 0.0,
        trip: 0.0,
        destination_group: 0.0,
    };
    for (s, &p) in belief.iter().enumerate() {
        let f = pomdp.hidden_features[s];
        out.queue_len += p * f.queue_len;
        out.trip += p * f.trip;
        out.destination_group += p * f.destination_group;
    }
    out
}

fn pomdp_action_for_observation(pomdp: &POMDPDispatchTuning, obs_id: usize) -> usize {
    let belief = pomdp_belief_for_observation(pomdp, obs_id);
    let mut best_a = 0usize;
    let mut best_q = f64::NEG_INFINITY;
    for a in 0..pomdp.q.first().map(|r| r.len()).unwrap_or(0) {
        let mut q = 0.0;
        for (s, &p) in belief.iter().enumerate() {
            q += p * pomdp.q[s][a];
        }
        if q > best_q {
            best_q = q;
            best_a = a;
        }
    }
    best_a
}

fn entropy_bits(p: &[f64]) -> f64 {
    p.iter().filter(|&&x| x > 0.0).map(|&x| -x * x.log2()).sum()
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
    let candidates = [
        0i64,
        1,
        5,
        17,
        43,
        87,
        129,
        173,
        tuning.num_states as i64 - 1,
    ];
    let mut interesting: Vec<usize> = Vec::new();
    for &s in &candidates {
        if s >= 0 && (s as usize) < tuning.num_states && !interesting.contains(&(s as usize)) {
            interesting.push(s as usize);
        }
    }
    MDPDispatchTuningSummary {
        model: tuning.model,
        observability: tuning.observability,
        num_states: tuning.num_states,
        hidden_states: tuning.pomdp.as_ref().map(|p| p.hidden_state_labels.len()),
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
    let mut belief_entropy_sum = 0.0;
    let mut belief_entropy_count = 0usize;
    let mut hidden_counts: Vec<(String, usize)> = Vec::new();
    // marginal: variable -> bin -> action -> count, preserving insertion order.
    let mut marginal: Vec<(String, Vec<(String, Vec<(String, usize)>)>)> = Vec::new();

    for row in log {
        bump(&mut action_counts, &row.action);
        if let Some(h) = row.belief_entropy {
            belief_entropy_sum += h;
            belief_entropy_count += 1;
        }
        if let Some(belief) = &row.belief {
            if let Some((label, _)) = belief.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                bump(&mut hidden_counts, label);
            }
        }
        let state_key = format!("{}|{}", row.state, row.action);
        match state_counts.iter_mut().find(|(k, _)| *k == state_key) {
            Some((_, ts)) => ts.count += 1,
            None => state_counts.push((
                state_key,
                TopState {
                    state: row.state.clone(),
                    action: row.action.clone(),
                    count: 1,
                },
            )),
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
        .map(|(action, count)| ActionCount {
            action,
            count,
            share: if total > 0 {
                count as f64 / total as f64
            } else {
                0.0
            },
        })
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
                        share: if count > 0 {
                            dominant_count as f64 / count as f64
                        } else {
                            0.0
                        },
                    }
                })
                .collect();
            bins.sort_by(|a, b| a.bin.cmp(&b.bin));
            MarginalRow { variable, bins }
        })
        .collect();
    marginals.sort_by(|a, b| a.variable.cmp(&b.variable));

    let mut dominant_hidden_states: Vec<ActionCount> = hidden_counts
        .into_iter()
        .map(|(action, count)| ActionCount {
            action,
            count,
            share: if belief_entropy_count > 0 {
                count as f64 / belief_entropy_count as f64
            } else {
                0.0
            },
        })
        .collect();
    dominant_hidden_states.sort_by(|a, b| b.count.cmp(&a.count));

    MDPRunDiagnostics {
        total_decisions: total,
        action_counts: action_count_rows,
        top_states,
        marginals,
        mean_belief_entropy: if belief_entropy_count > 0 {
            Some(belief_entropy_sum / belief_entropy_count as f64)
        } else {
            None
        },
        dominant_hidden_states,
    }
}

fn bump(counts: &mut Vec<(String, usize)>, key: &str) {
    match counts.iter_mut().find(|(k, _)| k == key) {
        Some((_, c)) => *c += 1,
        None => counts.push((key.to_string(), 1)),
    }
}

fn compare_to_baseline(
    result: &HighriseElevatorResult,
    baseline: &HighriseElevatorResult,
) -> MarginalComparison {
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
        let marginal_name = if tuning.observability == MDPObservability::DestinationDispatch {
            "batch"
        } else {
            "wait"
        };
        if tuning.model == DispatchDecisionModel::Pomdp {
            let hidden = run
                .dominant_hidden_states
                .iter()
                .take(2)
                .map(|r| format!("{} {:.0}%", r.action, 100.0 * r.share))
                .collect::<Vec<_>>()
                .join(", ");
            out += &format!(
                " POMDP uses QMDP over {} hidden demand states and {} call-only observations, then this run exercised {} belief decisions. Observed actions: {}. Mean belief entropy {:.2} bits; dominant hidden beliefs: {}. Learned weights favor destination grouping={:.2}, distance={:.2}, wait={:.2}.",
                tuning.hidden_states.unwrap_or(0),
                tuning.num_states,
                run.total_decisions,
                format_action_shares(run, 3),
                run.mean_belief_entropy.unwrap_or(0.0),
                if hidden.is_empty() { "none".to_string() } else { hidden },
                w.destination_group,
                w.distance,
                w.wait
            );
        } else {
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
        .map(|r| {
            format!(
                "{} {}/{} ({:.0}%)",
                r.action,
                r.count,
                run.total_decisions,
                100.0 * r.share
            )
        })
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
    ahead.sort_by(|a, b| {
        (a - current)
            .abs()
            .partial_cmp(&(b - current).abs())
            .unwrap()
    });
    if let Some(&first) = ahead.first() {
        return Some(first);
    }
    let mut all: Vec<f64> = car
        .passengers
        .iter()
        .map(|p| p.borrow().to_floor as f64)
        .collect();
    all.sort_by(|a, b| {
        (a - current)
            .abs()
            .partial_cmp(&(b - current).abs())
            .unwrap()
    });
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
                && (floor as f64 - car.current_floor).abs()
                    < (best.unwrap() as f64 - car.current_floor).abs());
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
    queue
        .iter()
        .map(|p| (p.borrow().to_floor - p.borrow().from_floor).abs() as f64)
        .sum::<f64>()
        / queue.len() as f64
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
    let mut fields = vec![
        ("model", jstr(t.model.slug())),
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
    ];
    if let Some(n) = t.hidden_states {
        fields.push(("hiddenStates", jnum(n as f64)));
    }
    jobj(fields)
}

fn run_diag_json(r: &MDPRunDiagnostics) -> JsonValue {
    let mut fields = vec![
        ("totalDecisions", jnum(r.total_decisions as f64)),
        (
            "actionCounts",
            JsonValue::Array(
                r.action_counts
                    .iter()
                    .map(|c| {
                        jobj(vec![
                            ("action", jstr(&c.action)),
                            ("count", jnum(c.count as f64)),
                            ("share", jnum(c.share)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "topStates",
            JsonValue::Array(
                r.top_states
                    .iter()
                    .map(|s| {
                        jobj(vec![
                            ("state", jstr(&s.state)),
                            ("action", jstr(&s.action)),
                            ("count", jnum(s.count as f64)),
                        ])
                    })
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
    ];
    if let Some(h) = r.mean_belief_entropy {
        fields.push(("meanBeliefEntropy", jnum(h)));
        fields.push((
            "dominantHiddenStates",
            JsonValue::Array(
                r.dominant_hidden_states
                    .iter()
                    .map(|c| {
                        jobj(vec![
                            ("state", jstr(&c.action)),
                            ("count", jnum(c.count as f64)),
                            ("share", jnum(c.share)),
                        ])
                    })
                    .collect(),
            ),
        ));
    }
    jobj(fields)
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
        Ok(s) => s
            .split(',')
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .filter_map(HighrisePolicy::from_slug)
            .collect(),
        Err(_) => HIGHRISE_POLICIES.to_vec(),
    };
    let authorities: Vec<DecisionAuthority> = match std::env::var("AUTHORITIES") {
        Ok(s) => s
            .split(',')
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .filter_map(DecisionAuthority::from_slug)
            .collect(),
        Err(_) => DECISION_AUTHORITIES.to_vec(),
    };
    let record_every = env_f64_opt(
        "RECORD_EVERY",
        (1.0_f64).max((env_f64_opt("ANIM_DT", 2.0) / cfg.step_size).round()),
    ) as i64;

    let mut mdp_tunings: Vec<(HighrisePolicy, MDPDispatchTuning)> = Vec::new();
    for &policy in &policies {
        if is_decision_process_policy(policy) && !mdp_tunings.iter().any(|(p, _)| *p == policy) {
            let tuning = if policy == HighrisePolicy::PomdpBelief {
                optimize_highrise_dispatch_pomdp()
            } else {
                optimize_highrise_dispatch_mdp(observability_for_policy(policy))
            };
            mdp_tunings.push((policy, tuning));
        }
    }

    println!("# High-rise elevator simulation");
    println!(
        "#   {} floors, {} shafts, capacity {}",
        cfg.n_floors, cfg.n_elevators, cfg.capacity
    );
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
        if tuning.model == DispatchDecisionModel::Pomdp {
            println!(
                "#   {} QMDP: {} hidden states, {} observations, {} actions, {} sweeps, max|ΔV|={:e}",
                policy.label(),
                tuning.pomdp.as_ref().map(|p| p.hidden_state_labels.len()).unwrap_or(0),
                tuning.num_states,
                tuning.actions.len(),
                tuning.iterations,
                tuning.final_delta
            );
        } else {
            println!(
                "#   {} VI: {} abstract states, {} actions, {} sweeps, max|ΔV|={:e}",
                policy.label(),
                tuning.num_states,
                tuning.actions.len(),
                tuning.iterations,
                tuning.final_delta
            );
        }
        println!(
            "#   {} learned weights: distance={:.3}, trip={:.3}, queue={:.3}, wait={:.3}, sameDir={:.3}, destGroup={:.3}",
            tuning.model.slug().to_uppercase(),
            w.distance, w.trip, w.queue, w.wait, w.same_direction, w.destination_group
        );
    }

    let mut results: Vec<HighriseElevatorResult> = Vec::new();
    let mut variants: Vec<AnimationVariant> = Vec::new();
    for &authority in &authorities {
        for &policy in &policies {
            let tuning = mdp_tunings
                .iter()
                .find(|(p, _)| *p == policy)
                .map(|(_, t)| t.clone());
            let (mut result, animation) = run_highrise_elevators(
                &cfg,
                policy,
                &schedule,
                HighriseRunOptions {
                    authority,
                    mdp_tuning: tuning,
                    record_every_ticks: record_every,
                },
            );
            if is_decision_process_policy(policy) {
                if let Some(baseline) = results.iter().find(|r| {
                    r.authority == authority && r.policy == HighrisePolicy::LowestTotalTime
                }) {
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
                let model = result
                    .mdp_tuning
                    .as_ref()
                    .map(|t| t.model.slug().to_uppercase())
                    .unwrap_or_else(|| "MDP".to_string());
                println!(
                    "#   {model} observed actions: {}",
                    format_action_shares(run, 3)
                );
                println!(
                    "#   {model} {} marginal: {}",
                    marginal_name,
                    format_marginal(run, marginal_name)
                );
                if let Some(h) = run.mean_belief_entropy {
                    println!(
                        "#   POMDP mean belief entropy {:.3} bits; dominant beliefs: {}",
                        h,
                        format_action_shares(
                            &MDPRunDiagnostics {
                                total_decisions: run.total_decisions,
                                action_counts: run.dominant_hidden_states.clone(),
                                top_states: Vec::new(),
                                marginals: Vec::new(),
                                mean_belief_entropy: None,
                                dominant_hidden_states: Vec::new(),
                            },
                            3
                        )
                    );
                }
            }
            let mut controls = HashMap::new();
            controls.insert("policy".to_string(), policy.label().to_string());
            controls.insert("authority".to_string(), authority.label().to_string());
            variants.push(AnimationVariant {
                id: format!("{}-{}", policy.slug(), authority.slug()),
                label: format!("{} / {}", policy.label(), authority.label()),
                controls: Some(controls),
                summary: Some(variant_summary(&result)),
                animation,
            });
            results.push(result);
        }
    }

    let _ = std::fs::create_dir_all("out");
    let asset_dir = std::path::Path::new("out").join("elevator-highrise");
    let _ = std::fs::create_dir_all(&asset_dir);

    let mut external_variants: Vec<ExternalAnimationVariant> = Vec::new();
    let mut external_payload_bytes = 0usize;
    for variant in &variants {
        let rel = format!("elevator-highrise/{}.json", variant.id);
        let path = std::path::Path::new("out").join(&rel);
        let payload = variant.animation.to_json().to_string();
        external_payload_bytes += payload.len();
        let _ = std::fs::write(&path, payload);
        external_variants.push(ExternalAnimationVariant {
            id: variant.id.clone(),
            label: variant.label.clone(),
            href: rel,
            summary: variant.summary.clone(),
            controls: variant.controls.clone(),
        });
    }

    let html_path = "out/elevator-highrise.html";
    let html_opts = AnimationSetOptions {
        title: Some("High-rise elevator dispatch policies".to_string()),
        subtitle: Some(format!(
            "{} floors, {} shafts, dt={}s, {} arrivals. Switch policy and decision authority.",
            cfg.n_floors,
            cfg.n_elevators,
            js_num(cfg.step_size),
            schedule.len()
        )),
        selector_label: None,
    };
    let inline_highrise = std::env::var("INLINE_ANIMATION_DATA").as_deref() == Ok("1")
        || std::env::var("INLINE_HIGHRISE_ANIMATION").as_deref() == Ok("1");
    let html = if inline_highrise {
        build_html_set(&variants, &html_opts)
    } else {
        build_html_set_external(&external_variants, &html_opts)
    };
    let _ = std::fs::write(html_path, html);

    let tunings_json = JsonValue::Object(
        mdp_tunings
            .iter()
            .map(|(policy, tuning)| {
                (
                    policy.slug().to_string(),
                    tuning_summary_json(&summarize_mdp_tuning(tuning)),
                )
            })
            .collect(),
    );
    let json = jobj(vec![
        ("schedule", schedule_json(&schedule)),
        ("mdpTunings", tunings_json),
        (
            "results",
            JsonValue::Array(results.iter().map(result_json).collect()),
        ),
    ]);
    let json_path = "out/elevator-highrise-results.json";
    let _ = std::fs::write(json_path, json.to_string_pretty(2));

    println!();
    println!("# wrote {html_path}");
    println!(
        "# wrote {} lazy animation payloads under out/elevator-highrise/ ({:.1} MB total)",
        external_variants.len(),
        external_payload_bytes as f64 / (1024.0 * 1024.0)
    );
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
        for obs in [
            MDPObservability::CallOnly,
            MDPObservability::DestinationDispatch,
        ] {
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

    #[test]
    fn call_only_models_do_not_score_hidden_destination_features() {
        let mdp = optimize_highrise_dispatch_mdp(MDPObservability::CallOnly);
        let pomdp = optimize_highrise_dispatch_pomdp();
        let sparse = PickupFeatures {
            distance: 6.0,
            oldest_wait: 20.0,
            queue_len: 1.0,
            trip: 5.0,
            same_side: 1.0,
            max_group: 1.0,
        };
        let dense = PickupFeatures {
            queue_len: 12.0,
            trip: 42.0,
            max_group: 9.0,
            ..sparse.clone()
        };
        let mdp_sparse = score_pickup(&sparse, HighrisePolicy::MdpCallOnly, Some(&mdp));
        let mdp_dense = score_pickup(&dense, HighrisePolicy::MdpCallOnly, Some(&mdp));
        assert!(
            (mdp_sparse - mdp_dense).abs() < 1e-12,
            "call-only MDP leaked hidden queue/destination features"
        );
        let pomdp_sparse = score_pickup(&sparse, HighrisePolicy::PomdpBelief, Some(&pomdp));
        let pomdp_dense = score_pickup(&dense, HighrisePolicy::PomdpBelief, Some(&pomdp));
        assert!(
            (pomdp_sparse - pomdp_dense).abs() < 1e-12,
            "POMDP score should use belief expectations, not hidden features"
        );
    }

    #[test]
    fn pomdp_beliefs_normalize_and_policy_uses_legal_actions() {
        let tuning = optimize_highrise_dispatch_pomdp();
        let pomdp = tuning.pomdp.as_ref().expect("POMDP tuning");
        let mut actions = HashSet::new();
        for obs in 0..tuning.num_states {
            let b = pomdp_belief_for_observation(pomdp, obs);
            let sum: f64 = b.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "belief sum={sum}");
            assert!(b.iter().all(|p| *p >= 0.0 && p.is_finite()));
            let a = pomdp_action_for_observation(pomdp, obs);
            assert!(a < tuning.actions.len());
            actions.insert(a);
        }
        assert!(
            actions.len() >= 2,
            "POMDP policy should react to different call-only observations"
        );
    }

    #[test]
    fn abstract_mdp_outcomes_are_valid_distributions() {
        for obs in [
            MDPObservability::CallOnly,
            MDPObservability::DestinationDispatch,
        ] {
            for s in 0..mdp_num_states(obs) {
                for a in 0..mdp_action_profiles().len() {
                    let outcomes = abstract_dispatch_outcomes(s, a, obs);
                    let total: f64 = outcomes.iter().map(|o| o.prob).sum();
                    assert!((total - 1.0).abs() < 1e-12, "s={s} a={a} total={total}");
                    for o in outcomes {
                        assert!(o.prob >= 0.0 && o.prob <= 1.0);
                        assert!(o.next_state < mdp_num_states(obs));
                        assert!(o.reward.is_finite());
                    }
                }
            }
        }
    }
}
