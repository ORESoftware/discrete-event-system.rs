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

use crate::des::general::des_base::neural_network::{NeuralNetworkLike, TrainableNeuralNetwork};
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};
use crate::des::shared::capabilities::SeededRandom;

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

const DISPATCH_FEATURE_DIM: usize = 10;

/// Belief over hidden demand at a floor: `empty`, `waiting`, `crowded`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElevatorFloorDemandBelief {
    pub empty: f64,
    pub waiting: f64,
    pub crowded: f64,
}

impl ElevatorFloorDemandBelief {
    pub fn empty() -> Self {
        ElevatorFloorDemandBelief {
            empty: 1.0,
            waiting: 0.0,
            crowded: 0.0,
        }
    }

    fn weights(self) -> [f64; 3] {
        [self.empty, self.waiting, self.crowded]
    }

    fn from_weights(w: [f64; 3]) -> Self {
        ElevatorFloorDemandBelief {
            empty: w[0],
            waiting: w[1],
            crowded: w[2],
        }
    }
}

/// Public car snapshot used by FEL dispatch policies and learning traces.
#[derive(Clone, Debug, PartialEq)]
pub struct ElevatorCarDispatchState {
    pub floor: usize,
    /// -1 = down, 0 = idle, +1 = up.
    pub dir: i8,
    pub doors_open: bool,
    pub active: bool,
    pub moving: bool,
    pub onboard: usize,
    pub capacity: usize,
}

/// Observation at the decision boundary where a hall call is claimed by a shaft.
#[derive(Clone, Debug, PartialEq)]
pub struct ElevatorDispatchObservation {
    pub time: f64,
    pub call_floor: usize,
    pub waiting_at_floor: usize,
    pub floors: usize,
    pub demand_belief: ElevatorFloorDemandBelief,
    pub cars: Vec<ElevatorCarDispatchState>,
}

impl ElevatorDispatchObservation {
    pub fn shafts(&self) -> usize {
        self.cars.len()
    }
}

/// A recorded FEL decision and the policy that made it.
#[derive(Clone, Debug, PartialEq)]
pub struct ElevatorDispatchDecision {
    pub time: f64,
    pub call_floor: usize,
    pub action_car: usize,
    pub policy: String,
}

/// POMDP belief-update trace emitted by FEL runs that use partial-observation
/// demand estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct ElevatorPomdpBeliefTrace {
    pub time: f64,
    pub floor: usize,
    pub action: String,
    pub observation: String,
    pub belief: ElevatorFloorDemandBelief,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DispatchChoice {
    Defer,
    Hold,
    Claim(usize),
}

/// Pluggable hall-call claiming policy for the FEL elevator.
///
/// The default [`ElevatorDispatchPolicy::Look`] preserves the current shared
/// LOOK/nearest-reachable heuristic. MDP and neural variants let a learned or
/// planned policy choose which shaft owns each newly visible call.
#[derive(Clone, Debug)]
pub enum ElevatorDispatchPolicy {
    Look,
    /// Tabular policy over `(call_floor, car_0_floor, ..., car_n_floor)`.
    MdpTable {
        floors: usize,
        shafts: usize,
        policy: Vec<usize>,
    },
    /// Score each candidate shaft with a neural network over the fixed feature
    /// vector returned by [`elevator_dispatch_features`].
    NeuralScorer {
        network: FeedForwardNetwork,
    },
    /// POMDP/QMDP floor-demand policy. A noisy hall-call belief chooses between
    /// holding for more evidence and dispatching the closest useful shaft.
    PomdpBelief {
        dispatch_margin: f64,
    },
    /// Online neural TD scorer updated at each dispatch decision.
    NeuralTdScorer {
        network: FeedForwardNetwork,
        learning_rate: f64,
        gamma: f64,
        updates: usize,
        loss_history: Vec<f64>,
    },
}

impl Default for ElevatorDispatchPolicy {
    fn default() -> Self {
        ElevatorDispatchPolicy::Look
    }
}

impl ElevatorDispatchPolicy {
    pub fn label(&self) -> &'static str {
        match self {
            ElevatorDispatchPolicy::Look => "look",
            ElevatorDispatchPolicy::MdpTable { .. } => "mdp-table",
            ElevatorDispatchPolicy::NeuralScorer { .. } => "neural-scorer",
            ElevatorDispatchPolicy::PomdpBelief { .. } => "pomdp-belief",
            ElevatorDispatchPolicy::NeuralTdScorer { .. } => "neural-td",
        }
    }

    fn choose(&mut self, obs: &ElevatorDispatchObservation) -> DispatchChoice {
        if obs.cars.is_empty() {
            return DispatchChoice::Defer;
        }
        match self {
            ElevatorDispatchPolicy::Look => DispatchChoice::Defer,
            ElevatorDispatchPolicy::MdpTable {
                floors,
                shafts,
                policy,
            } => {
                if *floors != obs.floors || *shafts != obs.cars.len() {
                    return DispatchChoice::Defer;
                }
                let s = elevator_dispatch_state_index(obs);
                policy
                    .get(s)
                    .copied()
                    .map(|a| DispatchChoice::Claim(a.min(obs.cars.len() - 1)))
                    .unwrap_or(DispatchChoice::Defer)
            }
            ElevatorDispatchPolicy::NeuralScorer { network } => {
                if network.input_dim() != DISPATCH_FEATURE_DIM || network.output_dim() != 1 {
                    return DispatchChoice::Defer;
                }
                let mut best = None;
                let mut best_score = f64::NEG_INFINITY;
                for car_id in 0..obs.cars.len() {
                    let features = elevator_dispatch_features(obs, car_id);
                    let score = network.predict(&features)[0];
                    if score > best_score {
                        best_score = score;
                        best = Some(car_id);
                    }
                }
                best.map(DispatchChoice::Claim)
                    .unwrap_or(DispatchChoice::Defer)
            }
            ElevatorDispatchPolicy::NeuralTdScorer {
                network,
                learning_rate,
                gamma,
                updates,
                loss_history,
            } => {
                if network.input_dim() != DISPATCH_FEATURE_DIM || network.output_dim() != 1 {
                    return DispatchChoice::Defer;
                }
                let targets = neural_td_dispatch_targets(network, obs, *gamma);
                let mut loss = 0.0;
                for (features, target) in targets {
                    loss += network
                        .train_sample(&features, &[target], *learning_rate)
                        .loss;
                    *updates += 1;
                }
                loss_history.push(loss / obs.cars.len().max(1) as f64);
                best_neural_dispatch_car(network, obs)
                    .map(DispatchChoice::Claim)
                    .unwrap_or(DispatchChoice::Defer)
            }
            ElevatorDispatchPolicy::PomdpBelief { dispatch_margin } => {
                let values = elevator_floor_pomdp_belief_action_values(obs.demand_belief);
                if values.dispatch + *dispatch_margin < values.hold {
                    DispatchChoice::Hold
                } else {
                    best_dispatch_car(obs)
                        .map(DispatchChoice::Claim)
                        .unwrap_or(DispatchChoice::Defer)
                }
            }
        }
    }
}

/// Options for neural imitation of the abstract MDP dispatch policy.
#[derive(Clone, Debug)]
pub struct ElevatorNeuralDispatchTrainingOptions {
    pub epochs: usize,
    pub learning_rate: f64,
    pub hidden_layers: Vec<usize>,
    pub seed: u32,
}

impl Default for ElevatorNeuralDispatchTrainingOptions {
    fn default() -> Self {
        ElevatorNeuralDispatchTrainingOptions {
            epochs: 40,
            learning_rate: 0.08,
            hidden_layers: vec![12],
            seed: 7,
        }
    }
}

/// Result of training a neural scorer to imitate the dispatch MDP table.
#[derive(Clone, Debug)]
pub struct ElevatorNeuralDispatchTrainingResult {
    pub policy: ElevatorDispatchPolicy,
    pub loss_history: Vec<f64>,
    pub samples: usize,
    pub mdp_states: usize,
}

/// Options for online neural TD dispatch learning inside the FEL run.
#[derive(Clone, Debug)]
pub struct ElevatorNeuralTdDispatchOptions {
    pub learning_rate: f64,
    pub gamma: f64,
    pub hidden_layers: Vec<usize>,
    pub seed: u32,
}

impl Default for ElevatorNeuralTdDispatchOptions {
    fn default() -> Self {
        ElevatorNeuralTdDispatchOptions {
            learning_rate: 0.05,
            gamma: 0.85,
            hidden_layers: vec![12],
            seed: 17,
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
    home_floor: usize,
    floor: usize,
    dir: Dir,
    doors_open: bool,
    active: bool,
    moving: bool,
    boarding_claim: Option<usize>,
    onboard: Vec<usize>,
}

impl CarState {
    fn new(home_floor: usize) -> Self {
        CarState {
            home_floor,
            floor: home_floor,
            dir: Dir::Idle,
            doors_open: false,
            active: false,
            moving: false,
            boarding_claim: None,
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
    dispatch_policy: ElevatorDispatchPolicy,
    cars: Vec<CarState>,
    waiting: Vec<VecDeque<Passenger>>,
    hall_claims: Vec<Option<usize>>,
    held_floors: Vec<bool>,
    held_recheck_scheduled: Vec<bool>,
    demand_beliefs: Vec<ElevatorFloorDemandBelief>,
    decisions: Vec<ElevatorDispatchDecision>,
    pomdp_beliefs: Vec<ElevatorPomdpBeliefTrace>,
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
        dispatch_policy: ElevatorDispatchPolicy,
    ) -> Self {
        ElevWorld {
            rng: Rng::new(seed),
            floors,
            capacity,
            travel,
            dwell,
            arrival_rate: rate,
            dispatch_policy,
            cars: (0..shafts)
                .map(|i| CarState::new(home_floor(i, shafts, floors)))
                .collect(),
            waiting: (0..floors).map(|_| VecDeque::new()).collect(),
            hall_claims: vec![None; floors],
            held_floors: vec![false; floors],
            held_recheck_scheduled: vec![false; floors],
            demand_beliefs: vec![ElevatorFloorDemandBelief::empty(); floors],
            decisions: Vec::new(),
            pomdp_beliefs: Vec::new(),
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
                    "home": car.home_floor,
                    "floor": car.floor,
                    "dir": car.dir.label(),
                    "doors": car.doors_open,
                    "active": car.active,
                    "moving": car.moving,
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

    fn heuristic_hall_owner(&self, floor: usize) -> Option<usize> {
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
                    + if a.active { 0.15 } else { 0.0 }
                    + (*a_id as f64 * 0.01);
                let score_b = self.look_distance(b, floor)
                    + b.onboard.len() as f64 * 0.35
                    + if b.active { 0.15 } else { 0.0 }
                    + (*b_id as f64 * 0.01);
                score_a.total_cmp(&score_b).then_with(|| a_id.cmp(b_id))
            })
            .map(|(id, _)| id)
    }

    fn hall_owner(&self, floor: usize) -> Option<usize> {
        if self.waiting[floor].is_empty() {
            return None;
        }
        if self.held_floors.get(floor).copied().unwrap_or(false) {
            return None;
        }
        if let Some(car_id) = self.hall_claims.get(floor).and_then(|claim| *claim) {
            if self
                .cars
                .get(car_id)
                .map(|car| car.onboard.len() < self.capacity)
                .unwrap_or(false)
            {
                return Some(car_id);
            }
        }
        self.heuristic_hall_owner(floor)
    }

    fn dispatch_observation(&self, time: f64, floor: usize) -> ElevatorDispatchObservation {
        ElevatorDispatchObservation {
            time,
            call_floor: floor,
            waiting_at_floor: self.waiting[floor].len(),
            floors: self.floors,
            demand_belief: self
                .demand_beliefs
                .get(floor)
                .copied()
                .unwrap_or_else(ElevatorFloorDemandBelief::empty),
            cars: self
                .cars
                .iter()
                .map(|car| ElevatorCarDispatchState {
                    floor: car.floor,
                    dir: dir_code(car.dir),
                    doors_open: car.doors_open,
                    active: car.active,
                    moving: car.moving,
                    onboard: car.onboard.len(),
                    capacity: self.capacity,
                })
                .collect(),
        }
    }

    fn claim_floor(&mut self, time: f64, floor: usize) -> Option<usize> {
        if floor >= self.floors || self.waiting[floor].is_empty() {
            return None;
        }
        let obs = self.dispatch_observation(time, floor);
        let choice = self.dispatch_policy.choose(&obs);
        let chosen = match choice {
            DispatchChoice::Claim(car_id) => self
                .cars
                .get(car_id)
                .map(|car| car.onboard.len() < self.capacity)
                .unwrap_or(false)
                .then_some(car_id)
                .or_else(|| self.heuristic_hall_owner(floor)),
            DispatchChoice::Hold => {
                self.hall_claims[floor] = None;
                self.held_floors[floor] = true;
                return None;
            }
            DispatchChoice::Defer => self.heuristic_hall_owner(floor),
        };
        if let Some(car_id) = chosen {
            self.hall_claims[floor] = Some(car_id);
            self.held_floors[floor] = false;
            self.decisions.push(ElevatorDispatchDecision {
                time,
                call_floor: floor,
                action_car: car_id,
                policy: self.dispatch_policy.label().to_string(),
            });
        }
        chosen
    }

    fn observe_floor_demand(&mut self, time: f64, floor: usize, action: PomdpDemandAction) {
        if floor >= self.floors {
            return;
        }
        let state = true_demand_state(self.waiting[floor].len());
        let observation = sample_demand_observation(&mut self.rng, state);
        let prior = self.demand_beliefs[floor];
        let posterior = update_floor_demand_belief(prior, action, observation);
        self.demand_beliefs[floor] = posterior;
        self.pomdp_beliefs.push(ElevatorPomdpBeliefTrace {
            time,
            floor,
            action: action.label().to_string(),
            observation: observation.label().to_string(),
            belief: posterior,
        });
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

fn home_floor(car_id: usize, shafts: usize, floors: usize) -> usize {
    if shafts <= 1 {
        0
    } else {
        let top = floors.saturating_sub(1);
        ((car_id * top) + (shafts - 1) / 2) / (shafts - 1)
    }
}

fn dir_code(dir: Dir) -> i8 {
    match dir {
        Dir::Down => -1,
        Dir::Idle => 0,
        Dir::Up => 1,
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
    eng.world
        .observe_floor_demand(now, origin, PomdpDemandAction::Hold);
    eng.world.claim_floor(now, origin);
    schedule_held_recheck(eng, origin);
    record(eng, "arrival");

    wake_idle_cars(eng);
}

/// The car has finished traversing one floor in its current direction.
fn car_step(eng: &mut Engine<ElevWorld>, car_id: usize) {
    if car_id >= eng.world.cars.len() {
        return;
    }
    if !eng.world.cars[car_id].moving || eng.world.cars[car_id].doors_open {
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
    eng.world.cars[car_id].moving = false;
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
        eng.world.cars[car_id].moving = false;
        eng.world.cars[car_id].active = true;
        eng.world.cars[car_id].doors_open = true;
        eng.world.cars[car_id].boarding_claim = need_board.then_some(f);
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
    let may_board =
        eng.world.cars[car_id].boarding_claim == Some(f) || eng.world.hall_owner(f) == Some(car_id);
    while may_board
        && !eng.world.waiting[f].is_empty()
        && eng.world.cars[car_id].onboard.len() < eng.world.capacity
    {
        let p = eng.world.waiting[f].pop_front().expect("nonempty");
        eng.world.total_wait += now - p.arrived;
        eng.world.boarded += 1;
        eng.world.cars[car_id].onboard.push(p.dest);
    }
    eng.world
        .observe_floor_demand(now, f, PomdpDemandAction::Dispatch);
    if eng.world.waiting[f].is_empty() {
        eng.world.hall_claims[f] = None;
        eng.world.held_floors[f] = false;
    } else {
        eng.world.claim_floor(now, f);
        schedule_held_recheck(eng, f);
    }
    eng.world.cars[car_id].doors_open = false;
    eng.world.cars[car_id].boarding_claim = None;
    record(eng, &format!("doors_close:{car_id}"));
    decide_step(eng, car_id);
    wake_idle_cars(eng);
}

fn dispatch_recheck(eng: &mut Engine<ElevWorld>, floor: usize) {
    if floor >= eng.world.floors {
        return;
    }
    eng.world.held_recheck_scheduled[floor] = false;
    if eng.world.waiting[floor].is_empty() {
        eng.world.held_floors[floor] = false;
        eng.world.hall_claims[floor] = None;
        return;
    }
    let now = eng.now();
    eng.world
        .observe_floor_demand(now, floor, PomdpDemandAction::Hold);
    eng.world.claim_floor(now, floor);
    schedule_held_recheck(eng, floor);
    wake_idle_cars(eng);
}

fn schedule_held_recheck(eng: &mut Engine<ElevWorld>, floor: usize) {
    if floor >= eng.world.floors
        || !eng.world.held_floors[floor]
        || eng.world.held_recheck_scheduled[floor]
    {
        return;
    }
    eng.world.held_recheck_scheduled[floor] = true;
    let delay = eng.world.dwell.max(eng.world.travel).max(0.1);
    eng.schedule_after(delay, move |eng| dispatch_recheck(eng, floor));
}

/// LOOK dispatch: continue in the current direction while targets lie ahead,
/// else reverse, else idle (and stop scheduling — the FEL skips to the next
/// arrival).
fn decide_step(eng: &mut Engine<ElevWorld>, car_id: usize) {
    if car_id >= eng.world.cars.len() {
        return;
    }
    if eng.world.cars[car_id].moving || eng.world.cars[car_id].doors_open {
        return;
    }
    let targets = eng.world.targets_for(car_id);
    if targets.is_empty() {
        eng.world.cars[car_id].dir = Dir::Idle;
        eng.world.cars[car_id].active = false;
        eng.world.cars[car_id].moving = false;
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
        eng.world.cars[car_id].moving = false;
        record(eng, &format!("idle:{car_id}"));
        return;
    }
    eng.world.cars[car_id].dir = dir;
    eng.world.cars[car_id].active = true;
    eng.world.cars[car_id].moving = true;
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
#[derive(Clone, Debug)]
pub struct ElevatorConfig {
    pub floors: usize,
    pub shafts: usize,
    pub capacity: usize,
    pub travel: f64,
    pub dwell: f64,
    pub arrival_rate: f64,
    pub horizon: f64,
    pub seed: u64,
    pub dispatch_policy: ElevatorDispatchPolicy,
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
            dispatch_policy: ElevatorDispatchPolicy::Look,
        }
    }
}

/// Run the FEL elevator and return `{ meta, frames }` ready for the animation
/// renderer.
pub fn run_fel_elevator(cfg: &ElevatorConfig) -> Value {
    run_fel_elevator_with_policy(cfg, cfg.dispatch_policy.clone())
}

/// Run the FEL elevator with an explicit hall-call dispatch policy.
pub fn run_fel_elevator_with_policy(
    cfg: &ElevatorConfig,
    dispatch_policy: ElevatorDispatchPolicy,
) -> Value {
    let floors = cfg.floors.max(2);
    let shafts = cfg.shafts.max(1);
    let capacity = cfg.capacity.max(1);
    let travel = finite_at_least(cfg.travel, 0.0, ElevatorConfig::default().travel);
    let dwell = finite_at_least(cfg.dwell, 0.0, ElevatorConfig::default().dwell);
    let arrival_rate = finite_at_least(
        cfg.arrival_rate,
        f64::MIN_POSITIVE,
        ElevatorConfig::default().arrival_rate,
    );
    let horizon = finite_at_least(cfg.horizon, 0.0, ElevatorConfig::default().horizon);
    let mut eng = Engine::new(ElevWorld::new(
        cfg.seed,
        floors,
        shafts,
        capacity,
        travel,
        dwell,
        arrival_rate,
        dispatch_policy,
    ));
    record(&mut eng, "start");
    let first = eng.world.rng.exp(arrival_rate);
    eng.schedule_after(first, passenger_arrival);
    eng.run_until(horizon);

    let events = eng.events_processed();
    let w = &eng.world;
    let policy_label = w.dispatch_policy.label().to_string();
    let (online_updates, online_loss_last) = match &w.dispatch_policy {
        ElevatorDispatchPolicy::NeuralTdScorer {
            updates,
            loss_history,
            ..
        } => (*updates, loss_history.last().copied()),
        _ => (0, None),
    };
    let policy_state = elevator_dispatch_policy_state_json(&w.dispatch_policy);
    let decisions: Vec<Value> = w
        .decisions
        .iter()
        .map(|d| {
            json!({
                "t": d.time,
                "floor": d.call_floor,
                "car": d.action_car,
                "policy": d.policy,
            })
        })
        .collect();
    let pomdp_beliefs: Vec<Value> = w
        .pomdp_beliefs
        .iter()
        .map(|b| {
            json!({
                "t": b.time,
                "floor": b.floor,
                "action": b.action,
                "observation": b.observation,
                "belief": {
                    "empty": b.belief.empty,
                    "waiting": b.belief.waiting,
                    "crowded": b.belief.crowded,
                }
            })
        })
        .collect();
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
            "boarded": eng.world.boarded,
            "served": eng.world.served,
            "meanWait": mean_wait,
            "dispatchPolicy": policy_label,
            "dispatchDecisions": decisions.len(),
            "pomdpBeliefUpdates": pomdp_beliefs.len(),
            "onlineLearningUpdates": online_updates,
            "onlineLearningLossLast": online_loss_last,
        },
        "decisions": decisions,
        "pomdpBeliefs": pomdp_beliefs,
        "policyState": policy_state,
        "frames": frames,
    })
}

/// Serializable final state for a FEL elevator dispatch policy.
///
/// This is the stable handoff payload for Postgres persistence and non-HTML
/// renderers: MDP policies keep their table, neural policies keep weights, and
/// POMDP policies keep their belief-control settings.
pub fn elevator_dispatch_policy_state_json(policy: &ElevatorDispatchPolicy) -> Value {
    match policy {
        ElevatorDispatchPolicy::Look => json!({
            "$schema": "des/fel-elevator-policy-state/v1",
            "kind": "look",
        }),
        ElevatorDispatchPolicy::MdpTable {
            floors,
            shafts,
            policy,
        } => json!({
            "$schema": "des/fel-elevator-policy-state/v1",
            "kind": "mdp-table",
            "floors": floors,
            "shafts": shafts,
            "table": policy,
        }),
        ElevatorDispatchPolicy::NeuralScorer { network } => json!({
            "$schema": "des/fel-elevator-policy-state/v1",
            "kind": "neural-scorer",
            "network": elevator_neural_network_snapshot_json(network),
        }),
        ElevatorDispatchPolicy::PomdpBelief { dispatch_margin } => json!({
            "$schema": "des/fel-elevator-policy-state/v1",
            "kind": "pomdp-belief",
            "dispatchMargin": dispatch_margin,
        }),
        ElevatorDispatchPolicy::NeuralTdScorer {
            network,
            learning_rate,
            gamma,
            updates,
            loss_history,
        } => json!({
            "$schema": "des/fel-elevator-policy-state/v1",
            "kind": "neural-td",
            "learningRate": learning_rate,
            "gamma": gamma,
            "updates": updates,
            "lossHistory": loss_history,
            "network": elevator_neural_network_snapshot_json(network),
        }),
    }
}

pub fn elevator_neural_network_snapshot_json(network: &FeedForwardNetwork) -> Value {
    json!({
        "inputDim": network.input_dim,
        "outputDim": network.output_dim,
        "parameterCount": network.num_parameters(),
        "l2Norm": network.l2_norm(),
        "layers": network.layers.iter().map(|layer| {
            json!({
                "activation": elevator_activation_label(layer.activation),
                "weights": &layer.weights,
                "biases": &layer.biases,
            })
        }).collect::<Vec<_>>(),
    })
}

fn elevator_activation_label(activation: ActivationName) -> &'static str {
    match activation {
        ActivationName::Linear => "linear",
        ActivationName::Sigmoid => "sigmoid",
        ActivationName::Tanh => "tanh",
        ActivationName::Relu => "relu",
    }
}

fn finite_at_least(value: f64, min: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.max(min)
    } else {
        fallback.max(min)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PomdpDemandAction {
    Hold,
    Dispatch,
}

impl PomdpDemandAction {
    fn index(self) -> usize {
        match self {
            PomdpDemandAction::Hold => 0,
            PomdpDemandAction::Dispatch => 1,
        }
    }

    fn label(self) -> &'static str {
        match self {
            PomdpDemandAction::Hold => "hold",
            PomdpDemandAction::Dispatch => "dispatch",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PomdpDemandObservation {
    Quiet,
    Call,
}

impl PomdpDemandObservation {
    fn index(self) -> usize {
        match self {
            PomdpDemandObservation::Quiet => 0,
            PomdpDemandObservation::Call => 1,
        }
    }

    fn label(self) -> &'static str {
        match self {
            PomdpDemandObservation::Quiet => "quiet",
            PomdpDemandObservation::Call => "call",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElevatorFloorPomdpActionValues {
    pub hold: f64,
    pub dispatch: f64,
}

const FLOOR_POMDP_TRANSITION: [[[f64; 3]; 2]; 3] = [
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0]],
    [[0.0, 0.6, 0.4], [1.0, 0.0, 0.0]],
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
];

const FLOOR_POMDP_OBSERVATION: [[f64; 2]; 3] = [[0.85, 0.15], [0.30, 0.70], [0.10, 0.90]];

const FLOOR_POMDP_REWARD: [[f64; 2]; 3] = [[0.0, -3.0], [-1.0, 8.0], [-3.0, 15.0]];

fn true_demand_state(waiting: usize) -> usize {
    match waiting {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

fn sample_demand_observation(rng: &mut Rng, state: usize) -> PomdpDemandObservation {
    let p_quiet = FLOOR_POMDP_OBSERVATION[state.min(2)][0];
    if rng.unit() < p_quiet {
        PomdpDemandObservation::Quiet
    } else {
        PomdpDemandObservation::Call
    }
}

fn update_floor_demand_belief(
    prior: ElevatorFloorDemandBelief,
    action: PomdpDemandAction,
    observation: PomdpDemandObservation,
) -> ElevatorFloorDemandBelief {
    let a = action.index();
    let o = observation.index();
    let prior = prior.weights();
    let mut predicted = [0.0; 3];
    for s in 0..3 {
        for sp in 0..3 {
            predicted[sp] += prior[s] * FLOOR_POMDP_TRANSITION[s][a][sp];
        }
    }

    let mut total = 0.0;
    for sp in 0..3 {
        predicted[sp] *= FLOOR_POMDP_OBSERVATION[sp][o];
        total += predicted[sp];
    }
    if !total.is_finite() || total <= 0.0 {
        return ElevatorFloorDemandBelief {
            empty: 1.0 / 3.0,
            waiting: 1.0 / 3.0,
            crowded: 1.0 / 3.0,
        };
    }
    for x in &mut predicted {
        *x /= total;
    }
    ElevatorFloorDemandBelief::from_weights(predicted)
}

fn floor_pomdp_q_values() -> [[f64; 2]; 3] {
    let gamma = 0.95;
    let mut v = [0.0; 3];
    let mut q = [[0.0; 2]; 3];
    for _ in 0..256 {
        let mut next = [0.0; 3];
        for s in 0..3 {
            for a in 0..2 {
                q[s][a] = FLOOR_POMDP_REWARD[s][a]
                    + gamma
                        * FLOOR_POMDP_TRANSITION[s][a]
                            .iter()
                            .zip(v.iter())
                            .map(|(p, value)| p * value)
                            .sum::<f64>();
            }
            next[s] = q[s][0].max(q[s][1]);
        }
        let delta = (0..3).map(|s| (next[s] - v[s]).abs()).fold(0.0, f64::max);
        v = next;
        if delta < 1e-10 {
            break;
        }
    }
    q
}

pub fn elevator_floor_pomdp_belief_action_values(
    belief: ElevatorFloorDemandBelief,
) -> ElevatorFloorPomdpActionValues {
    let q = floor_pomdp_q_values();
    let weights = belief.weights();
    ElevatorFloorPomdpActionValues {
        hold: (0..3).map(|s| weights[s] * q[s][0]).sum(),
        dispatch: (0..3).map(|s| weights[s] * q[s][1]).sum(),
    }
}

fn look_distance_from_dispatch_obs(
    floors: usize,
    car: &ElevatorCarDispatchState,
    floor: usize,
) -> f64 {
    if car.floor == floor {
        return 0.0;
    }
    let top = floors.saturating_sub(1);
    match car.dir {
        1 if floor >= car.floor => (floor - car.floor) as f64,
        1 => (top - car.floor + top - floor) as f64,
        -1 if floor <= car.floor => (car.floor - floor) as f64,
        -1 => (car.floor + floor) as f64,
        _ => car.floor.abs_diff(floor) as f64,
    }
}

fn best_dispatch_car(obs: &ElevatorDispatchObservation) -> Option<usize> {
    obs.cars
        .iter()
        .enumerate()
        .filter(|(_, car)| car.onboard < car.capacity)
        .min_by(|(a_id, a), (b_id, b)| {
            let score_a = look_distance_from_dispatch_obs(obs.floors, a, obs.call_floor)
                + a.onboard as f64 * 0.35
                + if a.active { 0.15 } else { 0.0 }
                + (*a_id as f64 * 0.01);
            let score_b = look_distance_from_dispatch_obs(obs.floors, b, obs.call_floor)
                + b.onboard as f64 * 0.35
                + if b.active { 0.15 } else { 0.0 }
                + (*b_id as f64 * 0.01);
            score_a.total_cmp(&score_b).then_with(|| a_id.cmp(b_id))
        })
        .map(|(id, _)| id)
}

fn best_neural_dispatch_car(
    network: &FeedForwardNetwork,
    obs: &ElevatorDispatchObservation,
) -> Option<usize> {
    let mut best = None;
    let mut best_score = f64::NEG_INFINITY;
    for car_id in 0..obs.cars.len() {
        if obs.cars[car_id].onboard >= obs.cars[car_id].capacity {
            continue;
        }
        let score = network.predict(&elevator_dispatch_features(obs, car_id))[0];
        if score > best_score {
            best_score = score;
            best = Some(car_id);
        }
    }
    best
}

fn neural_td_dispatch_targets(
    network: &FeedForwardNetwork,
    obs: &ElevatorDispatchObservation,
    gamma: f64,
) -> Vec<(Vec<f64>, f64)> {
    (0..obs.cars.len())
        .map(|action| {
            let features = elevator_dispatch_features(obs, action);
            let car = &obs.cars[action];
            let distance = look_distance_from_dispatch_obs(obs.floors, car, obs.call_floor);
            let load_penalty = car.onboard as f64 / car.capacity.max(1) as f64;
            let immediate = -distance - 0.35 * load_penalty;
            let mut next_obs = obs.clone();
            if let Some(next_car) = next_obs.cars.get_mut(action) {
                next_car.floor = obs.call_floor;
                next_car.dir = 0;
                next_car.active = false;
                next_car.moving = false;
                next_car.doors_open = false;
            }
            let bootstrap = (0..next_obs.cars.len())
                .map(|next_action| {
                    network.predict(&elevator_dispatch_features(&next_obs, next_action))[0]
                })
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0);
            (features, immediate + gamma * bootstrap)
        })
        .collect()
}

/// Feature vector used by neural dispatch scoring. The network scores one
/// candidate shaft at a time.
pub fn elevator_dispatch_features(obs: &ElevatorDispatchObservation, car_id: usize) -> Vec<f64> {
    let car = &obs.cars[car_id];
    let top = obs.floors.saturating_sub(1).max(1) as f64;
    let capacity = car.capacity.max(1) as f64;
    let floor = car.floor.min(obs.floors.saturating_sub(1));
    let call = obs.call_floor.min(obs.floors.saturating_sub(1));
    let signed_delta = call as f64 - floor as f64;
    let dir = car.dir as f64;
    let moving_toward = if dir == 0.0 {
        0.0
    } else if signed_delta == 0.0 || signed_delta.signum() == dir.signum() {
        1.0
    } else {
        -1.0
    };
    vec![
        1.0,
        call as f64 / top,
        floor as f64 / top,
        signed_delta.abs() / top,
        signed_delta / top,
        dir,
        moving_toward,
        car.onboard as f64 / capacity,
        if car.active { 1.0 } else { 0.0 },
        if car.doors_open { 1.0 } else { 0.0 },
    ]
}

pub fn elevator_dispatch_feature_dim() -> usize {
    DISPATCH_FEATURE_DIM
}

fn dispatch_state_count(floors: usize, shafts: usize) -> usize {
    let floors = floors.max(2);
    floors * pow_usize(floors, shafts.max(1))
}

fn pow_usize(base: usize, exp: usize) -> usize {
    (0..exp).fold(1usize, |acc, _| acc.saturating_mul(base))
}

fn encode_dispatch_state(floors: usize, call_floor: usize, car_floors: &[usize]) -> usize {
    let mut idx = call_floor.min(floors - 1);
    let mut stride = floors;
    for &floor in car_floors {
        idx += stride * floor.min(floors - 1);
        stride *= floors;
    }
    idx
}

fn decode_dispatch_state(floors: usize, shafts: usize, state: usize) -> (usize, Vec<usize>) {
    let call_floor = state % floors;
    let mut rest = state / floors;
    let mut car_floors = Vec::with_capacity(shafts);
    for _ in 0..shafts {
        car_floors.push(rest % floors);
        rest /= floors;
    }
    (call_floor, car_floors)
}

fn elevator_dispatch_state_index(obs: &ElevatorDispatchObservation) -> usize {
    let car_floors: Vec<usize> = obs.cars.iter().map(|car| car.floor).collect();
    encode_dispatch_state(obs.floors, obs.call_floor, &car_floors)
}

fn dispatch_observation_from_state(
    floors: usize,
    shafts: usize,
    state: usize,
) -> ElevatorDispatchObservation {
    let (call_floor, car_floors) = decode_dispatch_state(floors, shafts, state);
    ElevatorDispatchObservation {
        time: 0.0,
        call_floor,
        waiting_at_floor: 1,
        floors,
        demand_belief: ElevatorFloorDemandBelief {
            empty: 0.05,
            waiting: 0.80,
            crowded: 0.15,
        },
        cars: car_floors
            .into_iter()
            .map(|floor| ElevatorCarDispatchState {
                floor,
                dir: 0,
                doors_open: false,
                active: false,
                moving: false,
                onboard: 0,
                capacity: 1,
            })
            .collect(),
    }
}

/// Abstract shaft-assignment MDP aligned with the FEL dispatch boundary.
///
/// State is `(call_floor, car_0_floor, ..., car_n_floor)`. Action is the shaft
/// assigned to the current call. Serving a call moves that car to the call floor
/// and then samples the next call uniformly. The reward is negative travel
/// distance, which gives a compact value-iteration baseline for dispatch.
pub fn elevator_dispatch_mdp_model(floors: usize, shafts: usize) -> MDPSpec {
    let floors = floors.max(2);
    let shafts = shafts.max(1);
    let num_states = dispatch_state_count(floors, shafts);
    MDPSpec {
        num_states,
        num_actions: Box::new(move |_s| shafts),
        outcomes: Box::new(move |s, a| {
            let (call_floor, mut car_floors) = decode_dispatch_state(floors, shafts, s);
            let action = a.min(shafts - 1);
            let distance = car_floors[action].abs_diff(call_floor) as f64;
            car_floors[action] = call_floor;
            let reward = -distance;
            let p = 1.0 / floors as f64;
            (0..floors)
                .map(|next_call| Outcome {
                    prob: p,
                    reward,
                    next_state: encode_dispatch_state(floors, next_call, &car_floors),
                })
                .collect()
        }),
        is_terminal: None,
        terminal_reward: None,
        state_label: Some(Box::new(move |s| {
            let (call, cars) = decode_dispatch_state(floors, shafts, s);
            format!("call@{call};cars={cars:?}")
        })),
        action_label: Some(Box::new(|a| format!("shaft-{a}"))),
    }
}

/// Solve the abstract dispatch MDP into a table policy usable by the FEL run.
pub fn solve_elevator_dispatch_mdp_policy(cfg: &ElevatorConfig) -> ElevatorDispatchPolicy {
    let floors = cfg.floors.max(2);
    let shafts = cfg.shafts.max(1);
    let state_count = dispatch_state_count(floors, shafts);
    let result = value_iteration(
        elevator_dispatch_mdp_model(floors, shafts),
        VIOptions {
            gamma: 0.85,
            tol: 1e-8,
            max_iter: 1500,
            random_tie_break: false,
            ..Default::default()
        },
    );
    let mut policy = vec![0usize; state_count];
    for (i, &a) in result.policy.iter().enumerate().take(state_count) {
        policy[i] = if a < 0 {
            0
        } else {
            (a as usize).min(shafts - 1)
        };
    }
    ElevatorDispatchPolicy::MdpTable {
        floors,
        shafts,
        policy,
    }
}

/// Construct a POMDP/QMDP demand-belief policy for the FEL dispatcher.
pub fn elevator_pomdp_belief_dispatch_policy(dispatch_margin: f64) -> ElevatorDispatchPolicy {
    ElevatorDispatchPolicy::PomdpBelief { dispatch_margin }
}

/// Construct an online neural TD dispatch policy that learns during the FEL run.
pub fn elevator_neural_td_dispatch_policy(
    opts: &ElevatorNeuralTdDispatchOptions,
) -> ElevatorDispatchPolicy {
    let mut rng = SeededRandom::new(opts.seed);
    let network = FeedForwardNetwork::random(
        &RandomNetworkSpec {
            input_dim: DISPATCH_FEATURE_DIM,
            hidden_layers: opts.hidden_layers.clone(),
            output_dim: 1,
            hidden_activation: ActivationName::Tanh,
            output_activation: ActivationName::Linear,
            weight_scale: Some(0.01),
        },
        &mut rng,
    );
    ElevatorDispatchPolicy::NeuralTdScorer {
        network,
        learning_rate: opts.learning_rate,
        gamma: opts.gamma,
        updates: 0,
        loss_history: Vec::new(),
    }
}

/// Train a neural dispatch scorer to imitate the abstract MDP policy table.
///
/// This is intentionally an imitation bootstrap, not the final RL loop: it gives
/// the FEL a working neural policy surface immediately, and the same decision
/// trace can later be used for online TD or policy-gradient updates.
pub fn train_elevator_neural_dispatch_policy(
    cfg: &ElevatorConfig,
    opts: &ElevatorNeuralDispatchTrainingOptions,
) -> ElevatorNeuralDispatchTrainingResult {
    let floors = cfg.floors.max(2);
    let shafts = cfg.shafts.max(1);
    let state_count = dispatch_state_count(floors, shafts);
    let mdp_policy = solve_elevator_dispatch_mdp_policy(cfg);
    let policy_table = match &mdp_policy {
        ElevatorDispatchPolicy::MdpTable { policy, .. } => policy.clone(),
        _ => unreachable!("solver always returns a table policy"),
    };

    let mut rng = SeededRandom::new(opts.seed);
    let mut network = FeedForwardNetwork::random(
        &RandomNetworkSpec {
            input_dim: DISPATCH_FEATURE_DIM,
            hidden_layers: opts.hidden_layers.clone(),
            output_dim: 1,
            hidden_activation: ActivationName::Tanh,
            output_activation: ActivationName::Sigmoid,
            weight_scale: None,
        },
        &mut rng,
    );

    let samples = state_count * shafts;
    let mut loss_history = Vec::with_capacity(opts.epochs);
    for _ in 0..opts.epochs {
        let mut loss = 0.0;
        for state in 0..state_count {
            let obs = dispatch_observation_from_state(floors, shafts, state);
            let target_action = policy_table[state].min(shafts - 1);
            for action in 0..shafts {
                let x = elevator_dispatch_features(&obs, action);
                let y = if action == target_action { 1.0 } else { 0.0 };
                loss += network.train_sample(&x, &[y], opts.learning_rate).loss;
            }
        }
        loss_history.push(loss / samples.max(1) as f64);
    }

    ElevatorNeuralDispatchTrainingResult {
        policy: ElevatorDispatchPolicy::NeuralScorer { network },
        loss_history,
        samples,
        mdp_states: state_count,
    }
}

/// Run the FEL elevator under baseline LOOK, abstract-MDP dispatch, and neural
/// imitation dispatch, while also returning the canonical MDP/POMDP specs.
pub fn run_fel_elevator_learning_suite(
    cfg: &ElevatorConfig,
    neural_opts: &ElevatorNeuralDispatchTrainingOptions,
) -> Value {
    let baseline = run_fel_elevator_with_policy(cfg, ElevatorDispatchPolicy::Look);
    let mdp_policy = solve_elevator_dispatch_mdp_policy(cfg);
    let mdp = run_fel_elevator_with_policy(cfg, mdp_policy);
    let pomdp = run_fel_elevator_with_policy(cfg, elevator_pomdp_belief_dispatch_policy(0.0));
    let neural_td = run_fel_elevator_with_policy(
        cfg,
        elevator_neural_td_dispatch_policy(&ElevatorNeuralTdDispatchOptions::default()),
    );
    let neural = train_elevator_neural_dispatch_policy(cfg, neural_opts);
    let neural_loss_history = neural.loss_history.clone();
    let neural_samples = neural.samples;
    let neural_mdp_states = neural.mdp_states;
    let neural_run = run_fel_elevator_with_policy(cfg, neural.policy);

    json!({
        "$schema": "des/fel-elevator-learning/v1",
        "runs": {
            "look": baseline,
            "mdpDispatch": mdp,
            "pomdpDispatch": pomdp,
            "neuralTdDispatch": neural_td,
            "neuralDispatch": neural_run,
        },
        "training": {
            "neuralImitation": {
                "lossHistory": neural_loss_history,
                "samples": neural_samples,
                "mdpStates": neural_mdp_states,
            }
        },
        "planningSpecs": {
            "mdp": elevator_mdp_spec(),
            "pomdp": elevator_pomdp_spec(),
        }
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
main{max-width:1120px;margin:0 auto;padding:22px 20px 60px}
h1{font-size:1.4rem;margin:0 0 4px}
.sub{color:#9aa4b2;margin:0 0 14px;font-size:.9rem;line-height:1.5;max-width:82ch}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}
.chip{font-size:.78rem;border:1px solid #2b3344;border-radius:6px;padding:3px 9px;color:#c9d4e3;background:#0f1422}
.chip b{color:#fff}
.stage{display:grid;grid-template-columns:minmax(320px,360px) minmax(420px,1fr);gap:16px;align-items:start}
.panel{border:1px solid #21262d;border-radius:8px;background:#0f1422;padding:14px;min-width:0}
.stats{display:flex;gap:16px;flex-wrap:wrap;font-size:.82rem;color:#9aa4b2;margin:0 0 10px}
.stats b{color:#e6edf3;font-variant-numeric:tabular-nums}
.clock{color:#fff;font-weight:600}
svg{display:block;width:100%}
.shaft-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:10px}
.shaft{border:1px solid #273244;border-radius:6px;background:#0b1021;padding:8px;min-height:58px}
.shaft b{display:block;color:#e6edf3;font-size:.78rem;margin-bottom:4px}
.shaft span{display:block;color:#9aa4b2;font-size:.72rem;line-height:1.35;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.shaft.open{border-color:#22c55e}
.shaft.move{border-color:#3b82f6}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:16px;border-top:1px solid #21262d;padding-top:14px}
button{font:inherit;font-size:.85rem;cursor:pointer;border-radius:8px;padding:7px 12px;border:1px solid #2b3344;background:#161b22;color:#e6edf3}
button:hover{border-color:#3b82f6}
button.primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
.controls label{font-size:.8rem;color:#9aa4b2;display:inline-flex;align-items:center;gap:8px}
#scrub{flex:1;min-width:160px}
.legend{display:flex;gap:16px;align-items:center;font-size:.78rem;color:#9aa4b2;margin:14px 0 4px}
.legend .sw{width:14px;height:3px;border-radius:2px;display:inline-block}
@media (max-width:820px){
  main{padding:18px 12px 42px}
  .stage{grid-template-columns:1fr}
  .panel{padding:12px}
  .shaft-strip{grid-template-columns:repeat(auto-fit,minmax(92px,1fr))}
}
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
    <div class="shaft-strip" id="shaftState"></div>
  </div>
  <div class="panel">
    <div class="stats">
      <span>in&nbsp;car <b id="incar">0</b></span>
      <span>waiting <b id="waiting">0</b></span>
      <span>served <b id="served">0</b></span>
      <span>event <b id="event">start</b></span>
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
const M = DATA.meta || {};
const F = Math.max(2, Number(M.floors) || 2);
const T = Math.max(1, Number(M.horizon) || 1);
const shaftCount = Math.max(1, Number(M.shafts) || 1);
const sourceFrames = Array.isArray(DATA.frames) && DATA.frames.length ? DATA.frames : [{
  t:0, car:0, dir:'idle', doors:false, wait:Array(F).fill(0), inCar:0, served:0, events:0, kind:'start'
}];
const FR = sourceFrames.map(normalizeFrame);

function homeFloor(i){
  return shaftCount <= 1 ? 0 : Math.round(i * (F - 1) / (shaftCount - 1));
}
function normalizeDir(dir){
  return dir === 'up' || dir === 'down' ? dir : 'idle';
}
function normalizeFrame(fr){
  const wait = Array.isArray(fr.wait) ? fr.wait.slice(0, F).map(function(v){return Math.max(0, Number(v) || 0);}) : [];
  while(wait.length < F) wait.push(0);
  const rawCars = Array.isArray(fr.cars) && fr.cars.length ? fr.cars : [{
    id:0, home:0, floor:fr.car, dir:fr.dir, doors:fr.doors, active:fr.dir !== 'idle', moving:fr.dir !== 'idle', inCar:fr.inCar
  }];
  const cars = [];
  for(let i=0;i<shaftCount;i++){
    const src = rawCars[i] || {};
    const floor = Math.min(F - 1, Math.max(0, Math.round(Number(src.floor ?? homeFloor(i)) || 0)));
    const load = Math.max(0, Number(src.inCar) || 0);
    cars.push({
      id:i,
      home:Math.min(F - 1, Math.max(0, Math.round(Number(src.home ?? homeFloor(i)) || 0))),
      floor:floor,
      dir:normalizeDir(src.dir),
      doors:!!src.doors,
      active:!!src.active,
      moving:!!src.moving,
      inCar:load
    });
  }
  return {
    t:Math.max(0, Number(fr.t) || 0),
    car:Math.min(F - 1, Math.max(0, Math.round(Number(fr.car) || 0))),
    dir:String(fr.dir || 'idle'),
    doors:!!fr.doors,
    cars:cars,
    wait:wait,
    inCar:Math.max(0, Number(fr.inCar) || cars.reduce(function(a,c){return a+c.inCar;},0)),
    served:Math.max(0, Number(fr.served) || 0),
    events:Math.max(0, Number(fr.events) || 0),
    kind:String(fr.kind || '')
  };
}

(function(){
  const c=document.getElementById('chips');
  const it=[['floors',F],['shafts',shaftCount],['capacity/shaft',M.capacity],['arrival &lambda;',M.arrivalRate+' /s'],
    ['travel',M.travel+' s/floor'],['dwell',M.dwell+' s'],['horizon',M.horizon+' s'],
    ['FEL events',Number(M.events || 0).toLocaleString()],['arrivals',M.arrivals],['served',M.served],
    ['mean wait',Number(M.meanWait || 0).toFixed(1)+' s'],['dispatch',M.dispatchPolicy || 'look'],
    ['claims',M.dispatchDecisions || 0],['belief updates',M.pomdpBeliefUpdates || 0],
    ['online updates',M.onlineLearningUpdates || 0]];
  c.innerHTML=it.map(function(p){return '<span class="chip">'+p[0]+' <b>'+p[1]+'</b></span>';}).join('');
})();

function frameAt(t){
  let lo=0,hi=FR.length-1,ans=0;
  while(lo<=hi){const m=(lo+hi)>>1; if(FR[m].t<=t){ans=m;lo=m+1;} else hi=m-1;}
  return FR[ans];
}

// ---- building renderer ----
const BW=360, BH=420, MTOP=18, MBOT=22, SHX=132, SHG=8;
const SHW=Math.max(28,Math.min(48,(BW-SHX-16-(shaftCount-1)*SHG)/shaftCount));
const lane = (BH-MTOP-MBOT)/F;
function floorY(f){ return MTOP + (F-1-f)*lane; }  // floor 0 at the bottom
function carsOf(fr){ return fr.cars; }
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

function drawShaftState(fr){
  const html=carsOf(fr).map(function(car){
    const state=car.doors?'open':(car.moving?'move':'');
    const dir=car.dir==='up'?'up':(car.dir==='down'?'down':'idle');
    return '<div class="shaft '+state+'"><b>S'+(car.id+1)+'</b><span>floor '+car.floor+' / home '+car.home+'</span><span>'+dir+' · load '+car.inCar+'</span></div>';
  }).join('');
  document.getElementById('shaftState').innerHTML=html;
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
  drawBuilding(fr); drawShaftState(fr); drawChart(tPlay);
  document.getElementById('clock').textContent=tPlay.toFixed(1)+'s';
  document.getElementById('events').textContent=fr.events.toLocaleString();
  const active=carsOf(fr).filter(function(c){return c.active || c.doors;}).length;
  document.getElementById('dir').textContent=active===0?'idle':active+' active';
  document.getElementById('incar').textContent=fr.inCar;
  document.getElementById('waiting').textContent=fr.wait.reduce(function(a,b){return a+b;},0);
  document.getElementById('served').textContent=fr.served;
  document.getElementById('event').textContent=fr.kind || 'event';
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
        let capacity = data["meta"]["capacity"].as_u64().unwrap() as usize;
        assert_eq!(shafts, 3, "default FEL elevator should use three shafts");
        assert!(served > 0, "expected some passengers served");
        assert!(frames.len() > 10, "expected a frame stream");
        let homes: Vec<usize> = frames[0]["cars"]
            .as_array()
            .unwrap()
            .iter()
            .map(|car| car["home"].as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            homes,
            vec![0, 3, 5],
            "three shafts park across the building"
        );
        // No car may ever be outside the building.
        for f in frames {
            let cars = f["cars"].as_array().expect("frame includes shaft states");
            assert_eq!(cars.len(), shafts, "frame should carry every shaft");
            for car in cars {
                let floor = car["floor"].as_u64().unwrap() as usize;
                assert!(floor < floors, "car floor out of range: {floor}");
                let in_car = car["inCar"].as_u64().unwrap() as usize;
                assert!(in_car <= capacity, "car exceeded capacity: {in_car}");
                let moving = car["moving"].as_bool().unwrap();
                let doors = car["doors"].as_bool().unwrap();
                if moving {
                    assert!(!doors, "car cannot move with doors open");
                    assert_ne!(
                        car["dir"].as_str().unwrap(),
                        "idle",
                        "moving car needs direction"
                    );
                }
            }
        }
    }

    #[test]
    fn fel_elevator_sanitizes_extreme_config_values() {
        let data = run_fel_elevator(&ElevatorConfig {
            floors: 0,
            shafts: 0,
            capacity: 0,
            travel: f64::NAN,
            dwell: f64::INFINITY,
            arrival_rate: f64::NEG_INFINITY,
            horizon: 8.0,
            seed: 7,
            dispatch_policy: ElevatorDispatchPolicy::Look,
        });
        assert_eq!(data["meta"]["floors"], 2);
        assert_eq!(data["meta"]["shafts"], 1);
        assert_eq!(data["meta"]["capacity"], 1);
        assert_eq!(data["meta"]["travel"], ElevatorConfig::default().travel);
        assert_eq!(data["meta"]["dwell"], ElevatorConfig::default().dwell);
        assert_eq!(
            data["meta"]["arrivalRate"],
            ElevatorConfig::default().arrival_rate
        );
        assert_eq!(data["meta"]["horizon"], 8.0);
        for frame in data["frames"].as_array().unwrap() {
            assert_eq!(frame["cars"].as_array().unwrap().len(), 1);
            assert_eq!(frame["wait"].as_array().unwrap().len(), 2);
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

    #[test]
    fn elevator_dispatch_mdp_policy_drives_fel_claims() {
        let cfg = ElevatorConfig {
            floors: 4,
            shafts: 2,
            horizon: 45.0,
            seed: 19,
            ..Default::default()
        };
        let policy = solve_elevator_dispatch_mdp_policy(&cfg);
        let data = run_fel_elevator_with_policy(&cfg, policy);
        assert_eq!(data["meta"]["dispatchPolicy"], "mdp-table");
        assert!(
            data["meta"]["dispatchDecisions"].as_u64().unwrap() > 0,
            "MDP policy should claim calls"
        );
        assert!(
            data["meta"]["served"].as_u64().unwrap() > 0,
            "MDP-driven FEL should still serve passengers"
        );
        for decision in data["decisions"].as_array().unwrap() {
            assert!(decision["car"].as_u64().unwrap() < cfg.shafts as u64);
            assert!(decision["floor"].as_u64().unwrap() < cfg.floors as u64);
        }
    }

    #[test]
    fn elevator_neural_policy_trains_and_drives_fel_claims() {
        let cfg = ElevatorConfig {
            floors: 3,
            shafts: 2,
            horizon: 45.0,
            seed: 23,
            ..Default::default()
        };
        let trained = train_elevator_neural_dispatch_policy(
            &cfg,
            &ElevatorNeuralDispatchTrainingOptions {
                epochs: 6,
                learning_rate: 0.12,
                hidden_layers: vec![6],
                seed: 11,
            },
        );
        assert_eq!(trained.mdp_states, 27);
        assert_eq!(trained.samples, 54);
        assert_eq!(trained.loss_history.len(), 6);
        assert!(trained.loss_history.iter().all(|loss| loss.is_finite()));

        let data = run_fel_elevator_with_policy(&cfg, trained.policy);
        assert_eq!(data["meta"]["dispatchPolicy"], "neural-scorer");
        assert!(
            data["meta"]["dispatchDecisions"].as_u64().unwrap() > 0,
            "neural policy should claim calls"
        );
        assert!(
            data["meta"]["served"].as_u64().unwrap() > 0,
            "neural-driven FEL should still serve passengers"
        );
    }

    #[test]
    fn elevator_pomdp_policy_updates_beliefs_and_drives_fel_claims() {
        let cfg = ElevatorConfig {
            floors: 4,
            shafts: 2,
            horizon: 45.0,
            seed: 31,
            ..Default::default()
        };
        let data = run_fel_elevator_with_policy(&cfg, elevator_pomdp_belief_dispatch_policy(0.0));
        assert_eq!(data["meta"]["dispatchPolicy"], "pomdp-belief");
        assert!(
            data["meta"]["pomdpBeliefUpdates"].as_u64().unwrap() > 0,
            "POMDP policy should update noisy floor-demand beliefs"
        );
        assert!(
            data["meta"]["dispatchDecisions"].as_u64().unwrap() > 0,
            "POMDP policy should claim calls after belief updates"
        );
        assert!(
            data["meta"]["served"].as_u64().unwrap() > 0,
            "POMDP-driven FEL should still serve passengers"
        );
        let belief = &data["pomdpBeliefs"].as_array().unwrap()[0]["belief"];
        let total = belief["empty"].as_f64().unwrap()
            + belief["waiting"].as_f64().unwrap()
            + belief["crowded"].as_f64().unwrap();
        assert!((total - 1.0).abs() < 1e-9, "belief should normalize");
    }

    #[test]
    fn elevator_neural_td_policy_learns_during_fel_run() {
        let cfg = ElevatorConfig {
            floors: 4,
            shafts: 2,
            horizon: 45.0,
            seed: 37,
            ..Default::default()
        };
        let data = run_fel_elevator_with_policy(
            &cfg,
            elevator_neural_td_dispatch_policy(&ElevatorNeuralTdDispatchOptions {
                learning_rate: 0.04,
                gamma: 0.80,
                hidden_layers: vec![6],
                seed: 19,
            }),
        );
        assert_eq!(data["meta"]["dispatchPolicy"], "neural-td");
        assert!(
            data["meta"]["onlineLearningUpdates"].as_u64().unwrap() > 0,
            "online neural TD policy should update during dispatch"
        );
        assert!(
            data["meta"]["onlineLearningLossLast"]
                .as_f64()
                .unwrap()
                .is_finite(),
            "online neural TD loss should be finite"
        );
        assert!(
            data["meta"]["served"].as_u64().unwrap() > 0,
            "online neural TD FEL should still serve passengers"
        );
    }

    #[test]
    fn elevator_learning_suite_returns_runs_training_and_specs() {
        let cfg = ElevatorConfig {
            floors: 3,
            shafts: 2,
            horizon: 30.0,
            seed: 29,
            ..Default::default()
        };
        let suite = run_fel_elevator_learning_suite(
            &cfg,
            &ElevatorNeuralDispatchTrainingOptions {
                epochs: 3,
                learning_rate: 0.12,
                hidden_layers: vec![5],
                seed: 13,
            },
        );
        assert_eq!(suite["$schema"], "des/fel-elevator-learning/v1");
        assert_eq!(
            suite["runs"]["look"]["meta"]["dispatchPolicy"], "look",
            "suite should include baseline LOOK run"
        );
        assert_eq!(
            suite["runs"]["mdpDispatch"]["meta"]["dispatchPolicy"], "mdp-table",
            "suite should include MDP-driven run"
        );
        assert_eq!(
            suite["runs"]["pomdpDispatch"]["meta"]["dispatchPolicy"], "pomdp-belief",
            "suite should include POMDP-driven run"
        );
        assert!(
            suite["runs"]["pomdpDispatch"]["meta"]["pomdpBeliefUpdates"]
                .as_u64()
                .unwrap()
                > 0,
            "suite POMDP run should carry belief updates"
        );
        assert_eq!(
            suite["runs"]["neuralTdDispatch"]["meta"]["dispatchPolicy"], "neural-td",
            "suite should include online neural TD run"
        );
        assert!(
            suite["runs"]["neuralTdDispatch"]["meta"]["onlineLearningUpdates"]
                .as_u64()
                .unwrap()
                > 0,
            "suite neural TD run should learn online"
        );
        assert_eq!(
            suite["runs"]["neuralDispatch"]["meta"]["dispatchPolicy"], "neural-scorer",
            "suite should include neural-driven run"
        );
        assert_eq!(suite["training"]["neuralImitation"]["samples"], 54);
        assert_eq!(
            suite["planningSpecs"]["mdp"]["$schema"], "des/mdp/v1",
            "suite should expose canonical MDP spec"
        );
        assert_eq!(
            suite["planningSpecs"]["pomdp"]["$schema"], "des/pomdp/v1",
            "suite should expose canonical POMDP spec"
        );
    }
}
