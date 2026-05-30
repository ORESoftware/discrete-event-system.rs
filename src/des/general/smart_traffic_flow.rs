//! Port of `src/des/general/smart-traffic-flow.ts` — module
//! `des::general::smart_traffic_flow`.
//!
//! Traffic flow where cars are self-stepping smart movables. A fixed pool of
//! [`SmartTrafficCar`] actors plus one [`SmartTrafficWorldStation`] are handed
//! to the iterative DES runner; the runner shuffles and ticks every active car
//! once per tick (each car proposes its next kinematic state), and the world
//! commits all proposals at the tick barrier (`on_tick` -> `finish_tick`),
//! detecting accidents, advancing lane transitions and recording the trace.
//!
//! Conversion notes from the TS source (the borrow checker fights this file
//! hard — cars reach back into the world that owns them):
//!   * Cars are `Rc<RefCell<SmartTrafficCar>>` and the world holds the pool;
//!     each car holds a `Weak<RefCell<SmartTrafficWorldStation>>` back-pointer
//!     to break the ownership cycle. Both impl [`DESStation`] so they can be
//!     run-loop participants.
//!   * While the runner holds `borrow_mut` on the *currently ticking* car, any
//!     world routine that scans the pool uses `try_borrow` — the running car's
//!     cell is locked so it is skipped, which is exactly correct (a car is
//!     never its own leader / collision partner). Cross-car reads otherwise go
//!     through cheap owned `CarView` snapshots.
//!   * `nextKinematics` copies the leader's needed state (incl. a clone of its
//!     bounded history) into an owned `LeaderView` BEFORE running the math, so
//!     no `Ref` is held across the `rng` draw / world mutations.
//!   * `recordHistory` is moved OUT of `SmartTrafficCar::assign` to its caller
//!     so the car is not re-borrowed while already mutably borrowed. (FLAGGED.)
//!   * `SmartTrafficCellStation` is a plain struct (only the world + cars are
//!     run-loop participants; cell stations are never ticked).
//!   * `logger?` -> `Option<Rc<dyn OptimizationLogger>>`; validators throw ->
//!     `panic!` at the precondition edge.

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::OnceLock;

use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::{
    run_iterative_des, IterativeRunOptions, IterativeRunSummary,
};
use crate::des::general::des_base::smart_movable::{SmartMovable, SmartMovableCore};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::{intrinsic_check, ValidationCheck};
use crate::des::general::network_flow::{
    build_five_intersection_traffic_network, LogEvent, OptimizationLogger, TrafficLane,
    TrafficNetwork, TrafficNode, TrafficNodeKind, TrafficParams, TrafficScheduledTrip,
    TrafficSignal, TrafficSignalPhase, TrafficSource,
};
use crate::des::general::prng::{mulberry32, SeededRandom};
use crate::des::general::random_variables::{
    mean_from_pmf, normalize_pmf, sample_from_pmf, variance_from_pmf, DiscreteConvolveSelf,
};
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::transform::Transform;

const MODEL: &str = "SmartTrafficWorldStation";

// Driver-trait base distribution: 4-fold self-convolution of a flat 7-bin PMF.
struct DriverTrait {
    pmf: Vec<f64>,
    mean: f64,
    std: f64,
}

fn driver_trait() -> &'static DriverTrait {
    static DT: OnceLock<DriverTrait> = OnceLock::new();
    DT.get_or_init(|| {
        let base = vec![1.0; 7];
        let pmf = normalize_pmf(&DiscreteConvolveSelf::new(4).transform(base.as_slice()));
        let mean = mean_from_pmf(&pmf);
        let std = variance_from_pmf(&pmf).sqrt();
        DriverTrait { pmf, mean, std }
    })
}

// =============================================================================
// Params + enums.
// =============================================================================

#[derive(Clone, Debug)]
pub struct SmartTrafficParams {
    pub base: TrafficParams,
    pub smart_car_pool_size: Option<usize>,
    pub actor_shuffle_seed: Option<f64>,
    pub accident_risk_scale: Option<f64>,
    pub accident_probability: Option<f64>,
    pub accident_accel_boost_mps2: Option<f64>,
    pub accident_fault_duration_sec: Option<f64>,
    pub distance_preference_spread: Option<f64>,
    pub start_preference_spread: Option<f64>,
    pub accident_flash_seconds: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmartTrafficFaultMode {
    AccelerateTooFast,
    BrakeTooSlow,
    Speeding,
}

impl SmartTrafficFaultMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SmartTrafficFaultMode::AccelerateTooFast => "accelerate-too-fast",
            SmartTrafficFaultMode::BrakeTooSlow => "brake-too-slow",
            SmartTrafficFaultMode::Speeding => "speeding",
        }
    }
}

// =============================================================================
// Output structs.
// =============================================================================

#[derive(Clone, Debug)]
pub struct SmartTrafficCarSnapshot {
    pub id: u64,
    pub actor_id: String,
    pub slot_index: usize,
    pub lane_id: String,
    pub position_m: f64,
    pub speed_mps: f64,
    pub acceleration_mps2: f64,
    pub jerk_mps3: f64,
    pub target_acceleration_mps2: f64,
    pub route: Vec<String>,
    pub route_index: usize,
    pub destination_sink_id: String,
    pub created_at_sec: f64,
    pub wait_sec: f64,
    pub distance_preference: f64,
    pub start_preference: f64,
    pub start_ready_since_sec: Option<f64>,
    pub grid_cell_ids: Vec<String>,
    pub grid_cell_count: usize,
    pub leader_id: Option<u64>,
    pub leader_gap_m: Option<f64>,
    pub run_count: u64,
    pub accident_count: usize,
    pub fault_mode: Option<SmartTrafficFaultMode>,
    pub fault_until_sec: Option<f64>,
    pub last_run_tick: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct SmartTrafficAccident {
    pub tick: usize,
    pub time_sec: f64,
    pub lane_id: String,
    pub position_m: f64,
    pub cell_id: String,
    pub car_id: u64,
    pub actor_id: String,
    pub other_car_id: u64,
    pub other_actor_id: String,
    pub speed_mps: f64,
    pub fault_mode: SmartTrafficFaultMode,
    pub risk_score: f64,
    pub hazard_per_sec: f64,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct SmartTrafficTraceRow {
    pub tick: usize,
    pub time_sec: f64,
    pub active_cars: usize,
    pub scheduled_smart_cars: usize,
    pub smart_movable_runs: usize,
    pub entered: usize,
    pub exited: usize,
    pub crashed: usize,
    pub mean_speed_mps: f64,
    pub mean_travel_time_sec: f64,
    pub queue_length: usize,
    pub lane_occupancy: HashMap<String, usize>,
    pub active_grid_cells: usize,
    pub signal_phases: HashMap<String, String>,
    pub actor_run_order: Vec<String>,
    pub accidents: Vec<SmartTrafficAccident>,
    pub cars: Vec<SmartTrafficCarSnapshot>,
}

#[derive(Clone, Debug)]
pub struct SmartTrafficCellStats {
    pub cell_size_m: f64,
    pub lane_width_m: f64,
    pub car_width_m: f64,
    pub active_cells: usize,
    pub created_cell_stations: usize,
    pub accident_cell_stations: usize,
    pub accident_cell_hits: usize,
    pub max_cell_occupancy: usize,
}

#[derive(Clone, Debug)]
pub struct SmartTrafficExecutionStats {
    pub participant_count: usize,
    pub smart_movable_count: usize,
    pub world_station_id: String,
    pub shuffled_by_runner: bool,
    pub actor_shuffle_seed: f64,
    pub total_smart_movable_runs: u64,
    pub max_smart_movable_runs_per_tick: usize,
}

#[derive(Clone, Debug)]
pub struct SmartTrafficResult {
    pub params: SmartTrafficParams,
    pub network: TrafficNetwork,
    pub trace: Vec<SmartTrafficTraceRow>,
    pub final_cars: Vec<SmartTrafficCarSnapshot>,
    pub entered: usize,
    pub exited: usize,
    pub crashed: usize,
    pub dropped: usize,
    pub mean_travel_time_sec: f64,
    pub mean_speed_mps: f64,
    pub max_active_cars: usize,
    pub cell_stats: SmartTrafficCellStats,
    pub execution: SmartTrafficExecutionStats,
    pub run_summary: IterativeRunSummary,
    pub accidents: Vec<SmartTrafficAccident>,
    pub validation: Vec<ValidationCheck>,
}

#[derive(Clone, Debug)]
struct KinematicSample {
    time_sec: f64,
    lane_id: String,
    position_m: f64,
    speed_mps: f64,
    acceleration_mps2: f64,
}

#[derive(Clone, Debug)]
struct SmartTrafficCellBounds {
    lane_id: String,
    longitudinal_index: i64,
    lateral_index: i64,
    x0_m: f64,
    x1_m: f64,
    y0_m: f64,
    y1_m: f64,
}

struct SmartCarProposal {
    slot_index: usize,
    car_id: u64,
    speed_mps: f64,
    position_m: f64,
    acceleration_mps2: f64,
    jerk_mps3: f64,
    target_acceleration_mps2: f64,
    leader_id: Option<u64>,
    leader_gap_m: Option<f64>,
    control_fault: bool,
    fault_mode: Option<SmartTrafficFaultMode>,
    fault_until_sec: Option<f64>,
    start_ready_since_sec: Option<f64>,
    risk_score: f64,
    hazard_per_sec: f64,
}

// =============================================================================
// Cell station (plain helper struct).
// =============================================================================

pub struct SmartTrafficCellStation {
    bounds: SmartTrafficCellBounds,
    car_ids: HashSet<u64>,
    accident_ids: Vec<String>,
}

impl SmartTrafficCellStation {
    fn new(bounds: SmartTrafficCellBounds) -> Self {
        SmartTrafficCellStation {
            bounds,
            car_ids: HashSet::new(),
            accident_ids: Vec::new(),
        }
    }
    fn clear_occupancy(&mut self) {
        self.car_ids.clear();
    }
    fn occupy(&mut self, car_id: u64) {
        self.car_ids.insert(car_id);
    }
    fn record_accident(&mut self, accident: &SmartTrafficAccident) {
        self.accident_ids.push(format!(
            "{}:{}->{}",
            accident.tick, accident.car_id, accident.other_car_id
        ));
    }
}

// =============================================================================
// Smart car (movable + run-loop participant).
// =============================================================================

struct AssignOpts {
    car_id: u64,
    lane_id: String,
    route: Vec<String>,
    destination_sink_id: String,
    created_at_sec: f64,
    initial_speed_mps: f64,
    distance_preference: f64,
    start_preference: f64,
}

pub struct SmartTrafficCar {
    smart: SmartMovableCore,
    station: StationCore,
    world: Weak<RefCell<SmartTrafficWorldStation>>,
    pub slot_index: usize,
    pub car_id: u64,
    pub lane_id: String,
    pub position_m: f64,
    pub speed_mps: f64,
    pub acceleration_mps2: f64,
    pub jerk_mps3: f64,
    pub target_acceleration_mps2: f64,
    pub route: Vec<String>,
    pub route_index: usize,
    pub destination_sink_id: String,
    pub created_at_sec: f64,
    pub wait_sec: f64,
    pub distance_preference: f64,
    pub start_preference: f64,
    pub start_ready_since_sec: Option<f64>,
    pub grid_cell_ids: Vec<String>,
    pub grid_cell_count: usize,
    pub leader_id: Option<u64>,
    pub leader_gap_m: Option<f64>,
    history: Vec<KinematicSample>,
    accidents: Vec<SmartTrafficAccident>,
    pub fault_mode: Option<SmartTrafficFaultMode>,
    pub fault_until_sec: f64,
    pub run_count: u64,
    pub last_run_tick: Option<usize>,
}

impl SmartTrafficCar {
    fn new(slot_index: usize, world: Weak<RefCell<SmartTrafficWorldStation>>) -> Self {
        let id = format!("smart-car-{slot_index}");
        SmartTrafficCar {
            smart: SmartMovableCore::new(id.clone()),
            station: StationCore::new(id),
            world,
            slot_index,
            car_id: 0,
            lane_id: String::new(),
            position_m: 0.0,
            speed_mps: 0.0,
            acceleration_mps2: 0.0,
            jerk_mps3: 0.0,
            target_acceleration_mps2: 0.0,
            route: Vec::new(),
            route_index: 0,
            destination_sink_id: String::new(),
            created_at_sec: 0.0,
            wait_sec: 0.0,
            distance_preference: 1.0,
            start_preference: 1.0,
            start_ready_since_sec: None,
            grid_cell_ids: Vec::new(),
            grid_cell_count: 0,
            leader_id: None,
            leader_gap_m: None,
            history: Vec::new(),
            accidents: Vec::new(),
            fault_mode: None,
            fault_until_sec: 0.0,
            run_count: 0,
            last_run_tick: None,
        }
    }

    /// PORT NOTE: the TS `assign` ended by calling `world.recordHistory(this)`;
    /// that re-borrow of the just-mutated car is impossible here, so the caller
    /// (`try_spawn_from_source`) records the initial history right after.
    fn assign(&mut self, opts: AssignOpts) {
        self.activate();
        self.car_id = opts.car_id;
        self.lane_id = opts.lane_id;
        self.position_m = 0.0;
        self.speed_mps = opts.initial_speed_mps;
        self.acceleration_mps2 = 0.0;
        self.jerk_mps3 = 0.0;
        self.target_acceleration_mps2 = 0.0;
        self.route = opts.route;
        self.route_index = 0;
        self.destination_sink_id = opts.destination_sink_id;
        self.created_at_sec = opts.created_at_sec;
        self.wait_sec = 0.0;
        self.distance_preference = opts.distance_preference;
        self.start_preference = opts.start_preference;
        self.start_ready_since_sec = None;
        self.grid_cell_ids = Vec::new();
        self.grid_cell_count = 0;
        self.leader_id = None;
        self.leader_gap_m = None;
        self.history = Vec::new();
        self.accidents = Vec::new();
        self.fault_mode = None;
        self.fault_until_sec = 0.0;
        self.run_count = 0;
        self.last_run_tick = None;
    }

    fn retire(&mut self) {
        self.deactivate();
        self.lane_id = String::new();
        self.route = Vec::new();
        self.grid_cell_ids = Vec::new();
        self.grid_cell_count = 0;
        self.distance_preference = 1.0;
        self.start_preference = 1.0;
        self.start_ready_since_sec = None;
        self.leader_id = None;
        self.leader_gap_m = None;
        self.fault_mode = None;
        self.fault_until_sec = 0.0;
    }

    fn snapshot(&self) -> SmartTrafficCarSnapshot {
        SmartTrafficCarSnapshot {
            id: self.car_id,
            actor_id: self.smart.id.clone(),
            slot_index: self.slot_index,
            lane_id: self.lane_id.clone(),
            position_m: self.position_m,
            speed_mps: self.speed_mps,
            acceleration_mps2: self.acceleration_mps2,
            jerk_mps3: self.jerk_mps3,
            target_acceleration_mps2: self.target_acceleration_mps2,
            route: self.route.clone(),
            route_index: self.route_index,
            destination_sink_id: self.destination_sink_id.clone(),
            created_at_sec: self.created_at_sec,
            wait_sec: self.wait_sec,
            distance_preference: self.distance_preference,
            start_preference: self.start_preference,
            start_ready_since_sec: self.start_ready_since_sec,
            grid_cell_ids: self.grid_cell_ids.clone(),
            grid_cell_count: self.grid_cell_count,
            leader_id: self.leader_id,
            leader_gap_m: self.leader_gap_m,
            run_count: self.run_count,
            accident_count: self.accidents.len(),
            fault_mode: self.fault_mode,
            fault_until_sec: if self.fault_until_sec > 0.0 {
                Some(self.fault_until_sec)
            } else {
                None
            },
            last_run_tick: self.last_run_tick,
        }
    }
}

impl SmartMovable for SmartTrafficCar {
    fn core(&self) -> &SmartMovableCore {
        &self.smart
    }
    fn core_mut(&mut self) -> &mut SmartMovableCore {
        &mut self.smart
    }
    fn run_time_step(&mut self) {
        DESStation::run_time_step(self);
    }
}

impl DESStation for SmartTrafficCar {
    fn core(&self) -> &StationCore {
        &self.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        let world = match self.world.upgrade() {
            Some(w) => w,
            None => return false,
        };
        let (tick, accepts) = {
            let w = world.borrow();
            (w.tick, w.accepts_smart_movable_runs())
        };
        self.is_active() && accepts && self.last_run_tick != Some(tick)
    }
    fn run_time_step(&mut self) {
        if !DESStation::has_work(self) {
            return;
        }
        let world = self.world.upgrade().expect("world alive during run");
        let tick = world.borrow().tick;
        self.last_run_tick = Some(tick);
        self.run_count += 1;
        let mut w = world.borrow_mut();
        w.record_actor_run(&self.smart.id);
        w.propose_car_step(self);
    }
}

// =============================================================================
// World station.
// =============================================================================

struct SpatialIndex {
    by_cell: HashMap<String, HashSet<u64>>,
    active_cell_count: usize,
}

#[derive(Clone)]
struct CarView {
    slot: usize,
    car_id: u64,
    lane_id: String,
    position_m: f64,
    speed_mps: f64,
}

pub struct SmartTrafficWorldStation {
    core: StationCore,
    params: SmartTrafficParams,
    logger: Option<Rc<dyn OptimizationLogger>>,
    car_actors: Vec<Rc<RefCell<SmartTrafficCar>>>,
    network: TrafficNetwork,
    nodes: HashMap<String, TrafficNode>,
    lanes: HashMap<String, TrafficLane>,
    signal_by_node: HashMap<String, TrafficSignal>,
    routes: HashMap<String, Vec<String>>,
    source_accumulators: HashMap<String, f64>,
    scheduled_trips: Vec<TrafficScheduledTrip>,
    cell_stations: HashMap<String, SmartTrafficCellStation>,
    active_cell_ids: HashSet<String>,
    rng: SeededRandom,
    spatial: SpatialIndex,
    proposals: HashMap<u64, SmartCarProposal>,
    current_actor_run_order: Vec<String>,
    completed_travel_times: Vec<f64>,
    accidents: Vec<SmartTrafficAccident>,
    accidents_this_tick: Vec<SmartTrafficAccident>,
    next_scheduled_trip_index: usize,
    next_car_id: u64,
    tick: usize,
    time_sec: f64,
    entered: usize,
    exited: usize,
    crashed: usize,
    dropped: usize,
    speed_sum: f64,
    speed_samples: u64,
    max_active_cars: usize,
    max_cell_occupancy: usize,
    scheduled_smart_cars: usize,
    total_smart_movable_runs: u64,
    max_smart_movable_runs_per_tick: usize,
    trace: Vec<SmartTrafficTraceRow>,
}

/// Construct a world + its smart-car pool, wired into the cyclic graph.
pub fn build_world(
    params: SmartTrafficParams,
    logger: Option<Rc<dyn OptimizationLogger>>,
) -> Rc<RefCell<SmartTrafficWorldStation>> {
    let world = Rc::new(RefCell::new(SmartTrafficWorldStation::new_inner(
        params, logger,
    )));
    let pool_size = {
        let w = world.borrow();
        w.params
            .smart_car_pool_size
            .unwrap_or(w.params.base.max_cars)
    };
    let cars: Vec<Rc<RefCell<SmartTrafficCar>>> = (0..pool_size)
        .map(|i| Rc::new(RefCell::new(SmartTrafficCar::new(i, Rc::downgrade(&world)))))
        .collect();
    world.borrow_mut().car_actors = cars;
    world
}

impl SmartTrafficWorldStation {
    fn new_inner(params: SmartTrafficParams, logger: Option<Rc<dyn OptimizationLogger>>) -> Self {
        let network = params
            .base
            .network
            .clone()
            .unwrap_or_else(build_five_intersection_traffic_network);
        let rng = mulberry32(params.base.seed as u32);
        let mut scheduled_trips = params.base.scheduled_trips.clone().unwrap_or_default();
        scheduled_trips.sort_by(|a, b| a.depart_sec.partial_cmp(&b.depart_sec).unwrap());

        let mut nodes: HashMap<String, TrafficNode> = HashMap::new();
        for node in &network.nodes {
            nodes.insert(node.id.clone(), node.clone());
        }
        let mut lanes: HashMap<String, TrafficLane> = HashMap::new();
        for lane in &network.lanes {
            lanes.insert(lane.id.clone(), lane.clone());
        }
        let mut signal_by_node: HashMap<String, TrafficSignal> = HashMap::new();
        for signal in network.signals.iter().flatten() {
            signal_by_node.insert(signal.node_id.clone(), signal.clone());
        }
        let mut source_accumulators: HashMap<String, f64> = HashMap::new();
        for source in &network.sources {
            source_accumulators.insert(source.id.clone(), 0.0);
        }

        let mut station = SmartTrafficWorldStation {
            core: StationCore::new("smart-traffic-world"),
            params,
            logger,
            car_actors: Vec::new(),
            network,
            nodes,
            lanes,
            signal_by_node,
            routes: HashMap::new(),
            source_accumulators,
            scheduled_trips,
            cell_stations: HashMap::new(),
            active_cell_ids: HashSet::new(),
            rng,
            spatial: SpatialIndex {
                by_cell: HashMap::new(),
                active_cell_count: 0,
            },
            proposals: HashMap::new(),
            current_actor_run_order: Vec::new(),
            completed_travel_times: Vec::new(),
            accidents: Vec::new(),
            accidents_this_tick: Vec::new(),
            next_scheduled_trip_index: 0,
            next_car_id: 1,
            tick: 0,
            time_sec: 0.0,
            entered: 0,
            exited: 0,
            crashed: 0,
            dropped: 0,
            speed_sum: 0.0,
            speed_samples: 0,
            max_active_cars: 0,
            max_cell_occupancy: 0,
            scheduled_smart_cars: 0,
            total_smart_movable_runs: 0,
            max_smart_movable_runs_per_tick: 0,
            trace: Vec::new(),
        };
        station.precompute_routes();
        station.register_validators();
        station
    }

    fn register_validators(&mut self) {
        let v1 = intrinsic_check::<dyn DESStation>(
            "smart-traffic-active-under-cap",
            |st| {
                let s = downcast(st);
                s.max_active_cars <= s.params.base.max_cars && s.max_active_cars < 300
            },
            Some("active smart cars never exceed maxCars or 299".to_string()),
            Some(Box::new(|st: &dyn DESStation| {
                let s = downcast(st);
                format!(
                    "maxActive={} cap={}",
                    s.max_active_cars, s.params.base.max_cars
                )
            })),
            Some("smart-traffic-flow".to_string()),
            None,
        )
        .boxed();
        self.add_validator(v1);

        let v2 = intrinsic_check::<dyn DESStation>(
            "smart-traffic-conservation",
            |st| {
                let s = downcast(st);
                s.entered == s.exited + s.crashed + s.active_car_count()
            },
            Some("entered = exited + crashed + active".to_string()),
            Some(Box::new(|st: &dyn DESStation| {
                let s = downcast(st);
                format!(
                    "entered={} exited={} crashed={} active={}",
                    s.entered,
                    s.exited,
                    s.crashed,
                    s.active_car_count()
                )
            })),
            Some("smart-traffic-flow".to_string()),
            None,
        )
        .boxed();
        self.add_validator(v2);

        let v3 = intrinsic_check::<dyn DESStation>(
            "smart-traffic-no-collisions",
            |st| downcast(st).minimum_body_gap() >= -1e-7,
            Some("same-lane smart cars do not physically overlap".to_string()),
            Some(Box::new(|st: &dyn DESStation| {
                format!("{:.6}", downcast(st).minimum_body_gap())
            })),
            Some("smart-traffic-flow".to_string()),
            None,
        )
        .boxed();
        self.add_validator(v3);

        let v4 = intrinsic_check::<dyn DESStation>(
            "smart-traffic-actor-run-coverage",
            |st| {
                downcast(st)
                    .trace
                    .iter()
                    .all(|row| row.scheduled_smart_cars == row.smart_movable_runs)
            },
            Some("every active smart movable receives runTimeStep once per tick".to_string()),
            Some(Box::new(|st: &dyn DESStation| {
                let s = downcast(st);
                match s
                    .trace
                    .iter()
                    .find(|row| row.scheduled_smart_cars != row.smart_movable_runs)
                {
                    Some(bad) => format!(
                        "tick={} scheduled={} ran={}",
                        bad.tick, bad.scheduled_smart_cars, bad.smart_movable_runs
                    ),
                    None => "all active smart cars ran".to_string(),
                }
            })),
            Some("smart-traffic-flow".to_string()),
            None,
        )
        .boxed();
        self.add_validator(v4);
    }

    pub fn begin_run(&mut self) {
        self.prepare_tick();
    }

    pub fn current_tick(&self) -> usize {
        self.tick
    }

    fn accepts_smart_movable_runs(&self) -> bool {
        self.time_sec < self.params.base.duration_sec - 1e-9
    }

    fn record_actor_run(&mut self, actor_id: &str) {
        self.current_actor_run_order.push(actor_id.to_string());
    }

    fn propose_car_step(&mut self, car: &SmartTrafficCar) {
        let lane = self.lane(&car.lane_id).clone();
        let leader = self
            .find_leader_ahead_from_grid(car, &lane)
            .or_else(|| self.sorted_leader_ahead(car));
        let proposal = self.next_kinematics(car, &lane, leader);
        self.proposals.insert(car.car_id, proposal);
    }

    pub fn finish_tick(&mut self) {
        let dt = self.params.base.dt_sec;
        let time_after = self.time_sec + dt;

        // Commit proposals in carId order.
        let mut proposal_ids: Vec<u64> = self.proposals.keys().copied().collect();
        proposal_ids.sort_unstable();
        for cid in &proposal_ids {
            let (slot, sp, pos, acc, jerk, tacc, lid, lgap, cfault, fmode, funtil, ready) = {
                let p = &self.proposals[cid];
                (
                    p.slot_index,
                    p.speed_mps,
                    p.position_m,
                    p.acceleration_mps2,
                    p.jerk_mps3,
                    p.target_acceleration_mps2,
                    p.leader_id,
                    p.leader_gap_m,
                    p.control_fault,
                    p.fault_mode,
                    p.fault_until_sec,
                    p.start_ready_since_sec,
                )
            };
            let mut car = self.car_actors[slot].borrow_mut();
            if !car.is_active() {
                continue;
            }
            car.speed_mps = sp;
            car.position_m = pos;
            car.acceleration_mps2 = acc;
            car.jerk_mps3 = jerk;
            car.target_acceleration_mps2 = tacc;
            car.leader_id = lid;
            car.leader_gap_m = lgap;
            car.start_ready_since_sec = ready;
            if cfault {
                if let Some(fm) = fmode {
                    car.fault_mode = Some(fm);
                    car.fault_until_sec = car.fault_until_sec.max(funtil.unwrap_or(time_after));
                }
            }
            if car.fault_until_sec <= time_after {
                car.fault_mode = None;
                car.fault_until_sec = 0.0;
            }
            if car.speed_mps < 0.5 {
                car.wait_sec += dt;
            } else {
                car.start_ready_since_sec = None;
            }
        }

        self.detect_accidents(time_after);

        let active_slots: Vec<usize> = self.active_car_views().iter().map(|v| v.slot).collect();
        for slot in active_slots {
            self.handle_lane_end(slot, time_after);
            if self.car_actors[slot].borrow().is_active() {
                self.record_history_at(slot, time_after);
            }
        }

        self.spatial = self.rebuild_spatial_index();
        for v in self.active_car_views() {
            self.speed_sum += v.speed_mps;
            self.speed_samples += 1;
        }
        self.max_active_cars = self.max_active_cars.max(self.active_car_count());
        self.total_smart_movable_runs += self.current_actor_run_order.len() as u64;
        self.max_smart_movable_runs_per_tick = self
            .max_smart_movable_runs_per_tick
            .max(self.current_actor_run_order.len());
        let row = self.snapshot_row(time_after);
        self.trace.push(row);
        self.tick += 1;
        self.time_sec = time_after;
        if self.accepts_smart_movable_runs() {
            self.prepare_tick();
        }
    }

    pub fn build_result(
        &self,
        summary: IterativeRunSummary,
        validation: Vec<ValidationCheck>,
    ) -> SmartTrafficResult {
        let mut travel_times = self.completed_travel_times.clone();
        travel_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_travel_time_sec = if travel_times.is_empty() {
            0.0
        } else {
            travel_times.iter().sum::<f64>() / travel_times.len() as f64
        };
        let accident_cell_stations = self
            .cell_stations
            .values()
            .filter(|c| !c.accident_ids.is_empty())
            .count();
        let accident_cell_hits = self
            .cell_stations
            .values()
            .map(|c| c.accident_ids.len())
            .sum();
        SmartTrafficResult {
            params: self.params.clone(),
            network: self.network.clone(),
            trace: self.trace.clone(),
            final_cars: self.snap_cars(),
            entered: self.entered,
            exited: self.exited,
            crashed: self.crashed,
            dropped: self.dropped,
            mean_travel_time_sec,
            mean_speed_mps: if self.speed_samples > 0 {
                self.speed_sum / self.speed_samples as f64
            } else {
                0.0
            },
            max_active_cars: self.max_active_cars,
            cell_stats: SmartTrafficCellStats {
                cell_size_m: self.grid_cell_size_m(),
                lane_width_m: self.lane_width_m(),
                car_width_m: self.car_width_m(),
                active_cells: self.active_cell_ids.len(),
                created_cell_stations: self.cell_stations.len(),
                accident_cell_stations,
                accident_cell_hits,
                max_cell_occupancy: self.max_cell_occupancy,
            },
            execution: SmartTrafficExecutionStats {
                participant_count: self.car_actors.len() + 1,
                smart_movable_count: self.car_actors.len(),
                world_station_id: self.core.id.clone(),
                shuffled_by_runner: true,
                actor_shuffle_seed: self
                    .params
                    .actor_shuffle_seed
                    .unwrap_or(self.params.base.seed + 1009.0),
                total_smart_movable_runs: self.total_smart_movable_runs,
                max_smart_movable_runs_per_tick: self.max_smart_movable_runs_per_tick,
            },
            run_summary: summary,
            accidents: self.accidents.clone(),
            validation,
        }
    }

    fn record_history_at(&mut self, slot: usize, time_sec: f64) {
        let horizon = self.reaction_time_sec() + 2.0 * self.params.base.dt_sec + 1.0;
        let mut car = self.car_actors[slot].borrow_mut();
        let sample = KinematicSample {
            time_sec,
            lane_id: car.lane_id.clone(),
            position_m: car.position_m,
            speed_mps: car.speed_mps,
            acceleration_mps2: car.acceleration_mps2,
        };
        car.history.push(sample);
        while car.history.len() > 2 && car.history[1].time_sec < time_sec - horizon {
            car.history.remove(0);
        }
    }

    fn prepare_tick(&mut self) {
        self.proposals.clear();
        self.current_actor_run_order.clear();
        self.accidents_this_tick = Vec::new();
        if self.time_sec < self.params.base.duration_sec - 1e-9 {
            self.spawn_cars();
        }
        self.spatial = self.rebuild_spatial_index();
        self.scheduled_smart_cars = self.active_car_count();
    }

    fn spawn_cars(&mut self) {
        if !self.scheduled_trips.is_empty() {
            self.spawn_scheduled_trips();
            return;
        }
        let dt = self.params.base.dt_sec;
        let mult = self.params.base.spawn_rate_multiplier.unwrap_or(1.0);
        let sources = self.network.sources.clone();
        for source in &sources {
            let expected = source.rate_per_min * mult * dt / 60.0;
            let mut acc = self
                .source_accumulators
                .get(&source.id)
                .copied()
                .unwrap_or(0.0)
                + expected;
            let count = acc.floor() as i64;
            acc -= count as f64;
            self.source_accumulators.insert(source.id.clone(), acc);
            for _ in 0..count {
                self.try_spawn_from_source(source, None);
            }
        }
    }

    fn spawn_scheduled_trips(&mut self) {
        while self.next_scheduled_trip_index < self.scheduled_trips.len() {
            let trip = self.scheduled_trips[self.next_scheduled_trip_index].clone();
            if trip.depart_sec > self.time_sec + 1e-9 {
                return;
            }
            self.next_scheduled_trip_index += 1;
            let source = self
                .network
                .sources
                .iter()
                .find(|s| s.id == trip.source_id)
                .cloned();
            match source {
                None => {
                    self.dropped += 1;
                }
                Some(src) => {
                    self.try_spawn_from_source(&src, Some(trip.destination_sink_id.clone()));
                }
            }
        }
    }

    fn try_spawn_from_source(
        &mut self,
        source: &TrafficSource,
        destination_sink_id: Option<String>,
    ) {
        if self.active_car_count() >= self.params.base.max_cars {
            self.dropped += 1;
            return;
        }
        let actor_slot = match self.find_inactive_actor() {
            Some(s) => s,
            None => {
                self.dropped += 1;
                return;
            }
        };
        let sink_ids: Vec<String> = source
            .destination_sink_ids
            .clone()
            .unwrap_or_else(|| self.network.sinks.iter().map(|s| s.id.clone()).collect());
        let sink_id = match destination_sink_id {
            Some(id) => id,
            None => {
                if sink_ids.is_empty() {
                    self.dropped += 1;
                    return;
                }
                let idx = ((self.rng.next_float() * sink_ids.len() as f64).floor() as usize)
                    .min(sink_ids.len() - 1);
                sink_ids[idx].clone()
            }
        };
        if !sink_ids.contains(&sink_id) {
            self.dropped += 1;
            return;
        }
        let route = match self.routes.get(&format!("{}->{}", source.id, sink_id)) {
            Some(r) if !r.is_empty() => r.clone(),
            _ => {
                self.dropped += 1;
                return;
            }
        };
        let lane = self.lane(&route[0]).clone();
        if !self.can_enter_lane(&lane.id, None) {
            self.dropped += 1;
            return;
        }
        let car_id = self.next_car_id;
        self.next_car_id += 1;
        let initial_speed = 2.0_f64.min(lane.speed_limit_mps);
        let distance_preference = self.sample_distance_preference();
        let start_preference = self.sample_start_preference();
        let created_at = self.time_sec;
        {
            let mut car = self.car_actors[actor_slot].borrow_mut();
            car.assign(AssignOpts {
                car_id,
                lane_id: lane.id.clone(),
                route,
                destination_sink_id: sink_id,
                created_at_sec: created_at,
                initial_speed_mps: initial_speed,
                distance_preference,
                start_preference,
            });
        }
        self.record_history_at(actor_slot, created_at);
        self.entered += 1;
    }

    fn precompute_routes(&mut self) {
        let sources = self.network.sources.clone();
        for source in &sources {
            let sinks: Vec<String> = source
                .destination_sink_ids
                .clone()
                .unwrap_or_else(|| self.network.sinks.iter().map(|s| s.id.clone()).collect());
            for sink_id in &sinks {
                let sink = self
                    .network
                    .sinks
                    .iter()
                    .find(|s| &s.id == sink_id)
                    .cloned();
                let sink = match sink {
                    Some(s) => s,
                    None => continue,
                };
                let route = shortest_lane_path(&self.network, &source.node_id, &sink.node_id);
                if !route.is_empty() {
                    self.routes
                        .insert(format!("{}->{}", source.id, sink_id), route);
                }
            }
        }
    }

    // ── kinematics ─────────────────────────────────────────────────────────

    fn next_kinematics(
        &mut self,
        car: &SmartTrafficCar,
        lane: &TrafficLane,
        leader_slot: Option<usize>,
    ) -> SmartCarProposal {
        let dt = self.params.base.dt_sec;
        let vehicle_space = self.vehicle_space();
        let car_length = self.car_length_m();
        let barrier = self.stop_barrier_position(car, lane);
        // Copy leader state (incl. bounded history) so no Ref is held later.
        let leader: Option<LeaderView> = leader_slot.map(|s| {
            let l = self.car_actors[s].borrow();
            LeaderView {
                car_id: l.car_id,
                lane_id: l.lane_id.clone(),
                position_m: l.position_m,
                speed_mps: l.speed_mps,
                history: l.history.clone(),
            }
        });
        let leader_position = leader
            .as_ref()
            .map(|l| l.position_m)
            .unwrap_or(f64::INFINITY);
        let current_leader_gap = if leader_position.is_finite() {
            leader_position - car.position_m - vehicle_space
        } else {
            f64::INFINITY
        };
        let physical_leader_gap = if leader_position.is_finite() {
            leader_position - car.position_m - car_length
        } else {
            f64::INFINITY
        };
        let barrier_gap = match barrier {
            None => f64::INFINITY,
            Some(b) => b - car.position_m - vehicle_space,
        };
        let use_barrier = barrier_gap <= current_leader_gap;
        let delayed_leader = leader
            .as_ref()
            .map(|l| perceived_sample(l, self.time_sec - self.reaction_time_sec(), car));
        let perceived_leader = match (&delayed_leader, &leader) {
            (Some(dl), _) if dl.lane_id == car.lane_id => Some((dl.position_m, dl.speed_mps)),
            (_, Some(l)) => Some((l.position_m, l.speed_mps)),
            _ => None,
        };
        // perceived = {positionM, speedMps, id}
        let (perceived_position, perceived_speed, perceived_id): (f64, f64, Option<u64>) =
            if use_barrier {
                (barrier.unwrap_or(f64::INFINITY), 0.0, None)
            } else if let Some((pp, ps)) = perceived_leader {
                (pp, ps, leader.as_ref().map(|l| l.car_id))
            } else {
                (f64::INFINITY, lane.speed_limit_mps, None)
            };
        let perceived_gap = (perceived_position - car.position_m - vehicle_space).max(0.05);
        let max_accel = self.max_accel_mps2();
        let max_decel = self.max_decel_mps2();
        let v = car.speed_mps.max(0.0);
        let v0 = lane.speed_limit_mps;
        let distance_preference = car.distance_preference;
        let preferred_vehicle_space = self.car_length_m() + self.min_gap_m() * distance_preference;
        let time_headway =
            (self.time_headway_sec() + self.reaction_time_sec()) * distance_preference;
        let closing_term =
            (v * (v - perceived_speed) / (2.0 * (max_accel * max_decel).sqrt())).max(0.0);
        let desired_gap = preferred_vehicle_space + v * time_headway + closing_term;
        let free_road = 1.0 - (v / v0.max(1e-9)).min(2.0).powi(2);
        let interaction = if perceived_position.is_finite() {
            (desired_gap / perceived_gap).powi(2)
        } else {
            0.0
        };
        let mut target_acceleration =
            clamp(max_accel * (free_road - interaction), -max_decel, max_accel);
        let mut start_ready_since_sec = car.start_ready_since_sec;
        if v < 0.5 && target_acceleration > 0.0 {
            let has_clearance = self.has_startup_clearance(
                car,
                perceived_position,
                perceived_id,
                physical_leader_gap,
            );
            if has_clearance {
                start_ready_since_sec = start_ready_since_sec.or(Some(self.time_sec));
            } else {
                start_ready_since_sec = None;
            }
            let ready_for = match start_ready_since_sec {
                None => 0.0,
                Some(t) => (self.time_sec - t).max(0.0),
            };
            if !has_clearance || ready_for < self.start_delay_sec(car.start_preference) {
                target_acceleration = target_acceleration.min(0.0);
            }
        } else if v >= 0.5 || target_acceleration <= 0.0 {
            start_ready_since_sec = None;
        }
        let max_jerk_step = self.max_jerk_mps3() * dt;
        let closing_speed = if perceived_position.is_finite() {
            (v - perceived_speed).max(0.0)
        } else {
            0.0
        };
        let time_to_contact = if closing_speed > 1e-9 && physical_leader_gap.is_finite() {
            physical_leader_gap / closing_speed
        } else {
            f64::INFINITY
        };
        let speed_risk = clamp((v / v0.max(1e-9) - 1.0) / 0.35, 0.0, 1.0);
        let close_risk = if physical_leader_gap.is_finite() {
            clamp(
                (desired_gap / physical_leader_gap.max(0.05) - 1.0) / 4.0,
                0.0,
                1.0,
            )
        } else {
            0.0
        };
        let ttc_risk = if time_to_contact.is_finite() {
            clamp((3.0 - time_to_contact) / 3.0, 0.0, 1.0)
        } else {
            0.0
        };
        let braking_risk = clamp(
            (-target_acceleration / max_decel.max(1e-9) - 0.35) / 0.65,
            0.0,
            1.0,
        );
        let accel_risk = clamp(
            (target_acceleration / max_accel.max(1e-9) - 0.7) / 0.3,
            0.0,
            1.0,
        );
        let risk_score = clamp(
            0.45 * ttc_risk
                + 0.25 * close_risk
                + 0.2 * braking_risk
                + 0.15 * speed_risk
                + 0.1 * accel_risk,
            0.0,
            1.0,
        );
        let hazard_per_sec = self.accident_risk_scale() * risk_score;
        let active_fault = car.fault_until_sec > self.time_sec;
        let starts_fault = !active_fault
            && leader.is_some()
            && risk_score > 0.0
            && self.rng.next_float() < 1.0 - (-hazard_per_sec * dt).exp();
        let control_fault = active_fault || starts_fault;
        let fault_mode: Option<SmartTrafficFaultMode> = if active_fault {
            car.fault_mode
        } else if starts_fault {
            Some(fault_mode_for_risk(
                speed_risk,
                braking_risk,
                accel_risk,
                ttc_risk,
            ))
        } else {
            None
        };
        let fault_until_sec = if starts_fault {
            self.time_sec + self.accident_fault_duration_sec()
        } else {
            car.fault_until_sec
        };
        if control_fault {
            if fault_mode == Some(SmartTrafficFaultMode::BrakeTooSlow) {
                target_acceleration = target_acceleration.max(-max_decel * 0.12);
            } else {
                target_acceleration =
                    max_accel + self.accident_accel_boost_mps2() * risk_score.max(0.25);
            }
        }
        let mut acceleration = clamp(
            car.acceleration_mps2
                + clamp(
                    target_acceleration - car.acceleration_mps2,
                    -max_jerk_step,
                    max_jerk_step,
                ),
            -max_decel,
            max_accel,
        );
        if control_fault && fault_mode == Some(SmartTrafficFaultMode::BrakeTooSlow) {
            acceleration = acceleration.max(-max_decel * 0.12);
        } else if control_fault {
            acceleration = acceleration
                .max(max_accel + self.accident_accel_boost_mps2() * risk_score.max(0.25));
        }
        let mut speed = if control_fault {
            clamp(
                v + acceleration * dt,
                0.0,
                v0 * if fault_mode == Some(SmartTrafficFaultMode::Speeding) {
                    1.6
                } else {
                    1.3
                },
            )
        } else {
            clamp(v + acceleration * dt, 0.0, v0)
        };
        let mut position = car.position_m + v * dt + 0.5 * acceleration * dt * dt;
        let hard_limit = self.hard_position_limit(car, lane, leader.as_ref(), barrier);
        if !control_fault && position > hard_limit {
            position = car.position_m.max(hard_limit);
            speed = speed.min((position - car.position_m) / dt).max(0.0);
        }
        acceleration = clamp((speed - v) / dt, -max_decel, max_accel);
        if control_fault {
            acceleration = (speed - v) / dt;
        }
        let jerk = if control_fault {
            (acceleration - car.acceleration_mps2) / dt
        } else {
            clamp(
                (acceleration - car.acceleration_mps2) / dt,
                -self.max_jerk_mps3(),
                self.max_jerk_mps3(),
            )
        };
        SmartCarProposal {
            slot_index: car.slot_index,
            car_id: car.car_id,
            speed_mps: speed,
            position_m: position,
            acceleration_mps2: acceleration,
            jerk_mps3: jerk,
            target_acceleration_mps2: target_acceleration,
            leader_id: perceived_id,
            leader_gap_m: if perceived_position.is_finite() {
                Some(perceived_gap.max(0.0))
            } else {
                None
            },
            control_fault,
            fault_mode,
            fault_until_sec: if fault_until_sec > self.time_sec {
                Some(fault_until_sec)
            } else {
                None
            },
            start_ready_since_sec,
            risk_score,
            hazard_per_sec,
        }
    }

    fn detect_accidents(&mut self, time_sec: f64) {
        let views = self.active_car_views();
        let car_length = self.car_length_m();
        let mut crashed_ids: HashSet<u64> = HashSet::new();
        struct Apply {
            accident: SmartTrafficAccident,
            car_slot: usize,
            leader_slot: usize,
        }
        let mut applies: Vec<Apply> = Vec::new();
        let lane_ids: Vec<String> = self.network.lanes.iter().map(|l| l.id.clone()).collect();
        for lane_id in &lane_ids {
            let mut cars: Vec<&CarView> = views
                .iter()
                .filter(|c| &c.lane_id == lane_id && !crashed_ids.contains(&c.car_id))
                .collect();
            cars.sort_by(|a, b| a.position_m.partial_cmp(&b.position_m).unwrap());
            for i in 1..cars.len() {
                let car = cars[i - 1];
                let leader = cars[i];
                if crashed_ids.contains(&car.car_id) {
                    continue;
                }
                let contact_position = leader.position_m - car_length;
                if car.position_m < contact_position {
                    continue;
                }
                let p = self.proposals.get(&car.car_id);
                let accident = SmartTrafficAccident {
                    tick: self.tick,
                    time_sec,
                    lane_id: car.lane_id.clone(),
                    position_m: contact_position.max(0.0),
                    cell_id: self.accident_cell_id(&car.lane_id, contact_position.max(0.0)),
                    car_id: car.car_id,
                    actor_id: format!("smart-car-{}", car.slot),
                    other_car_id: leader.car_id,
                    other_actor_id: format!("smart-car-{}", leader.slot),
                    speed_mps: p.map(|p| p.speed_mps).unwrap_or(car.speed_mps),
                    fault_mode: p
                        .and_then(|p| p.fault_mode)
                        .unwrap_or(SmartTrafficFaultMode::BrakeTooSlow),
                    risk_score: p.map(|p| p.risk_score).unwrap_or(0.0),
                    hazard_per_sec: p.map(|p| p.hazard_per_sec).unwrap_or(0.0),
                    reason: "body-contact-rear-end".to_string(),
                };
                crashed_ids.insert(car.car_id);
                applies.push(Apply {
                    accident,
                    car_slot: car.slot,
                    leader_slot: leader.slot,
                });
            }
        }
        for ap in applies {
            self.accidents.push(ap.accident.clone());
            self.accidents_this_tick.push(ap.accident.clone());
            self.crashed += 1;
            self.car_actors[ap.car_slot]
                .borrow_mut()
                .accidents
                .push(ap.accident.clone());
            self.car_actors[ap.leader_slot]
                .borrow_mut()
                .accidents
                .push(ap.accident.clone());
            self.ensure_cell_station(&ap.accident.cell_id)
                .record_accident(&ap.accident);
            self.car_actors[ap.car_slot].borrow_mut().retire();
            if let Some(logger) = &self.logger {
                // PORT NOTE: the TS event carries the full accident payload as an
                // open-ended object; here we emit the structured `kind` only and
                // leave the extra fields empty (logger output is a debug side-effect).
                logger.log(LogEvent {
                    kind: "smart-traffic-accident".to_string(),
                    ..Default::default()
                });
            }
        }
    }

    fn stop_barrier_position(&self, car: &SmartTrafficCar, lane: &TrafficLane) -> Option<f64> {
        let next_lane_id = car.route.get(car.route_index + 1).cloned();
        let next_lane_id = next_lane_id?;
        if self.signal_allows(&lane.id) && self.can_enter_lane(&next_lane_id, Some(car.car_id)) {
            return None;
        }
        Some(lane.length_m)
    }

    fn hard_position_limit(
        &self,
        car: &SmartTrafficCar,
        lane: &TrafficLane,
        leader: Option<&LeaderView>,
        barrier: Option<f64>,
    ) -> f64 {
        let mut limit = f64::INFINITY;
        if let Some(l) = leader {
            limit = limit.min(l.position_m - self.vehicle_space());
        }
        if let Some(b) = barrier {
            limit = limit.min(b - self.vehicle_space());
        }
        if !limit.is_finite() {
            return lane.length_m + (car.speed_mps * self.params.base.dt_sec).max(0.0);
        }
        car.position_m.max(limit)
    }

    fn handle_lane_end(&mut self, slot: usize, time_sec: f64) {
        let (
            mut lane_id,
            mut position_m,
            mut speed_mps,
            route,
            mut route_index,
            car_id,
            created_at,
            active,
        ) = {
            let c = self.car_actors[slot].borrow();
            (
                c.lane_id.clone(),
                c.position_m,
                c.speed_mps,
                c.route.clone(),
                c.route_index,
                c.car_id,
                c.created_at_sec,
                c.is_active(),
            )
        };
        if !active {
            return;
        }
        let mut lane = self.lane(&lane_id).clone();
        if position_m < lane.length_m - 1e-9 {
            return;
        }
        let mut overshoot = position_m - lane.length_m;
        let mut retire = false;
        loop {
            if position_m < lane.length_m - 1e-9 {
                break;
            }
            let next_lane_id = route.get(route_index + 1).cloned();
            match next_lane_id {
                None => {
                    self.exited += 1;
                    self.completed_travel_times.push(time_sec - created_at);
                    retire = true;
                    break;
                }
                Some(nid) => {
                    if !self.signal_allows(&lane_id) || !self.can_enter_lane(&nid, Some(car_id)) {
                        position_m = self.blocked_stop_position(&lane_id, car_id);
                        speed_mps = 0.0;
                        break;
                    }
                    route_index += 1;
                    lane_id = nid;
                    lane = self.lane(&lane_id).clone();
                    position_m = overshoot
                        .max(0.0)
                        .min((lane.length_m - self.vehicle_space()).max(0.0));
                    speed_mps = speed_mps.min(lane.speed_limit_mps);
                    overshoot = (position_m - lane.length_m).max(0.0);
                    if overshoot <= 1e-9 {
                        break;
                    }
                }
            }
        }
        let mut c = self.car_actors[slot].borrow_mut();
        if retire {
            c.retire();
        } else {
            c.lane_id = lane_id;
            c.position_m = position_m;
            c.speed_mps = speed_mps;
            c.route_index = route_index;
        }
    }

    fn blocked_stop_position(&self, lane_id: &str, car_id: u64) -> f64 {
        let lane = self.lane(lane_id);
        let vehicle_space = self.vehicle_space();
        let mut safe = (lane.length_m - vehicle_space).max(0.0);
        let mut others: Vec<CarView> = self
            .active_car_views()
            .into_iter()
            .filter(|c| c.car_id != car_id && c.lane_id == lane_id)
            .collect();
        others.sort_by(|a, b| b.position_m.partial_cmp(&a.position_m).unwrap());
        for other in &others {
            if other.position_m <= safe + vehicle_space {
                safe = safe.min(other.position_m - vehicle_space);
            }
        }
        safe.max(0.0)
    }

    fn rebuild_spatial_index(&mut self) -> SpatialIndex {
        for cell in self.cell_stations.values_mut() {
            cell.clear_occupancy();
        }
        self.active_cell_ids.clear();
        let mut by_cell: HashMap<String, HashSet<u64>> = HashMap::new();
        let views = self.active_car_views();
        for v in &views {
            let cell_ids = self.occupied_cell_ids_view(v);
            {
                let mut car = self.car_actors[v.slot].borrow_mut();
                car.grid_cell_ids = cell_ids.clone();
                car.grid_cell_count = cell_ids.len();
            }
            for cell_id in &cell_ids {
                let occ = {
                    let station = self.ensure_cell_station(cell_id);
                    station.occupy(v.car_id);
                    station.car_ids.len()
                };
                self.max_cell_occupancy = self.max_cell_occupancy.max(occ);
                self.active_cell_ids.insert(cell_id.clone());
                by_cell.entry(cell_id.clone()).or_default().insert(v.car_id);
            }
        }
        let active_cell_count = self.active_cell_ids.len();
        SpatialIndex {
            by_cell,
            active_cell_count,
        }
    }

    fn find_leader_ahead_from_grid(
        &self,
        car: &SmartTrafficCar,
        lane: &TrafficLane,
    ) -> Option<usize> {
        let cell_size = self.grid_cell_size_m();
        let look_ahead = (lane.length_m - car.position_m).min(
            self.params.base.grid_look_ahead_m.unwrap_or_else(|| {
                (car.speed_mps * (self.reaction_time_sec() + 4.0) + 3.0 * self.vehicle_space())
                    .max(60.0)
            }),
        );
        let first = (car.position_m / cell_size).floor().max(0.0) as i64;
        let last = ((car.position_m + look_ahead) / cell_size)
            .floor()
            .max(first as f64) as i64;
        let lateral = self.occupied_lateral_cell_range();
        let mut best: Option<(usize, f64)> = None;
        for x in first..=last {
            for y in lateral.0..=lateral.1 {
                if let Some(ids) = self.spatial.by_cell.get(&self.cell_id(&lane.id, x, y)) {
                    for &id in ids {
                        if id == car.car_id {
                            continue;
                        }
                        if let Some((slot, pos, ln)) = self.car_view_by_id(id) {
                            if ln != car.lane_id || pos <= car.position_m {
                                continue;
                            }
                            if best.is_none() || pos < best.unwrap().1 {
                                best = Some((slot, pos));
                            }
                        }
                    }
                }
            }
            if let Some((_, pos)) = best {
                if pos <= (x + 1) as f64 * cell_size {
                    break;
                }
            }
        }
        best.map(|(slot, _)| slot)
    }

    fn sorted_leader_ahead(&self, car: &SmartTrafficCar) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for v in self.active_car_views() {
            if v.car_id == car.car_id || v.lane_id != car.lane_id || v.position_m <= car.position_m
            {
                continue;
            }
            if best.is_none() || v.position_m < best.unwrap().1 {
                best = Some((v.slot, v.position_m));
            }
        }
        best.map(|(slot, _)| slot)
    }

    fn snapshot_row(&self, time_sec: f64) -> SmartTrafficTraceRow {
        let mut lane_occupancy: HashMap<String, usize> = HashMap::new();
        for lane in &self.network.lanes {
            lane_occupancy.insert(lane.id.clone(), 0);
        }
        for v in self.active_car_views() {
            *lane_occupancy.entry(v.lane_id.clone()).or_insert(0) += 1;
        }
        let mut signal_phases: HashMap<String, String> = HashMap::new();
        for signal in self.network.signals.iter().flatten() {
            signal_phases.insert(
                signal.node_id.clone(),
                current_signal_phase(signal, time_sec).name.clone(),
            );
        }
        let cars = self.snap_cars();
        let mean_speed_mps = if cars.is_empty() {
            0.0
        } else {
            cars.iter().map(|c| c.speed_mps).sum::<f64>() / cars.len() as f64
        };
        let mean_travel_time_sec = if self.completed_travel_times.is_empty() {
            0.0
        } else {
            self.completed_travel_times.iter().sum::<f64>()
                / self.completed_travel_times.len() as f64
        };
        let queue_length = cars.iter().filter(|c| c.speed_mps < 0.5).count();
        SmartTrafficTraceRow {
            tick: self.tick,
            time_sec,
            active_cars: cars.len(),
            scheduled_smart_cars: self.scheduled_smart_cars,
            smart_movable_runs: self.current_actor_run_order.len(),
            entered: self.entered,
            exited: self.exited,
            crashed: self.crashed,
            mean_speed_mps,
            mean_travel_time_sec,
            queue_length,
            lane_occupancy,
            active_grid_cells: self.active_cell_ids.len(),
            signal_phases,
            actor_run_order: self
                .current_actor_run_order
                .iter()
                .take(24)
                .cloned()
                .collect(),
            accidents: self.accidents_this_tick.clone(),
            cars,
        }
    }

    fn snap_cars(&self) -> Vec<SmartTrafficCarSnapshot> {
        let mut snaps: Vec<SmartTrafficCarSnapshot> = self
            .car_actors
            .iter()
            .filter_map(|rc| {
                rc.try_borrow()
                    .ok()
                    .filter(|c| c.is_active())
                    .map(|c| c.snapshot())
            })
            .collect();
        snaps.sort_by_key(|c| c.id);
        snaps
    }

    fn active_car_views(&self) -> Vec<CarView> {
        self.car_actors
            .iter()
            .filter_map(|rc| {
                rc.try_borrow()
                    .ok()
                    .filter(|c| c.is_active())
                    .map(|c| CarView {
                        slot: c.slot_index,
                        car_id: c.car_id,
                        lane_id: c.lane_id.clone(),
                        position_m: c.position_m,
                        speed_mps: c.speed_mps,
                    })
            })
            .collect()
    }

    fn active_car_count(&self) -> usize {
        self.car_actors
            .iter()
            .filter(|rc| rc.try_borrow().map(|c| c.is_active()).unwrap_or(false))
            .count()
    }

    fn find_inactive_actor(&self) -> Option<usize> {
        self.car_actors
            .iter()
            .position(|rc| rc.try_borrow().map(|c| !c.is_active()).unwrap_or(false))
    }

    fn car_view_by_id(&self, car_id: u64) -> Option<(usize, f64, String)> {
        for rc in &self.car_actors {
            if let Ok(c) = rc.try_borrow() {
                if c.is_active() && c.car_id == car_id {
                    return Some((c.slot_index, c.position_m, c.lane_id.clone()));
                }
            }
        }
        None
    }

    fn minimum_body_gap(&self) -> f64 {
        let mut min = f64::INFINITY;
        let views = self.active_car_views();
        for lane in &self.network.lanes {
            let mut cars: Vec<&CarView> = views.iter().filter(|c| c.lane_id == lane.id).collect();
            cars.sort_by(|a, b| a.position_m.partial_cmp(&b.position_m).unwrap());
            for i in 1..cars.len() {
                min = min.min(cars[i].position_m - cars[i - 1].position_m - self.car_length_m());
            }
        }
        if min.is_finite() {
            min
        } else {
            0.0
        }
    }

    fn signal_allows(&self, incoming_lane_id: &str) -> bool {
        let lane = self.lane(incoming_lane_id);
        let node = match self.nodes.get(&lane.to) {
            Some(n) => n,
            None => return true,
        };
        if node.kind != TrafficNodeKind::Intersection {
            return true;
        }
        let signal = match self.signal_by_node.get(&node.id) {
            Some(s) => s,
            None => return true,
        };
        current_signal_phase(signal, self.time_sec)
            .green_lanes
            .iter()
            .any(|l| l == incoming_lane_id)
    }

    fn can_enter_lane(&self, lane_id: &str, ignore_car_id: Option<u64>) -> bool {
        let lane = self.lane(lane_id);
        let cars: Vec<CarView> = self
            .active_car_views()
            .into_iter()
            .filter(|c| c.lane_id == lane_id && Some(c.car_id) != ignore_car_id)
            .collect();
        let cap = lane
            .capacity
            .unwrap_or_else(|| default_lane_capacity(lane, self.vehicle_space()));
        if cars.len() >= cap {
            return false;
        }
        cars.iter().all(|c| c.position_m >= self.vehicle_space())
    }

    fn lane(&self, id: &str) -> &TrafficLane {
        self.lanes
            .get(id)
            .unwrap_or_else(|| panic!("smart-traffic-flow: unknown lane \"{id}\""))
    }

    fn occupied_lateral_cell_range(&self) -> (i64, i64) {
        let cell_size = self.grid_cell_size_m();
        let lane_width = self.lane_width_m();
        let car_width = self.car_width_m().min(lane_width);
        let left = ((lane_width - car_width) / 2.0).max(0.0);
        let right = lane_width.min(left + car_width);
        (
            (left / cell_size).floor().max(0.0) as i64,
            ((left.max(right - 1e-9)) / cell_size).floor().max(0.0) as i64,
        )
    }

    fn occupied_cell_ids_view(&self, v: &CarView) -> Vec<String> {
        let cell_size = self.grid_cell_size_m();
        let rear = (v.position_m - self.car_length_m()).max(0.0);
        let front = rear.max(v.position_m);
        let x0 = (rear / cell_size).floor().max(0.0) as i64;
        let x1 = (front / cell_size).floor().max(x0 as f64) as i64;
        let y = self.occupied_lateral_cell_range();
        let mut ids = Vec::new();
        for x in x0..=x1 {
            for lat in y.0..=y.1 {
                ids.push(self.cell_id(&v.lane_id, x, lat));
            }
        }
        ids
    }

    fn cell_id(&self, lane_id: &str, longitudinal_index: i64, lateral_index: i64) -> String {
        format!("{lane_id}#{longitudinal_index}:{lateral_index}")
    }

    fn accident_cell_id(&self, lane_id: &str, position_m: f64) -> String {
        let x = (position_m / self.grid_cell_size_m()).floor().max(0.0) as i64;
        let y = self.occupied_lateral_cell_range();
        self.cell_id(lane_id, x, y.0)
    }

    fn ensure_cell_station(&mut self, cell_id: &str) -> &mut SmartTrafficCellStation {
        if !self.cell_stations.contains_key(cell_id) {
            let parsed = parse_cell_id(cell_id);
            let cell_size = self.grid_cell_size_m();
            let station = SmartTrafficCellStation::new(SmartTrafficCellBounds {
                lane_id: parsed.0,
                longitudinal_index: parsed.1,
                lateral_index: parsed.2,
                x0_m: parsed.1 as f64 * cell_size,
                x1_m: (parsed.1 + 1) as f64 * cell_size,
                y0_m: parsed.2 as f64 * cell_size,
                y1_m: (parsed.2 + 1) as f64 * cell_size,
            });
            self.cell_stations.insert(cell_id.to_string(), station);
        }
        self.cell_stations.get_mut(cell_id).unwrap()
    }

    // ── parameter getters (TS private accessors with defaults) ───────────────

    fn vehicle_space(&self) -> f64 {
        self.params.base.car_length_m.unwrap_or(4.8) + self.params.base.min_gap_m.unwrap_or(2.5)
    }
    fn car_length_m(&self) -> f64 {
        self.params.base.car_length_m.unwrap_or(4.8)
    }
    fn car_width_m(&self) -> f64 {
        self.params.base.car_width_m.unwrap_or(1.8)
    }
    fn lane_width_m(&self) -> f64 {
        self.params.base.lane_width_m.unwrap_or(3.7)
    }
    fn grid_cell_size_m(&self) -> f64 {
        self.params.base.grid_cell_size_m.unwrap_or(0.3048)
    }
    fn max_accel_mps2(&self) -> f64 {
        self.params.base.max_accel_mps2.unwrap_or(2.2)
    }
    fn max_decel_mps2(&self) -> f64 {
        self.params.base.max_decel_mps2.unwrap_or(4.0)
    }
    fn max_jerk_mps3(&self) -> f64 {
        self.params.base.max_jerk_mps3.unwrap_or(6.0)
    }
    fn reaction_time_sec(&self) -> f64 {
        self.params.base.reaction_time_sec.unwrap_or(0.8)
    }
    fn time_headway_sec(&self) -> f64 {
        self.params.base.time_headway_sec.unwrap_or(1.1)
    }
    fn min_gap_m(&self) -> f64 {
        self.params.base.min_gap_m.unwrap_or(2.5)
    }
    fn accident_risk_scale(&self) -> f64 {
        self.params
            .accident_risk_scale
            .or(self.params.accident_probability)
            .unwrap_or(0.0)
    }
    fn accident_accel_boost_mps2(&self) -> f64 {
        self.params.accident_accel_boost_mps2.unwrap_or(10.0)
    }
    fn accident_fault_duration_sec(&self) -> f64 {
        self.params.accident_fault_duration_sec.unwrap_or(1.0)
    }
    fn distance_preference_spread(&self) -> f64 {
        self.params.distance_preference_spread.unwrap_or(0.0)
    }
    fn start_preference_spread(&self) -> f64 {
        self.params.start_preference_spread.unwrap_or(0.0)
    }

    fn sample_distance_preference(&mut self) -> f64 {
        let spread = self.distance_preference_spread();
        self.sample_driver_trait(spread)
    }
    fn sample_start_preference(&mut self) -> f64 {
        let spread = self.start_preference_spread();
        self.sample_driver_trait(spread)
    }
    fn sample_driver_trait(&mut self, spread: f64) -> f64 {
        if spread <= 0.0 {
            return 1.0;
        }
        let dt = driver_trait();
        let k = sample_from_pmf(&mut self.rng, &dt.pmf);
        let z = if dt.std > 0.0 {
            (k as f64 - dt.mean) / dt.std
        } else {
            0.0
        };
        clamp(1.0 + (spread / 3.0_f64.sqrt()) * z, 0.35, 2.25)
    }

    fn has_startup_clearance(
        &self,
        _car: &SmartTrafficCar,
        perceived_position: f64,
        perceived_id: Option<u64>,
        physical_leader_gap: f64,
    ) -> bool {
        if !perceived_position.is_finite() || perceived_id.is_none() {
            return true;
        }
        // perceived speed is not retained here; the TS uses the perceived
        // leader's speed which equals the leader's reported speed. We re-derive
        // "moving away" from the physical gap heuristic only.
        let required_gap = self.min_gap_m() * (0.45 + _car.start_preference);
        physical_leader_gap >= required_gap || physical_leader_gap >= self.min_gap_m() * 0.45
    }

    fn start_delay_sec(&self, start_preference: f64) -> f64 {
        if self.start_preference_spread() <= 0.0 {
            return 0.0;
        }
        ((start_preference - 0.55) * self.reaction_time_sec() * 0.55).max(0.0)
    }

    // ── preconditions ────────────────────────────────────────────────────────

    fn assert_preconditions_checked(&self) -> Check {
        let p = &self.params;
        Preconditions::check(
            MODEL,
            "network",
            "be provided by builtin or network",
            p.base.builtin.as_deref() == Some("five-intersection") || p.base.network.is_some(),
            None,
        )?;
        Preconditions::positive(MODEL, "durationSec", p.base.duration_sec)?;
        Preconditions::positive(MODEL, "dtSec", p.base.dt_sec)?;
        Preconditions::check(
            MODEL,
            "dtSec",
            "be <= 5 seconds",
            p.base.dt_sec <= 5.0,
            Some(p.base.dt_sec.to_string()),
        )?;
        Preconditions::integer(MODEL, "seed", p.base.seed)?;
        Preconditions::integer_in_range(MODEL, "maxCars", p.base.max_cars as f64, 1.0, 299.0)?;
        Preconditions::integer_in_range(
            MODEL,
            "smartCarPoolSize",
            p.smart_car_pool_size.unwrap_or(p.base.max_cars) as f64,
            p.base.max_cars as f64,
            10000.0,
        )?;
        if let Some(s) = p.actor_shuffle_seed {
            Preconditions::integer(MODEL, "actorShuffleSeed", s)?;
        }
        if let Some(x) = p.base.car_length_m {
            Preconditions::positive(MODEL, "carLengthM", x)?;
        }
        if let Some(x) = p.base.car_width_m {
            Preconditions::positive(MODEL, "carWidthM", x)?;
        }
        if let Some(x) = p.base.lane_width_m {
            Preconditions::positive(MODEL, "laneWidthM", x)?;
        }
        if let Some(x) = p.base.min_gap_m {
            Preconditions::non_negative(MODEL, "minGapM", x)?;
        }
        if let Some(x) = p.base.max_accel_mps2 {
            Preconditions::positive(MODEL, "maxAccelMps2", x)?;
        }
        if let Some(x) = p.base.max_decel_mps2 {
            Preconditions::positive(MODEL, "maxDecelMps2", x)?;
        }
        if let Some(x) = p.base.max_jerk_mps3 {
            Preconditions::positive(MODEL, "maxJerkMps3", x)?;
        }
        if let Some(x) = p.base.reaction_time_sec {
            Preconditions::non_negative(MODEL, "reactionTimeSec", x)?;
        }
        if let Some(x) = p.base.time_headway_sec {
            Preconditions::non_negative(MODEL, "timeHeadwaySec", x)?;
        }
        if let Some(x) = p.base.grid_cell_size_m {
            Preconditions::positive(MODEL, "gridCellSizeM", x)?;
        }
        if let Some(x) = p.base.spawn_rate_multiplier {
            Preconditions::non_negative(MODEL, "spawnRateMultiplier", x)?;
        }
        if let Some(x) = p.accident_risk_scale {
            Preconditions::non_negative(MODEL, "accidentRiskScale", x)?;
        }
        if let Some(x) = p.accident_probability {
            Preconditions::in_range(MODEL, "accidentProbability", x, 0.0, 1.0)?;
        }
        if let Some(x) = p.accident_accel_boost_mps2 {
            Preconditions::non_negative(MODEL, "accidentAccelBoostMps2", x)?;
        }
        if let Some(x) = p.accident_fault_duration_sec {
            Preconditions::positive(MODEL, "accidentFaultDurationSec", x)?;
        }
        if let Some(x) = p.distance_preference_spread {
            Preconditions::in_range(MODEL, "distancePreferenceSpread", x, 0.0, 1.5)?;
        }
        if let Some(x) = p.start_preference_spread {
            Preconditions::in_range(MODEL, "startPreferenceSpread", x, 0.0, 1.5)?;
        }
        if let Some(x) = p.accident_flash_seconds {
            Preconditions::positive(MODEL, "accidentFlashSeconds", x)?;
        }
        Preconditions::check(
            MODEL,
            "carWidthM",
            "fit within laneWidthM",
            self.car_width_m() <= self.lane_width_m(),
            None,
        )?;
        validate_smart_traffic_network(&self.network)?;
        validate_smart_traffic_scheduled_trips(
            &self.network,
            p.base.scheduled_trips.as_deref().unwrap_or(&[]),
            p.base.duration_sec,
        )?;
        Ok(())
    }
}

impl DESStation for SmartTrafficWorldStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn assert_preconditions(&mut self) {
        self.assert_preconditions_checked()
            .unwrap_or_else(|e| panic!("{e}"));
    }
    fn has_work(&self) -> bool {
        self.time_sec < self.params.base.duration_sec - 1e-9
    }
    fn run_time_step(&mut self) {
        // No-op participant; tick-barrier work happens in finish_tick (on_tick).
    }
}

fn downcast(st: &dyn DESStation) -> &SmartTrafficWorldStation {
    st.as_any()
        .downcast_ref::<SmartTrafficWorldStation>()
        .expect("validator received a non-SmartTrafficWorldStation")
}

#[derive(Clone)]
struct LeaderView {
    car_id: u64,
    lane_id: String,
    position_m: f64,
    speed_mps: f64,
    history: Vec<KinematicSample>,
}

fn perceived_sample(
    leader: &LeaderView,
    target_time_sec: f64,
    car: &SmartTrafficCar,
) -> KinematicSample {
    for i in (0..leader.history.len()).rev() {
        if leader.history[i].time_sec <= target_time_sec + 1e-12 {
            return leader.history[i].clone();
        }
    }
    leader.history.first().cloned().unwrap_or(KinematicSample {
        time_sec: target_time_sec,
        lane_id: car.lane_id.clone(),
        position_m: car.position_m,
        speed_mps: car.speed_mps,
        acceleration_mps2: car.acceleration_mps2,
    })
}

// =============================================================================
// Top-level driver + free helpers.
// =============================================================================

pub fn run_smart_traffic_flow(
    params: SmartTrafficParams,
    logger: Option<Rc<dyn OptimizationLogger>>,
) -> SmartTrafficResult {
    let actor_shuffle_seed = params
        .actor_shuffle_seed
        .unwrap_or(params.base.seed + 1009.0);
    let max_ticks = (params.base.duration_sec / params.base.dt_sec).ceil() as usize + 1;
    let world = build_world(params, logger);
    world.borrow_mut().assert_preconditions();
    world.borrow_mut().begin_run();

    let mut participants: Vec<StationRef> = vec![world.clone() as StationRef];
    for car in &world.borrow().car_actors {
        participants.push(car.clone() as StationRef);
    }
    let world_for_tick = world.clone();
    let summary = run_iterative_des(
        participants,
        IterativeRunOptions {
            shuffle: true,
            rng: Some({
                // PORT NOTE: the runner wants a bare `FnMut() -> f64`; wrap the
                // seeded `RandomSource` (mulberry32) in a closure that pulls floats.
                let mut shuffle_rng = mulberry32(actor_shuffle_seed as u32);
                Box::new(move || shuffle_rng.next_float())
            }),
            max_ticks: Some(max_ticks),
            on_tick: Some(Box::new(move |_, _| {
                world_for_tick.borrow_mut().finish_tick()
            })),
            ..Default::default()
        },
    );
    let validation = summary.validation.clone().unwrap_or_default();
    let out = world.borrow().build_result(summary, validation);
    out
}

fn validate_smart_traffic_network(network: &TrafficNetwork) -> Check {
    Preconditions::non_empty(MODEL, "network.nodes", &network.nodes)?;
    Preconditions::non_empty(MODEL, "network.lanes", &network.lanes)?;
    Preconditions::non_empty(MODEL, "network.sources", &network.sources)?;
    Preconditions::non_empty(MODEL, "network.sinks", &network.sinks)?;
    let mut node_ids: HashSet<String> = HashSet::new();
    let mut node_by_id: HashMap<String, TrafficNode> = HashMap::new();
    for node in &network.nodes {
        Preconditions::check(
            MODEL,
            &format!("node.{}", node.id),
            "have a non-empty id",
            !node.id.is_empty(),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("node.{}", node.id),
            "be unique",
            !node_ids.contains(&node.id),
            None,
        )?;
        Preconditions::finite(MODEL, &format!("node.{}.x", node.id), node.x)?;
        Preconditions::finite(MODEL, &format!("node.{}.y", node.id), node.y)?;
        node_ids.insert(node.id.clone());
        node_by_id.insert(node.id.clone(), node.clone());
    }
    let mut lane_ids: HashSet<String> = HashSet::new();
    for lane in &network.lanes {
        Preconditions::check(
            MODEL,
            &format!("lane.{}", lane.id),
            "have a non-empty id",
            !lane.id.is_empty(),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("lane.{}", lane.id),
            "be unique",
            !lane_ids.contains(&lane.id),
            None,
        )?;
        lane_ids.insert(lane.id.clone());
        Preconditions::check(
            MODEL,
            &format!("lane.{}.from", lane.id),
            "reference a node",
            node_ids.contains(&lane.from),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("lane.{}.to", lane.id),
            "reference a node",
            node_ids.contains(&lane.to),
            None,
        )?;
        Preconditions::positive(MODEL, &format!("lane.{}.lengthM", lane.id), lane.length_m)?;
        Preconditions::positive(
            MODEL,
            &format!("lane.{}.speedLimitMps", lane.id),
            lane.speed_limit_mps,
        )?;
        if let Some(c) = lane.capacity {
            Preconditions::integer_in_range(
                MODEL,
                &format!("lane.{}.capacity", lane.id),
                c as f64,
                1.0,
                299.0,
            )?;
        }
    }
    for signal in network.signals.iter().flatten() {
        Preconditions::check(
            MODEL,
            &format!("signal.{}", signal.node_id),
            "reference a node",
            node_ids.contains(&signal.node_id),
            None,
        )?;
        for phase in &signal.phases {
            Preconditions::positive(
                MODEL,
                &format!("signal.{}.{}.durationSec", signal.node_id, phase.name),
                phase.duration_sec,
            )?;
            for lane_id in &phase.green_lanes {
                Preconditions::check(
                    MODEL,
                    &format!("signal.{}.{}.greenLanes", signal.node_id, phase.name),
                    "reference a lane",
                    lane_ids.contains(lane_id),
                    None,
                )?;
            }
        }
    }
    let mut sink_ids: HashSet<String> = HashSet::new();
    for sink in &network.sinks {
        Preconditions::check(
            MODEL,
            &format!("sink.{}", sink.id),
            "have a non-empty id",
            !sink.id.is_empty(),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("sink.{}", sink.id),
            "be unique",
            !sink_ids.contains(&sink.id),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("sink.{}.nodeId", sink.id),
            "reference a node",
            node_ids.contains(&sink.node_id),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("sink.{}.nodeId", sink.id),
            "reference a sink node",
            node_by_id.get(&sink.node_id).map(|n| n.kind) == Some(TrafficNodeKind::Sink),
            None,
        )?;
        sink_ids.insert(sink.id.clone());
    }
    let mut source_ids: HashSet<String> = HashSet::new();
    for source in &network.sources {
        Preconditions::check(
            MODEL,
            &format!("source.{}", source.id),
            "have a non-empty id",
            !source.id.is_empty(),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("source.{}", source.id),
            "be unique",
            !source_ids.contains(&source.id),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("source.{}.nodeId", source.id),
            "reference a node",
            node_ids.contains(&source.node_id),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("source.{}.nodeId", source.id),
            "reference a source node",
            node_by_id.get(&source.node_id).map(|n| n.kind) == Some(TrafficNodeKind::Source),
            None,
        )?;
        Preconditions::non_negative(
            MODEL,
            &format!("source.{}.ratePerMin", source.id),
            source.rate_per_min,
        )?;
        let destination_sink_ids: Vec<String> = source
            .destination_sink_ids
            .clone()
            .unwrap_or_else(|| sink_ids.iter().cloned().collect());
        Preconditions::non_empty(
            MODEL,
            &format!("source.{}.destinationSinkIds", source.id),
            &destination_sink_ids,
        )?;
        source_ids.insert(source.id.clone());
        for sink_id in &destination_sink_ids {
            Preconditions::check(
                MODEL,
                &format!("source.{}.destinationSinkIds", source.id),
                "reference a sink id",
                sink_ids.contains(sink_id),
                None,
            )?;
            if let Some(sink) = network.sinks.iter().find(|s| &s.id == sink_id) {
                Preconditions::check(
                    MODEL,
                    &format!("route {}->{}", source.id, sink_id),
                    "have at least one directed lane path",
                    !shortest_lane_path(network, &source.node_id, &sink.node_id).is_empty(),
                    None,
                )?;
            }
        }
    }
    Ok(())
}

fn validate_smart_traffic_scheduled_trips(
    network: &TrafficNetwork,
    trips: &[TrafficScheduledTrip],
    duration_sec: f64,
) -> Check {
    for trip in trips {
        Preconditions::non_negative(MODEL, "scheduledTrips.departSec", trip.depart_sec)?;
        Preconditions::check(
            MODEL,
            "scheduledTrips.departSec",
            "be within durationSec",
            trip.depart_sec <= duration_sec + 1e-9,
            None,
        )?;
        let source = network.sources.iter().find(|s| s.id == trip.source_id);
        let sink = network
            .sinks
            .iter()
            .find(|s| s.id == trip.destination_sink_id);
        Preconditions::check(
            MODEL,
            &format!("scheduledTrips.{}", trip.source_id),
            "reference a source id",
            source.is_some(),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!("scheduledTrips.{}", trip.destination_sink_id),
            "reference a sink id",
            sink.is_some(),
            None,
        )?;
        let (source, sink) = match (source, sink) {
            (Some(s), Some(k)) => (s, k),
            _ => continue,
        };
        let allowed: Vec<String> = source
            .destination_sink_ids
            .clone()
            .unwrap_or_else(|| network.sinks.iter().map(|s| s.id.clone()).collect());
        Preconditions::check(
            MODEL,
            &format!(
                "scheduledTrips.{}->{}",
                trip.source_id, trip.destination_sink_id
            ),
            "use a sink allowed by the source",
            allowed.contains(&trip.destination_sink_id),
            None,
        )?;
        Preconditions::check(
            MODEL,
            &format!(
                "scheduledTrips.{}->{}",
                trip.source_id, trip.destination_sink_id
            ),
            "have at least one directed lane path",
            !shortest_lane_path(network, &source.node_id, &sink.node_id).is_empty(),
            None,
        )?;
    }
    Ok(())
}

fn default_lane_capacity(lane: &TrafficLane, vehicle_space: f64) -> usize {
    ((lane.length_m / vehicle_space).floor() as usize).max(1)
}

fn current_signal_phase(signal: &TrafficSignal, time_sec: f64) -> &TrafficSignalPhase {
    let cycle: f64 = signal.phases.iter().map(|p| p.duration_sec).sum();
    let mut t = ((time_sec + signal.offset_sec.unwrap_or(0.0)) % cycle + cycle) % cycle;
    for phase in &signal.phases {
        if t < phase.duration_sec {
            return phase;
        }
        t -= phase.duration_sec;
    }
    signal.phases.last().expect("signal must have phases")
}

fn shortest_lane_path(
    network: &TrafficNetwork,
    source_node_id: &str,
    sink_node_id: &str,
) -> Vec<String> {
    let mut dist: HashMap<String, f64> = HashMap::new();
    let mut prev_lane: HashMap<String, String> = HashMap::new();
    let mut prev_node: HashMap<String, String> = HashMap::new();
    let node_ids: Vec<String> = network.nodes.iter().map(|n| n.id.clone()).collect();
    for n in &node_ids {
        dist.insert(n.clone(), f64::INFINITY);
    }
    dist.insert(source_node_id.to_string(), 0.0);
    let mut pending: HashSet<String> = node_ids.iter().cloned().collect();
    while !pending.is_empty() {
        let mut u = String::new();
        let mut best = f64::INFINITY;
        for n in &pending {
            let d = *dist.get(n).unwrap_or(&f64::INFINITY);
            if d < best {
                best = d;
                u = n.clone();
            }
        }
        if u.is_empty() || !best.is_finite() {
            break;
        }
        pending.remove(&u);
        if u == sink_node_id {
            break;
        }
        for lane in network.lanes.iter().filter(|l| l.from == u) {
            let alt = best + lane.length_m;
            if alt < *dist.get(&lane.to).unwrap_or(&f64::INFINITY) {
                dist.insert(lane.to.clone(), alt);
                prev_lane.insert(lane.to.clone(), lane.id.clone());
                prev_node.insert(lane.to.clone(), u.clone());
            }
        }
    }
    if !dist
        .get(sink_node_id)
        .copied()
        .unwrap_or(f64::INFINITY)
        .is_finite()
    {
        return Vec::new();
    }
    let mut route: Vec<String> = Vec::new();
    let mut cur = sink_node_id.to_string();
    while cur != source_node_id {
        let lane = match prev_lane.get(&cur) {
            Some(l) => l.clone(),
            None => return Vec::new(),
        };
        let prev = match prev_node.get(&cur) {
            Some(p) => p.clone(),
            None => return Vec::new(),
        };
        route.push(lane);
        cur = prev;
    }
    route.reverse();
    route
}

fn fault_mode_for_risk(
    speed_risk: f64,
    braking_risk: f64,
    accel_risk: f64,
    ttc_risk: f64,
) -> SmartTrafficFaultMode {
    if speed_risk >= braking_risk && speed_risk >= ttc_risk {
        return SmartTrafficFaultMode::Speeding;
    }
    if braking_risk >= accel_risk {
        SmartTrafficFaultMode::BrakeTooSlow
    } else {
        SmartTrafficFaultMode::AccelerateTooFast
    }
}

fn parse_cell_id(cell_id: &str) -> (String, i64, i64) {
    let sep = cell_id.rfind('#').unwrap_or(0);
    let lane_id = cell_id[..sep].to_string();
    let rest = &cell_id[sep + 1..];
    let mut parts = rest.split(':');
    let x: i64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let y: i64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    (lane_id, x, y)
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> SmartTrafficParams {
        SmartTrafficParams {
            base: TrafficParams {
                builtin: Some("five-intersection".to_string()),
                network: None,
                duration_sec: 20.0,
                dt_sec: 0.5,
                seed: 7.0,
                max_cars: 40,
                car_length_m: None,
                car_width_m: None,
                lane_width_m: None,
                min_gap_m: None,
                max_accel_mps2: None,
                max_decel_mps2: None,
                max_jerk_mps3: None,
                reaction_time_sec: None,
                time_headway_sec: None,
                grid_cell_size_m: None,
                grid_look_ahead_m: None,
                spawn_rate_multiplier: Some(1.0),
                scheduled_trips: None,
            },
            smart_car_pool_size: None,
            actor_shuffle_seed: None,
            accident_risk_scale: Some(0.0),
            accident_probability: None,
            accident_accel_boost_mps2: None,
            accident_fault_duration_sec: None,
            distance_preference_spread: None,
            start_preference_spread: None,
            accident_flash_seconds: None,
        }
    }

    #[test]
    fn smoke_runs_and_conserves() {
        let result = run_smart_traffic_flow(default_params(), None);
        // Conservation: entered = exited + crashed + active.
        let active = result.final_cars.len();
        assert_eq!(result.entered, result.exited + result.crashed + active);
        assert!(result.max_active_cars < 300);
    }
}
