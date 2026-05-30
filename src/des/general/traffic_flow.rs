//! Port of `src/des/general/traffic-flow.ts` — module
//! `des::general::traffic_flow`.
//!
//! A small continuous-ish traffic DES: a stationary road grid coordinates
//! intersections (signal controllers), road links (segments holding continuous
//! car positions / speeds / exit credits) and moving cars that flow
//! Source -> Link -> Intersection -> Link -> Sink.
//!
//! Conversion notes from the TS source:
//!   * Only the [`TrafficGridStation`] is a run-loop participant (the TS
//!     `runTrafficSimulation` runs `runIterativeDES([grid])`). `TrafficCar`,
//!     `IntersectionStation` and `RoadLinkStation` are therefore plain structs
//!     here, NOT `DESStation` impls (their `extends DESStation` / `runTimeStep`
//!     no-ops were never driven by the runner). [`TrafficCar`] still embeds a
//!     [`BasicMovingEntity`] (the TS `extends BasicMovingEntity`) and calls
//!     `do_finish` on completion.
//!   * `RoadLinkStation.step(ctx)` took `canLeave` / `reserveExit` callbacks
//!     that re-entered the grid; those callbacks alias the grid that owns the
//!     link, so the per-link stepping is inlined into the grid over owned cars
//!     (`std::mem::take` pulls the car vector out, releasing the link borrow,
//!     so `can_leave` may read the other links).
//!   * Routing (`shortestNextLink`) depends only on the STATIC topology (node
//!     ids + link `lengthM` weights), so it is precomputed into a `RouteEdge`
//!     adjacency and memoised in `next_link_cache`.
//!   * `Map<number,_>` -> `HashMap<usize,_>`; `Map<string,_>` ->
//!     `HashMap<String,_>`; explicit `link_order` preserves insertion order.
//!   * `mulberry32(seed)` -> [`SeededRandom`]; the field is stored for fidelity
//!     even though the TS source never draws from it.
//!   * `Preconditions` throw -> `Result`; the constructor panics on an invalid
//!     problem (matching the TS throw).

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::des::entity_moving::moving::{BasicMovingEntity, MovingEntity};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::{
    assert_no_validation_failures, run_iterative_des, IterativeRunOptions,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::intrinsic_check;
use crate::des::general::max_flow::{solve_max_flow, MaxFlowEdge, MaxFlowProblem};
use crate::des::general::prng::{mulberry32, SeededRandom};

const MODEL: &str = "traffic-flow";

// =============================================================================
// Specs.
// =============================================================================

#[derive(Clone, Debug)]
pub struct TrafficNodeSpec {
    pub id: usize,
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub signal_offset_sec: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct TrafficLinkSpec {
    pub id: String,
    pub from: usize,
    pub to: usize,
    pub length_m: f64,
    pub speed_limit_mps: f64,
    pub capacity: Option<usize>,
    pub discharge_per_min: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct TrafficSourceSpec {
    pub id: String,
    pub node: usize,
    pub dest_node: usize,
    pub rate_per_min: f64,
    pub max_generated: Option<u64>,
    pub start_sec: Option<f64>,
    pub end_sec: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct TrafficProblem {
    pub nodes: Vec<TrafficNodeSpec>,
    pub links: Vec<TrafficLinkSpec>,
    pub sources: Vec<TrafficSourceSpec>,
    pub duration_sec: f64,
    pub dt_sec: f64,
    pub max_cars: usize,
    pub min_gap_m: f64,
    pub accel_mps2: f64,
    pub signal_cycle_sec: f64,
    pub drain_after_sources_sec: Option<f64>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct TrafficCarSnapshot {
    pub car_id: u64,
    pub origin_node: usize,
    pub dest_node: usize,
    pub birth_time_sec: f64,
    pub exit_time_sec: Option<f64>,
    pub current_link_id: Option<String>,
    pub position_m: f64,
    pub speed_mps: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalAxis {
    Ew,
    Ns,
    All,
}

// =============================================================================
// Moving entity: TrafficCar.
// =============================================================================

/// A car flowing through the grid (TS `class TrafficCar extends
/// BasicMovingEntity implements Token`).
pub struct TrafficCar {
    pub car_id: u64,
    pub origin_node: usize,
    pub dest_node: usize,
    pub birth_time_sec: f64,
    pub exit_time_sec: Option<f64>,
    pub current_link_id: Option<String>,
    pub position_m: f64,
    pub speed_mps: f64,
    moving: BasicMovingEntity,
}

impl TrafficCar {
    pub fn new(car_id: u64, origin_node: usize, dest_node: usize, birth_time_sec: f64) -> Self {
        TrafficCar {
            car_id,
            origin_node,
            dest_node,
            birth_time_sec,
            exit_time_sec: None,
            current_link_id: None,
            position_m: 0.0,
            speed_mps: 0.0,
            moving: BasicMovingEntity::new(),
        }
    }

    pub fn snapshot(&self) -> TrafficCarSnapshot {
        TrafficCarSnapshot {
            car_id: self.car_id,
            origin_node: self.origin_node,
            dest_node: self.dest_node,
            birth_time_sec: self.birth_time_sec,
            exit_time_sec: self.exit_time_sec,
            current_link_id: self.current_link_id.clone(),
            position_m: self.position_m,
            speed_mps: self.speed_mps,
        }
    }

    fn finish(&mut self) {
        self.moving.do_finish();
    }
}

// =============================================================================
// Intersection (signal controller) — plain helper struct.
// =============================================================================

pub struct IntersectionStation {
    pub spec: TrafficNodeSpec,
    controlled: bool,
    cycle_sec: f64,
    offset_sec: f64,
}

impl IntersectionStation {
    fn new(spec: TrafficNodeSpec, degree: usize, cycle_sec: f64) -> Self {
        let offset_sec = spec.signal_offset_sec.unwrap_or(0.0);
        IntersectionStation {
            spec,
            controlled: degree > 2,
            cycle_sec,
            offset_sec,
        }
    }

    pub fn axis_at(&self, time_sec: f64) -> SignalAxis {
        if !self.controlled {
            return SignalAxis::All;
        }
        let phase = positive_modulo(time_sec + self.offset_sec, self.cycle_sec);
        if phase < self.cycle_sec / 2.0 {
            SignalAxis::Ew
        } else {
            SignalAxis::Ns
        }
    }

    pub fn allows(&self, axis: SignalAxis, time_sec: f64) -> bool {
        let active = self.axis_at(time_sec);
        active == SignalAxis::All || active == axis
    }
}

// =============================================================================
// Road link — plain struct (stepping is driven by the grid).
// =============================================================================

pub struct RoadLinkStation {
    pub spec: TrafficLinkSpec,
    pub capacity: usize,
    pub discharge_per_min: f64,
    pub cars: Vec<TrafficCar>,
    exit_credit: f64,
    occupancy_area: f64,
    max_occupancy: usize,
    entered: u64,
    exited: u64,
}

impl RoadLinkStation {
    fn new(spec: TrafficLinkSpec, min_gap_m: f64) -> Self {
        let capacity = spec
            .capacity
            .unwrap_or_else(|| ((spec.length_m / min_gap_m).floor() as usize).max(1));
        let discharge_per_min = spec.discharge_per_min.unwrap_or(30.0);
        RoadLinkStation {
            spec,
            capacity,
            discharge_per_min,
            cars: Vec::new(),
            exit_credit: 0.0,
            occupancy_area: 0.0,
            max_occupancy: 0,
            entered: 0,
            exited: 0,
        }
    }

    fn can_accept_entry(&self, min_gap_m: f64, reserved_incoming: usize) -> bool {
        if self.cars.len() + reserved_incoming >= self.capacity {
            return false;
        }
        if reserved_incoming > 0 {
            return false;
        }
        for car in &self.cars {
            if car.position_m < min_gap_m {
                return false;
            }
        }
        true
    }

    fn insert_at_entry(&mut self, mut car: TrafficCar) {
        car.current_link_id = Some(self.spec.id.clone());
        car.position_m = 0.0;
        car.speed_mps = car.speed_mps.min(self.spec.speed_limit_mps);
        self.cars.push(car);
        self.entered += 1;
    }

    fn stats(&self, duration_sec: f64) -> TrafficLinkStats {
        TrafficLinkStats {
            id: self.spec.id.clone(),
            from: self.spec.from,
            to: self.spec.to,
            capacity: self.capacity as f64,
            entered: self.entered as f64,
            exited: self.exited as f64,
            final_occupancy: self.cars.len() as f64,
            max_occupancy: self.max_occupancy as f64,
            avg_occupancy: self.occupancy_area / duration_sec.max(1.0),
        }
    }
}

/// A car ready to leave its link this tick (TS `interface PendingExit`).
struct PendingExit {
    car: TrafficCar,
    from_link_id: String,
    at_node: usize,
}

#[derive(Clone, Debug)]
struct SourceState {
    spec: TrafficSourceSpec,
    pending: f64,
    generated: u64,
    blocked_attempts: u64,
}

// =============================================================================
// Output structs.
// =============================================================================

#[derive(Clone, Debug)]
pub struct TrafficLinkStats {
    pub id: String,
    pub from: usize,
    pub to: usize,
    pub capacity: f64,
    pub entered: f64,
    pub exited: f64,
    pub final_occupancy: f64,
    pub max_occupancy: f64,
    pub avg_occupancy: f64,
}

#[derive(Clone, Debug)]
pub struct TrafficTimeSample {
    pub t: f64,
    pub active_cars: f64,
    pub completed_cars: f64,
    pub generated_cars: f64,
}

#[derive(Clone, Debug)]
pub struct TrafficSimulationResult {
    pub generated_cars: f64,
    pub completed_cars: f64,
    pub active_cars: f64,
    pub max_active_cars: f64,
    pub blocked_source_attempts: f64,
    pub mean_travel_time_sec: f64,
    pub p95_travel_time_sec: f64,
    pub mean_speed_mps: f64,
    pub throughput_per_hour: f64,
    pub max_flow_upper_bound_per_min: f64,
    pub throughput_vs_max_flow: f64,
    pub total_simulated_sec: f64,
    pub link_stats: Vec<TrafficLinkStats>,
    pub time_series: Vec<TrafficTimeSample>,
    pub invariant_violations: Vec<String>,
}

// =============================================================================
// Routing topology (static).
// =============================================================================

#[derive(Clone, Debug)]
struct RouteEdge {
    link_id: String,
    to: usize,
    weight: f64,
}

// =============================================================================
// The grid station.
// =============================================================================

pub struct TrafficGridStation {
    core: StationCore,
    p: TrafficProblem,
    rng: SeededRandom,
    nodes_by_id: HashMap<usize, TrafficNodeSpec>,
    intersections: HashMap<usize, IntersectionStation>,
    links_by_id: HashMap<String, RoadLinkStation>,
    link_order: Vec<String>,
    node_ids: Vec<usize>,
    topo_outgoing: HashMap<usize, Vec<RouteEdge>>,
    incoming_reservations: HashMap<String, usize>,
    sources: Vec<SourceState>,
    completed: Vec<TrafficCar>,
    time_series: Vec<TrafficTimeSample>,
    invariant_violations: Vec<String>,
    next_link_cache: HashMap<String, Option<String>>,
    next_car_id: u64,
    time_sec: f64,
    max_active_cars: usize,
    speed_integral: f64,
    speed_samples: u64,
}

impl TrafficGridStation {
    pub fn new(p: TrafficProblem) -> Self {
        validate_traffic_problem(&p).unwrap_or_else(|e| panic!("{e}"));
        let rng = mulberry32(p.seed.unwrap_or(1));

        let mut nodes_by_id: HashMap<usize, TrafficNodeSpec> = HashMap::new();
        let mut node_ids: Vec<usize> = Vec::new();
        for n in &p.nodes {
            nodes_by_id.insert(n.id, n.clone());
            node_ids.push(n.id);
        }

        let mut degree: HashMap<usize, usize> = HashMap::new();
        for n in &p.nodes {
            degree.insert(n.id, 0);
        }
        let mut links_by_id: HashMap<String, RoadLinkStation> = HashMap::new();
        let mut link_order: Vec<String> = Vec::new();
        let mut topo_outgoing: HashMap<usize, Vec<RouteEdge>> = HashMap::new();
        for l in &p.links {
            *degree.entry(l.from).or_insert(0) += 1;
            *degree.entry(l.to).or_insert(0) += 1;
            links_by_id.insert(l.id.clone(), RoadLinkStation::new(l.clone(), p.min_gap_m));
            link_order.push(l.id.clone());
            topo_outgoing.entry(l.from).or_default().push(RouteEdge {
                link_id: l.id.clone(),
                to: l.to,
                weight: l.length_m,
            });
        }

        let mut intersections: HashMap<usize, IntersectionStation> = HashMap::new();
        for n in &p.nodes {
            intersections.insert(
                n.id,
                IntersectionStation::new(
                    n.clone(),
                    *degree.get(&n.id).unwrap_or(&0),
                    p.signal_cycle_sec,
                ),
            );
        }

        let sources: Vec<SourceState> = p
            .sources
            .iter()
            .map(|spec| SourceState {
                spec: spec.clone(),
                pending: 0.0,
                generated: 0,
                blocked_attempts: 0,
            })
            .collect();

        let mut station = TrafficGridStation {
            core: StationCore::new("traffic-grid"),
            p,
            rng,
            nodes_by_id,
            intersections,
            links_by_id,
            link_order,
            node_ids,
            topo_outgoing,
            incoming_reservations: HashMap::new(),
            sources,
            completed: Vec::new(),
            time_series: Vec::new(),
            invariant_violations: Vec::new(),
            next_link_cache: HashMap::new(),
            next_car_id: 1,
            time_sec: 0.0,
            max_active_cars: 0,
            speed_integral: 0.0,
            speed_samples: 0,
        };

        let conservation = intrinsic_check::<dyn DESStation>(
            "traffic.conservation",
            |st: &dyn DESStation| {
                let s = downcast(st);
                s.generated_cars() == s.completed.len() as u64 + s.active_cars() as u64
            },
            Some("generated = completed + active".to_string()),
            Some(Box::new(|st: &dyn DESStation| {
                let s = downcast(st);
                format!(
                    "generated={}, completed={}, active={}",
                    s.generated_cars(),
                    s.completed.len(),
                    s.active_cars()
                )
            })),
            Some("traffic-intrinsic".to_string()),
            None,
        )
        .boxed();
        station.add_validator(conservation);

        let car_cap = intrinsic_check::<dyn DESStation>(
            "traffic.car-cap",
            |st: &dyn DESStation| {
                let s = downcast(st);
                s.max_active_cars <= s.p.max_cars && s.max_active_cars < 300
            },
            Some("max active cars below configured cap and below 300".to_string()),
            Some(Box::new(|st: &dyn DESStation| {
                let s = downcast(st);
                format!("maxActive={}, cap={}", s.max_active_cars, s.p.max_cars)
            })),
            Some("traffic-intrinsic".to_string()),
            None,
        )
        .boxed();
        station.add_validator(car_cap);

        station
    }

    pub fn build_result(&self) -> TrafficSimulationResult {
        let mut travel_times: Vec<f64> = self
            .completed
            .iter()
            .map(|c| c.exit_time_sec.unwrap_or(self.time_sec) - c.birth_time_sec)
            .collect();
        travel_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_travel = travel_times.iter().sum::<f64>() / (travel_times.len().max(1) as f64);
        let p95 = if travel_times.is_empty() {
            f64::NAN
        } else {
            let idx = ((0.95 * (travel_times.len() as f64 - 1.0)).floor() as usize)
                .min(travel_times.len() - 1);
            travel_times[idx]
        };
        let max_flow = solve_max_flow(build_traffic_max_flow_problem(&self.p)).max_flow;
        let simulated_minutes = (self.p.duration_sec / 60.0).max(1e-9);
        let throughput_per_min = self.completed.len() as f64 / simulated_minutes;
        let link_stats: Vec<TrafficLinkStats> = self
            .link_order
            .iter()
            .map(|id| self.links_by_id[id].stats(self.time_sec))
            .collect();
        TrafficSimulationResult {
            generated_cars: self.generated_cars() as f64,
            completed_cars: self.completed.len() as f64,
            active_cars: self.active_cars() as f64,
            max_active_cars: self.max_active_cars as f64,
            blocked_source_attempts: self.sources.iter().map(|x| x.blocked_attempts).sum::<u64>()
                as f64,
            mean_travel_time_sec: mean_travel,
            p95_travel_time_sec: p95,
            mean_speed_mps: self.speed_integral / (self.speed_samples.max(1) as f64),
            throughput_per_hour: throughput_per_min * 60.0,
            max_flow_upper_bound_per_min: max_flow,
            throughput_vs_max_flow: throughput_per_min / max_flow.max(1e-9),
            total_simulated_sec: self.time_sec,
            link_stats,
            time_series: self.time_series.clone(),
            invariant_violations: self.invariant_violations.clone(),
        }
    }

    // ── per-tick phases ──────────────────────────────────────────────────────

    fn inject_sources(&mut self) {
        let time = self.time_sec;
        let dt = self.p.dt_sec;
        let max_cars = self.p.max_cars;
        let min_gap = self.p.min_gap_m;
        let duration = self.p.duration_sec;
        for si in 0..self.sources.len() {
            let s = self.sources[si].spec.clone();
            if time < s.start_sec.unwrap_or(0.0) || time > s.end_sec.unwrap_or(duration) {
                continue;
            }
            self.sources[si].pending += s.rate_per_min * dt / 60.0;
            while self.sources[si].pending >= 1.0 - 1e-12 {
                if let Some(mg) = s.max_generated {
                    if self.sources[si].generated >= mg {
                        self.sources[si].pending = 0.0;
                        break;
                    }
                }
                if self.active_cars() >= max_cars {
                    self.sources[si].blocked_attempts += 1;
                    break;
                }
                let next_link = match self.next_link_from(s.node, s.dest_node) {
                    Some(x) => x,
                    None => {
                        self.sources[si].blocked_attempts += 1;
                        self.sources[si].pending -= 1.0;
                        continue;
                    }
                };
                let reserved = self
                    .incoming_reservations
                    .get(&next_link)
                    .copied()
                    .unwrap_or(0);
                let can = self
                    .links_by_id
                    .get(&next_link)
                    .map(|l| l.can_accept_entry(min_gap, reserved))
                    .unwrap_or(false);
                if !can {
                    self.sources[si].blocked_attempts += 1;
                    break;
                }
                let car = TrafficCar::new(self.next_car_id, s.node, s.dest_node, time);
                self.next_car_id += 1;
                self.links_by_id
                    .get_mut(&next_link)
                    .unwrap()
                    .insert_at_entry(car);
                self.sources[si].generated += 1;
                self.sources[si].pending -= 1.0;
            }
        }
    }

    /// Step a single link's car kinematics (inlined `RoadLinkStation.step` with
    /// the grid callbacks resolved against `self`).
    fn step_link(&mut self, lid: &str) -> Vec<PendingExit> {
        let _time = self.time_sec;
        let dt = self.p.dt_sec;
        let min_gap = self.p.min_gap_m;
        let accel = self.p.accel_mps2;

        let spec: TrafficLinkSpec;
        let mut exit_credit: f64;
        let mut cars: Vec<TrafficCar>;
        {
            let link = self.links_by_id.get_mut(lid).unwrap();
            link.exit_credit += link.discharge_per_min * dt / 60.0;
            spec = link.spec.clone();
            exit_credit = link.exit_credit;
            cars = std::mem::take(&mut link.cars);
        }
        // Sort by position descending (front of the link first).
        cars.sort_by(|a, b| b.position_m.partial_cmp(&a.position_m).unwrap());

        let mut exits: Vec<PendingExit> = Vec::new();
        let mut survivors: Vec<TrafficCar> = Vec::new();
        let mut exited_delta: u64 = 0;
        let mut leader_pos: Option<f64> = None;
        let mut leader_exited = false;

        for (i, mut car) in cars.into_iter().enumerate() {
            let desired_speed = (car.speed_mps + accel * dt).min(spec.speed_limit_mps);
            let mut max_move = desired_speed * dt;
            if i > 0 && !leader_exited {
                let lp = leader_pos.unwrap_or(spec.length_m);
                max_move = max_move.min((lp - min_gap - car.position_m).max(0.0));
            } else {
                let can_leave = exit_credit >= 1.0 && self.can_leave(&spec, &car);
                if !can_leave {
                    max_move = max_move.min((spec.length_m - min_gap - car.position_m).max(0.0));
                }
            }
            let mv = max_move.max(0.0);
            car.position_m += mv;
            car.speed_mps = mv / dt;

            if car.position_m >= spec.length_m - 1e-9
                && exit_credit >= 1.0
                && self.can_leave(&spec, &car)
            {
                exit_credit -= 1.0;
                self.reserve_exit(&spec, &car);
                exited_delta += 1;
                leader_exited = true;
                exits.push(PendingExit {
                    car,
                    from_link_id: spec.id.clone(),
                    at_node: spec.to,
                });
            } else {
                car.position_m = car.position_m.min(spec.length_m - 1e-6);
                leader_pos = Some(car.position_m);
                leader_exited = false;
                survivors.push(car);
            }
        }

        {
            let link = self.links_by_id.get_mut(lid).unwrap();
            link.exit_credit = exit_credit;
            link.cars = survivors;
            link.exited += exited_delta;
            link.occupancy_area += link.cars.len() as f64 * dt;
            link.max_occupancy = link.max_occupancy.max(link.cars.len());
        }
        exits
    }

    fn can_leave(&mut self, link: &TrafficLinkSpec, car: &TrafficCar) -> bool {
        let to = link.to;
        let axis = self.axis_of(link);
        let allows = {
            let node = self
                .intersections
                .get(&to)
                .expect("intersection must exist");
            node.allows(axis, self.time_sec)
        };
        if !allows {
            return false;
        }
        if to == car.dest_node {
            return true;
        }
        let next_link_id = match self.next_link_from(to, car.dest_node) {
            Some(x) => x,
            None => return false,
        };
        let reserved = self
            .incoming_reservations
            .get(&next_link_id)
            .copied()
            .unwrap_or(0);
        let min_gap = self.p.min_gap_m;
        self.links_by_id
            .get(&next_link_id)
            .map(|next| next.can_accept_entry(min_gap, reserved))
            .unwrap_or(false)
    }

    fn reserve_exit(&mut self, link: &TrafficLinkSpec, car: &TrafficCar) {
        if link.to == car.dest_node {
            return;
        }
        let next = match self.next_link_from(link.to, car.dest_node) {
            Some(x) => x,
            None => return,
        };
        *self.incoming_reservations.entry(next).or_insert(0) += 1;
    }

    fn apply_exit(&mut self, ex: PendingExit) {
        let mut car = ex.car;
        if ex.at_node == car.dest_node {
            car.exit_time_sec = Some(self.time_sec + self.p.dt_sec);
            car.current_link_id = None;
            car.finish();
            self.completed.push(car);
            return;
        }
        let next_link_id = match self.next_link_from(ex.at_node, car.dest_node) {
            Some(x) => x,
            None => {
                self.invariant_violations.push(format!(
                    "car {} has no route from {} to {}",
                    car.car_id, ex.at_node, car.dest_node
                ));
                return;
            }
        };
        self.links_by_id
            .get_mut(&next_link_id)
            .unwrap()
            .insert_at_entry(car);
    }

    fn next_link_from(&mut self, node: usize, dest: usize) -> Option<String> {
        let key = format!("{node}->{dest}");
        if let Some(v) = self.next_link_cache.get(&key) {
            return v.clone();
        }
        let result = shortest_next_link(&self.node_ids, &self.topo_outgoing, node, dest);
        self.next_link_cache.insert(key, result.clone());
        result
    }

    fn axis_of(&self, link: &TrafficLinkSpec) -> SignalAxis {
        let a = &self.nodes_by_id[&link.from];
        let b = &self.nodes_by_id[&link.to];
        if (b.x - a.x).abs() >= (b.y - a.y).abs() {
            SignalAxis::Ew
        } else {
            SignalAxis::Ns
        }
    }

    fn active_cars(&self) -> usize {
        self.links_by_id.values().map(|l| l.cars.len()).sum()
    }

    fn generated_cars(&self) -> u64 {
        self.sources.iter().map(|x| x.generated).sum()
    }

    fn record_stats(&mut self) {
        let active = self.active_cars();
        self.max_active_cars = self.max_active_cars.max(active);
        for link in self.links_by_id.values() {
            for car in &link.cars {
                self.speed_integral += car.speed_mps;
                self.speed_samples += 1;
            }
        }
        if ((self.time_sec / self.p.dt_sec) % 10.0).abs() < 1e-9 {
            self.time_series.push(TrafficTimeSample {
                t: self.time_sec,
                active_cars: active as f64,
                completed_cars: self.completed.len() as f64,
                generated_cars: self.generated_cars() as f64,
            });
        }
    }

    fn record_invariants(&mut self) {
        let min_gap = self.p.min_gap_m;
        for id in &self.link_order.clone() {
            let (mut cars, capacity, length_m): (Vec<TrafficCarSnapshot>, usize, f64) = {
                let link = &self.links_by_id[id];
                (
                    link.cars.iter().map(|c| c.snapshot()).collect(),
                    link.capacity,
                    link.spec.length_m,
                )
            };
            cars.sort_by(|a, b| a.position_m.partial_cmp(&b.position_m).unwrap());
            if cars.len() > capacity {
                self.invariant_violations.push(format!(
                    "{}: occupancy {} exceeds cap {}",
                    id,
                    cars.len(),
                    capacity
                ));
            }
            for i in 0..cars.len() {
                let c = &cars[i];
                if c.position_m < -1e-6 || c.position_m > length_m + 1e-6 {
                    self.invariant_violations.push(format!(
                        "{}: car {} out of bounds at {}",
                        id, c.car_id, c.position_m
                    ));
                }
                if i > 0 && cars[i].position_m - cars[i - 1].position_m < min_gap - 1e-6 {
                    self.invariant_violations.push(format!(
                        "{}: car gap violation {}/{}",
                        id,
                        cars[i - 1].car_id,
                        cars[i].car_id
                    ));
                }
            }
        }
    }
}

impl DESStation for TrafficGridStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        let drain_until = self.p.duration_sec + self.p.drain_after_sources_sec.unwrap_or(300.0);
        self.time_sec < self.p.duration_sec
            || (self.active_cars() > 0 && self.time_sec < drain_until)
    }
    fn run_time_step(&mut self) {
        self.incoming_reservations.clear();
        if self.time_sec < self.p.duration_sec {
            self.inject_sources();
        }
        let mut pending_exits: Vec<PendingExit> = Vec::new();
        for lid in &self.link_order.clone() {
            let exits = self.step_link(lid);
            pending_exits.extend(exits);
        }
        for ex in pending_exits {
            self.apply_exit(ex);
        }
        self.record_invariants();
        self.record_stats();
        self.time_sec += self.p.dt_sec;
    }
}

fn downcast(st: &dyn DESStation) -> &TrafficGridStation {
    st.as_any()
        .downcast_ref::<TrafficGridStation>()
        .expect("validator received a non-TrafficGridStation")
}

fn shortest_next_link(
    node_ids: &[usize],
    outgoing: &HashMap<usize, Vec<RouteEdge>>,
    node: usize,
    dest: usize,
) -> Option<String> {
    if node == dest {
        return None;
    }
    let mut dist: HashMap<usize, f64> = node_ids.iter().map(|&id| (id, f64::INFINITY)).collect();
    let mut prev_node: HashMap<usize, usize> = HashMap::new();
    let mut prev_link: HashMap<usize, String> = HashMap::new();
    let mut unsettled: HashSet<usize> = node_ids.iter().copied().collect();
    dist.insert(node, 0.0);
    while !unsettled.is_empty() {
        let mut u: Option<usize> = None;
        let mut best = f64::INFINITY;
        for &id in &unsettled {
            let d = *dist.get(&id).unwrap_or(&f64::INFINITY);
            if d < best {
                best = d;
                u = Some(id);
            }
        }
        let u = match u {
            Some(x) if best.is_finite() => x,
            _ => break,
        };
        unsettled.remove(&u);
        if u == dest {
            break;
        }
        if let Some(edges) = outgoing.get(&u) {
            for e in edges {
                if !unsettled.contains(&e.to) {
                    continue;
                }
                let nd = best + e.weight;
                if nd < *dist.get(&e.to).unwrap_or(&f64::INFINITY) {
                    dist.insert(e.to, nd);
                    prev_node.insert(e.to, u);
                    prev_link.insert(e.to, e.link_id.clone());
                }
            }
        }
    }
    prev_link.get(&dest)?;
    let mut cur = dest;
    let mut first = prev_link.get(&cur).unwrap().clone();
    while *prev_node.get(&cur).unwrap_or(&node) != node {
        cur = *prev_node.get(&cur).unwrap();
        first = prev_link.get(&cur).unwrap().clone();
    }
    Some(first)
}

// =============================================================================
// Validation, max-flow upper bound, builders.
// =============================================================================

pub fn validate_traffic_problem(p: &TrafficProblem) -> Check {
    Preconditions::non_empty(MODEL, "nodes", &p.nodes)?;
    Preconditions::non_empty(MODEL, "links", &p.links)?;
    Preconditions::non_empty(MODEL, "sources", &p.sources)?;
    Preconditions::positive(MODEL, "durationSec", p.duration_sec)?;
    Preconditions::positive(MODEL, "dtSec", p.dt_sec)?;
    Preconditions::integer_in_range(MODEL, "maxCars", p.max_cars as f64, 1.0, 299.0)?;
    Preconditions::positive(MODEL, "minGapM", p.min_gap_m)?;
    Preconditions::positive(MODEL, "accelMps2", p.accel_mps2)?;
    Preconditions::positive(MODEL, "signalCycleSec", p.signal_cycle_sec)?;

    let mut node_ids: HashSet<usize> = HashSet::new();
    for n in &p.nodes {
        Preconditions::check(
            MODEL,
            &format!("node {}", n.id),
            "be unique",
            !node_ids.contains(&n.id),
            Some(n.id.to_string()),
        )?;
        node_ids.insert(n.id);
    }
    let mut link_ids: HashSet<String> = HashSet::new();
    for l in &p.links {
        Preconditions::check(
            MODEL,
            &format!("link {}", l.id),
            "be unique",
            !link_ids.contains(&l.id),
            Some(l.id.clone()),
        )?;
        link_ids.insert(l.id.clone());
        Preconditions::check(
            MODEL,
            &format!("{}.from", l.id),
            "reference a node",
            node_ids.contains(&l.from),
            Some(l.from.to_string()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("{}.to", l.id),
            "reference a node",
            node_ids.contains(&l.to),
            Some(l.to.to_string()),
        )?;
        Preconditions::positive(MODEL, &format!("{}.lengthM", l.id), l.length_m)?;
        Preconditions::positive(MODEL, &format!("{}.speedLimitMps", l.id), l.speed_limit_mps)?;
        if let Some(c) = l.capacity {
            Preconditions::integer_in_range(
                MODEL,
                &format!("{}.capacity", l.id),
                c as f64,
                1.0,
                299.0,
            )?;
        }
        if let Some(d) = l.discharge_per_min {
            Preconditions::positive(MODEL, &format!("{}.dischargePerMin", l.id), d)?;
        }
    }

    let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
    for l in &p.links {
        outgoing.entry(l.from).or_default().push(l.to);
    }
    for s in &p.sources {
        Preconditions::check(
            MODEL,
            &format!("{}.node", s.id),
            "reference a node",
            node_ids.contains(&s.node),
            Some(s.node.to_string()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("{}.destNode", s.id),
            "reference a node",
            node_ids.contains(&s.dest_node),
            Some(s.dest_node.to_string()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("{}.node != destNode", s.id),
            "hold",
            s.node != s.dest_node,
            Some(format!("[{}, {}]", s.node, s.dest_node)),
        )?;
        Preconditions::non_negative(MODEL, &format!("{}.ratePerMin", s.id), s.rate_per_min)?;
        if let Some(mg) = s.max_generated {
            Preconditions::integer_in_range(
                MODEL,
                &format!("{}.maxGenerated", s.id),
                mg as f64,
                0.0,
                1e6,
            )?;
        }
        if let Some(st) = s.start_sec {
            Preconditions::non_negative(MODEL, &format!("{}.startSec", s.id), st)?;
        }
        if let Some(en) = s.end_sec {
            Preconditions::non_negative(MODEL, &format!("{}.endSec", s.id), en)?;
        }
        if let (Some(st), Some(en)) = (s.start_sec, s.end_sec) {
            Preconditions::check(
                MODEL,
                &format!("{}.startSec <= endSec", s.id),
                "hold",
                st <= en,
                Some(format!("[{st}, {en}]")),
            )?;
        }
        Preconditions::check(
            MODEL,
            &format!("{}.route", s.id),
            "exist in directed link graph",
            has_directed_path(s.node, s.dest_node, &outgoing),
            Some(format!("[{}, {}]", s.node, s.dest_node)),
        )?;
    }
    if let Some(d) = p.drain_after_sources_sec {
        Preconditions::non_negative(MODEL, "drainAfterSourcesSec", d)?;
    }
    Ok(())
}

pub fn run_traffic_simulation(p: &TrafficProblem) -> TrafficSimulationResult {
    let grid = Rc::new(RefCell::new(TrafficGridStation::new(p.clone())));
    let drain = p.drain_after_sources_sec.unwrap_or(300.0);
    let max_ticks = ((p.duration_sec + drain) / p.dt_sec).ceil() as usize + 5;
    let summary = run_iterative_des(
        vec![grid.clone() as StationRef],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_ticks),
            ..Default::default()
        },
    );
    assert_no_validation_failures(&summary, "traffic").unwrap_or_else(|e| panic!("{e}"));
    let out = grid.borrow().build_result();
    out
}

pub fn build_traffic_max_flow_problem(p: &TrafficProblem) -> MaxFlowProblem {
    validate_traffic_problem(p).unwrap_or_else(|e| panic!("{e}"));
    let super_source = p.nodes.len();
    let super_sink = p.nodes.len() + 1;
    let max_demand: f64 = p.sources.iter().map(|x| x.rate_per_min).sum();
    // `[...new Set(sources.map(destNode))]` preserves first-seen order.
    let mut sink_nodes_ordered: Vec<usize> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for s in &p.sources {
        if seen.insert(s.dest_node) {
            sink_nodes_ordered.push(s.dest_node);
        }
    }
    let mut edges: Vec<MaxFlowEdge> = Vec::new();
    for s in &p.sources {
        edges.push(MaxFlowEdge {
            from: super_source,
            to: s.node,
            capacity: s.rate_per_min,
            name: Some(format!("src-{}", s.id)),
        });
    }
    for l in &p.links {
        edges.push(MaxFlowEdge {
            from: l.from,
            to: l.to,
            capacity: l.discharge_per_min.unwrap_or(30.0),
            name: Some(l.id.clone()),
        });
    }
    for n in &sink_nodes_ordered {
        edges.push(MaxFlowEdge {
            from: *n,
            to: super_sink,
            capacity: max_demand.max(1.0),
            name: Some(format!("sink-{n}")),
        });
    }
    MaxFlowProblem {
        num_nodes: p.nodes.len() + 2,
        source: super_source,
        sink: super_sink,
        edges,
    }
}

pub fn build_default_traffic_problem() -> TrafficProblem {
    let nodes = vec![
        TrafficNodeSpec {
            id: 0,
            name: "W".to_string(),
            x: 0.0,
            y: 1.0,
            signal_offset_sec: None,
        },
        TrafficNodeSpec {
            id: 1,
            name: "C".to_string(),
            x: 1.0,
            y: 1.0,
            signal_offset_sec: None,
        },
        TrafficNodeSpec {
            id: 2,
            name: "E".to_string(),
            x: 2.0,
            y: 1.0,
            signal_offset_sec: None,
        },
        TrafficNodeSpec {
            id: 3,
            name: "N".to_string(),
            x: 1.0,
            y: 2.0,
            signal_offset_sec: None,
        },
        TrafficNodeSpec {
            id: 4,
            name: "S".to_string(),
            x: 1.0,
            y: 0.0,
            signal_offset_sec: None,
        },
    ];
    let mk = |id: &str, from: usize, to: usize| TrafficLinkSpec {
        id: id.to_string(),
        from,
        to,
        length_m: 180.0,
        speed_limit_mps: 13.4,
        capacity: Some(24),
        discharge_per_min: Some(30.0),
    };
    TrafficProblem {
        nodes,
        links: vec![
            mk("W-C", 0, 1),
            mk("C-W", 1, 0),
            mk("C-E", 1, 2),
            mk("E-C", 2, 1),
            mk("N-C", 3, 1),
            mk("C-N", 1, 3),
            mk("S-C", 4, 1),
            mk("C-S", 1, 4),
        ],
        sources: vec![
            TrafficSourceSpec {
                id: "west-to-east".to_string(),
                node: 0,
                dest_node: 2,
                rate_per_min: 12.0,
                max_generated: Some(90),
                start_sec: None,
                end_sec: None,
            },
            TrafficSourceSpec {
                id: "north-to-south".to_string(),
                node: 3,
                dest_node: 4,
                rate_per_min: 9.0,
                max_generated: Some(70),
                start_sec: None,
                end_sec: None,
            },
            TrafficSourceSpec {
                id: "south-to-east".to_string(),
                node: 4,
                dest_node: 2,
                rate_per_min: 6.0,
                max_generated: Some(50),
                start_sec: None,
                end_sec: None,
            },
            TrafficSourceSpec {
                id: "east-to-west".to_string(),
                node: 2,
                dest_node: 0,
                rate_per_min: 5.0,
                max_generated: Some(30),
                start_sec: None,
                end_sec: None,
            },
        ],
        duration_sec: 600.0,
        dt_sec: 1.0,
        max_cars: 240,
        min_gap_m: 7.5,
        accel_mps2: 2.0,
        signal_cycle_sec: 60.0,
        drain_after_sources_sec: Some(420.0),
        seed: Some(7),
    }
}

fn positive_modulo(x: f64, m: f64) -> f64 {
    ((x % m) + m) % m
}

fn has_directed_path(source: usize, sink: usize, outgoing: &HashMap<usize, Vec<usize>>) -> bool {
    let mut seen: HashSet<usize> = HashSet::new();
    seen.insert(source);
    let mut q: Vec<usize> = vec![source];
    let mut qi = 0;
    while qi < q.len() {
        let u = q[qi];
        qi += 1;
        if u == sink {
            return true;
        }
        if let Some(neighbours) = outgoing.get(&u) {
            for &v in neighbours {
                if seen.contains(&v) {
                    continue;
                }
                seen.insert(v);
                q.push(v);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_problem_validates_and_runs() {
        let p = build_default_traffic_problem();
        assert!(validate_traffic_problem(&p).is_ok());
        let result = run_traffic_simulation(&p);
        // Conservation: generated = completed + active.
        let conserved = result.completed_cars + result.active_cars;
        assert!((result.generated_cars - conserved).abs() < 0.5);
        assert!(result.generated_cars > 0.0);
        assert!(result.max_active_cars < 300.0);
        assert!(result.max_flow_upper_bound_per_min > 0.0);
    }

    #[test]
    fn rejects_self_loop_source() {
        let mut p = build_default_traffic_problem();
        p.sources[0].dest_node = p.sources[0].node;
        assert!(validate_traffic_problem(&p).is_err());
    }
}
