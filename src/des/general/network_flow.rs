//! Port of `src/des/general/network-flow.ts` — augmenting-path max-flow + a
//! fixed-step traffic-flow DES on a road grid.
//!
//! Max-flow is modelled as one augmenting path per DES tick. Traffic flow is a
//! continuous-time fixed-step simulation where the stationary grid owns lanes,
//! intersections, sources, and sinks; cars are moving tokens whose feasible
//! motion is constrained by headway, downstream capacity, and signal phases.
//!
//! NOTE: despite the "network-flow" name this file implements *max-flow* (not a
//! transportation / min-cost-flow LP) plus the traffic micro-simulation; the
//! tests therefore cover the max-flow optimum and traffic conservation.
//!
//! MIGRATION NOTES
//!   * INJECT RNG: `mulberry32(seed)` seeds the traffic sim via the [`SeededRandom`]
//!     capability (mulberry32) instead of importing a closure.
//!   * String-keyed maps → `HashMap<String, _>`; numeric car ids → `HashSet<u64>`.
//!     HashMap iteration order is not guaranteed; per-tick car updates are
//!     snapshot-style (read pre-update state, apply after) so processing order is
//!     irrelevant to the result. Car-id collections are sorted where it makes a
//!     reported aggregate deterministic.
//!   * The grid owns cars/cells via interior maps keyed by id (arena-style),
//!     avoiding `Rc<RefCell>` cycles.
//!   * `optionalLogger?: OptimizationLogger` → `Option<Box<dyn OptimizationLogger>>`.
//!     FLAG: the header suggested `Option<&dyn ..>`, but a station stored behind
//!     `Rc<RefCell<dyn DESStation>>` must be `'static`, so the logger is owned.
//!   * throwing precondition guards → `Preconditions` `Result`s `.unwrap()`-ed in
//!     `assert_preconditions` (construction-time invariant → panic).
//!   * `class .. implements Token` → plain payload structs emitted as `Rc<dyn Any>`
//!     (the Rust station framework has no `Token` trait; tokens are `Rc<dyn Any>`).

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{
    DESStation, StationCore, StationRef, DEFAULT_CHANNEL,
};
use crate::des::general::des_base::validation::{intrinsic_check, ValidationCheck};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// =============================================================================
// Logger
// =============================================================================

/// Log severity (TS `'trace' | 'debug' | 'info' | 'warn' | 'error'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// A structured log event (TS `{kind; level?; [key]: unknown}`). The open-ended
/// extra payload becomes a stringified `fields` map.
#[derive(Clone, Debug, Default)]
pub struct LogEvent {
    pub kind: String,
    pub level: Option<LogLevel>,
    pub fields: HashMap<String, String>,
}

/// Optimisation logger sink (TS `interface OptimizationLogger`).
pub trait OptimizationLogger {
    fn log(&self, event: LogEvent);
}

// =============================================================================
// Max-flow optimization by augmenting-path DES ticks.
// =============================================================================

/// A directed capacitated edge (TS `interface FlowEdge`).
#[derive(Clone, Debug, PartialEq)]
pub struct FlowEdge {
    pub from: usize,
    pub to: usize,
    pub capacity: f64,
    pub name: Option<String>,
}

/// Max-flow problem parameters (TS `interface MaxFlowParams`).
#[derive(Clone, Debug)]
pub struct MaxFlowParams {
    pub num_nodes: usize,
    pub source: usize,
    pub sink: usize,
    pub edges: Vec<FlowEdge>,
    pub max_augmentations: Option<usize>,
    pub node_coordinates: Option<Vec<(f64, f64)>>,
    pub node_names: Option<Vec<String>>,
}

/// An edge with realised flow + residual (TS `interface FlowEdgeResult`).
#[derive(Clone, Debug, PartialEq)]
pub struct FlowEdgeResult {
    pub from: usize,
    pub to: usize,
    pub capacity: f64,
    pub name: Option<String>,
    pub flow: f64,
    pub residual: f64,
}

/// One augmentation row (TS `interface MaxFlowTraceRow`).
#[derive(Clone, Debug, PartialEq)]
pub struct MaxFlowTraceRow {
    pub iter: usize,
    pub path_nodes: Vec<usize>,
    pub path_edges: Vec<usize>,
    pub bottleneck: f64,
    pub value: f64,
}

/// The residual min-cut (TS `interface MaxFlowMinCut`).
#[derive(Clone, Debug)]
pub struct MaxFlowMinCut {
    pub source_side: Vec<usize>,
    pub sink_side: Vec<usize>,
    pub cut_edges: Vec<usize>,
    pub capacity: f64,
}

/// Full max-flow result (TS `interface MaxFlowResult`).
#[derive(Clone, Debug)]
pub struct MaxFlowResult {
    pub params: MaxFlowParams,
    pub max_flow: f64,
    pub edge_flows: Vec<FlowEdgeResult>,
    pub min_cut: MaxFlowMinCut,
    pub trace: Vec<MaxFlowTraceRow>,
    pub validation: Vec<ValidationCheck>,
}

/// A directed residual move along an original edge (TS private `interface
/// ResidualStep`). `dir` is `+1` (forward) or `-1` (backward).
#[derive(Clone, Copy, Debug, PartialEq)]
struct ResidualStep {
    edge: usize,
    dir: i32,
}

/// Token carrying one augmentation row (TS `class AugmentingPathToken implements
/// Token`).
#[derive(Clone, Debug)]
pub struct AugmentingPathToken {
    pub row: MaxFlowTraceRow,
}

struct ParentEntry {
    prev: isize,
    step: ResidualStep,
}

struct FoundPath {
    nodes: Vec<usize>,
    steps: Vec<ResidualStep>,
}

/// Max-flow optimisation station (TS `class MaxFlowOptimizationStation extends
/// DESStation`).
pub struct MaxFlowOptimizationStation {
    core: StationCore,
    pub params: MaxFlowParams,
    logger: Option<Box<dyn OptimizationLogger>>,
    flow: Vec<f64>,
    pub trace: Vec<MaxFlowTraceRow>,
    done: bool,
    value: f64,
}

impl MaxFlowOptimizationStation {
    pub fn new(params: MaxFlowParams, logger: Option<Box<dyn OptimizationLogger>>) -> Self {
        let flow = vec![0.0; params.edges.len()];
        let mut st = MaxFlowOptimizationStation {
            core: StationCore::new("max-flow"),
            params,
            logger,
            flow,
            trace: Vec::new(),
            done: false,
            value: 0.0,
        };
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "max-flow-capacity-feasible",
                |s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<MaxFlowOptimizationStation>()
                        .unwrap();
                    st.edge_flows()
                        .iter()
                        .all(|e| e.flow >= -1e-8 && e.flow <= e.capacity + 1e-8)
                },
                Some("0 <= flow <= capacity on every edge".to_string()),
                None,
                Some("max-flow".to_string()),
                None,
            )
            .boxed(),
        );
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "max-flow-conservation",
                |s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<MaxFlowOptimizationStation>()
                        .unwrap();
                    st.flow_conservation_ok()
                },
                Some("inflow equals outflow at every transshipment node".to_string()),
                None,
                Some("max-flow".to_string()),
                None,
            )
            .boxed(),
        );
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "max-flow-min-cut-tight",
                |s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<MaxFlowOptimizationStation>()
                        .unwrap();
                    (st.current_value() - st.min_cut().capacity).abs() <= 1e-7
                },
                Some("max-flow value equals residual min-cut capacity".to_string()),
                Some(Box::new(|s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<MaxFlowOptimizationStation>()
                        .unwrap();
                    format!(
                        "flow={:.6} cut={:.6}",
                        st.current_value(),
                        st.min_cut().capacity
                    )
                })),
                Some("max-flow".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    pub fn current_value(&self) -> f64 {
        self.value
    }

    pub fn edge_flows(&self) -> Vec<FlowEdgeResult> {
        self.params
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| FlowEdgeResult {
                from: e.from,
                to: e.to,
                capacity: e.capacity,
                name: e.name.clone(),
                flow: self.flow[i],
                residual: e.capacity - self.flow[i],
            })
            .collect()
    }

    pub fn result(&self, validation: Vec<ValidationCheck>) -> MaxFlowResult {
        MaxFlowResult {
            params: self.params.clone(),
            max_flow: self.value,
            edge_flows: self.edge_flows(),
            min_cut: self.min_cut(),
            trace: self.trace.clone(),
            validation,
        }
    }

    pub fn min_cut(&self) -> MaxFlowMinCut {
        let seen = self.residual_reachable();
        let mut source_side: Vec<usize> = Vec::new();
        let mut sink_side: Vec<usize> = Vec::new();
        for (v, is_seen) in seen.iter().enumerate().take(self.params.num_nodes) {
            if *is_seen {
                source_side.push(v);
            } else {
                sink_side.push(v);
            }
        }
        let mut cut_edges: Vec<usize> = Vec::new();
        let mut capacity = 0.0;
        for i in 0..self.params.edges.len() {
            let e = &self.params.edges[i];
            if seen[e.from] && !seen[e.to] {
                cut_edges.push(i);
                capacity += e.capacity;
            }
        }
        MaxFlowMinCut {
            source_side,
            sink_side,
            cut_edges,
            capacity,
        }
    }

    fn flow_conservation_ok(&self) -> bool {
        let mut balance = vec![0.0_f64; self.params.num_nodes];
        for i in 0..self.params.edges.len() {
            let e = &self.params.edges[i];
            balance[e.from] -= self.flow[i];
            balance[e.to] += self.flow[i];
        }
        for v in 0..balance.len() {
            if v == self.params.source || v == self.params.sink {
                continue;
            }
            if balance[v].abs() > 1e-7 {
                return false;
            }
        }
        (balance[self.params.sink] - self.value).abs() <= 1e-7
            && (balance[self.params.source] + self.value).abs() <= 1e-7
    }

    fn residual_capacity(&self, step: &ResidualStep) -> f64 {
        let e = &self.params.edges[step.edge];
        if step.dir == 1 {
            e.capacity - self.flow[step.edge]
        } else {
            self.flow[step.edge]
        }
    }

    fn neighbors(&self, u: usize) -> Vec<(usize, ResidualStep)> {
        let mut out: Vec<(usize, ResidualStep)> = Vec::new();
        for i in 0..self.params.edges.len() {
            let e = &self.params.edges[i];
            if e.from == u && e.capacity - self.flow[i] > 1e-9 {
                out.push((e.to, ResidualStep { edge: i, dir: 1 }));
            }
            if e.to == u && self.flow[i] > 1e-9 {
                out.push((e.from, ResidualStep { edge: i, dir: -1 }));
            }
        }
        out
    }

    fn find_augmenting_path(&self) -> Option<FoundPath> {
        let n = self.params.num_nodes;
        let mut parent: Vec<Option<ParentEntry>> = (0..n).map(|_| None).collect();
        let mut q: Vec<usize> = vec![self.params.source];
        parent[self.params.source] = Some(ParentEntry {
            prev: -1,
            step: ResidualStep { edge: 0, dir: 1 },
        });
        let mut qi = 0;
        while qi < q.len() {
            let u = q[qi];
            qi += 1;
            if u == self.params.sink {
                break;
            }
            for nb in self.neighbors(u) {
                if parent[nb.0].is_some() {
                    continue;
                }
                parent[nb.0] = Some(ParentEntry {
                    prev: u as isize,
                    step: nb.1,
                });
                q.push(nb.0);
            }
        }
        parent[self.params.sink].as_ref()?;
        let mut nodes: Vec<usize> = Vec::new();
        let mut steps: Vec<ResidualStep> = Vec::new();
        let mut cur = self.params.sink;
        while cur != self.params.source {
            let p = match &parent[cur] {
                Some(p) => p,
                None => return None,
            };
            nodes.push(cur);
            steps.push(p.step);
            cur = p.prev as usize;
        }
        nodes.push(self.params.source);
        nodes.reverse();
        steps.reverse();
        Some(FoundPath { nodes, steps })
    }

    fn residual_reachable(&self) -> Vec<bool> {
        let mut seen = vec![false; self.params.num_nodes];
        let mut q: Vec<usize> = vec![self.params.source];
        seen[self.params.source] = true;
        let mut qi = 0;
        while qi < q.len() {
            let u = q[qi];
            qi += 1;
            for nb in self.neighbors(u) {
                if seen[nb.0] {
                    continue;
                }
                seen[nb.0] = true;
                q.push(nb.0);
            }
        }
        seen
    }
}

impl DESStation for MaxFlowOptimizationStation {
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
        let p = &self.params;
        let m = "MaxFlowOptimizationStation";
        Preconditions::integer_in_range(m, "numNodes", p.num_nodes as f64, 2.0, 10000.0).unwrap();
        Preconditions::integer_in_range(
            m,
            "source",
            p.source as f64,
            0.0,
            (p.num_nodes - 1) as f64,
        )
        .unwrap();
        Preconditions::integer_in_range(m, "sink", p.sink as f64, 0.0, (p.num_nodes - 1) as f64)
            .unwrap();
        Preconditions::check(
            m,
            "sink",
            "differ from source",
            p.sink != p.source,
            Some(p.sink.to_string()),
        )
        .unwrap();
        Preconditions::non_empty(m, "edges", &p.edges).unwrap();
        for i in 0..p.edges.len() {
            let e = &p.edges[i];
            Preconditions::integer_in_range(
                m,
                &format!("edges[{i}].from"),
                e.from as f64,
                0.0,
                (p.num_nodes - 1) as f64,
            )
            .unwrap();
            Preconditions::integer_in_range(
                m,
                &format!("edges[{i}].to"),
                e.to as f64,
                0.0,
                (p.num_nodes - 1) as f64,
            )
            .unwrap();
            Preconditions::check(
                m,
                &format!("edges[{i}]"),
                "not be a self-loop",
                e.from != e.to,
                None,
            )
            .unwrap();
            Preconditions::non_negative(m, &format!("edges[{i}].capacity"), e.capacity).unwrap();
        }
        if let Some(ma) = p.max_augmentations {
            Preconditions::integer_in_range(
                m,
                "maxAugmentations",
                ma as f64,
                1.0,
                9_007_199_254_740_991.0,
            )
            .unwrap();
        }
        if let Some(nc) = &p.node_coordinates {
            Preconditions::length_eq(m, "nodeCoordinates", nc, p.num_nodes).unwrap();
        }
        if let Some(nn) = &p.node_names {
            Preconditions::length_eq(m, "nodeNames", nn, p.num_nodes).unwrap();
        }
    }

    fn has_work(&self) -> bool {
        !self.done
    }

    fn run_time_step(&mut self) {
        if self.done {
            return;
        }
        let max_aug = self.params.max_augmentations.unwrap_or(usize::MAX);
        if self.trace.len() >= max_aug {
            self.done = true;
            return;
        }
        let path = match self.find_augmenting_path() {
            Some(p) => p,
            None => {
                self.done = true;
                return;
            }
        };
        let mut bottleneck = f64::INFINITY;
        for step in &path.steps {
            bottleneck = bottleneck.min(self.residual_capacity(step));
        }
        for step in &path.steps {
            self.flow[step.edge] += step.dir as f64 * bottleneck;
        }
        self.value += bottleneck;
        let row = MaxFlowTraceRow {
            iter: self.trace.len() + 1,
            path_nodes: path.nodes.clone(),
            path_edges: path.steps.iter().map(|s| s.edge).collect(),
            bottleneck,
            value: self.value,
        };
        self.trace.push(row.clone());
        if let Some(logger) = &self.logger {
            let mut fields = HashMap::new();
            fields.insert("iter".to_string(), row.iter.to_string());
            fields.insert("bottleneck".to_string(), row.bottleneck.to_string());
            fields.insert("value".to_string(), row.value.to_string());
            logger.log(LogEvent {
                kind: "max-flow-augment".to_string(),
                level: Some(LogLevel::Info),
                fields,
            });
        }
        self.core_mut()
            .emit(Rc::new(AugmentingPathToken { row }), DEFAULT_CHANNEL);
    }
}

/// Run a max-flow optimisation to completion (TS `runMaxFlow`).
pub fn run_max_flow(
    params: MaxFlowParams,
    logger: Option<Box<dyn OptimizationLogger>>,
) -> MaxFlowResult {
    let max_ticks = params
        .max_augmentations
        .unwrap_or(params.edges.len() * params.num_nodes + 1)
        + 2;
    let station = Rc::new(RefCell::new(MaxFlowOptimizationStation::new(
        params, logger,
    )));
    let summary = run_iterative_des(
        vec![station.clone() as StationRef],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_ticks),
            ..Default::default()
        },
    );
    let checks = summary.validation.unwrap_or_default();
    let result = station.borrow().result(checks);
    result
}

// =============================================================================
// Continuous-time traffic flow on a stationary grid.
// =============================================================================

/// Node role in the road network (TS `type TrafficNodeKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrafficNodeKind {
    Source,
    Intersection,
    Sink,
}

/// A network node (TS `interface TrafficNode`).
#[derive(Clone, Debug)]
pub struct TrafficNode {
    pub id: String,
    pub kind: TrafficNodeKind,
    pub x: f64,
    pub y: f64,
}

/// A directed lane (TS `interface TrafficLane`).
#[derive(Clone, Debug)]
pub struct TrafficLane {
    pub id: String,
    pub from: String,
    pub to: String,
    pub length_m: f64,
    pub speed_limit_mps: f64,
    pub capacity: Option<usize>,
}

/// One phase of a signal cycle (TS `interface TrafficSignalPhase`).
#[derive(Clone, Debug)]
pub struct TrafficSignalPhase {
    pub name: String,
    pub green_lanes: Vec<String>,
    pub duration_sec: f64,
}

/// A traffic signal at an intersection (TS `interface TrafficSignal`).
#[derive(Clone, Debug)]
pub struct TrafficSignal {
    pub node_id: String,
    pub phases: Vec<TrafficSignalPhase>,
    pub offset_sec: Option<f64>,
}

/// A spawn point (TS `interface TrafficSource`).
#[derive(Clone, Debug)]
pub struct TrafficSource {
    pub id: String,
    pub node_id: String,
    pub rate_per_min: f64,
    pub destination_sink_ids: Option<Vec<String>>,
}

/// A drain point (TS `interface TrafficSink`).
#[derive(Clone, Debug)]
pub struct TrafficSink {
    pub id: String,
    pub node_id: String,
}

/// The road network (TS `interface TrafficNetwork`).
#[derive(Clone, Debug)]
pub struct TrafficNetwork {
    pub nodes: Vec<TrafficNode>,
    pub lanes: Vec<TrafficLane>,
    pub signals: Option<Vec<TrafficSignal>>,
    pub sources: Vec<TrafficSource>,
    pub sinks: Vec<TrafficSink>,
}

/// A pre-scheduled trip (TS `interface TrafficScheduledTrip`). Carried for parity
/// with the source interface; the simulation does not consume it.
#[derive(Clone, Debug)]
pub struct TrafficScheduledTrip {
    pub depart_sec: f64,
    pub source_id: String,
    pub destination_sink_id: String,
}

/// Simulation parameters (TS `interface TrafficParams`).
#[derive(Clone, Debug)]
pub struct TrafficParams {
    pub builtin: Option<String>,
    pub network: Option<TrafficNetwork>,
    pub duration_sec: f64,
    pub dt_sec: f64,
    pub seed: f64,
    pub max_cars: usize,
    pub car_length_m: Option<f64>,
    pub car_width_m: Option<f64>,
    pub lane_width_m: Option<f64>,
    pub min_gap_m: Option<f64>,
    pub max_accel_mps2: Option<f64>,
    pub max_decel_mps2: Option<f64>,
    pub max_jerk_mps3: Option<f64>,
    pub reaction_time_sec: Option<f64>,
    pub time_headway_sec: Option<f64>,
    pub grid_cell_size_m: Option<f64>,
    pub grid_look_ahead_m: Option<f64>,
    pub spawn_rate_multiplier: Option<f64>,
    pub scheduled_trips: Option<Vec<TrafficScheduledTrip>>,
}

/// An immutable per-tick car snapshot (TS `interface TrafficCarSnapshot`).
#[derive(Clone, Debug)]
pub struct TrafficCarSnapshot {
    pub id: u64,
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
    pub grid_cell_ids: Vec<String>,
    pub grid_cell_count: usize,
    pub leader_id: Option<u64>,
    pub leader_gap_m: Option<f64>,
}

/// One row of the per-tick trace (TS `interface TrafficTraceRow`).
#[derive(Clone, Debug)]
pub struct TrafficTraceRow {
    pub tick: usize,
    pub time_sec: f64,
    pub active_cars: usize,
    pub entered: usize,
    pub exited: usize,
    pub mean_speed_mps: f64,
    pub mean_travel_time_sec: f64,
    pub queue_length: usize,
    pub lane_occupancy: HashMap<String, usize>,
    pub active_grid_cells: usize,
    pub signal_phases: HashMap<String, String>,
    pub cars: Vec<TrafficCarSnapshot>,
}

/// Spatial-grid summary (TS `interface TrafficCellStats`).
#[derive(Clone, Debug)]
pub struct TrafficCellStats {
    pub cell_size_m: f64,
    pub lane_width_m: f64,
    pub car_width_m: f64,
    pub active_cells: usize,
    pub created_cell_stations: usize,
    pub max_cell_occupancy: usize,
}

/// Full traffic-simulation result (TS `interface TrafficResult`).
#[derive(Clone, Debug)]
pub struct TrafficResult {
    pub params: TrafficParams,
    pub network: TrafficNetwork,
    pub trace: Vec<TrafficTraceRow>,
    pub final_cars: Vec<TrafficCarSnapshot>,
    pub entered: usize,
    pub exited: usize,
    pub dropped: usize,
    pub mean_travel_time_sec: f64,
    pub mean_speed_mps: f64,
    pub max_active_cars: usize,
    pub cell_stats: TrafficCellStats,
    pub validation: Vec<ValidationCheck>,
}

/// A past kinematic sample for reaction-delay perception (TS private
/// `interface TrafficKinematicSample`).
#[derive(Clone, Debug)]
struct TrafficKinematicSample {
    time_sec: f64,
    lane_id: String,
    position_m: f64,
    speed_mps: f64,
    #[allow(dead_code)]
    acceleration_mps2: f64,
}

/// The mutable car state owned by the grid (TS private `interface TrafficCar`).
#[derive(Clone, Debug)]
struct TrafficCar {
    id: u64,
    lane_id: String,
    position_m: f64,
    speed_mps: f64,
    acceleration_mps2: f64,
    jerk_mps3: f64,
    target_acceleration_mps2: f64,
    route: Vec<String>,
    route_index: usize,
    destination_sink_id: String,
    created_at_sec: f64,
    wait_sec: f64,
    grid_cell_ids: Vec<String>,
    grid_cell_count: usize,
    leader_id: Option<u64>,
    leader_gap_m: Option<f64>,
    history: Vec<TrafficKinematicSample>,
}

/// Token carrying a car snapshot (TS `class CarToken implements Token`).
#[derive(Clone, Debug)]
pub struct CarToken {
    pub car: TrafficCarSnapshot,
}

/// Bounds of one spatial-grid cell (TS private `interface TrafficCellBounds`).
#[derive(Clone, Debug)]
pub struct TrafficCellBounds {
    pub lane_id: String,
    pub longitudinal_index: usize,
    pub lateral_index: usize,
    pub x0_m: f64,
    pub x1_m: f64,
    pub y0_m: f64,
    pub y1_m: f64,
}

/// A spatial-grid occupancy cell station (TS `class TrafficCellStation extends
/// DESStation`). Used purely as an occupancy container; never run by the runner.
pub struct TrafficCellStation {
    core: StationCore,
    pub bounds: TrafficCellBounds,
    pub car_ids: HashSet<u64>,
}

impl TrafficCellStation {
    pub fn new(bounds: TrafficCellBounds) -> Self {
        let id = format!(
            "traffic-cell-{}-{}-{}",
            bounds.lane_id, bounds.longitudinal_index, bounds.lateral_index
        );
        TrafficCellStation {
            core: StationCore::new(id),
            bounds,
            car_ids: HashSet::new(),
        }
    }
    pub fn clear_occupancy(&mut self) {
        self.car_ids.clear();
    }
    pub fn occupy(&mut self, car_id: u64) {
        self.car_ids.insert(car_id);
    }
}

impl DESStation for TrafficCellStation {
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
        false
    }
    fn run_time_step(&mut self) {}
}

/// Spatial index built each tick (TS private `interface TrafficSpatialIndex`).
struct TrafficSpatialIndex {
    by_cell: HashMap<String, HashSet<u64>>,
    #[allow(dead_code)]
    cell_ids_by_car: HashMap<u64, Vec<String>>,
    #[allow(dead_code)]
    active_cell_count: usize,
}

/// Optional grid construction options (TS private `interface TrafficOptions`).
#[derive(Default)]
pub struct TrafficOptions {
    pub logger: Option<Box<dyn OptimizationLogger>>,
}

/// Kinematic update produced by [`TrafficGridStation::next_kinematics`].
struct KinResult {
    speed: f64,
    position: f64,
    acceleration: f64,
    jerk: f64,
    target_acceleration: f64,
    leader_id: Option<u64>,
    leader_gap_m: Option<f64>,
}

struct CarUpdate {
    car_id: u64,
    kin: KinResult,
}

/// The stationary road grid that owns lanes, cells, and cars (TS `class
/// TrafficGridStation extends DESStation`).
pub struct TrafficGridStation {
    core: StationCore,
    params: TrafficParams,
    options: TrafficOptions,
    network: TrafficNetwork,
    nodes: HashMap<String, TrafficNode>,
    lanes: HashMap<String, TrafficLane>,
    signal_by_node: HashMap<String, TrafficSignal>,
    #[allow(dead_code)]
    outgoing: HashMap<String, Vec<String>>,
    routes: HashMap<String, Vec<String>>,
    source_accumulators: HashMap<String, f64>,
    cars: HashMap<u64, TrafficCar>,
    cell_stations: HashMap<String, TrafficCellStation>,
    active_cell_ids: HashSet<String>,
    rng: SeededRandom,
    next_car_id: u64,
    tick: usize,
    entered: usize,
    exited: usize,
    dropped: usize,
    travel_time_sum: f64,
    speed_sum: f64,
    speed_samples: usize,
    max_active_cars: usize,
    max_cell_occupancy: usize,
    pub trace: Vec<TrafficTraceRow>,
}

impl TrafficGridStation {
    pub fn new(params: TrafficParams, options: TrafficOptions) -> Self {
        let network = params
            .network
            .clone()
            .unwrap_or_else(build_five_intersection_traffic_network);
        let rng = mulberry32(params.seed as u32);
        let mut nodes: HashMap<String, TrafficNode> = HashMap::new();
        for node in &network.nodes {
            nodes.insert(node.id.clone(), node.clone());
        }
        let mut lanes: HashMap<String, TrafficLane> = HashMap::new();
        let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
        for lane in &network.lanes {
            lanes.insert(lane.id.clone(), lane.clone());
            outgoing
                .entry(lane.from.clone())
                .or_default()
                .push(lane.id.clone());
        }
        let mut signal_by_node: HashMap<String, TrafficSignal> = HashMap::new();
        for signal in network.signals.iter().flatten() {
            signal_by_node.insert(signal.node_id.clone(), signal.clone());
        }
        let mut source_accumulators: HashMap<String, f64> = HashMap::new();
        for source in &network.sources {
            source_accumulators.insert(source.id.clone(), 0.0);
        }

        let mut st = TrafficGridStation {
            core: StationCore::new("traffic-flow-grid"),
            params,
            options,
            network,
            nodes,
            lanes,
            signal_by_node,
            outgoing,
            routes: HashMap::new(),
            source_accumulators,
            cars: HashMap::new(),
            cell_stations: HashMap::new(),
            active_cell_ids: HashSet::new(),
            rng,
            next_car_id: 1,
            tick: 0,
            entered: 0,
            exited: 0,
            dropped: 0,
            travel_time_sum: 0.0,
            speed_sum: 0.0,
            speed_samples: 0,
            max_active_cars: 0,
            max_cell_occupancy: 0,
            trace: Vec::new(),
        };
        st.assert_preconditions();
        st.precompute_routes();

        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "traffic-active-under-cap",
                |s| {
                    let st = s.as_any().downcast_ref::<TrafficGridStation>().unwrap();
                    st.max_active_cars <= st.params.max_cars
                },
                Some("active car count never exceeds maxCars".to_string()),
                Some(Box::new(|s| {
                    let st = s.as_any().downcast_ref::<TrafficGridStation>().unwrap();
                    st.max_active_cars.to_string()
                })),
                Some("traffic-flow".to_string()),
                None,
            )
            .boxed(),
        );
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "traffic-conservation",
                |s| {
                    let st = s.as_any().downcast_ref::<TrafficGridStation>().unwrap();
                    st.entered == st.exited + st.cars.len()
                },
                Some("entered = exited + active".to_string()),
                Some(Box::new(|s| {
                    let st = s.as_any().downcast_ref::<TrafficGridStation>().unwrap();
                    format!(
                        "entered={} exited={} active={}",
                        st.entered,
                        st.exited,
                        st.cars.len()
                    )
                })),
                Some("traffic-flow".to_string()),
                None,
            )
            .boxed(),
        );
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "traffic-no-collisions",
                |s| {
                    let st = s.as_any().downcast_ref::<TrafficGridStation>().unwrap();
                    st.minimum_headway() >= -1e-7
                },
                Some("same-lane cars remain separated by carLength+minGap".to_string()),
                Some(Box::new(|s| {
                    let st = s.as_any().downcast_ref::<TrafficGridStation>().unwrap();
                    format!("{:.6}", st.minimum_headway())
                })),
                Some("traffic-flow".to_string()),
                None,
            )
            .boxed(),
        );
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "traffic-grid-cell-size",
                |s| {
                    let st = s.as_any().downcast_ref::<TrafficGridStation>().unwrap();
                    st.grid_cell_size_m() <= 0.3048 + 1e-12
                },
                Some(
                    "default/selected spatial grid cells are at most about one foot on a side"
                        .to_string(),
                ),
                Some(Box::new(|s| {
                    let st = s.as_any().downcast_ref::<TrafficGridStation>().unwrap();
                    format!("{:.4} m", st.grid_cell_size_m())
                })),
                Some("traffic-flow".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    pub fn result(&self, validation: Vec<ValidationCheck>) -> TrafficResult {
        let final_cars = self.snap_cars();
        let mean_travel = if self.exited > 0 {
            self.travel_time_sum / self.exited as f64
        } else {
            0.0
        };
        let mean_speed = if self.speed_samples > 0 {
            self.speed_sum / self.speed_samples as f64
        } else {
            0.0
        };
        TrafficResult {
            params: self.params.clone(),
            network: self.network.clone(),
            trace: self.trace.clone(),
            final_cars,
            entered: self.entered,
            exited: self.exited,
            dropped: self.dropped,
            mean_travel_time_sec: mean_travel,
            mean_speed_mps: mean_speed,
            max_active_cars: self.max_active_cars,
            cell_stats: TrafficCellStats {
                cell_size_m: self.grid_cell_size_m(),
                lane_width_m: self.lane_width_m(),
                car_width_m: self.car_width_m(),
                active_cells: self.active_cell_ids.len(),
                created_cell_stations: self.cell_stations.len(),
                max_cell_occupancy: self.max_cell_occupancy,
            },
            validation,
        }
    }

    fn precompute_routes(&mut self) {
        let sources = self.network.sources.clone();
        let sinks_all: Vec<String> = self.network.sinks.iter().map(|s| s.id.clone()).collect();
        for source in &sources {
            let sink_ids = source
                .destination_sink_ids
                .clone()
                .unwrap_or_else(|| sinks_all.clone());
            for sink_id in &sink_ids {
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

    fn spawn_cars(&mut self, time_sec: f64) {
        let dt = self.params.dt_sec;
        let mult = self.params.spawn_rate_multiplier.unwrap_or(1.0);
        let sources = self.network.sources.clone();
        for source in &sources {
            let expected = source.rate_per_min * mult * dt / 60.0;
            let mut acc = self
                .source_accumulators
                .get(&source.id)
                .copied()
                .unwrap_or(0.0)
                + expected;
            let count = acc.floor();
            acc -= count;
            self.source_accumulators.insert(source.id.clone(), acc);
            let count = count.max(0.0) as usize;
            for _ in 0..count {
                self.try_spawn_from_source(source, time_sec);
            }
        }
    }

    fn try_spawn_from_source(&mut self, source: &TrafficSource, time_sec: f64) {
        if self.cars.len() >= self.params.max_cars {
            self.dropped += 1;
            return;
        }
        let sink_ids = source
            .destination_sink_ids
            .clone()
            .unwrap_or_else(|| self.network.sinks.iter().map(|s| s.id.clone()).collect());
        if sink_ids.is_empty() {
            self.dropped += 1;
            return;
        }
        let mut idx = (self.rng.next_float() * sink_ids.len() as f64).floor() as usize;
        if idx >= sink_ids.len() {
            idx = sink_ids.len() - 1;
        }
        let sink_id = sink_ids[idx].clone();
        let route = self
            .routes
            .get(&format!("{}->{}", source.id, sink_id))
            .cloned();
        let route = match route {
            Some(r) if !r.is_empty() => r,
            _ => {
                self.dropped += 1;
                return;
            }
        };
        let lane = self.lane(&route[0]);
        if !self.can_enter_lane(&lane.id, None) {
            self.dropped += 1;
            return;
        }
        let mut car = TrafficCar {
            id: self.next_car_id,
            lane_id: lane.id.clone(),
            position_m: 0.0,
            speed_mps: 2.0_f64.min(lane.speed_limit_mps),
            acceleration_mps2: 0.0,
            jerk_mps3: 0.0,
            target_acceleration_mps2: 0.0,
            route: route.clone(),
            route_index: 0,
            destination_sink_id: sink_id,
            created_at_sec: time_sec,
            wait_sec: 0.0,
            grid_cell_ids: Vec::new(),
            grid_cell_count: 0,
            leader_id: None,
            leader_gap_m: None,
            history: Vec::new(),
        };
        self.next_car_id += 1;
        let horizon = self.reaction_time_sec() + 2.0 * self.params.dt_sec + 1.0;
        Self::push_history(&mut car, time_sec, horizon);
        let id = car.id;
        let snap = to_car_snapshot(&car);
        self.cars.insert(id, car);
        self.entered += 1;
        self.core_mut()
            .emit(Rc::new(CarToken { car: snap }), DEFAULT_CHANNEL);
    }

    fn advance_cars(&mut self, time_sec: f64) {
        let dt = self.params.dt_sec;
        let spatial = self.rebuild_spatial_index();
        let lane_groups = self.cars_by_lane();
        let mut updates: Vec<CarUpdate> = Vec::new();
        for (lane_id, mut ids) in lane_groups {
            let lane = self.lane(&lane_id);
            ids.sort_by(|&a, &b| {
                let pa = self.cars.get(&a).map(|c| c.position_m).unwrap_or(0.0);
                let pb = self.cars.get(&b).map(|c| c.position_m).unwrap_or(0.0);
                pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
            });
            for i in 0..ids.len() {
                let car_id = ids[i];
                let car = match self.cars.get(&car_id) {
                    Some(c) => c.clone(),
                    None => continue,
                };
                let sorted_leader_id = if i == 0 { None } else { Some(ids[i - 1]) };
                let grid_leader = self.find_leader_ahead_from_grid(&car, &lane, &spatial);
                let leader_id = grid_leader.or(sorted_leader_id);
                let leader = leader_id.and_then(|id| self.cars.get(&id).cloned());
                let kin = self.next_kinematics(&car, &lane, leader.as_ref(), time_sec);
                updates.push(CarUpdate { car_id, kin });
            }
        }
        let horizon = self.reaction_time_sec() + 2.0 * self.params.dt_sec + 1.0;
        for u in updates {
            if !self.cars.contains_key(&u.car_id) {
                continue;
            }
            {
                let car = self.cars.get_mut(&u.car_id).unwrap();
                car.jerk_mps3 = u.kin.jerk;
                car.acceleration_mps2 = u.kin.acceleration;
                car.target_acceleration_mps2 = u.kin.target_acceleration;
                car.leader_id = u.kin.leader_id;
                car.leader_gap_m = u.kin.leader_gap_m;
                car.speed_mps = u.kin.speed;
                car.position_m = u.kin.position;
                if u.kin.speed < 0.5 {
                    car.wait_sec += dt;
                }
            }
            self.speed_sum += u.kin.speed;
            self.speed_samples += 1;
            self.handle_lane_end(u.car_id, time_sec + dt);
            if let Some(car) = self.cars.get_mut(&u.car_id) {
                Self::push_history(car, time_sec + dt, horizon);
            }
        }
    }

    fn next_kinematics(
        &self,
        car: &TrafficCar,
        lane: &TrafficLane,
        leader: Option<&TrafficCar>,
        time_sec: f64,
    ) -> KinResult {
        let dt = self.params.dt_sec;
        let vehicle_space = self.vehicle_space();
        let barrier = self.stop_barrier_position(car, lane, time_sec);
        let leader_position = leader.map(|l| l.position_m).unwrap_or(f64::INFINITY);
        let current_leader_gap = if leader_position.is_finite() {
            leader_position - car.position_m - vehicle_space
        } else {
            f64::INFINITY
        };
        let barrier_gap = match barrier {
            None => f64::INFINITY,
            Some(b) => b - car.position_m - vehicle_space,
        };
        let use_barrier = barrier_gap <= current_leader_gap;

        let delayed_leader =
            leader.map(|l| self.perceived_sample(l, time_sec - self.reaction_time_sec()));
        // perceivedLeader: the delayed sample if it stayed in this lane, else the
        // (live) leader car.
        let perceived_leader: Option<(f64, f64)> = match (&delayed_leader, leader) {
            (Some(dl), _) if dl.lane_id == car.lane_id => Some((dl.position_m, dl.speed_mps)),
            (_, Some(l)) => Some((l.position_m, l.speed_mps)),
            _ => None,
        };
        // perceived: (position, speed, leaderId). id is always the live leader's.
        let perceived: (f64, f64, Option<u64>) = if use_barrier {
            (barrier.unwrap_or(f64::INFINITY), 0.0, None)
        } else if let Some((pos, spd)) = perceived_leader {
            (pos, spd, leader.map(|l| l.id))
        } else {
            (f64::INFINITY, lane.speed_limit_mps, None)
        };

        let perceived_gap = 0.05_f64.max(perceived.0 - car.position_m - vehicle_space);
        let max_accel = self.max_accel_mps2();
        let max_decel = self.max_decel_mps2();
        let v = 0.0_f64.max(car.speed_mps);
        let v0 = lane.speed_limit_mps;
        let time_headway = self.time_headway_sec() + self.reaction_time_sec();
        let closing_term =
            0.0_f64.max(v * (v - perceived.1) / (2.0 * (max_accel * max_decel).sqrt()));
        let desired_gap = vehicle_space + v * time_headway + closing_term;
        let free_road = {
            let ratio = (v / 1e-9_f64.max(v0)).min(2.0);
            1.0 - ratio * ratio
        };
        let interaction = if perceived.0.is_finite() {
            let r = desired_gap / perceived_gap;
            r * r
        } else {
            0.0
        };
        let target_acceleration =
            clamp(max_accel * (free_road - interaction), -max_decel, max_accel);
        let max_jerk_step = self.max_jerk_mps3() * dt;
        let acceleration = clamp(
            car.acceleration_mps2
                + clamp(
                    target_acceleration - car.acceleration_mps2,
                    -max_jerk_step,
                    max_jerk_step,
                ),
            -max_decel,
            max_accel,
        );
        let mut speed = clamp(v + acceleration * dt, 0.0, v0);
        let mut position = car.position_m + v * dt + 0.5 * acceleration * dt * dt;

        let hard_limit = self.hard_position_limit(car, lane, leader, barrier);
        if position > hard_limit {
            position = car.position_m.max(hard_limit);
            speed = 0.0_f64.max(speed.min((position - car.position_m) / dt));
        }
        let realized_acceleration = clamp((speed - v) / dt, -max_decel, max_accel);
        let jerk = clamp(
            (realized_acceleration - car.acceleration_mps2) / dt,
            -self.max_jerk_mps3(),
            self.max_jerk_mps3(),
        );
        KinResult {
            speed,
            position,
            acceleration: realized_acceleration,
            jerk,
            target_acceleration,
            leader_id: perceived.2,
            leader_gap_m: if perceived.0.is_finite() {
                Some(0.0_f64.max(perceived_gap))
            } else {
                None
            },
        }
    }

    fn stop_barrier_position(
        &self,
        car: &TrafficCar,
        lane: &TrafficLane,
        time_sec: f64,
    ) -> Option<f64> {
        let next_lane_id = car.route.get(car.route_index + 1)?;
        if self.signal_allows(&lane.id, time_sec) && self.can_enter_lane(next_lane_id, Some(car.id))
        {
            return None;
        }
        Some(lane.length_m)
    }

    fn hard_position_limit(
        &self,
        car: &TrafficCar,
        lane: &TrafficLane,
        leader: Option<&TrafficCar>,
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
            return lane.length_m + 0.0_f64.max(car.speed_mps * self.params.dt_sec);
        }
        car.position_m.max(limit)
    }

    fn handle_lane_end(&mut self, car_id: u64, time_sec: f64) {
        let mut car = match self.cars.get(&car_id) {
            Some(c) => c.clone(),
            None => return,
        };
        let mut lane = self.lane(&car.lane_id);
        if car.position_m < lane.length_m - 1e-9 {
            return;
        }
        let mut overshoot = car.position_m - lane.length_m;
        loop {
            if !(car.position_m >= lane.length_m - 1e-9) {
                break;
            }
            let next_lane_id = car.route.get(car.route_index + 1).cloned();
            match next_lane_id {
                None => {
                    self.cars.remove(&car_id);
                    self.exited += 1;
                    self.travel_time_sum += time_sec - car.created_at_sec;
                    return;
                }
                Some(next) => {
                    if !self.signal_allows(&lane.id, time_sec)
                        || !self.can_enter_lane(&next, Some(car_id))
                    {
                        car.position_m = self.blocked_stop_position(&car);
                        car.speed_mps = 0.0;
                        self.cars.insert(car_id, car);
                        return;
                    }
                    car.route_index += 1;
                    car.lane_id = next.clone();
                    lane = self.lane(&car.lane_id);
                    car.position_m = 0.0_f64
                        .max(overshoot)
                        .min(0.0_f64.max(lane.length_m - self.vehicle_space()));
                    car.speed_mps = car.speed_mps.min(lane.speed_limit_mps);
                    overshoot = 0.0_f64.max(car.position_m - lane.length_m);
                    if overshoot <= 1e-9 {
                        self.cars.insert(car_id, car);
                        return;
                    }
                }
            }
        }
        self.cars.insert(car_id, car);
    }

    fn blocked_stop_position(&self, car: &TrafficCar) -> f64 {
        let lane = self.lane(&car.lane_id);
        let vehicle_space = self.vehicle_space();
        let mut safe = 0.0_f64.max(lane.length_m - vehicle_space);
        let mut others: Vec<&TrafficCar> = self
            .cars
            .values()
            .filter(|c| c.id != car.id && c.lane_id == car.lane_id)
            .collect();
        others.sort_by(|a, b| {
            b.position_m
                .partial_cmp(&a.position_m)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for other in others {
            if other.position_m <= safe + vehicle_space {
                safe = safe.min(other.position_m - vehicle_space);
            }
        }
        0.0_f64.max(safe)
    }

    fn signal_allows(&self, incoming_lane_id: &str, time_sec: f64) -> bool {
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
        current_signal_phase(signal, time_sec)
            .green_lanes
            .iter()
            .any(|l| l == incoming_lane_id)
    }

    fn can_enter_lane(&self, lane_id: &str, ignore_car_id: Option<u64>) -> bool {
        let lane = self.lane(lane_id);
        let cars: Vec<&TrafficCar> = self
            .cars
            .values()
            .filter(|c| c.lane_id == lane_id && Some(c.id) != ignore_car_id)
            .collect();
        let cap = lane
            .capacity
            .unwrap_or_else(|| default_lane_capacity(&lane, self.vehicle_space()));
        if cars.len() >= cap {
            return false;
        }
        cars.iter().all(|c| c.position_m >= self.vehicle_space())
    }

    fn rebuild_spatial_index(&mut self) -> TrafficSpatialIndex {
        for cell in self.cell_stations.values_mut() {
            cell.clear_occupancy();
        }
        self.active_cell_ids.clear();
        let mut by_cell: HashMap<String, HashSet<u64>> = HashMap::new();
        let mut cell_ids_by_car: HashMap<u64, Vec<String>> = HashMap::new();
        let mut car_ids: Vec<u64> = self.cars.keys().copied().collect();
        car_ids.sort_unstable();
        for id in car_ids {
            let cell_ids = {
                let car = self.cars.get(&id).unwrap();
                self.occupied_cell_ids(car)
            };
            if let Some(car) = self.cars.get_mut(&id) {
                car.grid_cell_ids = cell_ids.clone();
                car.grid_cell_count = cell_ids.len();
            }
            cell_ids_by_car.insert(id, cell_ids.clone());
            for cell_id in &cell_ids {
                self.ensure_cell_station(cell_id);
                if let Some(station) = self.cell_stations.get_mut(cell_id) {
                    station.occupy(id);
                    self.max_cell_occupancy = self.max_cell_occupancy.max(station.car_ids.len());
                }
                self.active_cell_ids.insert(cell_id.clone());
                by_cell.entry(cell_id.clone()).or_default().insert(id);
            }
        }
        let active_cell_count = self.active_cell_ids.len();
        TrafficSpatialIndex {
            by_cell,
            cell_ids_by_car,
            active_cell_count,
        }
    }

    fn find_leader_ahead_from_grid(
        &self,
        car: &TrafficCar,
        lane: &TrafficLane,
        spatial: &TrafficSpatialIndex,
    ) -> Option<u64> {
        let cell_size = self.grid_cell_size_m();
        let look_ahead = (lane.length_m - car.position_m).min(
            self.params.grid_look_ahead_m.unwrap_or_else(|| {
                60.0_f64.max(
                    car.speed_mps * (self.reaction_time_sec() + 4.0) + 3.0 * self.vehicle_space(),
                )
            }),
        );
        let first = 0.0_f64.max((car.position_m / cell_size).floor()) as usize;
        let last =
            first.max(0.0_f64.max(((car.position_m + look_ahead) / cell_size).floor()) as usize);
        let lateral = self.occupied_lateral_cell_range();
        let mut best: Option<u64> = None;
        let mut best_pos = f64::INFINITY;
        for x in first..=last {
            for y in lateral.0..=lateral.1 {
                if let Some(ids) = spatial.by_cell.get(&self.cell_id(&lane.id, x, y)) {
                    for &id in ids {
                        if id == car.id {
                            continue;
                        }
                        if let Some(other) = self.cars.get(&id) {
                            if other.lane_id != car.lane_id || other.position_m <= car.position_m {
                                continue;
                            }
                            if best.is_none() || other.position_m < best_pos {
                                best = Some(id);
                                best_pos = other.position_m;
                            }
                        }
                    }
                }
            }
            if best.is_some() && best_pos <= (x + 1) as f64 * cell_size {
                break;
            }
        }
        best
    }

    fn cars_by_lane(&self) -> HashMap<String, Vec<u64>> {
        let mut groups: HashMap<String, Vec<u64>> = HashMap::new();
        for car in self.cars.values() {
            groups.entry(car.lane_id.clone()).or_default().push(car.id);
        }
        groups
    }

    fn snapshot(&self, time_sec: f64) -> TrafficTraceRow {
        let mut lane_occupancy: HashMap<String, usize> = HashMap::new();
        for lane in &self.network.lanes {
            lane_occupancy.insert(lane.id.clone(), 0);
        }
        for car in self.cars.values() {
            *lane_occupancy.entry(car.lane_id.clone()).or_insert(0) += 1;
        }
        let mut signal_phases: HashMap<String, String> = HashMap::new();
        for signal in self.network.signals.iter().flatten() {
            signal_phases.insert(
                signal.node_id.clone(),
                current_signal_phase(signal, time_sec).name,
            );
        }
        let cars = self.snap_cars();
        let mean_speed = if !cars.is_empty() {
            cars.iter().map(|c| c.speed_mps).sum::<f64>() / cars.len() as f64
        } else {
            0.0
        };
        let mean_travel = if self.exited > 0 {
            self.travel_time_sum / self.exited as f64
        } else {
            0.0
        };
        let queue_length = cars.iter().filter(|c| c.speed_mps < 0.5).count();
        TrafficTraceRow {
            tick: self.tick,
            time_sec,
            active_cars: cars.len(),
            entered: self.entered,
            exited: self.exited,
            mean_speed_mps: mean_speed,
            mean_travel_time_sec: mean_travel,
            queue_length,
            lane_occupancy,
            active_grid_cells: self.active_cell_ids.len(),
            signal_phases,
            cars,
        }
    }

    fn snap_cars(&self) -> Vec<TrafficCarSnapshot> {
        let mut cars: Vec<&TrafficCar> = self.cars.values().collect();
        cars.sort_by_key(|a| a.id);
        cars.iter().map(|c| to_car_snapshot(c)).collect()
    }

    fn minimum_headway(&self) -> f64 {
        let mut min = f64::INFINITY;
        let groups = self.cars_by_lane();
        for (_lane, mut ids) in groups {
            ids.sort_by(|&a, &b| {
                self.cars[&a]
                    .position_m
                    .partial_cmp(&self.cars[&b].position_m)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for i in 1..ids.len() {
                let gap = self.cars[&ids[i]].position_m
                    - self.cars[&ids[i - 1]].position_m
                    - self.vehicle_space();
                min = min.min(gap);
            }
        }
        if min.is_finite() {
            min
        } else {
            0.0
        }
    }

    fn lane(&self, id: &str) -> TrafficLane {
        match self.lanes.get(id) {
            Some(l) => l.clone(),
            None => panic!("traffic-flow: unknown lane \"{id}\""),
        }
    }

    fn vehicle_space(&self) -> f64 {
        self.params.car_length_m.unwrap_or(4.8) + self.params.min_gap_m.unwrap_or(2.5)
    }
    fn car_length_m(&self) -> f64 {
        self.params.car_length_m.unwrap_or(4.8)
    }
    fn car_width_m(&self) -> f64 {
        self.params.car_width_m.unwrap_or(1.8)
    }
    fn lane_width_m(&self) -> f64 {
        self.params.lane_width_m.unwrap_or(3.7)
    }
    fn grid_cell_size_m(&self) -> f64 {
        self.params.grid_cell_size_m.unwrap_or(0.3048)
    }
    fn max_accel_mps2(&self) -> f64 {
        self.params.max_accel_mps2.unwrap_or(2.2)
    }
    fn max_decel_mps2(&self) -> f64 {
        self.params.max_decel_mps2.unwrap_or(4.0)
    }
    fn max_jerk_mps3(&self) -> f64 {
        self.params.max_jerk_mps3.unwrap_or(6.0)
    }
    fn reaction_time_sec(&self) -> f64 {
        self.params.reaction_time_sec.unwrap_or(0.8)
    }
    fn time_headway_sec(&self) -> f64 {
        self.params.time_headway_sec.unwrap_or(1.1)
    }

    fn occupied_lateral_cell_range(&self) -> (usize, usize) {
        let cell_size = self.grid_cell_size_m();
        let lane_width = self.lane_width_m();
        let car_width = self.car_width_m().min(lane_width);
        let left = 0.0_f64.max((lane_width - car_width) / 2.0);
        let right = lane_width.min(left + car_width);
        (
            0.0_f64.max((left / cell_size).floor()) as usize,
            0.0_f64.max((left.max(right - 1e-9) / cell_size).floor()) as usize,
        )
    }

    fn occupied_cell_ids(&self, car: &TrafficCar) -> Vec<String> {
        let cell_size = self.grid_cell_size_m();
        let rear = 0.0_f64.max(car.position_m - self.car_length_m());
        let front = rear.max(car.position_m);
        let x0 = 0.0_f64.max((rear / cell_size).floor()) as usize;
        let x1 = x0.max(0.0_f64.max((front / cell_size).floor()) as usize);
        let y = self.occupied_lateral_cell_range();
        let mut ids: Vec<String> = Vec::new();
        for x in x0..=x1 {
            for lat in y.0..=y.1 {
                ids.push(self.cell_id(&car.lane_id, x, lat));
            }
        }
        ids
    }

    fn cell_id(&self, lane_id: &str, longitudinal_index: usize, lateral_index: usize) -> String {
        format!("{lane_id}#{longitudinal_index}:{lateral_index}")
    }

    fn ensure_cell_station(&mut self, cell_id: &str) {
        if self.cell_stations.contains_key(cell_id) {
            return;
        }
        let (lane_id, x, y) = self.parse_cell_id(cell_id);
        let cell_size = self.grid_cell_size_m();
        let station = TrafficCellStation::new(TrafficCellBounds {
            lane_id,
            longitudinal_index: x,
            lateral_index: y,
            x0_m: x as f64 * cell_size,
            x1_m: (x + 1) as f64 * cell_size,
            y0_m: y as f64 * cell_size,
            y1_m: (y + 1) as f64 * cell_size,
        });
        self.cell_stations.insert(cell_id.to_string(), station);
    }

    fn parse_cell_id(&self, cell_id: &str) -> (String, usize, usize) {
        let sep = cell_id.rfind('#').expect("traffic-flow: malformed cell id");
        let lane_id = cell_id[..sep].to_string();
        let rest = &cell_id[sep + 1..];
        let mut it = rest.split(':');
        let x: usize = it.next().unwrap().parse().unwrap();
        let y: usize = it.next().unwrap().parse().unwrap();
        (lane_id, x, y)
    }

    fn perceived_sample(&self, car: &TrafficCar, target_time_sec: f64) -> TrafficKinematicSample {
        for i in (0..car.history.len()).rev() {
            if car.history[i].time_sec <= target_time_sec + 1e-12 {
                return car.history[i].clone();
            }
        }
        car.history
            .first()
            .cloned()
            .unwrap_or(TrafficKinematicSample {
                time_sec: target_time_sec,
                lane_id: car.lane_id.clone(),
                position_m: car.position_m,
                speed_mps: car.speed_mps,
                acceleration_mps2: car.acceleration_mps2,
            })
    }

    fn push_history(car: &mut TrafficCar, time_sec: f64, horizon: f64) {
        car.history.push(TrafficKinematicSample {
            time_sec,
            lane_id: car.lane_id.clone(),
            position_m: car.position_m,
            speed_mps: car.speed_mps,
            acceleration_mps2: car.acceleration_mps2,
        });
        while car.history.len() > 2 && car.history[1].time_sec < time_sec - horizon {
            car.history.remove(0);
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

    fn assert_preconditions(&mut self) {
        let m = "TrafficGridStation";
        let p = &self.params;
        Preconditions::check(
            m,
            "network",
            "be provided by builtin or network",
            matches!(p.builtin.as_deref(), Some("five-intersection")) || p.network.is_some(),
            None,
        )
        .unwrap();
        Preconditions::positive(m, "durationSec", p.duration_sec).unwrap();
        Preconditions::positive(m, "dtSec", p.dt_sec).unwrap();
        Preconditions::check(
            m,
            "dtSec",
            "be <= 5 seconds",
            p.dt_sec <= 5.0,
            Some(p.dt_sec.to_string()),
        )
        .unwrap();
        Preconditions::integer(m, "seed", p.seed).unwrap();
        Preconditions::integer_in_range(m, "maxCars", p.max_cars as f64, 1.0, 299.0).unwrap();
        if let Some(v) = p.car_length_m {
            Preconditions::positive(m, "carLengthM", v).unwrap();
        }
        if let Some(v) = p.car_width_m {
            Preconditions::positive(m, "carWidthM", v).unwrap();
        }
        if let Some(v) = p.lane_width_m {
            Preconditions::positive(m, "laneWidthM", v).unwrap();
        }
        if let Some(v) = p.min_gap_m {
            Preconditions::non_negative(m, "minGapM", v).unwrap();
        }
        if let Some(v) = p.max_accel_mps2 {
            Preconditions::positive(m, "maxAccelMps2", v).unwrap();
        }
        if let Some(v) = p.max_decel_mps2 {
            Preconditions::positive(m, "maxDecelMps2", v).unwrap();
        }
        if let Some(v) = p.max_jerk_mps3 {
            Preconditions::positive(m, "maxJerkMps3", v).unwrap();
        }
        if let Some(v) = p.reaction_time_sec {
            Preconditions::non_negative(m, "reactionTimeSec", v).unwrap();
        }
        if let Some(v) = p.time_headway_sec {
            Preconditions::non_negative(m, "timeHeadwaySec", v).unwrap();
        }
        if let Some(v) = p.grid_cell_size_m {
            Preconditions::positive(m, "gridCellSizeM", v).unwrap();
        }
        if let Some(v) = p.grid_look_ahead_m {
            Preconditions::positive(m, "gridLookAheadM", v).unwrap();
        }
        if let Some(v) = p.spawn_rate_multiplier {
            Preconditions::non_negative(m, "spawnRateMultiplier", v).unwrap();
        }
        Preconditions::check(
            m,
            "carWidthM",
            "fit within laneWidthM",
            self.car_width_m() <= self.lane_width_m(),
            None,
        )
        .unwrap();
        validate_traffic_network(&self.network);
        let max_ticks = (p.duration_sec / p.dt_sec).ceil();
        Preconditions::integer_in_range(m, "tick count", max_ticks, 1.0, 100000.0).unwrap();
    }

    fn has_work(&self) -> bool {
        self.tick as f64 * self.params.dt_sec < self.params.duration_sec - 1e-9
    }

    fn run_time_step(&mut self) {
        let time_sec = self.tick as f64 * self.params.dt_sec;
        self.spawn_cars(time_sec);
        self.advance_cars(time_sec);
        self.rebuild_spatial_index();
        self.max_active_cars = self.max_active_cars.max(self.cars.len());
        let snap = self.snapshot(time_sec);
        self.trace.push(snap);
        if let Some(logger) = &self.options.logger {
            let mut fields = HashMap::new();
            fields.insert("tick".to_string(), self.tick.to_string());
            fields.insert("timeSec".to_string(), time_sec.to_string());
            fields.insert("activeCars".to_string(), self.cars.len().to_string());
            fields.insert("entered".to_string(), self.entered.to_string());
            fields.insert("exited".to_string(), self.exited.to_string());
            fields.insert("dropped".to_string(), self.dropped.to_string());
            logger.log(LogEvent {
                kind: "traffic-flow-tick".to_string(),
                level: Some(LogLevel::Debug),
                fields,
            });
        }
        self.tick += 1;
    }
}

// =============================================================================
// Free functions / network builders
// =============================================================================

fn to_car_snapshot(car: &TrafficCar) -> TrafficCarSnapshot {
    TrafficCarSnapshot {
        id: car.id,
        lane_id: car.lane_id.clone(),
        position_m: car.position_m,
        speed_mps: car.speed_mps,
        acceleration_mps2: car.acceleration_mps2,
        jerk_mps3: car.jerk_mps3,
        target_acceleration_mps2: car.target_acceleration_mps2,
        route: car.route.clone(),
        route_index: car.route_index,
        destination_sink_id: car.destination_sink_id.clone(),
        created_at_sec: car.created_at_sec,
        wait_sec: car.wait_sec,
        grid_cell_ids: car.grid_cell_ids.clone(),
        grid_cell_count: car.grid_cell_count,
        leader_id: car.leader_id,
        leader_gap_m: car.leader_gap_m,
    }
}

fn validate_traffic_network(network: &TrafficNetwork) {
    let m = "TrafficGridStation";
    Preconditions::non_empty(m, "network.nodes", &network.nodes).unwrap();
    Preconditions::non_empty(m, "network.lanes", &network.lanes).unwrap();
    Preconditions::non_empty(m, "network.sources", &network.sources).unwrap();
    Preconditions::non_empty(m, "network.sinks", &network.sinks).unwrap();
    let mut node_ids: HashSet<String> = HashSet::new();
    for node in &network.nodes {
        Preconditions::check(
            m,
            "node.id",
            "be unique and non-empty",
            !node.id.is_empty() && !node_ids.contains(&node.id),
            Some(node.id.clone()),
        )
        .unwrap();
        node_ids.insert(node.id.clone());
        Preconditions::finite(m, &format!("node.{}.x", node.id), node.x).unwrap();
        Preconditions::finite(m, &format!("node.{}.y", node.id), node.y).unwrap();
    }
    let mut lane_ids: HashSet<String> = HashSet::new();
    for lane in &network.lanes {
        Preconditions::check(
            m,
            "lane.id",
            "be unique and non-empty",
            !lane.id.is_empty() && !lane_ids.contains(&lane.id),
            Some(lane.id.clone()),
        )
        .unwrap();
        lane_ids.insert(lane.id.clone());
        Preconditions::check(
            m,
            &format!("lane.{}.from", lane.id),
            "reference a node",
            node_ids.contains(&lane.from),
            Some(lane.from.clone()),
        )
        .unwrap();
        Preconditions::check(
            m,
            &format!("lane.{}.to", lane.id),
            "reference a node",
            node_ids.contains(&lane.to),
            Some(lane.to.clone()),
        )
        .unwrap();
        Preconditions::positive(m, &format!("lane.{}.lengthM", lane.id), lane.length_m).unwrap();
        Preconditions::positive(
            m,
            &format!("lane.{}.speedLimitMps", lane.id),
            lane.speed_limit_mps,
        )
        .unwrap();
        if let Some(cap) = lane.capacity {
            Preconditions::integer_in_range(
                m,
                &format!("lane.{}.capacity", lane.id),
                cap as f64,
                1.0,
                10000.0,
            )
            .unwrap();
        }
    }
    for signal in network.signals.iter().flatten() {
        Preconditions::check(
            m,
            &format!("signal.{}.nodeId", signal.node_id),
            "reference a node",
            node_ids.contains(&signal.node_id),
            Some(signal.node_id.clone()),
        )
        .unwrap();
        Preconditions::non_empty(
            m,
            &format!("signal.{}.phases", signal.node_id),
            &signal.phases,
        )
        .unwrap();
        for phase in &signal.phases {
            Preconditions::positive(
                m,
                &format!("signal.{}.phase.{}.durationSec", signal.node_id, phase.name),
                phase.duration_sec,
            )
            .unwrap();
            for lane_id in &phase.green_lanes {
                Preconditions::check(
                    m,
                    &format!("signal.{}.greenLanes", signal.node_id),
                    "reference a lane",
                    lane_ids.contains(lane_id),
                    Some(lane_id.clone()),
                )
                .unwrap();
            }
        }
    }
    for source in &network.sources {
        Preconditions::check(
            m,
            &format!("source.{}.nodeId", source.id),
            "reference a source node",
            node_ids.contains(&source.node_id),
            Some(source.node_id.clone()),
        )
        .unwrap();
        Preconditions::non_negative(
            m,
            &format!("source.{}.ratePerMin", source.id),
            source.rate_per_min,
        )
        .unwrap();
    }
    let sink_ids: HashSet<String> = network.sinks.iter().map(|s| s.id.clone()).collect();
    for sink in &network.sinks {
        Preconditions::check(
            m,
            &format!("sink.{}.nodeId", sink.id),
            "reference a sink node",
            node_ids.contains(&sink.node_id),
            Some(sink.node_id.clone()),
        )
        .unwrap();
    }
    for source in &network.sources {
        let dests = source
            .destination_sink_ids
            .clone()
            .unwrap_or_else(|| sink_ids.iter().cloned().collect());
        for sink_id in &dests {
            Preconditions::check(
                m,
                &format!("source.{}.destinationSinkIds", source.id),
                "reference a sink id",
                sink_ids.contains(sink_id),
                Some(sink_id.clone()),
            )
            .unwrap();
            if let Some(sink) = network.sinks.iter().find(|s| &s.id == sink_id) {
                Preconditions::check(
                    m,
                    &format!("route {}->{}", source.id, sink_id),
                    "have at least one directed lane path",
                    !shortest_lane_path(network, &source.node_id, &sink.node_id).is_empty(),
                    None,
                )
                .unwrap();
            }
        }
    }
}

fn default_lane_capacity(lane: &TrafficLane, vehicle_space: f64) -> usize {
    1.max((lane.length_m / vehicle_space).floor() as usize)
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    lo.max(hi.min(x))
}

fn current_signal_phase(signal: &TrafficSignal, time_sec: f64) -> TrafficSignalPhase {
    let cycle: f64 = signal.phases.iter().map(|p| p.duration_sec).sum();
    let mut t = ((time_sec + signal.offset_sec.unwrap_or(0.0)) % cycle + cycle) % cycle;
    for phase in &signal.phases {
        if t < phase.duration_sec {
            return phase.clone();
        }
        t -= phase.duration_sec;
    }
    signal.phases[signal.phases.len() - 1].clone()
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
        let lane = prev_lane.get(&cur).cloned();
        let prev = prev_node.get(&cur).cloned();
        match (lane, prev) {
            (Some(l), Some(p)) => {
                route.push(l);
                cur = p;
            }
            _ => return Vec::new(),
        }
    }
    route.reverse();
    route
}

/// Build the canonical five-intersection demo network (TS
/// `buildFiveIntersectionTrafficNetwork`).
pub fn build_five_intersection_traffic_network() -> TrafficNetwork {
    let nodes: Vec<TrafficNode> = vec![
        TrafficNode {
            id: "W".to_string(),
            kind: TrafficNodeKind::Source,
            x: 0.0,
            y: 1.0,
        },
        TrafficNode {
            id: "S0".to_string(),
            kind: TrafficNodeKind::Source,
            x: 1.0,
            y: 2.0,
        },
        TrafficNode {
            id: "N2".to_string(),
            kind: TrafficNodeKind::Source,
            x: 3.0,
            y: 0.0,
        },
        TrafficNode {
            id: "I0".to_string(),
            kind: TrafficNodeKind::Intersection,
            x: 1.0,
            y: 1.0,
        },
        TrafficNode {
            id: "I1".to_string(),
            kind: TrafficNodeKind::Intersection,
            x: 2.0,
            y: 1.0,
        },
        TrafficNode {
            id: "I2".to_string(),
            kind: TrafficNodeKind::Intersection,
            x: 3.0,
            y: 1.0,
        },
        TrafficNode {
            id: "I3".to_string(),
            kind: TrafficNodeKind::Intersection,
            x: 4.0,
            y: 1.0,
        },
        TrafficNode {
            id: "I4".to_string(),
            kind: TrafficNodeKind::Intersection,
            x: 5.0,
            y: 1.0,
        },
        TrafficNode {
            id: "E".to_string(),
            kind: TrafficNodeKind::Sink,
            x: 6.0,
            y: 1.0,
        },
        TrafficNode {
            id: "N1".to_string(),
            kind: TrafficNodeKind::Sink,
            x: 2.0,
            y: 0.0,
        },
        TrafficNode {
            id: "S4".to_string(),
            kind: TrafficNodeKind::Sink,
            x: 5.0,
            y: 2.0,
        },
    ];
    let mk = |id: &str, from: &str, to: &str, length_m: f64| TrafficLane {
        id: id.to_string(),
        from: from.to_string(),
        to: to.to_string(),
        length_m,
        speed_limit_mps: 13.5,
        capacity: None,
    };
    let lanes: Vec<TrafficLane> = vec![
        mk("W-I0", "W", "I0", 90.0),
        mk("S0-I0", "S0", "I0", 85.0),
        mk("I0-I1", "I0", "I1", 120.0),
        mk("I1-I2", "I1", "I2", 120.0),
        mk("N2-I2", "N2", "I2", 90.0),
        mk("I2-I3", "I2", "I3", 120.0),
        mk("I3-I4", "I3", "I4", 120.0),
        mk("I4-E", "I4", "E", 100.0),
        mk("I1-N1", "I1", "N1", 80.0),
        mk("I4-S4", "I4", "S4", 85.0),
    ];
    let phase = |name: &str, green_lanes: Vec<&str>, duration_sec: f64| TrafficSignalPhase {
        name: name.to_string(),
        green_lanes: green_lanes.into_iter().map(|s| s.to_string()).collect(),
        duration_sec,
    };
    let signals: Vec<TrafficSignal> = vec![
        TrafficSignal {
            node_id: "I0".to_string(),
            phases: vec![
                phase("main", vec!["W-I0"], 28.0),
                phase("side", vec!["S0-I0"], 16.0),
            ],
            offset_sec: None,
        },
        TrafficSignal {
            node_id: "I1".to_string(),
            phases: vec![phase("main", vec!["I0-I1"], 30.0)],
            offset_sec: None,
        },
        TrafficSignal {
            node_id: "I2".to_string(),
            phases: vec![
                phase("main", vec!["I1-I2"], 26.0),
                phase("side", vec!["N2-I2"], 18.0),
            ],
            offset_sec: Some(5.0),
        },
        TrafficSignal {
            node_id: "I3".to_string(),
            phases: vec![phase("main", vec!["I2-I3"], 30.0)],
            offset_sec: None,
        },
        TrafficSignal {
            node_id: "I4".to_string(),
            phases: vec![phase("main", vec!["I3-I4"], 26.0)],
            offset_sec: None,
        },
    ];
    let sources: Vec<TrafficSource> = vec![
        TrafficSource {
            id: "west".to_string(),
            node_id: "W".to_string(),
            rate_per_min: 18.0,
            destination_sink_ids: Some(vec![
                "east".to_string(),
                "north1".to_string(),
                "south4".to_string(),
            ]),
        },
        TrafficSource {
            id: "south0".to_string(),
            node_id: "S0".to_string(),
            rate_per_min: 7.0,
            destination_sink_ids: Some(vec![
                "east".to_string(),
                "north1".to_string(),
                "south4".to_string(),
            ]),
        },
        TrafficSource {
            id: "north2".to_string(),
            node_id: "N2".to_string(),
            rate_per_min: 8.0,
            destination_sink_ids: Some(vec!["east".to_string(), "south4".to_string()]),
        },
    ];
    let sinks: Vec<TrafficSink> = vec![
        TrafficSink {
            id: "east".to_string(),
            node_id: "E".to_string(),
        },
        TrafficSink {
            id: "north1".to_string(),
            node_id: "N1".to_string(),
        },
        TrafficSink {
            id: "south4".to_string(),
            node_id: "S4".to_string(),
        },
    ];
    TrafficNetwork {
        nodes,
        lanes,
        signals: Some(signals),
        sources,
        sinks,
    }
}

/// Run a fixed-step traffic simulation (TS `runTrafficFlow`).
pub fn run_traffic_flow(
    params: TrafficParams,
    logger: Option<Box<dyn OptimizationLogger>>,
) -> TrafficResult {
    let max_ticks = (params.duration_sec / params.dt_sec).ceil() as usize + 1;
    let station = Rc::new(RefCell::new(TrafficGridStation::new(
        params,
        TrafficOptions { logger },
    )));
    let summary = run_iterative_des(
        vec![station.clone() as StationRef],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_ticks),
            ..Default::default()
        },
    );
    let checks = summary.validation.unwrap_or_default();
    let result = station.borrow().result(checks);
    result
}

#[cfg(test)]
mod tests {
    //! Tests for the network-flow models.
    //!
    //! The max-flow cases solve small graphs whose analytic optimum is known and
    //! confirm the value equals the residual min-cut capacity. The traffic case
    //! runs the builtin five-intersection grid briefly and checks the flow
    //! conservation identity entered equals exited plus active, plus the cap on
    //! active vehicles.

    use super::*;

    fn edge(from: usize, to: usize, capacity: f64) -> FlowEdge {
        FlowEdge {
            from,
            to,
            capacity,
            name: None,
        }
    }

    #[test]
    fn max_flow_diamond_optimum() {
        // 0→1(3), 0→2(2), 1→3(2), 2→3(3), 1→2(1). Source 0, sink 3 ⇒ max flow 5.
        let params = MaxFlowParams {
            num_nodes: 4,
            source: 0,
            sink: 3,
            edges: vec![
                edge(0, 1, 3.0),
                edge(0, 2, 2.0),
                edge(1, 3, 2.0),
                edge(2, 3, 3.0),
                edge(1, 2, 1.0),
            ],
            max_augmentations: None,
            node_coordinates: None,
            node_names: None,
        };
        let res = run_max_flow(params, None);
        assert!(
            (res.max_flow - 5.0).abs() < 1e-9,
            "max flow = {}",
            res.max_flow
        );
        assert!((res.min_cut.capacity - res.max_flow).abs() < 1e-9);
    }

    #[test]
    fn max_flow_chain_bottleneck() {
        // 0→1(10), 1→2(4), 2→3(10). Source 0, sink 3 ⇒ max flow 4.
        let params = MaxFlowParams {
            num_nodes: 4,
            source: 0,
            sink: 3,
            edges: vec![edge(0, 1, 10.0), edge(1, 2, 4.0), edge(2, 3, 10.0)],
            max_augmentations: None,
            node_coordinates: None,
            node_names: None,
        };
        let res = run_max_flow(params, None);
        assert!(
            (res.max_flow - 4.0).abs() < 1e-9,
            "max flow = {}",
            res.max_flow
        );
        assert_eq!(res.min_cut.cut_edges, vec![1]);
        assert!((res.min_cut.capacity - 4.0).abs() < 1e-9);
    }

    fn small_traffic_params(seed: f64) -> TrafficParams {
        TrafficParams {
            builtin: Some("five-intersection".to_string()),
            network: None,
            duration_sec: 8.0,
            dt_sec: 1.0,
            seed,
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
            spawn_rate_multiplier: None,
            scheduled_trips: None,
        }
    }

    #[test]
    fn traffic_flow_conserves_cars() {
        let res = run_traffic_flow(small_traffic_params(7.0), None);
        // Conservation identity: every spawned car is either still active or has exited.
        assert_eq!(res.entered, res.exited + res.final_cars.len());
        assert!(res.max_active_cars <= res.params.max_cars);
        // Some cars should have spawned over 8 seconds on the busy west corridor.
        assert!(res.entered > 0, "expected at least one spawned car");
    }
}
