//! Port of `src/des/general/internal-solver-network.ts` — runnable DES station
//! networks for common optimization / search problems with a wall-clock cap.
//!
//! Solvers are stationary entities, incumbent / best solutions are moving
//! tokens, and a wall-clock checker station provides the "cap most runs around
//! three minutes" stop condition without relying on external solvers.
//!
//! ## TS to Rust mapping
//!
//!   * `SOLUTION_CHANNEL` / `STOP_CHANNEL` constants stay `&str` channel ids.
//!   * `type InternalSolverKind = '...'` (string union) becomes the
//!     [`InternalSolverKind`] enum; the `'bellman-ford' | 'dijkstra'` algorithm
//!     union becomes [`ShortestPathAlgorithm`].
//!   * The various `interface Solver*/Snapshot*/*Params/*Result` become structs.
//!   * `bestState: unknown` and `metadata?: Record<string, unknown>` are
//!     represented as the [`SolverBestState`] enum and an ordered list of
//!     `(String, MetaValue)` pairs respectively.
//!   * `class *Token` become plain structs carried as `Rc<dyn Any>` tokens.
//!   * `interface SnapshotProvider` becomes the [`SnapshotProvider`] trait.
//!   * `WallClockCheckerStation` uses a wall-clock time source: per the TS
//!     migration header we inject a [`Clock`](crate::des::shared::capabilities::Clock)
//!     (defaulting to `SystemClock`) instead of calling the system clock
//!     directly, which keeps the three-minute cap deterministic / testable.
//!   * `mulberry32(seed)` in the SA / GA solvers becomes an injected
//!     `RandomSource` (the optimizers create their own `mulberry32(seed)`).
//!
//! ## Inheritance handling (flagged)
//!
//! `ObservableTSPSAOptimizer extends TSPSAOptimizer` and
//! `ObservableTSPGAOptimizer extends TSPGAOptimizer` override the
//! `onAccept` / `onReject` / `onGeneration` / `onFinish` hooks to additionally
//! emit a progress snapshot. Rust has no inheritance, so each observable wrapper
//! EMBEDS the base optimizer (`inner`), delegates every hook trait method to it,
//! and overrides the three / two notification hooks to delegate-then-emit. The
//! wrapper owns its own `StationCore` (the one wired into the network) and the
//! base optimizer's intrinsic / ground-truth validators are re-registered on the
//! wrapper (the base's validators downcast to the base type, which the wrapper is
//! not, so they cannot be shared directly).

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::population_optimizer::{PopulationOptimizer, PopulationState};
use crate::des::general::des_base::runner::{
    run_iterative_des, IterativeRunOptions, IterativeRunSummary, RunReason,
};
use crate::des::general::des_base::single_state_optimizer::{
    SingleStateOptimizer, SingleStateState,
};
use crate::des::general::des_base::station::{AnyToken, DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::{
    intrinsic_check, monotonicity_validator, Monotonicity, ValidationCheck,
};
use crate::des::general::ga_des::{TSPGAOptimizer, TSPGAOptions};
use crate::des::general::genetic_tsp::{
    build_pentagon_tsp, build_random_tsp, held_karp_exact, is_permutation, tour_length, HeldKarpResult,
    InitMode, TSPInstance, Tour,
};
use crate::des::general::prng::mulberry32;
use crate::des::general::sa_des::{CoolingSchedule, Moves, TSPSAOptimizer, TSPSAOptions, temperature_at};
use crate::des::general::shortest_path_des::{
    build_random_graph, build_small_chain_graph, Graph,
};
use crate::des::shared::capabilities::{Clock, RandomSource, SystemClock};

// =============================================================================
// Channels
// =============================================================================

/// Channel for incumbent / best solution snapshots.
pub const SOLUTION_CHANNEL: &str = "solution";
/// Channel for the wall-clock stop signal.
pub const STOP_CHANNEL: &str = "stop";

/// JavaScript `Number.MAX_SAFE_INTEGER`, used in integer-range preconditions.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

// =============================================================================
// Enums (TS string unions)
// =============================================================================

/// Which solver a network runs. (TS `type InternalSolverKind`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternalSolverKind {
    ShortestPath,
    KnapsackDp,
    KnapsackSa,
    TspSa,
    TspGa,
    TspHeldKarp,
}

impl InternalSolverKind {
    pub fn as_str(self) -> &'static str {
        match self {
            InternalSolverKind::ShortestPath => "shortest-path",
            InternalSolverKind::KnapsackDp => "knapsack-dp",
            InternalSolverKind::KnapsackSa => "knapsack-sa",
            InternalSolverKind::TspSa => "tsp-sa",
            InternalSolverKind::TspGa => "tsp-ga",
            InternalSolverKind::TspHeldKarp => "tsp-held-karp",
        }
    }
}

/// Shortest-path algorithm. (TS `'bellman-ford' | 'dijkstra'`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortestPathAlgorithm {
    BellmanFord,
    Dijkstra,
}

/// Built-in shortest-path graph. (TS `builtin?: 'small-chain'`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortestPathBuiltin {
    SmallChain,
}

/// Built-in TSP instance. (TS `builtin?: 'pentagon' | 'random'`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TspBuiltin {
    Pentagon,
    Random,
}

/// Run status. (TS `'complete' | 'time-limit' | 'tick-limit'`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternalSolverStatus {
    Complete,
    TimeLimit,
    TickLimit,
}

/// Network-node role. (TS `'solver' | 'checker' | 'sink' | 'source'`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverNodeRole {
    Solver,
    Checker,
    Sink,
    Source,
}

/// A single heterogeneous metadata value. (TS `Record<string, unknown>` values
/// are either numbers or booleans in this file.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetaValue {
    Number(f64),
    Bool(bool),
}

/// Solver-specific best-state payload. (TS `bestState: unknown`.)
#[derive(Clone, Debug)]
pub enum SolverBestState {
    ShortestPath {
        distance: Vec<f64>,
        predecessor: Vec<i64>,
        algorithm: ShortestPathAlgorithm,
        has_negative_cycle_from_source: bool,
    },
    Knapsack {
        selected: Vec<f64>,
        value: f64,
        weight: f64,
        capacity: f64,
    },
    Tour {
        tour: Tour,
        length: f64,
    },
}

// =============================================================================
// Progress payload + network description structs
// =============================================================================

/// One incumbent snapshot. (TS `interface SolverProgressPayload`.)
#[derive(Clone, Debug)]
pub struct SolverProgressPayload {
    pub solver_id: String,
    pub solver_kind: InternalSolverKind,
    pub tick: usize,
    pub iteration: usize,
    pub objective: f64,
    pub best_state: SolverBestState,
    pub feasible: bool,
    pub done: bool,
    pub metadata: Vec<(String, MetaValue)>,
}

/// Stop-signal payload. (TS `StopSignalToken['payload']`.)
#[derive(Clone, Debug)]
pub struct StopSignalPayload {
    pub checker_id: String,
    pub elapsed_ms: f64,
    pub budget_ms: f64,
    pub tick: usize,
}

/// A stationary entity in the network description.
#[derive(Clone, Debug)]
pub struct SolverNetworkNode {
    pub id: String,
    pub kind: String,
    pub role: SolverNodeRole,
}

/// A moving entity in the network description.
#[derive(Clone, Debug)]
pub struct SolverNetworkMovingEntity {
    pub id: String,
    pub kind: String,
    pub token_type: String,
}

/// A wiring edge in the network description.
#[derive(Clone, Debug)]
pub struct SolverNetworkEdge {
    pub from: String,
    pub to: String,
    pub moving_entity: String,
    pub channel: String,
}

/// Network topology description. (TS `interface SolverNetworkDescription`.)
#[derive(Clone, Debug)]
pub struct SolverNetworkDescription {
    pub stationary_entities: Vec<SolverNetworkNode>,
    pub moving_entities: Vec<SolverNetworkMovingEntity>,
    pub edges: Vec<SolverNetworkEdge>,
}

/// Wall-clock accounting block in the run result.
#[derive(Clone, Debug)]
pub struct WallClockReport {
    pub budget_ms: f64,
    pub elapsed_ms: f64,
    pub checks: usize,
    pub expired: bool,
}

/// Final result of a network run. (TS `interface InternalSolverRunResult`.)
#[derive(Clone, Debug)]
pub struct InternalSolverRunResult {
    pub kind: InternalSolverKind,
    pub status: InternalSolverStatus,
    pub run_summary: IterativeRunSummary,
    pub best: SolverProgressPayload,
    pub trace: Vec<SolverProgressPayload>,
    pub stop_signals: Vec<StopSignalPayload>,
    pub wall_clock: WallClockReport,
    pub network: SolverNetworkDescription,
    pub validation: Vec<ValidationCheck>,
}

// =============================================================================
// Param structs
// =============================================================================

/// Random-graph spec. (TS `ShortestPathSolverParams.randomGraph`.)
#[derive(Clone, Debug)]
pub struct RandomGraphSpec {
    pub num_nodes: usize,
    pub edge_prob: f64,
    pub w_min: f64,
    pub w_max: f64,
    pub seed: u32,
}

/// Shortest-path solver params. (TS `interface ShortestPathSolverParams`.)
#[derive(Clone, Debug)]
pub struct ShortestPathSolverParams {
    pub algorithm: ShortestPathAlgorithm,
    pub source: usize,
    pub builtin: Option<ShortestPathBuiltin>,
    pub graph: Option<Graph>,
    pub random_graph: Option<RandomGraphSpec>,
}

/// Knapsack solver params. (TS `interface KnapsackParams`.)
#[derive(Clone, Debug)]
pub struct KnapsackParams {
    pub values: Vec<f64>,
    pub weights: Vec<f64>,
    pub capacity: f64,
    pub seed: Option<u32>,
    pub max_iterations: Option<usize>,
    pub cooling: Option<CoolingSchedule>,
    pub stall_limit: Option<usize>,
    pub penalty: Option<f64>,
}

/// Partial SA options. (TS `Partial<TSPSAOptions>`.)
#[derive(Clone, Debug, Default)]
pub struct TSPSAOptionsPartial {
    pub cooling: Option<CoolingSchedule>,
    pub max_iterations: Option<usize>,
    pub seed: Option<u32>,
    pub init: Option<InitMode>,
    pub moves: Option<Moves>,
    pub penalty_per_violation: Option<f64>,
    pub trace_stride: Option<usize>,
    pub stall_limit: Option<usize>,
}

/// Partial GA options. (TS `Partial<TSPGAOptions>`.)
#[derive(Clone, Debug, Default)]
pub struct TSPGAOptionsPartial {
    pub pop_size: Option<usize>,
    pub num_generations: Option<usize>,
    pub tournament_size: Option<usize>,
    pub crossover_prob: Option<f64>,
    pub mutation_prob: Option<f64>,
    pub elitism: Option<usize>,
    pub seed: Option<u32>,
    pub init: Option<InitMode>,
    pub penalty_per_violation: Option<f64>,
}

/// TSP solver params. (TS `interface TSPSolverParams`.)
#[derive(Clone, Debug, Default)]
pub struct TSPSolverParams {
    pub builtin: Option<TspBuiltin>,
    pub n: Option<usize>,
    pub seed: Option<u32>,
    pub coordinates: Option<Vec<(f64, f64)>>,
    pub distance: Option<Vec<Vec<f64>>>,
    pub precedence: Option<Vec<(usize, usize)>>,
    pub sa: Option<TSPSAOptionsPartial>,
    pub ga: Option<TSPGAOptionsPartial>,
}

/// Top-level run params. (TS `interface InternalSolverRunParams`.)
#[derive(Clone, Debug)]
pub struct InternalSolverRunParams {
    pub kind: InternalSolverKind,
    pub time_limit_ms: Option<f64>,
    pub max_ticks: Option<usize>,
    pub check_every_ticks: Option<usize>,
    pub shortest_path: Option<ShortestPathSolverParams>,
    pub knapsack: Option<KnapsackParams>,
    pub tsp: Option<TSPSolverParams>,
}

// =============================================================================
// Tokens
// =============================================================================

/// Carries a [`SolverProgressPayload`]. (TS `class SolverSolutionToken`.)
pub struct SolverSolutionToken {
    pub payload: SolverProgressPayload,
}

impl SolverSolutionToken {
    pub fn new(payload: SolverProgressPayload) -> Self {
        SolverSolutionToken { payload }
    }
}

/// Carries a [`StopSignalPayload`]. (TS `class StopSignalToken`.)
pub struct StopSignalToken {
    pub payload: StopSignalPayload,
}

impl StopSignalToken {
    pub fn new(payload: StopSignalPayload) -> Self {
        StopSignalToken { payload }
    }
}

// =============================================================================
// SnapshotProvider trait
// =============================================================================

/// Stations that can produce a progress snapshot. (TS `interface
/// SnapshotProvider`.)
pub trait SnapshotProvider {
    fn snapshot(&self, done: bool) -> SolverProgressPayload;
}

// =============================================================================
// WallClockCheckerStation
// =============================================================================

/// Emits a stop signal once the injected clock passes `budget_ms`. (TS
/// `class WallClockCheckerStation`.)
pub struct WallClockCheckerStation {
    core: StationCore,
    pub budget_ms: f64,
    pub check_every_ticks: usize,
    clock: Box<dyn Clock>,
    started_at: u128,
    tick: usize,
    expired_flag: bool,
    elapsed: f64,
    checks: usize,
}

impl WallClockCheckerStation {
    pub fn new(
        id: impl Into<String>,
        budget_ms: f64,
        check_every_ticks: usize,
        clock: Option<Box<dyn Clock>>,
    ) -> Self {
        let clock: Box<dyn Clock> = clock.unwrap_or_else(|| Box::new(SystemClock));
        let started_at = clock.now_ms();
        let id_str = id.into();
        let id_for_validator = id_str.clone();
        let mut st = WallClockCheckerStation {
            core: StationCore::new(id_str),
            budget_ms,
            check_every_ticks,
            clock,
            started_at,
            tick: 0,
            expired_flag: false,
            elapsed: 0.0,
            checks: 0,
        };
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                format!("{id_for_validator}.budget-nonnegative"),
                |s| downcast::<WallClockCheckerStation>(s).budget_ms >= 0.0,
                Some("budgetMs >= 0".to_string()),
                Some(Box::new(|s| {
                    downcast::<WallClockCheckerStation>(s).budget_ms.to_string()
                })),
                Some("wall-clock-checker".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    fn now_ms(&self) -> u128 {
        self.clock.now_ms()
    }

    fn elapsed_since_start(&self) -> f64 {
        ((self.now_ms() as i128) - (self.started_at as i128)).max(0) as f64
    }

    pub fn expired(&self) -> bool {
        self.expired_flag
    }

    pub fn elapsed_ms(&self) -> f64 {
        if self.expired_flag {
            self.elapsed
        } else {
            self.elapsed_since_start()
        }
    }

    pub fn num_checks(&self) -> usize {
        self.checks
    }
}

impl DESStation for WallClockCheckerStation {
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
        Preconditions_non_negative("WallClockCheckerStation", "budgetMs", self.budget_ms);
        Preconditions_integer_in_range(
            "WallClockCheckerStation",
            "checkEveryTicks",
            self.check_every_ticks as f64,
            1.0,
            MAX_SAFE_INTEGER,
        );
    }

    fn has_work(&self) -> bool {
        false
    }

    fn run_time_step(&mut self) {
        if self.expired_flag {
            return;
        }
        if self.tick % self.check_every_ticks == 0 {
            self.checks += 1;
            self.elapsed = self.elapsed_since_start();
            if self.elapsed >= self.budget_ms {
                self.expired_flag = true;
                let payload = StopSignalPayload {
                    checker_id: self.id().to_string(),
                    elapsed_ms: self.elapsed,
                    budget_ms: self.budget_ms,
                    tick: self.tick,
                };
                let token: AnyToken = Rc::new(StopSignalToken::new(payload));
                self.core.emit(token, STOP_CHANNEL);
            }
        }
        self.tick += 1;
    }
}

// =============================================================================
// SolutionSinkStation
// =============================================================================

/// Collects incumbent snapshots and stop signals. (TS `class
/// SolutionSinkStation`.)
pub struct SolutionSinkStation {
    core: StationCore,
    pub trace: Vec<SolverProgressPayload>,
    pub stops: Vec<StopSignalPayload>,
}

impl SolutionSinkStation {
    pub fn new(id: impl Into<String>) -> Self {
        SolutionSinkStation { core: StationCore::new(id), trace: Vec::new(), stops: Vec::new() }
    }

    pub fn best(&self) -> Option<SolverProgressPayload> {
        let mut best: Option<&SolverProgressPayload> = None;
        for row in &self.trace {
            match best {
                None => best = Some(row),
                Some(b) if row.objective <= b.objective => best = Some(row),
                _ => {}
            }
        }
        best.cloned()
    }
}

impl DESStation for SolutionSinkStation {
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

    fn run_time_step(&mut self) {
        for token in self.core.drain::<SolverSolutionToken>(SOLUTION_CHANNEL) {
            self.trace.push(token.payload.clone());
        }
        for token in self.core.drain::<StopSignalToken>(STOP_CHANNEL) {
            self.stops.push(token.payload.clone());
        }
    }
}

// =============================================================================
// ShortestPathSolverStation
// =============================================================================

struct PendingNode {
    node_id: usize,
    distance: f64,
}

/// Incremental Bellman–Ford / Dijkstra solver. (TS `class
/// ShortestPathSolverStation`.)
pub struct ShortestPathSolverStation {
    core: StationCore,
    graph: Graph,
    algorithm: ShortestPathAlgorithm,
    source: usize,
    distance: Vec<f64>,
    predecessor: Vec<i64>,
    dirty: Vec<bool>,
    settled: Vec<bool>,
    pending: Vec<PendingNode>,
    iter: usize,
    done: bool,
    waves: usize,
    negative_cycle: bool,
}

impl ShortestPathSolverStation {
    pub fn new(id: impl Into<String>, params: ShortestPathSolverParams) -> Self {
        let graph = graph_from_params(&params);
        let n = graph.num_nodes;
        let mut st = ShortestPathSolverStation {
            core: StationCore::new(id),
            graph,
            algorithm: params.algorithm,
            source: params.source,
            distance: vec![f64::INFINITY; n],
            predecessor: vec![-1; n],
            dirty: vec![false; n],
            settled: vec![false; n],
            pending: Vec::new(),
            iter: 0,
            done: false,
            waves: 0,
            negative_cycle: false,
        };
        st.distance[st.source] = 0.0;
        st.dirty[st.source] = true;
        st.pending.push(PendingNode { node_id: st.source, distance: 0.0 });
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "shortest-path-source-distance-zero",
                |s| {
                    let st = downcast::<ShortestPathSolverStation>(s);
                    st.distance[st.source] == 0.0
                },
                Some("distance[source] = 0".to_string()),
                Some(Box::new(|s| {
                    let st = downcast::<ShortestPathSolverStation>(s);
                    st.distance[st.source].to_string()
                })),
                Some("internal-solver-shortest-path".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    fn step_bellman_ford(&mut self) {
        self.iter += 1;
        let mut new_dirty = vec![false; self.graph.num_nodes];
        let mut any_change = false;
        for u in 0..self.graph.num_nodes {
            if !self.dirty[u] {
                continue;
            }
            let du = self.distance[u];
            for edge in &self.graph.edges[u] {
                self.waves += 1;
                let cand = du + edge.weight;
                if cand < self.distance[edge.to] - 1e-12 {
                    self.distance[edge.to] = cand;
                    self.predecessor[edge.to] = u as i64;
                    new_dirty[edge.to] = true;
                    any_change = true;
                }
            }
        }
        self.dirty = new_dirty;
        if !any_change {
            self.done = true;
        }
        if self.iter >= self.graph.num_nodes && any_change {
            self.negative_cycle = true;
            self.done = true;
        }
    }

    fn step_dijkstra(&mut self) {
        while !self.pending.is_empty() {
            self.pending.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            let top = self.pending.remove(0);
            if self.settled[top.node_id] {
                continue;
            }
            self.settled[top.node_id] = true;
            self.iter += 1;
            for edge in &self.graph.edges[top.node_id] {
                self.waves += 1;
                let cand = top.distance + edge.weight;
                if cand < self.distance[edge.to] - 1e-12 {
                    self.distance[edge.to] = cand;
                    self.predecessor[edge.to] = top.node_id as i64;
                    self.pending.push(PendingNode { node_id: edge.to, distance: cand });
                }
            }
            return;
        }
        self.done = true;
    }

    fn emit_snapshot(&mut self, done: bool) {
        let payload = self.snapshot(done);
        let token: AnyToken = Rc::new(SolverSolutionToken::new(payload));
        self.core.emit(token, SOLUTION_CHANNEL);
    }
}

impl DESStation for ShortestPathSolverStation {
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
        validate_graph(&self.graph);
        Preconditions_integer_in_range(
            "ShortestPathSolverStation",
            "source",
            self.source as f64,
            0.0,
            (self.graph.num_nodes as f64) - 1.0,
        );
        if self.algorithm == ShortestPathAlgorithm::Dijkstra {
            for u in 0..self.graph.num_nodes {
                for e in &self.graph.edges[u] {
                    Preconditions_non_negative(
                        "ShortestPathSolverStation",
                        &format!("edge {u}->{}", e.to),
                        e.weight,
                    );
                }
            }
        }
    }

    fn has_work(&self) -> bool {
        !self.done
    }

    fn run_time_step(&mut self) {
        if self.done {
            return;
        }
        if self.algorithm == ShortestPathAlgorithm::BellmanFord {
            self.step_bellman_ford();
        } else {
            self.step_dijkstra();
        }
        let done = self.done;
        self.emit_snapshot(done);
    }
}

impl SnapshotProvider for ShortestPathSolverStation {
    fn snapshot(&self, done: bool) -> SolverProgressPayload {
        let reachable = self.distance.iter().filter(|d| d.is_finite()).count();
        let unresolved_penalty = (self.graph.num_nodes - reachable) as f64 * 1e12;
        let objective = unresolved_penalty
            + self.distance.iter().map(|&d| if d.is_finite() { d } else { 0.0 }).sum::<f64>();
        SolverProgressPayload {
            solver_id: self.id().to_string(),
            solver_kind: InternalSolverKind::ShortestPath,
            tick: self.iter,
            iteration: self.iter,
            objective,
            best_state: SolverBestState::ShortestPath {
                distance: self.distance.clone(),
                predecessor: self.predecessor.clone(),
                algorithm: self.algorithm,
                has_negative_cycle_from_source: self.negative_cycle,
            },
            feasible: !self.negative_cycle,
            done,
            metadata: vec![
                ("reachable".to_string(), MetaValue::Number(reachable as f64)),
                ("wavesEmitted".to_string(), MetaValue::Number(self.waves as f64)),
            ],
        }
    }
}

// =============================================================================
// KnapsackDPStation
// =============================================================================

/// Knapsack solution bookkeeping shared by the DP / SA solvers.
#[derive(Clone, Debug)]
pub struct KnapsackSolution {
    pub selected: Vec<f64>,
    pub value: f64,
    pub weight: f64,
    pub capacity: f64,
}

/// Incremental 0/1 knapsack DP. (TS `class KnapsackDPStation`.)
pub struct KnapsackDPStation {
    core: StationCore,
    values: Vec<f64>,
    weights: Vec<f64>,
    capacity: f64,
    keep: Vec<Vec<bool>>,
    dp: Vec<f64>,
    item: usize,
    done: bool,
}

impl KnapsackDPStation {
    pub fn new(id: impl Into<String>, params: KnapsackParams) -> Self {
        let values = params.values.clone();
        let weights = params.weights.clone();
        let capacity = if params.capacity.is_finite() { params.capacity } else { -1.0 };
        validate_knapsack(&values, &weights, capacity);
        let cap = ((capacity + 1.0).max(0.0)) as usize;
        let mut st = KnapsackDPStation {
            core: StationCore::new(id),
            values: values.clone(),
            weights,
            capacity,
            keep: vec![vec![false; cap]; values.len()],
            dp: vec![0.0; cap],
            item: 0,
            done: false,
        };
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "knapsack-dp-capacity-feasible",
                |s| {
                    let st = downcast::<KnapsackDPStation>(s);
                    st.solution().weight <= st.capacity
                },
                Some("selected weight <= capacity".to_string()),
                Some(Box::new(|s| {
                    let st = downcast::<KnapsackDPStation>(s);
                    format!("{}/{}", st.solution().weight, st.capacity)
                })),
                Some("internal-solver-knapsack".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    fn emit_snapshot(&mut self, done: bool) {
        let payload = self.snapshot(done);
        let token: AnyToken = Rc::new(SolverSolutionToken::new(payload));
        self.core.emit(token, SOLUTION_CHANNEL);
    }

    fn solution(&self) -> KnapsackSolution {
        let mut selected = vec![0.0_f64; self.values.len()];
        let mut c = self.capacity as i64;
        let start = self.item.min(self.values.len());
        for i in (0..start).rev() {
            let ci = c as usize;
            if ci < self.keep[i].len() && self.keep[i][ci] {
                selected[i] = 1.0;
                c -= self.weights[i] as i64;
            }
        }
        knapsack_score(&self.values, &self.weights, self.capacity, &selected)
    }
}

impl DESStation for KnapsackDPStation {
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
        validate_knapsack(&self.values, &self.weights, self.capacity);
        Preconditions_integer_in_range("KnapsackDPStation", "capacity", self.capacity, 0.0, 100000.0);
        let cells = (self.values.len() as f64) * (self.capacity + 1.0);
        Preconditions_check(
            "KnapsackDPStation",
            "state space",
            "have at most 5,000,000 cells",
            cells <= 5_000_000.0,
            Some(cells.to_string()),
        );
    }

    fn has_work(&self) -> bool {
        !self.done
    }

    fn run_time_step(&mut self) {
        if self.done {
            return;
        }
        if self.item >= self.values.len() {
            self.done = true;
            self.emit_snapshot(true);
            return;
        }
        let i = self.item;
        let mut next = self.dp.clone();
        let cap = self.dp.len();
        for c in 0..cap {
            let w = self.weights[i];
            if w <= c as f64 {
                let cand = self.dp[c - w as usize] + self.values[i];
                if cand > next[c] {
                    next[c] = cand;
                    self.keep[i][c] = true;
                }
            }
        }
        self.dp = next;
        self.item += 1;
        if self.item >= self.values.len() {
            self.done = true;
        }
        let done = self.done;
        self.emit_snapshot(done);
    }
}

impl SnapshotProvider for KnapsackDPStation {
    fn snapshot(&self, done: bool) -> SolverProgressPayload {
        let sol = self.solution();
        SolverProgressPayload {
            solver_id: self.id().to_string(),
            solver_kind: InternalSolverKind::KnapsackDp,
            tick: self.item,
            iteration: self.item,
            objective: -sol.value,
            feasible: sol.weight <= self.capacity,
            best_state: SolverBestState::Knapsack {
                selected: sol.selected,
                value: sol.value,
                weight: sol.weight,
                capacity: sol.capacity,
            },
            done,
            metadata: vec![
                ("itemsProcessed".to_string(), MetaValue::Number(self.item as f64)),
                ("capacity".to_string(), MetaValue::Number(self.capacity)),
            ],
        }
    }
}

// =============================================================================
// KnapsackSAStation — SingleStateOptimizer<Vec<f64>> leaf
// =============================================================================

/// Simulated-annealing knapsack solver. (TS `class KnapsackSAStation extends
/// SingleStateOptimizer<number[]>`.)
pub struct KnapsackSAStation {
    core: StationCore,
    state: SingleStateState<Vec<f64>>,
    values: Vec<f64>,
    weights: Vec<f64>,
    capacity: f64,
    cooling: CoolingSchedule,
    max_iterations: usize,
    stall_limit: usize,
    penalty: f64,
    stall: usize,
    prev_best: f64,
}

impl KnapsackSAStation {
    pub fn new(id: impl Into<String>, params: KnapsackParams) -> Self {
        let seed = params.seed.unwrap_or(1);
        let values = params.values.clone();
        let weights = params.weights.clone();
        let capacity = if params.capacity.is_finite() { params.capacity } else { -1.0 };
        validate_knapsack(&values, &weights, capacity);
        let cooling = params.cooling.unwrap_or(CoolingSchedule::Geometric {
            t0: 50.0,
            alpha: 0.995,
            t_min: Some(1e-6),
        });
        let mut st = KnapsackSAStation {
            core: StationCore::new(id),
            state: SingleStateState::new(1, Box::new(mulberry32(seed))),
            values,
            weights,
            capacity,
            cooling,
            max_iterations: params.max_iterations.unwrap_or(5000),
            stall_limit: params.stall_limit.unwrap_or(0),
            penalty: params.penalty.unwrap_or(1e6),
            stall: 0,
            prev_best: f64::INFINITY,
        };
        st.bootstrap();
        st.prev_best = st.opt_state().best_cost;
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "knapsack-sa-best-is-finite",
                |s| downcast::<KnapsackSAStation>(s).get_best_cost().is_finite(),
                Some("finite best cost".to_string()),
                Some(Box::new(|s| downcast::<KnapsackSAStation>(s).get_best_cost().to_string())),
                Some("internal-solver-knapsack".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    fn after_step(&mut self) {
        let best_cost = self.opt_state().best_cost;
        if best_cost < self.prev_best {
            self.prev_best = best_cost;
            self.stall = 0;
        } else {
            self.stall += 1;
        }
        self.emit_snapshot(false);
    }

    fn emit_snapshot(&mut self, done: bool) {
        let payload = self.snapshot(done);
        let token: AnyToken = Rc::new(SolverSolutionToken::new(payload));
        self.core.emit(token, SOLUTION_CHANNEL);
    }
}

impl DESStation for KnapsackSAStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        self.optimizer_step();
    }
    fn has_work(&self) -> bool {
        self.optimizer_has_work()
    }
}

impl SingleStateOptimizer<Vec<f64>> for KnapsackSAStation {
    fn opt_state(&self) -> &SingleStateState<Vec<f64>> {
        &self.state
    }
    fn opt_state_mut(&mut self) -> &mut SingleStateState<Vec<f64>> {
        &mut self.state
    }

    fn initial_state(&self, _rng: &mut dyn RandomSource) -> Vec<f64> {
        let mut order: Vec<usize> = (0..self.values.len()).collect();
        order.sort_by(|&a, &b| {
            let ra = self.values[b] / self.weights[b];
            let rb = self.values[a] / self.weights[a];
            ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut x = vec![0.0_f64; self.values.len()];
        let mut w = 0.0;
        for i in order {
            if w + self.weights[i] <= self.capacity {
                x[i] = 1.0;
                w += self.weights[i];
            }
        }
        x
    }

    fn cost(&self, x: &Vec<f64>) -> f64 {
        let s = knapsack_score(&self.values, &self.weights, self.capacity, x);
        -s.value + self.penalty * (s.weight - self.capacity).max(0.0)
    }

    fn propose(&self, x: &Vec<f64>, rng: &mut dyn RandomSource) -> Vec<f64> {
        let j = (rng.next_float() * x.len() as f64).floor() as usize;
        let mut next = x.clone();
        next[j] = 1.0 - next[j];
        next
    }

    fn accept(
        &self,
        _current: &Vec<f64>,
        _candidate: &Vec<f64>,
        current_cost: f64,
        candidate_cost: f64,
        iter: usize,
        rng: &mut dyn RandomSource,
    ) -> bool {
        let delta = candidate_cost - current_cost;
        if delta <= 0.0 {
            return true;
        }
        let t = temperature_at(&self.cooling, iter);
        t > 0.0 && rng.next_float() < (-delta / t).exp()
    }

    fn should_stop(&self, iter: usize) -> bool {
        if iter >= self.max_iterations {
            return true;
        }
        if self.stall_limit > 0 && self.stall >= self.stall_limit {
            return true;
        }
        false
    }

    fn on_accept(&mut self, _candidate: &Vec<f64>, _delta: f64, _iter: usize) {
        self.after_step();
    }

    fn on_reject(&mut self, _candidate: &Vec<f64>, _delta: f64, _iter: usize) {
        self.after_step();
    }

    fn on_finish(&mut self) {
        self.emit_snapshot(true);
    }
}

impl SnapshotProvider for KnapsackSAStation {
    fn snapshot(&self, done: bool) -> SolverProgressPayload {
        let score = knapsack_score(&self.values, &self.weights, self.capacity, self.get_best());
        SolverProgressPayload {
            solver_id: self.id().to_string(),
            solver_kind: InternalSolverKind::KnapsackSa,
            tick: self.get_iteration(),
            iteration: self.get_iteration(),
            objective: self.get_best_cost(),
            feasible: score.weight <= self.capacity,
            best_state: SolverBestState::Knapsack {
                selected: score.selected,
                value: score.value,
                weight: score.weight,
                capacity: score.capacity,
            },
            done,
            metadata: vec![
                ("accepted".to_string(), MetaValue::Number(self.get_accepted_count() as f64)),
                ("improvements".to_string(), MetaValue::Number(self.get_improve_count() as f64)),
            ],
        }
    }
}

// =============================================================================
// ObservableTSPSAOptimizer — wraps TSPSAOptimizer, emits snapshots
// =============================================================================

/// SA-for-TSP solver that emits a progress snapshot after every accept / reject
/// / finish. (TS `class ObservableTSPSAOptimizer extends TSPSAOptimizer`.)
pub struct ObservableTSPSAOptimizer {
    core: StationCore,
    inner: TSPSAOptimizer,
    instance_ref: TSPInstance,
}

fn downcast_obs_sa(s: &dyn DESStation) -> &ObservableTSPSAOptimizer {
    s.as_any()
        .downcast_ref::<ObservableTSPSAOptimizer>()
        .expect("validator received a non-ObservableTSPSAOptimizer station")
}

impl ObservableTSPSAOptimizer {
    pub fn new(id: impl Into<String>, instance_ref: TSPInstance, opts: TSPSAOptions) -> Self {
        let id = id.into();
        let inner = TSPSAOptimizer::new(id.clone(), instance_ref.clone(), opts, false, None);
        let mut obs = ObservableTSPSAOptimizer { core: StationCore::new(id), inner, instance_ref };

        // Re-register the base optimizer's intrinsic / ground-truth validators,
        // downcasting to the wrapper (the base's own validators downcast to
        // TSPSAOptimizer, which the wrapper is not).
        obs.add_validator(
            monotonicity_validator::<dyn DESStation>(
                "sa.bestHistory.monotone",
                |s| downcast_obs_sa(s).opt_state().best_history.clone(),
                Monotonicity::NonIncreasing,
                1e-9,
                Some("sa-intrinsic".to_string()),
            )
            .boxed(),
        );
        obs.add_validator(
            intrinsic_check::<dyn DESStation>(
                "sa.best-is-valid-permutation",
                |s| {
                    let st = downcast_obs_sa(s);
                    is_permutation(st.get_best(), st.instance_ref.n)
                },
                Some("permutation of [0..n-1]".to_string()),
                Some(Box::new(|s| {
                    let st = downcast_obs_sa(s);
                    format!("n={}  bestLen={}", st.instance_ref.n, st.get_best().len())
                })),
                Some("sa-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        obs.add_validator(
            intrinsic_check::<dyn DESStation>(
                "sa.best-cost-nonnegative",
                |s| downcast_obs_sa(s).get_best_cost() >= 0.0,
                Some(">= 0".to_string()),
                Some(Box::new(|s| format!("bestCost={}", downcast_obs_sa(s).get_best_cost()))),
                Some("sa-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        if obs.instance_ref.n <= 12 && obs.instance_ref.precedence.is_none() {
            let exact = Rc::new(RefCell::new(None::<f64>));
            let e1 = exact.clone();
            let e2 = exact;
            obs.add_validator(
                intrinsic_check::<dyn DESStation>(
                    "sa.bestCost-vs-heldKarp-LB",
                    move |s| {
                        let st = downcast_obs_sa(s);
                        let mut cache = e1.borrow_mut();
                        if cache.is_none() {
                            *cache = Some(held_karp_exact(&st.instance_ref).length);
                        }
                        st.get_best_cost() >= cache.unwrap() - 1e-9
                    },
                    Some("bestCost >= heldKarp.length".to_string()),
                    Some(Box::new(move |s| {
                        let st = downcast_obs_sa(s);
                        let mut cache = e2.borrow_mut();
                        if cache.is_none() {
                            *cache = Some(held_karp_exact(&st.instance_ref).length);
                        }
                        format!("bestCost={:.4}  heldKarp={:.4}", st.get_best_cost(), cache.unwrap())
                    })),
                    Some("sa-ground-truth".to_string()),
                    Some("bestCost is below the true global optimum".to_string()),
                )
                .boxed(),
            );
        }
        obs
    }

    fn emit_snapshot(&mut self, done: bool) {
        let payload = self.snapshot(done);
        let token: AnyToken = Rc::new(SolverSolutionToken::new(payload));
        self.core.emit(token, SOLUTION_CHANNEL);
    }
}

impl DESStation for ObservableTSPSAOptimizer {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        self.optimizer_step();
    }
    fn has_work(&self) -> bool {
        self.optimizer_has_work()
    }
}

impl SingleStateOptimizer<Tour> for ObservableTSPSAOptimizer {
    fn opt_state(&self) -> &SingleStateState<Tour> {
        self.inner.opt_state()
    }
    fn opt_state_mut(&mut self) -> &mut SingleStateState<Tour> {
        self.inner.opt_state_mut()
    }
    fn initial_state(&self, rng: &mut dyn RandomSource) -> Tour {
        self.inner.initial_state(rng)
    }
    fn cost(&self, state: &Tour) -> f64 {
        self.inner.cost(state)
    }
    fn propose(&self, state: &Tour, rng: &mut dyn RandomSource) -> Tour {
        self.inner.propose(state, rng)
    }
    fn accept(
        &self,
        current: &Tour,
        candidate: &Tour,
        current_cost: f64,
        candidate_cost: f64,
        iter: usize,
        rng: &mut dyn RandomSource,
    ) -> bool {
        self.inner.accept(current, candidate, current_cost, candidate_cost, iter, rng)
    }
    fn should_stop(&self, iter: usize) -> bool {
        self.inner.should_stop(iter)
    }
    fn on_bootstrap(&mut self) {
        self.inner.on_bootstrap();
    }
    fn on_accept(&mut self, candidate: &Tour, delta: f64, iter: usize) {
        self.inner.on_accept(candidate, delta, iter);
        self.emit_snapshot(false);
    }
    fn on_reject(&mut self, candidate: &Tour, delta: f64, iter: usize) {
        self.inner.on_reject(candidate, delta, iter);
        self.emit_snapshot(false);
    }
    fn on_finish(&mut self) {
        self.emit_snapshot(true);
    }
}

impl SnapshotProvider for ObservableTSPSAOptimizer {
    fn snapshot(&self, done: bool) -> SolverProgressPayload {
        let best = self.get_best().clone();
        let length = tour_length(&self.instance_ref, &best);
        SolverProgressPayload {
            solver_id: self.id().to_string(),
            solver_kind: InternalSolverKind::TspSa,
            tick: self.get_iteration(),
            iteration: self.get_iteration(),
            objective: self.get_best_cost(),
            feasible: is_permutation(&best, self.instance_ref.n),
            best_state: SolverBestState::Tour { tour: best, length },
            done,
            metadata: vec![
                ("accepted".to_string(), MetaValue::Number(self.get_accepted_count() as f64)),
                ("improvements".to_string(), MetaValue::Number(self.get_improve_count() as f64)),
                ("n".to_string(), MetaValue::Number(self.instance_ref.n as f64)),
            ],
        }
    }
}

// =============================================================================
// ObservableTSPGAOptimizer — wraps TSPGAOptimizer, emits snapshots
// =============================================================================

/// GA-for-TSP solver that emits a snapshot every generation / on finish. (TS
/// `class ObservableTSPGAOptimizer extends TSPGAOptimizer`.)
pub struct ObservableTSPGAOptimizer {
    core: StationCore,
    inner: TSPGAOptimizer,
    instance_ref: TSPInstance,
}

fn downcast_obs_ga(s: &dyn DESStation) -> &ObservableTSPGAOptimizer {
    s.as_any()
        .downcast_ref::<ObservableTSPGAOptimizer>()
        .expect("validator received a non-ObservableTSPGAOptimizer station")
}

impl ObservableTSPGAOptimizer {
    pub fn new(id: impl Into<String>, instance_ref: TSPInstance, opts: TSPGAOptions) -> Self {
        let id = id.into();
        let inner = TSPGAOptimizer::new(id.clone(), instance_ref.clone(), opts, false, None);
        let mut obs = ObservableTSPGAOptimizer { core: StationCore::new(id), inner, instance_ref };

        if obs.inner.elite_count() >= 1 {
            obs.add_validator(
                monotonicity_validator::<dyn DESStation>(
                    "ga.bestHistory.monotone",
                    |s| downcast_obs_ga(s).opt_state().best_history.clone(),
                    Monotonicity::NonIncreasing,
                    1e-9,
                    Some("ga-intrinsic".to_string()),
                )
                .boxed(),
            );
        }
        obs.add_validator(
            intrinsic_check::<dyn DESStation>(
                "ga.best-is-valid-permutation",
                |s| {
                    let st = downcast_obs_ga(s);
                    is_permutation(st.get_best(), st.instance_ref.n)
                },
                Some("permutation of [0..n-1]".to_string()),
                Some(Box::new(|s| {
                    let st = downcast_obs_ga(s);
                    format!("n={}  bestLen={}", st.instance_ref.n, st.get_best().len())
                })),
                Some("ga-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        if obs.instance_ref.n <= 12 && obs.instance_ref.precedence.is_none() {
            let exact = Rc::new(RefCell::new(None::<f64>));
            let e1 = exact.clone();
            let e2 = exact;
            obs.add_validator(
                intrinsic_check::<dyn DESStation>(
                    "ga.bestLength-vs-heldKarp-LB",
                    move |s| {
                        let st = downcast_obs_ga(s);
                        let mut cache = e1.borrow_mut();
                        if cache.is_none() {
                            *cache = Some(held_karp_exact(&st.instance_ref).length);
                        }
                        st.get_best_fitness() >= cache.unwrap() - 1e-9
                    },
                    Some("bestLength >= heldKarp.length".to_string()),
                    Some(Box::new(move |s| {
                        let st = downcast_obs_ga(s);
                        let mut cache = e2.borrow_mut();
                        if cache.is_none() {
                            *cache = Some(held_karp_exact(&st.instance_ref).length);
                        }
                        format!("best={:.4}  heldKarp={:.4}", st.get_best_fitness(), cache.unwrap())
                    })),
                    Some("ga-ground-truth".to_string()),
                    Some("best length is below the global optimum".to_string()),
                )
                .boxed(),
            );
        }
        obs
    }

    fn emit_snapshot(&mut self, done: bool) {
        let payload = self.snapshot(done);
        let token: AnyToken = Rc::new(SolverSolutionToken::new(payload));
        self.core.emit(token, SOLUTION_CHANNEL);
    }
}

impl DESStation for ObservableTSPGAOptimizer {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        self.generation_step();
    }
    fn has_work(&self) -> bool {
        self.optimizer_has_work()
    }
}

impl PopulationOptimizer<Tour> for ObservableTSPGAOptimizer {
    fn opt_state(&self) -> &PopulationState<Tour> {
        self.inner.opt_state()
    }
    fn opt_state_mut(&mut self) -> &mut PopulationState<Tour> {
        self.inner.opt_state_mut()
    }
    fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<Tour> {
        self.inner.initial_population(size, rng)
    }
    fn evaluate(&self, individual: &Tour) -> f64 {
        self.inner.evaluate(individual)
    }
    fn select(&self, pop: &[Tour], fitness: &[f64], rng: &mut dyn RandomSource) -> Vec<Tour> {
        self.inner.select(pop, fitness, rng)
    }
    fn recombine(&self, parents: &[Tour], rng: &mut dyn RandomSource) -> Tour {
        self.inner.recombine(parents, rng)
    }
    fn mutate(&self, child: Tour, rng: &mut dyn RandomSource) -> Tour {
        self.inner.mutate(child, rng)
    }
    fn should_stop(&self, generation: usize) -> bool {
        self.inner.should_stop(generation)
    }
    fn elite_count(&self) -> usize {
        self.inner.elite_count()
    }
    fn on_bootstrap(&mut self) {
        self.inner.on_bootstrap();
    }
    fn on_generation(&mut self, gen: usize) {
        self.inner.on_generation(gen);
        self.emit_snapshot(false);
    }
    fn on_finish(&mut self) {
        self.emit_snapshot(true);
    }
}

impl SnapshotProvider for ObservableTSPGAOptimizer {
    fn snapshot(&self, done: bool) -> SolverProgressPayload {
        let best = self.get_best().clone();
        let length = tour_length(&self.instance_ref, &best);
        SolverProgressPayload {
            solver_id: self.id().to_string(),
            solver_kind: InternalSolverKind::TspGa,
            tick: self.get_generation(),
            iteration: self.get_generation(),
            objective: self.get_best_fitness(),
            feasible: is_permutation(&best, self.instance_ref.n),
            best_state: SolverBestState::Tour { tour: best, length },
            done,
            metadata: vec![
                ("n".to_string(), MetaValue::Number(self.instance_ref.n as f64)),
                ("population".to_string(), MetaValue::Number(self.get_population().len() as f64)),
            ],
        }
    }
}

// =============================================================================
// TSPHeldKarpStation
// =============================================================================

/// Exact Held–Karp TSP solver run in a single tick. (TS `class
/// TSPHeldKarpStation`.)
pub struct TSPHeldKarpStation {
    core: StationCore,
    instance: TSPInstance,
    done: bool,
    iter: usize,
    best: Option<HeldKarpResult>,
}

impl TSPHeldKarpStation {
    pub fn new(id: impl Into<String>, instance: TSPInstance) -> Self {
        TSPHeldKarpStation { core: StationCore::new(id), instance, done: false, iter: 0, best: None }
    }
}

impl DESStation for TSPHeldKarpStation {
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
        Preconditions_integer_in_range("TSPHeldKarpStation", "n", self.instance.n as f64, 3.0, 16.0);
    }

    fn has_work(&self) -> bool {
        !self.done
    }

    fn run_time_step(&mut self) {
        if self.done {
            return;
        }
        self.best = Some(held_karp_exact(&self.instance));
        self.done = true;
        self.iter += 1;
        let payload = self.snapshot(true);
        let token: AnyToken = Rc::new(SolverSolutionToken::new(payload));
        self.core.emit(token, SOLUTION_CHANNEL);
    }
}

impl SnapshotProvider for TSPHeldKarpStation {
    fn snapshot(&self, done: bool) -> SolverProgressPayload {
        let (tour, length) = match &self.best {
            Some(b) => (b.tour.clone(), b.length),
            None => (Vec::new(), f64::INFINITY),
        };
        let feasible = tour.is_empty() || is_permutation(&tour, self.instance.n);
        SolverProgressPayload {
            solver_id: self.id().to_string(),
            solver_kind: InternalSolverKind::TspHeldKarp,
            tick: self.iter,
            iteration: self.iter,
            objective: length,
            feasible,
            best_state: SolverBestState::Tour { tour, length },
            done,
            metadata: vec![
                ("n".to_string(), MetaValue::Number(self.instance.n as f64)),
                ("exact".to_string(), MetaValue::Bool(true)),
            ],
        }
    }
}

// =============================================================================
// Solver handle (heterogeneous solver station)
// =============================================================================

/// A type-erased handle to the concrete solver station so the driver can both
/// pipe it as a [`StationRef`] and call [`SnapshotProvider::snapshot`] on it.
/// (TS `buildSolverStation` returns `DESStation & SnapshotProvider`.)
enum SolverHandle {
    ShortestPath(Rc<RefCell<ShortestPathSolverStation>>),
    KnapsackDp(Rc<RefCell<KnapsackDPStation>>),
    KnapsackSa(Rc<RefCell<KnapsackSAStation>>),
    TspSa(Rc<RefCell<ObservableTSPSAOptimizer>>),
    TspGa(Rc<RefCell<ObservableTSPGAOptimizer>>),
    TspHeldKarp(Rc<RefCell<TSPHeldKarpStation>>),
}

impl SolverHandle {
    fn station_ref(&self) -> StationRef {
        match self {
            SolverHandle::ShortestPath(r) => r.clone() as StationRef,
            SolverHandle::KnapsackDp(r) => r.clone() as StationRef,
            SolverHandle::KnapsackSa(r) => r.clone() as StationRef,
            SolverHandle::TspSa(r) => r.clone() as StationRef,
            SolverHandle::TspGa(r) => r.clone() as StationRef,
            SolverHandle::TspHeldKarp(r) => r.clone() as StationRef,
        }
    }

    fn snapshot(&self, done: bool) -> SolverProgressPayload {
        match self {
            SolverHandle::ShortestPath(r) => r.borrow().snapshot(done),
            SolverHandle::KnapsackDp(r) => r.borrow().snapshot(done),
            SolverHandle::KnapsackSa(r) => r.borrow().snapshot(done),
            SolverHandle::TspSa(r) => r.borrow().snapshot(done),
            SolverHandle::TspGa(r) => r.borrow().snapshot(done),
            SolverHandle::TspHeldKarp(r) => r.borrow().snapshot(done),
        }
    }

    fn id(&self) -> String {
        match self {
            SolverHandle::ShortestPath(r) => r.borrow().id().to_string(),
            SolverHandle::KnapsackDp(r) => r.borrow().id().to_string(),
            SolverHandle::KnapsackSa(r) => r.borrow().id().to_string(),
            SolverHandle::TspSa(r) => r.borrow().id().to_string(),
            SolverHandle::TspGa(r) => r.borrow().id().to_string(),
            SolverHandle::TspHeldKarp(r) => r.borrow().id().to_string(),
        }
    }
}

// =============================================================================
// Driver
// =============================================================================

/// Wire up the solver, wall-clock checker, and sink; run the iterative DES; and
/// reduce the result. (TS `runInternalSolverNetwork`.)
pub fn run_internal_solver_network(params: InternalSolverRunParams) -> InternalSolverRunResult {
    let budget_ms = params.time_limit_ms.unwrap_or(180000.0);
    let checker = Rc::new(RefCell::new(WallClockCheckerStation::new(
        "wall-clock-checker",
        budget_ms,
        params.check_every_ticks.unwrap_or(1),
        None,
    )));
    let sink = Rc::new(RefCell::new(SolutionSinkStation::new("solution-sink")));
    let solver = build_solver_station(&params);

    let solver_ref = solver.station_ref();
    solver_ref.borrow_mut().core_mut().pipe(sink.clone() as StationRef, SOLUTION_CHANNEL, SOLUTION_CHANNEL);
    checker.borrow_mut().core_mut().pipe(sink.clone() as StationRef, STOP_CHANNEL, STOP_CHANNEL);

    let max_ticks = params.max_ticks.unwrap_or_else(|| default_max_ticks(&params));
    let stations: Vec<StationRef> =
        vec![solver_ref.clone(), checker.clone() as StationRef, sink.clone() as StationRef];

    let checker_for_stop = checker.clone();
    let summary = run_iterative_des(
        stations,
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_ticks),
            stop_when: Some(Box::new(move |_, _| checker_for_stop.borrow().expired())),
            ..Default::default()
        },
    );

    let reason_done = summary.reason == Some(RunReason::Done);
    let fallback = solver.snapshot(reason_done);
    let best = sink.borrow().best().unwrap_or(fallback);
    let expired = checker.borrow().expired();
    let status = if expired {
        InternalSolverStatus::TimeLimit
    } else if summary.reason == Some(RunReason::MaxTicks) {
        InternalSolverStatus::TickLimit
    } else {
        InternalSolverStatus::Complete
    };
    let validation = summary.validation.clone().unwrap_or_default();
    let wall_clock = WallClockReport {
        budget_ms,
        elapsed_ms: checker.borrow().elapsed_ms(),
        checks: checker.borrow().num_checks(),
        expired,
    };
    let network = describe_network(params.kind, &solver.id());
    let trace = sink.borrow().trace.clone();
    let stop_signals = sink.borrow().stops.clone();

    InternalSolverRunResult {
        kind: params.kind,
        status,
        run_summary: summary,
        best,
        trace,
        stop_signals,
        wall_clock,
        network,
        validation,
    }
}

fn build_solver_station(params: &InternalSolverRunParams) -> SolverHandle {
    match params.kind {
        InternalSolverKind::ShortestPath => SolverHandle::ShortestPath(Rc::new(RefCell::new(
            ShortestPathSolverStation::new(
                "shortest-path-solver",
                required(params.shortest_path.clone(), "shortestPath"),
            ),
        ))),
        InternalSolverKind::KnapsackDp => SolverHandle::KnapsackDp(Rc::new(RefCell::new(
            KnapsackDPStation::new("knapsack-dp-solver", required(params.knapsack.clone(), "knapsack")),
        ))),
        InternalSolverKind::KnapsackSa => SolverHandle::KnapsackSa(Rc::new(RefCell::new(
            KnapsackSAStation::new("knapsack-sa-solver", required(params.knapsack.clone(), "knapsack")),
        ))),
        InternalSolverKind::TspSa => {
            let tsp = required(params.tsp.clone(), "tsp");
            let inst = tsp_instance(&tsp);
            SolverHandle::TspSa(Rc::new(RefCell::new(ObservableTSPSAOptimizer::new(
                "tsp-sa-solver",
                inst,
                build_sa_options(&tsp),
            ))))
        }
        InternalSolverKind::TspGa => {
            let tsp = required(params.tsp.clone(), "tsp");
            let inst = tsp_instance(&tsp);
            SolverHandle::TspGa(Rc::new(RefCell::new(ObservableTSPGAOptimizer::new(
                "tsp-ga-solver",
                inst,
                build_ga_options(&tsp),
            ))))
        }
        InternalSolverKind::TspHeldKarp => {
            let tsp = required(params.tsp.clone(), "tsp");
            SolverHandle::TspHeldKarp(Rc::new(RefCell::new(TSPHeldKarpStation::new(
                "tsp-held-karp-solver",
                tsp_instance(&tsp),
            ))))
        }
    }
}

fn describe_network(kind: InternalSolverKind, solver_id: &str) -> SolverNetworkDescription {
    let kind_str = kind.as_str();
    SolverNetworkDescription {
        stationary_entities: vec![
            SolverNetworkNode {
                id: "initial-source".to_string(),
                kind: format!("{kind_str}-initial-source"),
                role: SolverNodeRole::Source,
            },
            SolverNetworkNode { id: solver_id.to_string(), kind: kind_str.to_string(), role: SolverNodeRole::Solver },
            SolverNetworkNode {
                id: "wall-clock-checker".to_string(),
                kind: "wall-clock-checker".to_string(),
                role: SolverNodeRole::Checker,
            },
            SolverNetworkNode {
                id: "solution-sink".to_string(),
                kind: "solution-sink".to_string(),
                role: SolverNodeRole::Sink,
            },
        ],
        moving_entities: vec![
            SolverNetworkMovingEntity {
                id: "SolverSolutionToken".to_string(),
                kind: "incumbent-solution".to_string(),
                token_type: "SolverSolutionToken".to_string(),
            },
            SolverNetworkMovingEntity {
                id: "StopSignalToken".to_string(),
                kind: "stop-signal".to_string(),
                token_type: "StopSignalToken".to_string(),
            },
        ],
        edges: vec![
            SolverNetworkEdge {
                from: "initial-source".to_string(),
                to: solver_id.to_string(),
                moving_entity: "initial-conditions".to_string(),
                channel: "constructor".to_string(),
            },
            SolverNetworkEdge {
                from: solver_id.to_string(),
                to: "solution-sink".to_string(),
                moving_entity: "SolverSolutionToken".to_string(),
                channel: SOLUTION_CHANNEL.to_string(),
            },
            SolverNetworkEdge {
                from: "wall-clock-checker".to_string(),
                to: "solution-sink".to_string(),
                moving_entity: "StopSignalToken".to_string(),
                channel: STOP_CHANNEL.to_string(),
            },
        ],
    }
}

// =============================================================================
// Free helpers
// =============================================================================

fn build_sa_options(tsp: &TSPSolverParams) -> TSPSAOptions {
    let sa = tsp.sa.clone().unwrap_or_default();
    TSPSAOptions {
        cooling: sa.cooling.unwrap_or(CoolingSchedule::Geometric { t0: 100.0, alpha: 0.995, t_min: Some(1e-6) }),
        max_iterations: sa.max_iterations.unwrap_or(5000),
        seed: sa.seed.or(tsp.seed).unwrap_or(1),
        init: Some(sa.init.unwrap_or(InitMode::NearestNeighbor)),
        moves: Some(sa.moves.unwrap_or(Moves::Mixed)),
        penalty_per_violation: sa.penalty_per_violation,
        trace_stride: sa.trace_stride,
        stall_limit: sa.stall_limit,
    }
}

fn build_ga_options(tsp: &TSPSolverParams) -> TSPGAOptions {
    let ga = tsp.ga.clone().unwrap_or_default();
    TSPGAOptions {
        pop_size: ga.pop_size.unwrap_or(60),
        num_generations: ga.num_generations.unwrap_or(200),
        tournament_size: ga.tournament_size,
        crossover_prob: ga.crossover_prob,
        mutation_prob: ga.mutation_prob,
        elitism: ga.elitism,
        seed: ga.seed.or(tsp.seed).unwrap_or(1),
        init: Some(ga.init.unwrap_or(InitMode::NearestNeighbor)),
        penalty_per_violation: ga.penalty_per_violation,
    }
}

fn graph_from_params(params: &ShortestPathSolverParams) -> Graph {
    if let Some(ShortestPathBuiltin::SmallChain) = params.builtin {
        return build_small_chain_graph();
    }
    if let Some(rg) = &params.random_graph {
        let mut rng = mulberry32(rg.seed);
        return build_random_graph(rg.num_nodes, rg.edge_prob, rg.w_min, rg.w_max, &mut rng);
    }
    if let Some(g) = &params.graph {
        return g.clone();
    }
    panic!("shortest-path solver requires builtin, graph, or randomGraph");
}

fn tsp_instance(params: &TSPSolverParams) -> TSPInstance {
    match params.builtin {
        Some(TspBuiltin::Pentagon) => build_pentagon_tsp(params.n.unwrap_or(5), 50.0),
        Some(TspBuiltin::Random) => {
            build_random_tsp(params.n.unwrap_or(20), params.seed.unwrap_or(1), params.precedence.clone())
        }
        None => {
            if let (Some(coords), Some(dist)) = (&params.coordinates, &params.distance) {
                TSPInstance {
                    n: coords.len(),
                    coordinates: coords.clone(),
                    distance: dist.clone(),
                    precedence: params.precedence.clone(),
                }
            } else {
                build_pentagon_tsp(params.n.unwrap_or(5), 50.0)
            }
        }
    }
}

fn validate_graph(graph: &Graph) {
    Preconditions_integer_in_range("ShortestPathSolverStation", "numNodes", graph.num_nodes as f64, 1.0, 100000.0);
    Preconditions_check(
        "ShortestPathSolverStation",
        "edges.length",
        "equal numNodes",
        graph.edges.len() == graph.num_nodes,
        Some(graph.edges.len().to_string()),
    );
    for u in 0..graph.num_nodes {
        for edge in &graph.edges[u] {
            Preconditions_integer_in_range(
                "ShortestPathSolverStation",
                &format!("edge {u}.to"),
                edge.to as f64,
                0.0,
                (graph.num_nodes as f64) - 1.0,
            );
            Preconditions_finite("ShortestPathSolverStation", &format!("edge {u}->{}.weight", edge.to), edge.weight);
        }
    }
}

fn validate_knapsack(values: &[f64], weights: &[f64], capacity: f64) {
    use crate::des::general::des_base::preconditions::Preconditions;
    Preconditions::non_empty("KnapsackSolver", "values", values).unwrap_or_else(|e| panic!("{e}"));
    Preconditions::length_eq("KnapsackSolver", "weights", weights, values.len())
        .unwrap_or_else(|e| panic!("{e}"));
    Preconditions_all_finite("KnapsackSolver", "values", values);
    Preconditions_all_finite("KnapsackSolver", "weights", weights);
    Preconditions_integer_in_range("KnapsackSolver", "capacity", capacity, 0.0, MAX_SAFE_INTEGER);
    for (i, &w) in weights.iter().enumerate() {
        Preconditions_integer_in_range("KnapsackSolver", &format!("weights[{i}]"), w, 0.0, MAX_SAFE_INTEGER);
    }
}

fn knapsack_score(values: &[f64], weights: &[f64], capacity: f64, selected: &[f64]) -> KnapsackSolution {
    let mut value = 0.0;
    let mut weight = 0.0;
    for i in 0..selected.len() {
        value += selected[i] * values[i];
        weight += selected[i] * weights[i];
    }
    KnapsackSolution { selected: selected.to_vec(), value, weight, capacity }
}

fn default_max_ticks(params: &InternalSolverRunParams) -> usize {
    match params.kind {
        InternalSolverKind::ShortestPath => 100000,
        InternalSolverKind::KnapsackDp => {
            params.knapsack.as_ref().map(|k| k.values.len()).unwrap_or(1000) + 2
        }
        InternalSolverKind::KnapsackSa => {
            params.knapsack.as_ref().and_then(|k| k.max_iterations).unwrap_or(5000) + 2
        }
        InternalSolverKind::TspSa => {
            params.tsp.as_ref().and_then(|t| t.sa.as_ref()).and_then(|s| s.max_iterations).unwrap_or(5000) + 2
        }
        InternalSolverKind::TspGa => {
            params.tsp.as_ref().and_then(|t| t.ga.as_ref()).and_then(|g| g.num_generations).unwrap_or(200) + 2
        }
        InternalSolverKind::TspHeldKarp => 2,
    }
}

fn required<T>(value: Option<T>, name: &str) -> T {
    value.unwrap_or_else(|| panic!("internal-solver-network: {name} parameters required"))
}

/// Downcast a `&dyn DESStation` to a concrete station type for validators.
fn downcast<T: 'static>(s: &dyn DESStation) -> &T {
    s.as_any().downcast_ref::<T>().expect("validator received an unexpected station type")
}

// -----------------------------------------------------------------------------
// Precondition wrappers (the TS `Preconditions.*` throw on failure; here we
// `panic!` with the error message — these are fatal invariant violations).
// -----------------------------------------------------------------------------

#[allow(non_snake_case)]
fn Preconditions_finite(model: &str, param: &str, x: f64) {
    crate::des::general::des_base::preconditions::Preconditions::finite(model, param, x)
        .unwrap_or_else(|e| panic!("{e}"));
}

#[allow(non_snake_case)]
fn Preconditions_non_negative(model: &str, param: &str, x: f64) {
    crate::des::general::des_base::preconditions::Preconditions::non_negative(model, param, x)
        .unwrap_or_else(|e| panic!("{e}"));
}

#[allow(non_snake_case)]
fn Preconditions_integer_in_range(model: &str, param: &str, x: f64, lo: f64, hi: f64) {
    crate::des::general::des_base::preconditions::Preconditions::integer_in_range(model, param, x, lo, hi)
        .unwrap_or_else(|e| panic!("{e}"));
}

#[allow(non_snake_case)]
fn Preconditions_all_finite(model: &str, param: &str, arr: &[f64]) {
    crate::des::general::des_base::preconditions::Preconditions::all_finite(model, param, arr)
        .unwrap_or_else(|e| panic!("{e}"));
}

#[allow(non_snake_case)]
fn Preconditions_check(model: &str, param: &str, condition: &str, ok: bool, observed: Option<String>) {
    crate::des::general::des_base::preconditions::Preconditions::check(model, param, condition, ok, observed)
        .unwrap_or_else(|e| panic!("{e}"));
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! Smoke tests for the deterministic solvers (shortest-path on the built-in
    //! small chain and knapsack DP). The wall-clock budget is large so the cap
    //! never fires; we only assert the networks run to completion and produce a
    //! feasible incumbent.

    use super::*;

    fn base_params(kind: InternalSolverKind) -> InternalSolverRunParams {
        InternalSolverRunParams {
            kind,
            time_limit_ms: Some(180000.0),
            max_ticks: None,
            check_every_ticks: None,
            shortest_path: None,
            knapsack: None,
            tsp: None,
        }
    }

    #[test]
    fn shortest_path_small_chain_completes() {
        let mut p = base_params(InternalSolverKind::ShortestPath);
        p.shortest_path = Some(ShortestPathSolverParams {
            algorithm: ShortestPathAlgorithm::BellmanFord,
            source: 0,
            builtin: Some(ShortestPathBuiltin::SmallChain),
            graph: None,
            random_graph: None,
        });
        let result = run_internal_solver_network(p);
        assert_eq!(result.status, InternalSolverStatus::Complete);
        assert!(result.best.feasible);
        assert!(!result.trace.is_empty());
    }

    #[test]
    fn knapsack_dp_finds_feasible_optimum() {
        let mut p = base_params(InternalSolverKind::KnapsackDp);
        p.knapsack = Some(KnapsackParams {
            values: vec![60.0, 100.0, 120.0],
            weights: vec![10.0, 20.0, 30.0],
            capacity: 50.0,
            seed: None,
            max_iterations: None,
            cooling: None,
            stall_limit: None,
            penalty: None,
        });
        let result = run_internal_solver_network(p);
        assert_eq!(result.status, InternalSolverStatus::Complete);
        assert!(result.best.feasible);
        // Optimal value for this classic instance is 220 (items 2 and 3).
        if let SolverBestState::Knapsack { value, weight, .. } = &result.best.best_state {
            assert!(*weight <= 50.0);
            assert!((*value - 220.0).abs() < 1e-9, "expected optimal value 220, got {value}");
        } else {
            panic!("expected a knapsack best-state");
        }
    }

    #[test]
    fn knapsack_dp_objective_is_negative_value() {
        let mut p = base_params(InternalSolverKind::KnapsackDp);
        p.knapsack = Some(KnapsackParams {
            values: vec![10.0, 40.0, 30.0, 50.0],
            weights: vec![5.0, 4.0, 6.0, 3.0],
            capacity: 10.0,
            seed: None,
            max_iterations: None,
            cooling: None,
            stall_limit: None,
            penalty: None,
        });
        let result = run_internal_solver_network(p);
        // best objective = -value, so it should be <= 0.
        assert!(result.best.objective <= 0.0);
    }
}
