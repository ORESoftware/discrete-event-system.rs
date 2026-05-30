//! Port of `src/des/general/advanced-optimization-models.ts` — concrete
//! metaheuristics built on the shared advanced-optimization DES bases: particle
//! swarm optimization, ant-colony TSP, map-coloring CSP, MAX-SAT local search,
//! an SDP Max-Cut relaxation, and a Pareto portfolio archive.
//!
//! Each model wires a one-shot source station to a solver station to a
//! latest-token sink and runs the iterative DES loop. The solver stations plug
//! concrete hooks into the template-method bases in
//! [`des_base::advanced_optimization`](crate::des::general::des_base::advanced_optimization),
//! [`des_base::single_state_optimizer`](crate::des::general::des_base::single_state_optimizer),
//! and [`des_base::tree_search`](crate::des::general::des_base::tree_search).
//!
//! ## TS to Rust mapping
//!
//!   * The `ContinuousObjectiveName` string union becomes an enum dispatched in
//!     the swarm objective; `*Params` / `*Result` / `Point2` / `WeightedEdge` /
//!     `PortfolioAsset` / `ParetoPortfolioPoint` interfaces become structs. The
//!     public `*Params` keep `Option` fields (TS optional fields) and a private
//!     resolved `*Config` holds the defaulted values that flow in a start token.
//!   * `class *Station extends <Base>` becomes a struct embedding the base state
//!     struct and implementing the base hook trait plus `DESStation`. The TS
//!     `runTimeStep` "drain start, bootstrap, run base step, emit result"
//!     pattern is reproduced directly.
//!   * `mulberry32(seed)` becomes an injected seeded `RandomSource`.
//!   * `Math.round` is never used here; integer literals stay integers
//!     (`usize` / `i64`).
//!   * `Preconditions.*` throws become `panic!` via the [`require`] helper.
//!   * The generic `OptimizationStartToken<P>` carries the resolved config; the
//!     `ParetoCandidateSourceStation<T>` is ported as a generic struct.

#![allow(dead_code)]

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;

use crate::des::general::des_base::advanced_optimization::{
    dominates, vector_dot, ConstraintSatisfactionSearchStation, ConstraintSearchCore,
    ConstraintSearchNode, NumericSwarmOptimizerStation, NumericSwarmParticle, NumericSwarmState,
    ParetoArchiveStation, ParetoCandidateToken, PheromoneGraphSearchStation, PheromoneGraphState,
    SourceDrivenConstraintSatisfactionSearchStation, SourceDrivenCspState,
    UnitVectorRelaxationState, UnitVectorRelaxationStation,
};
use crate::des::general::des_base::learning_optimization::{
    LatestTokenSinkStation, SingleTokenSourceStation,
};
use crate::des::general::des_base::model_topology::{station_graph_topology, StationGraphTopology};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::single_state_optimizer::{SingleStateOptimizer, SingleStateState};
use crate::des::general::des_base::station::{
    AnyToken, DESStation, StationCore, StationRef, DEFAULT_CHANNEL,
};
use crate::des::general::des_base::tree_search::{
    NodeEvaluation, SearchObjective, TreeSearchCore, TreeSearchStation,
};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

// =============================================================================
// Shared helpers
// =============================================================================

/// `throw` on a failed precondition (fatal invariant violation).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

fn sq(x: f64) -> f64 {
    x * x
}

fn dist(a: &Point2, b: &Point2) -> f64 {
    (a.x - b.x).hypot(a.y - b.y)
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    lo.max(hi.min(x))
}

/// Generic start token carrying a model tag and the resolved config. (TS
/// `class OptimizationStartToken<P>`.)
#[derive(Clone)]
struct OptimizationStartToken<P> {
    model: String,
    params: P,
}

impl<P> OptimizationStartToken<P> {
    fn new(model: impl Into<String>, params: P) -> Self {
        OptimizationStartToken { model: model.into(), params }
    }
}

// =============================================================================
// 21/23. Particle Swarm Optimization
// =============================================================================

/// Continuous benchmark objective. (TS `type ContinuousObjectiveName`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuousObjectiveName {
    Sphere,
    Rastrigin,
    Rosenbrock,
}

#[derive(Clone, Debug, Default)]
pub struct ParticleSwarmParams {
    pub objective: Option<ContinuousObjectiveName>,
    pub dimension: Option<usize>,
    pub particles: Option<usize>,
    pub iterations: Option<usize>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub inertia: Option<f64>,
    pub cognitive: Option<f64>,
    pub social: Option<f64>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
struct ParticleSwarmConfig {
    objective: ContinuousObjectiveName,
    dimension: usize,
    particles: usize,
    iterations: usize,
    lower: f64,
    upper: f64,
    inertia: f64,
    cognitive: f64,
    social: f64,
    seed: u32,
}

#[derive(Clone, Debug)]
pub struct ParticleSwarmTraceRow {
    pub iteration: usize,
    pub best_value: f64,
    pub mean_value: f64,
    pub worst_value: f64,
}

#[derive(Clone, Debug)]
pub struct ParticleSwarmResult {
    pub best_position: Vec<f64>,
    pub best_value: f64,
    pub iterations: usize,
    pub trace: Vec<ParticleSwarmTraceRow>,
    pub topology: StationGraphTopology,
}

struct ParticleSwarmResultToken {
    result: ParticleSwarmResult,
}

const PARTICLE_SWARM_CH_START: &str = "particle-swarm-start";
const PARTICLE_SWARM_CH_RESULT: &str = "particle-swarm-result";

struct ParticleSwarmStation {
    core: StationCore,
    state: NumericSwarmState,
    objective_name: ContinuousObjectiveName,
    inertia: f64,
    cognitive: f64,
    social: f64,
    max_velocity: f64,
    started: bool,
    result_emitted: bool,
}

impl ParticleSwarmStation {
    fn new(config: ParticleSwarmConfig) -> Self {
        let lower = vec![config.lower; config.dimension];
        let upper = vec![config.upper; config.dimension];
        let state = NumericSwarmState::new(
            config.particles,
            config.dimension,
            config.iterations,
            lower,
            upper,
            Box::new(mulberry32(config.seed)),
        );
        ParticleSwarmStation {
            core: StationCore::new("particle-swarm-station"),
            state,
            objective_name: config.objective,
            inertia: config.inertia,
            cognitive: config.cognitive,
            social: config.social,
            max_velocity: 0.25 * (config.upper - config.lower),
            started: false,
            result_emitted: false,
        }
    }

    fn result(&self) -> ParticleSwarmResult {
        ParticleSwarmResult {
            best_position: self.get_best_position(),
            best_value: self.get_best_score(),
            iterations: self.get_iteration(),
            trace: self
                .swarm_state()
                .trace
                .iter()
                .map(|row| ParticleSwarmTraceRow {
                    iteration: row.iteration,
                    best_value: row.best_score,
                    mean_value: row.mean_score.unwrap_or(f64::NAN),
                    worst_value: row.worst_score.unwrap_or(f64::NAN),
                })
                .collect(),
            topology: station_graph_topology(
                &[
                    "particle-swarm-source".to_string(),
                    "particle-swarm-station".to_string(),
                    "particle-swarm-result-sink".to_string(),
                ],
                &[
                    "OptimizationStartToken<particle-swarm>".to_string(),
                    "OptimizationCandidateToken<swarm-particle>".to_string(),
                    "NumericSwarmParticle".to_string(),
                    "ParticleSwarmResultToken".to_string(),
                ],
            ),
        }
    }
}

impl DESStation for ParticleSwarmStation {
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
        require(self.assert_preconditions_swarm());
        require(Preconditions::non_negative(self.id(), "inertia", self.inertia));
        require(Preconditions::non_negative(self.id(), "cognitive", self.cognitive));
        require(Preconditions::non_negative(self.id(), "social", self.social));
    }

    fn has_work(&self) -> bool {
        if !self.started {
            return self.core.inbox_size(PARTICLE_SWARM_CH_START) > 0;
        }
        !self.result_emitted
    }

    fn run_time_step(&mut self) {
        if !self.started {
            let starts =
                self.core_mut().drain::<OptimizationStartToken<ParticleSwarmConfig>>(PARTICLE_SWARM_CH_START);
            if starts.is_empty() {
                return;
            }
            validate_particle_swarm_params("particle-swarm-source", &starts[starts.len() - 1].params);
            self.bootstrap();
            self.started = true;
            return;
        }
        if !self.swarm_state().finished {
            self.run_swarm_step();
        }
        if self.swarm_state().finished && !self.result_emitted {
            let result = self.result();
            let token: AnyToken = Rc::new(ParticleSwarmResultToken { result });
            self.core_mut().emit(token, PARTICLE_SWARM_CH_RESULT);
            self.result_emitted = true;
        }
    }
}

impl NumericSwarmOptimizerStation for ParticleSwarmStation {
    fn swarm_state(&self) -> &NumericSwarmState {
        &self.state
    }
    fn swarm_state_mut(&mut self) -> &mut NumericSwarmState {
        &mut self.state
    }

    fn objective(&self, position: &[f64]) -> f64 {
        match self.objective_name {
            ContinuousObjectiveName::Rastrigin => {
                10.0 * position.len() as f64
                    + position.iter().map(|&x| x * x - 10.0 * (2.0 * PI * x).cos()).sum::<f64>()
            }
            ContinuousObjectiveName::Rosenbrock => {
                let mut value = 0.0;
                let mut i = 0;
                while i + 1 < position.len() {
                    value += 100.0 * sq(position[i + 1] - sq(position[i])) + sq(1.0 - position[i]);
                    i += 1;
                }
                value
            }
            ContinuousObjectiveName::Sphere => position.iter().map(|&x| x * x).sum(),
        }
    }

    fn update_particle(
        &self,
        mut particle: NumericSwarmParticle,
        global_best: &[f64],
        _iteration: usize,
        rng: &mut dyn RandomSource,
    ) -> NumericSwarmParticle {
        for i in 0..particle.position.len() {
            let rp = rng.next_float();
            let rg = rng.next_float();
            let velocity = self.inertia * particle.velocity[i]
                + self.cognitive * rp * (particle.best_position[i] - particle.position[i])
                + self.social * rg * (global_best[i] - particle.position[i]);
            particle.velocity[i] = clamp(velocity, -self.max_velocity, self.max_velocity);
            particle.position[i] += particle.velocity[i];
        }
        particle
    }
}

fn validate_particle_swarm_params(model: &str, params: &ParticleSwarmConfig) {
    require(Preconditions::integer_in_range(model, "dimension", params.dimension as f64, 1.0, 1e6));
    require(Preconditions::integer_in_range(model, "particles", params.particles as f64, 1.0, 1e9));
    require(Preconditions::integer_in_range(model, "iterations", params.iterations as f64, 1.0, 1e9));
    require(Preconditions::finite(model, "lower", params.lower));
    require(Preconditions::finite(model, "upper", params.upper));
    require(Preconditions::check(
        model,
        "bounds",
        "satisfy lower < upper",
        params.lower < params.upper,
        Some(format!("[{}, {}]", params.lower, params.upper)),
    ));
    require(Preconditions::non_negative(model, "inertia", params.inertia));
    require(Preconditions::non_negative(model, "cognitive", params.cognitive));
    require(Preconditions::non_negative(model, "social", params.social));
}

pub fn run_particle_swarm(params: ParticleSwarmParams) -> ParticleSwarmResult {
    let config = ParticleSwarmConfig {
        objective: params.objective.unwrap_or(ContinuousObjectiveName::Sphere),
        dimension: params.dimension.unwrap_or(3),
        particles: params.particles.unwrap_or(32),
        iterations: params.iterations.unwrap_or(120),
        lower: params.lower.unwrap_or(-5.0),
        upper: params.upper.unwrap_or(5.0),
        inertia: params.inertia.unwrap_or(0.68),
        cognitive: params.cognitive.unwrap_or(1.45),
        social: params.social.unwrap_or(1.45),
        seed: params.seed.unwrap_or(11),
    };
    validate_particle_swarm_params("runParticleSwarm", &config);
    let max_ticks = config.iterations + 5;
    let source = Rc::new(RefCell::new(SingleTokenSourceStation::with_validator(
        "particle-swarm-source",
        PARTICLE_SWARM_CH_START,
        {
            let d = config.clone();
            move || OptimizationStartToken::new("particle-swarm", d.clone())
        },
        |t: &OptimizationStartToken<ParticleSwarmConfig>| {
            validate_particle_swarm_params("particle-swarm-source", &t.params)
        },
    )));
    let station = Rc::new(RefCell::new(ParticleSwarmStation::new(config)));
    let sink = Rc::new(RefCell::new(LatestTokenSinkStation::<ParticleSwarmResultToken>::new(
        "particle-swarm-result-sink",
        PARTICLE_SWARM_CH_RESULT,
    )));
    source.borrow_mut().core_mut().pipe(station.clone() as StationRef, PARTICLE_SWARM_CH_START, PARTICLE_SWARM_CH_START);
    station.borrow_mut().core_mut().pipe(sink.clone() as StationRef, PARTICLE_SWARM_CH_RESULT, PARTICLE_SWARM_CH_RESULT);
    run_iterative_des(
        vec![source as StationRef, station as StationRef, sink.clone() as StationRef],
        IterativeRunOptions { shuffle: false, max_ticks: Some(max_ticks), run_validators: false, ..Default::default() },
    );
    let latest = sink.borrow().latest.clone();
    let token = latest.unwrap_or_else(|| panic!("particle-swarm did not produce a result"));
    token.result.clone()
}

// =============================================================================
// 24. Ant Colony Optimization on a TSP graph
// =============================================================================

#[derive(Clone, Copy, Debug)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug, Default)]
pub struct AntColonyTSPParams {
    pub points: Option<Vec<Point2>>,
    pub ants: Option<usize>,
    pub iterations: Option<usize>,
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub evaporation: Option<f64>,
    pub deposit: Option<f64>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
struct AntColonyTSPConfig {
    points: Vec<Point2>,
    ants: usize,
    iterations: usize,
    alpha: f64,
    beta: f64,
    evaporation: f64,
    deposit: f64,
    seed: u32,
}

#[derive(Clone, Debug)]
pub struct AntColonyTSPTraceRow {
    pub iteration: usize,
    pub best_length: f64,
    pub mean_length: f64,
    pub worst_length: f64,
}

#[derive(Clone, Debug)]
pub struct AntColonyTSPResult {
    pub best_tour: Vec<usize>,
    pub best_length: f64,
    pub iterations: usize,
    pub trace: Vec<AntColonyTSPTraceRow>,
    pub topology: StationGraphTopology,
}

struct AntColonyTSPResultToken {
    result: AntColonyTSPResult,
}

const ANT_COLONY_TSP_CH_START: &str = "ant-colony-tsp-start";
const ANT_COLONY_TSP_CH_RESULT: &str = "ant-colony-tsp-result";

struct AntColonyTSPStation {
    core: StationCore,
    state: PheromoneGraphState,
    points: Vec<Point2>,
    started: bool,
    result_emitted: bool,
}

impl AntColonyTSPStation {
    fn new(config: AntColonyTSPConfig) -> Self {
        let state = PheromoneGraphState::new(
            config.points.len(),
            config.ants,
            config.iterations,
            config.alpha,
            config.beta,
            config.evaporation,
            config.deposit,
            Box::new(mulberry32(config.seed)),
        );
        AntColonyTSPStation {
            core: StationCore::new("ant-colony-tsp-station"),
            state,
            points: config.points.iter().map(|p| Point2 { x: p.x, y: p.y }).collect(),
            started: false,
            result_emitted: false,
        }
    }

    fn result(&self) -> AntColonyTSPResult {
        AntColonyTSPResult {
            best_tour: self.get_best_path(),
            best_length: self.get_best_cost(),
            iterations: self.get_aco_iteration(),
            trace: self
                .aco_state()
                .trace
                .iter()
                .map(|row| AntColonyTSPTraceRow {
                    iteration: row.iteration,
                    best_length: row.best_score,
                    mean_length: row.mean_score.unwrap_or(f64::NAN),
                    worst_length: row.worst_score.unwrap_or(f64::NAN),
                })
                .collect(),
            topology: station_graph_topology(
                &[
                    "ant-colony-tsp-source".to_string(),
                    "ant-colony-tsp-station".to_string(),
                    "ant-colony-tsp-result-sink".to_string(),
                ],
                &[
                    "OptimizationStartToken<ant-colony-tsp>".to_string(),
                    "GraphWalkToken".to_string(),
                    "pheromone-matrix-state".to_string(),
                    "AntColonyTSPResultToken".to_string(),
                ],
            ),
        }
    }
}

impl DESStation for AntColonyTSPStation {
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
        require(self.assert_preconditions_aco());
    }

    fn has_work(&self) -> bool {
        if !self.started {
            return self.core.inbox_size(ANT_COLONY_TSP_CH_START) > 0;
        }
        !self.result_emitted
    }

    fn run_time_step(&mut self) {
        if !self.started {
            let starts =
                self.core_mut().drain::<OptimizationStartToken<AntColonyTSPConfig>>(ANT_COLONY_TSP_CH_START);
            if starts.is_empty() {
                return;
            }
            validate_ant_colony_tsp_params("ant-colony-tsp-source", &starts[starts.len() - 1].params);
            self.started = true;
            return;
        }
        if !self.aco_state().finished {
            self.run_aco_step();
        }
        if self.aco_state().finished && !self.result_emitted {
            let result = self.result();
            let token: AnyToken = Rc::new(AntColonyTSPResultToken { result });
            self.core_mut().emit(token, ANT_COLONY_TSP_CH_RESULT);
            self.result_emitted = true;
        }
    }
}

impl PheromoneGraphSearchStation for AntColonyTSPStation {
    fn aco_state(&self) -> &PheromoneGraphState {
        &self.state
    }
    fn aco_state_mut(&mut self) -> &mut PheromoneGraphState {
        &mut self.state
    }

    fn path_cost(&self, path: &[usize]) -> f64 {
        let mut value = 0.0;
        for i in 1..path.len() {
            value += dist(&self.points[path[i - 1]], &self.points[path[i]]);
        }
        value
    }

    fn heuristic(&self, from: usize, to: usize) -> f64 {
        1.0 / (1e-9_f64).max(dist(&self.points[from], &self.points[to]))
    }
}

fn validate_ant_colony_tsp_params(model: &str, params: &AntColonyTSPConfig) {
    let points = &params.points;
    require(Preconditions::check(model, "points.length", "be at least 2", points.len() >= 2, Some(points.len().to_string())));
    let mut seen_points: HashSet<String> = HashSet::new();
    for (i, p) in points.iter().enumerate() {
        require(Preconditions::finite(model, &format!("points[{i}].x"), p.x));
        require(Preconditions::finite(model, &format!("points[{i}].y"), p.y));
        let key = format!("{}:{}", p.x, p.y);
        require(Preconditions::check(
            model,
            &format!("points[{i}]"),
            "be a unique coordinate",
            !seen_points.contains(&key),
            Some(format!("({}, {})", p.x, p.y)),
        ));
        seen_points.insert(key);
    }
    require(Preconditions::integer_in_range(model, "ants", params.ants as f64, 1.0, 1e9));
    require(Preconditions::integer_in_range(model, "iterations", params.iterations as f64, 1.0, 1e9));
    require(Preconditions::non_negative(model, "alpha", params.alpha));
    require(Preconditions::non_negative(model, "beta", params.beta));
    require(Preconditions::in_range(model, "evaporation", params.evaporation, 0.0, 1.0));
    require(Preconditions::positive(model, "deposit", params.deposit));
}

pub fn run_ant_colony_tsp(params: AntColonyTSPParams) -> AntColonyTSPResult {
    let points = params.points.unwrap_or_else(default_tsp_points);
    let ants = params.ants.unwrap_or_else(|| 12usize.max(points.len() * 3));
    let config = AntColonyTSPConfig {
        points,
        ants,
        iterations: params.iterations.unwrap_or(80),
        alpha: params.alpha.unwrap_or(1.0),
        beta: params.beta.unwrap_or(3.0),
        evaporation: params.evaporation.unwrap_or(0.28),
        deposit: params.deposit.unwrap_or(1.0),
        seed: params.seed.unwrap_or(5),
    };
    validate_ant_colony_tsp_params("runAntColonyTSP", &config);
    let max_ticks = config.iterations + 5;
    let source = Rc::new(RefCell::new(SingleTokenSourceStation::with_validator(
        "ant-colony-tsp-source",
        ANT_COLONY_TSP_CH_START,
        {
            let d = config.clone();
            move || OptimizationStartToken::new("ant-colony-tsp", d.clone())
        },
        |t: &OptimizationStartToken<AntColonyTSPConfig>| {
            validate_ant_colony_tsp_params("ant-colony-tsp-source", &t.params)
        },
    )));
    let station = Rc::new(RefCell::new(AntColonyTSPStation::new(config)));
    let sink = Rc::new(RefCell::new(LatestTokenSinkStation::<AntColonyTSPResultToken>::new(
        "ant-colony-tsp-result-sink",
        ANT_COLONY_TSP_CH_RESULT,
    )));
    source.borrow_mut().core_mut().pipe(station.clone() as StationRef, ANT_COLONY_TSP_CH_START, ANT_COLONY_TSP_CH_START);
    station.borrow_mut().core_mut().pipe(sink.clone() as StationRef, ANT_COLONY_TSP_CH_RESULT, ANT_COLONY_TSP_CH_RESULT);
    run_iterative_des(
        vec![source as StationRef, station as StationRef, sink.clone() as StationRef],
        IterativeRunOptions { shuffle: false, max_ticks: Some(max_ticks), run_validators: false, ..Default::default() },
    );
    let latest = sink.borrow().latest.clone();
    let token = latest.unwrap_or_else(|| panic!("ant-colony-tsp did not produce a result"));
    token.result.clone()
}

fn default_tsp_points() -> Vec<Point2> {
    vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 { x: 1.5, y: 0.3 },
        Point2 { x: 2.4, y: 1.7 },
        Point2 { x: 1.4, y: 2.8 },
        Point2 { x: -0.2, y: 2.2 },
        Point2 { x: -0.8, y: 0.9 },
    ]
}

// =============================================================================
// 25. Constraint Satisfaction Problem: map coloring
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct MapColoringCSPParams {
    pub variables: Option<Vec<String>>,
    pub colors: Option<Vec<String>>,
    pub edges: Option<Vec<(String, String)>>,
    pub max_nodes: Option<usize>,
}

#[derive(Clone, Debug)]
struct MapColoringCSPConfig {
    variables: Vec<String>,
    colors: Vec<String>,
    edges: Vec<(String, String)>,
    max_nodes: usize,
}

#[derive(Clone, Debug)]
pub struct MapColoringCSPResult {
    pub assignment: HashMap<String, String>,
    pub satisfied: bool,
    pub nodes_processed: usize,
    pub topology: StationGraphTopology,
}

struct MapColoringCSPResultToken {
    result: MapColoringCSPResult,
}

const MAP_COLORING_CH_START: &str = "map-coloring-csp-start";
const MAP_COLORING_CH_RESULT: &str = "map-coloring-csp-result";

struct MapColoringCSPStation {
    core: StationCore,
    search: TreeSearchCore<ConstraintSearchNode<String>>,
    csp: ConstraintSearchCore<String>,
    source: SourceDrivenCspState,
    edges: Vec<(String, String)>,
}

impl MapColoringCSPStation {
    fn new(config: MapColoringCSPConfig) -> Self {
        let mut domains: HashMap<String, Vec<String>> = HashMap::new();
        for variable in &config.variables {
            domains.insert(variable.clone(), config.colors.clone());
        }
        MapColoringCSPStation {
            core: StationCore::new("map-coloring-csp-station"),
            search: TreeSearchCore::new(SearchObjective::Maximise, config.max_nodes as f64),
            csp: ConstraintSearchCore::new(&config.variables, &domains),
            source: SourceDrivenCspState::new(MAP_COLORING_CH_START, MAP_COLORING_CH_RESULT),
            edges: config.edges.iter().map(|(a, b)| (a.clone(), b.clone())).collect(),
        }
    }

    fn check_assignment(&self, assignment: &HashMap<String, String>) -> bool {
        self.get_variables().iter().all(|v| assignment.contains_key(v)) && self.is_consistent(assignment)
    }

    fn result(&self) -> MapColoringCSPResult {
        let assignment = self.get_solution().unwrap_or_default();
        let satisfied = self.check_assignment(&assignment);
        let nodes_processed = self.get_nodes_processed();
        MapColoringCSPResult {
            assignment,
            satisfied,
            nodes_processed,
            topology: station_graph_topology(
                &[
                    "map-coloring-csp-source".to_string(),
                    "map-coloring-csp-station".to_string(),
                    "map-coloring-csp-result-sink".to_string(),
                ],
                &[
                    "OptimizationStartToken<map-coloring-csp>".to_string(),
                    "ConstraintAssignmentToken".to_string(),
                    "MapColoringCSPResultToken".to_string(),
                ],
            ),
        }
    }
}

impl DESStation for MapColoringCSPStation {
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
        require(self.assert_preconditions_csp());
        let known: HashSet<String> = self.get_variables().into_iter().collect();
        for (a, b) in &self.edges {
            require(Preconditions::check(
                self.core.id.as_str(),
                &format!("edge {a}-{b}"),
                "reference known variables",
                known.contains(a) && known.contains(b),
                Some(format!("[{a}, {b}]")),
            ));
        }
    }

    fn has_work(&self) -> bool {
        self.source_has_work()
    }

    fn run_time_step(&mut self) {
        self.source_run_time_step();
    }
}

impl TreeSearchStation<ConstraintSearchNode<String>> for MapColoringCSPStation {
    fn search_core(&self) -> &TreeSearchCore<ConstraintSearchNode<String>> {
        &self.search
    }
    fn search_core_mut(&mut self) -> &mut TreeSearchCore<ConstraintSearchNode<String>> {
        &mut self.search
    }
    fn pick_next(&mut self) -> Option<ConstraintSearchNode<String>> {
        self.csp_pick_next()
    }
    fn evaluate(&mut self, node: &ConstraintSearchNode<String>) -> NodeEvaluation {
        self.csp_evaluate(node)
    }
    fn expand(
        &mut self,
        node: &ConstraintSearchNode<String>,
        _ev: &NodeEvaluation,
    ) -> Vec<ConstraintSearchNode<String>> {
        self.csp_expand(node)
    }
    fn push_children(&mut self, children: Vec<ConstraintSearchNode<String>>) {
        self.csp_push_children(children);
    }
    fn should_prune(&self, node: &ConstraintSearchNode<String>, ev: &NodeEvaluation) -> bool {
        self.csp_should_prune(node, ev)
    }
    fn current_best_bound(&self) -> f64 {
        self.csp_current_best_bound()
    }
    fn on_incumbent_update(&mut self, node: &ConstraintSearchNode<String>, _value: f64) {
        self.csp_on_incumbent_update(node);
    }
}

impl ConstraintSatisfactionSearchStation<String> for MapColoringCSPStation {
    fn csp_core(&self) -> &ConstraintSearchCore<String> {
        &self.csp
    }
    fn csp_core_mut(&mut self) -> &mut ConstraintSearchCore<String> {
        &mut self.csp
    }
    fn is_consistent(&self, assignment: &HashMap<String, String>) -> bool {
        for (a, b) in &self.edges {
            if let (Some(va), Some(vb)) = (assignment.get(a), assignment.get(b)) {
                if va == vb {
                    return false;
                }
            }
        }
        true
    }
}

impl
    SourceDrivenConstraintSatisfactionSearchStation<
        String,
        OptimizationStartToken<MapColoringCSPConfig>,
        MapColoringCSPResultToken,
    > for MapColoringCSPStation
{
    fn source_state(&self) -> &SourceDrivenCspState {
        &self.source
    }
    fn source_state_mut(&mut self) -> &mut SourceDrivenCspState {
        &mut self.source
    }

    fn accept_start_token(&mut self, token: Rc<OptimizationStartToken<MapColoringCSPConfig>>) {
        require(Preconditions::check(
            self.core.id.as_str(),
            "start model",
            "match map-coloring-csp",
            token.model == "map-coloring-csp",
            Some(token.model.clone()),
        ));
        validate_map_coloring_csp_params(&format!("{}.start", self.core.id), &token.params);
    }

    fn make_result_token(&mut self) -> Rc<MapColoringCSPResultToken> {
        Rc::new(MapColoringCSPResultToken { result: self.result() })
    }
}

fn unique_strings(values: &[String]) -> bool {
    let set: HashSet<&String> = values.iter().collect();
    set.len() == values.len()
}

fn validate_map_coloring_csp_params(model: &str, params: &MapColoringCSPConfig) {
    require(Preconditions::non_empty(model, "variables", &params.variables));
    require(Preconditions::non_empty(model, "colors", &params.colors));
    require(Preconditions::integer_in_range(model, "maxNodes", params.max_nodes as f64, 1.0, 1e9));
    require(Preconditions::check(
        model,
        "variables",
        "be unique",
        unique_strings(&params.variables),
        Some(params.variables.join(",")),
    ));
    require(Preconditions::check(model, "colors", "be unique", unique_strings(&params.colors), Some(params.colors.join(","))));
    let known: HashSet<&String> = params.variables.iter().collect();
    for (i, edge) in params.edges.iter().enumerate() {
        // A `(String, String)` tuple is always length 2 (TS `Preconditions.lengthEq(..., 2)`).
        let (a, b) = edge;
        require(Preconditions::check(
            model,
            &format!("edges[{i}]"),
            "reference known variables",
            known.contains(a) && known.contains(b),
            Some(format!("[{a}, {b}]")),
        ));
        require(Preconditions::check(
            model,
            &format!("edges[{i}]"),
            "connect two distinct variables",
            a != b,
            Some(format!("[{a}, {b}]")),
        ));
    }
}

pub fn run_map_coloring_csp(params: MapColoringCSPParams) -> MapColoringCSPResult {
    let config = MapColoringCSPConfig {
        variables: params.variables.unwrap_or_else(default_map_coloring_variables),
        colors: params.colors.unwrap_or_else(default_map_coloring_colors),
        edges: params.edges.unwrap_or_else(default_map_coloring_edges),
        max_nodes: params.max_nodes.unwrap_or(10_000),
    };
    validate_map_coloring_csp_params("runMapColoringCSP", &config);
    let max_ticks = config.max_nodes + 4;
    let source = Rc::new(RefCell::new(SingleTokenSourceStation::with_validator(
        "map-coloring-csp-source",
        MAP_COLORING_CH_START,
        {
            let d = config.clone();
            move || OptimizationStartToken::new("map-coloring-csp", d.clone())
        },
        |t: &OptimizationStartToken<MapColoringCSPConfig>| {
            validate_map_coloring_csp_params("map-coloring-csp-source", &t.params)
        },
    )));
    let station = Rc::new(RefCell::new(MapColoringCSPStation::new(config)));
    let sink = Rc::new(RefCell::new(LatestTokenSinkStation::<MapColoringCSPResultToken>::new(
        "map-coloring-csp-result-sink",
        MAP_COLORING_CH_RESULT,
    )));
    source.borrow_mut().core_mut().pipe(station.clone() as StationRef, MAP_COLORING_CH_START, MAP_COLORING_CH_START);
    station.borrow_mut().core_mut().pipe(sink.clone() as StationRef, MAP_COLORING_CH_RESULT, MAP_COLORING_CH_RESULT);
    run_iterative_des(
        vec![source as StationRef, station as StationRef, sink.clone() as StationRef],
        IterativeRunOptions { shuffle: false, max_ticks: Some(max_ticks), run_validators: false, ..Default::default() },
    );
    let result_token = sink.borrow().latest.clone();
    require(Preconditions::check(
        "runMapColoringCSP",
        "result token",
        "be emitted by map-coloring-csp-station",
        result_token.is_some(),
        Some("map-coloring-csp-result-sink".to_string()),
    ));
    let token = result_token.unwrap_or_else(|| panic!("runMapColoringCSP: result token was not emitted"));
    token.result.clone()
}

fn default_map_coloring_variables() -> Vec<String> {
    ["WA", "NT", "SA", "Q", "NSW", "V", "T"].iter().map(|s| s.to_string()).collect()
}

fn default_map_coloring_colors() -> Vec<String> {
    ["red", "green", "blue"].iter().map(|s| s.to_string()).collect()
}

fn default_map_coloring_edges() -> Vec<(String, String)> {
    [
        ("WA", "NT"),
        ("WA", "SA"),
        ("NT", "SA"),
        ("NT", "Q"),
        ("SA", "Q"),
        ("SA", "NSW"),
        ("SA", "V"),
        ("Q", "NSW"),
        ("NSW", "V"),
    ]
    .iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect()
}

// =============================================================================
// 26. SAT / MAX-SAT local search
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct MaxSATParams {
    pub num_vars: Option<usize>,
    pub clauses: Option<Vec<Vec<i64>>>,
    pub iterations: Option<usize>,
    pub noise: Option<f64>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
struct MaxSATConfig {
    num_vars: usize,
    clauses: Vec<Vec<i64>>,
    iterations: usize,
    noise: f64,
    seed: u32,
}

#[derive(Clone, Debug)]
pub struct MaxSATTraceRow {
    pub iteration: usize,
    pub unsatisfied: f64,
}

#[derive(Clone, Debug)]
pub struct MaxSATResult {
    pub assignment: Vec<bool>,
    pub satisfied_clauses: usize,
    pub total_clauses: usize,
    pub all_satisfied: bool,
    pub iterations: usize,
    pub trace: Vec<MaxSATTraceRow>,
    pub topology: StationGraphTopology,
}

struct MaxSATResultToken {
    result: MaxSATResult,
}

const MAX_SAT_CH_START: &str = "max-sat-local-search-start";
const MAX_SAT_CH_RESULT: &str = "max-sat-local-search-result";

struct MaxSATLocalSearchStation {
    core: StationCore,
    state: SingleStateState<Vec<bool>>,
    num_vars: usize,
    clauses: Vec<Vec<i64>>,
    iterations: usize,
    noise: f64,
    started: bool,
    max_sat_result_emitted: bool,
}

impl MaxSATLocalSearchStation {
    fn new(config: MaxSATConfig) -> Self {
        let mut st = MaxSATLocalSearchStation {
            core: StationCore::new("max-sat-local-search-station"),
            state: SingleStateState::new(1, Box::new(mulberry32(config.seed))),
            num_vars: config.num_vars,
            clauses: config.clauses.clone(),
            iterations: config.iterations,
            noise: config.noise,
            started: false,
            max_sat_result_emitted: false,
        };
        st.assert_preconditions();
        st
    }

    fn result(&self) -> MaxSATResult {
        let assignment = self.get_best().clone();
        let satisfied_clauses = count_satisfied(&self.clauses, &assignment);
        MaxSATResult {
            satisfied_clauses,
            total_clauses: self.clauses.len(),
            all_satisfied: satisfied_clauses == self.clauses.len(),
            iterations: self.get_iteration(),
            trace: self
                .opt_state()
                .best_history
                .iter()
                .enumerate()
                .map(|(i, &unsatisfied)| MaxSATTraceRow { iteration: i, unsatisfied })
                .collect(),
            assignment,
            topology: station_graph_topology(
                &[
                    "max-sat-local-search-source".to_string(),
                    "max-sat-local-search-station".to_string(),
                    "max-sat-local-search-result-sink".to_string(),
                ],
                &[
                    "OptimizationStartToken<max-sat>".to_string(),
                    "boolean-assignment-state".to_string(),
                    "OptimizationCandidateToken<boolean[]>".to_string(),
                    "MaxSATResultToken".to_string(),
                ],
            ),
        }
    }
}

impl DESStation for MaxSATLocalSearchStation {
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
        require(Preconditions::integer_in_range(self.core.id.as_str(), "numVars", self.num_vars as f64, 1.0, 1e9));
        require(Preconditions::non_empty(self.core.id.as_str(), "clauses", &self.clauses));
        require(Preconditions::integer_in_range(self.core.id.as_str(), "iterations", self.iterations as f64, 1.0, 1e9));
        require(Preconditions::in_range(self.core.id.as_str(), "noise", self.noise, 0.0, 1.0));
        for i in 0..self.clauses.len() {
            require(Preconditions::non_empty(self.core.id.as_str(), &format!("clauses[{i}]"), &self.clauses[i]));
            for &lit in &self.clauses[i] {
                let var = lit.unsigned_abs() as usize;
                require(Preconditions::check(
                    self.core.id.as_str(),
                    &format!("literal {lit}"),
                    "refer to a variable in [1, numVars]",
                    var >= 1 && var <= self.num_vars,
                    Some(lit.to_string()),
                ));
            }
        }
    }

    fn has_work(&self) -> bool {
        if !self.started {
            return self.core.inbox_size(MAX_SAT_CH_START) > 0;
        }
        !self.max_sat_result_emitted
    }

    fn run_time_step(&mut self) {
        if !self.started {
            let starts = self.core_mut().drain::<OptimizationStartToken<MaxSATConfig>>(MAX_SAT_CH_START);
            if starts.is_empty() {
                return;
            }
            validate_max_sat_params("max-sat-local-search-source", &starts[starts.len() - 1].params);
            self.bootstrap();
            self.started = true;
            return;
        }
        if !self.opt_state().finished {
            self.optimizer_step();
        }
        if self.opt_state().finished && !self.max_sat_result_emitted {
            let result = self.result();
            let token: AnyToken = Rc::new(MaxSATResultToken { result });
            self.core_mut().emit(token, MAX_SAT_CH_RESULT);
            self.max_sat_result_emitted = true;
        }
    }
}

impl SingleStateOptimizer<Vec<bool>> for MaxSATLocalSearchStation {
    fn opt_state(&self) -> &SingleStateState<Vec<bool>> {
        &self.state
    }
    fn opt_state_mut(&mut self) -> &mut SingleStateState<Vec<bool>> {
        &mut self.state
    }

    fn initial_state(&self, rng: &mut dyn RandomSource) -> Vec<bool> {
        random_boolean_assignment(self.num_vars, rng)
    }

    fn cost(&self, state: &Vec<bool>) -> f64 {
        self.clauses.len() as f64 - count_satisfied(&self.clauses, state) as f64
    }

    fn propose(&self, state: &Vec<bool>, rng: &mut dyn RandomSource) -> Vec<bool> {
        let unsat: Vec<&Vec<i64>> = self.clauses.iter().filter(|clause| !clause_satisfied(clause, state)).collect();
        let mut next = state.clone();
        if unsat.is_empty() {
            return next;
        }
        let clause = unsat[(rng.next_float() * unsat.len() as f64).floor() as usize];
        let mut variable = (clause[(rng.next_float() * clause.len() as f64).floor() as usize].unsigned_abs() as usize) - 1;
        if rng.next_float() >= self.noise {
            let mut best_var = variable;
            let mut best_score = f64::NEG_INFINITY;
            for &lit in clause {
                let idx = (lit.unsigned_abs() as usize) - 1;
                let mut trial = state.clone();
                trial[idx] = !trial[idx];
                let score = count_satisfied(&self.clauses, &trial) as f64;
                if score > best_score {
                    best_score = score;
                    best_var = idx;
                }
            }
            variable = best_var;
        }
        next[variable] = !next[variable];
        next
    }

    fn accept(
        &self,
        _current: &Vec<bool>,
        _candidate: &Vec<bool>,
        _current_cost: f64,
        _candidate_cost: f64,
        _iter: usize,
        _rng: &mut dyn RandomSource,
    ) -> bool {
        true
    }

    fn should_stop(&self, iter: usize) -> bool {
        iter >= self.iterations || self.get_best_cost() == 0.0
    }
}

fn validate_max_sat_params(model: &str, params: &MaxSATConfig) {
    require(Preconditions::integer_in_range(model, "numVars", params.num_vars as f64, 1.0, 1e9));
    require(Preconditions::non_empty(model, "clauses", &params.clauses));
    require(Preconditions::integer_in_range(model, "iterations", params.iterations as f64, 1.0, 1e9));
    require(Preconditions::in_range(model, "noise", params.noise, 0.0, 1.0));
    for (i, clause) in params.clauses.iter().enumerate() {
        require(Preconditions::non_empty(model, &format!("clauses[{i}]"), clause));
        for &lit in clause {
            let var = lit.unsigned_abs() as usize;
            require(Preconditions::check(
                model,
                &format!("literal {lit}"),
                "refer to a variable in [1, numVars]",
                var >= 1 && var <= params.num_vars,
                Some(lit.to_string()),
            ));
        }
    }
}

pub fn run_max_sat_local_search(params: MaxSATParams) -> MaxSATResult {
    let clauses = match params.clauses {
        Some(c) if !c.is_empty() => c,
        _ => default_max_sat_clauses(),
    };
    let inferred_vars = clauses
        .iter()
        .flat_map(|clause| clause.iter().map(|lit| lit.unsigned_abs() as usize))
        .max()
        .unwrap_or(0);
    let config = MaxSATConfig {
        num_vars: params.num_vars.unwrap_or(inferred_vars),
        clauses,
        iterations: params.iterations.unwrap_or(300),
        noise: params.noise.unwrap_or(0.25),
        seed: params.seed.unwrap_or(13),
    };
    validate_max_sat_params("runMaxSATLocalSearch", &config);
    let max_ticks = config.iterations + 5;
    let source = Rc::new(RefCell::new(SingleTokenSourceStation::with_validator(
        "max-sat-local-search-source",
        MAX_SAT_CH_START,
        {
            let d = config.clone();
            move || OptimizationStartToken::new("max-sat-local-search", d.clone())
        },
        |t: &OptimizationStartToken<MaxSATConfig>| validate_max_sat_params("max-sat-local-search-source", &t.params),
    )));
    let station = Rc::new(RefCell::new(MaxSATLocalSearchStation::new(config)));
    let sink = Rc::new(RefCell::new(LatestTokenSinkStation::<MaxSATResultToken>::new(
        "max-sat-local-search-result-sink",
        MAX_SAT_CH_RESULT,
    )));
    source.borrow_mut().core_mut().pipe(station.clone() as StationRef, MAX_SAT_CH_START, MAX_SAT_CH_START);
    station.borrow_mut().core_mut().pipe(sink.clone() as StationRef, MAX_SAT_CH_RESULT, MAX_SAT_CH_RESULT);
    run_iterative_des(
        vec![source as StationRef, station as StationRef, sink.clone() as StationRef],
        IterativeRunOptions { shuffle: false, max_ticks: Some(max_ticks), run_validators: false, ..Default::default() },
    );
    let latest = sink.borrow().latest.clone();
    let token = latest.unwrap_or_else(|| panic!("max-sat-local-search did not produce a result"));
    token.result.clone()
}

fn random_boolean_assignment(num_vars: usize, rng: &mut dyn RandomSource) -> Vec<bool> {
    (0..num_vars).map(|_| rng.next_float() < 0.5).collect()
}

fn default_max_sat_clauses() -> Vec<Vec<i64>> {
    vec![
        vec![1, 2, -3],
        vec![-1, 3],
        vec![2, 4],
        vec![-2, -4],
        vec![1, -4],
        vec![-1, -2, 3],
    ]
}

fn literal_satisfied(lit: i64, assignment: &[bool]) -> bool {
    let value = assignment[(lit.unsigned_abs() as usize) - 1];
    if lit > 0 {
        value
    } else {
        !value
    }
}

fn clause_satisfied(clause: &[i64], assignment: &[bool]) -> bool {
    clause.iter().any(|&lit| literal_satisfied(lit, assignment))
}

fn count_satisfied(clauses: &[Vec<i64>], assignment: &[bool]) -> usize {
    clauses.iter().filter(|clause| clause_satisfied(clause, assignment)).count()
}

// =============================================================================
// 27. SDP relaxation: Max-Cut via rank-constrained unit vectors
// =============================================================================

#[derive(Clone, Debug)]
pub struct WeightedEdge {
    pub i: usize,
    pub j: usize,
    pub weight: f64,
}

#[derive(Clone, Debug, Default)]
pub struct SDPMaxCutParams {
    pub nodes: Option<usize>,
    pub edges: Option<Vec<WeightedEdge>>,
    pub rank: Option<usize>,
    pub iterations: Option<usize>,
    pub step_size: Option<f64>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
struct SDPMaxCutConfig {
    nodes: usize,
    edges: Vec<WeightedEdge>,
    rank: usize,
    iterations: usize,
    step_size: f64,
    seed: u32,
}

#[derive(Clone, Debug)]
pub struct SDPMaxCutTraceRow {
    pub iteration: usize,
    pub objective: f64,
}

#[derive(Clone, Debug)]
pub struct SDPMaxCutResult {
    pub sdp_value: f64,
    pub rounded_cut_value: f64,
    pub cut: Vec<f64>,
    pub gram_matrix: Vec<Vec<f64>>,
    pub iterations: usize,
    pub trace: Vec<SDPMaxCutTraceRow>,
    pub topology: StationGraphTopology,
}

struct SDPMaxCutResultToken {
    result: SDPMaxCutResult,
}

const SDP_MAXCUT_CH_START: &str = "sdp-maxcut-relaxation-start";
const SDP_MAXCUT_CH_RESULT: &str = "sdp-maxcut-relaxation-result";

struct MaxCutSDPStation {
    core: StationCore,
    state: UnitVectorRelaxationState,
    edges: Vec<WeightedEdge>,
    started: bool,
    result_emitted: bool,
}

impl MaxCutSDPStation {
    fn new(config: SDPMaxCutConfig) -> Self {
        let state = UnitVectorRelaxationState::new(
            config.nodes,
            config.rank,
            config.iterations,
            config.step_size,
            Box::new(mulberry32(config.seed)),
        );
        let mut st = MaxCutSDPStation {
            core: StationCore::new("sdp-maxcut-relaxation-station"),
            state,
            edges: config.edges.iter().cloned().collect(),
            started: false,
            result_emitted: false,
        };
        st.assert_preconditions();
        st
    }

    fn result(&self) -> SDPMaxCutResult {
        let (cut, rounded_value) = best_hyperplane_cut(&self.get_best_vectors(), &self.edges);
        SDPMaxCutResult {
            sdp_value: self.get_best_objective(),
            rounded_cut_value: rounded_value,
            cut,
            gram_matrix: self.get_gram_matrix(),
            iterations: self.get_uvr_iteration(),
            trace: self
                .uvr_state()
                .trace
                .iter()
                .map(|row| SDPMaxCutTraceRow { iteration: row.iteration, objective: row.objective })
                .collect(),
            topology: station_graph_topology(
                &[
                    "sdp-maxcut-relaxation-source".to_string(),
                    "sdp-maxcut-relaxation-station".to_string(),
                    "sdp-maxcut-relaxation-result-sink".to_string(),
                ],
                &[
                    "OptimizationStartToken<sdp-maxcut>".to_string(),
                    "unit-vector-state".to_string(),
                    "GramMatrixToken".to_string(),
                    "SDPMaxCutResultToken".to_string(),
                ],
            ),
        }
    }
}

impl DESStation for MaxCutSDPStation {
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
        require(self.assert_preconditions_uvr());
        let nodes = self.uvr_state().nodes;
        for edge in &self.edges {
            require(Preconditions::integer_in_range(self.core.id.as_str(), "edge.i", edge.i as f64, 0.0, (nodes - 1) as f64));
            require(Preconditions::integer_in_range(self.core.id.as_str(), "edge.j", edge.j as f64, 0.0, (nodes - 1) as f64));
            require(Preconditions::positive(self.core.id.as_str(), "edge.weight", edge.weight));
        }
    }

    fn has_work(&self) -> bool {
        if !self.started {
            return self.core.inbox_size(SDP_MAXCUT_CH_START) > 0;
        }
        !self.result_emitted
    }

    fn run_time_step(&mut self) {
        if !self.started {
            let starts = self.core_mut().drain::<OptimizationStartToken<SDPMaxCutConfig>>(SDP_MAXCUT_CH_START);
            if starts.is_empty() {
                return;
            }
            validate_sdp_max_cut_params("sdp-maxcut-relaxation-source", &starts[starts.len() - 1].params);
            self.bootstrap();
            self.started = true;
            return;
        }
        if !self.uvr_state().finished {
            self.run_relaxation_step();
        }
        if self.uvr_state().finished && !self.result_emitted {
            let result = self.result();
            let token: AnyToken = Rc::new(SDPMaxCutResultToken { result });
            self.core_mut().emit(token, SDP_MAXCUT_CH_RESULT);
            self.result_emitted = true;
        }
    }
}

impl UnitVectorRelaxationStation for MaxCutSDPStation {
    fn uvr_state(&self) -> &UnitVectorRelaxationState {
        &self.state
    }
    fn uvr_state_mut(&mut self) -> &mut UnitVectorRelaxationState {
        &mut self.state
    }

    fn objective(&self, vectors: &[Vec<f64>]) -> f64 {
        let mut value = 0.0;
        for edge in &self.edges {
            value += edge.weight * (1.0 - vector_dot(&vectors[edge.i], &vectors[edge.j])) / 2.0;
        }
        value
    }

    fn gradient(&self, vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let nodes = self.uvr_state().nodes;
        let rank = self.uvr_state().rank;
        let mut grad = vec![vec![0.0_f64; rank]; nodes];
        for edge in &self.edges {
            for k in 0..rank {
                grad[edge.i][k] += -0.5 * edge.weight * vectors[edge.j][k];
                grad[edge.j][k] += -0.5 * edge.weight * vectors[edge.i][k];
            }
        }
        grad
    }
}

fn validate_sdp_max_cut_params(model: &str, params: &SDPMaxCutConfig) {
    require(Preconditions::integer_in_range(model, "nodes", params.nodes as f64, 2.0, 1e6));
    require(Preconditions::integer_in_range(model, "rank", params.rank as f64, 1.0, 1e6));
    require(Preconditions::integer_in_range(model, "iterations", params.iterations as f64, 1.0, 1e9));
    require(Preconditions::positive(model, "stepSize", params.step_size));
    require(Preconditions::non_empty(model, "edges", &params.edges));
    for edge in &params.edges {
        require(Preconditions::integer_in_range(model, "edge.i", edge.i as f64, 0.0, (params.nodes - 1) as f64));
        require(Preconditions::integer_in_range(model, "edge.j", edge.j as f64, 0.0, (params.nodes - 1) as f64));
        require(Preconditions::positive(model, "edge.weight", edge.weight));
    }
}

pub fn run_sdp_max_cut_relaxation(params: SDPMaxCutParams) -> SDPMaxCutResult {
    let edges = match params.edges {
        Some(e) if !e.is_empty() => e,
        _ => default_max_cut_edges(),
    };
    let config = SDPMaxCutConfig {
        nodes: params.nodes.unwrap_or(5),
        edges,
        rank: params.rank.unwrap_or(3),
        iterations: params.iterations.unwrap_or(250),
        step_size: params.step_size.unwrap_or(0.08),
        seed: params.seed.unwrap_or(17),
    };
    validate_sdp_max_cut_params("runSDPMaxCutRelaxation", &config);
    let max_ticks = config.iterations + 5;
    let source = Rc::new(RefCell::new(SingleTokenSourceStation::with_validator(
        "sdp-maxcut-relaxation-source",
        SDP_MAXCUT_CH_START,
        {
            let d = config.clone();
            move || OptimizationStartToken::new("sdp-maxcut-relaxation", d.clone())
        },
        |t: &OptimizationStartToken<SDPMaxCutConfig>| validate_sdp_max_cut_params("sdp-maxcut-relaxation-source", &t.params),
    )));
    let station = Rc::new(RefCell::new(MaxCutSDPStation::new(config)));
    let sink = Rc::new(RefCell::new(LatestTokenSinkStation::<SDPMaxCutResultToken>::new(
        "sdp-maxcut-relaxation-result-sink",
        SDP_MAXCUT_CH_RESULT,
    )));
    source.borrow_mut().core_mut().pipe(station.clone() as StationRef, SDP_MAXCUT_CH_START, SDP_MAXCUT_CH_START);
    station.borrow_mut().core_mut().pipe(sink.clone() as StationRef, SDP_MAXCUT_CH_RESULT, SDP_MAXCUT_CH_RESULT);
    run_iterative_des(
        vec![source as StationRef, station as StationRef, sink.clone() as StationRef],
        IterativeRunOptions { shuffle: false, max_ticks: Some(max_ticks), run_validators: false, ..Default::default() },
    );
    let latest = sink.borrow().latest.clone();
    let token = latest.unwrap_or_else(|| panic!("sdp-maxcut-relaxation did not produce a result"));
    token.result.clone()
}

fn default_max_cut_edges() -> Vec<WeightedEdge> {
    vec![
        WeightedEdge { i: 0, j: 1, weight: 1.0 },
        WeightedEdge { i: 1, j: 2, weight: 1.0 },
        WeightedEdge { i: 2, j: 3, weight: 1.0 },
        WeightedEdge { i: 3, j: 4, weight: 1.0 },
        WeightedEdge { i: 4, j: 0, weight: 1.0 },
        WeightedEdge { i: 0, j: 2, weight: 0.7 },
        WeightedEdge { i: 1, j: 3, weight: 0.5 },
        WeightedEdge { i: 2, j: 4, weight: 0.8 },
    ]
}

fn best_hyperplane_cut(vectors: &[Vec<f64>], edges: &[WeightedEdge]) -> (Vec<f64>, f64) {
    let dim = vectors.first().map(|v| v.len()).unwrap_or(1);
    let mut directions: Vec<Vec<f64>> = vectors.to_vec();
    directions.extend(basis_directions(dim));
    let mut best_cut = vec![0.0; vectors.len()];
    let mut best_value = f64::NEG_INFINITY;
    for dir in &directions {
        let cut: Vec<f64> = vectors.iter().map(|v| if vector_dot(v, dir) >= 0.0 { 1.0 } else { -1.0 }).collect();
        let value = cut_value(&cut, edges);
        if value > best_value {
            best_value = value;
            best_cut = cut;
        }
    }
    (best_cut, best_value)
}

fn basis_directions(rank: usize) -> Vec<Vec<f64>> {
    (0..rank)
        .map(|i| {
            let mut v = vec![0.0; rank];
            v[i] = 1.0;
            v
        })
        .collect()
}

fn cut_value(cut: &[f64], edges: &[WeightedEdge]) -> f64 {
    let mut value = 0.0;
    for edge in edges {
        if cut[edge.i] != cut[edge.j] {
            value += edge.weight;
        }
    }
    value
}

// =============================================================================
// 29. Multi-objective optimization: Pareto portfolio archive
// =============================================================================

#[derive(Clone, Debug)]
pub struct PortfolioAsset {
    pub name: String,
    pub expected_return: f64,
    pub risk: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ParetoPortfolioParams {
    pub assets: Option<Vec<PortfolioAsset>>,
    pub samples: Option<usize>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct ParetoPortfolioPoint {
    pub weights: Vec<f64>,
    pub expected_return: f64,
    pub risk: f64,
}

#[derive(Clone, Debug)]
pub struct ParetoPortfolioResult {
    pub pareto_front: Vec<ParetoPortfolioPoint>,
    pub candidate_count: usize,
    pub hypervolume: f64,
    pub topology: StationGraphTopology,
}

/// Source station that emits a fixed batch of Pareto candidates exactly once.
/// (TS `class ParetoCandidateSourceStation<T>`.)
struct ParetoCandidateSourceStation<T> {
    core: StationCore,
    candidates: Vec<ParetoCandidateToken<T>>,
    emitted: bool,
}

impl<T: Clone + 'static> ParetoCandidateSourceStation<T> {
    fn new(id: impl Into<String>, candidates: Vec<ParetoCandidateToken<T>>) -> Self {
        ParetoCandidateSourceStation { core: StationCore::new(id), candidates, emitted: false }
    }
}

impl<T: Clone + 'static> DESStation for ParetoCandidateSourceStation<T> {
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
        require(Preconditions::non_empty(self.core.id.as_str(), "candidates", &self.candidates));
        for candidate in &self.candidates {
            require(Preconditions::non_empty(self.core.id.as_str(), "candidate.objectives", &candidate.objectives));
            require(Preconditions::all_finite(self.core.id.as_str(), "candidate.objectives", &candidate.objectives));
        }
    }

    fn has_work(&self) -> bool {
        !self.emitted
    }

    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        for candidate in &self.candidates {
            let token: AnyToken = Rc::new(candidate.clone());
            self.core.emit(token, DEFAULT_CHANNEL);
        }
        self.emitted = true;
    }
}

pub fn run_pareto_portfolio(params: ParetoPortfolioParams) -> ParetoPortfolioResult {
    let assets = match params.assets {
        Some(a) if !a.is_empty() => a,
        _ => default_portfolio_assets(),
    };
    let samples = params.samples.unwrap_or(240);
    require(Preconditions::integer_in_range("runParetoPortfolio", "samples", samples as f64, 1.0, 1e9));
    for asset in &assets {
        require(Preconditions::finite("runParetoPortfolio", &format!("{}.expectedReturn", asset.name), asset.expected_return));
        require(Preconditions::non_negative("runParetoPortfolio", &format!("{}.risk", asset.name), asset.risk));
    }
    let mut rng = mulberry32(params.seed.unwrap_or(19));
    let mut candidates: Vec<ParetoCandidateToken<ParetoPortfolioPoint>> = Vec::new();
    for i in 0..samples {
        let weights = random_simplex(assets.len(), &mut rng);
        let point = portfolio_point(&assets, &weights);
        let objectives = vec![point.risk, -point.expected_return];
        candidates.push(ParetoCandidateToken::with_generation(point, objectives, i));
    }
    for i in 0..assets.len() {
        let mut weights = vec![0.0; assets.len()];
        weights[i] = 1.0;
        let point = portfolio_point(&assets, &weights);
        let objectives = vec![point.risk, -point.expected_return];
        candidates.push(ParetoCandidateToken::with_generation(point, objectives, samples + i));
    }
    let max_ticks = candidates.len() + 3;
    let source = Rc::new(RefCell::new(ParetoCandidateSourceStation::new("pareto-portfolio-source", candidates)));
    let station = Rc::new(RefCell::new(ParetoArchiveStation::<ParetoPortfolioPoint>::new(
        "pareto-portfolio-archive",
        Vec::new(),
    )));
    source.borrow_mut().core_mut().pipe(station.clone() as StationRef, DEFAULT_CHANNEL, DEFAULT_CHANNEL);
    run_iterative_des(
        vec![source as StationRef, station.clone() as StationRef],
        IterativeRunOptions { shuffle: false, max_ticks: Some(max_ticks), run_validators: false, ..Default::default() },
    );
    let mut pareto_front: Vec<ParetoPortfolioPoint> =
        station.borrow().get_archive().into_iter().map(|row| row.candidate).collect();
    pareto_front.sort_by(|a, b| {
        a.risk
            .partial_cmp(&b.risk)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.expected_return.partial_cmp(&b.expected_return).unwrap_or(std::cmp::Ordering::Equal))
    });
    let candidate_count = station.borrow().get_processed_count();
    ParetoPortfolioResult {
        hypervolume: portfolio_hypervolume(&pareto_front),
        pareto_front,
        candidate_count,
        topology: station_graph_topology(
            &["pareto-portfolio-source".to_string(), "pareto-portfolio-archive".to_string()],
            &["ParetoCandidateToken<portfolio>".to_string(), "ParetoArchiveRow".to_string()],
        ),
    }
}

fn default_portfolio_assets() -> Vec<PortfolioAsset> {
    vec![
        PortfolioAsset { name: "cash".to_string(), expected_return: 0.02, risk: 0.01 },
        PortfolioAsset { name: "bonds".to_string(), expected_return: 0.045, risk: 0.06 },
        PortfolioAsset { name: "equity".to_string(), expected_return: 0.09, risk: 0.18 },
        PortfolioAsset { name: "growth".to_string(), expected_return: 0.13, risk: 0.30 },
    ]
}

fn random_simplex(n: usize, rng: &mut dyn RandomSource) -> Vec<f64> {
    let draws: Vec<f64> = (0..n).map(|_| -(1e-12_f64.max(rng.next_float())).ln()).collect();
    let total: f64 = draws.iter().sum();
    draws.iter().map(|x| x / total).collect()
}

fn portfolio_point(assets: &[PortfolioAsset], weights: &[f64]) -> ParetoPortfolioPoint {
    let mut expected_return = 0.0;
    let mut variance = 0.0;
    for i in 0..assets.len() {
        expected_return += weights[i] * assets[i].expected_return;
        variance += sq(weights[i] * assets[i].risk);
    }
    ParetoPortfolioPoint { weights: weights.to_vec(), expected_return, risk: variance.sqrt() }
}

fn portfolio_hypervolume(front: &[ParetoPortfolioPoint]) -> f64 {
    if front.is_empty() {
        return 0.0;
    }
    let max_risk = front.iter().map(|p| p.risk).fold(f64::NEG_INFINITY, f64::max) * 1.1;
    let min_return = front.iter().map(|p| p.expected_return).fold(f64::INFINITY, f64::min) * 0.9;
    let mut hv = 0.0;
    let mut prev_risk = 0.0;
    for point in front {
        let width = (point.risk - prev_risk).max(0.0);
        let height = (point.expected_return - min_return).max(0.0);
        hv += width * height;
        prev_risk = point.risk;
    }
    let tail_width = (max_risk - prev_risk).max(0.0);
    let last = &front[front.len() - 1];
    hv += tail_width * (last.expected_return - min_return).max(0.0);
    hv
}

pub fn pareto_front_is_nondominated(front: &[ParetoPortfolioPoint]) -> bool {
    let objectives: Vec<Vec<f64>> = front.iter().map(|p| vec![p.risk, -p.expected_return]).collect();
    for i in 0..objectives.len() {
        for j in 0..objectives.len() {
            if i != j && dominates(&objectives[j], &objectives[i]) {
                return false;
            }
        }
    }
    true
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! Deterministic smoke tests: each runner uses its default fixed seed, so
    //! these check that the wired DES pipeline produces a sane result.

    use super::*;

    #[test]
    fn particle_swarm_minimises_sphere() {
        let result = run_particle_swarm(ParticleSwarmParams::default());
        assert_eq!(result.iterations, 120);
        assert!(result.best_value.is_finite());
        assert!(result.best_value >= 0.0);
        assert!(result.best_value < 5.0, "best_value = {}", result.best_value);
        assert_eq!(result.best_position.len(), 3);
    }

    #[test]
    fn ant_colony_tsp_finds_a_tour() {
        let result = run_ant_colony_tsp(AntColonyTSPParams::default());
        // A closed tour over 6 points visits 6 nodes and returns to the start.
        assert_eq!(result.best_tour.len(), 7);
        assert!(result.best_length.is_finite() && result.best_length > 0.0);
    }

    #[test]
    fn map_coloring_csp_is_satisfied() {
        let result = run_map_coloring_csp(MapColoringCSPParams::default());
        assert!(result.satisfied);
        assert_eq!(result.assignment.len(), 7);
    }

    #[test]
    fn max_sat_satisfies_most_clauses() {
        let result = run_max_sat_local_search(MaxSATParams::default());
        assert_eq!(result.total_clauses, 6);
        assert!(result.satisfied_clauses >= 5, "satisfied = {}", result.satisfied_clauses);
    }

    #[test]
    fn sdp_max_cut_produces_a_cut() {
        let result = run_sdp_max_cut_relaxation(SDPMaxCutParams::default());
        assert_eq!(result.cut.len(), 5);
        assert!(result.rounded_cut_value.is_finite() && result.rounded_cut_value >= 0.0);
        assert!(result.sdp_value.is_finite());
    }

    #[test]
    fn pareto_portfolio_front_is_nondominated() {
        let result = run_pareto_portfolio(ParetoPortfolioParams::default());
        assert!(!result.pareto_front.is_empty());
        assert_eq!(result.candidate_count, 240 + 4);
        assert!(pareto_front_is_nondominated(&result.pareto_front));
    }
}
