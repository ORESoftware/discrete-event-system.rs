//! Port of `src/des/general/des-base/advanced-optimization.ts`.
//!
//! Station / token bases for advanced optimization families: numeric PSO swarm,
//! ant-colony graph search, CSP tree search, Pareto archive, and a
//! rank-constrained SDP / unit-vector relaxation. Concrete algorithms plug into
//! template methods while candidates, tours, assignments, Pareto points, and SDP
//! iterates move as typed tokens.
//!
//! ## Rust shape (faithful translation of the TS abstract classes)
//!
//!   * Movable token classes → plain `'static` structs. The ported `station.rs`
//!     has no `Token` marker trait (tokens are `Rc<dyn Any>`), so these are NOT
//!     `impl Token`; they are emitted as `Rc<dyn Any>` like every other token.
//!   * Each `abstract class … extends DESStation` → a hook trait (`: DESStation`)
//!     plus a state struct embedded by the concrete station (Rust traits hold no
//!     fields). Required hooks are required trait fns; the FINAL `runTimeStep` is
//!     a provided template step (`run_swarm_step` / `run_aco_step` /
//!     `run_relaxation_step`).
//!   * `ConstraintSatisfactionSearchStation<D> extends TreeSearchStation<…>` →
//!     trait [`ConstraintSatisfactionSearchStation`]`: TreeSearchStation<…>`. It
//!     PROVIDES the tree-search hook bodies as `csp_*` methods; a concrete CSP
//!     station's `TreeSearchStation` hooks delegate to them (Rust can't override
//!     a supertrait default from a subtrait). See the test for the wiring.
//!   * `SourceDrivenConstraintSatisfactionSearchStation<D, Start, Result>` →
//!     trait extending the above with a [`SourceDrivenCspState`] and
//!     `accept_start_token` / `make_result_token` hooks.
//!   * `ParetoArchiveStation<T>` is concrete (not abstract) → a struct.
//!   * `rng: () => number` → injected boxed
//!     [`RandomSource`](crate::des::shared::capabilities::RandomSource). NOTE the
//!     PSO `update_particle` hook GAINS a `rng: &mut dyn RandomSource` parameter
//!     (TS read `this.rng` from the protected base; a `&self` hook can't reach a
//!     `&mut` RNG, so it is threaded in explicitly).
//!   * `Record<string, D>` → `HashMap<String, D>`; spread-insert → `clone +
//!     insert`. `Preconditions.*` → [`Check`] for construction guards and
//!     `panic!`/`.expect()` for in-run invariants (matching the TS `throw`).
//!   * `normalize` / `vector_dot` / `gram` overlap `shared::linalg::VecOps` but
//!     are kept here to preserve the exported surface and the special-cased
//!     near-zero `normalize` (returns `e_0`).

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::station::{AnyToken, DESStation, DEFAULT_CHANNEL};
use crate::des::general::des_base::tree_search::{NodeEvaluation, TreeSearchStation};
use crate::des::shared::capabilities::RandomSource;

// -----------------------------------------------------------------------------
// Generic movable tokens
// -----------------------------------------------------------------------------

/// A scored optimization candidate in transit.
#[derive(Clone, Debug)]
pub struct OptimizationCandidateToken<T> {
    pub kind: String,
    pub candidate: T,
    pub score: f64,
    pub iteration: usize,
}

impl<T> OptimizationCandidateToken<T> {
    pub fn new(kind: String, candidate: T, score: f64, iteration: usize) -> Self {
        OptimizationCandidateToken {
            kind,
            candidate,
            score,
            iteration,
        }
    }
}

/// A constructed walk over graph nodes (default node type = `usize` index).
#[derive(Clone, Debug)]
pub struct GraphWalkToken<N = usize> {
    pub nodes: Vec<N>,
    pub cost: f64,
    pub iteration: usize,
}

impl<N> GraphWalkToken<N> {
    pub fn new(nodes: Vec<N>, cost: f64, iteration: usize) -> Self {
        GraphWalkToken {
            nodes,
            cost,
            iteration,
        }
    }
}

/// A (partial) constraint assignment in transit (default value type = `String`).
#[derive(Clone, Debug)]
pub struct ConstraintAssignmentToken<D = String> {
    pub assignment: HashMap<String, D>,
    pub depth: usize,
}

impl<D> ConstraintAssignmentToken<D> {
    pub fn new(assignment: HashMap<String, D>, depth: usize) -> Self {
        ConstraintAssignmentToken { assignment, depth }
    }
}

/// A multi-objective candidate considered for the Pareto archive.
#[derive(Clone, Debug)]
pub struct ParetoCandidateToken<T> {
    pub candidate: T,
    pub objectives: Vec<f64>,
    pub generation: usize,
}

impl<T> ParetoCandidateToken<T> {
    /// `generation` defaults to 0 (TS default parameter).
    pub fn new(candidate: T, objectives: Vec<f64>) -> Self {
        ParetoCandidateToken {
            candidate,
            objectives,
            generation: 0,
        }
    }
    pub fn with_generation(candidate: T, objectives: Vec<f64>, generation: usize) -> Self {
        ParetoCandidateToken {
            candidate,
            objectives,
            generation,
        }
    }
}

/// One trace row of an iterative optimizer (`mean`/`worst` optional).
#[derive(Clone, Copy, Debug)]
pub struct OptimizationTraceRow {
    pub iteration: usize,
    pub best_score: f64,
    pub mean_score: Option<f64>,
    pub worst_score: Option<f64>,
}

// -----------------------------------------------------------------------------
// Numeric swarm optimization
// -----------------------------------------------------------------------------

/// A single PSO particle.
#[derive(Clone, Debug)]
pub struct NumericSwarmParticle {
    pub id: String,
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub best_position: Vec<f64>,
    pub best_score: f64,
    pub score: f64,
}

/// Shared protected state of `NumericSwarmOptimizerStation`.
pub struct NumericSwarmState {
    pub particle_count: usize,
    pub dimension: usize,
    pub iterations: usize,
    pub lower_bound: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub rng: Option<Box<dyn RandomSource>>,
    pub particles: Vec<NumericSwarmParticle>,
    pub iteration: usize,
    pub finished: bool,
    pub best_position: Vec<f64>,
    pub best_score: f64,
    pub trace: Vec<OptimizationTraceRow>,
}

impl NumericSwarmState {
    pub fn new(
        particle_count: usize,
        dimension: usize,
        iterations: usize,
        lower_bound: Vec<f64>,
        upper_bound: Vec<f64>,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        NumericSwarmState {
            particle_count,
            dimension,
            iterations,
            lower_bound,
            upper_bound,
            rng: Some(rng),
            particles: Vec::new(),
            iteration: 0,
            finished: false,
            best_position: Vec::new(),
            best_score: f64::INFINITY,
            trace: Vec::new(),
        }
    }
}

/// Numeric swarm optimizer hook trait (`objective` / `update_particle`). The
/// template step is the provided [`run_swarm_step`](NumericSwarmOptimizerStation::run_swarm_step).
pub trait NumericSwarmOptimizerStation: DESStation {
    fn swarm_state(&self) -> &NumericSwarmState;
    fn swarm_state_mut(&mut self) -> &mut NumericSwarmState;

    // ── HOOKS (required) ───────────────────────────────────────────────────────

    fn objective(&self, position: &[f64]) -> f64;
    /// Produce the next particle. The injected RNG is threaded in (see header).
    fn update_particle(
        &self,
        particle: NumericSwarmParticle,
        global_best: &[f64],
        iteration: usize,
        rng: &mut dyn RandomSource,
    ) -> NumericSwarmParticle;

    // ── GUARDS ─────────────────────────────────────────────────────────────────

    fn assert_preconditions_swarm(&self) -> Check {
        let cls = self.id().to_string();
        let st = self.swarm_state();
        Preconditions::integer_in_range(&cls, "particleCount", st.particle_count as f64, 1.0, 1e9)?;
        Preconditions::integer_in_range(&cls, "dimension", st.dimension as f64, 1.0, 1e6)?;
        Preconditions::integer_in_range(&cls, "iterations", st.iterations as f64, 1.0, 1e9)?;
        Preconditions::length_eq(&cls, "lowerBound", &st.lower_bound, st.dimension)?;
        Preconditions::length_eq(&cls, "upperBound", &st.upper_bound, st.dimension)?;
        Preconditions::all_finite(&cls, "lowerBound", &st.lower_bound)?;
        Preconditions::all_finite(&cls, "upperBound", &st.upper_bound)?;
        for i in 0..st.dimension {
            Preconditions::check(
                &cls,
                &format!("lowerBound[{i}] < upperBound[{i}]"),
                "satisfy lower < upper",
                st.lower_bound[i] < st.upper_bound[i],
                Some(format!("[{}, {}]", st.lower_bound[i], st.upper_bound[i])),
            )?;
        }
        Ok(())
    }

    // ── TEMPLATE ───────────────────────────────────────────────────────────────

    fn bootstrap(&mut self) {
        self.assert_preconditions_swarm()
            .expect("swarm preconditions");
        let count = self.swarm_state().particle_count;
        let mut rng = self
            .swarm_state_mut()
            .rng
            .take()
            .expect("swarm: rng already in use");
        let mut particles: Vec<NumericSwarmParticle> = Vec::with_capacity(count);
        let mut best_score = f64::INFINITY;
        let mut best_position: Vec<f64> = Vec::new();
        for i in 0..count {
            let position = self.random_position(&mut *rng);
            let velocity = self.random_velocity(&mut *rng);
            let score = self.objective(&position);
            Preconditions::finite(self.id(), &format!("initial particle score {i}"), score)
                .expect("initial particle score finite");
            let p = NumericSwarmParticle {
                id: format!("particle-{i}"),
                best_position: position.clone(),
                position,
                velocity,
                best_score: score,
                score,
            };
            if p.best_score < best_score {
                best_score = p.best_score;
                best_position = p.best_position.clone();
            }
            particles.push(p);
        }
        self.swarm_state_mut().rng = Some(rng);
        {
            let st = self.swarm_state_mut();
            st.particles = particles;
            st.best_score = best_score;
            st.best_position = best_position;
        }
        self.record_trace_swarm();
    }

    fn run_swarm_step(&mut self) {
        if self.swarm_state().finished {
            return;
        }
        if self.swarm_state().iteration >= self.swarm_state().iterations {
            self.swarm_state_mut().finished = true;
            return;
        }
        let iteration = self.swarm_state().iteration;
        let dimension = self.swarm_state().dimension;
        let global_best = self.swarm_state().best_position.clone();
        let particles = self.swarm_state().particles.clone();
        let id = self.id().to_string();

        let mut rng = self
            .swarm_state_mut()
            .rng
            .take()
            .expect("swarm: rng already in use");
        let mut next_particles: Vec<NumericSwarmParticle> = Vec::with_capacity(particles.len());
        let mut best_score = self.swarm_state().best_score;
        let mut best_position = self.swarm_state().best_position.clone();
        let mut emits: Vec<(Vec<f64>, f64)> = Vec::new();

        for particle in particles {
            let mut next =
                self.update_particle(particle.clone(), &global_best, iteration, &mut *rng);
            Preconditions::length_eq(
                &id,
                &format!("{}.position", next.id),
                &next.position,
                dimension,
            )
            .expect("position length");
            Preconditions::length_eq(
                &id,
                &format!("{}.velocity", next.id),
                &next.velocity,
                dimension,
            )
            .expect("velocity length");
            Preconditions::all_finite(&id, &format!("{}.position", next.id), &next.position)
                .expect("position finite");
            Preconditions::all_finite(&id, &format!("{}.velocity", next.id), &next.velocity)
                .expect("velocity finite");
            next.position = self.clamp_position(&next.position);
            next.score = self.objective(&next.position);
            Preconditions::finite(&id, &format!("{}.score", next.id), next.score)
                .expect("score finite");
            if next.score < next.best_score {
                next.best_score = next.score;
                next.best_position = next.position.clone();
            }
            if next.best_score < best_score {
                best_score = next.best_score;
                best_position = next.best_position.clone();
            }
            emits.push((next.position.clone(), next.score));
            next_particles.push(next);
        }

        self.swarm_state_mut().rng = Some(rng);
        {
            let st = self.swarm_state_mut();
            st.particles = next_particles;
            st.best_score = best_score;
            st.best_position = best_position;
            st.iteration += 1;
        }
        for (position, score) in emits {
            let token: AnyToken = Rc::new(OptimizationCandidateToken::new(
                "swarm-particle".to_string(),
                position,
                score,
                iteration,
            ));
            self.core_mut().emit(token, DEFAULT_CHANNEL);
        }
        self.record_trace_swarm();
    }

    fn swarm_has_work(&self) -> bool {
        !self.swarm_state().finished
    }

    // ── ACCESSORS ──────────────────────────────────────────────────────────────

    fn get_best_position(&self) -> Vec<f64> {
        self.swarm_state().best_position.clone()
    }
    fn get_best_score(&self) -> f64 {
        self.swarm_state().best_score
    }
    fn get_particles(&self) -> Vec<NumericSwarmParticle> {
        self.swarm_state().particles.clone()
    }
    fn get_iteration(&self) -> usize {
        self.swarm_state().iteration
    }

    // ── INTERNAL HELPERS ─────────────────────────────────────────────────────────

    fn random_position(&self, rng: &mut dyn RandomSource) -> Vec<f64> {
        let st = self.swarm_state();
        st.lower_bound
            .iter()
            .enumerate()
            .map(|(i, &lo)| lo + rng.next_float() * (st.upper_bound[i] - lo))
            .collect()
    }

    fn random_velocity(&self, rng: &mut dyn RandomSource) -> Vec<f64> {
        let st = self.swarm_state();
        st.lower_bound
            .iter()
            .enumerate()
            .map(|(i, &lo)| {
                let span = st.upper_bound[i] - lo;
                (rng.next_float() * 2.0 - 1.0) * 0.2 * span
            })
            .collect()
    }

    fn clamp_position(&self, x: &[f64]) -> Vec<f64> {
        let st = self.swarm_state();
        x.iter()
            .enumerate()
            .map(|(i, &v)| st.lower_bound[i].max(st.upper_bound[i].min(v)))
            .collect()
    }

    fn record_trace_swarm(&mut self) {
        let (mean, worst, best, iteration) = {
            let st = self.swarm_state();
            let n = st.particles.len().max(1) as f64;
            let sum: f64 = st.particles.iter().map(|p| p.score).sum();
            let worst = st
                .particles
                .iter()
                .map(|p| p.score)
                .fold(f64::NEG_INFINITY, f64::max);
            (sum / n, worst, st.best_score, st.iteration)
        };
        self.swarm_state_mut().trace.push(OptimizationTraceRow {
            iteration,
            best_score: best,
            mean_score: Some(mean),
            worst_score: Some(worst),
        });
    }
}

// -----------------------------------------------------------------------------
// Pheromone-guided constructive graph search
// -----------------------------------------------------------------------------

/// Shared protected state of `PheromoneGraphSearchStation`.
pub struct PheromoneGraphState {
    pub node_count: usize,
    pub ants: usize,
    pub iterations: usize,
    pub alpha: f64,
    pub beta: f64,
    pub evaporation: f64,
    pub deposit: f64,
    pub rng: Option<Box<dyn RandomSource>>,
    pub pheromone: Vec<Vec<f64>>,
    pub iteration: usize,
    pub finished: bool,
    pub best_path: Vec<usize>,
    pub best_cost: f64,
    pub trace: Vec<OptimizationTraceRow>,
}

impl PheromoneGraphState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_count: usize,
        ants: usize,
        iterations: usize,
        alpha: f64,
        beta: f64,
        evaporation: f64,
        deposit: f64,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        PheromoneGraphState {
            node_count,
            ants,
            iterations,
            alpha,
            beta,
            evaporation,
            deposit,
            rng: Some(rng),
            pheromone: vec![vec![1.0; node_count]; node_count],
            iteration: 0,
            finished: false,
            best_path: Vec::new(),
            best_cost: f64::INFINITY,
            trace: Vec::new(),
        }
    }
}

/// Ant-colony graph-search hook trait (`path_cost` / `heuristic`). The template
/// step is the provided [`run_aco_step`](PheromoneGraphSearchStation::run_aco_step).
pub trait PheromoneGraphSearchStation: DESStation {
    fn aco_state(&self) -> &PheromoneGraphState;
    fn aco_state_mut(&mut self) -> &mut PheromoneGraphState;

    // ── HOOKS (required) ───────────────────────────────────────────────────────

    fn path_cost(&self, path: &[usize]) -> f64;
    fn heuristic(&self, from: usize, to: usize) -> f64;

    // ── GUARDS ─────────────────────────────────────────────────────────────────

    fn assert_preconditions_aco(&self) -> Check {
        let cls = self.id().to_string();
        let st = self.aco_state();
        Preconditions::integer_in_range(&cls, "nodeCount", st.node_count as f64, 2.0, 1e6)?;
        Preconditions::integer_in_range(&cls, "ants", st.ants as f64, 1.0, 1e9)?;
        Preconditions::integer_in_range(&cls, "iterations", st.iterations as f64, 1.0, 1e9)?;
        Preconditions::non_negative(&cls, "alpha", st.alpha)?;
        Preconditions::non_negative(&cls, "beta", st.beta)?;
        Preconditions::in_range(&cls, "evaporation", st.evaporation, 0.0, 1.0)?;
        Preconditions::positive(&cls, "deposit", st.deposit)?;
        Ok(())
    }

    // ── TEMPLATE ───────────────────────────────────────────────────────────────

    fn run_aco_step(&mut self) {
        if self.aco_state().finished {
            return;
        }
        if self.aco_state().iteration >= self.aco_state().iterations {
            self.aco_state_mut().finished = true;
            return;
        }
        let ants = self.aco_state().ants;
        let node_count = self.aco_state().node_count;
        let iteration = self.aco_state().iteration;
        let id = self.id().to_string();

        let mut rng = self
            .aco_state_mut()
            .rng
            .take()
            .expect("aco: rng already in use");
        let mut walks: Vec<(Vec<usize>, f64)> = Vec::new();
        let mut best_cost = self.aco_state().best_cost;
        let mut best_path = self.aco_state().best_path.clone();
        for ant in 0..ants {
            let path = self.construct_path(ant % node_count, &mut *rng);
            let cost = self.path_cost(&path);
            Preconditions::positive(&id, "pathCost", cost).expect("path cost positive");
            if cost < best_cost {
                best_cost = cost;
                best_path = path.clone();
            }
            walks.push((path, cost));
        }
        self.aco_state_mut().rng = Some(rng);
        self.evaporate();
        for (path, cost) in &walks {
            self.deposit_walk(path, *cost);
        }
        let mean = walks.iter().map(|w| w.1).sum::<f64>() / walks.len() as f64;
        let worst = walks.iter().map(|w| w.1).fold(f64::NEG_INFINITY, f64::max);
        {
            let st = self.aco_state_mut();
            st.trace.push(OptimizationTraceRow {
                iteration,
                best_score: best_cost,
                mean_score: Some(mean),
                worst_score: Some(worst),
            });
            st.best_cost = best_cost;
            st.best_path = best_path;
            st.iteration += 1;
        }
        for (path, cost) in walks {
            let token: AnyToken = Rc::new(GraphWalkToken::new(path, cost, iteration));
            self.core_mut().emit(token, DEFAULT_CHANNEL);
        }
    }

    fn aco_has_work(&self) -> bool {
        !self.aco_state().finished
    }

    // ── ACCESSORS ──────────────────────────────────────────────────────────────

    fn get_best_path(&self) -> Vec<usize> {
        self.aco_state().best_path.clone()
    }
    fn get_best_cost(&self) -> f64 {
        self.aco_state().best_cost
    }
    fn get_pheromone(&self) -> Vec<Vec<f64>> {
        self.aco_state().pheromone.clone()
    }
    fn get_aco_iteration(&self) -> usize {
        self.aco_state().iteration
    }

    // ── INTERNAL HELPERS ─────────────────────────────────────────────────────────

    fn construct_path(&self, start: usize, rng: &mut dyn RandomSource) -> Vec<usize> {
        let node_count = self.aco_state().node_count;
        let mut path = vec![start];
        let mut unvisited: Vec<usize> = (0..node_count).filter(|&i| i != start).collect();
        while !unvisited.is_empty() {
            let current = *path.last().expect("non-empty path");
            let next = self.pick_next(current, &unvisited, rng);
            path.push(next);
            unvisited.retain(|&x| x != next);
        }
        path.push(start);
        path
    }

    fn pick_next(&self, from: usize, options: &[usize], rng: &mut dyn RandomSource) -> usize {
        let st = self.aco_state();
        let weights: Vec<f64> = options
            .iter()
            .map(|&to| {
                let tau = st.pheromone[from][to].powf(st.alpha);
                let eta = self.heuristic(from, to).max(1e-12).powf(st.beta);
                tau * eta
            })
            .collect();
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            let idx = (rng.next_float() * options.len() as f64).floor() as usize;
            return options[idx.min(options.len() - 1)];
        }
        let mut r = rng.next_float() * total;
        for i in 0..options.len() {
            r -= weights[i];
            if r <= 0.0 {
                return options[i];
            }
        }
        options[options.len() - 1]
    }

    fn evaporate(&mut self) {
        let keep = 1.0 - self.aco_state().evaporation;
        let n = self.aco_state().node_count;
        let st = self.aco_state_mut();
        for i in 0..n {
            for j in 0..n {
                st.pheromone[i][j] = (st.pheromone[i][j] * keep).max(1e-9);
            }
        }
    }

    fn deposit_walk(&mut self, path: &[usize], cost: f64) {
        let amount = self.aco_state().deposit / cost.max(1e-12);
        let st = self.aco_state_mut();
        for i in 1..path.len() {
            let a = path[i - 1];
            let b = path[i];
            st.pheromone[a][b] += amount;
            st.pheromone[b][a] += amount;
        }
    }
}

// -----------------------------------------------------------------------------
// Constraint-satisfaction tree search
// -----------------------------------------------------------------------------

/// A node in the CSP search tree.
#[derive(Clone, Debug)]
pub struct ConstraintSearchNode<D = String> {
    pub assignment: HashMap<String, D>,
    pub depth: usize,
}

/// Shared protected state of `ConstraintSatisfactionSearchStation` (in addition
/// to the `TreeSearchCore` it embeds).
pub struct ConstraintSearchCore<D> {
    pub variables: Vec<String>,
    pub domains: HashMap<String, Vec<D>>,
    pub frontier: Vec<ConstraintSearchNode<D>>,
    pub solution: Option<HashMap<String, D>>,
}

impl<D: Clone> ConstraintSearchCore<D> {
    pub fn new(variables: &[String], domains: &HashMap<String, Vec<D>>) -> Self {
        let mut dom: HashMap<String, Vec<D>> = HashMap::new();
        for v in variables {
            dom.insert(v.clone(), domains.get(v).cloned().unwrap_or_default());
        }
        ConstraintSearchCore {
            variables: variables.to_vec(),
            domains: dom,
            frontier: vec![ConstraintSearchNode {
                assignment: HashMap::new(),
                depth: 0,
            }],
            solution: None,
        }
    }
}

/// CSP tree-search base. Extends [`TreeSearchStation`]: a concrete CSP station
/// implements the tree-search hooks by delegating to the provided `csp_*`
/// methods below (Rust cannot override supertrait defaults from a subtrait).
pub trait ConstraintSatisfactionSearchStation<D: Clone + 'static>:
    TreeSearchStation<ConstraintSearchNode<D>>
{
    fn csp_core(&self) -> &ConstraintSearchCore<D>;
    fn csp_core_mut(&mut self) -> &mut ConstraintSearchCore<D>;

    // ── HOOK (required) ──────────────────────────────────────────────────────────

    fn is_consistent(&self, assignment: &HashMap<String, D>) -> bool;

    // ── GUARDS ─────────────────────────────────────────────────────────────────

    fn assert_preconditions_csp(&self) -> Check {
        let cls = self.id().to_string();
        Preconditions::non_empty(&cls, "variables", &self.csp_core().variables)?;
        for v in &self.csp_core().variables {
            let empty: Vec<D> = Vec::new();
            let dom = self.csp_core().domains.get(v).unwrap_or(&empty);
            Preconditions::non_empty(&cls, &format!("domains.{v}"), dom)?;
        }
        Ok(())
    }

    // ── TREE-SEARCH HOOK BODIES (delegated to by the concrete station) ───────────

    fn csp_pick_next(&mut self) -> Option<ConstraintSearchNode<D>> {
        self.csp_core_mut().frontier.pop()
    }

    fn csp_evaluate(&mut self, node: &ConstraintSearchNode<D>) -> NodeEvaluation {
        if !self.is_consistent(&node.assignment) {
            return NodeEvaluation {
                bound: f64::NEG_INFINITY,
                is_leaf: true,
                value: None,
                is_feasible: false,
            };
        }
        let complete = self
            .csp_core()
            .variables
            .iter()
            .all(|v| node.assignment.contains_key(v));
        let nvars = self.csp_core().variables.len() as f64;
        NodeEvaluation {
            bound: nvars,
            is_leaf: complete,
            value: if complete { Some(nvars) } else { None },
            is_feasible: complete,
        }
    }

    fn csp_expand(&mut self, node: &ConstraintSearchNode<D>) -> Vec<ConstraintSearchNode<D>> {
        let Some(variable) = self.choose_variable(&node.assignment) else {
            return Vec::new();
        };
        let domain = self
            .csp_core()
            .domains
            .get(&variable)
            .cloned()
            .unwrap_or_default();
        let mut out: Vec<ConstraintSearchNode<D>> = Vec::new();
        for value in domain {
            let mut assignment = node.assignment.clone();
            assignment.insert(variable.clone(), value);
            if self.is_consistent(&assignment) {
                out.push(ConstraintSearchNode {
                    assignment,
                    depth: node.depth + 1,
                });
            }
        }
        out.reverse();
        out
    }

    fn csp_push_children(&mut self, children: Vec<ConstraintSearchNode<D>>) {
        self.csp_core_mut().frontier.extend(children);
    }

    fn csp_current_best_bound(&self) -> f64 {
        if !self.csp_core().frontier.is_empty() {
            self.csp_core().variables.len() as f64
        } else {
            f64::NEG_INFINITY
        }
    }

    fn csp_should_prune(&self, node: &ConstraintSearchNode<D>, ev: &NodeEvaluation) -> bool {
        !self.is_consistent(&node.assignment) || self.bound_is_dominated(ev.bound)
    }

    fn csp_on_incumbent_update(&mut self, node: &ConstraintSearchNode<D>) {
        let sol = node.assignment.clone();
        self.csp_core_mut().solution = Some(sol.clone());
        let token: AnyToken = Rc::new(ConstraintAssignmentToken::new(sol, node.depth));
        self.core_mut().emit(token, DEFAULT_CHANNEL);
    }

    // ── ACCESSORS ──────────────────────────────────────────────────────────────

    fn get_solution(&self) -> Option<HashMap<String, D>> {
        self.csp_core().solution.clone()
    }
    fn get_variables(&self) -> Vec<String> {
        self.csp_core().variables.clone()
    }
    fn get_domains(&self) -> HashMap<String, Vec<D>> {
        self.csp_core().domains.clone()
    }

    // ── INTERNAL ───────────────────────────────────────────────────────────────

    /// Minimum-remaining-values variable ordering heuristic.
    fn choose_variable(&self, assignment: &HashMap<String, D>) -> Option<String> {
        let mut best: Option<String> = None;
        let mut best_count = usize::MAX;
        for variable in &self.csp_core().variables {
            if assignment.contains_key(variable) {
                continue;
            }
            let mut count = 0usize;
            if let Some(domain) = self.csp_core().domains.get(variable) {
                for value in domain {
                    let mut trial = assignment.clone();
                    trial.insert(variable.clone(), value.clone());
                    if self.is_consistent(&trial) {
                        count += 1;
                    }
                }
            }
            if count < best_count {
                best = Some(variable.clone());
                best_count = count;
            }
        }
        best
    }
}

// -----------------------------------------------------------------------------
// Source-driven CSP wrapper
// -----------------------------------------------------------------------------

/// Shared state added by the source-driven CSP wrapper.
pub struct SourceDrivenCspState {
    pub started: bool,
    pub result_emitted: bool,
    pub start_channel: String,
    pub result_channel: String,
}

impl SourceDrivenCspState {
    pub fn new(start_channel: impl Into<String>, result_channel: impl Into<String>) -> Self {
        SourceDrivenCspState {
            started: false,
            result_emitted: false,
            start_channel: start_channel.into(),
            result_channel: result_channel.into(),
        }
    }
}

/// Source-driven CSP: enters through a `Start` token and exits through a
/// `Result` token, while the search itself uses the shared tree-search template.
pub trait SourceDrivenConstraintSatisfactionSearchStation<D, Start, Result>:
    ConstraintSatisfactionSearchStation<D>
where
    D: Clone + 'static,
    Start: 'static,
    Result: 'static,
{
    fn source_state(&self) -> &SourceDrivenCspState;
    fn source_state_mut(&mut self) -> &mut SourceDrivenCspState;

    fn accept_start_token(&mut self, token: Rc<Start>);
    fn make_result_token(&mut self) -> Rc<Result>;

    fn source_has_work(&self) -> bool {
        if !self.source_state().started {
            self.core().inbox_size(&self.source_state().start_channel) > 0
        } else {
            !self.source_state().result_emitted
        }
    }

    fn source_run_time_step(&mut self) {
        if !self.source_state().started {
            let ch = self.source_state().start_channel.clone();
            let starts = self.core_mut().drain::<Start>(&ch);
            Preconditions::check(
                self.id(),
                "start token count",
                "receive exactly one start token",
                starts.len() == 1,
                Some(starts.len().to_string()),
            )
            .expect("exactly one start token");
            self.accept_start_token(starts[0].clone());
            self.source_state_mut().started = true;
            return;
        }
        if !self.is_finished() {
            self.run_tree_search_step();
        }
        if self.is_finished() && !self.source_state().result_emitted {
            let token: AnyToken = self.make_result_token();
            let ch = self.source_state().result_channel.clone();
            self.core_mut().emit(token, &ch);
            self.source_state_mut().result_emitted = true;
        }
    }
}

// -----------------------------------------------------------------------------
// Pareto archive station
// -----------------------------------------------------------------------------

/// One row of a Pareto archive.
#[derive(Clone, Debug)]
pub struct ParetoArchiveRow<T> {
    pub candidate: T,
    pub objectives: Vec<f64>,
    pub generation: usize,
}

/// Maintains a non-dominated archive of candidates pulled from the inbox and an
/// explicit pending queue.
pub struct ParetoArchiveStation<T> {
    core: crate::des::general::des_base::station::StationCore,
    archive: Vec<ParetoArchiveRow<T>>,
    pending: VecDeque<ParetoCandidateToken<T>>,
    processed: usize,
    finished: bool,
}

impl<T: Clone + 'static> ParetoArchiveStation<T> {
    pub fn new(id: impl Into<String>, candidates: Vec<ParetoCandidateToken<T>>) -> Self {
        ParetoArchiveStation {
            core: crate::des::general::des_base::station::StationCore::new(id),
            archive: Vec::new(),
            pending: candidates.into_iter().collect(),
            processed: 0,
            finished: false,
        }
    }

    pub fn enqueue(&mut self, candidate: ParetoCandidateToken<T>) {
        self.pending.push_back(candidate);
        self.finished = false;
    }

    pub fn get_archive(&self) -> Vec<ParetoArchiveRow<T>> {
        self.archive.clone()
    }

    pub fn get_processed_count(&self) -> usize {
        self.processed
    }

    fn consider(&mut self, token: ParetoCandidateToken<T>) {
        let id = self.core.id.clone();
        Preconditions::non_empty(&id, "objectives", &token.objectives)
            .expect("objectives non-empty");
        Preconditions::all_finite(&id, "objectives", &token.objectives).expect("objectives finite");
        for row in &self.archive {
            if same_objectives(&row.objectives, &token.objectives) {
                return;
            }
            if dominates(&row.objectives, &token.objectives) {
                return;
            }
        }
        let obj = token.objectives.clone();
        self.archive.retain(|row| !dominates(&obj, &row.objectives));
        self.archive.push(ParetoArchiveRow {
            candidate: token.candidate,
            objectives: token.objectives,
            generation: token.generation,
        });
    }
}

impl<T: Clone + 'static> DESStation for ParetoArchiveStation<T> {
    fn core(&self) -> &crate::des::general::des_base::station::StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut crate::des::general::des_base::station::StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn has_work(&self) -> bool {
        !self.finished || !self.pending.is_empty() || self.core.inbox_size(DEFAULT_CHANNEL) > 0
    }

    fn run_time_step(&mut self) {
        let inbox = self.core.drain::<ParetoCandidateToken<T>>(DEFAULT_CHANNEL);
        for t in inbox {
            // Tokens arrive shared; clone out to own the candidate.
            self.pending.push_back((*t).clone());
        }
        if !self.pending.is_empty() {
            self.finished = false;
        }
        if self.finished {
            return;
        }
        let Some(next) = self.pending.pop_front() else {
            self.finished = true;
            return;
        };
        self.processed += 1;
        self.consider(next);
    }
}

/// True iff `a` Pareto-dominates `b` (minimisation: ≤ in all, < in at least one).
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    Preconditions::length_eq("dominates", "objective vector b", b, a.len()).expect("equal length");
    Preconditions::all_finite("dominates", "objective vector a", a).expect("a finite");
    Preconditions::all_finite("dominates", "objective vector b", b).expect("b finite");
    let mut strictly_better = false;
    for i in 0..a.len() {
        if a[i] > b[i] + 1e-12 {
            return false;
        }
        if a[i] < b[i] - 1e-12 {
            strictly_better = true;
        }
    }
    strictly_better
}

fn same_objectives(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > 1e-12 {
            return false;
        }
    }
    true
}

// -----------------------------------------------------------------------------
// Rank-constrained SDP / unit-vector relaxation station
// -----------------------------------------------------------------------------

/// One trace row of the relaxation.
#[derive(Clone, Copy, Debug)]
pub struct UnitVectorRelaxationTraceRow {
    pub iteration: usize,
    pub objective: f64,
}

/// Shared protected state of `UnitVectorRelaxationStation`.
pub struct UnitVectorRelaxationState {
    pub nodes: usize,
    pub rank: usize,
    pub iterations: usize,
    pub step_size: f64,
    pub rng: Option<Box<dyn RandomSource>>,
    pub vectors: Vec<Vec<f64>>,
    pub best_vectors: Vec<Vec<f64>>,
    pub best_objective: f64,
    pub iteration: usize,
    pub finished: bool,
    pub trace: Vec<UnitVectorRelaxationTraceRow>,
}

impl UnitVectorRelaxationState {
    pub fn new(
        nodes: usize,
        rank: usize,
        iterations: usize,
        step_size: f64,
        mut rng: Box<dyn RandomSource>,
    ) -> Self {
        let vectors: Vec<Vec<f64>> = (0..nodes)
            .map(|_| random_unit_vector(rank, &mut *rng))
            .collect();
        let best_vectors = vectors.clone();
        UnitVectorRelaxationState {
            nodes,
            rank,
            iterations,
            step_size,
            rng: Some(rng),
            vectors,
            best_vectors,
            best_objective: f64::NEG_INFINITY,
            iteration: 0,
            finished: false,
            trace: Vec::new(),
        }
    }
}

fn random_unit_vector(rank: usize, rng: &mut dyn RandomSource) -> Vec<f64> {
    let v: Vec<f64> = (0..rank).map(|_| rng.next_float() * 2.0 - 1.0).collect();
    normalize(&v)
}

/// Unit-vector relaxation hook trait (`objective` / `gradient`). The template
/// step is the provided [`run_relaxation_step`](UnitVectorRelaxationStation::run_relaxation_step).
pub trait UnitVectorRelaxationStation: DESStation {
    fn uvr_state(&self) -> &UnitVectorRelaxationState;
    fn uvr_state_mut(&mut self) -> &mut UnitVectorRelaxationState;

    // ── HOOKS (required) ───────────────────────────────────────────────────────

    fn objective(&self, vectors: &[Vec<f64>]) -> f64;
    fn gradient(&self, vectors: &[Vec<f64>]) -> Vec<Vec<f64>>;

    // ── GUARDS ─────────────────────────────────────────────────────────────────

    fn assert_preconditions_uvr(&self) -> Check {
        let cls = self.id().to_string();
        let st = self.uvr_state();
        Preconditions::integer_in_range(&cls, "nodes", st.nodes as f64, 2.0, 1e6)?;
        Preconditions::integer_in_range(&cls, "rank", st.rank as f64, 1.0, 1e6)?;
        Preconditions::integer_in_range(&cls, "iterations", st.iterations as f64, 1.0, 1e9)?;
        Preconditions::positive(&cls, "stepSize", st.step_size)?;
        Ok(())
    }

    // ── TEMPLATE ───────────────────────────────────────────────────────────────

    fn bootstrap(&mut self) {
        self.record_best_uvr();
    }

    fn run_relaxation_step(&mut self) {
        if self.uvr_state().finished {
            return;
        }
        if self.uvr_state().iteration >= self.uvr_state().iterations {
            self.uvr_state_mut().finished = true;
            return;
        }
        let id = self.id().to_string();
        let nodes = self.uvr_state().nodes;
        let rank = self.uvr_state().rank;
        let step = self.uvr_state().step_size;
        let grad = self.gradient(&self.uvr_state().vectors);
        Preconditions::length_eq(&id, "gradient", &grad, nodes).expect("gradient length");
        for i in 0..nodes {
            Preconditions::length_eq(&id, &format!("gradient[{i}]"), &grad[i], rank)
                .expect("gradient row length");
            Preconditions::all_finite(&id, &format!("gradient[{i}]"), &grad[i])
                .expect("gradient finite");
        }
        {
            let st = self.uvr_state_mut();
            for i in 0..nodes {
                for j in 0..rank {
                    st.vectors[i][j] += step * grad[i][j];
                }
                let normalized = normalize(&st.vectors[i]);
                st.vectors[i] = normalized;
            }
            st.iteration += 1;
        }
        self.record_best_uvr();
    }

    fn uvr_has_work(&self) -> bool {
        !self.uvr_state().finished
    }

    // ── ACCESSORS ──────────────────────────────────────────────────────────────

    fn get_vectors(&self) -> Vec<Vec<f64>> {
        self.uvr_state().vectors.clone()
    }
    fn get_best_vectors(&self) -> Vec<Vec<f64>> {
        self.uvr_state().best_vectors.clone()
    }
    fn get_best_objective(&self) -> f64 {
        self.uvr_state().best_objective
    }
    fn get_gram_matrix(&self) -> Vec<Vec<f64>> {
        gram(&self.uvr_state().best_vectors)
    }
    fn get_uvr_iteration(&self) -> usize {
        self.uvr_state().iteration
    }

    // ── INTERNAL ───────────────────────────────────────────────────────────────

    fn record_best_uvr(&mut self) {
        let value = self.objective(&self.uvr_state().vectors);
        Preconditions::finite(self.id(), "objective", value).expect("objective finite");
        let st = self.uvr_state_mut();
        st.trace.push(UnitVectorRelaxationTraceRow {
            iteration: st.iteration,
            objective: value,
        });
        if value > st.best_objective {
            st.best_objective = value;
            st.best_vectors = st.vectors.clone();
        }
    }
}

/// L2-normalize; near-zero norm returns `e_0` (TS special case).
pub fn normalize(v: &[f64]) -> Vec<f64> {
    Preconditions::non_empty("normalize", "v", v).expect("non-empty");
    Preconditions::all_finite("normalize", "v", v).expect("finite");
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        let mut out = vec![0.0; v.len()];
        out[0] = 1.0;
        return out;
    }
    v.iter().map(|x| x / norm).collect()
}

/// Dot product (lengths must match).
pub fn vector_dot(a: &[f64], b: &[f64]) -> f64 {
    Preconditions::length_eq("vectorDot", "b", b, a.len()).expect("equal length");
    Preconditions::all_finite("vectorDot", "a", a).expect("a finite");
    Preconditions::all_finite("vectorDot", "b", b).expect("b finite");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Gram matrix `G[i][j] = <v_i, v_j>`.
pub fn gram(vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
    vectors
        .iter()
        .map(|a| vectors.iter().map(|b| vector_dot(a, b)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::station::StationCore;
    use crate::des::general::des_base::tree_search::{
        NodeEvaluation, SearchObjective, TreeSearchCore, TreeSearchStation,
    };
    use crate::des::shared::capabilities::SeededRandom;

    // ── CSP B&B test ─────────────────────────────────────────────────────────────

    /// All-different over three variables `x, y, z` with domain `{1, 2, 3}`.
    /// The tree search (branch-and-bound base) must find a complete permutation.
    struct AllDifferent {
        core: StationCore,
        search: TreeSearchCore<ConstraintSearchNode<i32>>,
        csp: ConstraintSearchCore<i32>,
    }

    impl AllDifferent {
        fn new() -> Self {
            let variables: Vec<String> = vec!["x".into(), "y".into(), "z".into()];
            let mut domains: HashMap<String, Vec<i32>> = HashMap::new();
            for v in &variables {
                domains.insert(v.clone(), vec![1, 2, 3]);
            }
            AllDifferent {
                core: StationCore::new("all-different"),
                search: TreeSearchCore::new(SearchObjective::Maximise, f64::INFINITY),
                csp: ConstraintSearchCore::new(&variables, &domains),
            }
        }
    }

    impl DESStation for AllDifferent {
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
            self.run_tree_search_step();
        }
        fn has_work(&self) -> bool {
            !self.search.finished
        }
    }

    impl TreeSearchStation<ConstraintSearchNode<i32>> for AllDifferent {
        fn search_core(&self) -> &TreeSearchCore<ConstraintSearchNode<i32>> {
            &self.search
        }
        fn search_core_mut(&mut self) -> &mut TreeSearchCore<ConstraintSearchNode<i32>> {
            &mut self.search
        }
        fn pick_next(&mut self) -> Option<ConstraintSearchNode<i32>> {
            self.csp_pick_next()
        }
        fn evaluate(&mut self, node: &ConstraintSearchNode<i32>) -> NodeEvaluation {
            self.csp_evaluate(node)
        }
        fn expand(
            &mut self,
            node: &ConstraintSearchNode<i32>,
            _ev: &NodeEvaluation,
        ) -> Vec<ConstraintSearchNode<i32>> {
            self.csp_expand(node)
        }
        fn push_children(&mut self, children: Vec<ConstraintSearchNode<i32>>) {
            self.csp_push_children(children);
        }
        fn should_prune(&self, node: &ConstraintSearchNode<i32>, ev: &NodeEvaluation) -> bool {
            self.csp_should_prune(node, ev)
        }
        fn current_best_bound(&self) -> f64 {
            self.csp_current_best_bound()
        }
        fn on_incumbent_update(&mut self, node: &ConstraintSearchNode<i32>, _value: f64) {
            self.csp_on_incumbent_update(node);
        }
    }

    impl ConstraintSatisfactionSearchStation<i32> for AllDifferent {
        fn csp_core(&self) -> &ConstraintSearchCore<i32> {
            &self.csp
        }
        fn csp_core_mut(&mut self) -> &mut ConstraintSearchCore<i32> {
            &mut self.csp
        }
        fn is_consistent(&self, assignment: &HashMap<String, i32>) -> bool {
            let vals: Vec<i32> = assignment.values().copied().collect();
            for i in 0..vals.len() {
                for j in (i + 1)..vals.len() {
                    if vals[i] == vals[j] {
                        return false;
                    }
                }
            }
            true
        }
    }

    #[test]
    fn csp_branch_and_bound_finds_complete_assignment() {
        let mut csp = AllDifferent::new();
        assert!(csp.assert_preconditions_csp().is_ok());
        let mut guard = 0;
        while !csp.is_finished() {
            csp.run_time_step();
            guard += 1;
            assert!(guard < 10_000, "CSP search did not finish");
        }
        let solution = csp.get_solution().expect("a complete assignment");
        assert_eq!(solution.len(), 3);
        let mut vals: Vec<i32> = solution.values().copied().collect();
        vals.sort();
        assert_eq!(vals, vec![1, 2, 3]);
        // The complete assignment scores |variables| = 3 (the optimum bound).
        assert_eq!(csp.get_incumbent_value(), 3.0);
    }

    // ── Numeric swarm test ───────────────────────────────────────────────────────

    /// PSO minimising the 2-D sphere `f(x) = Σ x_i²` (global optimum 0 at the
    /// origin).
    struct SphereSwarm {
        core: StationCore,
        state: NumericSwarmState,
    }

    impl SphereSwarm {
        fn new(seed: u32) -> Self {
            SphereSwarm {
                core: StationCore::new("sphere-swarm"),
                state: NumericSwarmState::new(
                    20,
                    2,
                    80,
                    vec![-5.0, -5.0],
                    vec![5.0, 5.0],
                    Box::new(SeededRandom::new(seed)),
                ),
            }
        }
    }

    impl DESStation for SphereSwarm {
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
            self.run_swarm_step();
        }
        fn has_work(&self) -> bool {
            self.swarm_has_work()
        }
    }

    impl NumericSwarmOptimizerStation for SphereSwarm {
        fn swarm_state(&self) -> &NumericSwarmState {
            &self.state
        }
        fn swarm_state_mut(&mut self) -> &mut NumericSwarmState {
            &mut self.state
        }
        fn objective(&self, position: &[f64]) -> f64 {
            position.iter().map(|x| x * x).sum()
        }
        fn update_particle(
            &self,
            mut particle: NumericSwarmParticle,
            global_best: &[f64],
            _iteration: usize,
            rng: &mut dyn RandomSource,
        ) -> NumericSwarmParticle {
            let (w, c1, c2) = (0.7, 1.4, 1.4);
            for j in 0..particle.position.len() {
                let r1 = rng.next_float();
                let r2 = rng.next_float();
                particle.velocity[j] = w * particle.velocity[j]
                    + c1 * r1 * (particle.best_position[j] - particle.position[j])
                    + c2 * r2 * (global_best[j] - particle.position[j]);
                particle.position[j] += particle.velocity[j];
            }
            particle
        }
    }

    #[test]
    fn swarm_minimises_sphere_to_near_optimum() {
        let mut swarm = SphereSwarm::new(12345);
        swarm.bootstrap();
        assert_eq!(swarm.get_particles().len(), 20);
        let initial = swarm.get_best_score();
        while swarm.swarm_has_work() {
            swarm.run_swarm_step();
        }
        let best = swarm.get_best_score();
        assert!(best <= initial);
        assert!(best < 0.5, "best_score = {best}");
        assert_eq!(swarm.get_iteration(), 80);
    }

    // ── Pareto archive test ───────────────────────────────────────────────────────

    fn run_to_idle(st: &mut ParetoArchiveStation<&'static str>) {
        let mut guard = 0;
        while st.has_work() {
            st.run_time_step();
            guard += 1;
            assert!(guard < 1000, "archive did not settle");
        }
    }

    #[test]
    fn pareto_archive_keeps_only_nondominated() {
        let mut archive = ParetoArchiveStation::<&'static str>::new(
            "archive",
            vec![
                ParetoCandidateToken::new("a", vec![1.0, 4.0]),
                ParetoCandidateToken::new("b", vec![2.0, 2.0]),
                ParetoCandidateToken::new("c", vec![4.0, 1.0]),
                ParetoCandidateToken::new("d", vec![3.0, 3.0]), // dominated by b
            ],
        );
        run_to_idle(&mut archive);
        let rows = archive.get_archive();
        let mut objs: Vec<Vec<f64>> = rows.iter().map(|r| r.objectives.clone()).collect();
        objs.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(objs, vec![vec![1.0, 4.0], vec![2.0, 2.0], vec![4.0, 1.0]]);
        assert_eq!(archive.get_processed_count(), 4);
    }

    #[test]
    fn dominates_and_normalize_helpers() {
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
        assert!(!dominates(&[2.0, 2.0], &[2.0, 2.0])); // equal, not strictly better
        let n = normalize(&[3.0, 4.0]);
        assert!((n[0] - 0.6).abs() < 1e-12 && (n[1] - 0.8).abs() < 1e-12);
        // near-zero → e_0
        assert_eq!(normalize(&[0.0, 0.0]), vec![1.0, 0.0]);
        assert_eq!(vector_dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    }
}
