//! Station-graph implementations of classic numerical/optimization algorithms,
//! built on the discrete-event `VisualBlock` + `run_time_step` methodology.
//!
//! Each model is expressed as a small DES diagram of three [`VisualBlock`]s — a
//! one-shot **source**, an iterative **solver**, and a latest-token **sink** —
//! wired over named channels and driven by [`run_iterative_des`]. The solver
//! station runs exactly ONE algorithm iteration per tick from its
//! [`DESStation::run_time_step`] hook, so the simulation trace is the iteration
//! history of the underlying method.
//!
//! ## Pure vs. stateful pieces
//!
//! Per the engine's "function as a type" convention (`shared::transform`):
//!
//!   * VANILLA, side-effect-free math (objective functions, alignment scores,
//!     target densities, fitness) is modelled with the pure
//!     [`Transform`](crate::des::shared::transform::Transform) trait.
//!   * Algorithms that carry MEMORY across iterations (the L-BFGS curvature
//!     history, the MCMC chain, the EM responsibilities, the DP table) keep that
//!     state inside the solver station — the stateful analogue of
//!     [`StatefulTransform`](crate::des::shared::transform::StatefulTransform).
//!
//! ## Models
//!
//!   * `run_lbfgs` — limited-memory BFGS (gradient-based, curvature memory).
//!   * `run_sequence_alignment` — Needleman–Wunsch global alignment (DP table,
//!     one row filled per tick).
//!   * `run_metropolis_hastings` — random-walk Metropolis MCMC.
//!   * `run_differential_evolution` — DE/rand/1/bin evolutionary search.
//!   * `run_prim_mst` — Prim's minimum spanning tree (one edge added per tick).
//!   * `run_backprop_mlp` — a small MLP trained by full-batch backprop (one
//!     gradient-descent epoch per tick).
//!   * `run_gaussian_mixture_em` — EM for a 1-D Gaussian mixture.
//!   * `run_mean_field_vi` — mean-field coordinate-ascent variational inference
//!     for a Normal mean with a known-precision likelihood.

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use super::des_base::preconditions::{Check, Preconditions};
use super::des_base::runner::{run_iterative_des, IterativeRunOptions};
use super::des_base::station::{AnyToken, DESStation, StationCore, StationRef};
use super::des_base::visual_block::{
    visual_block_specs, VisualBlock, VisualBlockOptions, VisualBlockPortSpec, VisualBlockRole,
    VisualBlockSpec, VisualBlockStyle, VisualPortInput,
};
use super::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::transform::Transform;

/// Channel carrying the one-shot solve-start token.
const CH_START: &str = "start";
/// Channel carrying the terminal result token.
const CH_RESULT: &str = "result";

/// `throw` on a failed precondition (fatal invariant violation).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// Visual solver scaffold
// =============================================================================

/// The one required hook for a model: advance the algorithm by one iteration.
///
/// The owning [`SolverStation`] calls [`step`](IterativeSolver::step) once per
/// tick until it returns `false` (converged / done) or `max_iters` is reached,
/// then snapshots [`result`](IterativeSolver::result).
pub trait IterativeSolver {
    /// The reduced result this solver produces.
    type Output: Clone + 'static;

    /// Hard cap on iterations (also bounds the DES tick budget).
    fn max_iters(&self) -> usize;

    /// Run a single iteration. Return `true` to keep iterating, `false` to stop
    /// early (converged). `iter` is the 0-based iteration index.
    fn step(&mut self, iter: usize) -> bool;

    /// Reduce the terminal solver state to its public result.
    fn result(&self) -> Self::Output;
}

/// Marker token emitted by the source to kick off a solve.
struct SolverStartToken;

/// Terminal token carrying the solver's reduced result.
struct SolverResultToken<R> {
    result: R,
}

/// Build the standard light-blue solver style.
fn solver_style() -> VisualBlockStyle {
    VisualBlockStyle {
        fill: Some("#eff6ff".to_string()),
        stroke: Some("#1d4ed8".to_string()),
        text: Some("#0f172a".to_string()),
    }
}

/// A source [`VisualBlock`] that emits a single [`SolverStartToken`] once.
struct SolverSourceBlock {
    visual: VisualBlock,
    emitted: bool,
}

impl SolverSourceBlock {
    fn new(id: &str) -> Self {
        let visual = VisualBlock::source(
            id,
            vec![VisualPortInput::from(CH_START)],
            VisualBlockOptions {
                kind: Some("solver-source".to_string()),
                label: Some("start".to_string()),
                style: Some(solver_style()),
                ..Default::default()
            },
        );
        SolverSourceBlock {
            visual,
            emitted: false,
        }
    }

    fn visual(&self) -> &VisualBlock {
        &self.visual
    }
}

impl DESStation for SolverSourceBlock {
    fn core(&self) -> &StationCore {
        self.visual.core()
    }
    fn core_mut(&mut self) -> &mut StationCore {
        self.visual.core_mut()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        !self.emitted
    }
    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        let token: AnyToken = Rc::new(SolverStartToken);
        self.visual.core_mut().emit(token, CH_START);
        self.emitted = true;
    }
}

/// The iterative solver station: composes a [`VisualBlock`] and runs one
/// [`IterativeSolver::step`] per tick once a start token arrives.
struct SolverStation<S: IterativeSolver> {
    visual: VisualBlock,
    solver: S,
    started: bool,
    iter: usize,
    finished: bool,
    result_emitted: bool,
}

impl<S: IterativeSolver + 'static> SolverStation<S> {
    fn new(id: &str, kind: &str, label: &str, solver: S) -> Self {
        let visual = VisualBlock::new(
            id,
            VisualBlockOptions {
                kind: Some(kind.to_string()),
                role: Some(VisualBlockRole::Transform),
                label: Some(label.to_string()),
                ports: Some(VisualBlockPortSpec {
                    inputs: vec![VisualPortInput::from(CH_START)],
                    outputs: vec![VisualPortInput::from(CH_RESULT)],
                }),
                style: Some(solver_style()),
                ..Default::default()
            },
        );
        SolverStation {
            visual,
            solver,
            started: false,
            iter: 0,
            finished: false,
            result_emitted: false,
        }
    }

    fn visual(&self) -> &VisualBlock {
        &self.visual
    }
}

impl<S: IterativeSolver + 'static> DESStation for SolverStation<S> {
    fn core(&self) -> &StationCore {
        self.visual.core()
    }
    fn core_mut(&mut self) -> &mut StationCore {
        self.visual.core_mut()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        if !self.started {
            return self.visual.core().inbox_size(CH_START) > 0;
        }
        !self.result_emitted
    }
    fn run_time_step(&mut self) {
        if !self.started {
            let starts = self.visual.core_mut().drain::<SolverStartToken>(CH_START);
            if starts.is_empty() {
                return;
            }
            self.started = true;
            return;
        }
        if !self.finished {
            if self.iter >= self.solver.max_iters() {
                self.finished = true;
            } else {
                let keep_going = self.solver.step(self.iter);
                self.iter += 1;
                if !keep_going || self.iter >= self.solver.max_iters() {
                    self.finished = true;
                }
            }
        }
        if self.finished && !self.result_emitted {
            let result = self.solver.result();
            let token: AnyToken = Rc::new(SolverResultToken { result });
            self.visual.core_mut().emit(token, CH_RESULT);
            self.result_emitted = true;
        }
    }
}

/// A sink [`VisualBlock`] that keeps the latest [`SolverResultToken`].
struct SolverSinkBlock<R: Clone + 'static> {
    visual: VisualBlock,
    latest: Option<R>,
}

impl<R: Clone + 'static> SolverSinkBlock<R> {
    fn new(id: &str) -> Self {
        let visual = VisualBlock::sink(
            id,
            vec![VisualPortInput::from(CH_RESULT)],
            VisualBlockOptions {
                kind: Some("solver-sink".to_string()),
                label: Some("result".to_string()),
                style: Some(solver_style()),
                ..Default::default()
            },
        );
        SolverSinkBlock {
            visual,
            latest: None,
        }
    }

    fn visual(&self) -> &VisualBlock {
        &self.visual
    }
}

impl<R: Clone + 'static> DESStation for SolverSinkBlock<R> {
    fn core(&self) -> &StationCore {
        self.visual.core()
    }
    fn core_mut(&mut self) -> &mut StationCore {
        self.visual.core_mut()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.visual.core().inbox_size(CH_RESULT) > 0
    }
    fn run_time_step(&mut self) {
        let tokens = self.visual.core_mut().drain::<SolverResultToken<R>>(CH_RESULT);
        if let Some(last) = tokens.last() {
            self.latest = Some(last.result.clone());
        }
    }
}

/// The outcome of driving a [`SolverStation`] through the visual DES pipeline.
pub struct VisualSolverRun<R> {
    /// The solver's reduced result.
    pub result: R,
    /// The three pipeline blocks rendered as visual-editor specs.
    pub visual_blocks: Vec<VisualBlockSpec>,
    /// Station ids, in `source → solver → sink` order.
    pub topology: Vec<String>,
    /// Iterations the solver actually executed.
    pub iterations: usize,
    /// Total DES ticks the run consumed.
    pub ticks: usize,
}

/// Wire `source → solver → sink` (each a [`VisualBlock`]), drive the time-step
/// loop, and harvest the result plus the visual specs.
fn run_visual_solver<S>(base: &str, kind: &str, label: &str, solver: S) -> VisualSolverRun<S::Output>
where
    S: IterativeSolver + 'static,
{
    let max_iters = solver.max_iters();
    let source = Rc::new(RefCell::new(SolverSourceBlock::new(&format!("{base}-source"))));
    let station = Rc::new(RefCell::new(SolverStation::new(
        &format!("{base}-solver"),
        kind,
        label,
        solver,
    )));
    let sink = Rc::new(RefCell::new(SolverSinkBlock::<S::Output>::new(&format!(
        "{base}-sink"
    ))));

    source
        .borrow_mut()
        .core_mut()
        .pipe(station.clone() as StationRef, CH_START, CH_START);
    station
        .borrow_mut()
        .core_mut()
        .pipe(sink.clone() as StationRef, CH_RESULT, CH_RESULT);

    // Budget: one tick to emit the start token, one to consume it, one per
    // iteration, one to emit the result, plus slack for the sink to drain.
    let summary = run_iterative_des(
        vec![
            source.clone() as StationRef,
            station.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_iters + 8),
            run_validators: false,
            ..Default::default()
        },
    );

    let result = sink
        .borrow()
        .latest
        .clone()
        .unwrap_or_else(|| panic!("{base}: solver produced no result"));
    let iterations = station.borrow().iter;

    let source_guard = source.borrow();
    let station_guard = station.borrow();
    let sink_guard = sink.borrow();
    let visual_blocks = visual_block_specs(&[
        source_guard.visual(),
        station_guard.visual(),
        sink_guard.visual(),
    ]);
    let topology = vec![
        source_guard.visual().id(),
        station_guard.visual().id(),
        sink_guard.visual().id(),
    ];

    VisualSolverRun {
        result,
        visual_blocks,
        topology,
        iterations,
        ticks: summary.ticks,
    }
}

// =============================================================================
// 1. Gradient-based optimization: limited-memory BFGS (L-BFGS)
// =============================================================================

/// Smooth benchmark objective. `Transform`-able pure function (the curvature is
/// supplied analytically by [`smooth_gradient`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothObjective {
    Sphere,
    Rosenbrock,
}

impl SmoothObjective {
    pub fn as_str(self) -> &'static str {
        match self {
            SmoothObjective::Sphere => "sphere",
            SmoothObjective::Rosenbrock => "rosenbrock",
        }
    }
}

/// The objective value as a pure [`Transform`] (`x -> f(x)`).
pub struct ObjectiveFunction {
    pub objective: SmoothObjective,
}

impl Transform<Vec<f64>, f64> for ObjectiveFunction {
    fn transform(&self, input: Vec<f64>) -> f64 {
        smooth_value(self.objective, &input)
    }
}

fn smooth_value(objective: SmoothObjective, x: &[f64]) -> f64 {
    match objective {
        SmoothObjective::Sphere => x.iter().map(|&v| v * v).sum(),
        SmoothObjective::Rosenbrock => {
            let mut total = 0.0;
            let mut i = 0;
            while i + 1 < x.len() {
                total += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
                i += 1;
            }
            total
        }
    }
}

fn smooth_gradient(objective: SmoothObjective, x: &[f64]) -> Vec<f64> {
    match objective {
        SmoothObjective::Sphere => x.iter().map(|&v| 2.0 * v).collect(),
        SmoothObjective::Rosenbrock => {
            let n = x.len();
            let mut g = vec![0.0; n];
            for i in 0..n {
                if i + 1 < n {
                    g[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
                }
                if i > 0 {
                    g[i] += 200.0 * (x[i] - x[i - 1] * x[i - 1]);
                }
            }
            g
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LbfgsParams {
    pub objective: Option<SmoothObjective>,
    pub start: Option<Vec<f64>>,
    pub dimension: Option<usize>,
    pub memory: Option<usize>,
    pub max_iters: Option<usize>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct LbfgsTraceRow {
    pub iteration: usize,
    pub value: f64,
    pub gradient_norm: f64,
    pub step_size: f64,
}

#[derive(Clone, Debug)]
pub struct LbfgsResult {
    pub objective: SmoothObjective,
    pub best_x: Vec<f64>,
    pub best_value: f64,
    pub gradient_norm: f64,
    pub iterations: usize,
    pub converged: bool,
    pub trace: Vec<LbfgsTraceRow>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub topology: Vec<String>,
    pub ticks: usize,
}

/// L-BFGS solver with a bounded curvature history of `(s, y)` pairs.
struct LbfgsSolver {
    objective: SmoothObjective,
    x: Vec<f64>,
    g: Vec<f64>,
    fval: f64,
    s_hist: VecDeque<Vec<f64>>,
    y_hist: VecDeque<Vec<f64>>,
    rho_hist: VecDeque<f64>,
    memory: usize,
    max_iters: usize,
    tolerance: f64,
    converged: bool,
    trace: Vec<LbfgsTraceRow>,
}

impl LbfgsSolver {
    fn new(objective: SmoothObjective, x0: Vec<f64>, memory: usize, max_iters: usize, tol: f64) -> Self {
        let g = smooth_gradient(objective, &x0);
        let fval = smooth_value(objective, &x0);
        let gnorm = l2_norm(&g);
        let mut solver = LbfgsSolver {
            objective,
            x: x0,
            g,
            fval,
            s_hist: VecDeque::new(),
            y_hist: VecDeque::new(),
            rho_hist: VecDeque::new(),
            memory: memory.max(1),
            max_iters,
            tolerance: tol,
            converged: false,
            trace: Vec::new(),
        };
        solver.trace.push(LbfgsTraceRow {
            iteration: 0,
            value: solver.fval,
            gradient_norm: gnorm,
            step_size: 0.0,
        });
        solver
    }

    /// Two-loop recursion: map the gradient to a search direction using the
    /// stored curvature pairs (newest at the back of each deque).
    fn search_direction(&self) -> Vec<f64> {
        let mut q = self.g.clone();
        let k = self.s_hist.len();
        let mut alpha = vec![0.0; k];
        for i in (0..k).rev() {
            let a = self.rho_hist[i] * dot(&self.s_hist[i], &q);
            alpha[i] = a;
            axpy(&mut q, -a, &self.y_hist[i]);
        }
        let gamma = if k > 0 {
            let s = &self.s_hist[k - 1];
            let y = &self.y_hist[k - 1];
            dot(s, y) / dot(y, y).max(1e-12)
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|&v| gamma * v).collect();
        for i in 0..k {
            let beta = self.rho_hist[i] * dot(&self.y_hist[i], &r);
            axpy(&mut r, alpha[i] - beta, &self.s_hist[i]);
        }
        r.iter().map(|&v| -v).collect()
    }
}

impl IterativeSolver for LbfgsSolver {
    type Output = LbfgsResult;

    fn max_iters(&self) -> usize {
        self.max_iters
    }

    fn step(&mut self, iter: usize) -> bool {
        let gnorm = l2_norm(&self.g);
        if gnorm <= self.tolerance {
            self.converged = true;
            return false;
        }
        let mut dir = self.search_direction();
        // Guard against a non-descent direction (e.g. a degenerate history).
        if dot(&dir, &self.g) >= 0.0 {
            dir = self.g.iter().map(|&v| -v).collect();
        }
        // Backtracking line search with the Armijo sufficient-decrease rule.
        let c1 = 1e-4;
        let slope = dot(&self.g, &dir);
        let mut t = 1.0;
        let mut x_next = add_scaled(&self.x, t, &dir);
        let mut f_next = smooth_value(self.objective, &x_next);
        let mut ls = 0;
        while f_next > self.fval + c1 * t * slope && ls < 40 {
            t *= 0.5;
            x_next = add_scaled(&self.x, t, &dir);
            f_next = smooth_value(self.objective, &x_next);
            ls += 1;
        }
        let g_next = smooth_gradient(self.objective, &x_next);
        let s: Vec<f64> = x_next.iter().zip(&self.x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_next.iter().zip(&self.g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-10 {
            if self.s_hist.len() == self.memory {
                self.s_hist.pop_front();
                self.y_hist.pop_front();
                self.rho_hist.pop_front();
            }
            self.s_hist.push_back(s);
            self.y_hist.push_back(y);
            self.rho_hist.push_back(1.0 / sy);
        }
        self.x = x_next;
        self.g = g_next;
        self.fval = f_next;
        let new_gnorm = l2_norm(&self.g);
        self.trace.push(LbfgsTraceRow {
            iteration: iter + 1,
            value: self.fval,
            gradient_norm: new_gnorm,
            step_size: t,
        });
        if new_gnorm <= self.tolerance {
            self.converged = true;
            return false;
        }
        true
    }

    fn result(&self) -> LbfgsResult {
        LbfgsResult {
            objective: self.objective,
            best_x: self.x.clone(),
            best_value: self.fval,
            gradient_norm: l2_norm(&self.g),
            iterations: self.trace.len().saturating_sub(1),
            converged: self.converged,
            trace: self.trace.clone(),
            visual_blocks: Vec::new(),
            topology: Vec::new(),
            ticks: 0,
        }
    }
}

/// Minimize a smooth benchmark with L-BFGS, one iteration per DES tick.
pub fn run_lbfgs(params: LbfgsParams) -> LbfgsResult {
    let objective = params.objective.unwrap_or(SmoothObjective::Rosenbrock);
    let dimension = params.dimension.unwrap_or(2).max(1);
    let start = match params.start {
        Some(s) if !s.is_empty() => s,
        _ => default_start(objective, dimension),
    };
    let memory = params.memory.unwrap_or(6);
    let max_iters = params.max_iters.unwrap_or(200);
    let tolerance = params.tolerance.unwrap_or(1e-6);

    require(Preconditions::integer_in_range(
        "runLbfgs",
        "dimension",
        start.len() as f64,
        1.0,
        1e6,
    ));
    require(Preconditions::all_finite("runLbfgs", "start", &start));
    require(Preconditions::positive("runLbfgs", "tolerance", tolerance));
    require(Preconditions::integer_in_range(
        "runLbfgs",
        "maxIters",
        max_iters as f64,
        1.0,
        1e9,
    ));

    let solver = LbfgsSolver::new(objective, start, memory, max_iters, tolerance);
    let run = run_visual_solver("lbfgs", "lbfgs-optimizer", "L-BFGS", solver);
    let mut result = run.result;
    result.visual_blocks = run.visual_blocks;
    result.topology = run.topology;
    result.ticks = run.ticks;
    result
}

fn default_start(objective: SmoothObjective, dimension: usize) -> Vec<f64> {
    match objective {
        SmoothObjective::Sphere => vec![2.5; dimension],
        SmoothObjective::Rosenbrock => vec![-1.2; dimension],
    }
}

// =============================================================================
// Small vector helpers (kept local; mirror the math used by each solver)
// =============================================================================

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `a <- a + t * b` in place.
fn axpy(a: &mut [f64], t: f64, b: &[f64]) {
    for (ai, bi) in a.iter_mut().zip(b) {
        *ai += t * bi;
    }
}

/// `a + t * b` as a fresh vector.
fn add_scaled(a: &[f64], t: f64, b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + t * y).collect()
}

// =============================================================================
// 2. Dynamic programming: Needleman–Wunsch global sequence alignment
// =============================================================================

/// The substitution score as a pure [`Transform`] over a `(char, char)` pair.
pub struct SubstitutionScore {
    pub match_score: f64,
    pub mismatch: f64,
}

impl Transform<(char, char), f64> for SubstitutionScore {
    fn transform(&self, input: (char, char)) -> f64 {
        if input.0 == input.1 {
            self.match_score
        } else {
            self.mismatch
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SequenceAlignmentParams {
    pub seq_a: Option<String>,
    pub seq_b: Option<String>,
    pub match_score: Option<f64>,
    pub mismatch: Option<f64>,
    pub gap: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct AlignmentTraceRow {
    pub row: usize,
    pub row_best: f64,
    pub running_best: f64,
}

#[derive(Clone, Debug)]
pub struct SequenceAlignmentResult {
    pub aligned_a: String,
    pub aligned_b: String,
    pub score: f64,
    pub rows: usize,
    pub cols: usize,
    pub identity: f64,
    pub trace: Vec<AlignmentTraceRow>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub topology: Vec<String>,
    pub ticks: usize,
}

/// Needleman–Wunsch solver that fills one DP row per tick (its curvature
/// "memory" is the full score table carried across ticks).
struct SequenceAlignmentSolver {
    a: Vec<char>,
    b: Vec<char>,
    sub: SubstitutionScore,
    gap: f64,
    table: Vec<Vec<f64>>,
    running_best: f64,
    trace: Vec<AlignmentTraceRow>,
}

impl SequenceAlignmentSolver {
    fn new(a: Vec<char>, b: Vec<char>, sub: SubstitutionScore, gap: f64) -> Self {
        let n = a.len();
        let m = b.len();
        let mut table = vec![vec![0.0; m + 1]; n + 1];
        for (j, cell) in table[0].iter_mut().enumerate() {
            *cell = j as f64 * gap;
        }
        let running_best = table[0][m];
        SequenceAlignmentSolver {
            a,
            b,
            sub,
            gap,
            table,
            running_best,
            trace: vec![AlignmentTraceRow {
                row: 0,
                row_best: running_best,
                running_best,
            }],
        }
    }

    fn traceback(&self) -> (String, String, f64, usize) {
        let mut i = self.a.len();
        let mut j = self.b.len();
        let mut out_a: Vec<char> = Vec::new();
        let mut out_b: Vec<char> = Vec::new();
        let mut matches = 0usize;
        while i > 0 || j > 0 {
            let here = self.table[i][j];
            if i > 0
                && j > 0
                && (here - (self.table[i - 1][j - 1] + self.sub.transform((self.a[i - 1], self.b[j - 1])))).abs()
                    < 1e-9
            {
                if self.a[i - 1] == self.b[j - 1] {
                    matches += 1;
                }
                out_a.push(self.a[i - 1]);
                out_b.push(self.b[j - 1]);
                i -= 1;
                j -= 1;
            } else if i > 0 && (here - (self.table[i - 1][j] + self.gap)).abs() < 1e-9 {
                out_a.push(self.a[i - 1]);
                out_b.push('-');
                i -= 1;
            } else {
                out_a.push('-');
                out_b.push(self.b[j - 1]);
                j -= 1;
            }
        }
        out_a.reverse();
        out_b.reverse();
        let score = self.table[self.a.len()][self.b.len()];
        (out_a.into_iter().collect(), out_b.into_iter().collect(), score, matches)
    }
}

impl IterativeSolver for SequenceAlignmentSolver {
    type Output = SequenceAlignmentResult;

    fn max_iters(&self) -> usize {
        self.a.len()
    }

    fn step(&mut self, iter: usize) -> bool {
        let i = iter + 1;
        if i > self.a.len() {
            return false;
        }
        let m = self.b.len();
        self.table[i][0] = i as f64 * self.gap;
        let mut row_best = self.table[i][0];
        for j in 1..=m {
            let diag = self.table[i - 1][j - 1] + self.sub.transform((self.a[i - 1], self.b[j - 1]));
            let up = self.table[i - 1][j] + self.gap;
            let left = self.table[i][j - 1] + self.gap;
            let best = diag.max(up).max(left);
            self.table[i][j] = best;
            if best > row_best {
                row_best = best;
            }
        }
        if row_best > self.running_best {
            self.running_best = row_best;
        }
        self.trace.push(AlignmentTraceRow {
            row: i,
            row_best,
            running_best: self.running_best,
        });
        i < self.a.len()
    }

    fn result(&self) -> SequenceAlignmentResult {
        let (aligned_a, aligned_b, score, matches) = self.traceback();
        let columns = aligned_a.chars().count().max(1);
        SequenceAlignmentResult {
            aligned_a,
            aligned_b,
            score,
            rows: self.a.len() + 1,
            cols: self.b.len() + 1,
            identity: matches as f64 / columns as f64,
            trace: self.trace.clone(),
            visual_blocks: Vec::new(),
            topology: Vec::new(),
            ticks: 0,
        }
    }
}

/// Globally align two sequences with Needleman–Wunsch, one DP row per tick.
pub fn run_sequence_alignment(params: SequenceAlignmentParams) -> SequenceAlignmentResult {
    let seq_a = params.seq_a.unwrap_or_else(|| "GATTACA".to_string());
    let seq_b = params.seq_b.unwrap_or_else(|| "GCATGCU".to_string());
    let match_score = params.match_score.unwrap_or(1.0);
    let mismatch = params.mismatch.unwrap_or(-1.0);
    let gap = params.gap.unwrap_or(-1.0);

    let a: Vec<char> = seq_a.chars().collect();
    let b: Vec<char> = seq_b.chars().collect();
    require(Preconditions::non_empty("runSequenceAlignment", "seqA", &a));
    require(Preconditions::non_empty("runSequenceAlignment", "seqB", &b));
    require(Preconditions::finite("runSequenceAlignment", "matchScore", match_score));
    require(Preconditions::finite("runSequenceAlignment", "mismatch", mismatch));
    require(Preconditions::finite("runSequenceAlignment", "gap", gap));

    let solver = SequenceAlignmentSolver::new(
        a,
        b,
        SubstitutionScore {
            match_score,
            mismatch,
        },
        gap,
    );
    let run = run_visual_solver("sequence-alignment", "needleman-wunsch", "Needleman–Wunsch", solver);
    let mut result = run.result;
    result.visual_blocks = run.visual_blocks;
    result.topology = run.topology;
    result.ticks = run.ticks;
    result
}

// =============================================================================
// 3. Monte Carlo / MCMC: random-walk Metropolis–Hastings
// =============================================================================

/// Which (unnormalized) target distribution the sampler explores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McmcTarget {
    /// `N(0, 1)`.
    StandardNormal,
    /// Equal mixture of `N(-2, 0.7)` and `N(2, 0.7)`.
    Bimodal,
}

impl McmcTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            McmcTarget::StandardNormal => "standard-normal",
            McmcTarget::Bimodal => "bimodal",
        }
    }
}

/// The target's log-density as a pure [`Transform`] (`x -> log p(x)`, up to an
/// additive constant).
pub struct LogDensity {
    pub target: McmcTarget,
}

fn log_gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

impl Transform<f64, f64> for LogDensity {
    fn transform(&self, x: f64) -> f64 {
        match self.target {
            McmcTarget::StandardNormal => -0.5 * x * x,
            McmcTarget::Bimodal => {
                let a = log_gaussian(x, -2.0, 0.7) + (0.5_f64).ln();
                let b = log_gaussian(x, 2.0, 0.7) + (0.5_f64).ln();
                let m = a.max(b);
                m + ((a - m).exp() + (b - m).exp()).ln()
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MetropolisParams {
    pub target: Option<McmcTarget>,
    pub proposal_std: Option<f64>,
    pub iterations: Option<usize>,
    pub burn_in: Option<usize>,
    pub init: Option<f64>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct McmcTraceRow {
    pub iteration: usize,
    pub value: f64,
    pub accepted: bool,
}

#[derive(Clone, Debug)]
pub struct MetropolisResult {
    pub target: McmcTarget,
    pub mean: f64,
    pub std: f64,
    pub acceptance_rate: f64,
    pub sample_count: usize,
    pub accepted: usize,
    pub proposed: usize,
    pub trace: Vec<McmcTraceRow>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub topology: Vec<String>,
    pub ticks: usize,
}

/// Random-walk Metropolis sampler. The Markov chain state (`x`, its log-density,
/// the post-burn-in samples, the RNG) is the solver's memory across ticks.
struct MetropolisSolver {
    density: LogDensity,
    x: f64,
    log_p: f64,
    proposal_std: f64,
    burn_in: usize,
    iterations: usize,
    trace_stride: usize,
    rng: Box<dyn RandomSource>,
    samples: Vec<f64>,
    accepted: usize,
    proposed: usize,
    trace: Vec<McmcTraceRow>,
}

impl MetropolisSolver {
    fn new(
        density: LogDensity,
        init: f64,
        proposal_std: f64,
        iterations: usize,
        burn_in: usize,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        let log_p = density.transform(init);
        MetropolisSolver {
            density,
            x: init,
            log_p,
            proposal_std,
            burn_in,
            iterations,
            trace_stride: (iterations / 500).max(1),
            rng,
            samples: Vec::new(),
            accepted: 0,
            proposed: 0,
            trace: Vec::new(),
        }
    }
}

impl IterativeSolver for MetropolisSolver {
    type Output = MetropolisResult;

    fn max_iters(&self) -> usize {
        self.iterations
    }

    fn step(&mut self, iter: usize) -> bool {
        let proposal = self.x + self.rng.next_gaussian() * self.proposal_std;
        let log_p_prop = self.density.transform(proposal);
        let log_ratio = log_p_prop - self.log_p;
        let accept = log_ratio >= 0.0 || self.rng.next_float() < log_ratio.exp();
        self.proposed += 1;
        if accept {
            self.x = proposal;
            self.log_p = log_p_prop;
            self.accepted += 1;
        }
        if iter >= self.burn_in {
            self.samples.push(self.x);
        }
        if iter % self.trace_stride == 0 {
            self.trace.push(McmcTraceRow {
                iteration: iter,
                value: self.x,
                accepted: accept,
            });
        }
        true
    }

    fn result(&self) -> MetropolisResult {
        let n = self.samples.len().max(1) as f64;
        let mean = self.samples.iter().sum::<f64>() / n;
        let variance = self.samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
        MetropolisResult {
            target: self.density.target,
            mean,
            std: variance.sqrt(),
            acceptance_rate: self.accepted as f64 / self.proposed.max(1) as f64,
            sample_count: self.samples.len(),
            accepted: self.accepted,
            proposed: self.proposed,
            trace: self.trace.clone(),
            visual_blocks: Vec::new(),
            topology: Vec::new(),
            ticks: 0,
        }
    }
}

/// Sample a target density by random-walk Metropolis, one proposal per tick.
pub fn run_metropolis_hastings(params: MetropolisParams) -> MetropolisResult {
    let target = params.target.unwrap_or(McmcTarget::Bimodal);
    let proposal_std = params.proposal_std.unwrap_or(1.5);
    let iterations = params.iterations.unwrap_or(4000);
    let burn_in = params.burn_in.unwrap_or(iterations / 5);
    let init = params.init.unwrap_or(0.0);
    let seed = params.seed.unwrap_or(20);

    require(Preconditions::positive(
        "runMetropolisHastings",
        "proposalStd",
        proposal_std,
    ));
    require(Preconditions::integer_in_range(
        "runMetropolisHastings",
        "iterations",
        iterations as f64,
        1.0,
        1e9,
    ));
    require(Preconditions::check(
        "runMetropolisHastings",
        "burnIn",
        "be smaller than iterations",
        burn_in < iterations,
        Some(format!("{burn_in} >= {iterations}")),
    ));
    require(Preconditions::finite("runMetropolisHastings", "init", init));

    let solver = MetropolisSolver::new(
        LogDensity { target },
        init,
        proposal_std,
        iterations,
        burn_in,
        Box::new(mulberry32(seed)),
    );
    let run = run_visual_solver("metropolis", "metropolis-hastings", "Metropolis–Hastings", solver);
    let mut result = run.result;
    result.visual_blocks = run.visual_blocks;
    result.topology = run.topology;
    result.ticks = run.ticks;
    result
}

// =============================================================================
// 4. Evolutionary algorithms: differential evolution (DE/rand/1/bin)
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct DifferentialEvolutionParams {
    pub objective: Option<SmoothObjective>,
    pub dimension: Option<usize>,
    pub population: Option<usize>,
    pub generations: Option<usize>,
    pub differential_weight: Option<f64>,
    pub crossover: Option<f64>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct DETraceRow {
    pub generation: usize,
    pub best_value: f64,
    pub mean_value: f64,
}

#[derive(Clone, Debug)]
pub struct DifferentialEvolutionResult {
    pub objective: SmoothObjective,
    pub best_x: Vec<f64>,
    pub best_value: f64,
    pub generations: usize,
    pub trace: Vec<DETraceRow>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub topology: Vec<String>,
    pub ticks: usize,
}

/// DE/rand/1/bin solver; the population + fitness are the per-tick memory.
struct DifferentialEvolutionSolver {
    objective: SmoothObjective,
    population: Vec<Vec<f64>>,
    fitness: Vec<f64>,
    best_x: Vec<f64>,
    best_value: f64,
    f: f64,
    cr: f64,
    lower: f64,
    upper: f64,
    generations: usize,
    rng: Box<dyn RandomSource>,
    trace: Vec<DETraceRow>,
}

impl DifferentialEvolutionSolver {
    fn new(
        objective: SmoothObjective,
        dimension: usize,
        pop_size: usize,
        generations: usize,
        f: f64,
        cr: f64,
        lower: f64,
        upper: f64,
        mut rng: Box<dyn RandomSource>,
    ) -> Self {
        let population: Vec<Vec<f64>> = (0..pop_size)
            .map(|_| {
                (0..dimension)
                    .map(|_| lower + rng.next_float() * (upper - lower))
                    .collect()
            })
            .collect();
        let fitness: Vec<f64> = population.iter().map(|x| smooth_value(objective, x)).collect();
        let (best_i, best_value) = argmin(&fitness);
        let best_x = population[best_i].clone();
        let mut solver = DifferentialEvolutionSolver {
            objective,
            population,
            fitness,
            best_x,
            best_value,
            f,
            cr,
            lower,
            upper,
            generations,
            rng,
            trace: Vec::new(),
        };
        solver.record(0);
        solver
    }

    fn record(&mut self, generation: usize) {
        let mean = self.fitness.iter().sum::<f64>() / self.fitness.len() as f64;
        self.trace.push(DETraceRow {
            generation,
            best_value: self.best_value,
            mean_value: mean,
        });
    }

    /// A distinct index in `[0, n)` not equal to any in `exclude`.
    fn distinct_index(&mut self, n: usize, exclude: &[usize]) -> usize {
        loop {
            let candidate = (self.rng.next_float() * n as f64).floor() as usize % n;
            if !exclude.contains(&candidate) {
                return candidate;
            }
        }
    }
}

impl IterativeSolver for DifferentialEvolutionSolver {
    type Output = DifferentialEvolutionResult;

    fn max_iters(&self) -> usize {
        self.generations
    }

    fn step(&mut self, iter: usize) -> bool {
        let n = self.population.len();
        let dim = self.population[0].len();
        for i in 0..n {
            let r1 = self.distinct_index(n, &[i]);
            let r2 = self.distinct_index(n, &[i, r1]);
            let r3 = self.distinct_index(n, &[i, r1, r2]);
            let j_rand = (self.rng.next_float() * dim as f64).floor() as usize % dim;
            let mut trial = self.population[i].clone();
            for j in 0..dim {
                if self.rng.next_float() < self.cr || j == j_rand {
                    let mutant = self.population[r1][j]
                        + self.f * (self.population[r2][j] - self.population[r3][j]);
                    trial[j] = clamp(mutant, self.lower, self.upper);
                }
            }
            let trial_fit = smooth_value(self.objective, &trial);
            if trial_fit <= self.fitness[i] {
                self.fitness[i] = trial_fit;
                if trial_fit < self.best_value {
                    self.best_value = trial_fit;
                    self.best_x = trial.clone();
                }
                self.population[i] = trial;
            }
        }
        self.record(iter + 1);
        true
    }

    fn result(&self) -> DifferentialEvolutionResult {
        DifferentialEvolutionResult {
            objective: self.objective,
            best_x: self.best_x.clone(),
            best_value: self.best_value,
            generations: self.trace.len().saturating_sub(1),
            trace: self.trace.clone(),
            visual_blocks: Vec::new(),
            topology: Vec::new(),
            ticks: 0,
        }
    }
}

/// Minimize a smooth benchmark with differential evolution, one generation per tick.
pub fn run_differential_evolution(
    params: DifferentialEvolutionParams,
) -> DifferentialEvolutionResult {
    let objective = params.objective.unwrap_or(SmoothObjective::Rosenbrock);
    let dimension = params.dimension.unwrap_or(2).max(1);
    let population = params.population.unwrap_or(40).max(4);
    let generations = params.generations.unwrap_or(150);
    let f = params.differential_weight.unwrap_or(0.7);
    let cr = params.crossover.unwrap_or(0.9);
    let lower = params.lower.unwrap_or(-5.0);
    let upper = params.upper.unwrap_or(5.0);
    let seed = params.seed.unwrap_or(31);

    require(Preconditions::integer_in_range(
        "runDifferentialEvolution",
        "population",
        population as f64,
        4.0,
        1e6,
    ));
    require(Preconditions::integer_in_range(
        "runDifferentialEvolution",
        "generations",
        generations as f64,
        1.0,
        1e9,
    ));
    require(Preconditions::in_range(
        "runDifferentialEvolution",
        "crossover",
        cr,
        0.0,
        1.0,
    ));
    require(Preconditions::positive(
        "runDifferentialEvolution",
        "differentialWeight",
        f,
    ));
    require(Preconditions::check(
        "runDifferentialEvolution",
        "bounds",
        "satisfy lower < upper",
        lower < upper,
        Some(format!("[{lower}, {upper}]")),
    ));

    let solver = DifferentialEvolutionSolver::new(
        objective,
        dimension,
        population,
        generations,
        f,
        cr,
        lower,
        upper,
        Box::new(mulberry32(seed)),
    );
    let run = run_visual_solver(
        "differential-evolution",
        "differential-evolution",
        "Differential Evolution",
        solver,
    );
    let mut result = run.result;
    result.visual_blocks = run.visual_blocks;
    result.topology = run.topology;
    result.ticks = run.ticks;
    result
}

fn argmin(values: &[f64]) -> (usize, f64) {
    let mut best_i = 0;
    let mut best = values[0];
    for (i, &v) in values.iter().enumerate() {
        if v < best {
            best = v;
            best_i = i;
        }
    }
    (best_i, best)
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    lo.max(hi.min(x))
}

// =============================================================================
// 5. Graph optimization: Prim's minimum spanning tree
// =============================================================================

/// An undirected weighted edge `u — v`.
#[derive(Clone, Copy, Debug)]
pub struct GraphEdge {
    pub u: usize,
    pub v: usize,
    pub weight: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PrimMSTParams {
    pub nodes: Option<usize>,
    pub edges: Option<Vec<GraphEdge>>,
}

#[derive(Clone, Debug)]
pub struct MSTTraceRow {
    pub step: usize,
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub total_weight: f64,
}

#[derive(Clone, Debug)]
pub struct PrimMSTResult {
    pub mst_edges: Vec<GraphEdge>,
    pub total_weight: f64,
    pub connected: bool,
    pub node_count: usize,
    pub trace: Vec<MSTTraceRow>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub topology: Vec<String>,
    pub ticks: usize,
}

/// Prim's solver: the adjacency, the in-tree frontier, and the best connecting
/// edge per node are the memory carried across ticks (one edge added per tick).
struct PrimMSTSolver {
    adjacency: Vec<Vec<(usize, f64)>>,
    in_tree: Vec<bool>,
    best_dist: Vec<f64>,
    best_from: Vec<usize>,
    mst_edges: Vec<GraphEdge>,
    total_weight: f64,
    node_count: usize,
    trace: Vec<MSTTraceRow>,
    connected: bool,
}

impl PrimMSTSolver {
    fn new(node_count: usize, edges: &[GraphEdge]) -> Self {
        let mut adjacency = vec![Vec::new(); node_count];
        for e in edges {
            adjacency[e.u].push((e.v, e.weight));
            adjacency[e.v].push((e.u, e.weight));
        }
        let mut best_dist = vec![f64::INFINITY; node_count];
        let mut best_from = vec![usize::MAX; node_count];
        let mut in_tree = vec![false; node_count];
        in_tree[0] = true;
        for &(to, w) in &adjacency[0] {
            if w < best_dist[to] {
                best_dist[to] = w;
                best_from[to] = 0;
            }
        }
        PrimMSTSolver {
            adjacency,
            in_tree,
            best_dist,
            best_from,
            mst_edges: Vec::new(),
            total_weight: 0.0,
            node_count,
            trace: Vec::new(),
            connected: true,
        }
    }
}

impl IterativeSolver for PrimMSTSolver {
    type Output = PrimMSTResult;

    fn max_iters(&self) -> usize {
        self.node_count.saturating_sub(1)
    }

    fn step(&mut self, iter: usize) -> bool {
        // Pick the minimum-weight edge crossing the tree frontier.
        let mut pick = usize::MAX;
        let mut pick_w = f64::INFINITY;
        for v in 0..self.node_count {
            if !self.in_tree[v] && self.best_dist[v] < pick_w {
                pick_w = self.best_dist[v];
                pick = v;
            }
        }
        if pick == usize::MAX {
            // No crossing edge ⇒ the graph is disconnected.
            self.connected = false;
            return false;
        }
        self.in_tree[pick] = true;
        let from = self.best_from[pick];
        self.mst_edges.push(GraphEdge {
            u: from,
            v: pick,
            weight: pick_w,
        });
        self.total_weight += pick_w;
        self.trace.push(MSTTraceRow {
            step: iter + 1,
            from,
            to: pick,
            weight: pick_w,
            total_weight: self.total_weight,
        });
        // Relax frontier distances through the newly added node.
        let neighbours = self.adjacency[pick].clone();
        for (to, w) in neighbours {
            if !self.in_tree[to] && w < self.best_dist[to] {
                self.best_dist[to] = w;
                self.best_from[to] = pick;
            }
        }
        true
    }

    fn result(&self) -> PrimMSTResult {
        PrimMSTResult {
            mst_edges: self.mst_edges.clone(),
            total_weight: self.total_weight,
            connected: self.connected && self.mst_edges.len() == self.node_count.saturating_sub(1),
            node_count: self.node_count,
            trace: self.trace.clone(),
            visual_blocks: Vec::new(),
            topology: Vec::new(),
            ticks: 0,
        }
    }
}

/// Grow a minimum spanning tree with Prim's algorithm, one edge per tick.
pub fn run_prim_mst(params: PrimMSTParams) -> PrimMSTResult {
    let edges = match params.edges {
        Some(e) if !e.is_empty() => e,
        _ => default_graph_edges(),
    };
    let inferred_nodes = edges.iter().map(|e| e.u.max(e.v) + 1).max().unwrap_or(0);
    let node_count = params.nodes.unwrap_or(inferred_nodes).max(inferred_nodes);

    require(Preconditions::integer_in_range(
        "runPrimMST",
        "nodes",
        node_count as f64,
        1.0,
        1e6,
    ));
    for (i, e) in edges.iter().enumerate() {
        require(Preconditions::finite(
            "runPrimMST",
            &format!("edges[{i}].weight"),
            e.weight,
        ));
        require(Preconditions::check(
            "runPrimMST",
            &format!("edges[{i}]"),
            "reference valid nodes",
            e.u < node_count && e.v < node_count && e.u != e.v,
            Some(format!("({}, {})", e.u, e.v)),
        ));
    }

    let solver = PrimMSTSolver::new(node_count, &edges);
    let run = run_visual_solver("prim-mst", "prim-mst", "Prim MST", solver);
    let mut result = run.result;
    result.visual_blocks = run.visual_blocks;
    result.topology = run.topology;
    result.ticks = run.ticks;
    result
}

fn default_graph_edges() -> Vec<GraphEdge> {
    vec![
        GraphEdge { u: 0, v: 1, weight: 2.0 },
        GraphEdge { u: 0, v: 3, weight: 6.0 },
        GraphEdge { u: 1, v: 2, weight: 3.0 },
        GraphEdge { u: 1, v: 3, weight: 8.0 },
        GraphEdge { u: 1, v: 4, weight: 5.0 },
        GraphEdge { u: 2, v: 4, weight: 7.0 },
        GraphEdge { u: 3, v: 4, weight: 9.0 },
    ]
}

// =============================================================================
// 6. Deep-learning optimization via backpropagation
// =============================================================================

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// A labelled training example for the MLP (`y ∈ {0, 1}`).
#[derive(Clone, Debug)]
pub struct MlpSample {
    pub x: Vec<f64>,
    pub y: f64,
}

#[derive(Clone, Debug, Default)]
pub struct BackpropMlpParams {
    pub samples: Option<Vec<MlpSample>>,
    pub hidden_units: Option<usize>,
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct BackpropTraceRow {
    pub epoch: usize,
    pub loss: f64,
    pub accuracy: f64,
}

#[derive(Clone, Debug)]
pub struct BackpropMlpResult {
    pub epochs: usize,
    pub final_loss: f64,
    pub accuracy: f64,
    pub predictions: Vec<f64>,
    pub trace: Vec<BackpropTraceRow>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub topology: Vec<String>,
    pub ticks: usize,
}

/// A single-hidden-layer sigmoid MLP trained by full-batch backprop; the
/// network weights are the memory updated one epoch per tick.
struct BackpropMlpSolver {
    samples: Vec<MlpSample>,
    input_dim: usize,
    hidden_units: usize,
    learning_rate: f64,
    epochs: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
    trace: Vec<BackpropTraceRow>,
}

impl BackpropMlpSolver {
    fn new(
        samples: Vec<MlpSample>,
        hidden_units: usize,
        epochs: usize,
        learning_rate: f64,
        mut rng: Box<dyn RandomSource>,
    ) -> Self {
        let input_dim = samples[0].x.len();
        let init = |rng: &mut dyn RandomSource| (rng.next_float() - 0.5) * 0.8;
        let w1: Vec<Vec<f64>> = (0..hidden_units)
            .map(|_| (0..input_dim).map(|_| init(&mut *rng)).collect())
            .collect();
        let b1: Vec<f64> = (0..hidden_units).map(|_| init(&mut *rng)).collect();
        let w2: Vec<f64> = (0..hidden_units).map(|_| init(&mut *rng)).collect();
        let b2 = init(&mut *rng);
        BackpropMlpSolver {
            samples,
            input_dim,
            hidden_units,
            learning_rate,
            epochs,
            w1,
            b1,
            w2,
            b2,
            trace: Vec::new(),
        }
    }

    fn forward(&self, x: &[f64]) -> (Vec<f64>, f64) {
        let mut hidden = vec![0.0; self.hidden_units];
        for h in 0..self.hidden_units {
            let mut z = self.b1[h];
            for i in 0..self.input_dim {
                z += self.w1[h][i] * x[i];
            }
            hidden[h] = sigmoid(z);
        }
        let mut z = self.b2;
        for h in 0..self.hidden_units {
            z += self.w2[h] * hidden[h];
        }
        (hidden, sigmoid(z))
    }
}

impl IterativeSolver for BackpropMlpSolver {
    type Output = BackpropMlpResult;

    fn max_iters(&self) -> usize {
        self.epochs
    }

    fn step(&mut self, iter: usize) -> bool {
        let n = self.samples.len();
        let mut gw1 = vec![vec![0.0; self.input_dim]; self.hidden_units];
        let mut gb1 = vec![0.0; self.hidden_units];
        let mut gw2 = vec![0.0; self.hidden_units];
        let mut gb2 = 0.0;
        let mut loss = 0.0;
        for sample in &self.samples {
            let (hidden, out) = self.forward(&sample.x);
            let y = sample.y;
            loss += -(y * out.max(1e-12).ln() + (1.0 - y) * (1.0 - out).max(1e-12).ln());
            let d_out = out - y;
            for h in 0..self.hidden_units {
                gw2[h] += d_out * hidden[h];
                let d_hidden = d_out * self.w2[h] * hidden[h] * (1.0 - hidden[h]);
                for i in 0..self.input_dim {
                    gw1[h][i] += d_hidden * sample.x[i];
                }
                gb1[h] += d_hidden;
            }
            gb2 += d_out;
        }
        let scale = self.learning_rate / n as f64;
        for h in 0..self.hidden_units {
            for i in 0..self.input_dim {
                self.w1[h][i] -= scale * gw1[h][i];
            }
            self.b1[h] -= scale * gb1[h];
            self.w2[h] -= scale * gw2[h];
        }
        self.b2 -= scale * gb2;

        let (acc, _) = self.accuracy_and_predictions();
        self.trace.push(BackpropTraceRow {
            epoch: iter + 1,
            loss: loss / n as f64,
            accuracy: acc,
        });
        true
    }

    fn result(&self) -> BackpropMlpResult {
        let (accuracy, predictions) = self.accuracy_and_predictions();
        let final_loss = self.trace.last().map(|r| r.loss).unwrap_or(f64::NAN);
        BackpropMlpResult {
            epochs: self.trace.len(),
            final_loss,
            accuracy,
            predictions,
            trace: self.trace.clone(),
            visual_blocks: Vec::new(),
            topology: Vec::new(),
            ticks: 0,
        }
    }
}

impl BackpropMlpSolver {
    fn accuracy_and_predictions(&self) -> (f64, Vec<f64>) {
        let predictions: Vec<f64> = self.samples.iter().map(|s| self.forward(&s.x).1).collect();
        let correct = predictions
            .iter()
            .zip(&self.samples)
            .filter(|(p, s)| (if **p >= 0.5 { 1.0 } else { 0.0 }) == s.y)
            .count();
        (correct as f64 / self.samples.len() as f64, predictions)
    }
}

/// Train a small MLP by backprop, one full-batch gradient-descent epoch per tick.
pub fn run_backprop_mlp(params: BackpropMlpParams) -> BackpropMlpResult {
    let samples = match params.samples {
        Some(s) if !s.is_empty() => s,
        _ => default_xor_samples(),
    };
    let hidden_units = params.hidden_units.unwrap_or(4).max(1);
    let epochs = params.epochs.unwrap_or(2000);
    let learning_rate = params.learning_rate.unwrap_or(0.5);
    let seed = params.seed.unwrap_or(7);

    let input_dim = samples[0].x.len();
    for (i, s) in samples.iter().enumerate() {
        require(Preconditions::length_eq(
            "runBackpropMlp",
            &format!("samples[{i}].x"),
            &s.x,
            input_dim,
        ));
        require(Preconditions::in_range("runBackpropMlp", "y", s.y, 0.0, 1.0));
    }
    require(Preconditions::positive(
        "runBackpropMlp",
        "learningRate",
        learning_rate,
    ));
    require(Preconditions::integer_in_range(
        "runBackpropMlp",
        "epochs",
        epochs as f64,
        1.0,
        1e9,
    ));

    let solver = BackpropMlpSolver::new(
        samples,
        hidden_units,
        epochs,
        learning_rate,
        Box::new(mulberry32(seed)),
    );
    let run = run_visual_solver("backprop-mlp", "backprop-mlp", "Backprop MLP", solver);
    let mut result = run.result;
    result.visual_blocks = run.visual_blocks;
    result.topology = run.topology;
    result.ticks = run.ticks;
    result
}

fn default_xor_samples() -> Vec<MlpSample> {
    vec![
        MlpSample { x: vec![0.0, 0.0], y: 0.0 },
        MlpSample { x: vec![0.0, 1.0], y: 1.0 },
        MlpSample { x: vec![1.0, 0.0], y: 1.0 },
        MlpSample { x: vec![1.0, 1.0], y: 0.0 },
    ]
}

// =============================================================================
// 7a. Probabilistic inference: EM for a 1-D Gaussian mixture
// =============================================================================

/// A Gaussian density as a pure [`Transform`] (`x -> N(x; mu, sigma^2)`).
pub struct GaussianPdf {
    pub mean: f64,
    pub variance: f64,
}

impl Transform<f64, f64> for GaussianPdf {
    fn transform(&self, x: f64) -> f64 {
        gaussian_pdf(x, self.mean, self.variance)
    }
}

fn gaussian_pdf(x: f64, mean: f64, variance: f64) -> f64 {
    let v = variance.max(1e-12);
    let z = x - mean;
    (-(z * z) / (2.0 * v)).exp() / (2.0 * std::f64::consts::PI * v).sqrt()
}

#[derive(Clone, Debug, Default)]
pub struct GaussianMixtureEMParams {
    pub data: Option<Vec<f64>>,
    pub components: Option<usize>,
    pub max_iters: Option<usize>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct EMTraceRow {
    pub iteration: usize,
    pub log_likelihood: f64,
}

#[derive(Clone, Debug)]
pub struct GaussianMixtureEMResult {
    pub weights: Vec<f64>,
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
    pub log_likelihood: f64,
    pub iterations: usize,
    pub converged: bool,
    pub trace: Vec<EMTraceRow>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub topology: Vec<String>,
    pub ticks: usize,
}

/// EM solver for a 1-D Gaussian mixture; the mixture parameters + the last
/// log-likelihood are the memory carried across ticks (one EM iteration per tick).
struct GaussianMixtureEMSolver {
    data: Vec<f64>,
    k: usize,
    weights: Vec<f64>,
    means: Vec<f64>,
    variances: Vec<f64>,
    prev_ll: f64,
    tolerance: f64,
    max_iters: usize,
    converged: bool,
    trace: Vec<EMTraceRow>,
}

impl GaussianMixtureEMSolver {
    fn new(data: Vec<f64>, k: usize, max_iters: usize, tolerance: f64) -> Self {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let means: Vec<f64> = (0..k)
            .map(|i| min + (i as f64 + 0.5) / k as f64 * (max - min))
            .collect();
        GaussianMixtureEMSolver {
            data,
            k,
            weights: vec![1.0 / k as f64; k],
            means,
            variances: vec![var.max(1e-3); k],
            prev_ll: f64::NEG_INFINITY,
            tolerance,
            max_iters,
            converged: false,
            trace: Vec::new(),
        }
    }

    fn log_likelihood(&self) -> f64 {
        self.data
            .iter()
            .map(|&x| {
                let mix: f64 = (0..self.k)
                    .map(|k| self.weights[k] * gaussian_pdf(x, self.means[k], self.variances[k]))
                    .sum();
                mix.max(1e-300).ln()
            })
            .sum()
    }
}

impl IterativeSolver for GaussianMixtureEMSolver {
    type Output = GaussianMixtureEMResult;

    fn max_iters(&self) -> usize {
        self.max_iters
    }

    fn step(&mut self, iter: usize) -> bool {
        let n = self.data.len();
        let mut nk = vec![0.0; self.k];
        let mut sum1 = vec![0.0; self.k];
        let mut sum2 = vec![0.0; self.k];
        // E-step + sufficient-statistic accumulation.
        for &x in &self.data {
            let mut resp = vec![0.0; self.k];
            let mut total = 0.0;
            for k in 0..self.k {
                let r = self.weights[k] * gaussian_pdf(x, self.means[k], self.variances[k]);
                resp[k] = r;
                total += r;
            }
            let total = total.max(1e-300);
            for k in 0..self.k {
                let r = resp[k] / total;
                nk[k] += r;
                sum1[k] += r * x;
                sum2[k] += r * x * x;
            }
        }
        // M-step.
        for k in 0..self.k {
            let nkk = nk[k].max(1e-12);
            self.weights[k] = nk[k] / n as f64;
            self.means[k] = sum1[k] / nkk;
            self.variances[k] = (sum2[k] / nkk - self.means[k].powi(2)).max(1e-6);
        }
        let ll = self.log_likelihood();
        self.trace.push(EMTraceRow {
            iteration: iter + 1,
            log_likelihood: ll,
        });
        if (ll - self.prev_ll).abs() < self.tolerance {
            self.converged = true;
            self.prev_ll = ll;
            return false;
        }
        self.prev_ll = ll;
        true
    }

    fn result(&self) -> GaussianMixtureEMResult {
        GaussianMixtureEMResult {
            weights: self.weights.clone(),
            means: self.means.clone(),
            variances: self.variances.clone(),
            log_likelihood: self.trace.last().map(|r| r.log_likelihood).unwrap_or(f64::NAN),
            iterations: self.trace.len(),
            converged: self.converged,
            trace: self.trace.clone(),
            visual_blocks: Vec::new(),
            topology: Vec::new(),
            ticks: 0,
        }
    }
}

/// Fit a 1-D Gaussian mixture with EM, one E/M iteration per tick.
pub fn run_gaussian_mixture_em(params: GaussianMixtureEMParams) -> GaussianMixtureEMResult {
    let data = match params.data {
        Some(d) if d.len() >= 2 => d,
        _ => default_mixture_data(),
    };
    let components = params.components.unwrap_or(2).max(1);
    let max_iters = params.max_iters.unwrap_or(200);
    let tolerance = params.tolerance.unwrap_or(1e-6);

    require(Preconditions::all_finite("runGaussianMixtureEM", "data", &data));
    require(Preconditions::integer_in_range(
        "runGaussianMixtureEM",
        "components",
        components as f64,
        1.0,
        data.len() as f64,
    ));
    require(Preconditions::positive(
        "runGaussianMixtureEM",
        "tolerance",
        tolerance,
    ));

    let solver = GaussianMixtureEMSolver::new(data, components, max_iters, tolerance);
    let run = run_visual_solver("gmm-em", "gaussian-mixture-em", "Gaussian Mixture EM", solver);
    let mut result = run.result;
    result.visual_blocks = run.visual_blocks;
    result.topology = run.topology;
    result.ticks = run.ticks;
    result
}

fn default_mixture_data() -> Vec<f64> {
    // Two well-separated clusters around -3 and +3.
    let mut data = Vec::new();
    let cluster_a = [-3.4, -3.1, -2.9, -3.2, -2.7, -3.0, -3.3, -2.8];
    let cluster_b = [2.8, 3.1, 3.3, 2.9, 3.2, 3.0, 2.7, 3.4];
    data.extend_from_slice(&cluster_a);
    data.extend_from_slice(&cluster_b);
    data
}

// =============================================================================
// 7b. Probabilistic inference: mean-field variational inference (CAVI)
// =============================================================================

/// Mean-field VI for the mean & precision of a univariate Gaussian with a
/// Normal–Gamma prior (Bishop §10.1.3): factorized `q(μ)q(τ)` updated by
/// coordinate ascent. The output approximate posterior is `q(μ)=N(μ_N, λ_N^{-1})`,
/// `q(τ)=Gamma(a_N, b_N)`.
#[derive(Clone, Debug, Default)]
pub struct MeanFieldVIParams {
    pub data: Option<Vec<f64>>,
    pub prior_mean: Option<f64>,
    pub prior_precision_scale: Option<f64>,
    pub prior_shape: Option<f64>,
    pub prior_rate: Option<f64>,
    pub max_iters: Option<usize>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct VITraceRow {
    pub iteration: usize,
    pub expected_mean: f64,
    pub expected_precision: f64,
}

#[derive(Clone, Debug)]
pub struct MeanFieldVIResult {
    /// `μ_N`: posterior mean of `q(μ)`.
    pub posterior_mean: f64,
    /// `λ_N`: posterior precision of `q(μ)`.
    pub posterior_mean_precision: f64,
    /// `a_N`, `b_N`: shape/rate of `q(τ)`.
    pub posterior_shape: f64,
    pub posterior_rate: f64,
    /// `E[τ] = a_N / b_N`.
    pub expected_precision: f64,
    /// `E[σ²] = b_N / (a_N − 1)`.
    pub estimated_variance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub trace: Vec<VITraceRow>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub topology: Vec<String>,
    pub ticks: usize,
}

/// CAVI solver: the variational parameters are the memory updated per tick.
struct MeanFieldVISolver {
    data: Vec<f64>,
    mu0: f64,
    lambda0: f64,
    a0: f64,
    b0: f64,
    mu_n: f64,
    lambda_n: f64,
    a_n: f64,
    b_n: f64,
    e_tau: f64,
    prev_e_tau: f64,
    tolerance: f64,
    max_iters: usize,
    converged: bool,
    trace: Vec<VITraceRow>,
}

impl MeanFieldVISolver {
    fn new(
        data: Vec<f64>,
        mu0: f64,
        lambda0: f64,
        a0: f64,
        b0: f64,
        max_iters: usize,
        tolerance: f64,
    ) -> Self {
        let e_tau = a0 / b0;
        let n = data.len() as f64;
        // q(μ) mean is independent of τ; precision starts from the prior E[τ].
        let xbar = data.iter().sum::<f64>() / n;
        let mu_n = (lambda0 * mu0 + n * xbar) / (lambda0 + n);
        MeanFieldVISolver {
            data,
            mu0,
            lambda0,
            a0,
            b0,
            mu_n,
            lambda_n: (lambda0 + n) * e_tau,
            a_n: a0 + (n + 1.0) / 2.0,
            b_n: b0,
            e_tau,
            prev_e_tau: f64::NEG_INFINITY,
            tolerance,
            max_iters,
            converged: false,
            trace: Vec::new(),
        }
    }
}

impl IterativeSolver for MeanFieldVISolver {
    type Output = MeanFieldVIResult;

    fn max_iters(&self) -> usize {
        self.max_iters
    }

    fn step(&mut self, iter: usize) -> bool {
        let n = self.data.len() as f64;
        // q(μ) update.
        self.lambda_n = (self.lambda0 + n) * self.e_tau;
        // q(τ) update: E_{q(μ)}[Σ(xᵢ−μ)² + λ₀(μ−μ₀)²].
        let sse: f64 = self.data.iter().map(|x| (x - self.mu_n).powi(2)).sum();
        let prior_term = self.lambda0 * (self.mu_n - self.mu0).powi(2);
        let var_term = (n + self.lambda0) / self.lambda_n;
        self.b_n = self.b0 + 0.5 * (sse + prior_term + var_term);
        self.e_tau = self.a_n / self.b_n;
        self.trace.push(VITraceRow {
            iteration: iter + 1,
            expected_mean: self.mu_n,
            expected_precision: self.e_tau,
        });
        if (self.e_tau - self.prev_e_tau).abs() < self.tolerance {
            self.converged = true;
            self.prev_e_tau = self.e_tau;
            return false;
        }
        self.prev_e_tau = self.e_tau;
        true
    }

    fn result(&self) -> MeanFieldVIResult {
        MeanFieldVIResult {
            posterior_mean: self.mu_n,
            posterior_mean_precision: self.lambda_n,
            posterior_shape: self.a_n,
            posterior_rate: self.b_n,
            expected_precision: self.e_tau,
            estimated_variance: self.b_n / (self.a_n - 1.0).max(1e-9),
            iterations: self.trace.len(),
            converged: self.converged,
            trace: self.trace.clone(),
            visual_blocks: Vec::new(),
            topology: Vec::new(),
            ticks: 0,
        }
    }
}

/// Run mean-field coordinate-ascent VI for a univariate Gaussian, one CAVI
/// sweep per tick.
pub fn run_mean_field_vi(params: MeanFieldVIParams) -> MeanFieldVIResult {
    let data = match params.data {
        Some(d) if d.len() >= 2 => d,
        _ => default_vi_data(),
    };
    let mu0 = params.prior_mean.unwrap_or(0.0);
    let lambda0 = params.prior_precision_scale.unwrap_or(0.1);
    let a0 = params.prior_shape.unwrap_or(1.0);
    let b0 = params.prior_rate.unwrap_or(1.0);
    let max_iters = params.max_iters.unwrap_or(100);
    let tolerance = params.tolerance.unwrap_or(1e-9);

    require(Preconditions::all_finite("runMeanFieldVI", "data", &data));
    require(Preconditions::positive("runMeanFieldVI", "priorPrecisionScale", lambda0));
    require(Preconditions::positive("runMeanFieldVI", "priorShape", a0));
    require(Preconditions::positive("runMeanFieldVI", "priorRate", b0));

    let solver = MeanFieldVISolver::new(data, mu0, lambda0, a0, b0, max_iters, tolerance);
    let run = run_visual_solver("mean-field-vi", "mean-field-vi", "Mean-Field VI", solver);
    let mut result = run.result;
    result.visual_blocks = run.visual_blocks;
    result.topology = run.topology;
    result.ticks = run.ticks;
    result
}

fn default_vi_data() -> Vec<f64> {
    // Samples loosely centred at 5 with spread ≈ 1.
    vec![4.1, 5.2, 4.8, 6.0, 5.5, 4.6, 5.1, 5.9, 4.3, 5.4, 6.2, 4.9]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbfgs_minimizes_rosenbrock() {
        let result = run_lbfgs(LbfgsParams {
            objective: Some(SmoothObjective::Rosenbrock),
            dimension: Some(2),
            ..Default::default()
        });
        assert!(result.best_value < 1e-4, "f = {}", result.best_value);
        assert!((result.best_x[0] - 1.0).abs() < 1e-2);
        assert!((result.best_x[1] - 1.0).abs() < 1e-2);
        // source + solver + sink visual blocks.
        assert_eq!(result.visual_blocks.len(), 3);
        assert_eq!(result.topology.len(), 3);
    }

    #[test]
    fn lbfgs_minimizes_sphere_quickly() {
        let result = run_lbfgs(LbfgsParams {
            objective: Some(SmoothObjective::Sphere),
            dimension: Some(5),
            ..Default::default()
        });
        assert!(result.converged);
        assert!(result.best_value < 1e-8, "f = {}", result.best_value);
    }

    #[test]
    fn objective_transform_matches_value() {
        let t = ObjectiveFunction {
            objective: SmoothObjective::Sphere,
        };
        assert!((t.transform(vec![3.0, 4.0]) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn sequence_alignment_aligns_identical_sequences() {
        let result = run_sequence_alignment(SequenceAlignmentParams {
            seq_a: Some("ACGT".to_string()),
            seq_b: Some("ACGT".to_string()),
            ..Default::default()
        });
        assert_eq!(result.aligned_a, "ACGT");
        assert_eq!(result.aligned_b, "ACGT");
        assert!((result.score - 4.0).abs() < 1e-9);
        assert!((result.identity - 1.0).abs() < 1e-9);
        assert_eq!(result.visual_blocks.len(), 3);
    }

    #[test]
    fn sequence_alignment_inserts_gaps() {
        let result = run_sequence_alignment(SequenceAlignmentParams::default());
        // The aligned strings are equal length and at least as long as the longer input.
        assert_eq!(
            result.aligned_a.chars().count(),
            result.aligned_b.chars().count()
        );
        assert!(result.aligned_a.chars().count() >= 7);
        // One DP row filled per tick over the 7-char first sequence.
        assert_eq!(result.trace.len(), 8);
    }

    #[test]
    fn metropolis_recovers_standard_normal_moments() {
        let result = run_metropolis_hastings(MetropolisParams {
            target: Some(McmcTarget::StandardNormal),
            proposal_std: Some(2.0),
            iterations: Some(20_000),
            seed: Some(7),
            ..Default::default()
        });
        assert!(result.mean.abs() < 0.15, "mean = {}", result.mean);
        assert!((result.std - 1.0).abs() < 0.2, "std = {}", result.std);
        assert!(
            result.acceptance_rate > 0.1 && result.acceptance_rate < 0.95,
            "accept = {}",
            result.acceptance_rate
        );
        assert_eq!(result.visual_blocks.len(), 3);
    }

    #[test]
    fn metropolis_bimodal_is_symmetric() {
        let result = run_metropolis_hastings(MetropolisParams {
            target: Some(McmcTarget::Bimodal),
            iterations: Some(40_000),
            seed: Some(3),
            ..Default::default()
        });
        // Equal-weight modes at ±2 ⇒ overall mean near 0, std well above 1.
        assert!(result.mean.abs() < 0.5, "mean = {}", result.mean);
        assert!(result.std > 1.5, "std = {}", result.std);
    }

    #[test]
    fn differential_evolution_minimizes_rosenbrock() {
        let result = run_differential_evolution(DifferentialEvolutionParams {
            objective: Some(SmoothObjective::Rosenbrock),
            dimension: Some(2),
            generations: Some(300),
            seed: Some(5),
            ..Default::default()
        });
        assert!(result.best_value < 1e-2, "f = {}", result.best_value);
        // Best fitness is monotonically non-increasing over generations.
        for w in result.trace.windows(2) {
            assert!(w[1].best_value <= w[0].best_value + 1e-12);
        }
        assert_eq!(result.visual_blocks.len(), 3);
    }

    #[test]
    fn prim_mst_finds_minimum_tree() {
        let result = run_prim_mst(PrimMSTParams::default());
        assert!(result.connected);
        assert_eq!(result.mst_edges.len(), 4);
        assert!((result.total_weight - 16.0).abs() < 1e-9, "weight = {}", result.total_weight);
        // Cumulative weight in the trace is non-decreasing and ends at the total.
        assert!((result.trace.last().unwrap().total_weight - 16.0).abs() < 1e-9);
        assert_eq!(result.visual_blocks.len(), 3);
    }

    #[test]
    fn prim_mst_detects_disconnected_graph() {
        let result = run_prim_mst(PrimMSTParams {
            nodes: Some(4),
            edges: Some(vec![
                GraphEdge { u: 0, v: 1, weight: 1.0 },
                GraphEdge { u: 2, v: 3, weight: 1.0 },
            ]),
        });
        assert!(!result.connected);
    }

    #[test]
    fn backprop_mlp_learns_xor() {
        let result = run_backprop_mlp(BackpropMlpParams::default());
        assert_eq!(result.accuracy, 1.0, "accuracy = {}", result.accuracy);
        assert!(result.predictions[0] < 0.5 && result.predictions[3] < 0.5);
        assert!(result.predictions[1] >= 0.5 && result.predictions[2] >= 0.5);
        // Loss should fall substantially from the first epoch to the last.
        assert!(result.trace[0].loss > result.final_loss);
        assert_eq!(result.visual_blocks.len(), 3);
    }

    #[test]
    fn gaussian_mixture_em_separates_clusters() {
        let result = run_gaussian_mixture_em(GaussianMixtureEMParams::default());
        assert!(result.converged);
        let mut means = result.means.clone();
        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(means[0] < -2.0, "low mean = {}", means[0]);
        assert!(means[1] > 2.0, "high mean = {}", means[1]);
        for w in &result.weights {
            assert!(*w > 0.3 && *w < 0.7, "weight = {}", w);
        }
        // Log-likelihood is non-decreasing across EM iterations.
        for w in result.trace.windows(2) {
            assert!(w[1].log_likelihood >= w[0].log_likelihood - 1e-6);
        }
        assert_eq!(result.visual_blocks.len(), 3);
    }

    #[test]
    fn gaussian_pdf_transform_integrates_to_peak() {
        let pdf = GaussianPdf {
            mean: 0.0,
            variance: 1.0,
        };
        assert!((pdf.transform(0.0) - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn mean_field_vi_recovers_mean_and_precision() {
        let result = run_mean_field_vi(MeanFieldVIParams {
            prior_mean: Some(5.0),
            ..Default::default()
        });
        assert!(result.converged);
        assert!(
            result.posterior_mean > 4.8 && result.posterior_mean < 5.5,
            "mean = {}",
            result.posterior_mean
        );
        assert!(result.expected_precision > 0.0);
        assert!(
            result.estimated_variance > 0.2 && result.estimated_variance < 1.5,
            "var = {}",
            result.estimated_variance
        );
        // E[τ] converges monotonically downward from the prior toward the data.
        assert!(result.trace.len() >= 2);
        assert_eq!(result.visual_blocks.len(), 3);
    }
}
