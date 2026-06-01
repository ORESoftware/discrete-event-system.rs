//! Reusable **source → solver → sink** pipeline for one-iteration-per-tick algorithms.
//!
//! Models that advance by discrete iterations (gradient descent, DP rows, MCMC
//! steps, EM updates, …) implement [`IterativeSolver`] and call
//! [`run_visual_solver`] to wire three [`VisualBlock`] stations, drive
//! [`run_iterative_des`], and harvest a [`VisualSolverRun`] with visual specs.
//!
//! This is the shared base family for composable numerical solvers. Concrete
//! algorithms only supply the iteration hook; the DES topology and visual
//! integration are identical across models.

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use super::runner::{run_iterative_des, IterativeRunOptions};
use super::station::{AnyToken, DESStation, StationCore, StationRef};
use super::visual_block::{
    visual_block_specs, VisualBlock, VisualBlockOptions, VisualBlockPortSpec, VisualBlockRole,
    VisualBlockSpec, VisualBlockStyle, VisualPortInput,
};

/// Channel carrying the one-shot solve-start token.
pub const CH_START: &str = "start";
/// Channel carrying the terminal result token.
pub const CH_RESULT: &str = "result";

/// The one required hook for a model: advance the algorithm by one iteration.
///
/// The owning [`SolverStation`] calls [`step`](IterativeSolver::step) once per
/// tick until it returns `false` (converged / done) or [`max_iters`](IterativeSolver::max_iters)
/// is reached, then snapshots [`result`](IterativeSolver::result).
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
pub struct SolverStation<S: IterativeSolver> {
    visual: VisualBlock,
    solver: S,
    started: bool,
    iter: usize,
    finished: bool,
    result_emitted: bool,
}

impl<S: IterativeSolver + 'static> SolverStation<S> {
    /// Build a solver station with the given visual identity and algorithm state.
    pub fn new(id: &str, kind: &str, label: &str, solver: S) -> Self {
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

    /// The composed visual block (for spec harvesting).
    pub fn visual(&self) -> &VisualBlock {
        &self.visual
    }

    /// Iterations executed so far.
    pub fn iterations(&self) -> usize {
        self.iter
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
        let tokens = self
            .visual
            .core_mut()
            .drain::<SolverResultToken<R>>(CH_RESULT);
        if let Some(last) = tokens.last() {
            self.latest = Some(last.result.clone());
        }
    }
}

/// The outcome of driving a [`SolverStation`] through the visual DES pipeline.
#[derive(Clone, Debug)]
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
pub fn run_visual_solver<S>(
    base: &str,
    kind: &str,
    label: &str,
    solver: S,
) -> VisualSolverRun<S::Output>
where
    S: IterativeSolver + 'static,
{
    let max_iters = solver.max_iters();
    let source = Rc::new(RefCell::new(SolverSourceBlock::new(&format!(
        "{base}-source"
    ))));
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
    let iterations = station.borrow().iterations();

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
