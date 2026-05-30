//! Rust port of `src/des/general/des-base/fixed-point.ts`.

use super::station::{DESRunLoopEntity, DESStation, HasRunTimeStep, StationCore};
use super::validation::ValidationCheck;
use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/des-base/fixed-point.ts",
    "src/des/general/des_base/fixed_point.rs",
    &[
        "FixedPointOptions is a config struct.",
        "FixedPointIterationStation is a generic station over hook traits.",
        "Residual history is preserved as Vec<f64>.",
        "The template-method run_time_step mirrors the TS base class.",
    ],
    &["FixedPointIterationStation", "FixedPointOptions"],
);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedPointOptions {
    pub tol: f64,
    pub max_iter: usize,
    pub max_history_len: Option<usize>,
}

impl Default for FixedPointOptions {
    fn default() -> Self {
        Self {
            tol: 1e-9,
            max_iter: 5000,
            max_history_len: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
    Converged,
    MaxIter,
    Running,
}

pub trait FixedPointHooks<S> {
    fn initial_state(&mut self) -> S;
    fn apply_operator(&mut self, prev: &S) -> S;
    fn delta(&self, prev: &S, next: &S) -> f64;

    fn on_iteration(&mut self, _iter: usize, _delta: f64) {}
    fn on_converged(&mut self, _iter: usize, _delta: f64) {}
    fn on_max_iter(&mut self, _iter: usize, _delta: f64) {}
}

pub struct FixedPointIterationStation<S, H> {
    core: StationCore<Self>,
    current: S,
    iteration: usize,
    last_delta: f64,
    finished: bool,
    convergence_reason: ConvergenceReason,
    pub delta_history: Vec<f64>,
    tol: f64,
    max_iter: usize,
    max_history_len: Option<usize>,
    hooks: H,
}

impl<S, H> FixedPointIterationStation<S, H>
where
    H: FixedPointHooks<S>,
{
    pub fn new(id: impl Into<String>, mut hooks: H, opts: FixedPointOptions) -> Self {
        let current = hooks.initial_state();
        Self {
            core: StationCore::new(id),
            current,
            iteration: 0,
            last_delta: f64::INFINITY,
            finished: false,
            convergence_reason: ConvergenceReason::Running,
            delta_history: Vec::new(),
            tol: opts.tol,
            max_iter: opts.max_iter,
            max_history_len: opts.max_history_len,
            hooks,
        }
    }

    fn should_stop(&mut self) -> bool {
        if self.iteration >= self.max_iter {
            self.convergence_reason = ConvergenceReason::MaxIter;
            return true;
        }
        if self.iteration > 0 && self.last_delta < self.tol {
            self.convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        false
    }

    pub fn current(&self) -> &S {
        &self.current
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    pub fn last_delta(&self) -> f64 {
        self.last_delta
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn reason(&self) -> ConvergenceReason {
        self.convergence_reason
    }
}

impl<S, H> HasRunTimeStep for FixedPointIterationStation<S, H>
where
    H: FixedPointHooks<S>,
{
    fn run_time_step(&mut self) {
        if self.finished {
            return;
        }
        if self.should_stop() {
            self.finished = true;
            match self.convergence_reason {
                ConvergenceReason::Converged => {
                    self.hooks.on_converged(self.iteration, self.last_delta);
                }
                ConvergenceReason::MaxIter => {
                    self.hooks.on_max_iter(self.iteration, self.last_delta);
                }
                ConvergenceReason::Running => {}
            }
            return;
        }

        let next = self.hooks.apply_operator(&self.current);
        self.last_delta = self.hooks.delta(&self.current, &next);
        self.current = next;
        self.iteration += 1;
        if self
            .max_history_len
            .map(|max_len| self.delta_history.len() < max_len)
            .unwrap_or(true)
        {
            self.delta_history.push(self.last_delta);
        }
        self.hooks.on_iteration(self.iteration, self.last_delta);
    }
}

impl<S, H> DESRunLoopEntity for FixedPointIterationStation<S, H>
where
    H: FixedPointHooks<S>,
{
    fn id(&self) -> Option<&str> {
        Some(self.core.id())
    }

    fn has_work(&self) -> bool {
        !self.finished
    }

    fn num_validators(&self) -> usize {
        self.core.num_validators()
    }

    fn run_validation(&self) -> Vec<ValidationCheck> {
        self.core.run_validation(self)
    }
}

impl<S, H> DESStation for FixedPointIterationStation<S, H>
where
    H: FixedPointHooks<S>,
{
    fn core(&self) -> &StationCore<Self> {
        &self.core
    }

    fn core_mut(&mut self) -> &mut StationCore<Self> {
        &mut self.core
    }
}
