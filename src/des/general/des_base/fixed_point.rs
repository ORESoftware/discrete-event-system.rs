//! Port of `src/des/general/des-base/fixed-point.ts` — template-method base for
//! FIXED-POINT iteration: value/policy iteration, Jacobi / Gauss-Seidel,
//! alpha-vector / belief backups, Benders convergence, equilibrium computation.
//!
//! We iterate `x_{k+1} = T(x_k)` until `‖x_{k+1} − x_k‖ < tol` or `k ≥ maxIter`.
//!
//! ## Rust shape
//!
//! `abstract class FixedPointIterationStation<S> extends DESStation` becomes the
//! [`FixedPointIterationStation`] trait extending [`DESStation`]:
//!
//!   * Iteration state lives in [`FixedPointCore`] (the `current!`
//!     definite-assignment field becomes `Option<S>`, populated by
//!     [`FixedPointIterationStation::bootstrap`]).
//!   * Required hooks `initialState` / `applyOperator` / `delta` → required
//!     trait fns. `applyOperator` returns a NEW state (`&S -> S`).
//!   * Optional hooks `shouldStop` / `onIteration` / `onConverged` /
//!     `onMaxIter` → provided defaults.
//!   * The *final* `runTimeStep` → the provided template
//!     [`FixedPointIterationStation::fixed_point_run_time_step`]; concrete
//!     stations delegate to it from [`DESStation::run_time_step`].
//!   * `convergenceReason: 'converged'|'maxiter'|'running'` →
//!     [`ConvergenceReason`].
//!   * `maxIter`/`maxHistoryLen` default `Infinity` → `usize` resolved from
//!     `Option` (`usize::MAX` for unbounded history).

use super::station::DESStation;

/// Why iteration stopped (TS string union).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvergenceReason {
    Converged,
    MaxIter,
    Running,
}

/// Optional configuration (TS `interface FixedPointOptions`, all fields
/// optional). Resolved into [`FixedPointCore`] with the TS defaults.
#[derive(Clone, Debug, Default)]
pub struct FixedPointOptions {
    /// Convergence tolerance on `delta`. Default `1e-9`.
    pub tol: Option<f64>,
    /// Hard cap on iterations. Default `5000`.
    pub max_iter: Option<usize>,
    /// Cap on history length. Default unbounded (`usize::MAX`).
    pub max_history_len: Option<usize>,
}

/// Iteration state (the fields of the TS abstract class).
pub struct FixedPointCore<S> {
    current: Option<S>,
    pub iteration: usize,
    pub last_delta: f64,
    pub finished: bool,
    pub convergence_reason: ConvergenceReason,
    /// Per-iteration delta history (recorded up to `max_history_len`).
    pub delta_history: Vec<f64>,
    pub tol: f64,
    pub max_iter: usize,
    pub max_history_len: usize,
}

impl<S> FixedPointCore<S> {
    pub fn new(opts: FixedPointOptions) -> Self {
        FixedPointCore {
            current: None,
            iteration: 0,
            last_delta: f64::INFINITY,
            finished: false,
            convergence_reason: ConvergenceReason::Running,
            delta_history: Vec::new(),
            tol: opts.tol.unwrap_or(1e-9),
            max_iter: opts.max_iter.unwrap_or(5000),
            max_history_len: opts.max_history_len.unwrap_or(usize::MAX),
        }
    }
}

/// Template-method base for fixed-point iteration.
pub trait FixedPointIterationStation<S>: DESStation {
    /// Borrow iteration state.
    fn fp_core(&self) -> &FixedPointCore<S>;
    /// Mutably borrow iteration state.
    fn fp_core_mut(&mut self) -> &mut FixedPointCore<S>;

    // ── HOOKS (abstract) ───────────────────────────────────────────────────────

    /// Build `x_0`.
    fn initial_state(&self) -> S;
    /// Apply the operator `T(x_k)`. MUST return a NEW state (do not mutate `prev`).
    fn apply_operator(&mut self, prev: &S) -> S;
    /// Convergence metric — typically max-norm / L2-norm of `(next − prev)`.
    fn delta(&self, prev: &S, next: &S) -> f64;

    // ── BOOTSTRAP ──────────────────────────────────────────────────────────────

    /// Populate `current` from [`Self::initial_state`]. Concrete stations call
    /// this once after construction (mirrors the TS `bootstrap()` contract).
    fn bootstrap(&mut self) {
        let s0 = self.initial_state();
        self.fp_core_mut().current = Some(s0);
    }

    // ── HOOKS (optional override) ──────────────────────────────────────────────

    fn should_stop(&mut self, iter: usize, last_delta: f64) -> bool {
        if iter >= self.fp_core().max_iter {
            self.fp_core_mut().convergence_reason = ConvergenceReason::MaxIter;
            return true;
        }
        if iter > 0 && last_delta < self.fp_core().tol {
            self.fp_core_mut().convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        false
    }

    fn on_iteration(&mut self, _iter: usize, _delta: f64) {}
    fn on_converged(&mut self, _iter: usize, _delta: f64) {}
    fn on_max_iter(&mut self, _iter: usize, _delta: f64) {}

    // ── TEMPLATE METHOD (final) ────────────────────────────────────────────────

    fn fixed_point_run_time_step(&mut self)
    where
        S: Clone,
    {
        if self.fp_core().finished {
            return;
        }
        let iter = self.fp_core().iteration;
        let last = self.fp_core().last_delta;
        if self.should_stop(iter, last) {
            self.fp_core_mut().finished = true;
            match self.fp_core().convergence_reason {
                ConvergenceReason::Converged => self.on_converged(iter, last),
                ConvergenceReason::MaxIter => self.on_max_iter(iter, last),
                ConvergenceReason::Running => {}
            }
            return;
        }
        let current = self
            .fp_core()
            .current
            .clone()
            .expect("bootstrap() must be called before run");
        let next = self.apply_operator(&current);
        let d = self.delta(&current, &next);
        let max_hist = self.fp_core().max_history_len;
        {
            let c = self.fp_core_mut();
            c.last_delta = d;
            c.current = Some(next);
            c.iteration += 1;
            if c.delta_history.len() < max_hist {
                c.delta_history.push(d);
            }
        }
        let it = self.fp_core().iteration;
        self.on_iteration(it, d);
    }

    /// `hasWork` override: keep ticking until convergence.
    fn fixed_point_has_work(&self) -> bool {
        !self.fp_core().finished
    }

    // ── PUBLIC ACCESSORS ───────────────────────────────────────────────────────

    /// Current iterate (panics if [`Self::bootstrap`] has not run).
    fn current(&self) -> &S {
        self.fp_core()
            .current
            .as_ref()
            .expect("bootstrap() must be called before current()")
    }
    fn iteration(&self) -> usize {
        self.fp_core().iteration
    }
    fn get_last_delta(&self) -> f64 {
        self.fp_core().last_delta
    }
    fn is_finished(&self) -> bool {
        self.fp_core().finished
    }
    fn reason(&self) -> ConvergenceReason {
        self.fp_core().convergence_reason
    }
}

#[cfg(test)]
mod tests {
    use super::super::station::{DESStation, StationCore};
    use super::*;
    use std::any::Any;

    /// Banach fixed point of `x = cos(x)` (the Dottie number ≈ 0.739085).
    struct CosFix {
        core: StationCore,
        fp: FixedPointCore<f64>,
    }

    impl CosFix {
        fn new(opts: FixedPointOptions) -> Self {
            let mut s = CosFix {
                core: StationCore::new("cos-fix"),
                fp: FixedPointCore::new(opts),
            };
            s.bootstrap();
            s
        }
    }

    impl DESStation for CosFix {
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
            self.fixed_point_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.fixed_point_has_work()
        }
    }

    impl FixedPointIterationStation<f64> for CosFix {
        fn fp_core(&self) -> &FixedPointCore<f64> {
            &self.fp
        }
        fn fp_core_mut(&mut self) -> &mut FixedPointCore<f64> {
            &mut self.fp
        }
        fn initial_state(&self) -> f64 {
            1.0
        }
        fn apply_operator(&mut self, prev: &f64) -> f64 {
            prev.cos()
        }
        fn delta(&self, prev: &f64, next: &f64) -> f64 {
            (next - prev).abs()
        }
    }

    #[test]
    fn cos_iteration_converges() {
        let mut s = CosFix::new(FixedPointOptions {
            tol: Some(1e-12),
            ..Default::default()
        });
        let mut guard = 0;
        while s.has_work() {
            s.run_time_step();
            guard += 1;
            assert!(guard < 10_000, "did not converge");
        }
        assert!(s.is_finished());
        assert_eq!(s.reason(), ConvergenceReason::Converged);
        assert!(
            (*s.current() - 0.739_085_133_2).abs() < 1e-6,
            "x = {}",
            s.current()
        );
    }

    #[test]
    fn delta_history_is_recorded() {
        let mut s = CosFix::new(FixedPointOptions::default());
        for _ in 0..5 {
            s.run_time_step();
        }
        assert_eq!(s.iteration(), 5);
        assert_eq!(s.fp_core().delta_history.len(), 5);
        // contraction => deltas shrink monotonically
        let h = &s.fp_core().delta_history;
        assert!(h[1] < h[0] && h[4] < h[1]);
    }

    /// A non-contracting map `x -> x + 1` never converges => hits maxIter.
    struct Diverge {
        core: StationCore,
        fp: FixedPointCore<f64>,
    }

    impl DESStation for Diverge {
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
            self.fixed_point_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.fixed_point_has_work()
        }
    }

    impl FixedPointIterationStation<f64> for Diverge {
        fn fp_core(&self) -> &FixedPointCore<f64> {
            &self.fp
        }
        fn fp_core_mut(&mut self) -> &mut FixedPointCore<f64> {
            &mut self.fp
        }
        fn initial_state(&self) -> f64 {
            0.0
        }
        fn apply_operator(&mut self, prev: &f64) -> f64 {
            prev + 1.0
        }
        fn delta(&self, prev: &f64, next: &f64) -> f64 {
            (next - prev).abs()
        }
    }

    #[test]
    fn max_iter_stops_iteration() {
        let mut s = Diverge {
            core: StationCore::new("diverge"),
            fp: FixedPointCore::new(FixedPointOptions {
                max_iter: Some(5),
                ..Default::default()
            }),
        };
        s.bootstrap();
        while s.has_work() {
            s.run_time_step();
        }
        assert_eq!(s.reason(), ConvergenceReason::MaxIter);
        assert_eq!(s.iteration(), 5);
    }
}
