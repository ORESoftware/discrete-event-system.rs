//! Port of `src/des/general/des-base/single-state-optimizer.ts`.
//!
//! Template-method base for SINGLE-WALKER iterative optimisation (simulated
//! annealing, hill climbing, tabu search, threshold accepting, …) over a
//! generic state `S`.
//!
//! ## Problem shape
//!
//! Minimise `f(s)` over `s ∈ S` by repeatedly proposing a candidate
//! `s' ∈ N(s)` and conditionally accepting it. The DIFFERENTIATOR among the
//! algorithms in this family is the ACCEPTANCE rule:
//!
//!   * SA:        accept if `Δ ≤ 0` OR `rng() < exp(−Δ/T_iter)`
//!   * Hill climb: accept iff `Δ < 0`
//!   * Tabu:      accept best non-tabu candidate (uses memory)
//!   * Threshold: accept if `Δ ≤ τ_iter`
//!
//! ## Template-method mapping (TS `abstract class` → Rust)
//!
//! TypeScript modelled this as `abstract class SingleStateOptimizer<S> extends
//! DESStation` whose `runTimeStep` is a FINAL template method that calls
//! abstract hooks (`initialState`, `cost`, `propose`, `accept`, `clone`,
//! `shouldStop`) plus optional hooks (`onAccept`, `onReject`, `onBootstrap`,
//! `onFinish`). Concrete algorithms (SA, hill climb) subclass and override the
//! hooks. Rust has no abstract-method inheritance, so we split the class:
//!
//!   * [`SingleStateState`] — a plain struct holding the bookkeeping fields the
//!     TS base owned (`current`, `best`, costs, counters, history, the injected
//!     RNG, …). A concrete optimizer EMBEDS one of these and exposes it via
//!     `opt_state()` / `opt_state_mut()`.
//!   * [`SingleStateOptimizer`] — the hook trait (`: DESStation`). REQUIRED
//!     methods are the abstract hooks; the optional hooks have default impls.
//!     The template method itself is the PROVIDED method
//!     [`SingleStateOptimizer::optimizer_step`] (plus the bootstrap helpers and
//!     accessors), which calls the hooks. A concrete optimizer just delegates
//!     `DESStation::run_time_step` → `self.optimizer_step()` and
//!     `DESStation::has_work` → `self.optimizer_has_work()`.
//!
//! The injected `rng: () => number` becomes a boxed
//! [`RandomSource`](crate::des::shared::capabilities::RandomSource) stored in
//! [`SingleStateState`]; the template method threads it into the hooks as
//! `&mut dyn RandomSource` (it is temporarily moved out of the state while a
//! hook runs so that `&mut self` and `&mut rng` do not alias). `throw new
//! Error` (double-init, non-finite cost, >1 seed token, read-before-init) maps
//! to `panic!`. `number` → `f64`, indices/counters → `usize`.

use std::any::Any;
use std::rc::Rc;

use crate::des::general::des_base::station::{AnyToken, DESStation, StationCore};
use crate::des::shared::capabilities::RandomSource;

/// Channel carrying the one-shot initial-state seed token.
pub const SINGLE_STATE_INITIAL_CHANNEL: &str = "single-state-initial";
/// Channel carrying the terminal result snapshot token.
pub const SINGLE_STATE_RESULT_CHANNEL: &str = "single-state-result";

/// Seed token emitted by a source station to bootstrap the walker.
pub struct SingleStateInitialToken<S> {
    pub state: S,
}

impl<S> SingleStateInitialToken<S> {
    pub fn new(state: S) -> Self {
        SingleStateInitialToken { state }
    }
}

/// Immutable snapshot of optimiser progress at termination.
#[derive(Clone)]
pub struct SingleStateResultSnapshot<S> {
    pub best: S,
    pub best_cost: f64,
    pub current: S,
    pub current_cost: f64,
    pub iteration: usize,
    pub accepted_count: usize,
    pub improve_count: usize,
}

/// Terminal result token emitted on [`SINGLE_STATE_RESULT_CHANNEL`].
pub struct SingleStateResultToken<S> {
    pub snapshot: SingleStateResultSnapshot<S>,
}

impl<S> SingleStateResultToken<S> {
    pub fn new(snapshot: SingleStateResultSnapshot<S>) -> Self {
        SingleStateResultToken { snapshot }
    }
}

/// Source station that emits a single initial-state token exactly once.
///
/// Mirrors `SingleStateSourceStation<S>`: `initialState` and the optional
/// `validateInitialState` validator become boxed closures.
pub struct SingleStateSourceStation<S> {
    core: StationCore,
    emitted: bool,
    initial_state: Box<dyn FnMut() -> S>,
    validate_initial_state: Box<dyn FnMut(&S)>,
}

impl<S: 'static> SingleStateSourceStation<S> {
    /// Associated const mirroring `static readonly CH_INITIAL_STATE`.
    pub const CH_INITIAL_STATE: &'static str = SINGLE_STATE_INITIAL_CHANNEL;

    pub fn new(id: impl Into<String>, initial_state: impl FnMut() -> S + 'static) -> Self {
        SingleStateSourceStation {
            core: StationCore::new(id),
            emitted: false,
            initial_state: Box::new(initial_state),
            validate_initial_state: Box::new(|_| {}),
        }
    }

    pub fn with_validator(
        id: impl Into<String>,
        initial_state: impl FnMut() -> S + 'static,
        validate_initial_state: impl FnMut(&S) + 'static,
    ) -> Self {
        SingleStateSourceStation {
            core: StationCore::new(id),
            emitted: false,
            initial_state: Box::new(initial_state),
            validate_initial_state: Box::new(validate_initial_state),
        }
    }
}

impl<S: 'static> DESStation for SingleStateSourceStation<S> {
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
        !self.emitted
    }

    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        let state = (self.initial_state)();
        (self.validate_initial_state)(&state);
        let token: AnyToken = Rc::new(SingleStateInitialToken::new(state));
        self.core.emit(token, Self::CH_INITIAL_STATE);
        self.emitted = true;
    }
}

/// Sink station that keeps the latest result token.
pub struct SingleStateSinkStation<S> {
    core: StationCore,
    pub latest: Option<Rc<SingleStateResultToken<S>>>,
}

impl<S: 'static> SingleStateSinkStation<S> {
    pub const CH_RESULT: &'static str = SINGLE_STATE_RESULT_CHANNEL;

    pub fn new(id: impl Into<String>) -> Self {
        SingleStateSinkStation { core: StationCore::new(id), latest: None }
    }
}

impl<S: 'static> DESStation for SingleStateSinkStation<S> {
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
        self.core.inbox_size(Self::CH_RESULT) > 0
    }

    fn run_time_step(&mut self) {
        let tokens = self.core.drain::<SingleStateResultToken<S>>(Self::CH_RESULT);
        if let Some(last) = tokens.into_iter().last() {
            self.latest = Some(last);
        }
    }
}

/// Bookkeeping fields owned by the TS `abstract class`, factored into a struct
/// the concrete optimizer embeds. `current!`/`best!` definite-assignment fields
/// become `Option<S>` (two-phase init). The injected RNG is held here so the
/// template method can thread it into the hooks.
pub struct SingleStateState<S> {
    /// Walker's current position.
    pub current: Option<S>,
    /// Cost at `current`.
    pub current_cost: f64,
    /// Best position ever seen.
    pub best: Option<S>,
    /// Cost at `best` (lower is better).
    pub best_cost: f64,
    /// Iteration counter (one increment per step).
    pub iteration: usize,
    /// Number of times `accept` returned true.
    pub accepted_count: usize,
    /// Number of strict improvements (`Δ<0`) accepted.
    pub improve_count: usize,
    /// True after the runner terminates this station's loop.
    pub finished: bool,
    pub initialized: bool,
    result_emitted: bool,
    /// Best-cost history, downsampled by `trace_stride`.
    pub best_history: Vec<f64>,
    /// Current-cost history, downsampled.
    pub current_history: Vec<f64>,
    pub trace_stride: usize,
    /// RNG handed to the hooks (moved out transiently during a step).
    pub rng: Option<Box<dyn RandomSource>>,
}

impl<S> SingleStateState<S> {
    /// `trace_stride` defaults to 1 (TS `Math.max(1, opts.traceStride ?? 1)`).
    pub fn new(trace_stride: usize, rng: Box<dyn RandomSource>) -> Self {
        SingleStateState {
            current: None,
            current_cost: 0.0,
            best: None,
            best_cost: 0.0,
            iteration: 0,
            accepted_count: 0,
            improve_count: 0,
            finished: false,
            initialized: false,
            result_emitted: false,
            best_history: Vec::new(),
            current_history: Vec::new(),
            trace_stride: trace_stride.max(1),
            rng: Some(rng),
        }
    }
}

/// The single-state optimiser hook trait. REQUIRED methods are the TS abstract
/// hooks; optional hooks have default impls. The PROVIDED methods
/// (`bootstrap*`, `optimizer_step`, `optimizer_has_work`, accessors) make up the
/// template method and must NOT be overridden by concrete algorithms.
pub trait SingleStateOptimizer<S: Clone + 'static>: DESStation {
    /// Associated consts mirroring the TS static channel names.
    const CH_INITIAL_STATE: &'static str = SINGLE_STATE_INITIAL_CHANNEL;
    const CH_RESULT: &'static str = SINGLE_STATE_RESULT_CHANNEL;

    /// Borrow the embedded bookkeeping state.
    fn opt_state(&self) -> &SingleStateState<S>;
    /// Mutably borrow the embedded bookkeeping state.
    fn opt_state_mut(&mut self) -> &mut SingleStateState<S>;

    // ── HOOKS (required) ─────────────────────────────────────────────────────

    /// Build the initial walker state.
    fn initial_state(&self, rng: &mut dyn RandomSource) -> S;
    /// Compute scalar cost of a state (lower = better).
    fn cost(&self, state: &S) -> f64;
    /// Propose a neighbour `s' ∈ N(s)`. MUST NOT mutate `state`.
    fn propose(&self, state: &S, rng: &mut dyn RandomSource) -> S;
    /// Decide whether to move from `current` to `candidate`.
    fn accept(
        &self,
        current: &S,
        candidate: &S,
        current_cost: f64,
        candidate_cost: f64,
        iter: usize,
        rng: &mut dyn RandomSource,
    ) -> bool;
    /// Deep copy a state (for stashing in `best`). Defaults to `Clone`.
    fn clone_state(&self, state: &S) -> S {
        state.clone()
    }
    /// Return true to terminate the optimiser.
    fn should_stop(&self, iter: usize) -> bool;

    // ── HOOKS (optional) ───────────────────────────────────────────────────────

    fn on_accept(&mut self, _candidate: &S, _delta: f64, _iter: usize) {}
    fn on_reject(&mut self, _candidate: &S, _delta: f64, _iter: usize) {}
    fn on_bootstrap(&mut self) {}
    fn on_finish(&mut self) {}

    // ── BOOTSTRAP (template helpers) ──────────────────────────────────────────

    /// Seed `current`/`best` from `initial_state(rng)`. Call once after
    /// construction (the TS base could not call abstract methods in its ctor).
    fn bootstrap(&mut self) {
        let mut rng = self.opt_state_mut().rng.take().expect("rng already in use");
        let state = self.initial_state(&mut *rng);
        self.opt_state_mut().rng = Some(rng);
        self.bootstrap_from_state(state);
    }

    /// Source-driven bootstrap from an explicit initial state.
    fn bootstrap_from_state(&mut self, initial_state: S) {
        if self.opt_state().initialized {
            panic!("{}: initial state already supplied", self.id());
        }
        let current = self.clone_state(&initial_state);
        let current_cost = self.cost(&current);
        if !current_cost.is_finite() {
            panic!("{}: initial state cost must be finite; got {}", self.id(), current_cost);
        }
        let best = self.clone_state(&current);
        {
            let st = self.opt_state_mut();
            st.current = Some(current);
            st.current_cost = current_cost;
            st.best = Some(best);
            st.best_cost = current_cost;
            st.best_history.push(current_cost);
            st.current_history.push(current_cost);
            st.initialized = true;
        }
        self.on_bootstrap();
    }

    // ── TEMPLATE METHOD (do NOT override) ─────────────────────────────────────

    /// Drives one iteration. Concrete optimizers delegate
    /// `DESStation::run_time_step` to this.
    fn optimizer_step(&mut self) {
        if self.opt_state().finished {
            return;
        }
        if !self.opt_state().initialized {
            let seeds = self.core_mut().drain::<SingleStateInitialToken<S>>(Self::CH_INITIAL_STATE);
            if seeds.is_empty() {
                return;
            }
            if seeds.len() > 1 {
                panic!("{}: expected exactly one initial-state token, got {}", self.id(), seeds.len());
            }
            let state = seeds[0].state.clone();
            self.bootstrap_from_state(state);
            return;
        }
        if self.core().inbox_size(Self::CH_INITIAL_STATE) > 0 {
            panic!("{}: received an initial-state token after initialization", self.id());
        }
        let iter = self.opt_state().iteration;
        if self.should_stop(iter) {
            self.opt_state_mut().finished = true;
            self.on_finish();
            self.emit_result();
            return;
        }
        let current = self.clone_state(self.opt_state().current.as_ref().expect("initialized"));
        let current_cost = self.opt_state().current_cost;
        let mut rng = self.opt_state_mut().rng.take().expect("rng already in use");
        let candidate = self.propose(&current, &mut *rng);
        let cand_cost = self.cost(&candidate);
        let delta = cand_cost - current_cost;
        let ok = self.accept(&current, &candidate, current_cost, cand_cost, iter, &mut *rng);
        self.opt_state_mut().rng = Some(rng);
        if ok {
            if cand_cost < self.opt_state().best_cost {
                let new_best = self.clone_state(&candidate);
                let st = self.opt_state_mut();
                st.best_cost = cand_cost;
                st.best = Some(new_best);
            }
            {
                let st = self.opt_state_mut();
                st.current_cost = cand_cost;
                st.accepted_count += 1;
                if delta < 0.0 {
                    st.improve_count += 1;
                }
            }
            self.on_accept(&candidate, delta, iter);
            self.opt_state_mut().current = Some(candidate);
        } else {
            self.on_reject(&candidate, delta, iter);
        }
        {
            let st = self.opt_state_mut();
            if st.iteration % st.trace_stride == 0 {
                st.best_history.push(st.best_cost);
                st.current_history.push(st.current_cost);
            }
            st.iteration += 1;
        }
    }

    /// `hasWork` override: keep ticking while seeded or running.
    fn optimizer_has_work(&self) -> bool {
        self.core().inbox_size(Self::CH_INITIAL_STATE) > 0
            || (self.opt_state().initialized && !self.opt_state().finished)
    }

    // ── PUBLIC ACCESSORS ──────────────────────────────────────────────────────

    fn get_best(&self) -> &S {
        self.assert_initialized_for_read();
        self.opt_state().best.as_ref().expect("initialized")
    }
    fn get_best_cost(&self) -> f64 {
        self.assert_initialized_for_read();
        self.opt_state().best_cost
    }
    fn get_current(&self) -> &S {
        self.assert_initialized_for_read();
        self.opt_state().current.as_ref().expect("initialized")
    }
    fn get_current_cost(&self) -> f64 {
        self.assert_initialized_for_read();
        self.opt_state().current_cost
    }
    fn get_iteration(&self) -> usize {
        self.opt_state().iteration
    }
    fn get_accepted_count(&self) -> usize {
        self.opt_state().accepted_count
    }
    fn get_improve_count(&self) -> usize {
        self.opt_state().improve_count
    }
    fn is_finished(&self) -> bool {
        self.opt_state().finished
    }
    fn is_initialized(&self) -> bool {
        self.opt_state().initialized
    }

    fn emit_result(&mut self) {
        if self.opt_state().result_emitted {
            return;
        }
        let best = self.clone_state(self.opt_state().best.as_ref().expect("initialized"));
        let current = self.clone_state(self.opt_state().current.as_ref().expect("initialized"));
        let snapshot = {
            let st = self.opt_state();
            SingleStateResultSnapshot {
                best,
                best_cost: st.best_cost,
                current,
                current_cost: st.current_cost,
                iteration: st.iteration,
                accepted_count: st.accepted_count,
                improve_count: st.improve_count,
            }
        };
        let token: AnyToken = Rc::new(SingleStateResultToken::new(snapshot));
        self.core_mut().emit(token, Self::CH_RESULT);
        self.opt_state_mut().result_emitted = true;
    }

    fn assert_initialized_for_read(&self) {
        if !self.opt_state().initialized {
            panic!("{}: optimizer has not received an initial state", self.id());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::station::StationRef;
    use crate::des::shared::capabilities::SeededRandom;
    use std::cell::RefCell;

    /// Tiny concrete optimizer: minimise `(x - target)^2` over `x: f64` by
    /// hill-climbing with uniform neighbour proposals.
    struct Quadratic {
        core: StationCore,
        state: SingleStateState<f64>,
        target: f64,
        step: f64,
        max_iter: usize,
    }

    impl Quadratic {
        fn new(seed: u32, target: f64, step: f64, max_iter: usize) -> Self {
            Quadratic {
                core: StationCore::new("quad"),
                state: SingleStateState::new(1, Box::new(SeededRandom::new(seed))),
                target,
                step,
                max_iter,
            }
        }
    }

    impl DESStation for Quadratic {
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

    impl SingleStateOptimizer<f64> for Quadratic {
        fn opt_state(&self) -> &SingleStateState<f64> {
            &self.state
        }
        fn opt_state_mut(&mut self) -> &mut SingleStateState<f64> {
            &mut self.state
        }
        fn initial_state(&self, _rng: &mut dyn RandomSource) -> f64 {
            0.0
        }
        fn cost(&self, state: &f64) -> f64 {
            (state - self.target).powi(2)
        }
        fn propose(&self, state: &f64, rng: &mut dyn RandomSource) -> f64 {
            state + (rng.next_float() - 0.5) * 2.0 * self.step
        }
        fn accept(
            &self,
            _current: &f64,
            _candidate: &f64,
            current_cost: f64,
            candidate_cost: f64,
            _iter: usize,
            _rng: &mut dyn RandomSource,
        ) -> bool {
            candidate_cost <= current_cost
        }
        fn should_stop(&self, iter: usize) -> bool {
            iter >= self.max_iter
        }
    }

    #[test]
    fn hill_climb_converges() {
        let mut opt = Quadratic::new(42, 3.0, 0.5, 500);
        opt.bootstrap();
        assert_eq!(opt.get_current(), &0.0);
        while !opt.is_finished() {
            opt.run_time_step();
        }
        assert!(opt.is_finished());
        assert!(opt.get_best_cost() < 1.0, "best_cost = {}", opt.get_best_cost());
        assert!(opt.get_improve_count() > 0);
    }

    #[test]
    fn bootstraps_from_seed_token() {
        let mut opt = Quadratic::new(7, 3.0, 0.5, 100);
        let token: AnyToken = Rc::new(SingleStateInitialToken::new(5.0_f64));
        opt.core_mut().take(token, SINGLE_STATE_INITIAL_CHANNEL);
        assert!(!opt.is_initialized());
        opt.run_time_step();
        assert!(opt.is_initialized());
        assert_eq!(opt.get_current(), &5.0);
        assert_eq!(opt.get_best_cost(), 4.0);
    }

    #[test]
    fn source_optimizer_sink_pipeline() {
        let source = Rc::new(RefCell::new(SingleStateSourceStation::new("src", || 0.0_f64)));
        let opt = Rc::new(RefCell::new(Quadratic::new(99, 3.0, 0.6, 300)));
        let sink = Rc::new(RefCell::new(SingleStateSinkStation::<f64>::new("sink")));

        source
            .borrow_mut()
            .core_mut()
            .pipe(opt.clone() as StationRef, SINGLE_STATE_INITIAL_CHANNEL, SINGLE_STATE_INITIAL_CHANNEL);
        opt.borrow_mut()
            .core_mut()
            .pipe(sink.clone() as StationRef, SINGLE_STATE_RESULT_CHANNEL, SINGLE_STATE_RESULT_CHANNEL);

        source.borrow_mut().run_time_step();
        let mut guard = 0;
        while !opt.borrow().is_finished() {
            opt.borrow_mut().run_time_step();
            guard += 1;
            assert!(guard < 10_000, "optimizer did not finish");
        }
        sink.borrow_mut().run_time_step();
        let latest = sink.borrow().latest.clone().expect("result captured");
        assert!(latest.snapshot.best_cost < 1.0, "best_cost = {}", latest.snapshot.best_cost);
        assert_eq!(latest.snapshot.iteration, 300);
    }
}
