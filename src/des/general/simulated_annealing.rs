//! Port of `src/des/general/simulated-annealing.ts` — generic single-walker
//! Simulated Annealing as a discrete-event system, built on the
//! [`SingleStateOptimizer`] template-method base, with TSP and knapsack problem
//! adapters and a generic [`SAProblem`] interface so other combinatorial
//! problems can plug in.
//!
//! ## TS → Rust mapping
//!
//!   * `interface SAProblem<S>` (cost/neighbour/initial closures) → the
//!     [`SAProblem`] trait. The `(s) => …` / `(s, rng) => …` closures become
//!     `&self` methods; `neighbour`/`initial` take `&mut dyn RandomSource`. The
//!     optional `clone` is a provided default delegating to `S: Clone`.
//!   * `type CoolingSchedule` (discriminated union on `kind`) → the
//!     [`CoolingSchedule`] enum, matched in [`temperature_at`].
//!   * `interface SASolverOptions / SATickEvent / SAResult<S> / KnapsackInstance`
//!     → structs (`number` → `f64`, optionals → `Option<T>`, indices → `usize`).
//!   * `class SAOptimizer<S> extends SingleStateOptimizer<S>` → a struct
//!     `{ core, state: SingleStateState<S>, problem: Rc<dyn SAProblem<S>>, … }`
//!     that `impl DESStation` (delegating `run_time_step`/`has_work` to the base
//!     template methods) and `impl SingleStateOptimizer<S>`. The TS `accept`
//!     hook mutated instance fields (`currentT`/`currentCandCost`/…) but the
//!     trait's `accept` is `&self`, so those four are held in [`Cell`]s.
//!   * `runSimulatedAnnealing` / `buildTSPSAProblem` / `buildKnapsackSAProblem`
//!     → free functions. `runSimulatedAnnealing` shares ONE RNG stream between
//!     the source station and the optimizer via the local [`SharedRng`] adapter
//!     (the TS code shared a single `mulberry32` closure).
//!   * `throw` (empty result sink) → `panic!`; `Preconditions.finite` →
//!     `Preconditions::finite(..).unwrap_or_else(panic)`.
//!   * `console.debug` early-stop logging is dropped (logging only); behaviour
//!     (the return values of `should_stop`) is unchanged.
//!   * `export {checkPrecedence, isPermutation}` → `pub use` re-exports.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::single_state_optimizer::{
    SingleStateOptimizer, SingleStateSinkStation, SingleStateSourceStation, SingleStateState,
    SINGLE_STATE_INITIAL_CHANNEL, SINGLE_STATE_RESULT_CHANNEL,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::{
    intrinsic_check, monotonicity_validator, Monotonicity,
};
use crate::des::general::genetic_tsp::{tour_length, InitMode, TSPInstance, Tour};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

pub use crate::des::general::genetic_tsp::{check_precedence, is_permutation};

// =============================================================================
// RNG adapter — one shared stream between source + optimizer (see module docs).
// =============================================================================

/// A clonable handle to one shared `SeededRandom` stream.
#[derive(Clone)]
struct SharedRng(Rc<RefCell<SeededRandom>>);

impl SharedRng {
    fn new(seed: u32) -> Self {
        SharedRng(Rc::new(RefCell::new(mulberry32(seed))))
    }
}

impl RandomSource for SharedRng {
    fn next_float(&mut self) -> f64 {
        self.0.borrow_mut().next_float()
    }
}

// =============================================================================
// GENERIC PROBLEM INTERFACE
// =============================================================================

/// A combinatorial problem the SA solver can optimise. (TS `interface
/// SAProblem<S>`.) `S` is the state type; lower cost is better.
pub trait SAProblem<S: Clone> {
    /// Compute the (real-valued) cost of a state. Lower = better.
    fn cost(&self, s: &S) -> f64;
    /// Generate a NEIGHBOUR of the current state. Must NOT mutate `s`.
    fn neighbour(&self, s: &S, rng: &mut dyn RandomSource) -> S;
    /// Build the initial state.
    fn initial(&self, rng: &mut dyn RandomSource) -> S;
    /// Optional cheap clone (TS fell back to `structuredClone`); defaults to
    /// `S: Clone`.
    fn clone_state(&self, s: &S) -> S {
        s.clone()
    }
}

// =============================================================================
// COOLING SCHEDULES
// =============================================================================

/// Temperature schedule. (TS discriminated union on `kind`.) `Tmin` defaults to
/// `0` when absent.
#[derive(Clone, Copy, Debug)]
pub enum CoolingSchedule {
    Geometric {
        t0: f64,
        alpha: f64,
        t_min: Option<f64>,
    },
    Logarithmic {
        t0: f64,
        t_min: Option<f64>,
    },
    Linear {
        t0: f64,
        rate: f64,
        t_min: Option<f64>,
    },
    ExpRestart {
        t0: f64,
        alpha: f64,
        period: usize,
        t_min: Option<f64>,
    },
}

/// Temperature at iteration `k`. (TS `temperatureAt`.)
pub fn temperature_at(s: &CoolingSchedule, k: usize) -> f64 {
    match *s {
        CoolingSchedule::Geometric { t0, alpha, t_min } => {
            t_min.unwrap_or(0.0).max(t0 * alpha.powf(k as f64))
        }
        CoolingSchedule::Logarithmic { t0, t_min } => {
            t_min.unwrap_or(0.0).max(t0 / (2.0 + k as f64).ln())
        }
        CoolingSchedule::Linear { t0, rate, t_min } => {
            t_min.unwrap_or(0.0).max(t0 - rate * k as f64)
        }
        CoolingSchedule::ExpRestart {
            t0,
            alpha,
            period,
            t_min,
        } => t_min
            .unwrap_or(0.0)
            .max(t0 * alpha.powf((k % period) as f64)),
    }
}

// =============================================================================
// SOLVER OPTIONS, EVENTS, RESULTS
// =============================================================================

/// Solver configuration. (TS `interface SASolverOptions`.)
#[derive(Clone, Copy, Debug)]
pub struct SASolverOptions {
    /// Maximum number of iterations (ticks).
    pub max_iterations: usize,
    /// Cooling schedule.
    pub cooling: CoolingSchedule,
    /// Random seed.
    pub seed: Option<u32>,
    /// Stop if best has not improved for this many ticks. 0 = no early stop.
    pub stall_limit: Option<usize>,
    /// Print every tick to stderr (only for tiny problems).
    pub verbose: Option<bool>,
    /// Record full trace (every tick) — costs O(max_iterations) memory.
    pub record_trace: Option<bool>,
    /// Record-trace stride: keep one in N trace entries. Default 1.
    pub trace_stride: Option<usize>,
}

/// One per-tick trace record. (TS `interface SATickEvent`.)
#[derive(Clone, Copy, Debug)]
pub struct SATickEvent {
    pub k: usize,
    pub t: f64,
    pub current_cost: f64,
    pub candidate_cost: f64,
    pub delta: f64,
    pub accept: bool,
    pub accept_prob: f64,
    pub best_cost: f64,
}

/// Final result of an SA run. (TS `interface SAResult<S>`.)
#[derive(Clone, Debug)]
pub struct SAResult<S> {
    pub best_state: S,
    pub best_cost: f64,
    pub final_state: S,
    pub final_cost: f64,
    pub iterations: usize,
    pub accepted_count: usize,
    pub improve_count: usize,
    /// Per-tick trace if `record_trace = true`.
    pub trace: Option<Vec<SATickEvent>>,
    /// Per-record best-cost history (downsampled by `trace_stride`).
    pub best_history: Vec<f64>,
    /// Per-record current-cost history (downsampled).
    pub current_history: Vec<f64>,
    pub temperature_history: Vec<f64>,
}

// =============================================================================
// SAOptimizer<S> — SingleStateOptimizer<S> leaf
// =============================================================================

/// Concrete SA leaf of [`SingleStateOptimizer`]. Hooks: `initial_state`,
/// `cost`, `propose`, `accept` (Metropolis), `clone_state`, `should_stop`.
pub struct SAOptimizer<S: Clone + 'static> {
    core: StationCore,
    state: SingleStateState<S>,
    problem: Rc<dyn SAProblem<S>>,
    cooling: CoolingSchedule,
    max_iters: usize,
    stall_limit: usize,
    verbose: bool,
    record_trace: bool,
    /// Ticks since last best improvement.
    stall_count: usize,
    prev_best: f64,
    /// Captured every iteration when `record_trace`.
    trace: Vec<SATickEvent>,
    temperature_history: Vec<f64>,
    /// Current iteration's temperature (`&self` `accept` writes via `Cell`).
    current_t: Cell<f64>,
    current_cand_cost: Cell<f64>,
    current_accept_prob: Cell<f64>,
    current_accepted: Cell<bool>,
}

fn downcast_sa<S: Clone + 'static>(s: &dyn DESStation) -> &SAOptimizer<S> {
    s.as_any()
        .downcast_ref::<SAOptimizer<S>>()
        .expect("validator received a non-SAOptimizer station")
}

impl<S: Clone + 'static> SAOptimizer<S> {
    /// Construct the optimizer. `defer_bootstrap` mirrors the TS lifecycle flag;
    /// `rng` injects a (possibly shared) stream.
    pub fn new(
        problem: Rc<dyn SAProblem<S>>,
        options: SASolverOptions,
        defer_bootstrap: bool,
        rng: Option<Box<dyn RandomSource>>,
    ) -> Self {
        let seed = options.seed.unwrap_or(42);
        let rng: Box<dyn RandomSource> =
            rng.unwrap_or_else(|| Box::new(mulberry32(seed)) as Box<dyn RandomSource>);
        let trace_stride = options.trace_stride.unwrap_or(1).max(1);
        let mut opt = SAOptimizer {
            core: StationCore::new("simulated-annealing"),
            state: SingleStateState::new(trace_stride, rng),
            problem,
            cooling: options.cooling,
            max_iters: options.max_iterations,
            stall_limit: options.stall_limit.unwrap_or(0),
            verbose: options.verbose.unwrap_or(false),
            record_trace: options.record_trace.unwrap_or(false),
            stall_count: 0,
            prev_best: f64::INFINITY,
            trace: Vec::new(),
            temperature_history: Vec::new(),
            current_t: Cell::new(0.0),
            current_cand_cost: Cell::new(0.0),
            current_accept_prob: Cell::new(1.0),
            current_accepted: Cell::new(false),
        };
        if !defer_bootstrap {
            opt.bootstrap();
        }
        // TS seeds temperatureHistory with the iteration-0 temperature.
        opt.temperature_history
            .push(temperature_at(&opt.cooling, 0));

        // Intrinsic invariant: best-so-far is monotone non-increasing.
        opt.add_validator(
            monotonicity_validator::<dyn DESStation>(
                "sa.bestHistory.monotone",
                |s: &dyn DESStation| downcast_sa::<S>(s).opt_state().best_history.clone(),
                Monotonicity::NonIncreasing,
                1e-9,
                Some("simulated-annealing-intrinsic".to_string()),
            )
            .boxed(),
        );
        opt.add_validator(
            intrinsic_check::<dyn DESStation>(
                "sa.acceptedCount-le-iterations",
                |s: &dyn DESStation| {
                    let st = downcast_sa::<S>(s);
                    st.get_accepted_count() <= st.get_iteration()
                },
                Some("accepted ≤ iter".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    let st = downcast_sa::<S>(s);
                    format!(
                        "accepted={}  iter={}",
                        st.get_accepted_count(),
                        st.get_iteration()
                    )
                })),
                Some("simulated-annealing-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        opt
    }

    fn record_tick_event(&mut self, k: usize) {
        let stride = self.opt_state().trace_stride;
        if stride != 0 && k.is_multiple_of(stride) {
            self.temperature_history.push(self.current_t.get());
        }
        if self.record_trace {
            let cur = self.opt_state().current_cost;
            let best = self.opt_state().best_cost;
            self.trace.push(SATickEvent {
                k,
                t: self.current_t.get(),
                current_cost: cur,
                candidate_cost: self.current_cand_cost.get(),
                delta: self.current_cand_cost.get() - cur,
                accept: self.current_accepted.get(),
                accept_prob: self.current_accept_prob.get(),
                best_cost: best,
            });
        }
        if self.verbose {
            let cur = self.opt_state().current_cost;
            eprintln!(
                "SA  k={:>6}  T={:e}  cur={:.4}  cand={:.4}  Δ={:.4}  p={:.3}  {}  best={:.4}",
                k,
                self.current_t.get(),
                cur,
                self.current_cand_cost.get(),
                self.current_cand_cost.get() - cur,
                self.current_accept_prob.get(),
                if self.current_accepted.get() {
                    "ACC"
                } else {
                    "rej"
                },
                self.opt_state().best_cost,
            );
        }
    }
}

impl<S: Clone + 'static> DESStation for SAOptimizer<S> {
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

impl<S: Clone + 'static> SingleStateOptimizer<S> for SAOptimizer<S> {
    fn opt_state(&self) -> &SingleStateState<S> {
        &self.state
    }
    fn opt_state_mut(&mut self) -> &mut SingleStateState<S> {
        &mut self.state
    }

    fn initial_state(&self, rng: &mut dyn RandomSource) -> S {
        self.problem.initial(rng)
    }
    fn cost(&self, s: &S) -> f64 {
        self.problem.cost(s)
    }
    fn propose(&self, s: &S, rng: &mut dyn RandomSource) -> S {
        self.problem.neighbour(s, rng)
    }
    fn clone_state(&self, s: &S) -> S {
        self.problem.clone_state(s)
    }

    fn accept(
        &self,
        _current: &S,
        _candidate: &S,
        current_cost: f64,
        candidate_cost: f64,
        iter: usize,
        rng: &mut dyn RandomSource,
    ) -> bool {
        let t = temperature_at(&self.cooling, iter);
        self.current_t.set(t);
        self.current_cand_cost.set(candidate_cost);
        if t <= 0.0 {
            self.current_accept_prob.set(0.0);
            self.current_accepted.set(false);
            return false;
        }
        let delta = candidate_cost - current_cost;
        if delta <= 0.0 {
            self.current_accept_prob.set(1.0);
            self.current_accepted.set(true);
            return true;
        }
        let p = (-delta / t).exp();
        self.current_accept_prob.set(p);
        let accepted = rng.next_float() < p;
        self.current_accepted.set(accepted);
        accepted
    }

    fn should_stop(&self, iter: usize) -> bool {
        if iter >= self.max_iters {
            return true;
        }
        if self.stall_limit > 0 && self.stall_count >= self.stall_limit {
            return true;
        }
        if iter > 0 && temperature_at(&self.cooling, iter) <= 0.0 {
            return true;
        }
        false
    }

    fn on_accept(&mut self, _candidate: &S, _delta: f64, iter: usize) {
        let best_cost = self.opt_state().best_cost;
        if best_cost < self.prev_best {
            self.prev_best = best_cost;
            self.stall_count = 0;
        } else {
            self.stall_count += 1;
        }
        self.record_tick_event(iter);
    }

    fn on_reject(&mut self, _candidate: &S, _delta: f64, iter: usize) {
        self.stall_count += 1;
        self.record_tick_event(iter);
    }

    fn on_bootstrap(&mut self) {
        self.prev_best = self.opt_state().best_cost;
    }
}

// =============================================================================
// MAIN SOLVER
// =============================================================================

/// Run simulated annealing on a generic problem. Orchestrated by an
/// [`SAOptimizer`] running on `run_iterative_des` — each tick is one proposal +
/// Metropolis accept/reject. (TS `runSimulatedAnnealing`.)
pub fn run_simulated_annealing<S: Clone + 'static>(
    problem: Rc<dyn SAProblem<S>>,
    options: SASolverOptions,
) -> SAResult<S> {
    let seed = options.seed.unwrap_or(42);
    let rng = SharedRng::new(seed);

    let p_src = problem.clone();
    let mut src_rng = rng.clone();
    let p_val = problem.clone();
    let source = Rc::new(RefCell::new(SingleStateSourceStation::<S>::with_validator(
        "simulated-annealing-source",
        move || p_src.initial(&mut src_rng),
        move |state: &S| {
            let initial_cost = p_val.cost(state);
            Preconditions::finite("simulated-annealing-source", "initialCost", initial_cost)
                .unwrap_or_else(|e| panic!("{e}"));
        },
    )));

    let opt = Rc::new(RefCell::new(SAOptimizer::<S>::new(
        problem.clone(),
        options,
        true,
        Some(Box::new(rng.clone()) as Box<dyn RandomSource>),
    )));
    let sink = Rc::new(RefCell::new(SingleStateSinkStation::<S>::new(
        "simulated-annealing-sink",
    )));

    source.borrow_mut().core_mut().pipe(
        opt.clone() as StationRef,
        SINGLE_STATE_INITIAL_CHANNEL,
        SINGLE_STATE_INITIAL_CHANNEL,
    );
    opt.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        SINGLE_STATE_RESULT_CHANNEL,
        SINGLE_STATE_RESULT_CHANNEL,
    );

    run_iterative_des(
        vec![
            source as StationRef,
            opt.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            shuffle: false,
            ..Default::default()
        },
    );

    let latest = sink.borrow().latest.clone().unwrap_or_else(|| {
        panic!("simulated-annealing: result sink did not receive a final state")
    });
    let snapshot = latest.snapshot.clone();

    let opt_ref = opt.borrow();
    // SingleStateOptimizer::bootstrap pushes one initial entry; legacy semantics
    // record exactly `iterations` entries — drop the bootstrap entry to match.
    let best_history_full = opt_ref.opt_state().best_history.clone();
    let current_history_full = opt_ref.opt_state().current_history.clone();
    let temp_full = opt_ref.temperature_history.clone();
    let best_history = if best_history_full.len() > 1 {
        best_history_full[1..].to_vec()
    } else {
        Vec::new()
    };
    let current_history = if current_history_full.len() > 1 {
        current_history_full[1..].to_vec()
    } else {
        Vec::new()
    };
    let t_cut = best_history_full
        .len()
        .saturating_sub(1)
        .min(temp_full.len());
    let temperature_history = temp_full[..t_cut].to_vec();
    let trace = if options.record_trace.unwrap_or(false) {
        Some(opt_ref.trace.clone())
    } else {
        None
    };

    SAResult {
        best_state: snapshot.best,
        best_cost: snapshot.best_cost,
        final_state: snapshot.current,
        final_cost: snapshot.current_cost,
        iterations: snapshot.iteration,
        accepted_count: snapshot.accepted_count,
        improve_count: snapshot.improve_count,
        trace,
        best_history,
        current_history,
        temperature_history,
    }
}

// =============================================================================
// TSP ADAPTER
// =============================================================================

/// Neighbour move family. (TS `'2-opt' | 'or-opt' | 'mixed'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SAMove {
    TwoOpt,
    OrOpt,
    Mixed,
}

/// Options for [`build_tsp_sa_problem`]. (TS inline options object.)
#[derive(Clone, Debug, Default)]
pub struct TSPSAProblemOptions {
    /// Penalty per violated precedence pair (added to cost).
    pub penalty_per_violation: Option<f64>,
    /// Initial-state heuristic.
    pub init: Option<InitMode>,
    /// Move set. Default `Mixed` (2-opt + or-opt).
    pub moves: Option<SAMove>,
}

/// SA problem over TSP tours using the 2-opt / or-opt move families.
pub struct TspSaProblem {
    instance: TSPInstance,
    penalty: f64,
    init: InitMode,
    moves: SAMove,
    n: usize,
}

/// Build an [`SAProblem`] for a TSP instance. (TS `buildTSPSAProblem`.)
pub fn build_tsp_sa_problem(instance: TSPInstance, opts: TSPSAProblemOptions) -> TspSaProblem {
    let n = instance.n;
    TspSaProblem {
        instance,
        penalty: opts.penalty_per_violation.unwrap_or(1e6),
        init: opts.init.unwrap_or(InitMode::NearestNeighbor),
        moves: opts.moves.unwrap_or(SAMove::Mixed),
        n,
    }
}

impl SAProblem<Tour> for TspSaProblem {
    fn cost(&self, tour: &Tour) -> f64 {
        let mut c = tour_length(&self.instance, tour);
        if let Some(precedence) = &self.instance.precedence {
            for &(a, b) in precedence {
                let mut pos_a: i64 = -1;
                let mut pos_b: i64 = -1;
                for (i, &city) in tour.iter().enumerate() {
                    if city == a {
                        pos_a = i as i64;
                    }
                    if city == b {
                        pos_b = i as i64;
                    }
                }
                if pos_a >= 0 && pos_b >= 0 && pos_a >= pos_b {
                    c += self.penalty;
                }
            }
        }
        c
    }

    fn neighbour(&self, tour: &Tour, rng: &mut dyn RandomSource) -> Tour {
        let n = self.n;
        let use_move = match self.moves {
            SAMove::Mixed => {
                if rng.next_float() < 0.7 {
                    SAMove::TwoOpt
                } else {
                    SAMove::OrOpt
                }
            }
            other => other,
        };
        if use_move == SAMove::TwoOpt {
            let mut i = (rng.next_float() * n as f64).floor() as usize;
            let mut j = (rng.next_float() * n as f64).floor() as usize;
            if i > j {
                std::mem::swap(&mut i, &mut j);
            }
            if j - i < 1 {
                j = (n - 1).min(i + 1);
            }
            let mut next = tour.clone();
            let mut a = i;
            let mut b = j;
            while a < b {
                next.swap(a, b);
                a += 1;
                b -= 1;
            }
            return next;
        }
        // or-opt: extract a sub-segment of length L ∈ {1,2,3} and reinsert.
        let l = 1 + (rng.next_float() * 3.0).floor() as usize;
        if l >= n {
            return tour.clone();
        }
        let i = (rng.next_float() * (n - l + 1) as f64).floor() as usize;
        let seg: Vec<usize> = tour[i..i + l].to_vec();
        let mut remaining: Vec<usize> = tour[..i].to_vec();
        remaining.extend_from_slice(&tour[i + l..]);
        let insert_at = (rng.next_float() * (remaining.len() + 1) as f64).floor() as usize;
        if insert_at == i {
            return tour.clone();
        }
        let mut out: Vec<usize> = remaining[..insert_at].to_vec();
        out.extend_from_slice(&seg);
        out.extend_from_slice(&remaining[insert_at..]);
        out
    }

    fn initial(&self, rng: &mut dyn RandomSource) -> Tour {
        let n = self.n;
        if self.init == InitMode::NearestNeighbor {
            let start = (rng.next_float() * n as f64).floor() as usize;
            let mut tour: Tour = vec![start];
            let mut visited = vec![false; n];
            visited[start] = true;
            let mut cur = start;
            while tour.len() < n {
                let mut best_next: i64 = -1;
                let mut best_d = f64::INFINITY;
                for j in 0..n {
                    if visited[j] {
                        continue;
                    }
                    let d = self.instance.distance[cur][j];
                    if d < best_d {
                        best_d = d;
                        best_next = j as i64;
                    }
                }
                let bn = best_next as usize;
                tour.push(bn);
                visited[bn] = true;
                cur = bn;
            }
            return tour;
        }
        let mut t: Tour = (0..n).collect();
        for i in (1..t.len()).rev() {
            let j = (rng.next_float() * (i as f64 + 1.0)).floor() as usize;
            t.swap(i, j);
        }
        t
    }
}

// =============================================================================
// KNAPSACK ADAPTER
// =============================================================================

/// A 0/1 knapsack instance. (TS `interface KnapsackInstance`.)
#[derive(Clone, Debug)]
pub struct KnapsackInstance {
    pub values: Vec<f64>,
    pub weights: Vec<f64>,
    pub capacity: f64,
}

/// SA problem over a 0/1 knapsack. State is a length-n 0/1 vector; neighbours
/// flip a single bit; cost is `−value` plus a steep over-capacity penalty.
pub struct KnapsackSaProblem {
    inst: KnapsackInstance,
    penalty: f64,
    n: usize,
}

/// Build an [`SAProblem`] for a 0/1 knapsack. (TS `buildKnapsackSAProblem`,
/// whose `penalty = 1e6` default the caller supplies explicitly here.)
pub fn build_knapsack_sa_problem(inst: KnapsackInstance, penalty: f64) -> KnapsackSaProblem {
    let n = inst.values.len();
    KnapsackSaProblem { inst, penalty, n }
}

impl SAProblem<Vec<f64>> for KnapsackSaProblem {
    fn cost(&self, x: &Vec<f64>) -> f64 {
        let mut v = 0.0;
        let mut w = 0.0;
        for i in 0..self.n {
            v += self.inst.values[i] * x[i];
            w += self.inst.weights[i] * x[i];
        }
        -v + self.penalty * (w - self.inst.capacity).max(0.0)
    }

    fn neighbour(&self, x: &Vec<f64>, rng: &mut dyn RandomSource) -> Vec<f64> {
        let j = (rng.next_float() * self.n as f64).floor() as usize;
        let mut next = x.clone();
        next[j] = 1.0 - next[j];
        next
    }

    fn initial(&self, _rng: &mut dyn RandomSource) -> Vec<f64> {
        // Greedy by value/weight ratio, capped at capacity.
        let mut order: Vec<usize> = (0..self.n).collect();
        order.sort_by(|&a, &b| {
            let ra = self.inst.values[b] / self.inst.weights[b];
            let rb = self.inst.values[a] / self.inst.weights[a];
            ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut x = vec![0.0; self.n];
        let mut w = 0.0;
        for i in order {
            if w + self.inst.weights[i] <= self.inst.capacity {
                x[i] = 1.0;
                w += self.inst.weights[i];
            }
        }
        x
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! SA smoke tests with fixed seeds. The 1-D Rastrigin surface is multimodal
    //! (a global minimum at the origin surrounded by many integer-spaced local
    //! minima); Metropolis acceptance escapes the local basins and settles in a
    //! low-cost basin. A tiny 0/1 knapsack is solved to its known optimum, and
    //! the temperature schedules are checked against their closed forms.

    use super::*;

    /// 1-D Rastrigin: f(x) = x^2 - 10 cos(2 pi x) + 10. Global minimum f(0) = 0.
    struct Rastrigin1D {
        start: f64,
        step: f64,
    }

    impl SAProblem<f64> for Rastrigin1D {
        fn cost(&self, x: &f64) -> f64 {
            x * x - 10.0 * (2.0 * std::f64::consts::PI * x).cos() + 10.0
        }
        fn neighbour(&self, x: &f64, rng: &mut dyn RandomSource) -> f64 {
            x + (rng.next_float() - 0.5) * 2.0 * self.step
        }
        fn initial(&self, _rng: &mut dyn RandomSource) -> f64 {
            self.start
        }
    }

    fn opts(seed: u32) -> SASolverOptions {
        SASolverOptions {
            max_iterations: 6000,
            cooling: CoolingSchedule::Geometric {
                t0: 10.0,
                alpha: 0.998,
                t_min: Some(1e-4),
            },
            seed: Some(seed),
            stall_limit: None,
            verbose: None,
            record_trace: None,
            trace_stride: None,
        }
    }

    #[test]
    fn temperature_schedules() {
        let geo = CoolingSchedule::Geometric {
            t0: 10.0,
            alpha: 0.9,
            t_min: None,
        };
        assert!((temperature_at(&geo, 0) - 10.0).abs() < 1e-12);
        assert!((temperature_at(&geo, 1) - 9.0).abs() < 1e-12);

        let lin = CoolingSchedule::Linear {
            t0: 10.0,
            rate: 2.0,
            t_min: Some(1.0),
        };
        assert!((temperature_at(&lin, 3) - 4.0).abs() < 1e-12);
        assert!((temperature_at(&lin, 100) - 1.0).abs() < 1e-12);

        let restart = CoolingSchedule::ExpRestart {
            t0: 8.0,
            alpha: 0.5,
            period: 4,
            t_min: None,
        };
        assert!((temperature_at(&restart, 4) - 8.0).abs() < 1e-12);
        assert!((temperature_at(&restart, 5) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn minimises_multimodal_function() {
        let problem: Rc<dyn SAProblem<f64>> = Rc::new(Rastrigin1D {
            start: 3.7,
            step: 0.5,
        });
        let result = run_simulated_annealing(problem, opts(12345));
        // Settles into a low-cost integer basin (global f(0)=0, neighbours f(±1)≈1).
        assert!(result.best_cost < 2.0, "best_cost = {}", result.best_cost);
        assert!(result.best_cost <= result.final_cost + 1e-9 || result.best_cost >= 0.0);
        assert_eq!(result.iterations, 6000);
        // best_history is monotone non-increasing.
        for w in result.best_history.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "best_history not monotone: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn solves_tiny_knapsack() {
        // items (value, weight): (60,10),(100,20),(120,30); capacity 50.
        // Optimal 0/1 choice is {1,2}: value 220 (weight 50) → cost -220.
        let inst = KnapsackInstance {
            values: vec![60.0, 100.0, 120.0],
            weights: vec![10.0, 20.0, 30.0],
            capacity: 50.0,
        };
        let problem: Rc<dyn SAProblem<Vec<f64>>> =
            Rc::new(build_knapsack_sa_problem(inst.clone(), 1e6));
        // The knapsack's energy scale (values 100-220) is ~10x the Rastrigin's,
        // so SA needs a proportionally hotter start to accept the ~60-worse
        // intermediate state required to escape the greedy basin {0,1} -> {1,2}.
        let knapsack_opts = SASolverOptions {
            max_iterations: 8000,
            cooling: CoolingSchedule::Geometric {
                t0: 100.0,
                alpha: 0.999,
                t_min: Some(1e-4),
            },
            seed: Some(7),
            stall_limit: None,
            verbose: None,
            record_trace: None,
            trace_stride: None,
        };
        let result = run_simulated_annealing(problem, knapsack_opts);

        // Found a strictly better solution than the greedy start (value 160).
        assert!(
            result.best_cost <= -200.0,
            "best_cost = {}",
            result.best_cost
        );
        // The reported best is feasible (within capacity).
        let w: f64 = (0..3).map(|i| inst.weights[i] * result.best_state[i]).sum();
        assert!(w <= inst.capacity + 1e-9, "best weight {w} over capacity");
    }
}
