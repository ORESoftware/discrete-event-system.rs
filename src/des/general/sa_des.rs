//! Port of `src/des/general/sa-des.ts` — Simulated Annealing (and hill
//! climbing) over TSP tours as a DES, built on the [`SingleStateOptimizer`]
//! template-method base. The concrete leaf ([`TSPSAOptimizer`]) implements ONLY
//! the hooks; the iteration loop is the base's provided template method
//! (`optimizer_step`).
//!
//! ## TS → Rust mapping
//!
//!   * `type CoolingSchedule = {kind:'geometric'|…}` (discriminated union) →
//!     [`CoolingSchedule`] enum with per-variant fields, matched in
//!     [`temperature_at`].
//!   * `interface TSPSAOptions` / `interface SADESResult` → structs
//!     (`number` → `f64`/`usize`, optionals → `Option<T>`).
//!   * `class TSPSAOptimizer extends SingleStateOptimizer<Tour>` → a struct
//!     `{ core: StationCore, state: SingleStateState<Tour>, … }` that
//!     `impl DESStation` (delegating `run_time_step` → `optimizer_step`,
//!     `has_work` → `optimizer_has_work`) and `impl SingleStateOptimizer<Tour>`.
//!   * INHERITANCE: `class TSPHillClimber extends TSPSAOptimizer` (override only
//!     `accept`). Rust has no inheritance, so per the migration header we
//!     COMPOSE: an [`AcceptRule`] field selects Metropolis vs. strict-improvement
//!     acceptance inside the single `accept` hook. `TSPHillClimber` is a public
//!     type ALIAS for `TSPSAOptimizer`; hill-climber instances are built via
//!     [`TSPSAOptimizer::hill_climber`]. (Flagged design deviation.)
//!   * `init`/`moves` string-unions → reuse [`InitMode`] from `genetic_tsp` and
//!     the local [`Moves`] enum.
//!   * `mulberry32(seed)` closures → boxed [`RandomSource`] in the base state,
//!     threaded as `&mut dyn RandomSource`.
//!   * `throw` (empty result sink, non-permutation best) → `panic!`.
//!
//! ## Adapter (local, flagged)
//!
//!   * [`SharedRng`] — `Rc<RefCell<SeededRandom>>` newtype so the public drivers
//!     can share ONE RNG stream between the source station's initial-state
//!     generator and the optimizer (the TS code shared a single `mulberry32`
//!     closure between them).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::single_state_optimizer::{
    SingleStateOptimizer, SingleStateSinkStation, SingleStateSourceStation, SingleStateState,
    SINGLE_STATE_INITIAL_CHANNEL, SINGLE_STATE_RESULT_CHANNEL,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::{intrinsic_check, monotonicity_validator, Monotonicity};
use crate::des::general::genetic_tsp::{held_karp_exact, is_permutation, tour_length, InitMode, TSPInstance, Tour};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// =============================================================================
// RNG adapter (see module docs)
// =============================================================================

/// A clonable handle to a single shared `SeededRandom` stream.
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
// COOLING SCHEDULES
// =============================================================================

/// Temperature schedule. (TS discriminated union on `kind`.) `Tmin` defaults to
/// `0` when absent.
#[derive(Clone, Copy, Debug)]
pub enum CoolingSchedule {
    Geometric { t0: f64, alpha: f64, t_min: Option<f64> },
    Logarithmic { t0: f64, t_min: Option<f64> },
    Linear { t0: f64, rate: f64, t_min: Option<f64> },
    ExpRestart { t0: f64, alpha: f64, period: usize, t_min: Option<f64> },
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
        CoolingSchedule::ExpRestart { t0, alpha, period, t_min } => {
            t_min.unwrap_or(0.0).max(t0 * alpha.powf((k % period) as f64))
        }
    }
}

// =============================================================================
// Options
// =============================================================================

/// Neighbour-move kind. (TS `'2-opt' | 'or-opt' | 'mixed'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Moves {
    TwoOpt,
    OrOpt,
    Mixed,
}

/// Which acceptance rule the optimizer uses. SA = Metropolis; the
/// hill-climbing leaf overrides this to strict improvement only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AcceptRule {
    Metropolis,
    HillClimb,
}

/// SA configuration. (TS `interface TSPSAOptions`.)
#[derive(Clone, Copy, Debug)]
pub struct TSPSAOptions {
    pub cooling: CoolingSchedule,
    pub max_iterations: usize,
    pub seed: u32,
    pub init: Option<InitMode>,
    pub moves: Option<Moves>,
    /// Cost penalty per violated precedence pair.
    pub penalty_per_violation: Option<f64>,
    pub trace_stride: Option<usize>,
    /// Stop after this many iterations without best improvement. 0 = off.
    pub stall_limit: Option<usize>,
}

// =============================================================================
// TSPSAOptimizer — SingleStateOptimizer<Tour> leaf
// =============================================================================

/// SA-for-TSP concrete optimizer (also realises the hill climber via
/// [`AcceptRule::HillClimb`]). Implements the [`SingleStateOptimizer`] hook
/// trait; the iteration loop lives in the base's template method.
pub struct TSPSAOptimizer {
    core: StationCore,
    state: SingleStateState<Tour>,
    inst: TSPInstance,
    cooling: CoolingSchedule,
    max_iters: usize,
    init_mode: InitMode,
    moves: Moves,
    penalty: f64,
    stall_limit: usize,
    stall_since: usize,
    best_seen: f64,
    accept_rule: AcceptRule,
}

/// Hill-climbing leaf. (TS `class TSPHillClimber extends TSPSAOptimizer`.) Built
/// via [`TSPSAOptimizer::hill_climber`]; behaviourally identical to
/// `TSPSAOptimizer` except for [`AcceptRule::HillClimb`].
pub type TSPHillClimber = TSPSAOptimizer;

fn downcast_sa(s: &dyn DESStation) -> &TSPSAOptimizer {
    s.as_any().downcast_ref::<TSPSAOptimizer>().expect("validator received a non-TSPSAOptimizer station")
}

impl TSPSAOptimizer {
    /// Construct the SA optimizer (Metropolis acceptance). `defer_bootstrap`
    /// mirrors the TS lifecycle flag; `rng` injects a shared stream.
    pub fn new(
        id: impl Into<String>,
        inst: TSPInstance,
        opts: TSPSAOptions,
        defer_bootstrap: bool,
        rng: Option<Box<dyn RandomSource>>,
    ) -> Self {
        Self::with_rule(id, inst, opts, defer_bootstrap, rng, AcceptRule::Metropolis)
    }

    /// Construct the hill-climbing leaf (strict-improvement acceptance).
    pub fn hill_climber(
        id: impl Into<String>,
        inst: TSPInstance,
        opts: TSPSAOptions,
        defer_bootstrap: bool,
        rng: Option<Box<dyn RandomSource>>,
    ) -> Self {
        Self::with_rule(id, inst, opts, defer_bootstrap, rng, AcceptRule::HillClimb)
    }

    fn with_rule(
        id: impl Into<String>,
        inst: TSPInstance,
        opts: TSPSAOptions,
        defer_bootstrap: bool,
        rng: Option<Box<dyn RandomSource>>,
        accept_rule: AcceptRule,
    ) -> Self {
        let rng: Box<dyn RandomSource> =
            rng.unwrap_or_else(|| Box::new(mulberry32(opts.seed)) as Box<dyn RandomSource>);
        let mut opt = TSPSAOptimizer {
            core: StationCore::new(id),
            state: SingleStateState::new(opts.trace_stride.unwrap_or(1), rng),
            inst,
            cooling: opts.cooling,
            max_iters: opts.max_iterations,
            init_mode: opts.init.unwrap_or(InitMode::NearestNeighbor),
            moves: opts.moves.unwrap_or(Moves::Mixed),
            penalty: opts.penalty_per_violation.unwrap_or(1e6),
            stall_limit: opts.stall_limit.unwrap_or(0),
            stall_since: 0,
            best_seen: f64::INFINITY,
            accept_rule,
        };
        if !defer_bootstrap {
            opt.bootstrap();
        }

        // ── Intrinsic invariants ─────────────────────────────────────────────
        // Best-so-far history is monotone non-increasing by definition of SA.
        opt.add_validator(
            monotonicity_validator::<dyn DESStation>(
                "sa.bestHistory.monotone",
                |s: &dyn DESStation| downcast_sa(s).opt_state().best_history.clone(),
                Monotonicity::NonIncreasing,
                1e-9,
                Some("sa-intrinsic".to_string()),
            )
            .boxed(),
        );
        // Best is a valid permutation of n cities.
        opt.add_validator(
            intrinsic_check::<dyn DESStation>(
                "sa.best-is-valid-permutation",
                |s: &dyn DESStation| {
                    let st = downcast_sa(s);
                    is_permutation(st.get_best(), st.inst.n)
                },
                Some("permutation of [0..n-1]".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    let st = downcast_sa(s);
                    format!("n={}  bestLen={}", st.inst.n, st.get_best().len())
                })),
                Some("sa-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        // bestCost ≥ 0 for any valid Euclidean TSP.
        opt.add_validator(
            intrinsic_check::<dyn DESStation>(
                "sa.best-cost-nonnegative",
                |s: &dyn DESStation| downcast_sa(s).get_best_cost() >= 0.0,
                Some("≥ 0".to_string()),
                Some(Box::new(|s: &dyn DESStation| format!("bestCost={}", downcast_sa(s).get_best_cost()))),
                Some("sa-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );

        // ── Ground-truth: Held–Karp exact lower bound (small instances only) ─
        if opt.inst.n <= 12 && opt.inst.precedence.is_none() {
            let exact = Rc::new(RefCell::new(None::<f64>));
            let e1 = exact.clone();
            let e2 = exact.clone();
            opt.add_validator(
                intrinsic_check::<dyn DESStation>(
                    "sa.bestCost-vs-heldKarp-LB",
                    move |s: &dyn DESStation| {
                        let st = downcast_sa(s);
                        let mut cache = e1.borrow_mut();
                        if cache.is_none() {
                            *cache = Some(held_karp_exact(&st.inst).length);
                        }
                        st.get_best_cost() >= cache.unwrap() - 1e-9
                    },
                    Some("bestCost ≥ heldKarp.length".to_string()),
                    Some(Box::new(move |s: &dyn DESStation| {
                        let st = downcast_sa(s);
                        let mut cache = e2.borrow_mut();
                        if cache.is_none() {
                            *cache = Some(held_karp_exact(&st.inst).length);
                        }
                        format!("bestCost={:.4}  heldKarp={:.4}", st.get_best_cost(), cache.unwrap())
                    })),
                    Some("sa-ground-truth".to_string()),
                    Some("bestCost is below the true global optimum — would indicate a bug".to_string()),
                )
                .boxed(),
            );
        }

        opt
    }
}

impl DESStation for TSPSAOptimizer {
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

impl SingleStateOptimizer<Tour> for TSPSAOptimizer {
    fn opt_state(&self) -> &SingleStateState<Tour> {
        &self.state
    }
    fn opt_state_mut(&mut self) -> &mut SingleStateState<Tour> {
        &mut self.state
    }

    fn initial_state(&self, rng: &mut dyn RandomSource) -> Tour {
        initial_tour(&self.inst, self.init_mode, rng)
    }

    fn cost(&self, tour: &Tour) -> f64 {
        let mut c = tour_length(&self.inst, tour);
        if let Some(precedence) = &self.inst.precedence {
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

    fn propose(&self, tour: &Tour, rng: &mut dyn RandomSource) -> Tour {
        let mv = match self.moves {
            Moves::Mixed => {
                if rng.next_float() < 0.7 {
                    Moves::TwoOpt
                } else {
                    Moves::OrOpt
                }
            }
            other => other,
        };
        let n = tour.len();
        if mv == Moves::TwoOpt {
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
        let mut remain: Vec<usize> = tour[..i].to_vec();
        remain.extend_from_slice(&tour[i + l..]);
        let insert_at = (rng.next_float() * (remain.len() + 1) as f64).floor() as usize;
        if insert_at == i {
            return tour.clone();
        }
        let mut out: Vec<usize> = remain[..insert_at].to_vec();
        out.extend_from_slice(&seg);
        out.extend_from_slice(&remain[insert_at..]);
        out
    }

    /// Metropolis acceptance (SA) or strict-improvement (hill climbing),
    /// selected by [`AcceptRule`].
    fn accept(
        &self,
        _current: &Tour,
        _candidate: &Tour,
        current_cost: f64,
        candidate_cost: f64,
        iter: usize,
        rng: &mut dyn RandomSource,
    ) -> bool {
        match self.accept_rule {
            AcceptRule::HillClimb => candidate_cost < current_cost,
            AcceptRule::Metropolis => {
                let delta = candidate_cost - current_cost;
                if delta <= 0.0 {
                    return true;
                }
                let t = temperature_at(&self.cooling, iter);
                if t <= 0.0 {
                    return false;
                }
                rng.next_float() < (-delta / t).exp()
            }
        }
    }

    fn should_stop(&self, iter: usize) -> bool {
        if iter >= self.max_iters {
            return true;
        }
        if self.stall_limit > 0 && self.stall_since >= self.stall_limit {
            return true;
        }
        false
    }

    // ── optional overrides — track stall ─────────────────────────────────────

    fn on_accept(&mut self, _candidate: &Tour, _delta: f64, _iter: usize) {
        let best_cost = self.opt_state().best_cost;
        if best_cost < self.best_seen {
            self.best_seen = best_cost;
            self.stall_since = 0;
        } else {
            self.stall_since += 1;
        }
    }

    fn on_reject(&mut self, _candidate: &Tour, _delta: f64, _iter: usize) {
        self.stall_since += 1;
    }

    fn on_bootstrap(&mut self) {
        self.best_seen = self.opt_state().best_cost;
    }
}

// =============================================================================
// PUBLIC DRIVERS
// =============================================================================

/// Final result of an SA/hill-climber DES run. (TS `interface SADESResult`.)
#[derive(Clone, Debug)]
pub struct SADESResult {
    pub best_tour: Tour,
    pub best_cost: f64,
    pub iterations: usize,
    pub accepted_count: usize,
    pub improve_count: usize,
    pub best_history: Vec<f64>,
    pub current_history: Vec<f64>,
    pub ticks: usize,
}

/// Run SA over TSP tours. (TS `runTSPSADES`.)
pub fn run_tsp_sa_des(
    inst: TSPInstance,
    opts: TSPSAOptions,
    des_options: Option<IterativeRunOptions>,
) -> SADESResult {
    run_single_state_des(inst, opts, AcceptRule::Metropolis, "sa", des_options)
}

/// Run hill climbing over TSP tours. (TS `runTSPHillClimberDES`.)
pub fn run_tsp_hill_climber_des(
    inst: TSPInstance,
    opts: TSPSAOptions,
    des_options: Option<IterativeRunOptions>,
) -> SADESResult {
    run_single_state_des(inst, opts, AcceptRule::HillClimb, "hc", des_options)
}

/// Shared driver for both leaves (the two TS functions differ only in the
/// optimizer leaf and station ids).
fn run_single_state_des(
    inst: TSPInstance,
    opts: TSPSAOptions,
    accept_rule: AcceptRule,
    prefix: &str,
    des_options: Option<IterativeRunOptions>,
) -> SADESResult {
    let rng = SharedRng::new(opts.seed);
    let init_mode = opts.init.unwrap_or(InitMode::NearestNeighbor);

    let source_id = format!("{prefix}-source");
    let validate_id = source_id.clone();
    let inst_init = inst.clone();
    let inst_val = inst.clone();
    let mut src_rng = rng.clone();
    let source = Rc::new(RefCell::new(SingleStateSourceStation::<Tour>::with_validator(
        source_id,
        move || initial_tour(&inst_init, init_mode, &mut src_rng),
        move |tour: &Tour| validate_initial_tour(&validate_id, &inst_val, tour),
    )));

    let opt = Rc::new(RefCell::new(TSPSAOptimizer::with_rule(
        prefix.to_string(),
        inst.clone(),
        opts,
        true,
        Some(Box::new(rng.clone()) as Box<dyn RandomSource>),
        accept_rule,
    )));
    let sink = Rc::new(RefCell::new(SingleStateSinkStation::<Tour>::new(format!("{prefix}-sink"))));

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

    // TS forces `shuffle` off here (overriding the runner's default of `true`).
    let run_opts = des_options.unwrap_or(IterativeRunOptions { shuffle: false, ..Default::default() });
    let summary = run_iterative_des(
        vec![source as StationRef, opt.clone() as StationRef, sink.clone() as StationRef],
        run_opts,
    );

    let latest = sink
        .borrow()
        .latest
        .clone()
        .unwrap_or_else(|| panic!("{prefix}-des: result sink did not receive a final tour"));
    let snapshot = &latest.snapshot;
    let best = snapshot.best.clone();
    if !is_permutation(&best, inst.n) {
        panic!("{prefix}-des: best is not a valid permutation");
    }
    let opt_ref = opt.borrow();
    SADESResult {
        best_tour: best,
        best_cost: snapshot.best_cost,
        iterations: snapshot.iteration,
        accepted_count: snapshot.accepted_count,
        improve_count: snapshot.improve_count,
        best_history: opt_ref.opt_state().best_history.clone(),
        current_history: opt_ref.opt_state().current_history.clone(),
        ticks: summary.ticks,
    }
}

// =============================================================================
// Free helpers
// =============================================================================

fn initial_tour(inst: &TSPInstance, init_mode: InitMode, rng: &mut dyn RandomSource) -> Tour {
    let n = inst.n;
    if init_mode == InitMode::NearestNeighbor {
        let start = (rng.next_float() * n as f64).floor() as usize;
        let mut tour: Tour = vec![start];
        let mut seen = vec![false; n];
        seen[start] = true;
        let mut cur = start;
        while tour.len() < n {
            let mut best_next: i64 = -1;
            let mut best_d = f64::INFINITY;
            for j in 0..n {
                if seen[j] {
                    continue;
                }
                let d = inst.distance[cur][j];
                if d < best_d {
                    best_d = d;
                    best_next = j as i64;
                }
            }
            let bn = best_next as usize;
            tour.push(bn);
            seen[bn] = true;
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

fn validate_initial_tour(source_id: &str, inst: &TSPInstance, tour: &Tour) {
    Preconditions::check(
        source_id,
        "initial tour",
        &format!("be a permutation of {} cities", inst.n),
        is_permutation(tour, inst.n),
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    Preconditions::finite(source_id, "initial tour length", tour_length(inst, tour))
        .unwrap_or_else(|e| panic!("{e}"));
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! SA-DES smoke tests over tiny TSP instances with fixed seeds. The pentagon
    //! cost surface is multimodal (many local optima from 2-opt / or-opt moves);
    //! Metropolis acceptance escapes them and settles near the Held–Karp global
    //! optimum, while the hill climber only ever improves on its start.

    use super::*;
    use crate::des::general::genetic_tsp::build_pentagon_tsp;

    fn cooling() -> CoolingSchedule {
        CoolingSchedule::Geometric { t0: 50.0, alpha: 0.995, t_min: Some(1e-3) }
    }

    fn opts(seed: u32) -> TSPSAOptions {
        TSPSAOptions {
            cooling: cooling(),
            max_iterations: 4000,
            seed,
            init: None,
            moves: None,
            penalty_per_violation: None,
            trace_stride: None,
            stall_limit: None,
        }
    }

    #[test]
    fn temperature_schedules() {
        let geo = CoolingSchedule::Geometric { t0: 10.0, alpha: 0.9, t_min: None };
        assert!((temperature_at(&geo, 0) - 10.0).abs() < 1e-12);
        assert!((temperature_at(&geo, 1) - 9.0).abs() < 1e-12);

        let log = CoolingSchedule::Logarithmic { t0: 10.0, t_min: None };
        assert!((temperature_at(&log, 0) - 10.0 / 2.0_f64.ln()).abs() < 1e-9);

        let lin = CoolingSchedule::Linear { t0: 10.0, rate: 2.0, t_min: Some(1.0) };
        assert!((temperature_at(&lin, 3) - 4.0).abs() < 1e-12);
        assert!((temperature_at(&lin, 100) - 1.0).abs() < 1e-12); // clamped at t_min

        let restart = CoolingSchedule::ExpRestart { t0: 8.0, alpha: 0.5, period: 4, t_min: None };
        assert!((temperature_at(&restart, 4) - 8.0).abs() < 1e-12); // 4 % 4 == 0
        assert!((temperature_at(&restart, 5) - 4.0).abs() < 1e-12); // 5 % 4 == 1
    }

    #[test]
    fn sa_settles_near_optimal() {
        let inst = build_pentagon_tsp(5, 50.0);
        let optimal = held_karp_exact(&inst).length;
        let result = run_tsp_sa_des(inst.clone(), opts(2024), None);

        assert!(is_permutation(&result.best_tour, inst.n), "best must be a permutation");
        assert!(result.best_cost >= optimal - 1e-9, "cannot beat the global optimum");
        assert!(
            result.best_cost <= optimal * 1.05,
            "SA best {} should be near optimal {}",
            result.best_cost,
            optimal
        );
        assert_eq!(result.iterations, 4000);
    }

    #[test]
    fn hill_climber_only_improves() {
        let inst = build_pentagon_tsp(6, 40.0);
        let result = run_tsp_hill_climber_des(inst.clone(), opts(11), None);

        assert!(is_permutation(&result.best_tour, inst.n));
        // best_history is monotone non-increasing and ends at the final best.
        for w in result.best_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "best_history not monotone: {} -> {}", w[0], w[1]);
        }
        let first = *result.best_history.first().unwrap();
        assert!(result.best_cost <= first + 1e-9, "hill climber must not worsen the start");
    }
}
