//! Port of `src/des/general/ga-des.ts` — Genetic Algorithm as a DES, built on
//! the [`PopulationOptimizer`] template-method base. The concrete leaf
//! ([`TSPGAOptimizer`]) implements ONLY the hooks; the per-generation breeding
//! loop is the base's provided template method (`generation_step`).
//!
//! ## TS → Rust mapping
//!
//!   * `interface TSPGAOptions` / `interface GADESResult` → plain structs
//!     (`number` → `f64`/`usize`, optionals → `Option<T>`).
//!   * `class TSPGAOptimizer extends PopulationOptimizer<Tour>` → a struct
//!     `{ core: StationCore, state: PopulationState<Tour>, … }` that
//!     `impl DESStation` (delegating `run_time_step` → `generation_step`,
//!     `has_work` → `optimizer_has_work`) and `impl PopulationOptimizer<Tour>`
//!     (the hook trait).
//!   * `init: 'random' | 'nearest-neighbor'` → reuse [`InitMode`] from
//!     `genetic_tsp` (the string-union maps onto its enum).
//!   * `mulberry32(seed)` RNG closures → a boxed [`RandomSource`] stored in the
//!     base state, threaded into the hooks as `&mut dyn RandomSource`.
//!   * `throw` (result-sink empty, non-permutation best) → `panic!` (invariant).
//!   * The intrinsic / ground-truth validators are registered through the ported
//!     [`intrinsic_check`] / [`monotonicity_validator`] factories.
//!
//! ## Adapters (local, flagged)
//!
//!   * [`DynRandom`] — newtype bridging the hooks' `&mut dyn RandomSource` to the
//!     `genetic_tsp` operators, which are generic over `&mut impl RandomSource`
//!     (a `dyn` value is unsized and cannot satisfy the implicit `Sized` bound).
//!   * [`SharedRng`] — an `Rc<RefCell<SeededRandom>>` newtype so the public
//!     driver can share ONE RNG stream between the source station's
//!     initial-population generator and the optimizer's breeding loop (the TS
//!     code shared a single `mulberry32` closure between them).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::population_optimizer::{
    PopulationOptimizer, PopulationSinkStation, PopulationSourceStation, PopulationState,
    POPULATION_INITIAL_CHANNEL, POPULATION_RESULT_CHANNEL,
};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::{intrinsic_check, monotonicity_validator, Monotonicity};
use crate::des::general::genetic_tsp::{
    held_karp_exact, inversion_mutate, is_permutation, order_crossover, swap_mutate, tour_length,
    tournament_select, InitMode, TSPInstance, Tour,
};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// =============================================================================
// RNG adapters (see module docs)
// =============================================================================

/// Bridges a `&mut dyn RandomSource` to the `genetic_tsp` operators, which take
/// `&mut impl RandomSource` (a sized generic).
struct DynRandom<'a>(&'a mut dyn RandomSource);

impl RandomSource for DynRandom<'_> {
    fn next_float(&mut self) -> f64 {
        self.0.next_float()
    }
}

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
// Options
// =============================================================================

/// GA configuration. (TS `interface TSPGAOptions`.)
#[derive(Clone, Copy, Debug)]
pub struct TSPGAOptions {
    pub pop_size: usize,
    pub num_generations: usize,
    pub tournament_size: Option<usize>,
    pub crossover_prob: Option<f64>,
    pub mutation_prob: Option<f64>,
    pub elitism: Option<usize>,
    pub seed: u32,
    pub init: Option<InitMode>,
    /// Cost penalty per violated precedence pair.
    pub penalty_per_violation: Option<f64>,
}

// =============================================================================
// TSPGAOptimizer — PopulationOptimizer<Tour> leaf
// =============================================================================

/// GA-for-TSP concrete optimizer. Implements the [`PopulationOptimizer`] hook
/// trait; the breeding loop lives in the base's template method.
pub struct TSPGAOptimizer {
    core: StationCore,
    state: PopulationState<Tour>,
    inst: TSPInstance,
    num_generations: usize,
    tournament_k: usize,
    crossover_prob: f64,
    mutation_prob: f64,
    elite: usize,
    init_mode: InitMode,
    penalty: f64,
}

fn downcast_ga(s: &dyn DESStation) -> &TSPGAOptimizer {
    s.as_any().downcast_ref::<TSPGAOptimizer>().expect("validator received a non-TSPGAOptimizer station")
}

impl TSPGAOptimizer {
    /// Construct the optimizer. `defer_bootstrap` mirrors the TS lifecycle flag
    /// (skip the in-constructor bootstrap so a source station can seed the
    /// population instead); `rng` lets the caller inject a shared stream.
    pub fn new(
        id: impl Into<String>,
        inst: TSPInstance,
        opts: TSPGAOptions,
        defer_bootstrap: bool,
        rng: Option<Box<dyn RandomSource>>,
    ) -> Self {
        let rng: Box<dyn RandomSource> =
            rng.unwrap_or_else(|| Box::new(mulberry32(opts.seed)) as Box<dyn RandomSource>);
        let elite = opts.elitism.unwrap_or(2).min(opts.pop_size);
        let mut opt = TSPGAOptimizer {
            core: StationCore::new(id),
            state: PopulationState::new(opts.pop_size, rng),
            inst,
            num_generations: opts.num_generations,
            tournament_k: opts.tournament_size.unwrap_or(3),
            crossover_prob: opts.crossover_prob.unwrap_or(0.95),
            mutation_prob: opts.mutation_prob.unwrap_or(0.3),
            elite,
            init_mode: opts.init.unwrap_or(InitMode::Random),
            penalty: opts.penalty_per_violation.unwrap_or(1e6),
        };
        if !defer_bootstrap {
            opt.bootstrap();
        }

        // ── Intrinsic invariants ─────────────────────────────────────────────
        // With elitism ≥ 1, best-so-far history is monotone non-increasing.
        if opt.elite >= 1 {
            opt.add_validator(
                monotonicity_validator::<dyn DESStation>(
                    "ga.bestHistory.monotone",
                    |s: &dyn DESStation| downcast_ga(s).opt_state().best_history.clone(),
                    Monotonicity::NonIncreasing,
                    1e-9,
                    Some("ga-intrinsic".to_string()),
                )
                .boxed(),
            );
        }
        // best is a valid permutation of n cities.
        opt.add_validator(
            intrinsic_check::<dyn DESStation>(
                "ga.best-is-valid-permutation",
                |s: &dyn DESStation| {
                    let st = downcast_ga(s);
                    is_permutation(st.get_best_tour(), st.inst.n)
                },
                Some("permutation of [0..n-1]".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    let st = downcast_ga(s);
                    format!("n={}  bestLen={}", st.inst.n, st.get_best_tour().len())
                })),
                Some("ga-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );

        // ── Ground-truth: Held–Karp lower bound (small instances only) ───────
        if opt.inst.n <= 12 && opt.inst.precedence.is_none() {
            let exact = Rc::new(RefCell::new(None::<f64>));
            let e1 = exact.clone();
            let e2 = exact.clone();
            opt.add_validator(
                intrinsic_check::<dyn DESStation>(
                    "ga.bestLength-vs-heldKarp-LB",
                    move |s: &dyn DESStation| {
                        let st = downcast_ga(s);
                        let mut cache = e1.borrow_mut();
                        if cache.is_none() {
                            *cache = Some(held_karp_exact(&st.inst).length);
                        }
                        st.get_best_length() >= cache.unwrap() - 1e-9
                    },
                    Some("bestLength ≥ heldKarp.length".to_string()),
                    Some(Box::new(move |s: &dyn DESStation| {
                        let st = downcast_ga(s);
                        let mut cache = e2.borrow_mut();
                        if cache.is_none() {
                            *cache = Some(held_karp_exact(&st.inst).length);
                        }
                        format!("best={:.4}  heldKarp={:.4}", st.get_best_length(), cache.unwrap())
                    })),
                    Some("ga-ground-truth".to_string()),
                    Some("best length is below the global optimum".to_string()),
                )
                .boxed(),
            );
        }

        opt
    }

    // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

    pub fn get_best_tour(&self) -> &Tour {
        self.get_best()
    }
    pub fn get_best_length(&self) -> f64 {
        self.get_best_fitness()
    }
}

impl DESStation for TSPGAOptimizer {
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

impl PopulationOptimizer<Tour> for TSPGAOptimizer {
    fn opt_state(&self) -> &PopulationState<Tour> {
        &self.state
    }
    fn opt_state_mut(&mut self) -> &mut PopulationState<Tour> {
        &mut self.state
    }

    fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<Tour> {
        initial_tour_population(&self.inst, size, self.init_mode, &mut DynRandom(rng))
    }

    fn evaluate(&self, tour: &Tour) -> f64 {
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

    /// Tournament selection of TWO parents (for one offspring).
    fn select(&self, pop: &[Tour], fitness: &[f64], rng: &mut dyn RandomSource) -> Vec<Tour> {
        let mut w = DynRandom(rng);
        let i = tournament_select(fitness, self.tournament_k, &mut w);
        let mut j = tournament_select(fitness, self.tournament_k, &mut w);
        let mut tries = 0;
        while j == i && tries < 8 {
            j = tournament_select(fitness, self.tournament_k, &mut w);
            tries += 1;
        }
        vec![pop[i].clone(), pop[j].clone()]
    }

    /// Order crossover with probability `crossover_prob`; otherwise clone p1.
    fn recombine(&self, parents: &[Tour], rng: &mut dyn RandomSource) -> Tour {
        if rng.next_float() < self.crossover_prob {
            order_crossover(&parents[0], &parents[1], &mut DynRandom(rng))
        } else {
            parents[0].clone()
        }
    }

    /// Apply mutation with `mutation_prob`; mix of inversion (preserves perm)
    /// and swap.
    fn mutate(&self, child: Tour, rng: &mut dyn RandomSource) -> Tour {
        if rng.next_float() < self.mutation_prob {
            if rng.next_float() < 0.6 {
                inversion_mutate(&child, &mut DynRandom(rng))
            } else {
                swap_mutate(&child, &mut DynRandom(rng))
            }
        } else {
            child
        }
    }

    fn should_stop(&self, generation: usize) -> bool {
        generation >= self.num_generations
    }

    fn elite_count(&self) -> usize {
        self.elite
    }
}

// =============================================================================
// PUBLIC DRIVER
// =============================================================================

/// Final result of a GA-DES run. (TS `interface GADESResult`.)
#[derive(Clone, Debug)]
pub struct GADESResult {
    pub best_tour: Tour,
    pub best_length: f64,
    pub generations: usize,
    pub best_history: Vec<f64>,
    pub mean_history: Vec<f64>,
    pub ticks: usize,
}

/// Wire up `source → optimizer → sink`, run the iterative DES, and reduce the
/// terminal snapshot to a [`GADESResult`]. (TS `runTSPGADES`.)
pub fn run_tsp_ga_des(
    inst: TSPInstance,
    opts: TSPGAOptions,
    des_options: Option<IterativeRunOptions>,
) -> GADESResult {
    let rng = SharedRng::new(opts.seed);
    let init_mode = opts.init.unwrap_or(InitMode::Random);

    let inst_init = inst.clone();
    let inst_val = inst.clone();
    let mut src_rng = rng.clone();
    let source = Rc::new(RefCell::new(PopulationSourceStation::<Tour>::with_validator(
        "ga-source",
        move || initial_tour_population(&inst_init, opts.pop_size, init_mode, &mut src_rng),
        move |population: &[Tour]| {
            validate_initial_tour_population("ga-source", &inst_val, opts.pop_size, population)
        },
    )));

    let opt = Rc::new(RefCell::new(TSPGAOptimizer::new(
        "ga",
        inst.clone(),
        opts,
        true,
        Some(Box::new(rng.clone()) as Box<dyn RandomSource>),
    )));
    let sink = Rc::new(RefCell::new(PopulationSinkStation::<Tour>::new("ga-sink")));

    source.borrow_mut().core_mut().pipe(
        opt.clone() as StationRef,
        POPULATION_INITIAL_CHANNEL,
        POPULATION_INITIAL_CHANNEL,
    );
    opt.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        POPULATION_RESULT_CHANNEL,
        POPULATION_RESULT_CHANNEL,
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
        .expect("ga-des: result sink did not receive a final population");
    let snapshot = &latest.snapshot;
    let best = snapshot.best.clone();
    if !is_permutation(&best, inst.n) {
        panic!("ga-des: best is not a valid permutation");
    }
    let opt_ref = opt.borrow();
    GADESResult {
        best_tour: best,
        best_length: snapshot.best_fitness,
        generations: snapshot.generation,
        best_history: opt_ref.opt_state().best_history.clone(),
        mean_history: opt_ref.opt_state().mean_history.clone(),
        ticks: summary.ticks,
    }
}

// =============================================================================
// Free helpers
// =============================================================================

fn initial_tour_population(
    inst: &TSPInstance,
    size: usize,
    init_mode: InitMode,
    rng: &mut impl RandomSource,
) -> Vec<Tour> {
    let n = inst.n;
    let mut out: Vec<Tour> = Vec::new();
    if init_mode == InitMode::NearestNeighbor {
        let k = n.min(size);
        for s in 0..k {
            out.push(nearest_neighbor_tour(inst, s));
        }
        while out.len() < size {
            out.push(random_tour(n, rng));
        }
        return out;
    }
    for _ in 0..size {
        out.push(random_tour(n, rng));
    }
    out
}

fn nearest_neighbor_tour(inst: &TSPInstance, start: usize) -> Tour {
    let n = inst.n;
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
    tour
}

fn random_tour(n: usize, rng: &mut impl RandomSource) -> Tour {
    let mut t: Tour = (0..n).collect();
    for i in (1..t.len()).rev() {
        let j = (rng.next_float() * (i as f64 + 1.0)).floor() as usize;
        t.swap(i, j);
    }
    t
}

fn validate_initial_tour_population(
    source_id: &str,
    inst: &TSPInstance,
    expected_size: usize,
    population: &[Tour],
) {
    Preconditions::length_eq(source_id, "initial population", population, expected_size)
        .unwrap_or_else(|e| panic!("{e}"));
    for (i, tour) in population.iter().enumerate() {
        Preconditions::check(
            source_id,
            &format!("initial population[{i}]"),
            &format!("be a permutation of {} cities", inst.n),
            is_permutation(tour, inst.n),
            None,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        Preconditions::finite(source_id, &format!("initial population[{i}] length"), tour_length(inst, tour))
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! GA-DES smoke tests over tiny TSP instances with fixed seeds. The pentagon
    //! instance has a known global optimum (visit the vertices in polygon
    //! order), recovered exactly by Held–Karp; we assert the GA reaches it
    //! (within a small slack) and that the best-cost history is monotone.

    use super::*;
    use crate::des::general::genetic_tsp::build_pentagon_tsp;

    fn opts(seed: u32) -> TSPGAOptions {
        TSPGAOptions {
            pop_size: 50,
            num_generations: 120,
            tournament_size: None,
            crossover_prob: None,
            mutation_prob: None,
            elitism: Some(2),
            seed,
            init: None,
            penalty_per_violation: None,
        }
    }

    #[test]
    fn solves_small_tsp_near_optimal() {
        let inst = build_pentagon_tsp(5, 50.0);
        let optimal = held_karp_exact(&inst).length;
        let result = run_tsp_ga_des(inst.clone(), opts(12_345), None);

        assert!(is_permutation(&result.best_tour, inst.n), "best must be a permutation");
        assert!(
            result.best_length <= optimal * 1.05,
            "GA best {} should be near optimal {}",
            result.best_length,
            optimal
        );
        assert!(result.best_length >= optimal - 1e-9, "cannot beat the global optimum");
        assert_eq!(result.generations, 120);
        assert_eq!(result.best_history.len(), 121); // bootstrap + one per generation
    }

    #[test]
    fn best_history_is_monotone_non_increasing() {
        let inst = build_pentagon_tsp(6, 40.0);
        let result = run_tsp_ga_des(inst, opts(7), None);
        for w in result.best_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "best_history not monotone: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn direct_bootstrap_optimizer_runs_to_completion() {
        let inst = build_pentagon_tsp(5, 50.0);
        let lower_bound = held_karp_exact(&inst).length;
        let mut opt = TSPGAOptimizer::new("ga", inst.clone(), opts(99), false, None);
        assert_eq!(opt.get_population().len(), 50);
        while !opt.is_finished() {
            opt.run_time_step();
        }
        assert!(opt.is_finished());
        assert!(is_permutation(opt.get_best_tour(), inst.n));
        assert!(opt.get_best_length() >= lower_bound - 1e-9);
    }
}
