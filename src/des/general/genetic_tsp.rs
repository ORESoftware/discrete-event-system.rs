//! Port of `src/des/general/genetic-tsp.ts` — Genetic Algorithm for the
//! Travelling Salesman Problem, modelled as a discrete-event system (every
//! generation is a tick; selection, crossover, mutation, feasibility, fitness,
//! and replacement are stations; chromosomes are the movables).
//!
//! THE PROBLEM
//!   Given `n` cities at 2-D coordinates, find a permutation π minimising the
//!   closed Euclidean Hamiltonian-cycle length `Σ ‖coord(π_i) − coord(π_{i+1})‖`.
//!   Optional precedence constraints (`i` must appear before `j`) prune the
//!   search space, which is where branch cutting becomes meaningful.
//!
//! BRANCH CUTTING
//!   When precedence constraints are present, order-crossover can produce
//!   infeasible children. Three policies are supported:
//!     * `Cut`      — drop infeasible children, retry up to `retry_limit`
//!     * `Penalize` — accept but inflate the tour length by `penalty_per_violation`
//!     * `Repair`   — swap the violating cities until feasible (best-effort)
//!
//! MIGRATION NOTES
//!   * The TS file imports `PureTransform` from `shared/transform`; per the
//!     migration rules `PureTransform<I, O>` collapses onto the `Transform`
//!     trait. Each `class X extends PureTransform` becomes a unit struct
//!     implementing `Transform`, delegating to the corresponding free function
//!     (the canonical implementation lives in the free fn; the struct is the
//!     thin "behaviour object" wrapper).
//!   * The TS `rng: () => number` closures map to `&mut impl RandomSource`,
//!     called via `next_float()`; for the `Transform` structs the `&mut` RNG is
//!     carried inside the input struct so the trait's `&self` signature is kept.
//!   * **FLAGGED DEPENDENCIES NOT IN THE AVAILABLE LIST**: the TS file imports
//!     `PopulationOptimizer`, `runIterativeDES`, `intrinsicCheck`, and
//!     `monotonicityValidator` from `./des-base`. These are NOT yet ported on
//!     the Rust side (`des_base` currently only exposes `preconditions`), so the
//!     template-method GA driver from `PopulationOptimizer` and the single-station
//!     `runIterativeDES` loop are INLINED locally as `GeneticTspOptimizer` +
//!     `GeneticTspOptimizer::run`. The two intrinsic validators
//!     (`bestHistory` non-increasing, best is a permutation) are reproduced as
//!     `debug_assert!`s instead of the (unported) validator framework.
//!   * `Date.now()` timing is replaced by `std::time::Instant` (the TS migration
//!     header suggests a `Clock` capability; using `Instant` keeps the crate
//!     dependency-free — flagged as a minor deviation).

use std::time::Instant;

use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::linalg::Matrix;
use crate::des::shared::transform::Transform;

// =============================================================================
// Core public types
// =============================================================================

/// A TSP instance: city coordinates, the dense distance matrix, and optional
/// precedence constraints. (TS `interface TSPInstance`.)
#[derive(Clone, Debug)]
pub struct TSPInstance {
    pub n: usize,
    /// 2-D coordinates, one per city.
    pub coordinates: Vec<(f64, f64)>,
    /// `distance[i][j]` = Euclidean distance between city `i` and city `j`.
    pub distance: Matrix,
    /// Each `(i, j)` means "city `i` must appear somewhere before city `j`".
    pub precedence: Option<Vec<(usize, usize)>>,
}

/// A tour is a permutation of city indices. (TS `type Tour = number[]`.)
pub type Tour = Vec<usize>;

/// Shared Euclidean distance matrix builder.
fn dist_matrix(coords: &[(f64, f64)]) -> Matrix {
    let n = coords.len();
    let mut dist = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let dx = coords[i].0 - coords[j].0;
            let dy = coords[i].1 - coords[j].1;
            dist[i][j] = (dx * dx + dy * dy).sqrt();
        }
    }
    dist
}

// =============================================================================
// Instance builders
// =============================================================================

/// Input for [`BuildRandomTSP`]. (TS `interface BuildRandomTSPInput`.)
#[derive(Clone, Debug)]
pub struct BuildRandomTspInput {
    pub n: usize,
    pub seed: Option<u32>,
    pub precedence: Option<Vec<(usize, usize)>>,
}

/// Build a random TSP instance with cities uniformly in `[0, 100)²`.
pub struct BuildRandomTSP;

impl Transform<BuildRandomTspInput, TSPInstance> for BuildRandomTSP {
    fn transform(&self, input: BuildRandomTspInput) -> TSPInstance {
        let n = input.n;
        let seed = input.seed.unwrap_or(42);
        let mut rng = mulberry32(seed);
        let mut coords: Vec<(f64, f64)> = Vec::with_capacity(n);
        for _ in 0..n {
            coords.push((rng.next_float() * 100.0, rng.next_float() * 100.0));
        }
        let distance = dist_matrix(&coords);
        TSPInstance {
            n,
            coordinates: coords,
            distance,
            precedence: input.precedence,
        }
    }
}

/// Free-function form of [`BuildRandomTSP`] (TS `buildRandomTSP`, kept because
/// it is imported by sibling modules).
pub fn build_random_tsp(
    n: usize,
    seed: u32,
    precedence: Option<Vec<(usize, usize)>>,
) -> TSPInstance {
    BuildRandomTSP.transform(BuildRandomTspInput {
        n,
        seed: Some(seed),
        precedence,
    })
}

/// Input for [`BuildPentagonTSP`]. (TS `interface BuildPentagonTSPInput`.)
#[derive(Clone, Debug)]
pub struct BuildPentagonTspInput {
    pub n: Option<usize>,
    pub radius: Option<f64>,
}

/// A small, well-known instance for unit testing: `n` cities on a regular
/// polygon. The optimal tour visits them in order.
pub struct BuildPentagonTSP;

impl Transform<BuildPentagonTspInput, TSPInstance> for BuildPentagonTSP {
    fn transform(&self, input: BuildPentagonTspInput) -> TSPInstance {
        let n = input.n.unwrap_or(5);
        let radius = input.radius.unwrap_or(50.0);
        let mut coords: Vec<(f64, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            let a = (2.0 * std::f64::consts::PI * i as f64) / n as f64;
            coords.push((50.0 + radius * a.cos(), 50.0 + radius * a.sin()));
        }
        let distance = dist_matrix(&coords);
        TSPInstance {
            n,
            coordinates: coords,
            distance,
            precedence: None,
        }
    }
}

/// Free-function form of [`BuildPentagonTSP`] (TS `buildPentagonTSP`, kept
/// because it is imported by sibling modules).
pub fn build_pentagon_tsp(n: usize, radius: f64) -> TSPInstance {
    BuildPentagonTSP.transform(BuildPentagonTspInput {
        n: Some(n),
        radius: Some(radius),
    })
}

// =============================================================================
// Tour evaluation + feasibility
// =============================================================================

/// Shared input for tour-evaluation transforms. (TS `interface InstanceTourInput`.)
pub struct InstanceTourInput<'a> {
    pub instance: &'a TSPInstance,
    pub tour: &'a [usize],
}

/// Total closed-cycle length of a tour. (Imported by sibling modules — kept pub.)
pub fn tour_length(instance: &TSPInstance, tour: &[usize]) -> f64 {
    let n = instance.n;
    let mut s = 0.0;
    for i in 0..n {
        s += instance.distance[tour[i]][tour[(i + 1) % n]];
    }
    s
}

/// Behaviour-object wrapper for [`tour_length`]. (TS `class TourLength`.)
pub struct TourLength;

impl<'a> Transform<InstanceTourInput<'a>, f64> for TourLength {
    fn transform(&self, input: InstanceTourInput<'a>) -> f64 {
        tour_length(input.instance, input.tour)
    }
}

/// Returns `None` if the tour is feasible, otherwise the violating `(i, j)`
/// precedence pair. (Imported by sibling modules — kept pub.)
pub fn check_precedence(instance: &TSPInstance, tour: &[usize]) -> Option<(usize, usize)> {
    let precedence = match &instance.precedence {
        Some(p) => p,
        None => return None,
    };
    // Position map: where does city `c` appear in the tour?
    let mut pos = vec![-1_i64; instance.n];
    for (i, &c) in tour.iter().enumerate() {
        pos[c] = i as i64;
    }
    for &(a, b) in precedence {
        if pos[a] >= 0 && pos[b] >= 0 && pos[a] >= pos[b] {
            return Some((a, b));
        }
    }
    None
}

/// Behaviour-object wrapper for [`check_precedence`]. (TS `class CheckPrecedence`.)
pub struct CheckPrecedence;

impl<'a> Transform<InstanceTourInput<'a>, Option<(usize, usize)>> for CheckPrecedence {
    fn transform(&self, input: InstanceTourInput<'a>) -> Option<(usize, usize)> {
        check_precedence(input.instance, input.tour)
    }
}

/// Input for [`IsPermutation`]. (TS `interface IsPermutationInput`.)
pub struct IsPermutationInput<'a> {
    pub tour: &'a [usize],
    pub n: usize,
}

/// `true` iff `tour` is a permutation of `[0, n)`. (Imported by sibling
/// modules — kept pub.)
pub fn is_permutation(tour: &[usize], n: usize) -> bool {
    if tour.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &c in tour {
        if c >= n {
            return false;
        }
        if seen[c] {
            return false;
        }
        seen[c] = true;
    }
    true
}

/// Behaviour-object wrapper for [`is_permutation`]. (TS `class IsPermutation`.)
pub struct IsPermutation;

impl<'a> Transform<IsPermutationInput<'a>, bool> for IsPermutation {
    fn transform(&self, input: IsPermutationInput<'a>) -> bool {
        is_permutation(input.tour, input.n)
    }
}

// =============================================================================
// GA operators (stations in the DES)
// =============================================================================

/// Input for [`TournamentSelect`]. (TS `interface TournamentSelectInput`.)
pub struct TournamentSelectInput<'a, R: RandomSource> {
    pub population_lengths: &'a [f64],
    pub size: usize,
    pub rng: &'a mut R,
}

/// Tournament selection: sample `size` chromosomes uniformly at random, return
/// the index of the lowest-tour-length one.
pub fn tournament_select(
    population_lengths: &[f64],
    size: usize,
    rng: &mut impl RandomSource,
) -> usize {
    let len = population_lengths.len();
    let mut best_idx = (rng.next_float() * len as f64).floor() as usize;
    let mut best_len = population_lengths[best_idx];
    for _ in 1..size {
        let idx = (rng.next_float() * len as f64).floor() as usize;
        if population_lengths[idx] < best_len {
            best_len = population_lengths[idx];
            best_idx = idx;
        }
    }
    best_idx
}

/// Behaviour-object wrapper for [`tournament_select`]. (TS `class TournamentSelect`.)
pub struct TournamentSelect;

impl<'a, R: RandomSource> Transform<TournamentSelectInput<'a, R>, usize> for TournamentSelect {
    fn transform(&self, input: TournamentSelectInput<'a, R>) -> usize {
        tournament_select(input.population_lengths, input.size, input.rng)
    }
}

/// Input for [`OrderCrossover`]. (TS `interface OrderCrossoverInput`.)
pub struct OrderCrossoverInput<'a, R: RandomSource> {
    pub parent1: &'a [usize],
    pub parent2: &'a [usize],
    pub rng: &'a mut R,
}

/// Order-Crossover (OX): copy a random sub-segment of `parent1` into the child,
/// then fill remaining positions with `parent2`'s order (skipping duplicates).
/// The result is always a permutation.
pub fn order_crossover(parent1: &[usize], parent2: &[usize], rng: &mut impl RandomSource) -> Tour {
    let n = parent1.len();
    let mut a = (rng.next_float() * n as f64).floor() as usize;
    let mut b = (rng.next_float() * n as f64).floor() as usize;
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    let mut child = vec![usize::MAX; n];
    let mut in_child = vec![false; n];
    for i in a..=b {
        child[i] = parent1[i];
        in_child[parent1[i]] = true;
    }
    let mut p2cursor = (b + 1) % n;
    let mut cursor = (b + 1) % n;
    while cursor != a {
        while in_child[parent2[p2cursor]] {
            p2cursor = (p2cursor + 1) % n;
        }
        child[cursor] = parent2[p2cursor];
        in_child[parent2[p2cursor]] = true;
        p2cursor = (p2cursor + 1) % n;
        cursor = (cursor + 1) % n;
    }
    child
}

/// Behaviour-object wrapper for [`order_crossover`]. (TS `class OrderCrossover`.)
pub struct OrderCrossover;

impl<'a, R: RandomSource> Transform<OrderCrossoverInput<'a, R>, Tour> for OrderCrossover {
    fn transform(&self, input: OrderCrossoverInput<'a, R>) -> Tour {
        order_crossover(input.parent1, input.parent2, input.rng)
    }
}

/// Input for the mutation operators. (TS `interface MutateInput`.)
pub struct MutateInput<'a, R: RandomSource> {
    pub tour: &'a [usize],
    pub rng: &'a mut R,
}

/// Inversion mutation: reverse a random sub-segment (a single 2-opt move).
pub fn inversion_mutate(tour: &[usize], rng: &mut impl RandomSource) -> Tour {
    let n = tour.len();
    let mut a = (rng.next_float() * n as f64).floor() as usize;
    let mut b = (rng.next_float() * n as f64).floor() as usize;
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    let mut out = tour.to_vec();
    while a < b {
        out.swap(a, b);
        a += 1;
        b -= 1;
    }
    out
}

/// Behaviour-object wrapper for [`inversion_mutate`]. (TS `class InversionMutate`.)
pub struct InversionMutate;

impl<'a, R: RandomSource> Transform<MutateInput<'a, R>, Tour> for InversionMutate {
    fn transform(&self, input: MutateInput<'a, R>) -> Tour {
        inversion_mutate(input.tour, input.rng)
    }
}

/// Swap mutation: pick two distinct random positions and swap them.
pub fn swap_mutate(tour: &[usize], rng: &mut impl RandomSource) -> Tour {
    let n = tour.len();
    let a = (rng.next_float() * n as f64).floor() as usize;
    let mut b = (rng.next_float() * n as f64).floor() as usize;
    while b == a {
        b = (rng.next_float() * n as f64).floor() as usize;
    }
    let mut out = tour.to_vec();
    out.swap(a, b);
    out
}

/// Behaviour-object wrapper for [`swap_mutate`]. (TS `class SwapMutate`.)
pub struct SwapMutate;

impl<'a, R: RandomSource> Transform<MutateInput<'a, R>, Tour> for SwapMutate {
    fn transform(&self, input: MutateInput<'a, R>) -> Tour {
        swap_mutate(input.tour, input.rng)
    }
}

/// Input for [`RepairPrecedence`]. (TS `interface RepairPrecedenceInput`.)
pub struct RepairPrecedenceInput<'a> {
    pub instance: &'a TSPInstance,
    pub tour: &'a [usize],
    pub max_rounds: Option<usize>,
}

/// Result of a precedence-repair attempt. (TS `interface RepairPrecedenceResult`.)
#[derive(Clone, Debug)]
pub struct RepairPrecedenceResult {
    pub tour: Tour,
    pub feasible: bool,
}

/// Best-effort precedence repair: while a `(i, j)` constraint is violated, swap
/// the two cities. Repeat at most `max_rounds` times.
pub fn repair_precedence(
    instance: &TSPInstance,
    tour: &[usize],
    max_rounds: usize,
) -> RepairPrecedenceResult {
    if instance.precedence.is_none() {
        return RepairPrecedenceResult {
            tour: tour.to_vec(),
            feasible: true,
        };
    }
    let mut out = tour.to_vec();
    for _ in 0..max_rounds {
        match check_precedence(instance, &out) {
            None => {
                return RepairPrecedenceResult {
                    tour: out,
                    feasible: true,
                }
            }
            Some((a, b)) => {
                let pa = out.iter().position(|&x| x == a).unwrap();
                let pb = out.iter().position(|&x| x == b).unwrap();
                out.swap(pa, pb);
            }
        }
    }
    let feasible = check_precedence(instance, &out).is_none();
    RepairPrecedenceResult {
        tour: out,
        feasible,
    }
}

/// Behaviour-object wrapper for [`repair_precedence`]. (TS `class RepairPrecedence`.)
pub struct RepairPrecedence;

impl<'a> Transform<RepairPrecedenceInput<'a>, RepairPrecedenceResult> for RepairPrecedence {
    fn transform(&self, input: RepairPrecedenceInput<'a>) -> RepairPrecedenceResult {
        repair_precedence(input.instance, input.tour, input.max_rounds.unwrap_or(4))
    }
}

// =============================================================================
// 2-opt local search + nearest-neighbour construction
// =============================================================================

fn reverse_segment(tour: &mut [usize], mut lo: usize, mut hi: usize) {
    while lo < hi {
        tour.swap(lo, hi);
        lo += 1;
        hi -= 1;
    }
}

/// Input for [`TwoOptImprove`]. (TS `interface TwoOptImproveInput`.)
pub struct TwoOptImproveInput<'a> {
    pub instance: &'a TSPInstance,
    pub tour: &'a [usize],
    pub max_passes: Option<usize>,
}

/// First-improvement 2-opt: reverse the first improving segment, up to
/// `max_passes` passes.
pub fn two_opt_improve(instance: &TSPInstance, tour: &[usize], max_passes: usize) -> Tour {
    let n = tour.len();
    if n < 4 || max_passes == 0 {
        return tour.to_vec();
    }
    let mut out = tour.to_vec();
    let d = &instance.distance;
    for _pass in 0..max_passes {
        let mut improved = false;
        let mut i = 0;
        while i < n - 1 && !improved {
            let a = out[i];
            let b = out[(i + 1) % n];
            let mut k = i + 2;
            while k < n {
                if !(i == 0 && k == n - 1) {
                    let c = out[k];
                    let e = out[(k + 1) % n];
                    let delta = d[a][c] + d[b][e] - d[a][b] - d[c][e];
                    if delta < -1e-12 {
                        reverse_segment(&mut out, i + 1, k);
                        improved = true;
                        break;
                    }
                }
                k += 1;
            }
            i += 1;
        }
        if !improved {
            break;
        }
    }
    out
}

/// Behaviour-object wrapper for [`two_opt_improve`]. (TS `class TwoOptImprove`.)
pub struct TwoOptImprove;

impl<'a> Transform<TwoOptImproveInput<'a>, Tour> for TwoOptImprove {
    fn transform(&self, input: TwoOptImproveInput<'a>) -> Tour {
        two_opt_improve(input.instance, input.tour, input.max_passes.unwrap_or(1))
    }
}

fn nearest_neighbor_tour(instance: &TSPInstance, start: usize) -> Tour {
    let n = instance.n;
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
            let dd = instance.distance[cur][j];
            if dd < best_d {
                best_d = dd;
                best_next = j as i64;
            }
        }
        let bn = best_next as usize;
        tour.push(bn);
        visited[bn] = true;
        cur = bn;
    }
    tour
}

// =============================================================================
// GA solver options / results
// =============================================================================

/// Constraint-handling policy. (TS `'cut' | 'penalize' | 'repair'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feasibility {
    Cut,
    Penalize,
    Repair,
}

/// Initial-population strategy. (TS `'random' | 'nearest-neighbor'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InitMode {
    Random,
    NearestNeighbor,
}

/// Memetic local-search policy. (TS `'none' | 'two-opt'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LocalSearch {
    None,
    TwoOpt,
}

/// GA solver options; every field defaults if `None`. (TS `interface GASolverOptions`.)
///
/// `on_generation` is a `FnMut` callback invoked after each generation with
/// `(gen, &info)` — mirroring the TS `onGeneration` closure.
#[derive(Default)]
pub struct GASolverOptions {
    pub population_size: Option<usize>,
    pub num_generations: Option<usize>,
    pub tournament_size: Option<usize>,
    pub crossover_prob: Option<f64>,
    pub mutation_prob: Option<f64>,
    pub elitism: Option<usize>,
    pub seed: Option<u32>,
    pub feasibility: Option<Feasibility>,
    pub penalty_per_violation: Option<f64>,
    pub retry_limit: Option<usize>,
    pub init: Option<InitMode>,
    pub local_search: Option<LocalSearch>,
    pub local_search_prob: Option<f64>,
    pub local_search_passes: Option<usize>,
    pub on_generation: Option<Box<dyn FnMut(usize, &GenerationInfo)>>,
}

/// Per-generation summary passed to the `on_generation` callback. (TS
/// `interface GenerationInfo`.)
#[derive(Clone, Debug)]
pub struct GenerationInfo {
    pub best: f64,
    pub mean: f64,
    pub worst: f64,
    pub elite_tour: Tour,
    pub num_feasible_children: usize,
    pub num_infeasible_children: usize,
}

/// Final result of a GA run. (TS `interface GASolverResult`.)
#[derive(Clone, Debug)]
pub struct GASolverResult {
    pub best_tour: Tour,
    pub best_length: f64,
    pub per_generation_best: Vec<f64>,
    pub per_generation_mean: Vec<f64>,
    pub per_generation_elite: Vec<Tour>,
    pub total_feasible_evaluated: usize,
    pub total_infeasible_cut: usize,
    pub local_search_applications: usize,
    pub elapsed_ms: f64,
    pub performance: GAPerformanceStats,
    pub generations: usize,
}

/// Throughput / improvement statistics. (TS `interface GAPerformanceStats`.)
#[derive(Clone, Debug)]
pub struct GAPerformanceStats {
    pub elapsed_ms: f64,
    pub generations_per_second: f64,
    pub estimated_evaluations: usize,
    pub evaluations_per_second: f64,
    pub initial_best: f64,
    pub final_best: f64,
    pub absolute_improvement: f64,
    pub relative_improvement: f64,
}

/// Options with every field resolved to a concrete value (internal).
struct FilledOptions {
    population_size: usize,
    num_generations: usize,
    tournament_size: usize,
    crossover_prob: f64,
    mutation_prob: f64,
    elitism: usize,
    seed: u32,
    feasibility: Feasibility,
    penalty_per_violation: f64,
    retry_limit: usize,
    init: InitMode,
    local_search: LocalSearch,
    local_search_prob: f64,
    local_search_passes: usize,
    on_generation: Option<Box<dyn FnMut(usize, &GenerationInfo)>>,
}

// =============================================================================
// GeneticTspOptimizer — inlined `PopulationOptimizer<Tour>` leaf + driver loop.
//
// FLAGGED: `PopulationOptimizer` / `runIterativeDES` are not yet ported; the
// template-method breeding loop and the single-station run loop are inlined
// here. Hook semantics match the TS overrides one-to-one.
// =============================================================================

struct GeneticTspOptimizer {
    // ── genetic-tsp config ──
    inst: TSPInstance,
    num_gen: usize,
    tournament_k: usize,
    cx_prob: f64,
    mut_prob: f64,
    elite_n: usize,
    init_mode: InitMode,
    feas: Feasibility,
    penalty: f64,
    retry_limit: usize,
    local_search: LocalSearch,
    local_search_prob: f64,
    local_search_passes: usize,
    on_generation_cb: Option<Box<dyn FnMut(usize, &GenerationInfo)>>,

    // ── PopulationOptimizer base state ──
    pop_size: usize,
    population: Vec<Tour>,
    fitness: Vec<f64>,
    generation: usize,
    best: Tour,
    best_fitness: f64,
    finished: bool,
    initialized: bool,
    best_history: Vec<f64>,
    mean_history: Vec<f64>,
    worst_history: Vec<f64>,
    rng: SeededRandom,

    // ── genetic-tsp counters / history ──
    feas_count: usize,
    infeas_count: usize,
    local_search_applications: usize,
    per_gen_best: Vec<f64>,
    per_gen_mean: Vec<f64>,
    per_gen_elite: Vec<Tour>,
}

impl GeneticTspOptimizer {
    fn new(inst: TSPInstance, o: FilledOptions) -> Self {
        let elite_n = o.elitism.min(o.population_size);
        let mut opt = GeneticTspOptimizer {
            inst,
            num_gen: o.num_generations,
            tournament_k: o.tournament_size,
            cx_prob: o.crossover_prob,
            mut_prob: o.mutation_prob,
            elite_n,
            init_mode: o.init,
            feas: o.feasibility,
            penalty: o.penalty_per_violation,
            retry_limit: o.retry_limit,
            local_search: o.local_search,
            local_search_prob: o.local_search_prob,
            local_search_passes: o.local_search_passes,
            on_generation_cb: o.on_generation,
            pop_size: o.population_size,
            population: Vec::new(),
            fitness: Vec::new(),
            generation: 0,
            best: Vec::new(),
            best_fitness: f64::INFINITY,
            finished: false,
            initialized: false,
            best_history: Vec::new(),
            mean_history: Vec::new(),
            worst_history: Vec::new(),
            rng: mulberry32(o.seed),
            feas_count: 0,
            infeas_count: 0,
            local_search_applications: 0,
            per_gen_best: Vec::new(),
            per_gen_mean: Vec::new(),
            per_gen_elite: Vec::new(),
        };
        opt.bootstrap();
        opt
    }

    // ── bootstrap (seed population + fitness) ──

    fn bootstrap(&mut self) {
        let population = self.initial_population(self.pop_size);
        if population.len() != self.pop_size {
            panic!(
                "initialPopulation returned {} individuals, expected {}",
                population.len(),
                self.pop_size
            );
        }
        self.population = population;
        let fitness: Vec<f64> = self.population.iter().map(|x| self.evaluate(x)).collect();
        for (i, f) in fitness.iter().enumerate() {
            if !f.is_finite() {
                panic!("genetic-tsp: initial population fitness[{i}] must be finite; got {f}");
            }
        }
        self.fitness = fitness;
        self.record_best();
        self.initialized = true;
    }

    // ── hooks ──

    fn initial_population(&mut self, size: usize) -> Vec<Tour> {
        let n = self.inst.n;
        let mut out: Vec<Tour> = Vec::new();
        if self.init_mode == InitMode::NearestNeighbor {
            let seed_count = n.min(size);
            for i in 0..seed_count {
                out.push(nearest_neighbor_tour(&self.inst, i));
            }
            while out.len() < size {
                let mut t: Tour = Vec::new();
                let mut remaining: Vec<usize> = (0..n).collect();
                while !remaining.is_empty() {
                    let idx = (self.rng.next_float() * remaining.len() as f64).floor() as usize;
                    t.push(remaining.remove(idx));
                }
                out.push(t);
            }
            return out;
        }
        for _ in 0..size {
            let mut t: Tour = (0..n).collect();
            // Fisher–Yates shuffle (i = len-1 .. 1).
            for i in (1..t.len()).rev() {
                let j = (self.rng.next_float() * (i as f64 + 1.0)).floor() as usize;
                t.swap(i, j);
            }
            out.push(t);
        }
        out
    }

    fn evaluate(&self, t: &[usize]) -> f64 {
        let mut len = tour_length(&self.inst, t);
        if self.feas == Feasibility::Penalize && check_precedence(&self.inst, t).is_some() {
            len += self.penalty;
        }
        len
    }

    fn select_parents(&mut self) -> (Tour, Tour) {
        let i1 = tournament_select(&self.fitness, self.tournament_k, &mut self.rng);
        let i2 = tournament_select(&self.fitness, self.tournament_k, &mut self.rng);
        (self.population[i1].clone(), self.population[i2].clone())
    }

    fn recombine(&mut self, parents: &(Tour, Tour)) -> Tour {
        if self.rng.next_float() < self.cx_prob {
            order_crossover(&parents.0, &parents.1, &mut self.rng)
        } else {
            parents.0.clone()
        }
    }

    fn mutate(&mut self, child: Tour) -> Tour {
        let mut out = child;
        if self.rng.next_float() < self.mut_prob {
            out = if self.rng.next_float() < 0.5 {
                inversion_mutate(&out, &mut self.rng)
            } else {
                swap_mutate(&out, &mut self.rng)
            };
        }
        if self.local_search == LocalSearch::TwoOpt
            && self.rng.next_float() < self.local_search_prob
        {
            let improved = two_opt_improve(&self.inst, &out, self.local_search_passes);
            if tour_length(&self.inst, &improved) < tour_length(&self.inst, &out) - 1e-12 {
                self.local_search_applications += 1;
            }
            out = improved;
        }
        out
    }

    fn should_stop(&self, gen: usize) -> bool {
        gen >= self.num_gen
    }

    fn elite_count(&self) -> usize {
        self.elite_n
    }

    fn has_precedence(&self) -> bool {
        match &self.inst.precedence {
            Some(p) => !p.is_empty(),
            None => false,
        }
    }

    /// Constraint hook (TS `acceptChild`). `Cut` accepts iff feasible; `Repair`
    /// patches the child in place and always accepts; `Penalize`/no-precedence
    /// always accept.
    fn accept_child(&mut self, child: &mut Tour) -> bool {
        if !self.has_precedence() {
            self.feas_count += 1;
            return true;
        }
        if self.feas == Feasibility::Penalize {
            self.feas_count += 1;
            return true;
        }
        if self.feas == Feasibility::Repair {
            let r = repair_precedence(&self.inst, child, 4);
            for i in 0..r.tour.len() {
                child[i] = r.tour[i];
            }
            if r.feasible {
                self.feas_count += 1;
                return true;
            }
            self.infeas_count += 1;
            return true; // accept the partial-repair child anyway
        }
        // Cut
        if check_precedence(&self.inst, child).is_none() {
            self.feas_count += 1;
            return true;
        }
        false
    }

    fn child_retry_limit(&self) -> usize {
        if !self.has_precedence() {
            return 1;
        }
        if self.feas == Feasibility::Cut {
            self.retry_limit.max(1)
        } else {
            1
        }
    }

    fn on_child_rejected(&mut self, attempt: usize) {
        // Only count as "infeasible cut" once the entire retry budget is used up.
        if attempt >= self.child_retry_limit() {
            self.infeas_count += 1;
        }
    }

    fn on_generation(&mut self) {
        let min_len = self.fitness.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean_len = self.fitness.iter().sum::<f64>() / self.fitness.len() as f64;
        let max_len = self
            .fitness
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let elite_idx = self.fitness.iter().position(|&x| x == min_len).unwrap();
        let elite_tour = self.population[elite_idx].clone();
        self.per_gen_best.push(min_len);
        self.per_gen_mean.push(mean_len);
        self.per_gen_elite.push(elite_tour.clone());

        // Inlined intrinsic validator: best must be a permutation of [0, n).
        debug_assert!(
            is_permutation(&self.best, self.inst.n),
            "genetic-tsp.best-is-valid-permutation"
        );

        if let Some(cb) = self.on_generation_cb.as_mut() {
            let info = GenerationInfo {
                best: min_len,
                mean: mean_len,
                worst: max_len,
                elite_tour,
                num_feasible_children: self.feas_count,
                num_infeasible_children: self.infeas_count,
            };
            cb(self.generation - 1, &info);
        }
    }

    fn record_best(&mut self) {
        let mut best_idx = 0;
        let mut best_f = self.fitness[0];
        let mut mean = 0.0;
        let mut worst = f64::NEG_INFINITY;
        for i in 0..self.fitness.len() {
            let f = self.fitness[i];
            mean += f;
            if f < best_f {
                best_f = f;
                best_idx = i;
            }
            if f > worst {
                worst = f;
            }
        }
        mean /= self.fitness.len() as f64;
        if best_f < self.best_fitness {
            self.best_fitness = best_f;
            self.best = self.population[best_idx].clone();
        }
        self.best_history.push(self.best_fitness);
        self.mean_history.push(mean);
        self.worst_history.push(worst);

        // Inlined intrinsic validator: bestHistory is non-increasing.
        let h = &self.best_history;
        debug_assert!(
            h.len() < 2 || h[h.len() - 1] <= h[h.len() - 2] + 1e-12,
            "genetic-tsp.bestHistory.monotone"
        );
    }

    // ── template method (one generation) ──

    fn run_time_step(&mut self) {
        if self.finished {
            return;
        }
        if self.should_stop(self.generation) {
            self.finished = true;
            return;
        }
        let mut new_pop: Vec<Tour> = Vec::new();
        let mut new_fit: Vec<f64> = Vec::new();

        // Elitism — copy best k unchanged.
        let elite_k = self.elite_count().min(self.pop_size);
        if elite_k > 0 {
            let mut order: Vec<(f64, usize)> = self
                .fitness
                .iter()
                .enumerate()
                .map(|(i, &f)| (f, i))
                .collect();
            // Stable ascending sort by fitness (matches V8's stable Array.sort).
            order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for entry in order.iter().take(elite_k) {
                let idx = entry.1;
                new_pop.push(self.population[idx].clone());
                new_fit.push(entry.0);
            }
        }

        let retry_budget = self.child_retry_limit().max(1);
        while new_pop.len() < self.pop_size {
            let mut child: Tour = Vec::new();
            let mut accepted = false;
            for attempt in 0..retry_budget {
                let parents = self.select_parents();
                let c = self.recombine(&parents);
                let mut c = self.mutate(c);
                if self.accept_child(&mut c) {
                    child = c;
                    accepted = true;
                    break;
                }
                self.on_child_rejected(attempt);
                child = c;
            }
            if !accepted {
                self.on_child_rejected(retry_budget);
            }
            let f = self.evaluate(&child);
            new_pop.push(child);
            new_fit.push(f);
        }

        self.population = new_pop;
        self.fitness = new_fit;
        self.record_best();
        self.generation += 1;
        self.on_generation();
    }

    /// Inlined single-station `runIterativeDES`: tick until quiescent.
    fn run(&mut self) {
        while self.initialized && !self.finished {
            self.run_time_step();
        }
    }
}

/// Run the GA. (TS `runGeneticTSP`.) Each tick is exactly one generation.
pub fn run_genetic_tsp(instance: TSPInstance, options: GASolverOptions) -> GASolverResult {
    let t0 = Instant::now();
    let filled = FilledOptions {
        population_size: options.population_size.unwrap_or(100),
        num_generations: options.num_generations.unwrap_or(200),
        tournament_size: options.tournament_size.unwrap_or(3),
        crossover_prob: options.crossover_prob.unwrap_or(0.95),
        mutation_prob: options.mutation_prob.unwrap_or(0.3),
        elitism: options.elitism.unwrap_or(2),
        seed: options.seed.unwrap_or(1),
        feasibility: options.feasibility.unwrap_or(Feasibility::Cut),
        penalty_per_violation: options.penalty_per_violation.unwrap_or(1e6),
        retry_limit: options.retry_limit.unwrap_or(8),
        init: options.init.unwrap_or(InitMode::Random),
        local_search: options.local_search.unwrap_or(LocalSearch::None),
        local_search_prob: options.local_search_prob.unwrap_or(1.0),
        local_search_passes: options.local_search_passes.unwrap_or(1),
        on_generation: options.on_generation,
    };
    let pop_size = filled.population_size;
    let mut opt = GeneticTspOptimizer::new(instance, filled);
    opt.run();

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let best = opt.best.clone();
    let estimated_evaluations = pop_size * (opt.generation + 1);
    let initial_best = opt
        .per_gen_best
        .first()
        .copied()
        .unwrap_or(opt.best_fitness);
    let final_best = tour_length(&opt.inst, &best);
    let absolute_improvement = initial_best - final_best;
    let secs = (elapsed_ms / 1000.0).max(1e-9);

    GASolverResult {
        best_tour: best.clone(),
        best_length: final_best,
        per_generation_best: opt.per_gen_best.clone(),
        per_generation_mean: opt.per_gen_mean.clone(),
        per_generation_elite: opt.per_gen_elite.clone(),
        total_feasible_evaluated: opt.feas_count,
        total_infeasible_cut: opt.infeas_count,
        local_search_applications: opt.local_search_applications,
        elapsed_ms,
        performance: GAPerformanceStats {
            elapsed_ms,
            generations_per_second: opt.generation as f64 / secs,
            estimated_evaluations,
            evaluations_per_second: estimated_evaluations as f64 / secs,
            initial_best,
            final_best,
            absolute_improvement,
            relative_improvement: absolute_improvement / initial_best.abs().max(1e-9),
        },
        generations: opt.generation,
    }
}

// =============================================================================
// Lower bounds / reference solvers
// =============================================================================

/// Held–Karp exact result. (TS `interface HeldKarpResult`.)
#[derive(Clone, Debug)]
pub struct HeldKarpResult {
    pub tour: Tour,
    pub length: f64,
}

/// Held–Karp bitmask DP exact solver for small TSPs (`n ≤ 16`). `panic!`s for
/// larger instances (invariant violation, was `throw` in TS).
pub fn held_karp_exact(instance: &TSPInstance) -> HeldKarpResult {
    let n = instance.n;
    if n > 16 {
        panic!("Held–Karp only practical for n ≤ 16, got {n}");
    }
    let big_n = 1usize << n;
    // dp[mask * n + i] = min path length from 0, visiting `mask`, ending at i.
    let mut dp = vec![f64::INFINITY; big_n * n];
    let mut parent = vec![-1_i64; big_n * n];
    dp[n] = 0.0; // dp[(1) * n + 0]: start at city 0, mask = {0}
    for mask in 1..big_n {
        if mask & 1 == 0 {
            continue;
        }
        for i in 0..n {
            if mask & (1usize << i) == 0 {
                continue;
            }
            let cur = dp[mask * n + i];
            if !cur.is_finite() {
                continue;
            }
            for j in 0..n {
                if mask & (1usize << j) != 0 {
                    continue;
                }
                let new_mask = mask | (1usize << j);
                let cand = cur + instance.distance[i][j];
                if cand < dp[new_mask * n + j] {
                    dp[new_mask * n + j] = cand;
                    parent[new_mask * n + j] = i as i64;
                }
            }
        }
    }
    let full_mask = big_n - 1;
    let mut best_end = 1usize;
    let mut best_len = f64::INFINITY;
    for i in 1..n {
        let cand = dp[full_mask * n + i] + instance.distance[i][0];
        if cand < best_len {
            best_len = cand;
            best_end = i;
        }
    }
    // Reconstruct.
    let mut tour: Tour = Vec::new();
    let mut mask = full_mask;
    let mut cur: i64 = best_end as i64;
    while cur != -1 {
        let c = cur as usize;
        tour.push(c);
        let prev = parent[mask * n + c];
        mask ^= 1usize << c;
        cur = prev;
    }
    tour.reverse();
    HeldKarpResult {
        tour,
        length: best_len,
    }
}

/// Behaviour-object wrapper for [`held_karp_exact`]. (TS `class HeldKarpExact`.)
pub struct HeldKarpExact;

impl<'a> Transform<&'a TSPInstance, HeldKarpResult> for HeldKarpExact {
    fn transform(&self, instance: &'a TSPInstance) -> HeldKarpResult {
        held_karp_exact(instance)
    }
}

/// 1-tree lower bound: MST on cities `{1, …, n-1}` plus the two cheapest edges
/// from city 0.
pub fn one_tree_lower_bound(instance: &TSPInstance) -> f64 {
    let n = instance.n;
    let mut in_tree = vec![false; n];
    let mut min_edge = vec![f64::INFINITY; n];
    in_tree[0] = true; // exclude city 0
    let mut mst_cost = 0.0;
    if n < 2 {
        return 0.0;
    }
    in_tree[1] = true;
    for j in 2..n {
        min_edge[j] = instance.distance[1][j];
    }
    for _count in 1..(n - 1) {
        let mut best: i64 = -1;
        let mut best_val = f64::INFINITY;
        for j in 2..n {
            if !in_tree[j] && min_edge[j] < best_val {
                best_val = min_edge[j];
                best = j as i64;
            }
        }
        if best == -1 {
            break;
        }
        mst_cost += best_val;
        let b = best as usize;
        in_tree[b] = true;
        for k in 2..n {
            if !in_tree[k] && instance.distance[b][k] < min_edge[k] {
                min_edge[k] = instance.distance[b][k];
            }
        }
    }
    // Two cheapest edges from city 0.
    let mut edges0: Vec<f64> = Vec::new();
    for j in 1..n {
        edges0.push(instance.distance[0][j]);
    }
    edges0.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mst_cost + edges0.first().copied().unwrap_or(0.0) + edges0.get(1).copied().unwrap_or(0.0)
}

/// Behaviour-object wrapper for [`one_tree_lower_bound`]. (TS `class OneTreeLowerBound`.)
pub struct OneTreeLowerBound;

impl<'a> Transform<&'a TSPInstance, f64> for OneTreeLowerBound {
    fn transform(&self, instance: &'a TSPInstance) -> f64 {
        one_tree_lower_bound(instance)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit square: cities at the four corners.
    fn unit_square() -> TSPInstance {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let distance = dist_matrix(&coords);
        TSPInstance {
            n: 4,
            coordinates: coords,
            distance,
            precedence: None,
        }
    }

    #[test]
    fn tour_length_known_instance() {
        let inst = unit_square();
        // Perimeter tour 0→1→2→3→0 has length 4 (four unit edges).
        let len = tour_length(&inst, &[0, 1, 2, 3]);
        assert!((len - 4.0).abs() < 1e-9, "expected 4.0, got {len}");
        // Crossing tour 0→2→1→3→0 is longer (two diagonals + two unit edges).
        let crossing = tour_length(&inst, &[0, 2, 1, 3]);
        assert!(crossing > len);
    }

    #[test]
    fn is_permutation_true_and_false() {
        assert!(is_permutation(&[0, 1, 2, 3], 4));
        assert!(is_permutation(&[3, 1, 0, 2], 4));
        assert!(!is_permutation(&[0, 1, 1, 3], 4)); // duplicate
        assert!(!is_permutation(&[0, 1, 2], 4)); // wrong length
        assert!(!is_permutation(&[0, 1, 2, 4], 4)); // out of range
    }

    #[test]
    fn ga_pentagon_converges_near_optimal() {
        let inst = build_pentagon_tsp(5, 50.0);
        let optimal = held_karp_exact(&inst).length;

        let options = GASolverOptions {
            population_size: Some(40),
            num_generations: Some(80),
            seed: Some(7),
            local_search: Some(LocalSearch::TwoOpt),
            local_search_passes: Some(5),
            ..GASolverOptions::default()
        };
        let result = run_genetic_tsp(inst.clone(), options);

        assert!(
            is_permutation(&result.best_tour, inst.n),
            "best tour must be a valid permutation"
        );
        // Loose tolerance: GA + 2-opt should reach within 5% of optimum.
        assert!(
            result.best_length <= optimal * 1.05,
            "GA best {} should be near optimal {}",
            result.best_length,
            optimal
        );
        assert_eq!(result.generations, 80);
        assert_eq!(result.per_generation_best.len(), 80);
    }
}
