//! Generic genetic-algorithm driver with several population-update flavors.
//!
//! Fitness is **minimized**. The breeding loop mirrors `genetic_tsp` semantics
//! (tournament selection, elitism, optional child-retry) without inlining a
//! second copy for every problem domain.

use std::time::Instant;

use crate::des::general::des_base::population_optimizer::{PopulationOptimizer, PopulationState};
use crate::des::general::des_base::runner::{
    run_iterative_des, IterativeRunOptions, IterativeRunSummary,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

/// How the population advances each generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaFlavor {
    /// Full generational replacement with elitism (classic GA).
    Generational,
    /// Replace the single worst individual each offspring.
    SteadyState,
    /// μ+λ: keep parents, add λ children, truncate to μ.
    MuPlusLambda,
    /// Multiple sub-populations with periodic migration of elites.
    Island,
}

/// GA configuration (unset fields use documented defaults).
#[derive(Clone, Debug)]
pub struct GaOptions {
    pub population_size: usize,
    pub num_generations: usize,
    pub tournament_size: Option<usize>,
    pub crossover_prob: Option<f64>,
    pub mutation_prob: Option<f64>,
    pub elitism: Option<usize>,
    pub seed: Option<u32>,
    pub flavor: Option<GaFlavor>,
    /// Islands only: number of demes (default 4).
    pub num_islands: Option<usize>,
    /// Islands only: generations between migrations (default 5).
    pub migration_interval: Option<usize>,
    /// μ+λ only: offspring per generation (default `population_size`).
    pub lambda_offspring: Option<usize>,
    pub child_retry_limit: Option<usize>,
}

impl GaOptions {
    pub fn with_defaults(population_size: usize, num_generations: usize) -> Self {
        GaOptions {
            population_size,
            num_generations,
            tournament_size: None,
            crossover_prob: None,
            mutation_prob: None,
            elitism: None,
            seed: None,
            flavor: None,
            num_islands: None,
            migration_interval: None,
            lambda_offspring: None,
            child_retry_limit: None,
        }
    }
}

/// Per-generation telemetry.
#[derive(Clone, Debug)]
pub struct GaGenerationInfo<I> {
    pub generation: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub worst_fitness: f64,
    pub best_individual: I,
}

/// Final GA output.
#[derive(Clone, Debug)]
pub struct GaResult<I> {
    pub best: I,
    pub best_fitness: f64,
    pub per_generation_best: Vec<f64>,
    pub per_generation_mean: Vec<f64>,
    pub generations: usize,
    pub elapsed_ms: f64,
}

/// Builds an initial population for a population-based search.
pub trait PopulationInitializer<I: Clone> {
    fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<I>;
}

/// Scores one individual, with an overridable population-batch path.
pub trait FitnessEvaluator<I: Clone> {
    fn evaluate(&self, individual: &I) -> f64;

    /// Score a whole population. Override this when a problem can use SIMD,
    /// BLAS-shaped matrix batches, GPU kernels, or shared cached state.
    fn evaluate_population(&self, population: &[I]) -> Vec<f64> {
        population.iter().map(|x| self.evaluate(x)).collect()
    }
}

/// Genetic search operators shared by standalone and DES-station GA runners.
pub trait GeneticOperators<I: Clone> {
    fn crossover(&self, a: &I, b: &I, rng: &mut dyn RandomSource) -> I;
    fn mutate(&self, child: I, rng: &mut dyn RandomSource) -> I;
    /// Optional memetic polish (identity by default).
    fn local_search(&self, child: I) -> I {
        child
    }
    /// Constraint hook: return false to reject a child (retry up to limit).
    fn accept_child(&self, _child: &I) -> bool {
        true
    }
}

/// Problem-specific operators plugged into [`run_ga`] and [`EvolutionGaStation`].
pub trait GaProblem<I: Clone>:
    PopulationInitializer<I> + FitnessEvaluator<I> + GeneticOperators<I>
{
}

impl<I, P> GaProblem<I> for P
where
    I: Clone,
    P: PopulationInitializer<I> + FitnessEvaluator<I> + GeneticOperators<I>,
{
}

struct FilledGaOptions {
    pop: usize,
    gens: usize,
    tournament_k: usize,
    cx_prob: f64,
    mut_prob: f64,
    elite: usize,
    flavor: GaFlavor,
    num_islands: usize,
    migration_interval: usize,
    lambda: usize,
    retry_limit: usize,
    seed: u32,
}

fn fill_options(o: GaOptions) -> FilledGaOptions {
    FilledGaOptions {
        pop: o.population_size,
        gens: o.num_generations,
        tournament_k: o.tournament_size.unwrap_or(3),
        cx_prob: o.crossover_prob.unwrap_or(0.9),
        mut_prob: o.mutation_prob.unwrap_or(0.25),
        elite: o.elitism.unwrap_or(2).min(o.population_size),
        flavor: o.flavor.unwrap_or(GaFlavor::Generational),
        num_islands: o.num_islands.unwrap_or(4).max(1),
        migration_interval: o.migration_interval.unwrap_or(5).max(1),
        lambda: o.lambda_offspring.unwrap_or(o.population_size),
        retry_limit: o.child_retry_limit.unwrap_or(8).max(1),
        seed: o.seed.unwrap_or(42),
    }
}

fn tournament_pick<I: Clone>(
    pop: &[I],
    fitness: &[f64],
    k: usize,
    rng: &mut dyn RandomSource,
) -> usize {
    let n = pop.len();
    let mut best_i = (rng.next_float() * n as f64).floor() as usize % n;
    let mut best_f = fitness[best_i];
    for _ in 1..k {
        let j = (rng.next_float() * n as f64).floor() as usize % n;
        if fitness[j] < best_f {
            best_f = fitness[j];
            best_i = j;
        }
    }
    best_i
}

fn breed_one<I: Clone, P: GaProblem<I>>(
    problem: &P,
    pop: &[I],
    fitness: &[f64],
    o: &FilledGaOptions,
    rng: &mut dyn RandomSource,
) -> I {
    for attempt in 0..o.retry_limit {
        let p1 = tournament_pick(pop, fitness, o.tournament_k, rng);
        let p2 = tournament_pick(pop, fitness, o.tournament_k, rng);
        let mut child = if rng.next_float() < o.cx_prob {
            problem.crossover(&pop[p1], &pop[p2], rng)
        } else {
            pop[p1].clone()
        };
        if rng.next_float() < o.mut_prob {
            child = problem.mutate(child, rng);
        }
        child = problem.local_search(child);
        if problem.accept_child(&child) || attempt + 1 == o.retry_limit {
            return child;
        }
    }
    unreachable!()
}

fn eval_all<I: Clone, P: GaProblem<I>>(problem: &P, pop: &[I]) -> Vec<f64> {
    problem.evaluate_population(pop)
}

fn update_best<I: Clone>(best: &mut I, best_f: &mut f64, pop: &[I], fitness: &[f64]) {
    for (ind, f) in pop.iter().zip(fitness) {
        if *f < *best_f {
            *best_f = *f;
            *best = ind.clone();
        }
    }
}

fn mean_fitness(fitness: &[f64]) -> f64 {
    if fitness.is_empty() {
        return f64::INFINITY;
    }
    fitness.iter().sum::<f64>() / fitness.len() as f64
}

fn worst_index(fitness: &[f64]) -> usize {
    fitness
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn elite_indices(fitness: &[f64], elite: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..fitness.len()).collect();
    idx.sort_by(|&a, &b| fitness[a].total_cmp(&fitness[b]));
    idx.truncate(elite);
    idx
}

/// Run a genetic algorithm for `problem` with options `opts`.
pub fn run_ga<I: Clone, P: GaProblem<I>>(
    problem: P,
    opts: GaOptions,
    mut on_generation: Option<Box<dyn FnMut(&GaGenerationInfo<I>)>>,
) -> GaResult<I> {
    let o = fill_options(opts);
    let t0 = Instant::now();
    let mut rng = mulberry32(o.seed);

    match o.flavor {
        GaFlavor::Island => run_island_ga(problem, &o, &mut rng, &mut on_generation, t0),
        GaFlavor::MuPlusLambda => run_mu_plus_lambda(problem, &o, &mut rng, &mut on_generation, t0),
        GaFlavor::SteadyState => run_steady_state(problem, &o, &mut rng, &mut on_generation, t0),
        GaFlavor::Generational => run_generational(problem, &o, &mut rng, &mut on_generation, t0),
    }
}

fn run_generational<I: Clone, P: GaProblem<I>>(
    problem: P,
    o: &FilledGaOptions,
    rng: &mut SeededRandom,
    on_generation: &mut Option<Box<dyn FnMut(&GaGenerationInfo<I>)>>,
    t0: Instant,
) -> GaResult<I> {
    let mut pop = problem.initial_population(o.pop, rng);
    let mut fitness = eval_all(&problem, &pop);
    let mut best = pop[0].clone();
    let mut best_f = fitness[0];
    update_best(&mut best, &mut best_f, &pop, &fitness);
    let mut per_gen_best = Vec::with_capacity(o.gens);
    let mut per_gen_mean = Vec::with_capacity(o.gens);

    for gen in 0..o.gens {
        per_gen_best.push(best_f);
        per_gen_mean.push(mean_fitness(&fitness));
        if let Some(cb) = on_generation {
            cb(&GaGenerationInfo {
                generation: gen,
                best_fitness: best_f,
                mean_fitness: per_gen_mean[gen],
                worst_fitness: fitness[worst_index(&fitness)],
                best_individual: best.clone(),
            });
        }
        let elites: Vec<I> = elite_indices(&fitness, o.elite)
            .into_iter()
            .map(|i| pop[i].clone())
            .collect();
        let mut next = Vec::with_capacity(o.pop);
        next.extend(elites);
        while next.len() < o.pop {
            next.push(breed_one(&problem, &pop, &fitness, o, rng));
        }
        pop = next;
        fitness = eval_all(&problem, &pop);
        update_best(&mut best, &mut best_f, &pop, &fitness);
    }

    GaResult {
        best,
        best_fitness: best_f,
        per_generation_best: per_gen_best,
        per_generation_mean: per_gen_mean,
        generations: o.gens,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

fn run_steady_state<I: Clone, P: GaProblem<I>>(
    problem: P,
    o: &FilledGaOptions,
    rng: &mut SeededRandom,
    on_generation: &mut Option<Box<dyn FnMut(&GaGenerationInfo<I>)>>,
    t0: Instant,
) -> GaResult<I> {
    let mut pop = problem.initial_population(o.pop, rng);
    let mut fitness = eval_all(&problem, &pop);
    let mut best = pop[0].clone();
    let mut best_f = fitness[0];
    update_best(&mut best, &mut best_f, &pop, &fitness);
    let mut per_gen_best = Vec::with_capacity(o.gens);
    let mut per_gen_mean = Vec::with_capacity(o.gens);

    for gen in 0..o.gens {
        per_gen_best.push(best_f);
        per_gen_mean.push(mean_fitness(&fitness));
        if let Some(cb) = on_generation {
            cb(&GaGenerationInfo {
                generation: gen,
                best_fitness: best_f,
                mean_fitness: per_gen_mean[gen],
                worst_fitness: fitness[worst_index(&fitness)],
                best_individual: best.clone(),
            });
        }
        for _ in 0..o.pop {
            let child = breed_one(&problem, &pop, &fitness, o, rng);
            let f = problem.evaluate(&child);
            let w = worst_index(&fitness);
            if f < fitness[w] {
                pop[w] = child;
                fitness[w] = f;
            }
            update_best(&mut best, &mut best_f, &pop, &fitness);
        }
    }

    GaResult {
        best,
        best_fitness: best_f,
        per_generation_best: per_gen_best,
        per_generation_mean: per_gen_mean,
        generations: o.gens,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

fn run_mu_plus_lambda<I: Clone, P: GaProblem<I>>(
    problem: P,
    o: &FilledGaOptions,
    rng: &mut SeededRandom,
    on_generation: &mut Option<Box<dyn FnMut(&GaGenerationInfo<I>)>>,
    t0: Instant,
) -> GaResult<I> {
    let mu = o.pop;
    let mut pop = problem.initial_population(mu, rng);
    let mut fitness = eval_all(&problem, &pop);
    let mut best = pop[0].clone();
    let mut best_f = fitness[0];
    update_best(&mut best, &mut best_f, &pop, &fitness);
    let mut per_gen_best = Vec::with_capacity(o.gens);
    let mut per_gen_mean = Vec::with_capacity(o.gens);

    for gen in 0..o.gens {
        per_gen_best.push(best_f);
        per_gen_mean.push(mean_fitness(&fitness));
        if let Some(cb) = on_generation {
            cb(&GaGenerationInfo {
                generation: gen,
                best_fitness: best_f,
                mean_fitness: per_gen_mean[gen],
                worst_fitness: fitness[worst_index(&fitness)],
                best_individual: best.clone(),
            });
        }
        let mut pool: Vec<I> = pop.clone();
        let mut all_fit = fitness.clone();
        for _ in 0..o.lambda {
            pool.push(breed_one(&problem, &pop, &fitness, o, rng));
        }
        all_fit.extend(
            pool[mu..]
                .iter()
                .map(|x| problem.evaluate(x))
                .collect::<Vec<_>>(),
        );
        let mut ranked: Vec<usize> = (0..pool.len()).collect();
        ranked.sort_by(|&a, &b| all_fit[a].total_cmp(&all_fit[b]));
        ranked.truncate(mu);
        pop = ranked.iter().map(|&i| pool[i].clone()).collect();
        fitness = ranked.iter().map(|&i| all_fit[i]).collect();
        update_best(&mut best, &mut best_f, &pop, &fitness);
    }

    GaResult {
        best,
        best_fitness: best_f,
        per_generation_best: per_gen_best,
        per_generation_mean: per_gen_mean,
        generations: o.gens,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

fn run_island_ga<I: Clone, P: GaProblem<I>>(
    problem: P,
    o: &FilledGaOptions,
    rng: &mut SeededRandom,
    on_generation: &mut Option<Box<dyn FnMut(&GaGenerationInfo<I>)>>,
    t0: Instant,
) -> GaResult<I> {
    let islands = o.num_islands;
    let sub = (o.pop / islands).max(2);
    let mut demes: Vec<Vec<I>> = (0..islands)
        .map(|_| problem.initial_population(sub, rng))
        .collect();
    let mut deme_fit: Vec<Vec<f64>> = demes.iter().map(|d| eval_all(&problem, d)).collect();
    let mut best = demes[0][0].clone();
    let mut best_f = deme_fit[0][0];
    for (d, fit) in demes.iter().zip(deme_fit.iter()) {
        update_best(&mut best, &mut best_f, d, fit);
    }
    let mut per_gen_best = Vec::with_capacity(o.gens);
    let mut per_gen_mean = Vec::with_capacity(o.gens);
    let sub_opts = FilledGaOptions {
        pop: sub,
        gens: 1,
        elite: o.elite.min(sub),
        ..*o
    };

    for gen in 0..o.gens {
        let all_fit: Vec<f64> = deme_fit.iter().flatten().copied().collect();
        per_gen_best.push(best_f);
        per_gen_mean.push(mean_fitness(&all_fit));
        if let Some(cb) = on_generation {
            cb(&GaGenerationInfo {
                generation: gen,
                best_fitness: best_f,
                mean_fitness: per_gen_mean[gen],
                worst_fitness: all_fit[worst_index(&all_fit)],
                best_individual: best.clone(),
            });
        }
        for k in 0..islands {
            let elites: Vec<I> = elite_indices(&deme_fit[k], sub_opts.elite)
                .into_iter()
                .map(|i| demes[k][i].clone())
                .collect();
            let mut next = elites;
            while next.len() < sub {
                next.push(breed_one(&problem, &demes[k], &deme_fit[k], &sub_opts, rng));
            }
            demes[k] = next;
            deme_fit[k] = eval_all(&problem, &demes[k]);
            update_best(&mut best, &mut best_f, &demes[k], &deme_fit[k]);
        }
        if gen > 0 && gen % o.migration_interval == 0 {
            let mut migrants: Vec<I> = Vec::new();
            for k in 0..islands {
                let ei = elite_indices(&deme_fit[k], 1)[0];
                migrants.push(demes[k][ei].clone());
            }
            for k in 0..islands {
                let donor = (k + 1) % islands;
                let w = worst_index(&deme_fit[k]);
                demes[k][w] = migrants[donor].clone();
                deme_fit[k][w] = problem.evaluate(&demes[k][w]);
            }
        }
    }

    GaResult {
        best,
        best_fitness: best_f,
        per_generation_best: per_gen_best,
        per_generation_mean: per_gen_mean,
        generations: o.gens,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

/// Generic DES station wrapper for GA problems.
///
/// This station uses the shared [`PopulationOptimizer`] template-method base,
/// so one station tick performs one generational GA update. Standalone
/// [`run_ga`] remains the broader flavor runner (`SteadyState`, `MuPlusLambda`,
/// `Island`); this station is the animation/instrumentation-friendly
/// generational form.
pub struct EvolutionGaStation<I: Clone + 'static, P: GaProblem<I> + 'static> {
    core: StationCore,
    state: PopulationState<I>,
    problem: P,
    max_generations: usize,
    tournament_k: usize,
    crossover_prob: f64,
    mutation_prob: f64,
    elite: usize,
    retry_limit: usize,
    pub generation_events: Vec<GaGenerationInfo<I>>,
}

impl<I: Clone + 'static, P: GaProblem<I> + 'static> EvolutionGaStation<I, P> {
    pub fn new(id: impl Into<String>, problem: P, opts: GaOptions) -> Self {
        let o = fill_options(opts);
        EvolutionGaStation {
            core: StationCore::new(id),
            state: PopulationState::new(o.pop, Box::new(SeededRandom::new(o.seed))),
            problem,
            max_generations: o.gens,
            tournament_k: o.tournament_k,
            crossover_prob: o.cx_prob,
            mutation_prob: o.mut_prob,
            elite: o.elite,
            retry_limit: o.retry_limit,
            generation_events: Vec::with_capacity(o.gens + 1),
        }
    }

    fn current_generation_info(&self, generation: usize) -> Option<GaGenerationInfo<I>> {
        let st = self.opt_state();
        let best = st.best.as_ref()?.clone();
        if st.fitness.is_empty() {
            return None;
        }
        let mean = st.fitness.iter().sum::<f64>() / st.fitness.len() as f64;
        let worst = st.fitness.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(GaGenerationInfo {
            generation,
            best_fitness: st.best_fitness,
            mean_fitness: mean,
            worst_fitness: worst,
            best_individual: best,
        })
    }

    pub fn to_ga_result(&self, elapsed_ms: f64) -> GaResult<I> {
        let st = self.opt_state();
        GaResult {
            best: st.best.as_ref().expect("initialized").clone(),
            best_fitness: st.best_fitness,
            per_generation_best: st.best_history.clone(),
            per_generation_mean: st.mean_history.clone(),
            generations: st.generation,
            elapsed_ms,
        }
    }
}

impl<I: Clone + 'static, P: GaProblem<I> + 'static> DESStation for EvolutionGaStation<I, P> {
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

impl<I: Clone + 'static, P: GaProblem<I> + 'static> PopulationOptimizer<I>
    for EvolutionGaStation<I, P>
{
    fn opt_state(&self) -> &PopulationState<I> {
        &self.state
    }

    fn opt_state_mut(&mut self) -> &mut PopulationState<I> {
        &mut self.state
    }

    fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<I> {
        self.problem.initial_population(size, rng)
    }

    fn evaluate(&self, individual: &I) -> f64 {
        self.problem.evaluate(individual)
    }

    fn evaluate_population(&self, population: &[I]) -> Vec<f64> {
        self.problem.evaluate_population(population)
    }

    fn select(&self, pop: &[I], fitness: &[f64], rng: &mut dyn RandomSource) -> Vec<I> {
        let p1 = tournament_pick(pop, fitness, self.tournament_k, rng);
        let p2 = tournament_pick(pop, fitness, self.tournament_k, rng);
        vec![pop[p1].clone(), pop[p2].clone()]
    }

    fn recombine(&self, parents: &[I], rng: &mut dyn RandomSource) -> I {
        if parents.len() < 2 || rng.next_float() >= self.crossover_prob {
            return parents[0].clone();
        }
        self.problem.crossover(&parents[0], &parents[1], rng)
    }

    fn mutate(&self, child: I, rng: &mut dyn RandomSource) -> I {
        let child = if rng.next_float() < self.mutation_prob {
            self.problem.mutate(child, rng)
        } else {
            child
        };
        self.problem.local_search(child)
    }

    fn should_stop(&self, generation: usize) -> bool {
        generation >= self.max_generations
    }

    fn elite_count(&self) -> usize {
        self.elite
    }

    fn accept_child(&self, child: &I) -> bool {
        self.problem.accept_child(child)
    }

    fn child_retry_limit(&self) -> usize {
        self.retry_limit
    }

    fn on_bootstrap(&mut self) {
        if let Some(info) = self.current_generation_info(0) {
            self.generation_events.push(info);
        }
    }

    fn on_generation(&mut self, generation: usize) {
        if let Some(info) = self.current_generation_info(generation) {
            self.generation_events.push(info);
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvolutionGaDesResult<I> {
    pub ga: GaResult<I>,
    pub generation_events: Vec<GaGenerationInfo<I>>,
    pub run: IterativeRunSummary,
}

/// Run a generational GA as a one-station DES model.
pub fn run_ga_as_des<I, P>(problem: P, opts: GaOptions) -> EvolutionGaDesResult<I>
where
    I: Clone + 'static,
    P: GaProblem<I> + 'static,
{
    let t0 = Instant::now();
    let station = Rc::new(RefCell::new(EvolutionGaStation::new(
        "evolution-ga",
        problem,
        opts,
    )));
    station.borrow_mut().bootstrap();
    let station_ref: StationRef = station.clone();
    let run = run_iterative_des(
        vec![station_ref],
        IterativeRunOptions {
            shuffle: false,
            run_validators: false,
            ..Default::default()
        },
    );
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let borrowed = station.borrow();
    EvolutionGaDesResult {
        ga: borrowed.to_ga_result(elapsed_ms),
        generation_events: borrowed.generation_events.clone(),
        run,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::capabilities::RandomSource;

    #[derive(Clone)]
    struct RealVec(Vec<f64>);

    struct Sphere;

    impl PopulationInitializer<RealVec> for Sphere {
        fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<RealVec> {
            (0..size)
                .map(|_| RealVec((0..3).map(|_| rng.next_float() * 4.0 - 2.0).collect()))
                .collect()
        }
    }

    impl FitnessEvaluator<RealVec> for Sphere {
        fn evaluate(&self, individual: &RealVec) -> f64 {
            individual.0.iter().map(|x| x * x).sum()
        }
    }

    impl GeneticOperators<RealVec> for Sphere {
        fn crossover(&self, a: &RealVec, b: &RealVec, rng: &mut dyn RandomSource) -> RealVec {
            RealVec(
                a.0.iter()
                    .zip(&b.0)
                    .map(|(&x, &y)| if rng.next_float() < 0.5 { x } else { y })
                    .collect(),
            )
        }
        fn mutate(&self, mut child: RealVec, rng: &mut dyn RandomSource) -> RealVec {
            let i = (rng.next_float() * child.0.len() as f64).floor() as usize % child.0.len();
            child.0[i] += rng.next_float() * 0.4 - 0.2;
            child
        }
    }

    #[test]
    fn generational_finds_near_origin() {
        let r = run_ga(Sphere, GaOptions::with_defaults(40, 80), None);
        assert!(r.best_fitness < 0.05, "got {}", r.best_fitness);
    }

    #[test]
    fn des_station_records_generation_ticks() {
        let r = run_ga_as_des(Sphere, GaOptions::with_defaults(30, 20));
        assert_eq!(r.ga.generations, 20);
        assert_eq!(r.generation_events.first().unwrap().generation, 0);
        assert_eq!(r.generation_events.last().unwrap().generation, 20);
        assert!(r.ga.best_fitness < 0.2, "got {}", r.ga.best_fitness);
    }
}
