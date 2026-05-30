//! Port of `src/des/general/des-base/population-optimizer.ts`.
//!
//! Template-method base for POPULATION-BASED metaheuristics (genetic algorithm,
//! particle swarm, differential evolution, ant colony, evolution strategies, …)
//! over a generic individual `I`.
//!
//! ## Problem shape
//!
//! Minimise `f(x)` over `x ∈ X` by maintaining a population `P_t = {x_1, …,
//! x_K}` and generating `P_{t+1}` via `parents = SELECT(P_t, fitness)`,
//! `child = RECOMBINE(parents) ∘ MUTATE`, `P_{t+1} = REPLACE(P_t, child)` with
//! optional elitism. The DIFFERENTIATOR among algorithms is which operators are
//! plugged in (crossover vs velocity update vs difference vectors vs pheromone
//! construction).
//!
//! ## Template-method mapping (TS `abstract class` → Rust)
//!
//! `abstract class PopulationOptimizer<I> extends DESStation` had a FINAL
//! `runTimeStep` template method calling abstract hooks (`initialPopulation`,
//! `evaluate`, `select`, `recombine`, `mutate`, `clone`, `shouldStop`) plus
//! optional hooks (`eliteCount`, `onGeneration`, `onFinish`, `acceptChild`,
//! `childRetryLimit`, `onChildRejected`). Since Rust lacks abstract-method
//! inheritance we split it:
//!
//!   * [`PopulationState`] — bookkeeping fields the TS base owned (population,
//!     fitness, generation, best, histories, popSize, injected RNG). A concrete
//!     optimizer EMBEDS one and exposes it via `opt_state()` / `opt_state_mut()`.
//!   * [`PopulationOptimizer`] — the hook trait (`: DESStation`). REQUIRED
//!     methods are the abstract hooks; optional hooks have defaults. The
//!     template method is the PROVIDED [`PopulationOptimizer::generation_step`]
//!     (plus bootstrap helpers and accessors). A concrete optimizer delegates
//!     `DESStation::run_time_step` → `self.generation_step()` and
//!     `DESStation::has_work` → `self.optimizer_has_work()`.
//!
//! `rng: () => number` becomes a boxed
//! [`RandomSource`](crate::des::shared::capabilities::RandomSource) in the
//! state, threaded into the hooks as `&mut dyn RandomSource`. The elitism sort
//! over `(fitness, index)` uses `f64::total_cmp` for NaN-safety. `throw new
//! Error` (double-init, wrong population size, non-finite fitness, >1 seed,
//! read-before-init) maps to `panic!`. `number` → `f64`.

use std::any::Any;
use std::rc::Rc;

use crate::des::general::des_base::station::{AnyToken, DESStation, StationCore};
use crate::des::shared::capabilities::RandomSource;

/// Channel carrying the one-shot initial-population seed token.
pub const POPULATION_INITIAL_CHANNEL: &str = "population-initial";
/// Channel carrying the terminal result snapshot token.
pub const POPULATION_RESULT_CHANNEL: &str = "population-result";

/// Seed token carrying the initial population.
pub struct PopulationInitialToken<I> {
    pub population: Vec<I>,
}

impl<I> PopulationInitialToken<I> {
    pub fn new(population: Vec<I>) -> Self {
        PopulationInitialToken { population }
    }
}

/// Immutable snapshot of optimiser progress at termination.
#[derive(Clone)]
pub struct PopulationResultSnapshot<I> {
    pub best: I,
    pub best_fitness: f64,
    pub population: Vec<I>,
    pub fitness: Vec<f64>,
    pub generation: usize,
}

/// Terminal result token emitted on [`POPULATION_RESULT_CHANNEL`].
pub struct PopulationResultToken<I> {
    pub snapshot: PopulationResultSnapshot<I>,
}

impl<I> PopulationResultToken<I> {
    pub fn new(snapshot: PopulationResultSnapshot<I>) -> Self {
        PopulationResultToken { snapshot }
    }
}

/// Source station that emits a single initial-population token exactly once.
pub struct PopulationSourceStation<I> {
    core: StationCore,
    emitted: bool,
    initial_population: Box<dyn FnMut() -> Vec<I>>,
    validate_initial_population: Box<dyn FnMut(&[I])>,
}

impl<I: 'static> PopulationSourceStation<I> {
    pub const CH_INITIAL_POPULATION: &'static str = POPULATION_INITIAL_CHANNEL;

    pub fn new(
        id: impl Into<String>,
        initial_population: impl FnMut() -> Vec<I> + 'static,
    ) -> Self {
        PopulationSourceStation {
            core: StationCore::new(id),
            emitted: false,
            initial_population: Box::new(initial_population),
            validate_initial_population: Box::new(|_| {}),
        }
    }

    pub fn with_validator(
        id: impl Into<String>,
        initial_population: impl FnMut() -> Vec<I> + 'static,
        validate_initial_population: impl FnMut(&[I]) + 'static,
    ) -> Self {
        PopulationSourceStation {
            core: StationCore::new(id),
            emitted: false,
            initial_population: Box::new(initial_population),
            validate_initial_population: Box::new(validate_initial_population),
        }
    }
}

impl<I: 'static> DESStation for PopulationSourceStation<I> {
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
        let population = (self.initial_population)();
        (self.validate_initial_population)(&population);
        let token: AnyToken = Rc::new(PopulationInitialToken::new(population));
        self.core.emit(token, Self::CH_INITIAL_POPULATION);
        self.emitted = true;
    }
}

/// Sink station that keeps the latest result token.
pub struct PopulationSinkStation<I> {
    core: StationCore,
    pub latest: Option<Rc<PopulationResultToken<I>>>,
}

impl<I: 'static> PopulationSinkStation<I> {
    pub const CH_RESULT: &'static str = POPULATION_RESULT_CHANNEL;

    pub fn new(id: impl Into<String>) -> Self {
        PopulationSinkStation {
            core: StationCore::new(id),
            latest: None,
        }
    }
}

impl<I: 'static> DESStation for PopulationSinkStation<I> {
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
        let tokens = self.core.drain::<PopulationResultToken<I>>(Self::CH_RESULT);
        if let Some(last) = tokens.into_iter().last() {
            self.latest = Some(last);
        }
    }
}

/// Bookkeeping fields owned by the TS `abstract class`, factored into a struct
/// the concrete optimizer embeds. `best!: I` definite-assignment becomes
/// `Option<I>` (two-phase init). The injected RNG is held here.
pub struct PopulationState<I> {
    pub population: Vec<I>,
    /// Fitness, lower is better.
    pub fitness: Vec<f64>,
    pub generation: usize,
    pub best: Option<I>,
    pub best_fitness: f64,
    pub finished: bool,
    pub initialized: bool,
    result_emitted: bool,
    pub best_history: Vec<f64>,
    pub mean_history: Vec<f64>,
    pub worst_history: Vec<f64>,
    pub pop_size: usize,
    pub rng: Option<Box<dyn RandomSource>>,
}

impl<I> PopulationState<I> {
    pub fn new(pop_size: usize, rng: Box<dyn RandomSource>) -> Self {
        PopulationState {
            population: Vec::new(),
            fitness: Vec::new(),
            generation: 0,
            best: None,
            best_fitness: f64::INFINITY,
            finished: false,
            initialized: false,
            result_emitted: false,
            best_history: Vec::new(),
            mean_history: Vec::new(),
            worst_history: Vec::new(),
            pop_size,
            rng: Some(rng),
        }
    }
}

/// The population optimiser hook trait. REQUIRED methods are the TS abstract
/// hooks; optional hooks have default impls. The PROVIDED methods make up the
/// template method and must NOT be overridden by concrete algorithms.
pub trait PopulationOptimizer<I: Clone + 'static>: DESStation {
    const CH_INITIAL_POPULATION: &'static str = POPULATION_INITIAL_CHANNEL;
    const CH_RESULT: &'static str = POPULATION_RESULT_CHANNEL;

    fn opt_state(&self) -> &PopulationState<I>;
    fn opt_state_mut(&mut self) -> &mut PopulationState<I>;

    // ── HOOKS (required) ─────────────────────────────────────────────────────

    fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<I>;
    fn evaluate(&self, individual: &I) -> f64;
    /// Pick parents (≥ 1) for one offspring.
    fn select(&self, pop: &[I], fitness: &[f64], rng: &mut dyn RandomSource) -> Vec<I>;
    /// Combine parents into a child (GA crossover / PSO or DE update / …).
    fn recombine(&self, parents: &[I], rng: &mut dyn RandomSource) -> I;
    /// Apply mutation.
    fn mutate(&self, child: I, rng: &mut dyn RandomSource) -> I;
    /// Deep copy an individual. Defaults to `Clone`.
    fn clone_ind(&self, individual: &I) -> I {
        individual.clone()
    }
    fn should_stop(&self, generation: usize) -> bool;

    // ── HOOKS (optional) ───────────────────────────────────────────────────────

    /// Number of best individuals copied unchanged to the next generation.
    fn elite_count(&self) -> usize {
        0
    }
    fn on_bootstrap(&mut self) {}
    fn on_generation(&mut self, _gen: usize) {}
    fn on_finish(&mut self) {}
    /// Constraint-handling hook: return true to accept a freshly-bred child,
    /// false to retry. Default accepts every child.
    fn accept_child(&self, _child: &I) -> bool {
        true
    }
    /// Maximum breeding attempts per offspring slot. After exhaustion the LAST
    /// attempt is pushed even if `accept_child` returned false (preserving
    /// population size).
    fn child_retry_limit(&self) -> usize {
        1
    }
    /// Instrumentation hook fired when `accept_child` returns false.
    fn on_child_rejected(&mut self, _child: &I, _attempt: usize) {}

    // ── BOOTSTRAP (template helpers) ──────────────────────────────────────────

    /// Seed population + fitness from `initial_population(popSize, rng)`. Call
    /// once after construction.
    fn bootstrap(&mut self) {
        let mut rng = self.opt_state_mut().rng.take().expect("rng already in use");
        let size = self.opt_state().pop_size;
        let pop = self.initial_population(size, &mut *rng);
        self.opt_state_mut().rng = Some(rng);
        self.bootstrap_from_population(&pop);
    }

    /// Source-driven bootstrap from an explicit initial population.
    fn bootstrap_from_population(&mut self, initial_population: &[I]) {
        if self.opt_state().initialized {
            panic!("{}: initial population already supplied", self.id());
        }
        let population: Vec<I> = initial_population
            .iter()
            .map(|x| self.clone_ind(x))
            .collect();
        let pop_size = self.opt_state().pop_size;
        if population.len() != pop_size {
            panic!(
                "initialPopulation returned {} individuals, expected {}",
                population.len(),
                pop_size
            );
        }
        let fitness: Vec<f64> = population.iter().map(|x| self.evaluate(x)).collect();
        for (i, f) in fitness.iter().enumerate() {
            if !f.is_finite() {
                panic!(
                    "{}: initial population fitness[{}] must be finite; got {}",
                    self.id(),
                    i,
                    f
                );
            }
        }
        {
            let st = self.opt_state_mut();
            st.population = population;
            st.fitness = fitness;
        }
        self.record_best();
        self.opt_state_mut().initialized = true;
        self.on_bootstrap();
    }

    // ── TEMPLATE METHOD (do NOT override) ─────────────────────────────────────

    fn generation_step(&mut self) {
        if self.opt_state().finished {
            return;
        }
        if !self.opt_state().initialized {
            let seeds = self
                .core_mut()
                .drain::<PopulationInitialToken<I>>(Self::CH_INITIAL_POPULATION);
            if seeds.is_empty() {
                return;
            }
            if seeds.len() > 1 {
                panic!(
                    "{}: expected exactly one initial-population token, got {}",
                    self.id(),
                    seeds.len()
                );
            }
            let pop = seeds[0].population.clone();
            self.bootstrap_from_population(&pop);
            return;
        }
        if self.core().inbox_size(Self::CH_INITIAL_POPULATION) > 0 {
            panic!(
                "{}: received an initial-population token after initialization",
                self.id()
            );
        }
        let gen = self.opt_state().generation;
        if self.should_stop(gen) {
            self.opt_state_mut().finished = true;
            self.on_finish();
            self.emit_result();
            return;
        }
        let pop_size = self.opt_state().pop_size;
        let mut new_pop: Vec<I> = Vec::new();
        let mut new_fit: Vec<f64> = Vec::new();
        // Elitism — copy best k unchanged.
        let elite_k = self.elite_count().min(pop_size);
        if elite_k > 0 {
            let mut order: Vec<(f64, usize)> = self
                .opt_state()
                .fitness
                .iter()
                .copied()
                .enumerate()
                .map(|(i, f)| (f, i))
                .collect();
            order.sort_by(|a, b| a.0.total_cmp(&b.0));
            for entry in order.iter().take(elite_k) {
                let clone = self.clone_ind(&self.opt_state().population[entry.1]);
                new_pop.push(clone);
                new_fit.push(entry.0);
            }
        }
        let retry_budget = self.child_retry_limit().max(1);
        let mut rng = self.opt_state_mut().rng.take().expect("rng already in use");
        while new_pop.len() < pop_size {
            let mut child: Option<I> = None;
            let mut accepted = false;
            for attempt in 0..retry_budget {
                let parents = self.select(
                    &self.opt_state().population,
                    &self.opt_state().fitness,
                    &mut *rng,
                );
                let c = self.recombine(&parents, &mut *rng);
                let c = self.mutate(c, &mut *rng);
                let ok = self.accept_child(&c);
                child = Some(c);
                if ok {
                    accepted = true;
                    break;
                }
                self.on_child_rejected(child.as_ref().expect("bred"), attempt);
            }
            let child = child.expect("at least one breeding attempt");
            if !accepted {
                self.on_child_rejected(&child, retry_budget);
            }
            let fit = self.evaluate(&child);
            new_pop.push(child);
            new_fit.push(fit);
        }
        self.opt_state_mut().rng = Some(rng);
        {
            let st = self.opt_state_mut();
            st.population = new_pop;
            st.fitness = new_fit;
        }
        self.record_best();
        self.opt_state_mut().generation += 1;
        let gen = self.opt_state().generation;
        self.on_generation(gen);
    }

    fn optimizer_has_work(&self) -> bool {
        self.core().inbox_size(Self::CH_INITIAL_POPULATION) > 0
            || (self.opt_state().initialized && !self.opt_state().finished)
    }

    // ── INTERNALS ──────────────────────────────────────────────────────────────

    fn record_best(&mut self) {
        let (best_idx, best_f, mean, worst) = {
            let st = self.opt_state();
            let mut best_idx = 0usize;
            let mut best_f = st.fitness[0];
            let mut mean = 0.0;
            let mut worst = f64::NEG_INFINITY;
            for (i, &f) in st.fitness.iter().enumerate() {
                mean += f;
                if f < best_f {
                    best_f = f;
                    best_idx = i;
                }
                if f > worst {
                    worst = f;
                }
            }
            mean /= st.fitness.len() as f64;
            (best_idx, best_f, mean, worst)
        };
        if best_f < self.opt_state().best_fitness {
            let b = self.clone_ind(&self.opt_state().population[best_idx]);
            let st = self.opt_state_mut();
            st.best_fitness = best_f;
            st.best = Some(b);
        }
        let st = self.opt_state_mut();
        let bf = st.best_fitness;
        st.best_history.push(bf);
        st.mean_history.push(mean);
        st.worst_history.push(worst);
    }

    // ── PUBLIC ACCESSORS ──────────────────────────────────────────────────────

    fn get_population(&self) -> &[I] {
        self.assert_initialized_for_read();
        &self.opt_state().population
    }
    fn get_fitness(&self) -> &[f64] {
        self.assert_initialized_for_read();
        &self.opt_state().fitness
    }
    fn get_best(&self) -> &I {
        self.assert_initialized_for_read();
        self.opt_state().best.as_ref().expect("initialized")
    }
    fn get_best_fitness(&self) -> f64 {
        self.assert_initialized_for_read();
        self.opt_state().best_fitness
    }
    fn get_generation(&self) -> usize {
        self.opt_state().generation
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
        let best = self.clone_ind(self.opt_state().best.as_ref().expect("initialized"));
        let population: Vec<I> = self
            .opt_state()
            .population
            .iter()
            .map(|x| self.clone_ind(x))
            .collect();
        let snapshot = {
            let st = self.opt_state();
            PopulationResultSnapshot {
                best,
                best_fitness: st.best_fitness,
                population,
                fitness: st.fitness.clone(),
                generation: st.generation,
            }
        };
        let token: AnyToken = Rc::new(PopulationResultToken::new(snapshot));
        self.core_mut().emit(token, Self::CH_RESULT);
        self.opt_state_mut().result_emitted = true;
    }

    fn assert_initialized_for_read(&self) {
        if !self.opt_state().initialized {
            panic!(
                "{}: optimizer has not received an initial population",
                self.id()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::station::StationRef;
    use crate::des::shared::capabilities::SeededRandom;
    use std::cell::RefCell;

    /// Tiny GA over `x: f64` minimising `(x - target)^2`. Tournament-of-2
    /// selection, arithmetic crossover, Gaussian mutation, single elite.
    struct ScalarGa {
        core: StationCore,
        state: PopulationState<f64>,
        target: f64,
        mutation: f64,
        max_gen: usize,
    }

    impl ScalarGa {
        fn new(seed: u32, pop_size: usize, target: f64, mutation: f64, max_gen: usize) -> Self {
            ScalarGa {
                core: StationCore::new("ga"),
                state: PopulationState::new(pop_size, Box::new(SeededRandom::new(seed))),
                target,
                mutation,
                max_gen,
            }
        }
    }

    impl DESStation for ScalarGa {
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

    impl PopulationOptimizer<f64> for ScalarGa {
        fn opt_state(&self) -> &PopulationState<f64> {
            &self.state
        }
        fn opt_state_mut(&mut self) -> &mut PopulationState<f64> {
            &mut self.state
        }
        fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<f64> {
            (0..size).map(|_| (rng.next_float() - 0.5) * 12.0).collect()
        }
        fn evaluate(&self, individual: &f64) -> f64 {
            (individual - self.target).powi(2)
        }
        fn select(&self, pop: &[f64], fitness: &[f64], rng: &mut dyn RandomSource) -> Vec<f64> {
            let pick = |rng: &mut dyn RandomSource| {
                let a = rng.next_int(0, pop.len() as i64) as usize;
                let b = rng.next_int(0, pop.len() as i64) as usize;
                if fitness[a] <= fitness[b] {
                    pop[a]
                } else {
                    pop[b]
                }
            };
            vec![pick(rng), pick(rng)]
        }
        fn recombine(&self, parents: &[f64], _rng: &mut dyn RandomSource) -> f64 {
            parents.iter().sum::<f64>() / parents.len() as f64
        }
        fn mutate(&self, child: f64, rng: &mut dyn RandomSource) -> f64 {
            child + rng.next_gaussian() * self.mutation
        }
        fn should_stop(&self, generation: usize) -> bool {
            generation >= self.max_gen
        }
        fn elite_count(&self) -> usize {
            1
        }
    }

    #[test]
    fn ga_converges_with_elitism() {
        let mut ga = ScalarGa::new(123, 20, 4.0, 0.5, 80);
        ga.bootstrap();
        assert_eq!(ga.get_population().len(), 20);
        let initial_best = ga.get_best_fitness();
        while !ga.is_finished() {
            ga.run_time_step();
        }
        assert!(ga.is_finished());
        assert!(ga.get_best_fitness() <= initial_best);
        assert!(
            ga.get_best_fitness() < 1.0,
            "best_fitness = {}",
            ga.get_best_fitness()
        );
    }

    #[test]
    fn best_fitness_is_monotone_non_increasing() {
        let mut ga = ScalarGa::new(7, 16, 0.0, 0.4, 50);
        ga.bootstrap();
        while !ga.is_finished() {
            ga.run_time_step();
        }
        let hist = &ga.opt_state().best_history;
        for w in hist.windows(2) {
            assert!(
                w[1] <= w[0],
                "best_history not monotone: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn source_optimizer_sink_pipeline() {
        let pop: Vec<f64> = vec![-5.0, -2.0, 1.0, 6.0];
        let source = Rc::new(RefCell::new(PopulationSourceStation::new(
            "src",
            move || pop.clone(),
        )));
        let ga = Rc::new(RefCell::new(ScalarGa::new(55, 4, 2.0, 0.5, 60)));
        let sink = Rc::new(RefCell::new(PopulationSinkStation::<f64>::new("sink")));

        source.borrow_mut().core_mut().pipe(
            ga.clone() as StationRef,
            POPULATION_INITIAL_CHANNEL,
            POPULATION_INITIAL_CHANNEL,
        );
        ga.borrow_mut().core_mut().pipe(
            sink.clone() as StationRef,
            POPULATION_RESULT_CHANNEL,
            POPULATION_RESULT_CHANNEL,
        );

        source.borrow_mut().run_time_step();
        let mut guard = 0;
        while !ga.borrow().is_finished() {
            ga.borrow_mut().run_time_step();
            guard += 1;
            assert!(guard < 10_000, "optimizer did not finish");
        }
        sink.borrow_mut().run_time_step();
        let latest = sink.borrow().latest.clone().expect("result captured");
        assert_eq!(latest.snapshot.generation, 60);
        assert_eq!(latest.snapshot.population.len(), 4);
        assert!(
            latest.snapshot.best_fitness < 1.0,
            "best_fitness = {}",
            latest.snapshot.best_fitness
        );
    }
}
