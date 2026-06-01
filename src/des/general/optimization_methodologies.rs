//! Methodology support for choosing and running optimization strategies.
//!
//! This module sits above the individual solvers. It records the practical
//! distinction between convex/concave problems, where gradient methods have
//! strong guarantees, and rugged non-convex problems, where sampling and
//! population methods are useful exploration tools.
//!
//! Existing solver modules already provide gradient descent/Newton/BFGS,
//! simulated annealing, and genetic algorithms. This module adds:
//!
//! * a problem-profile recommender for gradient, annealing, Monte Carlo,
//!   replica exchange, and evolutionary search families;
//! * a generic box-constrained Monte Carlo optimizer;
//! * a generic box-constrained replica-exchange optimizer.

use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::transform::Transform;

/// Whether the objective is being minimized or maximized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

impl ObjectiveSense {
    /// Convert an objective value to an energy. Lower energy is always better.
    pub fn energy(self, value: f64) -> f64 {
        match self {
            ObjectiveSense::Minimize => value,
            ObjectiveSense::Maximize => -value,
        }
    }

    /// Compare two raw objective values using this sense.
    pub fn is_better(self, candidate: f64, incumbent: f64) -> bool {
        self.energy(candidate) < self.energy(incumbent)
    }
}

/// Curvature information relevant to first-order guarantees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurvatureClass {
    Convex,
    Concave,
    NonConvex,
    Unknown,
}

/// High-level method families the engine can route users toward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationMethodology {
    GradientBased,
    SimulatedAnnealing,
    MonteCarloSampling,
    ReplicaExchange,
    EvolutionaryAlgorithm,
}

/// How strongly a methodology fits the supplied problem profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationStrength {
    Discouraged,
    Optional,
    Support,
    Primary,
}

/// What kind of result guarantee a method can reasonably claim for a profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuaranteeLevel {
    GlobalOptimumUnderCurvature,
    StationaryPointOnly,
    ProbabilisticExploration,
    HeuristicSearch,
}

/// Minimal problem description used for methodology selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptimizationProblemProfile {
    pub sense: ObjectiveSense,
    pub curvature: CurvatureClass,
    pub differentiable: bool,
    pub gradient_available: bool,
    pub stochastic_objective: bool,
    pub discrete_or_combinatorial: bool,
    pub rugged_landscape: bool,
}

impl Default for OptimizationProblemProfile {
    fn default() -> Self {
        OptimizationProblemProfile {
            sense: ObjectiveSense::Minimize,
            curvature: CurvatureClass::Unknown,
            differentiable: false,
            gradient_available: false,
            stochastic_objective: false,
            discrete_or_combinatorial: false,
            rugged_landscape: false,
        }
    }
}

impl OptimizationProblemProfile {
    /// Canonical smooth convex minimization profile.
    pub fn convex_gradient_minimization() -> Self {
        OptimizationProblemProfile {
            sense: ObjectiveSense::Minimize,
            curvature: CurvatureClass::Convex,
            differentiable: true,
            gradient_available: true,
            stochastic_objective: false,
            discrete_or_combinatorial: false,
            rugged_landscape: false,
        }
    }

    /// A rugged non-convex profile, useful for protein-folding-like landscapes.
    pub fn rugged_nonconvex_minimization() -> Self {
        OptimizationProblemProfile {
            sense: ObjectiveSense::Minimize,
            curvature: CurvatureClass::NonConvex,
            differentiable: true,
            gradient_available: true,
            stochastic_objective: false,
            discrete_or_combinatorial: false,
            rugged_landscape: true,
        }
    }

    /// A combinatorial value-only profile, useful for genetics/search problems.
    pub fn combinatorial_search() -> Self {
        OptimizationProblemProfile {
            sense: ObjectiveSense::Minimize,
            curvature: CurvatureClass::NonConvex,
            differentiable: false,
            gradient_available: false,
            stochastic_objective: false,
            discrete_or_combinatorial: true,
            rugged_landscape: true,
        }
    }

    pub fn has_gradient_global_guarantee(self) -> bool {
        matches!(
            (self.sense, self.curvature),
            (ObjectiveSense::Minimize, CurvatureClass::Convex)
                | (ObjectiveSense::Maximize, CurvatureClass::Concave)
        ) && self.differentiable
            && self.gradient_available
    }

    pub fn needs_exploration(self) -> bool {
        self.rugged_landscape
            || self.discrete_or_combinatorial
            || self.stochastic_objective
            || matches!(
                self.curvature,
                CurvatureClass::NonConvex | CurvatureClass::Unknown
            )
    }
}

/// One methodology recommendation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MethodologyRecommendation {
    pub methodology: OptimizationMethodology,
    pub strength: RecommendationStrength,
    pub guarantee: GuaranteeLevel,
    pub rationale: &'static str,
}

/// Return the methodology fit list for a problem profile.
pub fn recommend_methodologies(
    profile: OptimizationProblemProfile,
) -> Vec<MethodologyRecommendation> {
    let gradient = if !profile.differentiable || !profile.gradient_available {
        MethodologyRecommendation {
            methodology: OptimizationMethodology::GradientBased,
            strength: RecommendationStrength::Discouraged,
            guarantee: GuaranteeLevel::StationaryPointOnly,
            rationale: "Gradient methods require a differentiable objective and usable gradients.",
        }
    } else if profile.has_gradient_global_guarantee() {
        MethodologyRecommendation {
            methodology: OptimizationMethodology::GradientBased,
            strength: RecommendationStrength::Primary,
            guarantee: GuaranteeLevel::GlobalOptimumUnderCurvature,
            rationale: "Convex minimization or concave maximization makes local optima global.",
        }
    } else {
        MethodologyRecommendation {
            methodology: OptimizationMethodology::GradientBased,
            strength: RecommendationStrength::Support,
            guarantee: GuaranteeLevel::StationaryPointOnly,
            rationale: "On non-convex objectives, gradient methods are useful but only certify stationarity.",
        }
    };

    let annealing_strength = if profile.needs_exploration() {
        RecommendationStrength::Primary
    } else {
        RecommendationStrength::Optional
    };
    let monte_carlo_strength = if !profile.gradient_available || profile.stochastic_objective {
        RecommendationStrength::Primary
    } else if profile.needs_exploration() {
        RecommendationStrength::Support
    } else {
        RecommendationStrength::Optional
    };
    let replica_strength = if profile.rugged_landscape
        && matches!(
            profile.curvature,
            CurvatureClass::NonConvex | CurvatureClass::Unknown
        ) {
        RecommendationStrength::Support
    } else {
        RecommendationStrength::Optional
    };
    let evolutionary_strength = if profile.discrete_or_combinatorial || !profile.differentiable {
        RecommendationStrength::Primary
    } else if profile.needs_exploration() {
        RecommendationStrength::Support
    } else {
        RecommendationStrength::Optional
    };

    let mut recs = vec![
        gradient,
        MethodologyRecommendation {
            methodology: OptimizationMethodology::SimulatedAnnealing,
            strength: annealing_strength,
            guarantee: GuaranteeLevel::ProbabilisticExploration,
            rationale: "Temperature-controlled uphill moves help explore rugged landscapes.",
        },
        MethodologyRecommendation {
            methodology: OptimizationMethodology::MonteCarloSampling,
            strength: monte_carlo_strength,
            guarantee: GuaranteeLevel::ProbabilisticExploration,
            rationale: "Random sampling gives a robust baseline when gradients are absent, noisy, or misleading.",
        },
        MethodologyRecommendation {
            methodology: OptimizationMethodology::ReplicaExchange,
            strength: replica_strength,
            guarantee: GuaranteeLevel::ProbabilisticExploration,
            rationale: "Multiple temperatures exchange states so hot chains can cross barriers and cold chains can refine.",
        },
        MethodologyRecommendation {
            methodology: OptimizationMethodology::EvolutionaryAlgorithm,
            strength: evolutionary_strength,
            guarantee: GuaranteeLevel::HeuristicSearch,
            rationale: "Population search handles discrete, discontinuous, and multi-modal objectives.",
        },
    ];
    recs.sort_by(|a, b| b.strength.cmp(&a.strength));
    recs
}

/// Axis-aligned finite bounds for continuous samplers.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxConstraints {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl BoxConstraints {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        BoxConstraints { lower, upper }
    }

    pub fn dimension(&self) -> usize {
        self.lower.len()
    }

    pub fn validate(&self) {
        assert!(
            !self.lower.is_empty(),
            "BoxConstraints requires at least one dimension"
        );
        assert_eq!(
            self.lower.len(),
            self.upper.len(),
            "BoxConstraints lower/upper dimension mismatch"
        );
        for (i, (&lo, &hi)) in self.lower.iter().zip(&self.upper).enumerate() {
            assert!(lo.is_finite(), "lower bound {i} is not finite");
            assert!(hi.is_finite(), "upper bound {i} is not finite");
            assert!(
                lo < hi,
                "lower bound must be < upper bound at dimension {i}"
            );
        }
    }

    fn sample(&self, rng: &mut dyn RandomSource) -> Vec<f64> {
        self.lower
            .iter()
            .zip(&self.upper)
            .map(|(&lo, &hi)| lo + rng.next_float() * (hi - lo))
            .collect()
    }

    fn span(&self, i: usize) -> f64 {
        self.upper[i] - self.lower[i]
    }

    fn clamp_value(&self, i: usize, value: f64) -> f64 {
        value.max(self.lower[i]).min(self.upper[i])
    }
}

/// A scalar objective over a finite box.
pub struct BoxConstrainedProblem<F> {
    pub objective: F,
    pub bounds: BoxConstraints,
    pub sense: ObjectiveSense,
}

impl<F> BoxConstrainedProblem<F> {
    pub fn new(objective: F, bounds: BoxConstraints, sense: ObjectiveSense) -> Self {
        BoxConstrainedProblem {
            objective,
            bounds,
            sense,
        }
    }
}

/// Options for uniform Monte Carlo search.
#[derive(Clone, Debug, PartialEq)]
pub struct MonteCarloOptions {
    pub samples: usize,
    pub seed: u32,
    pub record_trace: bool,
    pub trace_stride: usize,
}

impl Default for MonteCarloOptions {
    fn default() -> Self {
        MonteCarloOptions {
            samples: 10_000,
            seed: 42,
            record_trace: false,
            trace_stride: 1,
        }
    }
}

/// Downsampled Monte Carlo trace.
#[derive(Clone, Debug, PartialEq)]
pub struct MonteCarloTraceEntry {
    pub sample: usize,
    pub value: f64,
    pub best_value: f64,
    pub x: Vec<f64>,
}

/// Final Monte Carlo result.
#[derive(Clone, Debug, PartialEq)]
pub struct MonteCarloResult {
    pub best_x: Vec<f64>,
    pub best_value: f64,
    pub samples: usize,
    pub finite_samples: usize,
    pub trace: Vec<MonteCarloTraceEntry>,
}

/// Uniform random box search. Useful as a baseline or value-only optimizer.
pub struct MonteCarloSearch {
    pub options: MonteCarloOptions,
}

impl MonteCarloSearch {
    pub fn new(options: MonteCarloOptions) -> Self {
        MonteCarloSearch { options }
    }
}

impl Default for MonteCarloSearch {
    fn default() -> Self {
        MonteCarloSearch {
            options: MonteCarloOptions::default(),
        }
    }
}

impl<F> Transform<BoxConstrainedProblem<F>, MonteCarloResult> for MonteCarloSearch
where
    F: Fn(&[f64]) -> f64,
{
    fn transform(&self, problem: BoxConstrainedProblem<F>) -> MonteCarloResult {
        assert!(
            self.options.samples > 0,
            "MonteCarloSearch requires samples > 0"
        );
        problem.bounds.validate();
        let stride = self.options.trace_stride.max(1);
        let mut rng = mulberry32(self.options.seed);
        let mut best_x = Vec::new();
        let mut best_value = f64::NAN;
        let mut best_energy = f64::INFINITY;
        let mut finite_samples = 0;
        let mut trace = Vec::new();

        for i in 0..self.options.samples {
            let x = problem.bounds.sample(&mut rng);
            let value = (problem.objective)(&x);
            if !value.is_finite() {
                continue;
            }
            finite_samples += 1;
            let energy = problem.sense.energy(value);
            if energy < best_energy {
                best_energy = energy;
                best_value = value;
                best_x = x.clone();
            }
            if self.options.record_trace && (i + 1) % stride == 0 {
                trace.push(MonteCarloTraceEntry {
                    sample: i + 1,
                    value,
                    best_value,
                    x,
                });
            }
        }

        assert!(
            !best_x.is_empty(),
            "MonteCarloSearch saw no finite objective values"
        );
        MonteCarloResult {
            best_x,
            best_value,
            samples: self.options.samples,
            finite_samples,
            trace,
        }
    }
}

/// Options for replica-exchange random-walk search.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplicaExchangeOptions {
    pub iterations: usize,
    pub seed: u32,
    pub temperatures: Vec<f64>,
    /// Gaussian proposal scale as a fraction of each coordinate's box width.
    pub step_size: f64,
    pub swap_interval: usize,
    pub record_trace: bool,
    pub trace_stride: usize,
}

impl Default for ReplicaExchangeOptions {
    fn default() -> Self {
        ReplicaExchangeOptions {
            iterations: 2_000,
            seed: 42,
            temperatures: vec![0.1, 0.5, 2.0, 8.0],
            step_size: 0.08,
            swap_interval: 5,
            record_trace: false,
            trace_stride: 10,
        }
    }
}

#[derive(Clone, Debug)]
struct ReplicaChain {
    temperature: f64,
    x: Vec<f64>,
    value: f64,
    energy: f64,
}

/// Downsampled replica-exchange trace.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplicaExchangeTraceEntry {
    pub iteration: usize,
    pub replica: usize,
    pub temperature: f64,
    pub value: f64,
    pub best_value: f64,
}

/// Final replica-exchange result.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplicaExchangeResult {
    pub best_x: Vec<f64>,
    pub best_value: f64,
    pub iterations: usize,
    pub accepted_moves: usize,
    pub proposed_moves: usize,
    pub accepted_swaps: usize,
    pub proposed_swaps: usize,
    pub trace: Vec<ReplicaExchangeTraceEntry>,
}

impl ReplicaExchangeResult {
    pub fn move_acceptance_rate(&self) -> f64 {
        if self.proposed_moves == 0 {
            0.0
        } else {
            self.accepted_moves as f64 / self.proposed_moves as f64
        }
    }

    pub fn swap_acceptance_rate(&self) -> f64 {
        if self.proposed_swaps == 0 {
            0.0
        } else {
            self.accepted_swaps as f64 / self.proposed_swaps as f64
        }
    }
}

/// Replica-exchange Metropolis search over a finite continuous box.
pub struct ReplicaExchangeSearch {
    pub options: ReplicaExchangeOptions,
}

impl ReplicaExchangeSearch {
    pub fn new(options: ReplicaExchangeOptions) -> Self {
        ReplicaExchangeSearch { options }
    }
}

impl Default for ReplicaExchangeSearch {
    fn default() -> Self {
        ReplicaExchangeSearch {
            options: ReplicaExchangeOptions::default(),
        }
    }
}

impl<F> Transform<BoxConstrainedProblem<F>, ReplicaExchangeResult> for ReplicaExchangeSearch
where
    F: Fn(&[f64]) -> f64,
{
    fn transform(&self, problem: BoxConstrainedProblem<F>) -> ReplicaExchangeResult {
        assert!(
            self.options.iterations > 0,
            "ReplicaExchangeSearch requires iterations > 0"
        );
        assert!(
            self.options.step_size > 0.0 && self.options.step_size.is_finite(),
            "ReplicaExchangeSearch requires finite step_size > 0"
        );
        assert!(
            !self.options.temperatures.is_empty(),
            "ReplicaExchangeSearch requires at least one temperature"
        );
        for (i, &t) in self.options.temperatures.iter().enumerate() {
            assert!(
                t > 0.0 && t.is_finite(),
                "temperature {i} must be finite and > 0"
            );
        }
        problem.bounds.validate();

        let stride = self.options.trace_stride.max(1);
        let mut rng = mulberry32(self.options.seed);
        let mut chains = self
            .options
            .temperatures
            .iter()
            .map(|&temperature| {
                let x = finite_initial_point(&problem, &mut rng);
                let value = (problem.objective)(&x);
                ReplicaChain {
                    temperature,
                    energy: problem.sense.energy(value),
                    value,
                    x,
                }
            })
            .collect::<Vec<_>>();

        let mut best_x = chains[0].x.clone();
        let mut best_value = chains[0].value;
        let mut best_energy = chains[0].energy;
        for chain in &chains {
            if chain.energy < best_energy {
                best_energy = chain.energy;
                best_value = chain.value;
                best_x = chain.x.clone();
            }
        }

        let mut accepted_moves = 0;
        let mut proposed_moves = 0;
        let mut accepted_swaps = 0;
        let mut proposed_swaps = 0;
        let mut trace = Vec::new();

        for iter in 0..self.options.iterations {
            for chain in &mut chains {
                let proposal =
                    propose_box_step(&problem.bounds, &chain.x, self.options.step_size, &mut rng);
                let proposal_value = (problem.objective)(&proposal);
                if !proposal_value.is_finite() {
                    continue;
                }
                proposed_moves += 1;
                let proposal_energy = problem.sense.energy(proposal_value);
                let delta = proposal_energy - chain.energy;
                let accept = delta <= 0.0 || rng.next_float() < (-delta / chain.temperature).exp();
                if accept {
                    accepted_moves += 1;
                    chain.x = proposal;
                    chain.value = proposal_value;
                    chain.energy = proposal_energy;
                    if proposal_energy < best_energy {
                        best_energy = proposal_energy;
                        best_value = proposal_value;
                        best_x = chain.x.clone();
                    }
                }
            }

            if self.options.swap_interval > 0
                && (iter + 1) % self.options.swap_interval == 0
                && chains.len() > 1
            {
                for i in 0..(chains.len() - 1) {
                    proposed_swaps += 1;
                    let (left, right) = chains.split_at_mut(i + 1);
                    let a = &mut left[i];
                    let b = &mut right[0];
                    let beta_a = 1.0 / a.temperature;
                    let beta_b = 1.0 / b.temperature;
                    let log_accept = (beta_a - beta_b) * (a.energy - b.energy);
                    if log_accept >= 0.0 || rng.next_float().ln() < log_accept {
                        accepted_swaps += 1;
                        std::mem::swap(&mut a.x, &mut b.x);
                        std::mem::swap(&mut a.value, &mut b.value);
                        std::mem::swap(&mut a.energy, &mut b.energy);
                    }
                }
            }

            if self.options.record_trace && (iter + 1) % stride == 0 {
                for (replica, chain) in chains.iter().enumerate() {
                    trace.push(ReplicaExchangeTraceEntry {
                        iteration: iter + 1,
                        replica,
                        temperature: chain.temperature,
                        value: chain.value,
                        best_value,
                    });
                }
            }
        }

        ReplicaExchangeResult {
            best_x,
            best_value,
            iterations: self.options.iterations,
            accepted_moves,
            proposed_moves,
            accepted_swaps,
            proposed_swaps,
            trace,
        }
    }
}

fn finite_initial_point<F>(
    problem: &BoxConstrainedProblem<F>,
    rng: &mut dyn RandomSource,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    for _ in 0..10_000 {
        let x = problem.bounds.sample(rng);
        if (problem.objective)(&x).is_finite() {
            return x;
        }
    }
    panic!("ReplicaExchangeSearch could not sample a finite initial objective value");
}

fn propose_box_step(
    bounds: &BoxConstraints,
    x: &[f64],
    step_size: f64,
    rng: &mut dyn RandomSource,
) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let step = rng.next_gaussian() * step_size * bounds.span(i);
            bounds.clamp_value(i, v + step)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic(x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2)
    }

    #[test]
    fn convex_profile_recommends_gradient_with_global_guarantee() {
        let recs =
            recommend_methodologies(OptimizationProblemProfile::convex_gradient_minimization());
        let grad = recs
            .iter()
            .find(|r| r.methodology == OptimizationMethodology::GradientBased)
            .unwrap();
        assert_eq!(grad.strength, RecommendationStrength::Primary);
        assert_eq!(grad.guarantee, GuaranteeLevel::GlobalOptimumUnderCurvature);
    }

    #[test]
    fn rugged_nonconvex_profile_recommends_exploration() {
        let recs =
            recommend_methodologies(OptimizationProblemProfile::rugged_nonconvex_minimization());
        let grad = recs
            .iter()
            .find(|r| r.methodology == OptimizationMethodology::GradientBased)
            .unwrap();
        let annealing = recs
            .iter()
            .find(|r| r.methodology == OptimizationMethodology::SimulatedAnnealing)
            .unwrap();
        let replica = recs
            .iter()
            .find(|r| r.methodology == OptimizationMethodology::ReplicaExchange)
            .unwrap();

        assert_eq!(grad.guarantee, GuaranteeLevel::StationaryPointOnly);
        assert_eq!(annealing.strength, RecommendationStrength::Primary);
        assert_eq!(replica.strength, RecommendationStrength::Support);
    }

    #[test]
    fn combinatorial_profile_prefers_evolutionary_and_monte_carlo_over_gradient() {
        let recs = recommend_methodologies(OptimizationProblemProfile::combinatorial_search());
        let gradient = recs
            .iter()
            .find(|r| r.methodology == OptimizationMethodology::GradientBased)
            .unwrap();
        let evolutionary = recs
            .iter()
            .find(|r| r.methodology == OptimizationMethodology::EvolutionaryAlgorithm)
            .unwrap();
        let monte_carlo = recs
            .iter()
            .find(|r| r.methodology == OptimizationMethodology::MonteCarloSampling)
            .unwrap();

        assert_eq!(gradient.strength, RecommendationStrength::Discouraged);
        assert_eq!(evolutionary.strength, RecommendationStrength::Primary);
        assert_eq!(monte_carlo.strength, RecommendationStrength::Primary);
    }

    #[test]
    fn monte_carlo_search_finds_quadratic_basin() {
        let search = MonteCarloSearch::new(MonteCarloOptions {
            samples: 50_000,
            seed: 7,
            record_trace: true,
            trace_stride: 5_000,
        });
        let result = search.transform(BoxConstrainedProblem::new(
            quadratic,
            BoxConstraints::new(vec![-5.0, -5.0], vec![5.0, 5.0]),
            ObjectiveSense::Minimize,
        ));

        assert!(result.best_value < 0.02, "best={}", result.best_value);
        assert_eq!(result.samples, 50_000);
        assert_eq!(result.trace.len(), 10);
    }

    #[test]
    fn replica_exchange_search_runs_swaps_and_improves_rugged_objective() {
        fn rugged(x: &[f64]) -> f64 {
            let v = x[0];
            (v * v - 1.0).powi(2) + 0.05 * (12.0 * v).sin()
        }

        let search = ReplicaExchangeSearch::new(ReplicaExchangeOptions {
            iterations: 1_500,
            seed: 9,
            temperatures: vec![0.05, 0.2, 1.0, 4.0],
            step_size: 0.04,
            swap_interval: 4,
            record_trace: true,
            trace_stride: 100,
        });
        let result = search.transform(BoxConstrainedProblem::new(
            rugged,
            BoxConstraints::new(vec![-2.0], vec![2.0]),
            ObjectiveSense::Minimize,
        ));

        assert!(result.best_value < 0.02, "best={}", result.best_value);
        assert!(result.accepted_moves > 0);
        assert!(result.proposed_swaps > 0);
        assert!(!result.trace.is_empty());
    }
}
