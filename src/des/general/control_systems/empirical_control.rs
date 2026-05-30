//! Port of `src/des/general/control-systems/empirical-control.ts` — QUANTITATIVE
//! / EMPIRICAL estimation of controllability and observability.
//!
//! `observability-controllability` answers the binary structural question
//! analytically. This module measures the DEGREE — the min and max
//! controllability/observability — two complementary ways:
//!
//! A. Gramians (numerical): the controllability Gramian W_c and observability
//!    Gramian W_o; their eigenvalues are the squared singular values of the
//!    reachability / observability maps, so the smallest eigenvalue is the
//!    hardest direction and the largest is the easiest.
//!
//! B. Trials / simulation (no rank algebra): drive the system with many random
//!    control sequences and inspect the reached cloud's covariance; feed many
//!    random initial states through the noisy output map and reconstruct them by
//!    least squares; run random-policy MDP rollouts; and run Monte-Carlo POMDP
//!    trajectories with Bayesian belief tracking.
//!
//! Everything is types with methods (LinAlg / SymmetricEigen / MatrixInverse do
//! the numerics; a seedable Mulberry32 RNG drives the trials). The Mulberry32
//! bitstream is reproduced with `u32` wrapping arithmetic to match JS exactly.
#![allow(dead_code)]

use std::any::Any;
use std::rc::Rc;

use super::linear_algebra::{LinAlg, Matrix, MatrixInverse, SymmetricEigen, Vector};
use super::observability_controllability::{
    MarkovDecisionProcess, PartiallyObservableProcess, StateSpaceModel,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::des_base::transform_entity::{
    OutputChannel, PureTransformEntity, TransformContext, TransformEntity, TransformEntityCore,
    TransformEntityOptions, TransformResult,
};

fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// Deterministic RNG (seedable) — Mulberry32.
// =============================================================================

/// Mulberry32 PRNG. The `u32` wrapping arithmetic reproduces the JavaScript
/// `>>> 0` / `Math.imul` bitstream exactly, so seeded runs match the TS engine.
#[derive(Clone, Debug)]
pub struct Mulberry32 {
    state: u32,
}

impl Default for Mulberry32 {
    fn default() -> Self {
        Mulberry32 { state: 0x9e37_79b9 }
    }
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Mulberry32 { state: seed }
    }

    /// Uniform in [0, 1).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b_79f5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }

    /// Uniform in [-a, a].
    pub fn uniform(&mut self, a: f64) -> f64 {
        (self.next() * 2.0 - 1.0) * a
    }

    /// Standard normal via Box–Muller.
    pub fn normal(&mut self) -> f64 {
        let mut u = 0.0;
        let mut v = 0.0;
        while u == 0.0 {
            u = self.next();
        }
        while v == 0.0 {
            v = self.next();
        }
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// Sample an index from a pmf.
    pub fn categorical(&mut self, pmf: &[f64]) -> usize {
        let r = self.next();
        let mut acc = 0.0;
        for i in 0..pmf.len() {
            acc += pmf[i];
            if r <= acc {
                return i;
            }
        }
        pmf.len() - 1
    }
}

// =============================================================================
// Discrete linear system  x_{k+1} = Ad x_k + Bd u_k,  y_k = C x_k.
// =============================================================================

#[derive(Clone, Debug)]
pub struct DiscreteLinearSystem {
    pub ad: Matrix,
    pub bd: Matrix,
    pub c: Matrix,
}

impl DiscreteLinearSystem {
    pub fn new(ad: Matrix, bd: Matrix, c: Matrix) -> Self {
        let cls = "DiscreteLinearSystem";
        require(Preconditions::square_matrix(cls, "Ad", &ad));
        require(Preconditions::length_eq(cls, "Bd", &bd, ad.len()));
        require(Preconditions::length_eq(cls, "C[0]", &c[0], ad.len()));
        DiscreteLinearSystem {
            ad: LinAlg::copy(&ad),
            bd: LinAlg::copy(&bd),
            c: LinAlg::copy(&c),
        }
    }

    /// Forward-Euler discretisation of a continuous `StateSpaceModel`.
    pub fn from_continuous(model: &StateSpaceModel, dt: f64) -> Self {
        require(Preconditions::positive("DiscreteLinearSystem", "dt", dt));
        let n = model.state_dim();
        let ad = LinAlg::add(&LinAlg::identity(n), &LinAlg::scale(&model.a, dt));
        let bd = LinAlg::scale(&model.b, dt);
        DiscreteLinearSystem::new(ad, bd, model.c.clone())
    }

    pub fn state_dim(&self) -> usize {
        self.ad.len()
    }
    pub fn input_dim(&self) -> usize {
        LinAlg::cols(&self.bd)
    }
    pub fn output_dim(&self) -> usize {
        self.c.len()
    }

    pub fn step(&self, x: &[f64], u: &[f64]) -> Vector {
        let ax = LinAlg::mat_vec(&self.ad, x);
        let bu = LinAlg::mat_vec(&self.bd, u);
        (0..ax.len()).map(|i| ax[i] + bu[i]).collect()
    }

    /// Roll the state forward under an input sequence; returns terminal state.
    pub fn rollout(&self, x0: &[f64], inputs: &[Vector]) -> Vector {
        let mut x = x0.to_vec();
        for u in inputs {
            x = self.step(&x, u);
        }
        x
    }

    /// Output sequence of length H from x0 (default zero inputs).
    pub fn outputs(&self, x0: &[f64], h: usize, inputs: Option<&[Vector]>) -> Vec<Vector> {
        let mut ys: Vec<Vector> = Vec::new();
        let mut x = x0.to_vec();
        for k in 0..h {
            ys.push(LinAlg::mat_vec(&self.c, &x));
            let u: Vector = match inputs {
                Some(inp) => inp[k].clone(),
                None => vec![0.0; self.input_dim()],
            };
            x = self.step(&x, &u);
        }
        ys
    }

    /// Reachability map R = [ A^{H-1}B | … | AB | B ].
    pub fn reachability_map(&self, h: usize) -> Matrix {
        let mut blocks: Vec<Matrix> = Vec::new();
        for k in (0..h).rev() {
            blocks.push(LinAlg::mat_mul(&LinAlg::power(&self.ad, k), &self.bd));
        }
        LinAlg::hstack(&blocks)
    }

    /// Observability map O = [ C; CA; …; CA^{H-1} ].
    pub fn observability_map(&self, h: usize) -> Matrix {
        let mut blocks: Vec<Matrix> = Vec::new();
        let mut cap = self.c.clone();
        blocks.push(cap.clone());
        for _k in 1..h {
            cap = LinAlg::mat_mul(&cap, &self.ad);
            blocks.push(cap.clone());
        }
        LinAlg::vstack(&blocks)
    }
}

// =============================================================================
// A. GRAMIANS — quantitative controllability / observability degree.
// =============================================================================

/// Shared eigen-summary of a symmetric PSD Gramian. The eigen-decomposition is
/// computed eagerly in the constructor (the TS class held a lazy `SymmetricEigen`
/// whose getters cache; eager computation has identical behaviour and keeps the
/// accessors `&self`).
#[derive(Clone, Debug)]
pub struct GramianDegree {
    gramian: Matrix,
    values: Vector,  // ascending
    vectors: Matrix, // eigenvectors as columns, aligned with `values`
}

impl GramianDegree {
    pub fn new(gramian: Matrix) -> Self {
        let mut eig = SymmetricEigen::new(&gramian, 100);
        let values = eig.values();
        let vectors = eig.vectors();
        GramianDegree { gramian, values, vectors }
    }
    pub fn matrix(&self) -> &Matrix {
        &self.gramian
    }
    pub fn eigenvalues(&self) -> Vector {
        self.values.clone()
    }
    /// Min degree (hardest / weakest direction), clamped at 0 for PSD noise.
    pub fn min(&self) -> f64 {
        (0.0_f64).max(self.values[0])
    }
    /// Max degree (easiest / strongest direction).
    pub fn max(&self) -> f64 {
        self.values[self.values.len() - 1]
    }
    /// Direction (unit eigenvector) that is hardest to drive / see.
    pub fn weakest_direction(&self) -> Vector {
        LinAlg::transpose(&self.vectors)[0].clone()
    }
    /// Direction that is easiest to drive / see.
    pub fn strongest_direction(&self) -> Vector {
        let cols = LinAlg::transpose(&self.vectors);
        cols[cols.len() - 1].clone()
    }
    /// λ_max/λ_min — anisotropy; ∞ when a direction is uncontrollable/unobservable.
    pub fn condition_number(&self) -> f64 {
        let lo = self.min();
        if lo <= 0.0 {
            f64::INFINITY
        } else {
            self.max() / lo
        }
    }
}

/// W_c = Σ_{k=0}^{H-1} Ad^k Bd Bdᵀ (Adᵀ)^k.
#[derive(Clone, Debug)]
pub struct ControllabilityGramian {
    degree: GramianDegree,
}

impl ControllabilityGramian {
    pub fn new(sys: &DiscreteLinearSystem, horizon: usize) -> Self {
        require(Preconditions::integer_in_range(
            "ControllabilityGramian",
            "horizon",
            horizon as f64,
            1.0,
            100000.0,
        ));
        let n = sys.state_dim();
        let mut w = LinAlg::zeros(n, n);
        let mut a_pow = LinAlg::identity(n);
        let bbt = LinAlg::mat_mul(&sys.bd, &LinAlg::transpose(&sys.bd));
        for _k in 0..horizon {
            let term = LinAlg::mat_mul(&LinAlg::mat_mul(&a_pow, &bbt), &LinAlg::transpose(&a_pow));
            w = LinAlg::add(&w, &term);
            a_pow = LinAlg::mat_mul(&a_pow, &sys.ad);
        }
        ControllabilityGramian { degree: GramianDegree::new(w) }
    }

    pub fn matrix(&self) -> &Matrix {
        self.degree.matrix()
    }
    pub fn eigenvalues(&self) -> Vector {
        self.degree.eigenvalues()
    }
    pub fn min(&self) -> f64 {
        self.degree.min()
    }
    pub fn max(&self) -> f64 {
        self.degree.max()
    }
    pub fn weakest_direction(&self) -> Vector {
        self.degree.weakest_direction()
    }
    pub fn strongest_direction(&self) -> Vector {
        self.degree.strongest_direction()
    }
    pub fn condition_number(&self) -> f64 {
        self.degree.condition_number()
    }

    /// Minimum control energy to reach unit-norm state x* in the horizon:
    /// E = x*ᵀ W_c⁻¹ x*. Small λ ⇒ huge energy ⇒ weakly controllable.
    pub fn min_energy_to_reach(&self, target: &[f64]) -> f64 {
        let winv = MatrixInverse::new(&self.regularised(), None).inverse();
        let w = LinAlg::mat_vec(&winv, target);
        let mut e = 0.0;
        for i in 0..target.len() {
            e += target[i] * w[i];
        }
        e
    }

    fn regularised(&self) -> Matrix {
        let n = self.degree.matrix().len();
        LinAlg::add(self.degree.matrix(), &LinAlg::scale(&LinAlg::identity(n), 1e-12))
    }
}

/// W_o = Σ_{k=0}^{H-1} (Adᵀ)^k Cᵀ C Ad^k.
#[derive(Clone, Debug)]
pub struct ObservabilityGramian {
    degree: GramianDegree,
}

impl ObservabilityGramian {
    pub fn new(sys: &DiscreteLinearSystem, horizon: usize) -> Self {
        require(Preconditions::integer_in_range(
            "ObservabilityGramian",
            "horizon",
            horizon as f64,
            1.0,
            100000.0,
        ));
        let n = sys.state_dim();
        let mut w = LinAlg::zeros(n, n);
        let mut a_pow = LinAlg::identity(n);
        let ctc = LinAlg::mat_mul(&LinAlg::transpose(&sys.c), &sys.c);
        for _k in 0..horizon {
            let term = LinAlg::mat_mul(&LinAlg::mat_mul(&LinAlg::transpose(&a_pow), &ctc), &a_pow);
            w = LinAlg::add(&w, &term);
            a_pow = LinAlg::mat_mul(&a_pow, &sys.ad);
        }
        ObservabilityGramian { degree: GramianDegree::new(w) }
    }

    pub fn matrix(&self) -> &Matrix {
        self.degree.matrix()
    }
    pub fn eigenvalues(&self) -> Vector {
        self.degree.eigenvalues()
    }
    pub fn min(&self) -> f64 {
        self.degree.min()
    }
    pub fn max(&self) -> f64 {
        self.degree.max()
    }
    pub fn weakest_direction(&self) -> Vector {
        self.degree.weakest_direction()
    }
    pub fn strongest_direction(&self) -> Vector {
        self.degree.strongest_direction()
    }
    pub fn condition_number(&self) -> f64 {
        self.degree.condition_number()
    }
}

// =============================================================================
// B1. TRIAL-BASED CONTROLLABILITY — random shooting + least-squares targeting.
// =============================================================================

/// Least-squares (minimum-energy) open-loop controller: pick the input stack u
/// that drives x0=0 → target in H steps via the right pseudo-inverse of the
/// reachability map.
pub struct MinEnergyController {
    r: Matrix,       // reachability map (n × H·m)
    rrt_inv: Matrix, // (R Rᵀ + εI)⁻¹  (n × n)
}

impl MinEnergyController {
    pub fn new(sys: &DiscreteLinearSystem, horizon: usize, ridge: f64) -> Self {
        let r = sys.reachability_map(horizon);
        let n = sys.state_dim();
        let rrt = LinAlg::add(
            &LinAlg::mat_mul(&r, &LinAlg::transpose(&r)),
            &LinAlg::scale(&LinAlg::identity(n), ridge),
        );
        let rrt_inv = MatrixInverse::new(&rrt, None).inverse();
        MinEnergyController { r, rrt_inv }
    }

    /// Stacked input u* = Rᵀ (RRᵀ)⁻¹ target.
    pub fn input_for(&self, target: &[f64]) -> Vector {
        LinAlg::mat_vec(&LinAlg::transpose(&self.r), &LinAlg::mat_vec(&self.rrt_inv, target))
    }

    /// The state actually reached by u*.
    pub fn reached_state(&self, target: &[f64]) -> Vector {
        LinAlg::mat_vec(&self.r, &self.input_for(target))
    }

    /// ‖target − reached‖ — zero iff target lies in the controllable subspace.
    pub fn reach_error(&self, target: &[f64]) -> f64 {
        let reached = self.reached_state(target);
        let mut s = 0.0;
        for i in 0..target.len() {
            let d = target[i] - reached[i];
            s += d * d;
        }
        s.sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct MonteCarloControllabilityResult {
    pub trials: usize,
    /// principal-axis variances of the reached cloud (ascending).
    pub spread_eigenvalues: Vector,
    /// fraction of random unit targets reached within tolerance.
    pub target_success_rate: f64,
    /// max ‖x_H‖ observed across random-input rollouts.
    pub reach_radius: f64,
}

/// Options for [`MonteCarloControllability`]; `None` ⇒ TS default.
#[derive(Clone, Debug, Default)]
pub struct MonteCarloControllabilityOpts {
    pub trials: Option<usize>,
    pub input_bound: Option<f64>,
    pub target_radius: Option<f64>,
    pub tol: Option<f64>,
    pub seed: Option<u32>,
}

/// Drives the system with many random input sequences and analyses where the
/// state lands, plus how often a least-squares controller hits random targets.
pub struct MonteCarloControllability<'a> {
    sys: &'a DiscreteLinearSystem,
    horizon: usize,
    opts: MonteCarloControllabilityOpts,
}

impl<'a> MonteCarloControllability<'a> {
    pub fn new(
        sys: &'a DiscreteLinearSystem,
        horizon: usize,
        opts: MonteCarloControllabilityOpts,
    ) -> Self {
        MonteCarloControllability { sys, horizon, opts }
    }

    pub fn run(&self) -> MonteCarloControllabilityResult {
        let trials = self.opts.trials.unwrap_or(2000);
        let u_bound = self.opts.input_bound.unwrap_or(1.0);
        let target_radius = self.opts.target_radius.unwrap_or(1.0);
        let tol = self.opts.tol.unwrap_or(0.05);
        let mut rng = Mulberry32::new(self.opts.seed.unwrap_or(12345));
        let n = self.sys.state_dim();
        let m = self.sys.input_dim();

        // 1. Random-input rollouts → reached-state cloud.
        let mut cloud: Vec<Vector> = Vec::new();
        let mut reach_radius = 0.0_f64;
        for _t in 0..trials {
            let mut inputs: Vec<Vector> = Vec::new();
            for _k in 0..self.horizon {
                inputs.push((0..m).map(|_| rng.uniform(u_bound)).collect());
            }
            let x_t = self.sys.rollout(&vec![0.0; n], &inputs);
            let r = x_t.iter().fold(0.0, |a, v| a + v * v).sqrt();
            cloud.push(x_t);
            if r > reach_radius {
                reach_radius = r;
            }
        }
        let spread = SymmetricEigen::new(&self.covariance(&cloud), 100).values();

        // 2. Least-squares targeting of random unit-direction targets.
        let controller = MinEnergyController::new(self.sys, self.horizon, 1e-9);
        let mut hits = 0usize;
        let probes = trials.min(500);
        for _t in 0..probes {
            let dir: Vector = (0..n).map(|_| rng.normal()).collect();
            let norm = {
                let s = dir.iter().fold(0.0, |a, v| a + v * v).sqrt();
                if s == 0.0 {
                    1.0
                } else {
                    s
                }
            };
            let target: Vector = dir.iter().map(|v| (v / norm) * target_radius).collect();
            if controller.reach_error(&target) <= tol * target_radius {
                hits += 1;
            }
        }

        MonteCarloControllabilityResult {
            trials,
            spread_eigenvalues: spread,
            target_success_rate: hits as f64 / probes as f64,
            reach_radius,
        }
    }

    fn covariance(&self, cloud: &[Vector]) -> Matrix {
        let n = cloud[0].len();
        let len = cloud.len() as f64;
        let mut mean = vec![0.0; n];
        for x in cloud {
            for i in 0..n {
                mean[i] += x[i] / len;
            }
        }
        let mut cov = LinAlg::zeros(n, n);
        for x in cloud {
            for i in 0..n {
                for j in 0..n {
                    cov[i][j] += (x[i] - mean[i]) * (x[j] - mean[j]) / len;
                }
            }
        }
        cov
    }
}

// =============================================================================
// B2. TRIAL-BASED OBSERVABILITY — random states + noisy least-squares recovery.
// =============================================================================

#[derive(Clone, Debug)]
pub struct MonteCarloObservabilityResult {
    pub trials: usize,
    /// mean ‖x0 − x̂0‖ across trials at the given sensor-noise level.
    pub mean_reconstruction_error: f64,
    /// worst per-trial reconstruction error.
    pub worst_reconstruction_error: f64,
    /// observability-Gramian eigenvalues (ascending) for reference.
    pub gramian_eigenvalues: Vector,
}

/// Options for [`MonteCarloObservability`]; `None` ⇒ TS default.
#[derive(Clone, Debug, Default)]
pub struct MonteCarloObservabilityOpts {
    pub trials: Option<usize>,
    pub noise_std: Option<f64>,
    pub state_scale: Option<f64>,
    pub seed: Option<u32>,
}

/// Feeds many random initial states through the (noisy) output map and
/// reconstructs them by least squares.
pub struct MonteCarloObservability<'a> {
    sys: &'a DiscreteLinearSystem,
    horizon: usize,
    opts: MonteCarloObservabilityOpts,
}

impl<'a> MonteCarloObservability<'a> {
    pub fn new(
        sys: &'a DiscreteLinearSystem,
        horizon: usize,
        opts: MonteCarloObservabilityOpts,
    ) -> Self {
        MonteCarloObservability { sys, horizon, opts }
    }

    pub fn run(&self) -> MonteCarloObservabilityResult {
        let trials = self.opts.trials.unwrap_or(1000);
        let noise_std = self.opts.noise_std.unwrap_or(0.01);
        let scale = self.opts.state_scale.unwrap_or(1.0);
        let mut rng = Mulberry32::new(self.opts.seed.unwrap_or(2024));
        let n = self.sys.state_dim();
        let o = self.sys.observability_map(self.horizon);
        // Least-squares reconstruction operator: (OᵀO + εI)⁻¹ Oᵀ.
        let oto = LinAlg::add(
            &LinAlg::mat_mul(&LinAlg::transpose(&o), &o),
            &LinAlg::scale(&LinAlg::identity(n), 1e-9),
        );
        let recon = LinAlg::mat_mul(&MatrixInverse::new(&oto, None).inverse(), &LinAlg::transpose(&o));

        let mut sum = 0.0;
        let mut worst = 0.0_f64;
        for _t in 0..trials {
            let x0: Vector = (0..n).map(|_| rng.normal() * scale).collect();
            let ys = self.sys.outputs(&x0, self.horizon, None);
            let mut stacked: Vector = Vec::new();
            for y in &ys {
                for &yi in y {
                    stacked.push(yi + rng.normal() * noise_std);
                }
            }
            let xhat = LinAlg::mat_vec(&recon, &stacked);
            let mut e = 0.0;
            for i in 0..n {
                let d = x0[i] - xhat[i];
                e += d * d;
            }
            e = e.sqrt();
            sum += e;
            if e > worst {
                worst = e;
            }
        }
        MonteCarloObservabilityResult {
            trials,
            mean_reconstruction_error: sum / trials as f64,
            worst_reconstruction_error: worst,
            gramian_eigenvalues: ObservabilityGramian::new(self.sys, self.horizon).eigenvalues(),
        }
    }
}

// =============================================================================
// B3. MDP CONTROLLABILITY DEGREE — value iteration + random-policy rollouts.
// =============================================================================

/// Options for the random-policy reach estimators; `None` ⇒ TS default.
#[derive(Clone, Debug, Default)]
pub struct RandomPolicyOpts {
    pub episodes: Option<usize>,
    pub horizon: Option<usize>,
    pub seed: Option<u32>,
}

pub struct MdpControllabilityDegree<'a> {
    mdp: &'a MarkovDecisionProcess,
}

impl<'a> MdpControllabilityDegree<'a> {
    pub fn new(mdp: &'a MarkovDecisionProcess) -> Self {
        MdpControllabilityDegree { mdp }
    }

    /// Min expected number of steps to reach `target` from each state under the
    /// best action (Bellman value iteration). Unreachable ⇒ +∞.
    /// (TS defaults: `iters = 1000`, `tol = 1e-9`.)
    pub fn expected_hitting_times(&self, target: usize, iters: usize, tol: f64) -> Vector {
        let n = self.mdp.num_states;
        let mut v = vec![f64::INFINITY; n];
        v[target] = 0.0;
        for _it in 0..iters {
            let mut delta = 0.0_f64;
            let mut next = v.clone();
            for s in 0..n {
                if s == target {
                    continue;
                }
                let mut best = f64::INFINITY;
                for a in 0..self.mdp.num_actions {
                    let mut exp = 1.0;
                    let mut finite = true;
                    for t in 0..n {
                        let p = self.mdp.transition[a][s][t];
                        if p <= 0.0 {
                            continue;
                        }
                        if !v[t].is_finite() {
                            finite = false;
                            break;
                        }
                        exp += p * v[t];
                    }
                    if finite && exp < best {
                        best = exp;
                    }
                }
                next[s] = best;
                if best.is_finite() {
                    delta = delta.max((best - v[s]).abs());
                }
            }
            v = next;
            if delta < tol {
                break;
            }
        }
        v
    }

    /// Empirical reach frequency to `target` within `horizon` steps under a
    /// uniform-random policy.
    pub fn random_policy_reach_rate(&self, target: usize, opts: &RandomPolicyOpts) -> Vector {
        let episodes = opts.episodes.unwrap_or(400);
        let horizon = opts.horizon.unwrap_or(self.mdp.num_states * 4);
        let mut rng = Mulberry32::new(opts.seed.unwrap_or(7));
        let n = self.mdp.num_states;
        let mut rate = vec![0.0; n];
        for s0 in 0..n {
            let mut hits = 0usize;
            for _e in 0..episodes {
                let mut s = s0;
                for _k in 0..horizon {
                    if s == target {
                        hits += 1;
                        break;
                    }
                    let a = (rng.next() * self.mdp.num_actions as f64).floor() as usize;
                    s = rng.categorical(&self.mdp.transition[a][s]);
                }
                if s == target {
                    hits += 1; // counts terminal landing too
                }
            }
            rate[s0] = (1.0_f64).min(hits as f64 / episodes as f64);
        }
        rate
    }

    /// Controllability degree per target = mean reach rate from all sources.
    pub fn per_target_degree(&self, opts: &RandomPolicyOpts) -> Vector {
        let n = self.mdp.num_states;
        let mut deg = vec![0.0; n];
        for t in 0..n {
            let rates = self.random_policy_reach_rate(t, opts);
            deg[t] = rates.iter().sum::<f64>() / n as f64;
        }
        deg
    }
}

// =============================================================================
// B4. POMDP OBSERVABILITY DEGREE — Bayesian belief tracking + Monte-Carlo.
// =============================================================================

/// Maintains a belief (pmf over states) updated by Bayes' rule on
/// (action, observation) pairs.
pub struct BeliefTracker<'a> {
    pomdp: &'a PartiallyObservableProcess,
    belief: Vec<f64>,
}

impl<'a> BeliefTracker<'a> {
    pub fn new(pomdp: &'a PartiallyObservableProcess, prior: Option<Vec<f64>>) -> Self {
        let n = pomdp.mdp.num_states;
        let belief = match prior {
            Some(p) => p,
            None => vec![1.0 / n as f64; n],
        };
        BeliefTracker { pomdp, belief }
    }

    pub fn current(&self) -> Vec<f64> {
        self.belief.clone()
    }

    /// Predict with action a, then correct with observation o.
    pub fn update(&mut self, action: usize, observation: usize) {
        let n = self.pomdp.mdp.num_states;
        let mut predicted = vec![0.0; n];
        for s in 0..n {
            if self.belief[s] == 0.0 {
                continue;
            }
            for t in 0..n {
                predicted[t] += self.belief[s] * self.pomdp.mdp.transition[action][s][t];
            }
        }
        let mut z = 0.0;
        for t in 0..n {
            predicted[t] *= self.pomdp.observation[t][observation];
            z += predicted[t];
        }
        if z > 0.0 {
            for t in 0..n {
                predicted[t] /= z;
            }
        }
        self.belief = predicted;
    }

    /// Shannon entropy of the current belief (bits).
    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for &p in &self.belief {
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    /// Replace the belief directly (the TS code reached into the private field
    /// via an `as unknown` cast; here it is an explicit setter).
    pub fn set_belief(&mut self, belief: Vec<f64>) {
        self.belief = belief;
    }
}

#[derive(Clone, Debug)]
pub struct PomdpObservabilityResult {
    /// per-true-state probability mass assigned to the true state after H steps.
    pub hit_probability: Vector,
    /// per-true-state mean residual belief entropy (bits) after H steps.
    pub residual_entropy: Vector,
    /// overall observability degree in [0,1] (min over states of hit prob).
    pub min_degree: f64,
    pub max_degree: f64,
}

/// Runs many simulated trajectories from each true state, tracks the belief, and
/// measures how concentrated the belief becomes on the true state.
pub struct MonteCarloDistinguishability<'a> {
    pomdp: &'a PartiallyObservableProcess,
}

impl<'a> MonteCarloDistinguishability<'a> {
    pub fn new(pomdp: &'a PartiallyObservableProcess) -> Self {
        MonteCarloDistinguishability { pomdp }
    }

    pub fn run(&self, opts: &RandomPolicyOpts) -> PomdpObservabilityResult {
        let episodes = opts.episodes.unwrap_or(400);
        let horizon = opts.horizon.unwrap_or(self.pomdp.mdp.num_states * 4);
        let mut rng = Mulberry32::new(opts.seed.unwrap_or(99));
        let n = self.pomdp.mdp.num_states;
        let mut hit = vec![0.0; n];
        let mut ent = vec![0.0; n];

        for s0 in 0..n {
            for _e in 0..episodes {
                let mut tracker = BeliefTracker::new(self.pomdp, None);
                let mut s = s0;
                // First observation (no action yet) to seed the belief.
                let mut o = rng.categorical(&self.pomdp.observation[s]);
                self.bayes_observe(&mut tracker, o);
                for _k in 0..horizon {
                    let a = (rng.next() * self.pomdp.mdp.num_actions as f64).floor() as usize;
                    s = rng.categorical(&self.pomdp.mdp.transition[a][s]);
                    o = rng.categorical(&self.pomdp.observation[s]);
                    tracker.update(a, o);
                }
                hit[s0] += tracker.current()[s] / episodes as f64;
                ent[s0] += tracker.entropy() / episodes as f64;
            }
        }
        let min_degree = hit.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_degree = hit.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        PomdpObservabilityResult {
            hit_probability: hit,
            residual_entropy: ent,
            min_degree,
            max_degree,
        }
    }

    fn bayes_observe(&self, tracker: &mut BeliefTracker<'_>, observation: usize) {
        let mut b = tracker.current();
        let n = self.pomdp.mdp.num_states;
        let mut z = 0.0;
        for s in 0..n {
            b[s] *= self.pomdp.observation[s][observation];
            z += b[s];
        }
        if z > 0.0 {
            for s in 0..n {
                b[s] /= z;
            }
        }
        // Re-seed the tracker with the corrected prior.
        tracker.set_belief(b);
    }
}

// =============================================================================
// DES PIPELINE — empirical evaluators as stations.
// =============================================================================

pub struct EmpiricalChannels;

impl EmpiricalChannels {
    pub const SYSTEM: &'static str = "empirical-system";
    pub const MDP: &'static str = "empirical-mdp";
    pub const POMDP: &'static str = "empirical-pomdp";
    pub const REPORT: &'static str = "empirical-report";
}

#[derive(Clone, Debug)]
pub struct DiscreteSystemToken {
    pub label: String,
    pub sys: DiscreteLinearSystem,
    pub horizon: usize,
}

impl DiscreteSystemToken {
    pub fn new(label: String, sys: DiscreteLinearSystem, horizon: usize) -> Self {
        DiscreteSystemToken { label, sys, horizon }
    }
}

#[derive(Clone, Debug)]
pub struct MdpDegreeToken {
    pub label: String,
    pub mdp: MarkovDecisionProcess,
}

impl MdpDegreeToken {
    pub fn new(label: String, mdp: MarkovDecisionProcess) -> Self {
        MdpDegreeToken { label, mdp }
    }
}

#[derive(Clone, Debug)]
pub struct PomdpDegreeToken {
    pub label: String,
    pub pomdp: PartiallyObservableProcess,
}

impl PomdpDegreeToken {
    pub fn new(label: String, pomdp: PartiallyObservableProcess) -> Self {
        PomdpDegreeToken { label, pomdp }
    }
}

/// `type DegreeKind = 'lti-degree' | 'mdp-degree' | 'pomdp-degree'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegreeKind {
    LtiDegree,
    MdpDegree,
    PomdpDegree,
}

/// A quantitative min/max degree report flowing on the report channel.
#[derive(Clone, Debug)]
pub struct DegreeReportToken {
    pub label: String,
    pub kind: DegreeKind,
    pub min_controllability: f64,
    pub max_controllability: f64,
    pub min_observability: f64,
    pub max_observability: f64,
    pub detail: String,
}

impl DegreeReportToken {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        label: String,
        kind: DegreeKind,
        min_controllability: f64,
        max_controllability: f64,
        min_observability: f64,
        max_observability: f64,
        detail: String,
    ) -> Self {
        DegreeReportToken {
            label,
            kind,
            min_controllability,
            max_controllability,
            min_observability,
            max_observability,
            detail,
        }
    }
}

fn fmt_cond(c: f64) -> String {
    if c.is_finite() {
        format!("{c:.1e}")
    } else {
        "∞".to_string()
    }
}

pub struct DiscreteSystemSourceStation {
    core: StationCore,
    items: Vec<DiscreteSystemToken>,
    emitted: bool,
}

impl DiscreteSystemSourceStation {
    pub fn new(id: &str, items: Vec<DiscreteSystemToken>) -> Self {
        DiscreteSystemSourceStation { core: StationCore::new(id), items, emitted: false }
    }
}

impl DESStation for DiscreteSystemSourceStation {
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
        let items = self.items.clone();
        for it in items {
            self.core.emit(Rc::new(it), EmpiricalChannels::SYSTEM);
        }
        self.emitted = true;
    }
}

pub struct MdpDegreeSourceStation {
    core: StationCore,
    items: Vec<MdpDegreeToken>,
    emitted: bool,
}

impl MdpDegreeSourceStation {
    pub fn new(id: &str, items: Vec<MdpDegreeToken>) -> Self {
        MdpDegreeSourceStation { core: StationCore::new(id), items, emitted: false }
    }
}

impl DESStation for MdpDegreeSourceStation {
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
        let items = self.items.clone();
        for it in items {
            self.core.emit(Rc::new(it), EmpiricalChannels::MDP);
        }
        self.emitted = true;
    }
}

pub struct PomdpDegreeSourceStation {
    core: StationCore,
    items: Vec<PomdpDegreeToken>,
    emitted: bool,
}

impl PomdpDegreeSourceStation {
    pub fn new(id: &str, items: Vec<PomdpDegreeToken>) -> Self {
        PomdpDegreeSourceStation { core: StationCore::new(id), items, emitted: false }
    }
}

impl DESStation for PomdpDegreeSourceStation {
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
        let items = self.items.clone();
        for it in items {
            self.core.emit(Rc::new(it), EmpiricalChannels::POMDP);
        }
        self.emitted = true;
    }
}

/// Gramian-based min/max controllability & observability degree for an LTI.
pub struct LtiDegreeEvaluatorStation {
    tcore: TransformEntityCore<DiscreteSystemToken, DegreeReportToken>,
}

impl LtiDegreeEvaluatorStation {
    pub fn new(id: &str) -> Self {
        LtiDegreeEvaluatorStation {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![EmpiricalChannels::SYSTEM.to_string()],
                    output_channel: OutputChannel::Fixed(EmpiricalChannels::REPORT.to_string()),
                    ..Default::default()
                },
            ),
        }
    }
}

impl TransformEntity<DiscreteSystemToken, DegreeReportToken> for LtiDegreeEvaluatorStation {
    fn tcore(&self) -> &TransformEntityCore<DiscreteSystemToken, DegreeReportToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<DiscreteSystemToken, DegreeReportToken> {
        &mut self.tcore
    }
}

impl PureTransformEntity<DiscreteSystemToken, DegreeReportToken> for LtiDegreeEvaluatorStation {
    fn transform(
        &mut self,
        token: &DiscreteSystemToken,
        _ctx: &mut TransformContext<DegreeReportToken>,
    ) -> TransformResult<DegreeReportToken> {
        let wc = ControllabilityGramian::new(&token.sys, token.horizon);
        let wo = ObservabilityGramian::new(&token.sys, token.horizon);
        let detail = format!(
            "W_c λ∈[{:.2e}, {:.2e}] (cond {}); W_o λ∈[{:.2e}, {:.2e}] (cond {})",
            wc.min(),
            wc.max(),
            fmt_cond(wc.condition_number()),
            wo.min(),
            wo.max(),
            fmt_cond(wo.condition_number()),
        );
        TransformResult::One(DegreeReportToken::new(
            token.label.clone(),
            DegreeKind::LtiDegree,
            wc.min(),
            wc.max(),
            wo.min(),
            wo.max(),
            detail,
        ))
    }
}

impl DESStation for LtiDegreeEvaluatorStation {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
    fn has_work(&self) -> bool {
        false
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

/// Random-policy reach degree (controllability) for an MDP.
pub struct MdpDegreeEvaluatorStation {
    tcore: TransformEntityCore<MdpDegreeToken, DegreeReportToken>,
}

impl MdpDegreeEvaluatorStation {
    pub fn new(id: &str) -> Self {
        MdpDegreeEvaluatorStation {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![EmpiricalChannels::MDP.to_string()],
                    output_channel: OutputChannel::Fixed(EmpiricalChannels::REPORT.to_string()),
                    ..Default::default()
                },
            ),
        }
    }
}

impl TransformEntity<MdpDegreeToken, DegreeReportToken> for MdpDegreeEvaluatorStation {
    fn tcore(&self) -> &TransformEntityCore<MdpDegreeToken, DegreeReportToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<MdpDegreeToken, DegreeReportToken> {
        &mut self.tcore
    }
}

impl PureTransformEntity<MdpDegreeToken, DegreeReportToken> for MdpDegreeEvaluatorStation {
    fn transform(
        &mut self,
        token: &MdpDegreeToken,
        _ctx: &mut TransformContext<DegreeReportToken>,
    ) -> TransformResult<DegreeReportToken> {
        let deg = MdpControllabilityDegree::new(&token.mdp).per_target_degree(&RandomPolicyOpts::default());
        let min = deg.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = deg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let detail = format!(
            "random-policy reach degree per target: [{}]",
            deg.iter().map(|d| format!("{d:.2}")).collect::<Vec<String>>().join(", ")
        );
        TransformResult::One(DegreeReportToken::new(
            token.label.clone(),
            DegreeKind::MdpDegree,
            min,
            max,
            f64::NAN,
            f64::NAN,
            detail,
        ))
    }
}

impl DESStation for MdpDegreeEvaluatorStation {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
    fn has_work(&self) -> bool {
        false
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

/// Belief-tracking distinguishability degree (observability) for a POMDP.
pub struct PomdpDegreeEvaluatorStation {
    tcore: TransformEntityCore<PomdpDegreeToken, DegreeReportToken>,
}

impl PomdpDegreeEvaluatorStation {
    pub fn new(id: &str) -> Self {
        PomdpDegreeEvaluatorStation {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![EmpiricalChannels::POMDP.to_string()],
                    output_channel: OutputChannel::Fixed(EmpiricalChannels::REPORT.to_string()),
                    ..Default::default()
                },
            ),
        }
    }
}

impl TransformEntity<PomdpDegreeToken, DegreeReportToken> for PomdpDegreeEvaluatorStation {
    fn tcore(&self) -> &TransformEntityCore<PomdpDegreeToken, DegreeReportToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<PomdpDegreeToken, DegreeReportToken> {
        &mut self.tcore
    }
}

impl PureTransformEntity<PomdpDegreeToken, DegreeReportToken> for PomdpDegreeEvaluatorStation {
    fn transform(
        &mut self,
        token: &PomdpDegreeToken,
        _ctx: &mut TransformContext<DegreeReportToken>,
    ) -> TransformResult<DegreeReportToken> {
        let r = MonteCarloDistinguishability::new(&token.pomdp).run(&RandomPolicyOpts::default());
        let detail = format!(
            "belief hit-prob per state: [{}]; residual entropy: [{}] bits",
            r.hit_probability.iter().map(|d| format!("{d:.2}")).collect::<Vec<String>>().join(", "),
            r.residual_entropy.iter().map(|d| format!("{d:.2}")).collect::<Vec<String>>().join(", "),
        );
        TransformResult::One(DegreeReportToken::new(
            token.label.clone(),
            DegreeKind::PomdpDegree,
            f64::NAN,
            f64::NAN,
            r.min_degree,
            r.max_degree,
            detail,
        ))
    }
}

impl DESStation for PomdpDegreeEvaluatorStation {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
    fn has_work(&self) -> bool {
        false
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

pub struct DegreeReportSinkStation {
    core: StationCore,
    pub reports: Vec<Rc<DegreeReportToken>>,
}

impl DegreeReportSinkStation {
    pub fn new(id: &str) -> Self {
        DegreeReportSinkStation { core: StationCore::new(id), reports: Vec::new() }
    }
}

impl DESStation for DegreeReportSinkStation {
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
        self.core.inbox_size(EmpiricalChannels::REPORT) > 0
    }
    fn run_time_step(&mut self) {
        let drained = self.core.drain::<DegreeReportToken>(EmpiricalChannels::REPORT);
        self.reports.extend(drained);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::control_systems::observability_controllability::{MdpSpec, PomdpSpec};

    #[test]
    fn mulberry32_is_reproducible_and_bounded() {
        let mut a = Mulberry32::new(12345);
        let mut b = Mulberry32::new(12345);
        for _ in 0..1000 {
            let x = a.next();
            assert_eq!(x, b.next());
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn gramian_distinguishes_controllable_from_not() {
        // Controllable: distinct modes driven differently -> min eigenvalue > 0.
        let controllable = DiscreteLinearSystem::new(
            vec![vec![0.5, 0.0], vec![0.0, 0.3]],
            vec![vec![1.0], vec![1.0]],
            vec![vec![1.0, 0.0]],
        );
        let wc = ControllabilityGramian::new(&controllable, 100);
        assert!(wc.min() > 1e-2, "controllable min {}", wc.min());

        // Uncontrollable: identical dynamics, B drives both equally -> rank-1
        // Gramian with a (clamped) ~0 smallest eigenvalue.
        let uncontrollable = DiscreteLinearSystem::new(
            vec![vec![0.5, 0.0], vec![0.0, 0.5]],
            vec![vec![1.0], vec![1.0]],
            vec![vec![1.0, 0.0]],
        );
        let wc2 = ControllabilityGramian::new(&uncontrollable, 100);
        assert!(wc2.min() < 1e-6, "uncontrollable min {}", wc2.min());
        assert!(wc2.max() > 1.0);
    }

    #[test]
    fn min_energy_controller_reaches_controllable_targets() {
        let sys = DiscreteLinearSystem::new(
            vec![vec![0.5, 0.0], vec![0.0, 0.3]],
            vec![vec![1.0], vec![0.0]],
            vec![vec![1.0, 0.0]],
        );
        // Driving both states needs full controllability; here B only hits state
        // 0 so the reachable component of an in-subspace target is recovered.
        let mc = MinEnergyController::new(&sys, 5, 1e-9);
        let err = mc.reach_error(&[1.0, 0.0]);
        assert!(err < 1e-3, "reach error {err}");
    }

    #[test]
    fn belief_tracker_entropy_of_uniform_is_one_bit() {
        let pomdp = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
            num_observations: 2,
            observation: vec![vec![0.7, 0.3], vec![0.3, 0.7]],
        });
        let tracker = BeliefTracker::new(&pomdp, None);
        assert!((tracker.entropy() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mdp_reach_degree_high_for_fully_mixing() {
        let mdp = MarkovDecisionProcess::new(MdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
        });
        let deg = MdpControllabilityDegree::new(&mdp).per_target_degree(&RandomPolicyOpts {
            episodes: Some(100),
            ..Default::default()
        });
        assert!(deg.iter().all(|&d| d > 0.5), "degrees {deg:?}");
    }
}
