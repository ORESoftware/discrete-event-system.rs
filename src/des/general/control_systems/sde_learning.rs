//! Port of `src/des/general/control-systems/sde-learning.ts` — three
//! machine-learning algorithms for stochastic differential equations, one per
//! classic ML paradigm.
//!
//! ML-1 SYSTEM IDENTIFICATION (`SdeMaximumLikelihoodEstimator`): learn the
//! drift/diffusion parameters of an SDE from one observed sample path by Adam
//! gradient ascent on the Euler–Maruyama transition log-likelihood (supervised /
//! MLE).
//!
//! ML-2 FILTERING / INFERENCE (`EnsembleKalmanFilter` + DES station): track the
//! posterior over the hidden state online from noisy observations using a
//! Monte-Carlo ensemble and the Kalman analysis update (sequential Bayesian
//! estimation).
//!
//! ML-3 GENERATIVE MODELING (`DenoisingDiffusionModel` + tiny `Mlp`): learn the
//! score of a data distribution by denoising score matching and draw new samples
//! by integrating the reverse-time SDE (the DDPM discretisation of the
//! variance-preserving SDE) (generative).
//!
//! The parameter vector θ is unconstrained; positivity is enforced through exp()
//! reparametrisations inside `instantiate`. RNG draws use the seeded `Mulberry32`
//! for bit-reproducibility. `throw` invariants become `panic!`; all numerics are
//! `f64`.
#![allow(dead_code)]

use std::any::Any;
use std::collections::BTreeMap;

use super::empirical_control::Mulberry32;
use super::linear_algebra::{LinAlg, Matrix, MatrixInverse, Vector};
use super::stochastic_sde::{
    EulerMaruyamaIntegrator, GeometricBrownianMotion, OrnsteinUhlenbeck, SdeChannels,
    SdeEstimateToken, SdeObservationToken, SdeSystem,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::des_base::transform_entity::{
    MemoryTransformEntity, OutputChannel, TransformContext, TransformEntity, TransformEntityCore,
    TransformEntityOptions, TransformResult,
};

fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// ML-1. SDE PARAMETER ESTIMATION (maximum likelihood, Adam gradient ascent)
// =============================================================================

/// A parametric SDE family: maps an UNCONSTRAINED parameter vector θ to a
/// concrete `SdeSystem`, so the optimiser can search ℝ^k freely (positivity is
/// enforced through exp() reparametrisations inside `instantiate`).
pub trait ParametricSdeFamily {
    fn name(&self) -> String;
    fn param_dim(&self) -> usize;
    fn initial_guess(&self) -> Vector;
    fn instantiate(&self, theta: &[f64]) -> Box<dyn SdeSystem>;
    /// Human-readable named parameters for reporting.
    fn describe(&self, theta: &[f64]) -> BTreeMap<String, f64>;
}

/// GBM family θ = [μ, log σ].
#[derive(Clone, Copy, Debug, Default)]
pub struct GbmFamily;

impl ParametricSdeFamily for GbmFamily {
    fn name(&self) -> String {
        "GBM".to_string()
    }
    fn param_dim(&self) -> usize {
        2
    }
    fn initial_guess(&self) -> Vector {
        vec![0.0, 0.1_f64.ln()]
    }
    fn instantiate(&self, theta: &[f64]) -> Box<dyn SdeSystem> {
        Box::new(GeometricBrownianMotion::new(theta[0], theta[1].exp()))
    }
    fn describe(&self, theta: &[f64]) -> BTreeMap<String, f64> {
        let mut map = BTreeMap::new();
        map.insert("mu".to_string(), theta[0]);
        map.insert("sigma".to_string(), theta[1].exp());
        map
    }
}

/// OU family θ = [log θ, μ, log σ].
#[derive(Clone, Copy, Debug, Default)]
pub struct OuFamily;

impl ParametricSdeFamily for OuFamily {
    fn name(&self) -> String {
        "OU".to_string()
    }
    fn param_dim(&self) -> usize {
        3
    }
    fn initial_guess(&self) -> Vector {
        vec![0.5_f64.ln(), 0.0, 0.5_f64.ln()]
    }
    fn instantiate(&self, theta: &[f64]) -> Box<dyn SdeSystem> {
        Box::new(OrnsteinUhlenbeck::new(
            theta[0].exp(),
            theta[1],
            theta[2].exp(),
        ))
    }
    fn describe(&self, theta: &[f64]) -> BTreeMap<String, f64> {
        let mut map = BTreeMap::new();
        map.insert("theta".to_string(), theta[0].exp());
        map.insert("mu".to_string(), theta[1]);
        map.insert("sigma".to_string(), theta[2].exp());
        map
    }
}

pub struct MleFitResult {
    pub theta: Vector,
    pub params: BTreeMap<String, f64>,
    pub system: Box<dyn SdeSystem>,
    pub final_neg_log_lik: f64,
    pub iterations: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SdeMleOptions {
    pub iterations: Option<usize>,
    pub learning_rate: Option<f64>,
    pub fd_eps: Option<f64>,
}

/// Maximum-likelihood estimator for a parametric SDE family from one path.
pub struct SdeMaximumLikelihoodEstimator {
    opts: SdeMleOptions,
}

impl SdeMaximumLikelihoodEstimator {
    pub fn new(opts: SdeMleOptions) -> Self {
        SdeMaximumLikelihoodEstimator { opts }
    }

    /// Euler–Maruyama transition negative log-likelihood of `path` under θ.
    pub fn neg_log_likelihood(
        &self,
        family: &dyn ParametricSdeFamily,
        theta: &[f64],
        times: &[f64],
        path: &[Vector],
    ) -> f64 {
        let sys = family.instantiate(theta);
        let n = sys.dimension();
        let mut nll = 0.0;
        for k in 0..path.len().saturating_sub(1) {
            let dt = times[k + 1] - times[k];
            let f = sys.drift(times[k], &path[k]);
            let g = sys.diffusion(times[k], &path[k]);
            // cov = (g gᵀ) dt   (n×n), regularised to stay invertible.
            let mut cov = LinAlg::scale(&LinAlg::mat_mul(&g, &LinAlg::transpose(&g)), dt);
            for i in 0..n {
                cov[i][i] += 1e-12;
            }
            let mut r = vec![0.0; n];
            for i in 0..n {
                r[i] = path[k + 1][i] - (path[k][i] + f[i] * dt);
            }
            let inv = MatrixInverse::new(&cov, None).inverse();
            let cinvr = LinAlg::mat_vec(&inv, &r);
            let mut quad = 0.0;
            for i in 0..n {
                quad += r[i] * cinvr[i];
            }
            nll += 0.5
                * (n as f64 * (2.0 * std::f64::consts::PI).ln()
                    + self.determinant(&cov).max(1e-300).ln()
                    + quad);
        }
        nll
    }

    /// Fit θ by Adam descent on the transition NLL (central-difference grad).
    pub fn fit(
        &self,
        family: &dyn ParametricSdeFamily,
        times: &[f64],
        path: &[Vector],
    ) -> MleFitResult {
        require(Preconditions::non_empty(
            "SdeMaximumLikelihoodEstimator",
            "path",
            path,
        ));
        let iters = self.opts.iterations.unwrap_or(1500);
        let lr = self.opts.learning_rate.unwrap_or(0.05);
        let eps = self.opts.fd_eps.unwrap_or(1e-4);
        let d = family.param_dim();
        let mut theta = family.initial_guess();
        let mut m = vec![0.0; d];
        let mut v = vec![0.0; d];
        let b1 = 0.9;
        let b2 = 0.999;
        let mut last = 0.0;
        for it in 1..=iters {
            let mut grad = vec![0.0; d];
            for j in 0..d {
                let mut tp = theta.clone();
                tp[j] += eps;
                let mut tm = theta.clone();
                tm[j] -= eps;
                grad[j] = (self.neg_log_likelihood(family, &tp, times, path)
                    - self.neg_log_likelihood(family, &tm, times, path))
                    / (2.0 * eps);
            }
            for j in 0..d {
                m[j] = b1 * m[j] + (1.0 - b1) * grad[j];
                v[j] = b2 * v[j] + (1.0 - b2) * grad[j] * grad[j];
                let mhat = m[j] / (1.0 - b1.powi(it as i32));
                let vhat = v[j] / (1.0 - b2.powi(it as i32));
                theta[j] -= lr * mhat / (vhat.sqrt() + 1e-8);
            }
            if it == iters {
                last = self.neg_log_likelihood(family, &theta, times, path);
            }
        }
        let params = family.describe(&theta);
        let system = family.instantiate(&theta);
        MleFitResult {
            theta,
            params,
            system,
            final_neg_log_lik: last,
            iterations: iters,
        }
    }

    fn determinant(&self, m: &Matrix) -> f64 {
        let n = m.len();
        let mut a: Matrix = m.iter().cloned().collect();
        let mut det = 1.0;
        for col in 0..n {
            let mut piv = col;
            for i in (col + 1)..n {
                if a[i][col].abs() > a[piv][col].abs() {
                    piv = i;
                }
            }
            if a[piv][col].abs() < 1e-300 {
                return 0.0;
            }
            if piv != col {
                a.swap(piv, col);
                det = -det;
            }
            det *= a[col][col];
            for i in (col + 1)..n {
                let f = a[i][col] / a[col][col];
                for j in col..n {
                    a[i][j] -= f * a[col][j];
                }
            }
        }
        det
    }
}

// =============================================================================
// ML-2. ENSEMBLE KALMAN FILTER (sequential Bayesian state estimation)
// =============================================================================

pub struct EnkfOptions {
    pub ensemble_size: Option<usize>,
    /// H (p×n)
    pub observation_matrix: Matrix,
    /// diag R (length p)
    pub observation_noise_var: Vector,
    /// x̂₀ (length n)
    pub initial_mean: Vector,
    /// sqrt diag P₀ (length n)
    pub initial_std: Vector,
    pub seed: Option<u32>,
}

/// One filtering output: posterior mean and per-dimension variance.
pub struct EnkfEstimate {
    pub mean: Vector,
    pub variance: Vector,
}

/// Stochastic (perturbed-observation) Ensemble Kalman Filter. The forecast step
/// pushes every ensemble member through the SDE's Euler–Maruyama transition; the
/// analysis step nudges them toward the observation using the ensemble-estimated
/// covariance.
pub struct EnsembleKalmanFilter {
    em: EulerMaruyamaIntegrator,
    rng: Mulberry32,
    ensemble: Vec<Vector>,
    t: f64,
    n_members: usize,
    h: Matrix,
    r: Vector,
    sys: Box<dyn SdeSystem>,
    dt: f64,
}

impl EnsembleKalmanFilter {
    pub fn new(sys: Box<dyn SdeSystem>, dt: f64, opts: EnkfOptions) -> Self {
        require(Preconditions::positive("EnsembleKalmanFilter", "dt", dt));
        let n_members = opts.ensemble_size.unwrap_or(100);
        let h = LinAlg::copy(&opts.observation_matrix);
        let r = opts.observation_noise_var.clone();
        let mut rng = Mulberry32::new(opts.seed.unwrap_or(4242));
        let n = sys.dimension();
        let mut ensemble = Vec::with_capacity(n_members);
        for _ in 0..n_members {
            let mut member = Vec::with_capacity(n);
            for i in 0..n {
                member.push(opts.initial_mean[i] + rng.normal() * opts.initial_std[i]);
            }
            ensemble.push(member);
        }
        EnsembleKalmanFilter {
            em: EulerMaruyamaIntegrator::new(),
            rng,
            ensemble,
            t: 0.0,
            n_members,
            h,
            r,
            sys,
            dt,
        }
    }

    /// Forecast: advance each member one SDE step with independent noise.
    pub fn predict(&mut self) {
        let m = self.sys.noise_dimension();
        let ensemble = std::mem::take(&mut self.ensemble);
        let mut next = Vec::with_capacity(ensemble.len());
        for x in &ensemble {
            let dw = self.em.brownian_increment(m, self.dt, &mut self.rng);
            next.push(self.em.step(&*self.sys, self.t, x, self.dt, &dw));
        }
        self.ensemble = next;
        self.t += self.dt;
    }

    /// Analysis: perturbed-observation Kalman update with the ensemble covariance.
    pub fn update(&mut self, obs: &[f64]) {
        let n = self.sys.dimension();
        let p = LinAlg::rows(&self.h);
        let xbar = self.mean();
        // Anomalies A (n×N).
        let mut a = LinAlg::zeros(n, self.n_members);
        for j in 0..self.n_members {
            for i in 0..n {
                a[i][j] = self.ensemble[j][i] - xbar[i];
            }
        }
        let pf = LinAlg::scale(
            &LinAlg::mat_mul(&a, &LinAlg::transpose(&a)),
            1.0 / (self.n_members - 1) as f64,
        );
        let ht = LinAlg::transpose(&self.h);
        let pf_ht = LinAlg::mat_mul(&pf, &ht);
        let mut s = LinAlg::mat_mul(&self.h, &pf_ht);
        for i in 0..p {
            s[i][i] += self.r[i];
        }
        let k_gain = LinAlg::mat_mul(&pf_ht, &MatrixInverse::new(&s, None).inverse());
        let ensemble = std::mem::take(&mut self.ensemble);
        let mut next = Vec::with_capacity(ensemble.len());
        for x in &ensemble {
            let mut d_perturbed = Vec::with_capacity(p);
            for i in 0..p {
                d_perturbed.push(obs[i] + self.rng.normal() * self.r[i].sqrt());
            }
            let hx = LinAlg::mat_vec(&self.h, x);
            let mut innov = Vec::with_capacity(p);
            for i in 0..p {
                innov.push(d_perturbed[i] - hx[i]);
            }
            let corr = LinAlg::mat_vec(&k_gain, &innov);
            let mut updated = Vec::with_capacity(x.len());
            for i in 0..x.len() {
                updated.push(x[i] + corr[i]);
            }
            next.push(updated);
        }
        self.ensemble = next;
    }

    pub fn mean(&self) -> Vector {
        let n = self.sys.dimension();
        let mut out = vec![0.0; n];
        for x in &self.ensemble {
            for i in 0..n {
                out[i] += x[i] / self.n_members as f64;
            }
        }
        out
    }

    /// Per-dimension posterior variance (diagonal of ensemble covariance).
    pub fn variance(&self) -> Vector {
        let n = self.sys.dimension();
        let xbar = self.mean();
        let mut out = vec![0.0; n];
        for x in &self.ensemble {
            for i in 0..n {
                let d = x[i] - xbar[i];
                out[i] += d * d / (self.n_members - 1) as f64;
            }
        }
        out
    }

    /// One filtering step: forecast then assimilate the observation.
    pub fn step(&mut self, obs: &[f64]) -> EnkfEstimate {
        self.predict();
        self.update(obs);
        EnkfEstimate {
            mean: self.mean(),
            variance: self.variance(),
        }
    }
}

/// Streaming EnKF as a DES station: consumes observation tokens, emits state
/// estimate tokens. One observation per tick → one forecast+analysis.
pub struct EnsembleKalmanFilterStation {
    tcore: TransformEntityCore<SdeObservationToken, SdeEstimateToken>,
    /// The filter is the `MemoryTransformEntity` memory (TS `previous`).
    filter: EnsembleKalmanFilter,
}

impl EnsembleKalmanFilterStation {
    pub fn new(id: &str, filter: EnsembleKalmanFilter) -> Self {
        Self::with_channels(id, filter, SdeChannels::OBSERVATION, SdeChannels::ESTIMATE)
    }

    pub fn with_channels(
        id: &str,
        filter: EnsembleKalmanFilter,
        input_channel: &str,
        output_channel: &str,
    ) -> Self {
        EnsembleKalmanFilterStation {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![input_channel.to_string()],
                    output_channel: OutputChannel::Fixed(output_channel.to_string()),
                    ..Default::default()
                },
            ),
            filter,
        }
    }
}

impl TransformEntity<SdeObservationToken, SdeEstimateToken> for EnsembleKalmanFilterStation {
    fn tcore(&self) -> &TransformEntityCore<SdeObservationToken, SdeEstimateToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<SdeObservationToken, SdeEstimateToken> {
        &mut self.tcore
    }
}

impl MemoryTransformEntity<SdeObservationToken, SdeEstimateToken> for EnsembleKalmanFilterStation {
    fn transform_queued(
        &mut self,
        token: &SdeObservationToken,
        _ctx: &mut TransformContext<SdeEstimateToken>,
    ) -> TransformResult<SdeEstimateToken> {
        let est = self.filter.step(&token.obs);
        TransformResult::One(SdeEstimateToken::new(
            token.time,
            token.step,
            est.mean,
            est.variance,
        ))
    }
}

impl DESStation for EnsembleKalmanFilterStation {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        self.run_queued();
    }
    fn has_work(&self) -> bool {
        self.tcore().has_queued_input()
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

// =============================================================================
// ML-3. SCORE-BASED DIFFUSION MODEL (learned reverse-time SDE)
// =============================================================================

/// A minimal one-hidden-layer tanh MLP with manual backprop (scalar output),
/// trained by SGD on a squared-error target. Used as the noise predictor
/// ε_θ(x, t) of the diffusion model.
pub struct Mlp {
    w1: Matrix,
    b1: Vector,
    w2: Vector,
    b2: f64,
    input_dim: usize,
    hidden: usize,
}

impl Mlp {
    pub fn new(input_dim: usize, hidden: usize, rng: &mut Mulberry32) -> Self {
        let s = 1.0 / (input_dim as f64).sqrt();
        let mut w1 = Vec::with_capacity(hidden);
        for _ in 0..hidden {
            let mut row = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                row.push(rng.normal() * s);
            }
            w1.push(row);
        }
        let b1 = vec![0.0; hidden];
        let mut w2 = Vec::with_capacity(hidden);
        for _ in 0..hidden {
            w2.push(rng.normal() / (hidden as f64).sqrt());
        }
        Mlp {
            w1,
            b1,
            w2,
            b2: 0.0,
            input_dim,
            hidden,
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut out = self.b2;
        for h in 0..self.hidden {
            let mut z = self.b1[h];
            for i in 0..self.input_dim {
                z += self.w1[h][i] * x[i];
            }
            out += self.w2[h] * z.tanh();
        }
        out
    }

    /// Forward + backprop one example for loss ½(out − target)²; SGD update.
    pub fn train_example(&mut self, x: &[f64], target: f64, lr: f64) -> f64 {
        let mut a1 = vec![0.0; self.hidden];
        let mut out = self.b2;
        for h in 0..self.hidden {
            let mut z = self.b1[h];
            for i in 0..self.input_dim {
                z += self.w1[h][i] * x[i];
            }
            a1[h] = z.tanh();
            out += self.w2[h] * a1[h];
        }
        let d_out = out - target;
        for h in 0..self.hidden {
            let dz = d_out * self.w2[h] * (1.0 - a1[h] * a1[h]);
            self.w2[h] -= lr * d_out * a1[h];
            for i in 0..self.input_dim {
                self.w1[h][i] -= lr * dz * x[i];
            }
            self.b1[h] -= lr * dz;
        }
        self.b2 -= lr * d_out;
        0.5 * d_out * d_out
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DiffusionOptions {
    /// T discretisation steps of the VP-SDE
    pub steps: Option<usize>,
    pub beta_min: Option<f64>,
    pub beta_max: Option<f64>,
    pub hidden: Option<usize>,
    pub seed: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DiffusionTrainOptions {
    pub iterations: Option<usize>,
    pub learning_rate: Option<f64>,
}

/// Sample mean / std summary for quick validation against a data distribution.
pub struct SampleSummary {
    pub mean: f64,
    pub std: f64,
}

/// Denoising Diffusion Probabilistic Model on 1-D data — the discrete-time
/// variance-preserving SDE  dx = −½β(t)x dt + √β(t) dW  whose reverse-time
/// process is integrated using a learned noise predictor. Data is standardised
/// internally so the unit-variance prior matches.
pub struct DenoisingDiffusionModel {
    t_steps: usize,
    beta: Vector,
    alpha: Vector,
    alpha_bar: Vector,
    net: Mlp,
    rng: Mulberry32,
    data_mean: f64,
    data_std: f64,
}

impl DenoisingDiffusionModel {
    pub fn new(opts: DiffusionOptions) -> Self {
        let t_steps = opts.steps.unwrap_or(100);
        let b_min = opts.beta_min.unwrap_or(1e-4);
        // βmax chosen so ᾱ_T → ~0 at this (small) T, i.e. the forward process
        // actually reaches the N(0,1) prior the sampler starts from.
        let b_max = opts.beta_max.unwrap_or(0.2);
        let mut rng = Mulberry32::new(opts.seed.unwrap_or(7));
        let net = Mlp::new(2, opts.hidden.unwrap_or(64), &mut rng);
        let mut beta = vec![0.0; t_steps];
        let mut alpha = vec![0.0; t_steps];
        let mut alpha_bar = vec![0.0; t_steps];
        let mut abar = 1.0;
        for t in 0..t_steps {
            beta[t] = b_min + (b_max - b_min) * (t as f64 / (t_steps as f64 - 1.0));
            alpha[t] = 1.0 - beta[t];
            abar *= alpha[t];
            alpha_bar[t] = abar;
        }
        DenoisingDiffusionModel {
            t_steps,
            beta,
            alpha,
            alpha_bar,
            net,
            rng,
            data_mean: 0.0,
            data_std: 1.0,
        }
    }

    /// Train ε_θ(x_t, t/T) to predict the injected noise (denoising score matching).
    pub fn train(&mut self, data: &[f64], opts: DiffusionTrainOptions) -> f64 {
        require(Preconditions::non_empty(
            "DenoisingDiffusionModel",
            "data",
            data,
        ));
        self.data_mean = data.iter().sum::<f64>() / data.len() as f64;
        let var_d = data
            .iter()
            .map(|v| (v - self.data_mean).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        let std_dev = var_d.sqrt();
        self.data_std = if std_dev == 0.0 || std_dev.is_nan() {
            1.0
        } else {
            std_dev
        };
        let std: Vec<f64> = data
            .iter()
            .map(|v| (v - self.data_mean) / self.data_std)
            .collect();
        let iters = opts.iterations.unwrap_or(20_000);
        let lr = opts.learning_rate.unwrap_or(0.01);
        let mut last_loss = 0.0;
        for _ in 0..iters {
            let x0 = std[(self.rng.next() * std.len() as f64).floor() as usize];
            let t = (self.rng.next() * self.t_steps as f64).floor() as usize;
            let z = self.rng.normal();
            let ab = self.alpha_bar[t];
            let xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * z;
            last_loss =
                self.net
                    .train_example(&[xt, (t as f64 + 1.0) / self.t_steps as f64], z, lr);
        }
        last_loss
    }

    /// Draw `count` samples by ancestral reverse-time sampling, then de-standardise.
    pub fn sample(&mut self, count: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut x = self.rng.normal();
            for t in (0..self.t_steps).rev() {
                let eps = self
                    .net
                    .predict(&[x, (t as f64 + 1.0) / self.t_steps as f64]);
                let ab = self.alpha_bar[t];
                let mean =
                    (1.0 / self.alpha[t].sqrt()) * (x - (self.beta[t] / (1.0 - ab).sqrt()) * eps);
                x = if t > 0 {
                    mean + self.beta[t].sqrt() * self.rng.normal()
                } else {
                    mean
                };
            }
            out.push(x * self.data_std + self.data_mean);
        }
        out
    }

    /// Fraction of the original signal still present at the final forward step,
    /// √ᾱ_T. Near 0 means the forward SDE has reached the N(0,1) prior the reverse
    /// sampler starts from.
    pub fn terminal_signal_retention(&self) -> f64 {
        self.alpha_bar[self.t_steps - 1].sqrt()
    }

    /// Number of diffusion (discretised reverse-SDE) steps.
    pub fn num_steps(&self) -> usize {
        self.t_steps
    }

    /// Sample mean / std for quick validation against the data distribution.
    pub fn summarise(samples: &[f64]) -> SampleSummary {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let std =
            (samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
        SampleSummary { mean, std }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::des::general::des_base::station::StationRef;

    #[test]
    fn families_describe_and_instantiate() {
        let gbm = GbmFamily;
        assert_eq!(gbm.name(), "GBM");
        assert_eq!(gbm.param_dim(), 2);
        let p = gbm.describe(&[0.1, 0.2_f64.ln()]);
        assert!((p["mu"] - 0.1).abs() < 1e-12);
        assert!((p["sigma"] - 0.2).abs() < 1e-12);
        let sys = gbm.instantiate(&[0.1, 0.2_f64.ln()]);
        assert_eq!(sys.dimension(), 1);

        let ou = OuFamily;
        assert_eq!(ou.param_dim(), 3);
        let q = ou.describe(&[0.5_f64.ln(), 1.0, 0.3_f64.ln()]);
        assert!((q["theta"] - 0.5).abs() < 1e-12);
        assert!((q["mu"] - 1.0).abs() < 1e-12);
        assert!((q["sigma"] - 0.3).abs() < 1e-12);
    }

    fn gbm_path() -> (Vec<f64>, Vec<Vector>) {
        let gbm = GeometricBrownianMotion::new(0.08, 0.25);
        let em = EulerMaruyamaIntegrator::new();
        let mut rng = Mulberry32::new(123);
        let path = em.simulate(&gbm, &[1.0], 0.01, 400, &mut rng);
        (path.times, path.path)
    }

    #[test]
    fn mle_reduces_negative_log_likelihood() {
        let (times, path) = gbm_path();
        let est = SdeMaximumLikelihoodEstimator::new(SdeMleOptions {
            iterations: Some(400),
            learning_rate: Some(0.05),
            fd_eps: Some(1e-4),
        });
        let family = GbmFamily;
        let nll_initial = est.neg_log_likelihood(&family, &family.initial_guess(), &times, &path);
        let fit = est.fit(&family, &times, &path);
        assert_eq!(fit.iterations, 400);
        assert!(fit.final_neg_log_lik.is_finite());
        assert!(
            fit.final_neg_log_lik <= nll_initial + 1e-6,
            "did not improve NLL"
        );
        // sigma is well-identified from quadratic variation of a single path.
        assert!(
            (fit.params["sigma"] - 0.25).abs() < 0.1,
            "sigma_hat = {}",
            fit.params["sigma"]
        );
    }

    #[test]
    fn enkf_tracks_strong_observations() {
        let sys: Box<dyn SdeSystem> = Box::new(OrnsteinUhlenbeck::new(0.1, 0.0, 0.1));
        let mut filter = EnsembleKalmanFilter::new(
            sys,
            0.1,
            EnkfOptions {
                ensemble_size: Some(200),
                observation_matrix: vec![vec![1.0]],
                observation_noise_var: vec![0.01],
                initial_mean: vec![0.0],
                initial_std: vec![1.0],
                seed: Some(99),
            },
        );
        let mut last = filter.step(&[5.0]);
        for _ in 0..40 {
            last = filter.step(&[5.0]);
        }
        assert!(last.mean[0] > 3.0, "mean = {}", last.mean[0]);
        assert!(last.variance[0] > 0.0);
    }

    struct EstimateSink {
        core: StationCore,
        got: Vec<Rc<SdeEstimateToken>>,
    }
    impl DESStation for EstimateSink {
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
            let d = self.core.drain::<SdeEstimateToken>(SdeChannels::ESTIMATE);
            self.got.extend(d);
        }
    }

    #[test]
    fn enkf_station_emits_estimate_token() {
        let sys: Box<dyn SdeSystem> = Box::new(OrnsteinUhlenbeck::new(0.5, 0.0, 0.2));
        let filter = EnsembleKalmanFilter::new(
            sys,
            0.1,
            EnkfOptions {
                ensemble_size: Some(50),
                observation_matrix: vec![vec![1.0]],
                observation_noise_var: vec![0.05],
                initial_mean: vec![0.0],
                initial_std: vec![1.0],
                seed: Some(1),
            },
        );
        let sink = Rc::new(RefCell::new(EstimateSink {
            core: StationCore::new("est-sink"),
            got: Vec::new(),
        }));
        let mut station = EnsembleKalmanFilterStation::new("enkf", filter);
        station.tcore_mut().station.pipe(
            sink.clone() as StationRef,
            SdeChannels::ESTIMATE,
            SdeChannels::ESTIMATE,
        );
        station.take(
            Rc::new(SdeObservationToken::new(0.1, 1, vec![2.0])),
            SdeChannels::OBSERVATION,
        );
        station.run_time_step();
        sink.borrow_mut().run_time_step();
        let got = &sink.borrow().got;
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].step, 1);
        assert!((got[0].time - 0.1).abs() < 1e-12);
        assert_eq!(got[0].mean.len(), 1);
    }

    #[test]
    fn mlp_learns_a_single_example() {
        let mut rng = Mulberry32::new(3);
        let mut net = Mlp::new(2, 16, &mut rng);
        let x = [0.3, 0.6];
        let first = net.train_example(&x, 0.8, 0.05);
        let mut loss = first;
        for _ in 0..200 {
            loss = net.train_example(&x, 0.8, 0.05);
        }
        assert!(loss < first, "loss did not decrease: {first} -> {loss}");
        assert!((net.predict(&x) - 0.8).abs() < 0.1);
    }

    #[test]
    fn diffusion_builds_schedule_and_samples() {
        let mut model = DenoisingDiffusionModel::new(DiffusionOptions {
            steps: Some(50),
            beta_min: None,
            beta_max: None,
            hidden: Some(16),
            seed: Some(5),
        });
        assert_eq!(model.num_steps(), 50);
        let retention = model.terminal_signal_retention();
        assert!(
            retention > 0.0 && retention < 1.0,
            "retention = {retention}"
        );
        let data: Vec<f64> = (0..200)
            .map(|i| 3.0 + 0.5 * ((i as f64) * 0.1).sin())
            .collect();
        let loss = model.train(
            &data,
            DiffusionTrainOptions {
                iterations: Some(1000),
                learning_rate: Some(0.01),
            },
        );
        assert!(loss.is_finite());
        let samples = model.sample(16);
        assert_eq!(samples.len(), 16);
        assert!(samples.iter().all(|v| v.is_finite()));
        let summary = DenoisingDiffusionModel::summarise(&samples);
        assert!(summary.mean.is_finite() && summary.std.is_finite());
    }
}
