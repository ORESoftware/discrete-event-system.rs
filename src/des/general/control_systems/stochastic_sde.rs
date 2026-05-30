//! Port of `src/des/general/control-systems/stochastic-sde.ts` — stochastic
//! differential equations (Itô SDEs).
//!
//! The model is dX = f(X, t) dt + g(X, t) dW, an Itô SDE whose solution X_t is a
//! random process (not a deterministic function). This file provides the drift /
//! diffusion contract (`SdeSystem`), the canonical fixed-step Euler–Maruyama
//! solver, three concrete systems (geometric Brownian motion / Black–Scholes,
//! the mean-reverting Ornstein–Uhlenbeck process, and a stochastic DC motor with
//! additive process noise), and a self-clocking `SdePlantStation` that streams
//! truth and noisy observation tokens for the online ML estimators.
//!
//! Brownian increments are drawn from the seeded `Mulberry32` RNG (re-used from
//! `empirical_control`) so simulated paths are bit-for-bit reproducible. Vectors
//! and matrices are the `shared::linalg` aliases (not raw `Vec`). `throw`
//! invariants become `panic!`; all numerics are `f64`.
#![allow(dead_code)]

use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;

use super::empirical_control::Mulberry32;
use super::linear_algebra::{LinAlg, Matrix, Vector};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::station::{DESStation, StationCore};

fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// SDE contract + solver
// =============================================================================

/// An Itô SDE  dX = f(X,t) dt + g(X,t) dW  with state dim n and noise dim m.
pub trait SdeSystem {
    /// State dimension n.
    fn dimension(&self) -> usize;
    /// Brownian-motion dimension m.
    fn noise_dimension(&self) -> usize;
    /// Drift f(t, x) — length n.
    fn drift(&self, t: f64, x: &[f64]) -> Vector;
    /// Diffusion g(t, x) — n×m matrix multiplying dW.
    fn diffusion(&self, t: f64, x: &[f64]) -> Matrix;
}

/// One simulated sample path: aligned `times` and `path` of equal length.
pub struct SdePath {
    pub times: Vec<f64>,
    pub path: Vec<Vector>,
}

/// Fixed-step Euler–Maruyama:  x_{k+1} = x_k + f Δt + g √Δt ξ, ξ ~ N(0, I_m).
#[derive(Clone, Copy, Debug, Default)]
pub struct EulerMaruyamaIntegrator;

impl EulerMaruyamaIntegrator {
    pub fn new() -> Self {
        EulerMaruyamaIntegrator
    }

    /// One step given a pre-drawn Brownian increment dW (length m).
    pub fn step(&self, sys: &dyn SdeSystem, t: f64, x: &[f64], dt: f64, dw: &[f64]) -> Vector {
        let f = sys.drift(t, x);
        let g = sys.diffusion(t, x);
        let gdw = LinAlg::mat_vec(&g, dw);
        let mut out = vec![0.0; x.len()];
        for i in 0..x.len() {
            out[i] = x[i] + f[i] * dt + gdw[i];
        }
        out
    }

    /// Draw a Brownian increment dW = √Δt · ξ.
    pub fn brownian_increment(&self, m: usize, dt: f64, rng: &mut Mulberry32) -> Vector {
        let s = dt.sqrt();
        (0..m).map(|_| rng.normal() * s).collect()
    }

    /// Simulate one sample path; returns times[0..=steps] and path[0..=steps].
    pub fn simulate(
        &self,
        sys: &dyn SdeSystem,
        x0: &[f64],
        dt: f64,
        steps: usize,
        rng: &mut Mulberry32,
    ) -> SdePath {
        require(Preconditions::positive("EulerMaruyamaIntegrator", "dt", dt));
        let mut times = vec![0.0];
        let mut path: Vec<Vector> = vec![x0.to_vec()];
        let mut x = x0.to_vec();
        for k in 0..steps {
            let dw = self.brownian_increment(sys.noise_dimension(), dt, rng);
            x = self.step(sys, k as f64 * dt, &x, dt, &dw);
            times.push((k + 1) as f64 * dt);
            path.push(x.clone());
        }
        SdePath { times, path }
    }
}

// =============================================================================
// Concrete systems
// =============================================================================

/// dX = μ X dt + σ X dW — geometric Brownian motion (Black–Scholes asset).
#[derive(Clone, Debug)]
pub struct GeometricBrownianMotion {
    pub mu: f64,
    pub sigma: f64,
}

impl GeometricBrownianMotion {
    pub fn new(mu: f64, sigma: f64) -> Self {
        require(Preconditions::finite("GeometricBrownianMotion", "mu", mu));
        require(Preconditions::non_negative(
            "GeometricBrownianMotion",
            "sigma",
            sigma,
        ));
        GeometricBrownianMotion { mu, sigma }
    }

    /// Closed-form solution X_t = X_0 exp((μ−σ²/2)t + σ W_t).
    pub fn exact(&self, x0: f64, t: f64, wt: f64) -> f64 {
        x0 * ((self.mu - 0.5 * self.sigma * self.sigma) * t + self.sigma * wt).exp()
    }
    /// E[X_t] = X_0 e^{μt}.
    pub fn mean_at(&self, x0: f64, t: f64) -> f64 {
        x0 * (self.mu * t).exp()
    }
    /// Var[X_t] = X_0² e^{2μt}(e^{σ²t} − 1).
    pub fn var_at(&self, x0: f64, t: f64) -> f64 {
        x0 * x0 * (2.0 * self.mu * t).exp() * ((self.sigma * self.sigma * t).exp() - 1.0)
    }
}

impl SdeSystem for GeometricBrownianMotion {
    fn dimension(&self) -> usize {
        1
    }
    fn noise_dimension(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64]) -> Vector {
        vec![self.mu * x[0]]
    }
    fn diffusion(&self, _t: f64, x: &[f64]) -> Matrix {
        vec![vec![self.sigma * x[0]]]
    }
}

/// dX = θ(μ − X) dt + σ dW — Ornstein–Uhlenbeck (mean-reverting).
#[derive(Clone, Debug)]
pub struct OrnsteinUhlenbeck {
    pub theta: f64,
    pub mu: f64,
    pub sigma: f64,
}

impl OrnsteinUhlenbeck {
    pub fn new(theta: f64, mu: f64, sigma: f64) -> Self {
        require(Preconditions::positive("OrnsteinUhlenbeck", "theta", theta));
        require(Preconditions::non_negative(
            "OrnsteinUhlenbeck",
            "sigma",
            sigma,
        ));
        OrnsteinUhlenbeck { theta, mu, sigma }
    }

    /// Stationary mean of N(μ, σ²/(2θ)).
    pub fn stationary_mean(&self) -> f64 {
        self.mu
    }
    /// Stationary variance σ²/(2θ).
    pub fn stationary_variance(&self) -> f64 {
        (self.sigma * self.sigma) / (2.0 * self.theta)
    }
}

impl SdeSystem for OrnsteinUhlenbeck {
    fn dimension(&self) -> usize {
        1
    }
    fn noise_dimension(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64]) -> Vector {
        vec![self.theta * (self.mu - x[0])]
    }
    fn diffusion(&self, _t: f64, _x: &[f64]) -> Matrix {
        vec![vec![self.sigma]]
    }
}

#[derive(Clone, Debug)]
pub struct StochasticDcMotorSpec {
    pub resistance: f64,
    pub inductance: f64,
    pub back_emf_constant: f64,
    pub torque_constant: f64,
    pub inertia: f64,
    pub friction: f64,
    pub voltage: f64,
    pub load_torque: Option<f64>,
    pub current_noise: f64,
    pub speed_noise: f64,
}

/// Stochastic DC motor: the deterministic [i, ω] ODE plus additive process noise
/// on each state — di = (V−Ri−K_eω)/L dt + σ_i dW₁,
/// dω = (K_t i−Bω−T_L)/J dt + σ_ω dW₂.
pub struct StochasticDcMotor {
    p: StochasticDcMotorSpec,
}

impl StochasticDcMotor {
    pub fn new(p: StochasticDcMotorSpec) -> Self {
        let cls = "StochasticDcMotor";
        require(Preconditions::positive(cls, "inductance", p.inductance));
        require(Preconditions::positive(cls, "inertia", p.inertia));
        StochasticDcMotor { p }
    }
}

impl SdeSystem for StochasticDcMotor {
    fn dimension(&self) -> usize {
        2
    }
    fn noise_dimension(&self) -> usize {
        2
    }
    fn drift(&self, _t: f64, x: &[f64]) -> Vector {
        let i = x[0];
        let w = x[1];
        let p = &self.p;
        vec![
            (p.voltage - p.resistance * i - p.back_emf_constant * w) / p.inductance,
            (p.torque_constant * i - p.friction * w - p.load_torque.unwrap_or(0.0)) / p.inertia,
        ]
    }
    fn diffusion(&self, _t: f64, _x: &[f64]) -> Matrix {
        vec![
            vec![self.p.current_noise, 0.0],
            vec![0.0, self.p.speed_noise],
        ]
    }
}

// =============================================================================
// DES pipeline — a streaming SDE plant.
// =============================================================================

pub struct SdeChannels;

impl SdeChannels {
    pub const STATE: &'static str = "sde-state";
    pub const OBSERVATION: &'static str = "sde-observation";
    pub const ESTIMATE: &'static str = "sde-estimate";
}

#[derive(Clone, Debug)]
pub struct SdeStateToken {
    pub time: f64,
    pub step: usize,
    pub state: Vector,
}

impl SdeStateToken {
    pub fn new(time: f64, step: usize, state: Vector) -> Self {
        SdeStateToken { time, step, state }
    }
}

#[derive(Clone, Debug)]
pub struct SdeObservationToken {
    pub time: f64,
    pub step: usize,
    pub obs: Vector,
}

impl SdeObservationToken {
    pub fn new(time: f64, step: usize, obs: Vector) -> Self {
        SdeObservationToken { time, step, obs }
    }
}

#[derive(Clone, Debug)]
pub struct SdeEstimateToken {
    pub time: f64,
    pub step: usize,
    pub mean: Vector,
    pub variance: Vector,
}

impl SdeEstimateToken {
    pub fn new(time: f64, step: usize, mean: Vector, variance: Vector) -> Self {
        SdeEstimateToken {
            time,
            step,
            mean,
            variance,
        }
    }
}

pub struct SdePlantOptions {
    pub system: Box<dyn SdeSystem>,
    pub x0: Vector,
    pub dt: f64,
    pub steps: usize,
    /// observation matrix H (p×n); default = identity (observe full state).
    pub observation_matrix: Option<Matrix>,
    /// per-observation-channel measurement-noise std; default 0.
    pub observation_noise_std: Option<Vector>,
    pub seed: Option<u32>,
}

/// Self-clocking plant: each tick advances the SDE one Euler–Maruyama step and
/// emits the true state plus a noisy observation y = H x + v.
pub struct SdePlantStation {
    core: StationCore,
    em: EulerMaruyamaIntegrator,
    rng: Mulberry32,
    h: Matrix,
    obs_noise: Vector,
    x: Vector,
    k: usize,
    system: Box<dyn SdeSystem>,
    dt: f64,
    steps: usize,
    pub true_states: Vec<SdeStateToken>,
    pub observations: Vec<SdeObservationToken>,
}

impl SdePlantStation {
    pub fn new(id: &str, opts: SdePlantOptions) -> Self {
        require(Preconditions::positive("SdePlantStation", "dt", opts.dt));
        require(Preconditions::integer_in_range(
            "SdePlantStation",
            "steps",
            opts.steps as f64,
            1.0,
            10_000_000.0,
        ));
        let rng = Mulberry32::new(opts.seed.unwrap_or(20_260_529));
        let n = opts.system.dimension();
        let h = opts
            .observation_matrix
            .unwrap_or_else(|| LinAlg::identity(n));
        let obs_noise = opts
            .observation_noise_std
            .unwrap_or_else(|| vec![0.0; LinAlg::rows(&h)]);
        let x = opts.x0.clone();
        SdePlantStation {
            core: StationCore::new(id),
            em: EulerMaruyamaIntegrator::new(),
            rng,
            h,
            obs_noise,
            x,
            k: 0,
            system: opts.system,
            dt: opts.dt,
            steps: opts.steps,
            true_states: Vec::new(),
            observations: Vec::new(),
        }
    }
}

impl DESStation for SdePlantStation {
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
        self.k < self.steps
    }
    fn run_time_step(&mut self) {
        if self.k >= self.steps {
            return;
        }
        let t = self.k as f64 * self.dt;
        let m = self.system.noise_dimension();
        let dw = self.em.brownian_increment(m, self.dt, &mut self.rng);
        self.x = self.em.step(&*self.system, t, &self.x, self.dt, &dw);
        self.k += 1;
        let t_now = self.k as f64 * self.dt;
        let state_tok = SdeStateToken::new(t_now, self.k, self.x.clone());
        let y_clean = LinAlg::mat_vec(&self.h, &self.x);
        let mut y: Vector = Vec::with_capacity(y_clean.len());
        for (i, v) in y_clean.iter().enumerate() {
            y.push(v + self.rng.normal() * self.obs_noise[i]);
        }
        let obs_tok = SdeObservationToken::new(t_now, self.k, y);
        self.true_states.push(state_tok.clone());
        self.observations.push(obs_tok.clone());
        self.core.emit(Rc::new(state_tok), SdeChannels::STATE);
        self.core.emit(Rc::new(obs_tok), SdeChannels::OBSERVATION);
    }
}

/// Collects truth + estimate tokens and reports filtering accuracy.
pub struct SdeEstimateSinkStation {
    core: StationCore,
    pub estimates: Vec<Rc<SdeEstimateToken>>,
    pub truth: Vec<Rc<SdeStateToken>>,
}

impl SdeEstimateSinkStation {
    pub fn new(id: &str) -> Self {
        SdeEstimateSinkStation {
            core: StationCore::new(id),
            estimates: Vec::new(),
            truth: Vec::new(),
        }
    }

    /// Per-state-dimension RMSE between estimate.mean and the aligned truth.
    pub fn rmse_by_dimension(&self) -> Vector {
        let mut by_step: HashMap<usize, Rc<SdeStateToken>> = HashMap::new();
        for t in &self.truth {
            by_step.insert(t.step, t.clone());
        }
        let n = if self.truth.is_empty() {
            0
        } else {
            self.truth[0].state.len()
        };
        let mut sse = vec![0.0; n];
        let mut count = 0usize;
        for e in &self.estimates {
            let t = match by_step.get(&e.step) {
                Some(t) => t,
                None => continue,
            };
            count += 1;
            for i in 0..n {
                let d = e.mean[i] - t.state[i];
                sse[i] += d * d;
            }
        }
        sse.iter()
            .map(|s| {
                if count > 0 {
                    (s / count as f64).sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
}

impl DESStation for SdeEstimateSinkStation {
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
        self.core.inbox_size(SdeChannels::ESTIMATE) > 0
            || self.core.inbox_size(SdeChannels::STATE) > 0
    }
    fn run_time_step(&mut self) {
        let truth = self.core.drain::<SdeStateToken>(SdeChannels::STATE);
        self.truth.extend(truth);
        let estimates = self.core.drain::<SdeEstimateToken>(SdeChannels::ESTIMATE);
        self.estimates.extend(estimates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbm_closed_form_statistics() {
        let gbm = GeometricBrownianMotion::new(0.1, 0.2);
        assert!((gbm.exact(1.0, 0.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((gbm.mean_at(1.0, 0.0) - 1.0).abs() < 1e-12);
        assert!(gbm.var_at(1.0, 0.0).abs() < 1e-12);
        assert!((gbm.mean_at(1.0, 1.0) - 0.1_f64.exp()).abs() < 1e-12);
        assert_eq!(gbm.dimension(), 1);
        assert_eq!(gbm.diffusion(0.0, &[3.0])[0][0], 0.2 * 3.0);
    }

    #[test]
    fn ou_stationary_distribution() {
        let ou = OrnsteinUhlenbeck::new(0.5, 1.0, 0.3);
        assert!((ou.stationary_mean() - 1.0).abs() < 1e-12);
        assert!((ou.stationary_variance() - 0.09 / 1.0).abs() < 1e-12);
        // drift pulls toward mu: at x = mu the drift vanishes.
        assert!(ou.drift(0.0, &[1.0])[0].abs() < 1e-12);
        assert!(ou.drift(0.0, &[0.0])[0] > 0.0);
    }

    #[test]
    fn stochastic_dc_motor_drift_and_diffusion() {
        let m = StochasticDcMotor::new(StochasticDcMotorSpec {
            resistance: 1.0,
            inductance: 0.5,
            back_emf_constant: 0.01,
            torque_constant: 0.01,
            inertia: 0.01,
            friction: 0.1,
            voltage: 12.0,
            load_torque: None,
            current_noise: 0.05,
            speed_noise: 0.02,
        });
        assert_eq!(m.dimension(), 2);
        assert_eq!(m.noise_dimension(), 2);
        let d = m.drift(0.0, &[0.0, 0.0]);
        assert!((d[0] - 12.0 / 0.5).abs() < 1e-12);
        assert!(d[1].abs() < 1e-12);
        let g = m.diffusion(0.0, &[0.0, 0.0]);
        assert_eq!(g, vec![vec![0.05, 0.0], vec![0.0, 0.02]]);
    }

    #[test]
    fn euler_maruyama_matches_deterministic_growth_when_sigma_zero() {
        // σ = 0 => the SDE collapses to the ODE dx = 0.1 x dt; Euler converges to e^{0.1}.
        let gbm = GeometricBrownianMotion::new(0.1, 0.0);
        let em = EulerMaruyamaIntegrator::new();
        let mut rng = Mulberry32::new(1);
        let path = em.simulate(&gbm, &[1.0], 0.001, 1000, &mut rng);
        assert_eq!(path.path.len(), 1001);
        assert_eq!(path.times.len(), 1001);
        let last = path.path.last().unwrap()[0];
        assert!((last - 0.1_f64.exp()).abs() < 1e-2, "last = {last}");
    }

    #[test]
    fn plant_streams_truth_and_observations() {
        let mut plant = SdePlantStation::new(
            "sde",
            SdePlantOptions {
                system: Box::new(OrnsteinUhlenbeck::new(0.5, 1.0, 0.2)),
                x0: vec![0.0],
                dt: 0.01,
                steps: 50,
                observation_matrix: None,
                observation_noise_std: None,
                seed: Some(7),
            },
        );
        for _ in 0..50 {
            plant.run_time_step();
        }
        assert_eq!(plant.true_states.len(), 50);
        assert_eq!(plant.observations.len(), 50);
        // No observation noise => y == H x == x (identity).
        let last_state = plant.true_states.last().unwrap();
        let last_obs = plant.observations.last().unwrap();
        assert!((last_state.state[0] - last_obs.obs[0]).abs() < 1e-12);
        assert_eq!(last_state.step, 50);
    }

    #[test]
    fn plant_is_reproducible_for_a_fixed_seed() {
        let make = || {
            let mut plant = SdePlantStation::new(
                "sde",
                SdePlantOptions {
                    system: Box::new(GeometricBrownianMotion::new(0.05, 0.3)),
                    x0: vec![1.0],
                    dt: 0.01,
                    steps: 100,
                    observation_matrix: None,
                    observation_noise_std: Some(vec![0.1]),
                    seed: Some(42),
                },
            );
            for _ in 0..100 {
                plant.run_time_step();
            }
            plant.true_states.last().unwrap().state[0]
        };
        assert_eq!(make(), make());
    }
}
