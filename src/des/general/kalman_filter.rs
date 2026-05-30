//! Port of `src/des/general/kalman-filter.ts` — the LINEAR KALMAN FILTER
//! (Kalman 1960) on the canonical RADAR / GPS 1-D TRACKING problem.
//!
//! PROBLEM
//!   A 1-D point mass moves with random acceleration noise:
//!     x_{k+1} = A x_k + w_k,   w_k ∼ N(0, Q)
//!     y_k     = H x_k + v_k,   v_k ∼ N(0, R)
//!   with state x = [position, velocity]ᵀ, A = [[1, dt], [0, 1]], H = [1, 0]
//!   (position-only observation), Q from integrated acceleration noise, R the
//!   sensor variance.
//!
//! KALMAN UPDATE (Welch & Bishop 1995)
//!   PREDICT  x̂⁻ = A x̂                P⁻ = A P Aᵀ + Q
//!   UPDATE   K  = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹
//!            x̂  = x̂⁻ + K (y − H x̂⁻)   P  = (I − K H) P⁻
//!
//! [`run_radar_tracking`] wires a noisy constant-velocity plant → Kalman filter
//! and returns the trajectory, measurements, estimates, and RMSE diagnostics.
//!
//! STUBBED / INLINED (their real home, `general/des-base/control-blocks.ts`, is
//! not ported yet):
//!   * `PlantBlock` / `EstimatorBlock` / `ControllerBlock` / `runClosedLoop` /
//!     `VectorSignal` (the heavyweight signal-block framework) — FLAGGED as
//!     unported. For this specific wiring the closed loop reduces to a
//!     deterministic lock-step: the passive `NullController` never emits a
//!     control, so the plant always advances under u = 0 and the filter consumes
//!     one measurement per tick. That reduction is inlined directly into
//!     [`run_radar_tracking`], with [`RadarPlant`] kept as a local struct mirroring
//!     `PlantBlock`'s state/output history bookkeeping.
//!   * The `lqr-controller` matrix-helper re-exports (`matMul`, `matInv`, …) are
//!     replaced by `crate::des::shared::linalg`.
//!   * `mulberry32` ambient RNG — threaded explicitly as a `RandomSource` per the
//!     "inject capabilities" rule.

#![allow(dead_code)]

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};
use crate::des::general::prng::{mulberry32, RandomSource};
use crate::des::shared::linalg::{LinAlg, Matrix, MatrixInverse, Vector};

/// Box–Muller standard-normal sample using a `[0, 1)` RNG. Mirrors the TS
/// `gaussian(rng)` (single `u`-clamp, no rejection loop).
fn gaussian(rng: &mut impl RandomSource) -> f64 {
    let mut u = rng.next_float();
    let v = rng.next_float();
    if u < 1e-12 {
        u = 1e-12;
    }
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

// -----------------------------------------------------------------------------
// PLANT: noisy 1-D constant-velocity model with position-only sensor
// -----------------------------------------------------------------------------

/// Local equivalent of the TS `RadarPlant extends PlantBlock`: constant-velocity
/// motion with Gaussian acceleration noise and a position-only noisy sensor.
struct RadarPlant {
    state: Vector,
    dt: f64,
    proc_noise_std: f64,
    meas_noise_std: f64,
    state_history: Vec<Vector>,
    input_history: Vec<Vector>,
    output_history: Vec<Vector>,
    tick: usize,
    last_u: Vector,
}

impl RadarPlant {
    fn new(x0: [f64; 2], dt: f64, proc_noise_std: f64, meas_noise_std: f64) -> Self {
        RadarPlant {
            state: vec![x0[0], x0[1]],
            dt,
            proc_noise_std,
            meas_noise_std,
            state_history: vec![vec![x0[0], x0[1]]],
            input_history: Vec::new(),
            output_history: Vec::new(),
            tick: 0,
            // Plant is seeded with u = [0] and the passive controller never
            // overrides it, so the latest control is always zero.
            last_u: vec![0.0],
        }
    }

    /// Constant velocity + Gaussian acceleration noise.
    fn dynamics(&self, acc_noise: f64) -> Vector {
        let dt = self.dt;
        let x0 = self.state[0];
        let x1 = self.state[1];
        vec![
            x0 + dt * x1 + 0.5 * dt * dt * acc_noise,
            x1 + dt * acc_noise,
        ]
    }

    /// Advance one tick (drain control → dynamics → observe) and return the
    /// emitted measurement. Mirrors `PlantBlock::runTimeStep`.
    fn step(&mut self, rng: &mut impl RandomSource) -> Vector {
        let acc_noise = self.proc_noise_std * gaussian(rng);
        let x_new = self.dynamics(acc_noise);
        self.input_history.push(self.last_u.clone());
        self.state = x_new;
        self.state_history.push(self.state.clone());
        self.tick += 1;
        let y = vec![self.state[0] + self.meas_noise_std * gaussian(rng)];
        self.output_history.push(y.clone());
        y
    }
}

// -----------------------------------------------------------------------------
// KALMAN FILTER BLOCK
// -----------------------------------------------------------------------------

/// Specification of the Kalman model matrices (TS `KalmanFilterBlock` ctor arg).
#[derive(Clone, Debug)]
pub struct KalmanSpec {
    /// Initial state estimate.
    pub x0: Vector,
    /// Initial posterior covariance.
    pub p0: Matrix,
    /// State-transition matrix A.
    pub a: Matrix,
    /// Observation matrix H.
    pub h: Matrix,
    /// Process-noise covariance Q.
    pub q: Matrix,
    /// Measurement-noise covariance R.
    pub r: Matrix,
}

/// Linear Kalman filter as a DES estimator block. Drains a measurement on every
/// tick, runs ONE predict–update step, and emits the posterior estimate.
pub struct KalmanFilterBlock {
    /// State estimate.
    xhat: Vector,
    /// Posterior covariance.
    p: Matrix,
    a: Matrix,
    h: Matrix,
    q: Matrix,
    r: Matrix,
    /// History of emitted state estimates (TS `estimateHistory`).
    pub estimate_history: Vec<Vector>,
    /// History of consumed measurements (TS `measurementHistory`).
    pub measurement_history: Vec<Vector>,
}

impl KalmanFilterBlock {
    /// Construct, validating the model matrices. Guards return `Err`.
    pub fn new(spec: KalmanSpec) -> Result<Self, PreconditionError> {
        let cls = "KalmanFilterBlock";
        let n = spec.x0.len();
        Preconditions::check(cls, "x0.length", "be >= 1", n >= 1, Some(n.to_string()))?;
        Preconditions::all_finite(cls, "x0", &spec.x0)?;
        Preconditions::length_eq(cls, "A", &spec.a, n)?;
        Preconditions::rectangular_matrix(cls, "A", &spec.a)?;
        Preconditions::length_eq(cls, "A[0]", &spec.a[0], n)?;
        Preconditions::symmetric_matrix(cls, "P0", &spec.p0, 1e-9)?;
        Preconditions::length_eq(cls, "P0", &spec.p0, n)?;
        Preconditions::positive_semidefinite_diag(cls, "P0", &spec.p0, 1e-9)?;
        Preconditions::symmetric_matrix(cls, "Q", &spec.q, 1e-9)?;
        Preconditions::length_eq(cls, "Q", &spec.q, n)?;
        Preconditions::positive_semidefinite_diag(cls, "Q", &spec.q, 1e-9)?;
        Preconditions::rectangular_matrix(cls, "H", &spec.h)?;
        Preconditions::length_eq(cls, "H[0]", &spec.h[0], n)?;
        let m = spec.h.len();
        Preconditions::check(
            cls,
            "H.length (output dim m)",
            "be >= 1",
            m >= 1,
            Some(m.to_string()),
        )?;
        Preconditions::symmetric_matrix(cls, "R", &spec.r, 1e-9)?;
        Preconditions::length_eq(cls, "R", &spec.r, m)?;
        // R MUST be PD (we invert H P Hᵀ + R and need a strictly positive total
        // innovation covariance — zero measurement noise ⇒ degenerate).
        Preconditions::positive_definite_cholesky(cls, "R", &spec.r)?;
        Ok(KalmanFilterBlock {
            xhat: spec.x0.clone(),
            p: LinAlg::copy(&spec.p0),
            a: spec.a,
            h: spec.h,
            q: spec.q,
            r: spec.r,
            estimate_history: Vec::new(),
            measurement_history: Vec::new(),
        })
    }

    /// One predict–update step. `_u` is unused (process model is autonomous
    /// here) but kept for parity with the TS `update(y, u)` signature.
    pub fn update(&mut self, y: &[f64], _u: Option<&[f64]>) -> Vector {
        // PREDICT.
        let xhat_pred = LinAlg::mat_vec(&self.a, &self.xhat);
        let ap = LinAlg::mat_mul(&self.a, &self.p);
        let apat = LinAlg::mat_mul(&ap, &LinAlg::transpose(&self.a));
        let ppred = LinAlg::add(&apat, &self.q);
        // UPDATE: K = P Hᵀ (H P Hᵀ + R)⁻¹
        let hp = LinAlg::mat_mul(&self.h, &ppred);
        let hpht = LinAlg::mat_mul(&hp, &LinAlg::transpose(&self.h));
        let s = LinAlg::add(&hpht, &self.r);
        let s_inv = MatrixInverse::new(&s, None).inverse();
        let pht = LinAlg::mat_mul(&ppred, &LinAlg::transpose(&self.h));
        let k = LinAlg::mat_mul(&pht, &s_inv);
        // Innovation y − H x̂⁻.
        let hxhat = LinAlg::mat_vec(&self.h, &xhat_pred);
        let innov: Vector = y.iter().enumerate().map(|(i, yi)| yi - hxhat[i]).collect();
        // x̂ = x̂⁻ + K innov.
        let kinnov = LinAlg::mat_vec(&k, &innov);
        self.xhat = xhat_pred
            .iter()
            .enumerate()
            .map(|(i, v)| v + kinnov[i])
            .collect();
        // P = (I − K H) P⁻.
        let kh = LinAlg::mat_mul(&k, &self.h);
        let ident = LinAlg::identity(kh.len());
        self.p = LinAlg::mat_mul(&LinAlg::sub(&ident, &kh), &ppred);
        self.xhat.clone()
    }

    /// Consume one measurement: record it, run the filter, record the estimate.
    /// Mirrors `EstimatorBlock::runTimeStep`.
    fn run_time_step(&mut self, y: &[f64]) {
        self.measurement_history.push(y.to_vec());
        let xhat = self.update(y, None);
        self.estimate_history.push(xhat);
    }

    /// Current state estimate.
    pub fn get_estimate(&self) -> Vector {
        self.xhat.clone()
    }

    /// Current posterior covariance.
    pub fn get_covariance(&self) -> Matrix {
        LinAlg::copy(&self.p)
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Options for [`run_radar_tracking`] (TS `RadarTrackingOpts`).
#[derive(Clone, Debug, Default)]
pub struct RadarTrackingOpts {
    /// Initial true state [pos, vel]. Default [0, 1].
    pub x0: Option<[f64; 2]>,
    /// Sample period. Default 0.1.
    pub dt: Option<f64>,
    /// Number of simulation steps. Default 200.
    pub num_steps: Option<usize>,
    /// Process-noise std σ_w (acceleration). Default 0.1.
    pub proc_noise_std: Option<f64>,
    /// Sensor std σ_v. Default 1.0.
    pub meas_noise_std: Option<f64>,
    /// Initial covariance scale (P0 = scale·I). Default 10.
    pub p0_scale: Option<f64>,
    /// RNG seed. Default 1.
    pub seed: Option<u32>,
}

/// Result of [`run_radar_tracking`] (TS `RadarTrackingResult`).
#[derive(Clone, Debug)]
pub struct RadarTrackingResult {
    /// True trajectory (length numSteps+1).
    pub true_trajectory: Vec<Vector>,
    /// Raw measurements (length numSteps).
    pub measurements: Vec<Vector>,
    /// Filter estimates (length numSteps).
    pub estimates: Vec<Vector>,
    /// RMSE between true position and KF estimate.
    pub rmse_pos: f64,
    /// RMSE between true position and raw measurement (baseline).
    pub rmse_meas_pos: f64,
    /// Final position covariance trace.
    pub final_cov_trace: f64,
    /// Number of simulation steps.
    pub num_steps: usize,
}

/// Simulate the radar/GPS tracking problem: a noisy constant-velocity plant
/// driven open-loop (u = 0) with a Kalman filter consuming each measurement.
/// Parameter guards return `Err`.
pub fn run_radar_tracking(
    opts: RadarTrackingOpts,
) -> Result<RadarTrackingResult, PreconditionError> {
    let x0 = opts.x0.unwrap_or([0.0, 1.0]);
    let dt = opts.dt.unwrap_or(0.1);
    let num_steps = opts.num_steps.unwrap_or(200);
    let proc_noise_std = opts.proc_noise_std.unwrap_or(0.1);
    let meas_noise_std = opts.meas_noise_std.unwrap_or(1.0);
    let p0_scale = opts.p0_scale.unwrap_or(10.0);
    let cls = "runRadarTracking";
    Preconditions::length_eq(cls, "x0", &x0, 2)?;
    Preconditions::all_finite(cls, "x0", &x0)?;
    Preconditions::positive(cls, "dt", dt)?;
    Preconditions::integer_in_range(cls, "numSteps", num_steps as f64, 1.0, 1e9)?;
    Preconditions::non_negative(cls, "procNoiseStd", proc_noise_std)?;
    // Sensor noise must be strictly positive — KF inverts H P H' + R.
    Preconditions::positive(cls, "measNoiseStd", meas_noise_std)?;
    Preconditions::positive(cls, "P0Scale", p0_scale)?;
    let mut rng = mulberry32(opts.seed.unwrap_or(1));

    let mut plant = RadarPlant::new(x0, dt, proc_noise_std, meas_noise_std);

    // Build KF model matrices for constant-velocity dynamics.
    let a: Matrix = vec![vec![1.0, dt], vec![0.0, 1.0]];
    let h: Matrix = vec![vec![1.0, 0.0]];
    // Process noise from continuous-time acceleration variance σ_w²:
    //   Q = σ_w² · [[dt⁴/4, dt³/2], [dt³/2, dt²]]
    let sw2 = proc_noise_std * proc_noise_std;
    let q: Matrix = vec![
        vec![sw2 * dt * dt * dt * dt / 4.0, sw2 * dt * dt * dt / 2.0],
        vec![sw2 * dt * dt * dt / 2.0, sw2 * dt * dt],
    ];
    let r: Matrix = vec![vec![meas_noise_std * meas_noise_std]];
    // Initial estimate: first position at zero velocity, large P0 uncertainty.
    let mut kf = KalmanFilterBlock::new(KalmanSpec {
        x0: vec![x0[0], 0.0],
        p0: vec![vec![p0_scale, 0.0], vec![0.0, p0_scale]],
        a,
        h,
        q,
        r,
    })?;

    // Lock-step driver (passive controller emits nothing, so u stays 0).
    for _ in 0..num_steps {
        let y = plant.step(&mut rng);
        kf.run_time_step(&y);
    }

    // Compute diagnostics. The plant's measurement at tick t corresponds to the
    // state at tick t, so the estimate at index i aligns with state index i+1.
    let true_traj = &plant.state_history; // length numSteps+1
    let meas = &plant.output_history; // length numSteps
    let est = &kf.estimate_history; // length numSteps
    let mut rmse_est_sum = 0.0;
    let mut rmse_meas_sum = 0.0;
    let mut n: f64 = 0.0;
    for i in 0..est.len() {
        let true_pos = true_traj[i + 1][0];
        let e_pos = est[i][0];
        let m_pos = meas[i][0];
        rmse_est_sum += (true_pos - e_pos) * (true_pos - e_pos);
        rmse_meas_sum += (true_pos - m_pos) * (true_pos - m_pos);
        n += 1.0;
    }
    let rmse_pos = (rmse_est_sum / n.max(1.0)).sqrt();
    let rmse_meas_pos = (rmse_meas_sum / n.max(1.0)).sqrt();
    let p_final = kf.get_covariance();
    let final_cov_trace = p_final[0][0] + p_final[1][1];

    Ok(RadarTrackingResult {
        true_trajectory: true_traj.iter().map(|x| x.clone()).collect(),
        measurements: meas.iter().map(|y| y.clone()).collect(),
        estimates: est.iter().map(|x| x.clone()).collect(),
        rmse_pos,
        rmse_meas_pos,
        final_cov_trace,
        num_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_beats_raw_measurement() {
        let res = run_radar_tracking(RadarTrackingOpts {
            seed: Some(7),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(res.true_trajectory.len(), res.num_steps + 1);
        assert_eq!(res.estimates.len(), res.num_steps);
        // The Kalman estimate should track better than the raw noisy sensor.
        assert!(
            res.rmse_pos < res.rmse_meas_pos,
            "KF rmse {} should beat raw rmse {}",
            res.rmse_pos,
            res.rmse_meas_pos
        );
        // Posterior covariance should shrink well below the P0 = 10·I prior.
        assert!(res.final_cov_trace < 20.0);
    }

    #[test]
    fn estimate_converges_to_constant_signal() {
        // A scalar random-walk model fed a constant measurement should converge
        // its estimate toward that constant as covariance shrinks.
        let mut kf = KalmanFilterBlock::new(KalmanSpec {
            x0: vec![0.0],
            p0: vec![vec![10.0]],
            a: vec![vec![1.0]],
            h: vec![vec![1.0]],
            q: vec![vec![1e-6]],
            r: vec![vec![1.0]],
        })
        .unwrap();
        let target = 5.0;
        let mut xhat = kf.get_estimate();
        for _ in 0..500 {
            xhat = kf.update(&[target], None);
        }
        assert!(
            (xhat[0] - target).abs() < 0.1,
            "estimate {} should converge to {target}",
            xhat[0]
        );
        let p = kf.get_covariance();
        assert!(p[0][0] < 1.0, "covariance should shrink, got {}", p[0][0]);
    }

    #[test]
    fn rejects_singular_measurement_noise() {
        // R = 0 is not positive-definite ⇒ construction must fail.
        let bad = KalmanFilterBlock::new(KalmanSpec {
            x0: vec![0.0, 0.0],
            p0: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            a: vec![vec![1.0, 0.1], vec![0.0, 1.0]],
            h: vec![vec![1.0, 0.0]],
            q: vec![vec![1e-4, 0.0], vec![0.0, 1e-4]],
            r: vec![vec![0.0]],
        });
        assert!(bad.is_err());
    }
}
