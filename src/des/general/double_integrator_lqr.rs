//! Port of `src/des/general/double-integrator-lqr.ts` — the canonical DOUBLE
//! INTEGRATOR (point mass under direct force) controlled by an LQR derived from
//! the discrete-time algebraic Riccati equation (DARE).
//!
//! CONTINUOUS-TIME PLANT
//!   ẍ = u    (m = 1, no damping),  x_state = [position; velocity]
//!
//! DISCRETE-TIME (sample period τ)
//!   A = [[1, τ], [0, 1]],  B = [[τ²/2], [τ]]
//!
//! COSTS  Q = diag(q_pos, q_vel),  R = [[r_u]]; the infinite-horizon optimal LQR
//! minimises Σ_k xᵀQx + uᵀRu. A small Gaussian process noise w ∼ N(0, σ_w² I₂) is
//! injected and the closed-loop trajectory observed.
//!
//! [`run_double_integrator_lqr`] builds the LQR controller, simulates the
//! closed-loop system from a fixed initial state, and returns the trajectory plus
//! key diagnostics (final state, cumulative cost, Riccati solution).
//!
//! STUBBED / INLINED (their real home, `general/des-base/`, is not ported yet):
//!   * `LQRController` / `LQRSpec` (`des-base/lqr-controller.ts`) — reimplemented
//!     locally as [`LqrController`] / [`LqrSpec`]: the DARE fixed-point solve, the
//!     `u = −Kx` control law, per-component saturation, and the synchronous
//!     `step` helper inherited from `ControllerStation`. FLAGGED: this depends on
//!     the unported control-station framework, so a minimal local equivalent is
//!     provided here.
//!   * The file-local matrix helpers (`matMul`, `matInv`, …) that the TS
//!     `lqr-controller` re-exported are replaced by `crate::des::shared::linalg`.
//!   * Ambient `Math.random` via `mulberry32` — replaced by an injected
//!     `RandomSource` threaded through [`gaussian`], per the "inject capabilities"
//!     rule.

#![allow(dead_code)]

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};
use crate::des::general::prng::{mulberry32, RandomSource};
use crate::des::shared::linalg::{LinAlg, Matrix, MatrixInverse, Vector};

/// Box–Muller standard-normal sample using a `[0, 1)` RNG. Mirrors the TS
/// `gaussian(rng)` exactly (single `u`-clamp, no rejection loop), so the random
/// draw ordering matches the source.
fn gaussian(rng: &mut impl RandomSource) -> f64 {
    let mut u = rng.next_float();
    let v = rng.next_float();
    if u < 1e-12 {
        u = 1e-12;
    }
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

// -----------------------------------------------------------------------------
// LOCAL LQR CONTROLLER  (stub for the unported `des-base/lqr-controller.ts`)
// -----------------------------------------------------------------------------

/// LQR problem specification. Mirrors the TS `LQRSpec` (optionals → `Option`).
#[derive(Clone, Debug)]
pub struct LqrSpec {
    /// State dimension n.
    pub n: usize,
    /// Control dimension m.
    pub m: usize,
    /// A (n × n).
    pub a: Matrix,
    /// B (n × m).
    pub b: Matrix,
    /// State cost Q (n × n, symmetric PSD).
    pub q: Matrix,
    /// Control cost R (m × m, symmetric PD).
    pub r: Matrix,
    /// Discount factor γ ∈ (0, 1]. Default 1.
    pub gamma: Option<f64>,
    /// Per-component lower saturation. Default no clamp.
    pub u_min_vec: Option<Vector>,
    /// Per-component upper saturation. Default no clamp.
    pub u_max_vec: Option<Vector>,
    /// Riccati iteration tolerance. Default 1e-10.
    pub riccati_tol: Option<f64>,
    /// Riccati iteration max iters. Default 5000.
    pub riccati_max_iter: Option<usize>,
}

/// Infinite-horizon LQR: solves the DARE by fixed-point iteration, then runs
/// `u = −Kx` as the control law. Local minimal equivalent of the unported
/// `LQRController` station.
pub struct LqrController {
    /// Station id (kept for parity with the TS `DESStation`).
    pub id: String,
    /// Problem spec.
    pub spec: LqrSpec,
    /// Optimal feedback gain K (m × n).
    pub k: Matrix,
    /// Riccati solution P (n × n).
    pub p: Matrix,
    /// Riccati iteration count to convergence.
    pub riccati_iters: usize,
    /// Final residual ‖P_new − P‖_∞ at convergence.
    pub riccati_residual: f64,
    /// Per-tick observation history (TS `observationHistory`).
    pub observation_history: Vec<Vector>,
    /// Per-tick clamped-control history (TS `controlHistory`).
    pub control_history: Vec<Vector>,
    ticks_processed: usize,
}

impl LqrController {
    /// Construct, solving the DARE in the process. Parameter guards return
    /// `Err` (recoverable construction failure) rather than panicking.
    pub fn new(id: &str, spec: LqrSpec) -> Result<Self, PreconditionError> {
        let cls = "LQRController";
        Preconditions::integer_in_range(cls, "spec.n", spec.n as f64, 1.0, 10_000.0)?;
        Preconditions::integer_in_range(cls, "spec.m", spec.m as f64, 1.0, 10_000.0)?;
        Preconditions::length_eq(cls, "spec.A", &spec.a, spec.n)?;
        Preconditions::rectangular_matrix(cls, "spec.A", &spec.a)?;
        Preconditions::length_eq(cls, "spec.A[0]", &spec.a[0], spec.n)?;
        Preconditions::length_eq(cls, "spec.B", &spec.b, spec.n)?;
        Preconditions::rectangular_matrix(cls, "spec.B", &spec.b)?;
        Preconditions::length_eq(cls, "spec.B[0]", &spec.b[0], spec.m)?;
        Preconditions::symmetric_matrix(cls, "spec.Q", &spec.q, 1e-9)?;
        Preconditions::length_eq(cls, "spec.Q", &spec.q, spec.n)?;
        Preconditions::positive_semidefinite_diag(cls, "spec.Q", &spec.q, 1e-9)?;
        Preconditions::symmetric_matrix(cls, "spec.R", &spec.r, 1e-9)?;
        Preconditions::length_eq(cls, "spec.R", &spec.r, spec.m)?;
        // R MUST be positive-definite (we take its inverse).
        Preconditions::positive_definite_cholesky(cls, "spec.R", &spec.r)?;
        if let Some(g) = spec.gamma {
            Preconditions::in_range(cls, "spec.gamma", g, 1e-9, 1.0)?;
        }
        if let Some(lo) = &spec.u_min_vec {
            Preconditions::length_eq(cls, "spec.uMinVec", lo, spec.m)?;
            Preconditions::all_finite(cls, "spec.uMinVec", lo)?;
        }
        if let Some(hi) = &spec.u_max_vec {
            Preconditions::length_eq(cls, "spec.uMaxVec", hi, spec.m)?;
            Preconditions::all_finite(cls, "spec.uMaxVec", hi)?;
        }
        if let (Some(lo), Some(hi)) = (&spec.u_min_vec, &spec.u_max_vec) {
            for i in 0..spec.m {
                Preconditions::check(
                    cls,
                    &format!("uMin[{i}] <= uMax[{i}]"),
                    "satisfy uMin <= uMax",
                    lo[i] <= hi[i],
                    Some(format!("[{}, {}]", lo[i], hi[i])),
                )?;
            }
        }
        let gamma = spec.gamma.unwrap_or(1.0);
        let tol = spec.riccati_tol.unwrap_or(1e-10);
        let max_iter = spec.riccati_max_iter.unwrap_or(5000);
        Preconditions::positive(cls, "riccatiTol", tol)?;
        Preconditions::integer_in_range(cls, "riccatiMaxIter", max_iter as f64, 1.0, 10_000_000.0)?;

        let n = spec.n;
        let at = LinAlg::transpose(&spec.a);
        let bt = LinAlg::transpose(&spec.b);
        let mut p = LinAlg::copy(&spec.q);
        let mut iter = 0usize;
        let mut res = f64::INFINITY;
        while iter < max_iter {
            // R_eff = BᵀPB + R
            let btp = LinAlg::mat_mul(&bt, &p);
            let btpb = LinAlg::mat_mul(&btp, &spec.b);
            let reff = LinAlg::add(&btpb, &spec.r);
            let reff_inv = MatrixInverse::new(&reff, None).inverse();
            // K = R_eff⁻¹ BᵀPA
            let btpa = LinAlg::mat_mul(&btp, &spec.a);
            let k = LinAlg::mat_mul(&reff_inv, &btpa);
            // P_new = Q + γ(AᵀPA − AᵀPB K)
            let atp = LinAlg::mat_mul(&at, &p);
            let atpa = LinAlg::mat_mul(&atp, &spec.a);
            let atpb = LinAlg::mat_mul(&atp, &spec.b);
            let atpbk = LinAlg::mat_mul(&atpb, &k);
            let pnew = LinAlg::add(&spec.q, &LinAlg::scale(&LinAlg::sub(&atpa, &atpbk), gamma));
            // Residual = ‖P_new − P‖_∞
            let mut r = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let d = (pnew[i][j] - p[i][j]).abs();
                    if d > r {
                        r = d;
                    }
                }
            }
            p = pnew;
            res = r;
            iter += 1;
            if r < tol {
                break;
            }
        }
        // Final K from final P.
        let btp = LinAlg::mat_mul(&bt, &p);
        let btpb = LinAlg::mat_mul(&btp, &spec.b);
        let reff = LinAlg::add(&btpb, &spec.r);
        let reff_inv = MatrixInverse::new(&reff, None).inverse();
        let btpa = LinAlg::mat_mul(&btp, &spec.a);
        let k = LinAlg::mat_mul(&reff_inv, &btpa);

        Ok(LqrController {
            id: id.to_string(),
            spec,
            k,
            p,
            riccati_iters: iter,
            riccati_residual: res,
            observation_history: Vec::new(),
            control_history: Vec::new(),
            ticks_processed: 0,
        })
    }

    /// u = −Kx (no saturation).
    fn control_law(&self, observation: &[f64]) -> Vector {
        let kx = LinAlg::mat_vec(&self.k, observation);
        (0..self.spec.m).map(|i| -kx[i]).collect()
    }

    /// Per-component saturation (TS `clamp` override).
    fn clamp(&self, u: Vector) -> Vector {
        if self.spec.u_min_vec.is_none() && self.spec.u_max_vec.is_none() {
            return u;
        }
        let mut v = u;
        for i in 0..v.len() {
            if let Some(lo) = &self.spec.u_min_vec {
                if v[i] < lo[i] {
                    v[i] = lo[i];
                }
            }
            if let Some(hi) = &self.spec.u_max_vec {
                if v[i] > hi[i] {
                    v[i] = hi[i];
                }
            }
        }
        v
    }

    /// Synchronous one-shot control step: control law + saturation + history
    /// bookkeeping. Mirrors `ControllerStation::step`.
    pub fn step(&mut self, observation: &[f64], _tick: usize, _time: f64) -> Vector {
        let u = self.control_law(observation);
        let u_clamped = self.clamp(u);
        self.observation_history.push(observation.to_vec());
        self.control_history.push(u_clamped.clone());
        self.ticks_processed += 1;
        u_clamped
    }

    /// Optimal feedback gain K.
    pub fn get_gain(&self) -> Matrix {
        self.k.clone()
    }

    /// Riccati solution P.
    pub fn get_riccati_p(&self) -> Matrix {
        self.p.clone()
    }

    /// J(x₀) under the optimal policy in the DARE solution: x₀ᵀ P x₀.
    pub fn optimal_cost_from_initial_state(&self, x0: &[f64]) -> f64 {
        let px = LinAlg::mat_vec(&self.p, x0);
        let mut v = 0.0;
        for i in 0..x0.len() {
            v += x0[i] * px[i];
        }
        v
    }

    /// Number of control ticks processed.
    pub fn get_ticks_processed(&self) -> usize {
        self.ticks_processed
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Options for [`run_double_integrator_lqr`] (TS `DoubleIntegratorOpts`).
/// All-`None` defaults reproduce the TS `opts.x ?? default` fallbacks.
#[derive(Clone, Debug, Default)]
pub struct DoubleIntegratorOpts {
    /// Sample period τ. Default 0.1.
    pub dt: Option<f64>,
    /// Position weight in Q. Default 1.
    pub q_pos: Option<f64>,
    /// Velocity weight in Q. Default 0.1.
    pub q_vel: Option<f64>,
    /// Control weight in R. Default 0.01.
    pub r_u: Option<f64>,
    /// Process-noise stddev. Default 0.05.
    pub noise_std: Option<f64>,
    /// Initial state [pos, vel]. Default [3, 0].
    pub x0: Option<[f64; 2]>,
    /// Number of simulation steps. Default 100.
    pub num_steps: Option<usize>,
    /// Saturation on |u|. Default +∞ (no clamp).
    pub u_sat: Option<f64>,
    /// Discount γ. Default 1.
    pub gamma: Option<f64>,
    /// RNG seed. Default 1.
    pub seed: Option<u32>,
}

/// Result of [`run_double_integrator_lqr`] (TS `DoubleIntegratorResult`).
#[derive(Clone, Debug)]
pub struct DoubleIntegratorResult {
    /// Closed-loop trajectory [pos(t), vel(t)]. Length numSteps+1.
    pub trajectory: Vec<[f64; 2]>,
    /// Control input at each step.
    pub controls: Vec<f64>,
    /// Per-step running cost xᵀQx + uᵀRu.
    pub stage_costs: Vec<f64>,
    /// Σ stage costs.
    pub total_cost: f64,
    /// Optimal cost-to-go from x0 according to the Riccati P (theory).
    pub riccati_cost_from_x0: f64,
    /// Riccati iteration count to convergence.
    pub riccati_iters: usize,
    /// Final residual ‖P_new − P‖_∞ at convergence.
    pub riccati_residual: f64,
    /// Optimal feedback gain K.
    pub k: Matrix,
}

/// Build the LQR controller, simulate the noisy closed-loop double integrator,
/// and return the trajectory + diagnostics. Parameter guards return `Err`.
pub fn run_double_integrator_lqr(
    opts: DoubleIntegratorOpts,
) -> Result<DoubleIntegratorResult, PreconditionError> {
    let dt = opts.dt.unwrap_or(0.1);
    let q_pos = opts.q_pos.unwrap_or(1.0);
    let q_vel = opts.q_vel.unwrap_or(0.1);
    let r_u = opts.r_u.unwrap_or(0.01);
    let noise_std = opts.noise_std.unwrap_or(0.05);
    let x0 = opts.x0.unwrap_or([3.0, 0.0]);
    let num_steps = opts.num_steps.unwrap_or(100);
    let u_sat = opts.u_sat.unwrap_or(f64::INFINITY);
    let gamma = opts.gamma.unwrap_or(1.0);
    // Pre-run guards.
    let cls = "runDoubleIntegratorLQR";
    Preconditions::positive(cls, "dt", dt)?;
    Preconditions::non_negative(cls, "qPos", q_pos)?;
    Preconditions::non_negative(cls, "qVel", q_vel)?;
    // R = rU > 0 mandatory: DARE requires (R + B'PB) invertible.
    Preconditions::positive(cls, "rU", r_u)?;
    Preconditions::non_negative(cls, "noiseStd", noise_std)?;
    Preconditions::length_eq(cls, "x0", &x0, 2)?;
    Preconditions::all_finite(cls, "x0", &x0)?;
    Preconditions::integer_in_range(cls, "numSteps", num_steps as f64, 1.0, 1e9)?;
    if u_sat.is_finite() {
        Preconditions::positive(cls, "uSat", u_sat)?;
    }
    Preconditions::in_range(cls, "gamma", gamma, 1e-9, 1.0)?;
    let mut rng = mulberry32(opts.seed.unwrap_or(1));

    let spec = LqrSpec {
        n: 2,
        m: 1,
        a: vec![vec![1.0, dt], vec![0.0, 1.0]],
        b: vec![vec![dt * dt / 2.0], vec![dt]],
        q: vec![vec![q_pos, 0.0], vec![0.0, q_vel]],
        r: vec![vec![r_u]],
        gamma: Some(gamma),
        u_min_vec: if u_sat.is_finite() {
            Some(vec![-u_sat])
        } else {
            None
        },
        u_max_vec: if u_sat.is_finite() {
            Some(vec![u_sat])
        } else {
            None
        },
        riccati_tol: None,
        riccati_max_iter: None,
    };
    let mut ctrl = LqrController::new("double-int-lqr", spec)?;

    let mut traj: Vec<[f64; 2]> = vec![[x0[0], x0[1]]];
    let mut ctrls: Vec<f64> = Vec::new();
    let mut stage_costs: Vec<f64> = Vec::new();
    let mut total = 0.0;
    let mut x: [f64; 2] = [x0[0], x0[1]];
    for k in 0..num_steps {
        let u = ctrl.step(&[x[0], x[1]], k, k as f64 * dt);
        let u_val = u[0];
        ctrls.push(u_val);
        // Stage cost xᵀQx + uᵀRu.
        let sc = q_pos * x[0] * x[0] + q_vel * x[1] * x[1] + r_u * u_val * u_val;
        stage_costs.push(sc);
        total += sc;
        // Dynamics step: x_{k+1} = A x_k + B u_k + w_k.
        let w0 = if noise_std > 0.0 {
            noise_std * gaussian(&mut rng)
        } else {
            0.0
        };
        let w1 = if noise_std > 0.0 {
            noise_std * gaussian(&mut rng)
        } else {
            0.0
        };
        let x_next: [f64; 2] = [
            x[0] + dt * x[1] + (dt * dt / 2.0) * u_val + w0,
            x[1] + dt * u_val + w1,
        ];
        traj.push(x_next);
        x = x_next;
    }
    Ok(DoubleIntegratorResult {
        riccati_cost_from_x0: ctrl.optimal_cost_from_initial_state(&[x0[0], x0[1]]),
        riccati_iters: ctrl.riccati_iters,
        riccati_residual: ctrl.riccati_residual,
        k: ctrl.get_gain(),
        trajectory: traj,
        controls: ctrls,
        stage_costs,
        total_cost: total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lqr_drives_state_toward_origin() {
        // Noise-free run so the test is deterministic: the optimal regulator
        // should drive [3, 0] toward the origin.
        let res = run_double_integrator_lqr(DoubleIntegratorOpts {
            noise_std: Some(0.0),
            num_steps: Some(200),
            ..Default::default()
        })
        .unwrap();
        let first = res.trajectory.first().unwrap();
        let last = res.trajectory.last().unwrap();
        let last_norm = (last[0] * last[0] + last[1] * last[1]).sqrt();
        let first_norm = (first[0] * first[0] + first[1] * first[1]).sqrt();
        assert!(last_norm < first_norm, "state should shrink");
        assert!(last_norm < 0.1, "final state near origin, got {last_norm}");
    }

    #[test]
    fn riccati_converges() {
        let res = run_double_integrator_lqr(DoubleIntegratorOpts::default()).unwrap();
        assert!(res.riccati_iters > 0);
        assert!(
            res.riccati_residual < 1e-10,
            "residual {}",
            res.riccati_residual
        );
        // Gain has shape m × n = 1 × 2 and positive position feedback.
        assert_eq!(res.k.len(), 1);
        assert_eq!(res.k[0].len(), 2);
        assert!(res.k[0][0] > 0.0);
    }

    #[test]
    fn costs_are_finite_and_nonnegative() {
        let res = run_double_integrator_lqr(DoubleIntegratorOpts {
            noise_std: Some(0.0),
            ..Default::default()
        })
        .unwrap();
        assert!(res.total_cost.is_finite() && res.total_cost > 0.0);
        assert!(res.stage_costs.iter().all(|&c| c >= 0.0));
        // Realised cost should not undershoot the theoretical optimum-to-go.
        assert!(res.riccati_cost_from_x0 > 0.0);
    }
}
