//! Port of `src/des/general/des-base/lqr-controller.ts` — the infinite-horizon
//! LINEAR QUADRATIC REGULATOR, the canonical "stochastic control = MDP"
//! example.
//!
//! ## Control problem
//!
//!   Plant:  x_{k+1} = A x_k + B u_k + w_k     (w_k ∼ N(0, Σ_w), optional)
//!   Cost:   J = E[ Σ_k x_kᵀ Q x_k + u_kᵀ R u_k ]
//!
//!   Optimal control law (infinite-horizon, discounted gamma ∈ (0, 1]):
//!     u_k = −K x_k     where    K = (Bᵀ P B + R)^{−1} Bᵀ P A
//!   and P solves the discrete-time algebraic Riccati equation (DARE):
//!     P = Q + gamma Aᵀ P A − gamma Aᵀ P B (Bᵀ P B + R)^{−1} Bᵀ P A
//!
//! We solve the DARE by fixed-point iteration, then run the resulting affine
//! state-feedback law as the [`ControllerStation::control_law`] hook.
//!
//! ## Rust shape
//!
//!   * TS `class LQRController extends ControllerStation<Vec, Vec>` →
//!     [`LQRController`] struct embedding a [`StationCore`] + a
//!     `ControllerCore<Vector, Vector>`, implementing both [`DESStation`] and
//!     [`ControllerStation`].
//!   * The local matrix helpers (`matMul`, `matT`, `matInv`, …) that duplicated
//!     `shared/linalg.ts` are DELETED; we reuse
//!     [`crate::des::shared::linalg`] (`LinAlg` / `MatrixInverse`). The TS
//!     `export {matMul, …}` re-export goes away.
//!   * `matInv` (Gauss-Jordan, threw on singular) → [`MatrixInverse`] which
//!     `panic!`s on a singular matrix.
//!   * Non-ASCII `gamma`; all-`number` matrices → `f64` via `Matrix`/`Vector`.
//!   * `uMinVec?/uMaxVec?: Vec` → `Option<Vector>`.
//!   * TS overrides `clamp` for per-component saturation; here that is the
//!     [`Saturate`] impl for `Vec<f64>` (FLAGGED below), so the inherited
//!     template-method `clamp` does the work via `u_min()`/`u_max()`.
//!   * Construction-time `Preconditions.*` throws → `panic!` at the
//!     construction edge (TS constructor `throw` = invariant violation).

use super::controller::{ControllerCore, ControllerStation, Saturate};
use super::preconditions::{Check, Preconditions};
use super::station::{DESStation, StationCore};
use crate::des::shared::linalg::{LinAlg, Matrix, MatrixInverse, Vector};

use std::any::Any;

/// Tolerance for the symmetry / PSD structural guards (TS used the
/// `Preconditions` defaults; the ported guards take an explicit `tol`).
const STRUCTURAL_TOL: f64 = 1e-9;

/// Panic at the construction edge if a precondition guard fails (mirrors the TS
/// constructor `throw`). The `PreconditionError` `Display` carries the message.
fn require(c: Check) {
    if let Err(e) = c {
        panic!("{e}");
    }
}

/// FLAGGED: per-component saturation for vector controls.
///
/// `controller.rs` only implements [`Saturate`] for the scalar `f64` and notes
/// that "non-scalar `U` simply provides a no-op `Saturate` impl". The TS
/// `LQRController` instead overrides `clamp` to clamp **per component** against
/// `uMinVec`/`uMaxVec`. To stay faithful we implement [`Saturate`] for
/// `Vec<f64>` here (legal: the trait is local to this crate). NOTE: if another
/// ported module also adds `impl Saturate for Vec<f64>`, that is a duplicate
/// impl (`E0119`) — there must be exactly one crate-wide.
impl Saturate for Vec<f64> {
    fn saturate(self, lo: Option<Self>, hi: Option<Self>) -> Self {
        if lo.is_none() && hi.is_none() {
            return self;
        }
        let mut v = self;
        for i in 0..v.len() {
            if let Some(ref lo) = lo {
                if v[i] < lo[i] {
                    v[i] = lo[i];
                }
            }
            if let Some(ref hi) = hi {
                if v[i] > hi[i] {
                    v[i] = hi[i];
                }
            }
        }
        v
    }
}

/// Problem specification for an infinite-horizon LQR.
#[derive(Clone, Debug)]
pub struct LQRSpec {
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
    /// Discount factor gamma ∈ (0, 1]. Default 1.
    pub gamma: Option<f64>,
    /// Per-component saturation lower bound `[u_min]^m`. Default no clamp.
    pub u_min_vec: Option<Vector>,
    /// Per-component saturation upper bound `[u_max]^m`. Default no clamp.
    pub u_max_vec: Option<Vector>,
    /// Riccati iteration tolerance. Default 1e-10.
    pub riccati_tol: Option<f64>,
    /// Riccati iteration max iters. Default 5000.
    pub riccati_max_iter: Option<usize>,
}

/// Infinite-horizon LQR feedback controller.
pub struct LQRController {
    core: StationCore,
    ctrl: ControllerCore<Vector, Vector>,
    /// The validated problem spec.
    pub spec: LQRSpec,
    /// Optimal feedback gain K (m × n).
    pub k: Matrix,
    /// Riccati solution P (n × n).
    pub p: Matrix,
    /// Number of Riccati iterations performed.
    pub riccati_iters: usize,
    /// Final `‖P_new − P‖_∞` residual.
    pub riccati_residual: f64,
}

impl LQRController {
    /// Validate `spec`, solve the DARE, and cache the gain. Panics on an invalid
    /// spec (TS constructor `throw`).
    pub fn new(id: impl Into<String>, spec: LQRSpec) -> Self {
        let cls = "LQRController";
        // Pre-construction guards — DARE math is only valid if these hold.
        require(Preconditions::integer_in_range(cls, "spec.n", spec.n as f64, 1.0, 10_000.0));
        require(Preconditions::integer_in_range(cls, "spec.m", spec.m as f64, 1.0, 10_000.0));
        require(Preconditions::length_eq(cls, "spec.A", &spec.a, spec.n));
        require(Preconditions::rectangular_matrix(cls, "spec.A", &spec.a));
        require(Preconditions::length_eq(cls, "spec.A[0]", &spec.a[0], spec.n));
        require(Preconditions::length_eq(cls, "spec.B", &spec.b, spec.n));
        require(Preconditions::rectangular_matrix(cls, "spec.B", &spec.b));
        require(Preconditions::length_eq(cls, "spec.B[0]", &spec.b[0], spec.m));
        require(Preconditions::symmetric_matrix(cls, "spec.Q", &spec.q, STRUCTURAL_TOL));
        require(Preconditions::length_eq(cls, "spec.Q", &spec.q, spec.n));
        require(Preconditions::positive_semidefinite_diag(cls, "spec.Q", &spec.q, STRUCTURAL_TOL));
        require(Preconditions::symmetric_matrix(cls, "spec.R", &spec.r, STRUCTURAL_TOL));
        require(Preconditions::length_eq(cls, "spec.R", &spec.r, spec.m));
        // R MUST be positive-definite (we take its inverse). Cholesky test
        // catches user errors like R = 0.
        require(Preconditions::positive_definite_cholesky(cls, "spec.R", &spec.r));
        if let Some(gamma) = spec.gamma {
            require(Preconditions::in_range(cls, "spec.gamma", gamma, 1e-9, 1.0));
        }
        if let Some(lo) = &spec.u_min_vec {
            require(Preconditions::length_eq(cls, "spec.uMinVec", lo, spec.m));
            require(Preconditions::all_finite(cls, "spec.uMinVec", lo));
        }
        if let Some(hi) = &spec.u_max_vec {
            require(Preconditions::length_eq(cls, "spec.uMaxVec", hi, spec.m));
            require(Preconditions::all_finite(cls, "spec.uMaxVec", hi));
        }
        if let (Some(lo), Some(hi)) = (&spec.u_min_vec, &spec.u_max_vec) {
            for i in 0..spec.m {
                require(Preconditions::check(
                    cls,
                    &format!("uMin[{i}] <= uMax[{i}]"),
                    "satisfy uMin <= uMax",
                    lo[i] <= hi[i],
                    Some(format!("[{}, {}]", lo[i], hi[i])),
                ));
            }
        }
        let gamma = spec.gamma.unwrap_or(1.0);
        let tol = spec.riccati_tol.unwrap_or(1e-10);
        let max_iter = spec.riccati_max_iter.unwrap_or(5000);
        require(Preconditions::positive(cls, "riccatiTol", tol));
        require(Preconditions::integer_in_range(cls, "riccatiMaxIter", max_iter as f64, 1.0, 10_000_000.0));

        let n = spec.n;
        let bt = LinAlg::transpose(&spec.b);
        let at = LinAlg::transpose(&spec.a);

        let mut p = LinAlg::copy(&spec.q);
        let mut iter = 0usize;
        let mut res = f64::INFINITY;
        while iter < max_iter {
            // R_eff = Bᵀ P B + R
            let bt_p = LinAlg::mat_mul(&bt, &p);
            let bt_pb = LinAlg::mat_mul(&bt_p, &spec.b);
            let r_eff = LinAlg::add(&bt_pb, &spec.r);
            let r_eff_inv = MatrixInverse::new(&r_eff, None).inverse();
            // K = R_eff^{-1} Bᵀ P A
            let bt_pa = LinAlg::mat_mul(&bt_p, &spec.a);
            let k = LinAlg::mat_mul(&r_eff_inv, &bt_pa);
            // P_new = Q + gamma (Aᵀ P A − Aᵀ P B K)
            let at_p = LinAlg::mat_mul(&at, &p);
            let at_pa = LinAlg::mat_mul(&at_p, &spec.a);
            let at_pb = LinAlg::mat_mul(&at_p, &spec.b);
            let at_pbk = LinAlg::mat_mul(&at_pb, &k);
            let p_new = LinAlg::add(&spec.q, &LinAlg::scale(&LinAlg::sub(&at_pa, &at_pbk), gamma));
            // Residual = ‖P_new − P‖_∞
            let mut r = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let d = (p_new[i][j] - p[i][j]).abs();
                    if d > r {
                        r = d;
                    }
                }
            }
            p = p_new;
            res = r;
            iter += 1;
            if r < tol {
                break;
            }
        }

        // Final K from final P.
        let bt_p = LinAlg::mat_mul(&bt, &p);
        let bt_pb = LinAlg::mat_mul(&bt_p, &spec.b);
        let r_eff = LinAlg::add(&bt_pb, &spec.r);
        let r_eff_inv = MatrixInverse::new(&r_eff, None).inverse();
        let bt_pa = LinAlg::mat_mul(&bt_p, &spec.a);
        let k = LinAlg::mat_mul(&r_eff_inv, &bt_pa);

        LQRController {
            core: StationCore::new(id),
            ctrl: ControllerCore::new(),
            spec,
            k,
            p,
            riccati_iters: iter,
            riccati_residual: res,
        }
    }

    // ── PUBLIC ACCESSORS ──────────────────────────────────────────────────────

    /// Optimal feedback gain K.
    pub fn get_gain(&self) -> &Matrix {
        &self.k
    }

    /// Riccati solution P.
    pub fn get_riccati_p(&self) -> &Matrix {
        &self.p
    }

    /// J(x_0) under the optimal policy in the DARE solution: x_0ᵀ P x_0.
    pub fn optimal_cost_from_initial_state(&self, x0: &[f64]) -> f64 {
        let px = LinAlg::mat_vec(&self.p, x0);
        let mut v = 0.0;
        for i in 0..x0.len() {
            v += x0[i] * px[i];
        }
        v
    }
}

impl DESStation for LQRController {
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
        self.controller_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.controller_has_work()
    }
}

impl ControllerStation<Vector, Vector> for LQRController {
    fn controller_core(&self) -> &ControllerCore<Vector, Vector> {
        &self.ctrl
    }
    fn controller_core_mut(&mut self) -> &mut ControllerCore<Vector, Vector> {
        &mut self.ctrl
    }

    /// u = −K x (saturation is applied by the template-method `clamp`).
    fn control_law(&mut self, observation: &Vector, _tick: f64, _time: f64) -> Vector {
        let kx = LinAlg::mat_vec(&self.k, observation);
        let mut u = vec![0.0; self.spec.m];
        for i in 0..self.spec.m {
            u[i] = -kx[i];
        }
        u
    }

    fn u_min(&self) -> Option<Vector> {
        self.spec.u_min_vec.clone()
    }
    fn u_max(&self) -> Option<Vector> {
        self.spec.u_max_vec.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::linalg::VecOps;

    fn scalar_spec() -> LQRSpec {
        LQRSpec {
            n: 1,
            m: 1,
            a: vec![vec![1.0]],
            b: vec![vec![1.0]],
            q: vec![vec![1.0]],
            r: vec![vec![1.0]],
            gamma: None,
            u_min_vec: None,
            u_max_vec: None,
            riccati_tol: None,
            riccati_max_iter: None,
        }
    }

    #[test]
    fn scalar_gain_matches_analytic_and_drives_state_to_zero() {
        // For a=b=q=r=1, gamma=1 the DARE is P = P + 1 − P²/(P+1), giving the
        // golden ratio P = (1+√5)/2 and gain K = P/(P+1) = 1/φ ≈ 0.618034.
        let mut lqr = LQRController::new("lqr", scalar_spec());
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((lqr.k[0][0] - 1.0 / phi).abs() < 1e-6, "gain {} != {}", lqr.k[0][0], 1.0 / phi);
        assert!((lqr.p[0][0] - phi).abs() < 1e-6, "P {} != {}", lqr.p[0][0], phi);

        // Closed-loop x_{k+1} = x_k + u_k with u = step(x) drives x → 0.
        let a = lqr.spec.a.clone();
        let b = lqr.spec.b.clone();
        let mut x = vec![5.0];
        for k in 0..80 {
            let u = lqr.step(x.clone(), k as f64, k as f64);
            x = VecOps::add(&LinAlg::mat_vec(&a, &x), &LinAlg::mat_vec(&b, &u));
        }
        assert!(x[0].abs() < 1e-6, "state did not converge: {}", x[0]);

        // Optimal cost-to-go from x0 = [1] is x0ᵀ P x0 = P = φ.
        assert!((lqr.optimal_cost_from_initial_state(&[1.0]) - phi).abs() < 1e-6);
    }

    #[test]
    fn double_integrator_drives_state_to_zero() {
        // Double integrator: A = [[1,1],[0,1]], B = [[0],[1]], Q = I, R = [[1]].
        let spec = LQRSpec {
            n: 2,
            m: 1,
            a: vec![vec![1.0, 1.0], vec![0.0, 1.0]],
            b: vec![vec![0.0], vec![1.0]],
            q: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            r: vec![vec![1.0]],
            gamma: None,
            u_min_vec: None,
            u_max_vec: None,
            riccati_tol: None,
            riccati_max_iter: None,
        };
        let mut lqr = LQRController::new("lqr2", spec);
        let a = lqr.spec.a.clone();
        let b = lqr.spec.b.clone();
        let mut x = vec![3.0, -2.0];
        for k in 0..200 {
            let u = lqr.step(x.clone(), k as f64, k as f64);
            x = VecOps::add(&LinAlg::mat_vec(&a, &x), &LinAlg::mat_vec(&b, &u));
        }
        assert!(VecOps::norm2(&x) < 1e-3, "state did not converge: {x:?}");
    }

    #[test]
    fn clamp_saturates_control_per_component() {
        let mut spec = scalar_spec();
        spec.u_min_vec = Some(vec![-0.1]);
        spec.u_max_vec = Some(vec![0.1]);
        let mut lqr = LQRController::new("lqr-clamped", spec);
        // Large state ⇒ large raw control, clamped into [−0.1, 0.1].
        let u = lqr.step(vec![100.0], 0.0, 0.0);
        assert!(u[0] <= 0.1 + 1e-12 && u[0] >= -0.1 - 1e-12, "not clamped: {}", u[0]);
        // Raw u = −Kx ≈ −61.8 ⇒ clamp pins it to the lower bound.
        assert!((u[0] + 0.1).abs() < 1e-9, "expected lower-bound clamp, got {}", u[0]);
    }
}
