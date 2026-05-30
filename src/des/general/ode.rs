//! Port of `src/des/general/ode.ts` — fixed-step + adaptive ODE integrators.
//!
//! ODE solvers for first-order systems  y'(t) = f(t, y),  y(t₀) = y₀.
//!
//! All solvers operate on vector-valued `y` of any dimension. Higher-order ODEs
//! y'' = … are reduced to first-order by stacking `[y, y']` in the state vector
//! — `second_order_to_first_order` builds the stacked RHS.
//!
//! Methods (cost / order in parentheses):
//!   * `EulerIntegrator`         forward Euler            (1 feval/step, O(dt))
//!   * `HeunIntegrator`          improved Euler (RK2)     (2 fevals/step, O(dt²))
//!   * `RK4Integrator`           classical RK4            (4 fevals/step, O(dt⁴))
//!   * `RK45Integrator`          Dormand-Prince adaptive  (6 fevals/step, O(dt⁵))
//!   * `BackwardEulerIntegrator` implicit Euler           (Newton inner-iter, A-stable)
//!
//! Adaptive RK45 is the workhorse for non-stiff problems. Backward Euler is for
//! stiff problems where explicit methods would need impossibly small `dt`.
//!
//! TS mapping: the `PureTransform<I, O>` solver classes become config structs
//! implementing [`Transform`]. The RHS closure `(t, y[]) => y[]` becomes a
//! generic `F: Fn(f64, &[f64]) -> Vec<f64>` and the Jacobian `(t, y[]) => y[][]`
//! a generic `J: Fn(f64, &[f64]) -> Vec<Vec<f64>>`. The `@deprecated` free-fn
//! shims are dropped — call the structs directly.

use crate::des::shared::transform::Transform;

/// A solver trace: the time grid `t` and the state `y[i]` recorded at each `t[i]`.
#[derive(Clone, Debug)]
pub struct ODETrace {
    pub t: Vec<f64>,
    /// `y[i]` is the state vector at time `t[i]`.
    pub y: Vec<Vec<f64>>,
}

/// An initial-value problem y'(t)=f(t,y), y(t₀)=y₀, integrated over [t₀, t₁].
///
/// Bundles the positional (f, y0, t0, t1) arguments into one named input so the
/// integrators keep the `Transform<I, O>` shape. Generic over the RHS closure.
pub struct IVP<F>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    pub f: F,
    pub y0: Vec<f64>,
    pub t0: f64,
    pub t1: f64,
}

/// A stiff IVP that additionally carries the Jacobian J=∂f/∂y (or `None` to fall
/// back to fixed-point iteration), consumed by [`BackwardEulerIntegrator`].
pub struct StiffIVP<F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    pub f: F,
    pub j: Option<J>,
    pub y0: Vec<f64>,
    pub t0: f64,
    pub t1: f64,
}

// -----------------------------------------------------------------------------
// Vector helpers (file-local, mirroring the TS `vplus`/`vscale`/`vmax`).
// -----------------------------------------------------------------------------

/// `a + s·b`, elementwise.
fn vplus(a: &[f64], b: &[f64], s: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(v, bi)| v + s * bi).collect()
}

/// `a · s`, elementwise.
fn vscale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|v| v * s).collect()
}

/// Max absolute component (`‖a‖∞`).
fn vmax(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, v| m.max(v.abs()))
}

// -----------------------------------------------------------------------------
// Forward Euler.  y_{n+1} = y_n + dt · f(t_n, y_n)
// -----------------------------------------------------------------------------

/// Forward Euler integrator. Fixed step `dt` is config; the IVP is the input.
pub struct EulerIntegrator {
    pub dt: f64,
}

impl EulerIntegrator {
    pub fn new(dt: f64) -> Self {
        EulerIntegrator { dt }
    }
}

impl<F> Transform<IVP<F>, ODETrace> for EulerIntegrator
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    fn transform(&self, problem: IVP<F>) -> ODETrace {
        let IVP { f, y0, t0, t1 } = problem;
        let dt = self.dt;
        let mut t = vec![t0];
        let mut y = vec![y0.clone()];
        let mut tn = t0;
        let mut yn = y0;
        while tn + 0.5 * dt < t1 {
            let fnv = f(tn, &yn);
            yn = vplus(&yn, &fnv, dt);
            tn += dt;
            t.push(tn);
            y.push(yn.clone());
        }
        ODETrace { t, y }
    }
}

// -----------------------------------------------------------------------------
// Heun's method (RK2 / improved Euler). Predictor + corrector.
//   k1 = f(t,y);  k2 = f(t+dt, y+dt·k1)
//   y_{n+1} = y_n + dt/2 · (k1 + k2)
// -----------------------------------------------------------------------------

/// Heun's method (RK2 / improved Euler). Fixed step `dt` is config.
pub struct HeunIntegrator {
    pub dt: f64,
}

impl HeunIntegrator {
    pub fn new(dt: f64) -> Self {
        HeunIntegrator { dt }
    }
}

impl<F> Transform<IVP<F>, ODETrace> for HeunIntegrator
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    fn transform(&self, problem: IVP<F>) -> ODETrace {
        let IVP { f, y0, t0, t1 } = problem;
        let dt = self.dt;
        let mut t = vec![t0];
        let mut y = vec![y0.clone()];
        let mut tn = t0;
        let mut yn = y0;
        while tn + 0.5 * dt < t1 {
            let k1 = f(tn, &yn);
            let k2 = f(tn + dt, &vplus(&yn, &k1, dt));
            yn = vplus(&yn, &vplus(&k1, &k2, 1.0), dt / 2.0);
            tn += dt;
            t.push(tn);
            y.push(yn.clone());
        }
        ODETrace { t, y }
    }
}

// -----------------------------------------------------------------------------
// Classical RK4. The textbook fourth-order Runge–Kutta scheme.
// -----------------------------------------------------------------------------

/// Classical fourth-order Runge–Kutta (RK4). Fixed step `dt` is config.
pub struct RK4Integrator {
    pub dt: f64,
}

impl RK4Integrator {
    pub fn new(dt: f64) -> Self {
        RK4Integrator { dt }
    }
}

impl<F> Transform<IVP<F>, ODETrace> for RK4Integrator
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    fn transform(&self, problem: IVP<F>) -> ODETrace {
        let IVP { f, y0, t0, t1 } = problem;
        let dt = self.dt;
        let mut t = vec![t0];
        let mut y = vec![y0.clone()];
        let mut tn = t0;
        let mut yn = y0;
        while tn + 0.5 * dt < t1 {
            let k1 = f(tn, &yn);
            let k2 = f(tn + dt / 2.0, &vplus(&yn, &k1, dt / 2.0));
            let k3 = f(tn + dt / 2.0, &vplus(&yn, &k2, dt / 2.0));
            let k4 = f(tn + dt, &vplus(&yn, &k3, dt));
            // incr = (k1 + k4) + (2·k2 + 2·k3)
            let incr = vplus(
                &vplus(&k1, &k4, 1.0),
                &vplus(&vscale(&k2, 2.0), &vscale(&k3, 2.0), 1.0),
                1.0,
            );
            yn = vplus(&yn, &incr, dt / 6.0);
            tn += dt;
            t.push(tn);
            y.push(yn.clone());
        }
        ODETrace { t, y }
    }
}

// -----------------------------------------------------------------------------
// Dormand-Prince RK45 with adaptive step size.
// scipy.integrate.solve_ivp(method='RK45') uses the same Butcher tableau.
// -----------------------------------------------------------------------------

/// Options for [`RK45Integrator`]. `None` fields fall back to the documented
/// defaults, computed per-problem where they depend on `(t1 - t0)`.
#[derive(Clone, Debug, Default)]
pub struct RK45Options {
    /// Relative tolerance (default 1e-6).
    pub rtol: Option<f64>,
    /// Absolute tolerance (default 1e-9).
    pub atol: Option<f64>,
    /// Initial step size (default (t1-t0)/100).
    pub h_init: Option<f64>,
    /// Minimum step size (default 1e-12).
    pub h_min: Option<f64>,
    /// Maximum step size (default t1-t0).
    pub h_max: Option<f64>,
    /// Hard cap on the number of steps (default 1_000_000).
    pub max_steps: Option<usize>,
}

// Dormand-Prince Butcher tableau coefficients.
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;
const C6: f64 = 1.0;

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
const A71: f64 = 35.0 / 384.0;
const A72: f64 = 0.0;
const A73: f64 = 500.0 / 1113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2187.0 / 6784.0;
const A76: f64 = 11.0 / 84.0;
// E_i = b_i − b̂_i, used to estimate the error.
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

/// Dormand-Prince RK45 with adaptive step size. CONFIG (tolerances, step-size
/// bounds, step cap) lives on the struct; the IVP is the `transform` input.
#[derive(Default)]
pub struct RK45Integrator {
    pub opts: RK45Options,
}

impl RK45Integrator {
    pub fn new(opts: RK45Options) -> Self {
        RK45Integrator { opts }
    }
}


impl<F> Transform<IVP<F>, ODETrace> for RK45Integrator
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    fn transform(&self, problem: IVP<F>) -> ODETrace {
        let IVP { f, y0, t0, t1 } = problem;
        let opts = &self.opts;
        let rtol = opts.rtol.unwrap_or(1e-6);
        let atol = opts.atol.unwrap_or(1e-9);
        let h_init = opts.h_init.unwrap_or((t1 - t0) / 100.0);
        let h_min = opts.h_min.unwrap_or(1e-12);
        let h_max = opts.h_max.unwrap_or(t1 - t0);
        let max_steps = opts.max_steps.unwrap_or(1_000_000);

        let mut t = vec![t0];
        let mut y = vec![y0.clone()];
        let mut tn = t0;
        let mut yn = y0;
        let mut h = h_max.min(h_min.max(h_init));
        let n = yn.len();
        let mut step: usize = 0;
        while tn < t1 - 1e-15 {
            step += 1;
            if step > max_steps {
                eprintln!(
                    "[ode.rk45] exceeded maxSteps={max_steps} at t={tn} (target t1={t1}, current h={h}); integration aborted."
                );
                panic!("rk45: exceeded {max_steps} steps");
            }
            if tn + h > t1 {
                h = t1 - tn;
            }
            let k1 = f(tn, &yn);
            let k2 = f(tn + C2 * h, &vplus(&yn, &k1, h * A21));
            let yk3in: Vec<f64> = yn
                .iter()
                .enumerate()
                .map(|(i, v)| v + h * (A31 * k1[i] + A32 * k2[i]))
                .collect();
            let k3 = f(tn + C3 * h, &yk3in);
            let yk4in: Vec<f64> = yn
                .iter()
                .enumerate()
                .map(|(i, v)| v + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]))
                .collect();
            let k4 = f(tn + C4 * h, &yk4in);
            let yk5in: Vec<f64> = yn
                .iter()
                .enumerate()
                .map(|(i, v)| v + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]))
                .collect();
            let k5 = f(tn + C5 * h, &yk5in);
            let yk6in: Vec<f64> = yn
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i])
                })
                .collect();
            let k6 = f(tn + C6 * h, &yk6in);
            // 5th-order solution at t+h.
            let y5: Vec<f64> = yn
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v + h * (A71 * k1[i]
                        + A72 * k2[i]
                        + A73 * k3[i]
                        + A74 * k4[i]
                        + A75 * k5[i]
                        + A76 * k6[i])
                })
                .collect();
            let k7 = f(tn + h, &y5);
            // Error estimate (5th − 4th order).
            let mut err_norm = 0.0;
            for i in 0..n {
                let sci = atol + rtol * yn[i].abs().max(y5[i].abs());
                let ei = h
                    * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
                err_norm += (ei / sci) * (ei / sci);
            }
            err_norm = (err_norm / n as f64).sqrt();
            if err_norm <= 1.0 {
                tn += h;
                yn = y5;
                t.push(tn);
                y.push(yn.clone());
                // Step expansion factor (simple I controller).
                let factor = if err_norm == 0.0 {
                    5.0
                } else {
                    5.0_f64.min(0.9 * err_norm.powf(-1.0 / 5.0))
                };
                h = h_max.min(h_min.max(h * factor));
            } else {
                let factor = 0.1_f64.max(0.9 * err_norm.powf(-1.0 / 5.0));
                h = h_min.max(h * factor);
                if h <= h_min {
                    eprintln!(
                        "[ode.rk45] step size underflow at t={tn}: h={h} ≤ hMin={h_min} with errNorm={err_norm}; problem may be stiff (try backwardEuler)."
                    );
                    panic!("rk45: step underflow at t={tn}");
                }
            }
        }
        ODETrace { t, y }
    }
}

// -----------------------------------------------------------------------------
// Backward (implicit) Euler.  y_{n+1} = y_n + dt · f(t_{n+1}, y_{n+1}).
// Solves the implicit equation by Newton iteration using the Jacobian J
// (= ∂f/∂y). Use for stiff problems. Falls back to fixed-point if no J.
// -----------------------------------------------------------------------------

/// Backward (implicit) Euler for stiff systems. CONFIG (fixed step `dt`, Newton
/// tolerance and iteration cap) lives on the struct; the stiff IVP — which
/// carries the Jacobian `J` (or `None` for fixed-point fallback) — is the input.
pub struct BackwardEulerIntegrator {
    pub dt: f64,
    pub newton_tol: f64,
    pub newton_max_iter: usize,
}

impl BackwardEulerIntegrator {
    pub fn new(dt: f64, newton_tol: f64, newton_max_iter: usize) -> Self {
        BackwardEulerIntegrator { dt, newton_tol, newton_max_iter }
    }

    /// Construct with the TS default Newton settings (`tol=1e-10`, `maxIter=50`).
    pub fn with_dt(dt: f64) -> Self {
        BackwardEulerIntegrator { dt, newton_tol: 1e-10, newton_max_iter: 50 }
    }
}

impl<F, J> Transform<StiffIVP<F, J>, ODETrace> for BackwardEulerIntegrator
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    fn transform(&self, problem: StiffIVP<F, J>) -> ODETrace {
        let StiffIVP { f, j, y0, t0, t1 } = problem;
        let dt = self.dt;
        let newton_tol = self.newton_tol;
        let newton_max_iter = self.newton_max_iter;
        let mut t = vec![t0];
        let mut y = vec![y0.clone()];
        let mut tn = t0;
        let mut yn = y0;
        while tn + 0.5 * dt < t1 {
            let t_next = tn + dt;
            let mut y_next = yn.clone();
            let mut success = false;
            for _iter in 0..newton_max_iter {
                let f_next = f(t_next, &y_next);
                let g: Vec<f64> = y_next
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v - yn[i] - dt * f_next[i])
                    .collect();
                let g_norm = vmax(&g);
                if g_norm < newton_tol {
                    success = true;
                    break;
                }
                if let Some(jac) = &j {
                    let jmat = jac(t_next, &y_next);
                    let n = y_next.len();
                    let mut m: Vec<Vec<f64>> = Vec::with_capacity(n);
                    for i in 0..n {
                        let mut row = vec![0.0; n];
                        for (jj, cell) in row.iter_mut().enumerate() {
                            *cell = (if i == jj { 1.0 } else { 0.0 }) - dt * jmat[i][jj];
                        }
                        m.push(row);
                    }
                    let dy = solve_linear(&m, &g);
                    for i in 0..n {
                        y_next[i] -= dy[i];
                    }
                } else {
                    // Fixed-point: y^{k+1} = y_n + dt · f(t_{n+1}, y^k). Often diverges for stiff.
                    for i in 0..y_next.len() {
                        y_next[i] = yn[i] + dt * f_next[i];
                    }
                }
            }
            if !success && j.is_some() {
                eprintln!(
                    "[ode.backwardEuler] Newton iteration failed to converge (tol={newton_tol}, maxIter={newton_max_iter}) at t={tn}; Jacobian may be wrong or step dt={dt} too large."
                );
                panic!("backwardEuler: Newton failed at t={tn}");
            }
            yn = y_next;
            tn = t_next;
            t.push(tn);
            y.push(yn.clone());
        }
        ODETrace { t, y }
    }
}

/// Dense linear solve via Gaussian elimination with partial pivoting. Panics on
/// a singular matrix, matching the TS invariant violation (`throw`).
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().cloned().collect();
    let mut x = b.to_vec();
    for i in 0..n {
        let mut p = i;
        for k in (i + 1)..n {
            if m[k][i].abs() > m[p][i].abs() {
                p = k;
            }
        }
        if m[p][i].abs() < 1e-15 {
            eprintln!(
                "[ode.backwardEuler] singular Newton matrix (pivot {} at column {i}/{n}); cannot solve the implicit step.",
                m[p][i]
            );
            panic!("singular matrix in backwardEuler");
        }
        if p != i {
            m.swap(i, p);
            x.swap(i, p);
        }
        for k in (i + 1)..n {
            let factor = m[k][i] / m[i][i];
            for jj in i..n {
                m[k][jj] -= factor * m[i][jj];
            }
            x[k] -= factor * x[i];
        }
    }
    let mut y = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = x[i];
        for jj in (i + 1)..n {
            s -= m[i][jj] * y[jj];
        }
        y[i] = s / m[i][i];
    }
    y
}

// -----------------------------------------------------------------------------
// Helper: build the stacked first-order system for a 2nd-order ODE
//   y'' + p(t)·y' + q(t)·y = r(t)
// by setting state = [y, y']. Caller supplies p, q, r as closures.
// -----------------------------------------------------------------------------

/// Reduce y'' + p(t)·y' + q(t)·y = r(t) to a first-order RHS on state `[y, y']`.
pub fn second_order_to_first_order<P, Q, R>(p: P, q: Q, r: R) -> impl Fn(f64, &[f64]) -> Vec<f64>
where
    P: Fn(f64) -> f64,
    Q: Fn(f64) -> f64,
    R: Fn(f64) -> f64,
{
    move |t: f64, y: &[f64]| {
        // y[0] = y, y[1] = y'
        vec![y[1], r(t) - p(t) * y[1] - q(t) * y[0]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y' = y, y(0) = 1 ⇒ y(t) = eᵗ. RK4 integrates it to high accuracy.
    #[test]
    fn rk4_exponential_growth() {
        let trace = RK4Integrator::new(0.01).transform(IVP {
            f: |_t, y| vec![y[0]],
            y0: vec![1.0],
            t0: 0.0,
            t1: 1.0,
        });
        let last = trace.y.last().unwrap()[0];
        assert!((last - std::f64::consts::E).abs() < 1e-6);
    }

    /// Same problem, adaptive RK45: should also land on e within tolerance.
    #[test]
    fn rk45_exponential_growth() {
        let trace = RK45Integrator::default().transform(IVP {
            f: |_t, y| vec![y[0]],
            y0: vec![1.0],
            t0: 0.0,
            t1: 1.0,
        });
        let last = trace.y.last().unwrap()[0];
        assert!((last - std::f64::consts::E).abs() < 1e-4);
        // Final recorded time should reach t1.
        assert!((trace.t.last().unwrap() - 1.0).abs() < 1e-9);
    }

    /// Harmonic oscillator y'' = -y, y(0)=1, y'(0)=0 ⇒ y(t)=cos(t).
    #[test]
    fn rk4_harmonic_oscillator() {
        let rhs = second_order_to_first_order(|_t| 0.0, |_t| 1.0, |_t| 0.0);
        let trace = RK4Integrator::new(0.001).transform(IVP {
            f: rhs,
            y0: vec![1.0, 0.0],
            t0: 0.0,
            t1: std::f64::consts::PI,
        });
        let last = trace.y.last().unwrap();
        // y(π) = cos(π) = -1, y'(π) = -sin(π) = 0. Fixed-step RK4 stops at the
        // last grid point ≤ π (up to one dt short), so y' is bounded by ~dt
        // rather than machine-epsilon; RK4's truncation error itself is ~dt⁴.
        assert!((last[0] + 1.0).abs() < 1e-5);
        assert!(last[1].abs() < 2e-3);
    }

    /// Stiff decay y' = -15y, y(0)=1 ⇒ y(t)=e^{-15t}. Backward Euler stays
    /// stable with the analytical Jacobian even at a coarse step.
    #[test]
    fn backward_euler_stiff_decay() {
        let trace = BackwardEulerIntegrator::with_dt(0.05).transform(StiffIVP {
            f: |_t, y| vec![-15.0 * y[0]],
            j: Some(|_t: f64, _y: &[f64]| vec![vec![-15.0]]),
            y0: vec![1.0],
            t0: 0.0,
            t1: 1.0,
        });
        let last = trace.y.last().unwrap()[0];
        // Backward Euler is only O(dt) accurate but must remain bounded & positive.
        assert!(last > 0.0 && last < 0.1);
        let exact = (-15.0_f64).exp();
        assert!((last - exact).abs() < 0.05);
    }

    /// Forward Euler on y' = y over a single coarse step matches y0 + dt·y0.
    #[test]
    fn euler_single_step() {
        let trace = EulerIntegrator::new(1.0).transform(IVP {
            f: |_t, y| vec![y[0]],
            y0: vec![1.0],
            t0: 0.0,
            t1: 1.0,
        });
        // One step: tn + 0.5*dt < t1 holds once (0.5 < 1), then stops.
        assert_eq!(trace.t.len(), 2);
        assert!((trace.y[1][0] - 2.0).abs() < 1e-12);
    }
}
