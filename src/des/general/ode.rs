//! Rust port of `src/des/general/ode.ts`.

use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/ode.ts",
    "src/des/general/ode.rs",
    &[
        "RHS/Jac callbacks are Rust Fn bounds over slices and Vec rows.",
        "ODETrace is a serde struct with time points and state rows.",
        "Explicit solvers remain free functions; DES graph wrappers should live in their mapped station files.",
        "Fallible adaptive/implicit paths return OdeError instead of throwing.",
    ],
    &[
        "Jac",
        "ODETrace",
        "RHS",
        "backwardEuler",
        "euler",
        "rk2Heun",
        "rk4",
        "rk45",
        "secondOrderToFirstOrder",
    ],
);

pub type Rhs = dyn Fn(f64, &[f64]) -> Vec<f64>;
pub type Jac = dyn Fn(f64, &[f64]) -> Vec<Vec<f64>>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OdeTrace {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rk45Options {
    pub rtol: f64,
    pub atol: f64,
    pub h_init: Option<f64>,
    pub h_min: f64,
    pub h_max: Option<f64>,
    pub max_steps: usize,
}

impl Default for Rk45Options {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-9,
            h_init: None,
            h_min: 1e-12,
            h_max: None,
            max_steps: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum OdeError {
    #[error("{method}: dt must be positive and finite, got {dt}")]
    InvalidStep { method: &'static str, dt: f64 },
    #[error("rk45: exceeded {max_steps} steps at t={t}")]
    MaxSteps { max_steps: usize, t: f64 },
    #[error("rk45: step underflow at t={t}")]
    StepUnderflow { t: f64 },
    #[error("backward_euler: singular matrix")]
    SingularMatrix,
    #[error("backward_euler: Newton failed at t={t}")]
    NewtonFailed { t: f64 },
}

pub fn euler<F>(f: F, y0: &[f64], t0: f64, t1: f64, dt: f64) -> Result<OdeTrace, OdeError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_dt("euler", dt)?;
    let mut t = vec![t0];
    let mut y = vec![y0.to_vec()];
    let mut tn = t0;
    let mut yn = y0.to_vec();
    while tn + 0.5 * dt < t1 {
        let f_n = f(tn, &yn);
        yn = vplus(&yn, &f_n, dt);
        tn += dt;
        t.push(tn);
        y.push(yn.clone());
    }
    Ok(OdeTrace { t, y })
}

pub fn rk2_heun<F>(f: F, y0: &[f64], t0: f64, t1: f64, dt: f64) -> Result<OdeTrace, OdeError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_dt("rk2_heun", dt)?;
    let mut t = vec![t0];
    let mut y = vec![y0.to_vec()];
    let mut tn = t0;
    let mut yn = y0.to_vec();
    while tn + 0.5 * dt < t1 {
        let k1 = f(tn, &yn);
        let y_predict = vplus(&yn, &k1, dt);
        let k2 = f(tn + dt, &y_predict);
        let increment = vplus(&k1, &k2, 1.0);
        yn = vplus(&yn, &increment, dt / 2.0);
        tn += dt;
        t.push(tn);
        y.push(yn.clone());
    }
    Ok(OdeTrace { t, y })
}

pub fn rk4<F>(f: F, y0: &[f64], t0: f64, t1: f64, dt: f64) -> Result<OdeTrace, OdeError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_dt("rk4", dt)?;
    let mut t = vec![t0];
    let mut y = vec![y0.to_vec()];
    let mut tn = t0;
    let mut yn = y0.to_vec();
    while tn + 0.5 * dt < t1 {
        let k1 = f(tn, &yn);
        let y2 = vplus(&yn, &k1, dt / 2.0);
        let k2 = f(tn + dt / 2.0, &y2);
        let y3 = vplus(&yn, &k2, dt / 2.0);
        let k3 = f(tn + dt / 2.0, &y3);
        let y4 = vplus(&yn, &k3, dt);
        let k4 = f(tn + dt, &y4);
        let weighted_mid = vplus(&vscale(&k2, 2.0), &vscale(&k3, 2.0), 1.0);
        let increment = vplus(&vplus(&k1, &k4, 1.0), &weighted_mid, 1.0);
        yn = vplus(&yn, &increment, dt / 6.0);
        tn += dt;
        t.push(tn);
        y.push(yn.clone());
    }
    Ok(OdeTrace { t, y })
}

pub fn rk45<F>(f: F, y0: &[f64], t0: f64, t1: f64, opts: Rk45Options) -> Result<OdeTrace, OdeError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let h_init = opts.h_init.unwrap_or((t1 - t0) / 100.0);
    validate_dt("rk45", h_init)?;
    let h_max = opts.h_max.unwrap_or(t1 - t0);
    let mut t = vec![t0];
    let mut y = vec![y0.to_vec()];
    let mut tn = t0;
    let mut yn = y0.to_vec();
    let mut h = h_init.clamp(opts.h_min, h_max);
    let n = y0.len();
    let mut step = 0;

    while tn < t1 - 1e-15 {
        if step > opts.max_steps {
            return Err(OdeError::MaxSteps {
                max_steps: opts.max_steps,
                t: tn,
            });
        }
        step += 1;
        if tn + h > t1 {
            h = t1 - tn;
        }

        let k1 = f(tn, &yn);
        let k2 = f(tn + C2 * h, &add_scaled(&yn, h, &[(A21, &k1)]));
        let k3 = f(tn + C3 * h, &add_scaled(&yn, h, &[(A31, &k1), (A32, &k2)]));
        let k4 = f(
            tn + C4 * h,
            &add_scaled(&yn, h, &[(A41, &k1), (A42, &k2), (A43, &k3)]),
        );
        let k5 = f(
            tn + C5 * h,
            &add_scaled(&yn, h, &[(A51, &k1), (A52, &k2), (A53, &k3), (A54, &k4)]),
        );
        let k6 = f(
            tn + C6 * h,
            &add_scaled(
                &yn,
                h,
                &[(A61, &k1), (A62, &k2), (A63, &k3), (A64, &k4), (A65, &k5)],
            ),
        );
        let y5 = add_scaled(
            &yn,
            h,
            &[
                (A71, &k1),
                (A72, &k2),
                (A73, &k3),
                (A74, &k4),
                (A75, &k5),
                (A76, &k6),
            ],
        );
        let k7 = f(tn + h, &y5);

        let mut err_norm = 0.0;
        for i in 0..n {
            let sci = opts.atol + opts.rtol * yn[i].abs().max(y5[i].abs());
            let err_i =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            err_norm += (err_i / sci) * (err_i / sci);
        }
        err_norm = (err_norm / n as f64).sqrt();

        if err_norm <= 1.0 {
            tn += h;
            yn = y5;
            t.push(tn);
            y.push(yn.clone());
            let factor = if err_norm == 0.0 {
                5.0
            } else {
                (0.9 * err_norm.powf(-0.2)).min(5.0)
            };
            h = (h * factor).clamp(opts.h_min, h_max);
        } else {
            let factor = (0.9 * err_norm.powf(-0.2)).max(0.1);
            h = (h * factor).max(opts.h_min);
            if h <= opts.h_min {
                return Err(OdeError::StepUnderflow { t: tn });
            }
        }
    }

    Ok(OdeTrace { t, y })
}

pub fn backward_euler<F, J>(
    f: F,
    jac: Option<J>,
    y0: &[f64],
    t0: f64,
    t1: f64,
    dt: f64,
    newton_tol: f64,
    newton_max_iter: usize,
) -> Result<OdeTrace, OdeError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    validate_dt("backward_euler", dt)?;
    let mut t = vec![t0];
    let mut y = vec![y0.to_vec()];
    let mut tn = t0;
    let mut yn = y0.to_vec();

    while tn + 0.5 * dt < t1 {
        let t_next = tn + dt;
        let mut y_next = yn.clone();
        let mut success = false;
        for _ in 0..newton_max_iter {
            let f_next = f(t_next, &y_next);
            let g: Vec<f64> = y_next
                .iter()
                .enumerate()
                .map(|(i, value)| value - yn[i] - dt * f_next[i])
                .collect();
            if vmax(&g) < newton_tol {
                success = true;
                break;
            }
            if let Some(jac_fn) = &jac {
                let jacobian = jac_fn(t_next, &y_next);
                let n = y_next.len();
                let mut matrix = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        matrix[i][j] = if i == j { 1.0 } else { 0.0 } - dt * jacobian[i][j];
                    }
                }
                let dy = solve_linear(matrix, g)?;
                for i in 0..n {
                    y_next[i] -= dy[i];
                }
            } else {
                for i in 0..y_next.len() {
                    y_next[i] = yn[i] + dt * f_next[i];
                }
            }
        }
        if !success && jac.is_some() {
            return Err(OdeError::NewtonFailed { t: tn });
        }
        yn = y_next;
        tn = t_next;
        t.push(tn);
        y.push(yn.clone());
    }

    Ok(OdeTrace { t, y })
}

pub fn second_order_to_first_order<P, Q, R>(p: P, q: Q, r: R) -> impl Fn(f64, &[f64]) -> Vec<f64>
where
    P: Fn(f64) -> f64,
    Q: Fn(f64) -> f64,
    R: Fn(f64) -> f64,
{
    move |t, y| vec![y[1], r(t) - p(t) * y[1] - q(t) * y[0]]
}

fn validate_dt(method: &'static str, dt: f64) -> Result<(), OdeError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(OdeError::InvalidStep { method, dt });
    }
    Ok(())
}

fn vplus(a: &[f64], b: &[f64], scale: f64) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| left + scale * right)
        .collect()
}

fn vscale(a: &[f64], scale: f64) -> Vec<f64> {
    a.iter().map(|value| value * scale).collect()
}

fn vmax(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |max, value| max.max(value.abs()))
}

fn add_scaled(base: &[f64], h: f64, terms: &[(f64, &[f64])]) -> Vec<f64> {
    let mut out = base.to_vec();
    for (coef, vector) in terms {
        for i in 0..out.len() {
            out[i] += h * coef * vector[i];
        }
    }
    out
}

fn solve_linear(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>, OdeError> {
    let n = rhs.len();
    for i in 0..n {
        let mut pivot = i;
        for k in (i + 1)..n {
            if matrix[k][i].abs() > matrix[pivot][i].abs() {
                pivot = k;
            }
        }
        if matrix[pivot][i].abs() < 1e-15 {
            return Err(OdeError::SingularMatrix);
        }
        if pivot != i {
            matrix.swap(i, pivot);
            rhs.swap(i, pivot);
        }
        for k in (i + 1)..n {
            let factor = matrix[k][i] / matrix[i][i];
            for j in i..n {
                matrix[k][j] -= factor * matrix[i][j];
            }
            rhs[k] -= factor * rhs[i];
        }
    }
    let mut out = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for (j, value) in out.iter().enumerate().skip(i + 1) {
            sum -= matrix[i][j] * value;
        }
        out[i] = sum / matrix[i][i];
    }
    Ok(out)
}

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
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;
