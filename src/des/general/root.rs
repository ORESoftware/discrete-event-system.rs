//! Rust port of `src/des/general/root.ts`.

use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/root.ts",
    "src/des/general/root.rs",
    &[
        "RootResult is a serde struct mirroring the TypeScript return shape.",
        "Function and derivative inputs are generic Fn(f64) -> f64 callbacks.",
        "Thrown TypeScript validation paths become RootError variants.",
        "DES-visible solver stations should wrap these free functions in their own mapped modules.",
    ],
    &["RootResult", "bisection", "newton", "secant"],
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RootResult {
    pub root: f64,
    pub iterations: usize,
    pub converged: bool,
    pub final_residual: f64,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RootError {
    #[error("bisection: no sign change on [{a}, {b}]: f(a)={fa}, f(b)={fb}")]
    NoSignChange { a: f64, b: f64, fa: f64, fb: f64 },
    #[error("{method}: tolerance must be positive and finite, got {tol}")]
    InvalidTolerance { method: &'static str, tol: f64 },
    #[error("{method}: max_iter must be >= 1")]
    InvalidMaxIter { method: &'static str },
}

pub fn bisection<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<RootResult, RootError>
where
    F: Fn(f64) -> f64,
{
    validate_args("bisection", tol, max_iter)?;
    let mut fa = f(a);
    let fb = f(b);
    if fa * fb > 0.0 {
        return Err(RootError::NoSignChange { a, b, fa, fb });
    }
    let mut iter = 0;
    while iter < max_iter {
        let midpoint = 0.5 * (a + b);
        let fm = f(midpoint);
        if fm.abs() < tol || (b - a).abs() / 2.0 < tol {
            return Ok(RootResult {
                root: midpoint,
                iterations: iter + 1,
                converged: true,
                final_residual: fm.abs(),
            });
        }
        if fa * fm < 0.0 {
            b = midpoint;
        } else {
            a = midpoint;
            fa = fm;
        }
        iter += 1;
    }
    let midpoint = 0.5 * (a + b);
    Ok(RootResult {
        root: midpoint,
        iterations: iter,
        converged: false,
        final_residual: f(midpoint).abs(),
    })
}

pub fn newton<F, D>(
    f: F,
    df: D,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<RootResult, RootError>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    validate_args("newton", tol, max_iter)?;
    let mut x = x0;
    let mut fx = f(x);
    for i in 0..max_iter {
        if fx.abs() < tol {
            return Ok(RootResult {
                root: x,
                iterations: i,
                converged: true,
                final_residual: fx.abs(),
            });
        }
        let dfx = df(x);
        if dfx == 0.0 || !dfx.is_finite() {
            break;
        }
        let step = fx / dfx;
        let mut alpha = 1.0;
        let mut x_next = x - alpha * step;
        let mut f_next = f(x_next);
        let mut damp = 0;
        while f_next.abs() > fx.abs() && damp < 20 {
            alpha *= 0.5;
            x_next = x - alpha * step;
            f_next = f(x_next);
            damp += 1;
        }
        x = x_next;
        fx = f_next;
    }
    Ok(RootResult {
        root: x,
        iterations: max_iter,
        converged: fx.abs() < tol,
        final_residual: fx.abs(),
    })
}

pub fn secant<F>(
    f: F,
    mut x0: f64,
    mut x1: f64,
    tol: f64,
    max_iter: usize,
) -> Result<RootResult, RootError>
where
    F: Fn(f64) -> f64,
{
    validate_args("secant", tol, max_iter)?;
    let mut f0 = f(x0);
    let mut f1 = f(x1);
    for i in 0..max_iter {
        if f1.abs() < tol {
            return Ok(RootResult {
                root: x1,
                iterations: i,
                converged: true,
                final_residual: f1.abs(),
            });
        }
        if f0 == f1 {
            break;
        }
        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x1);
    }
    Ok(RootResult {
        root: x1,
        iterations: max_iter,
        converged: f1.abs() < tol,
        final_residual: f1.abs(),
    })
}

fn validate_args(method: &'static str, tol: f64, max_iter: usize) -> Result<(), RootError> {
    if !tol.is_finite() || tol <= 0.0 {
        return Err(RootError::InvalidTolerance { method, tol });
    }
    if max_iter < 1 {
        return Err(RootError::InvalidMaxIter { method });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_finders_converge_on_sqrt_two() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        assert!((bisection(f, 0.0, 2.0, 1e-12, 200).unwrap().root - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((newton(f, df, 1.0, 1e-12, 100).unwrap().root - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((secant(f, 0.0, 2.0, 1e-12, 100).unwrap().root - 2.0_f64.sqrt()).abs() < 1e-10);
    }
}
