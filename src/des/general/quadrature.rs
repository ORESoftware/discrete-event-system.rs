//! Rust port of `src/des/general/quadrature.ts`.

use crate::core::RandomSource;
use crate::des::general::prng::Mulberry32;
use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/quadrature.ts",
    "src/des/general/quadrature.rs",
    &[
        "QuadResult is a serde struct; integrands are Fn(f64) -> f64 callbacks.",
        "Thrown TypeScript validation errors become QuadError values.",
        "Gauss-Legendre node tables are fixed Rust match arms over supported orders.",
        "Monte Carlo routines take injected RandomSource implementations, with Mulberry32 default wrappers matching TypeScript.",
    ],
    &[
        "QuadResult",
        "adaptiveSimpson",
        "gaussLegendre",
        "monteCarlo",
        "monteCarloND",
        "simpson",
        "trapezoidal",
    ],
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuadResult {
    pub value: f64,
    pub evaluations: usize,
    pub stderr: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum QuadError {
    #[error("{method}: n must be {condition}, got {n}")]
    InvalidCount {
        method: &'static str,
        condition: &'static str,
        n: usize,
    },
    #[error("gauss_legendre: only n in {{2,3,4,5,7,10}} supported, got {0}")]
    UnsupportedGaussLegendreOrder(usize),
    #[error("monte_carlo_nd: lo and hi dimensions differ ({lo} vs {hi})")]
    DimensionMismatch { lo: usize, hi: usize },
}

pub fn trapezoidal<F>(f: F, a: f64, b: f64, n: usize) -> Result<QuadResult, QuadError>
where
    F: Fn(f64) -> f64,
{
    if n < 1 {
        return Err(QuadError::InvalidCount {
            method: "trapezoidal",
            condition: ">= 1",
            n,
        });
    }
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    Ok(QuadResult {
        value: sum * h,
        evaluations: n + 1,
        stderr: None,
    })
}

pub fn simpson<F>(f: F, a: f64, b: f64, n: usize) -> Result<QuadResult, QuadError>
where
    F: Fn(f64) -> f64,
{
    if n < 2 || !n.is_multiple_of(2) {
        return Err(QuadError::InvalidCount {
            method: "simpson",
            condition: "even and >= 2",
            n,
        });
    }
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
    }
    Ok(QuadResult {
        value: sum * h / 3.0,
        evaluations: n + 1,
        stderr: None,
    })
}

pub fn adaptive_simpson<F>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> Result<QuadResult, QuadError>
where
    F: Fn(f64) -> f64,
{
    let mut evaluations = 0;
    let midpoint = (a + b) / 2.0;
    let fa = f(a);
    let fb = f(b);
    let fm = f(midpoint);
    evaluations += 3;
    let whole = simpson_segment(a, fa, fb, fm, b);
    let value = adaptive_simpson_recurse(
        &f,
        &mut evaluations,
        a,
        fa,
        fb,
        fm,
        b,
        whole,
        tol,
        0,
        max_depth,
    );
    Ok(QuadResult {
        value,
        evaluations,
        stderr: None,
    })
}

pub fn adaptive_simpson_default<F>(f: F, a: f64, b: f64) -> Result<QuadResult, QuadError>
where
    F: Fn(f64) -> f64,
{
    adaptive_simpson(f, a, b, 1e-9, 40)
}

pub fn gauss_legendre<F>(f: F, a: f64, b: f64, n: usize) -> Result<QuadResult, QuadError>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) =
        gauss_legendre_table(n).ok_or(QuadError::UnsupportedGaussLegendreOrder(n))?;
    let half = (b - a) / 2.0;
    let mid = (a + b) / 2.0;
    let mut sum = 0.0;
    for i in 0..n {
        sum += weights[i] * f(half * nodes[i] + mid);
    }
    Ok(QuadResult {
        value: sum * half,
        evaluations: n,
        stderr: None,
    })
}

pub fn monte_carlo<F, R>(
    f: F,
    a: f64,
    b: f64,
    n: usize,
    rng: &mut R,
) -> Result<QuadResult, QuadError>
where
    F: Fn(f64) -> f64,
    R: RandomSource,
{
    if n < 1 {
        return Err(QuadError::InvalidCount {
            method: "monte_carlo",
            condition: ">= 1",
            n,
        });
    }
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for _ in 0..n {
        let x = a + (b - a) * rng.next_f64();
        let y = f(x);
        sum += y;
        sum_squares += y * y;
    }
    let mean = sum / n as f64;
    let variance = sum_squares / n as f64 - mean * mean;
    let width = b - a;
    Ok(QuadResult {
        value: mean * width,
        evaluations: n,
        stderr: Some(variance.max(0.0).sqrt() / (n as f64).sqrt() * width),
    })
}

pub fn monte_carlo_default<F>(f: F, a: f64, b: f64, n: usize) -> Result<QuadResult, QuadError>
where
    F: Fn(f64) -> f64,
{
    let mut rng = Mulberry32::new(1);
    let mut next = || rng.next_f64();
    monte_carlo(f, a, b, n, &mut next)
}

pub fn monte_carlo_nd<F, R>(
    f: F,
    lo: &[f64],
    hi: &[f64],
    n: usize,
    rng: &mut R,
) -> Result<QuadResult, QuadError>
where
    F: Fn(&[f64]) -> f64,
    R: RandomSource,
{
    if lo.len() != hi.len() {
        return Err(QuadError::DimensionMismatch {
            lo: lo.len(),
            hi: hi.len(),
        });
    }
    if n < 1 {
        return Err(QuadError::InvalidCount {
            method: "monte_carlo_nd",
            condition: ">= 1",
            n,
        });
    }
    let mut volume = 1.0;
    for k in 0..lo.len() {
        volume *= hi[k] - lo[k];
    }
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    let mut x = vec![0.0; lo.len()];
    for _ in 0..n {
        for k in 0..lo.len() {
            x[k] = lo[k] + (hi[k] - lo[k]) * rng.next_f64();
        }
        let y = f(&x);
        sum += y;
        sum_squares += y * y;
    }
    let mean = sum / n as f64;
    let variance = sum_squares / n as f64 - mean * mean;
    Ok(QuadResult {
        value: mean * volume,
        evaluations: n,
        stderr: Some(variance.max(0.0).sqrt() / (n as f64).sqrt() * volume),
    })
}

pub fn monte_carlo_nd_default<F>(
    f: F,
    lo: &[f64],
    hi: &[f64],
    n: usize,
) -> Result<QuadResult, QuadError>
where
    F: Fn(&[f64]) -> f64,
{
    let mut rng = Mulberry32::new(1);
    let mut next = || rng.next_f64();
    monte_carlo_nd(f, lo, hi, n, &mut next)
}

fn simpson_segment(a: f64, fa: f64, fb: f64, fm: f64, b: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

#[allow(clippy::too_many_arguments)]
fn adaptive_simpson_recurse<F>(
    f: &F,
    evaluations: &mut usize,
    a: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    b: f64,
    whole: f64,
    tol: f64,
    depth: usize,
    max_depth: usize,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let m = (a + b) / 2.0;
    let lm = (a + m) / 2.0;
    let rm = (m + b) / 2.0;
    let flm = f(lm);
    let frm = f(rm);
    *evaluations += 2;
    let left = simpson_segment(a, fa, fm, flm, m);
    let right = simpson_segment(m, fm, fb, frm, b);
    let err = (left + right - whole) / 15.0;
    if err.abs() <= tol || depth >= max_depth {
        return left + right + err;
    }
    adaptive_simpson_recurse(
        f,
        evaluations,
        a,
        fa,
        fm,
        flm,
        m,
        left,
        tol / 2.0,
        depth + 1,
        max_depth,
    ) + adaptive_simpson_recurse(
        f,
        evaluations,
        m,
        fm,
        fb,
        frm,
        b,
        right,
        tol / 2.0,
        depth + 1,
        max_depth,
    )
}

fn gauss_legendre_table(n: usize) -> Option<(&'static [f64], &'static [f64])> {
    match n {
        2 => Some((&[-0.5773502691896257, 0.5773502691896257], &[1.0, 1.0])),
        3 => Some((
            &[-0.7745966692414834, 0.0, 0.7745966692414834],
            &[0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
        )),
        4 => Some((
            &[
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            &[
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ],
        )),
        5 => Some((
            &[
                -0.906179845938664,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.906179845938664,
            ],
            &[
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ],
        )),
        7 => Some((
            &[
                -0.9491079123427585,
                -0.7415311855993945,
                -0.4058451513773972,
                0.0,
                0.4058451513773972,
                0.7415311855993945,
                0.9491079123427585,
            ],
            &[
                0.1294849661688697,
                0.2797053914892766,
                0.3818300505051189,
                0.4179591836734694,
                0.3818300505051189,
                0.2797053914892766,
                0.1294849661688697,
            ],
        )),
        10 => Some((
            &[
                -0.9739065285171717,
                -0.8650633666889845,
                -0.6794095682990244,
                -0.4333953941292472,
                -0.1488743389816312,
                0.1488743389816312,
                0.4333953941292472,
                0.6794095682990244,
                0.8650633666889845,
                0.9739065285171717,
            ],
            &[
                0.0666713443086881,
                0.1494513491505806,
                0.219086362515982,
                0.2692667193099963,
                0.2955242247147529,
                0.2955242247147529,
                0.2692667193099963,
                0.219086362515982,
                0.1494513491505806,
                0.0666713443086881,
            ],
        )),
        _ => None,
    }
}
