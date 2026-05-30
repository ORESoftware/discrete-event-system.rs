//! Port of `src/des/general/quadrature.ts` (module `des::general::quadrature`).
//!
//! Numerical integration (quadrature) — multiple methods. All rules take an
//! integrand `f`, integration limits `[a, b]`, and a method-specific accuracy
//! parameter, returning a [`QuadResult`] (value + evaluation count, plus an
//! optional Monte-Carlo stderr) so callers can compare cost across methods.
//!
//! Methods:
//!   * [`TrapezoidRule`]       composite trapezoid, `n` subintervals
//!   * [`SimpsonRule`]         composite Simpson 1/3, `n` even
//!   * [`AdaptiveSimpsonRule`] recursive Simpson with error gauge
//!   * [`GaussLegendreRule`]   n-point Gauss-Legendre (n ∈ {2,3,4,5,7,10})
//!   * [`MonteCarloIntegrator`] / [`monte_carlo_nd`]  random sampling
//!
//! Conversion notes (per the TS "RUST MIGRATION" header):
//!   * the integrand `f: (x: number) => number` becomes a generic `F: Fn(f64) -> f64`;
//!   * the multidimensional integrand becomes `F: Fn(&[f64]) -> f64`;
//!   * `mulberry32` is injected as a [`RandomSource`] (`SeededRandom`) instead of a
//!     global; deterministic rules need no RNG;
//!   * all numerics are `f64`; evaluation counts are `u64`;
//!   * `@deprecated` free-fn shims are dropped.

use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::transform::{StatefulTransform, Transform};

/// Result of a quadrature: the estimated integral, the number of integrand
/// evaluations performed, and (Monte-Carlo only) an uncertainty estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadResult {
    pub value: f64,
    pub evaluations: u64,
    /// Optional uncertainty estimate (Monte Carlo only).
    pub stderr: Option<f64>,
}

/// A one-dimensional integrand `f` over the interval `[a, b]`. Bundling the
/// positional `(f, a, b)` arguments keeps the quadrature rules in `Transform`
/// shape; the accuracy parameter (`n` / `tol`) lives on the rule struct.
pub struct Integrand1D<F>
where
    F: Fn(f64) -> f64,
{
    pub f: F,
    pub a: f64,
    pub b: f64,
}

impl<F> Integrand1D<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(f: F, a: f64, b: f64) -> Self {
        Integrand1D { f, a, b }
    }
}

/// A multidimensional integrand `f` over the box `[lo, hi]`.
pub struct IntegrandND<F>
where
    F: Fn(&[f64]) -> f64,
{
    pub f: F,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}

impl<F> IntegrandND<F>
where
    F: Fn(&[f64]) -> f64,
{
    pub fn new(f: F, lo: Vec<f64>, hi: Vec<f64>) -> Self {
        IntegrandND { f, lo, hi }
    }
}

// -----------------------------------------------------------------------------
// Composite trapezoid.
//   ∫_a^b f(x) dx ≈ h · (f(a)/2 + f(a+h) + … + f(b−h) + f(b)/2)
// -----------------------------------------------------------------------------

/// Composite trapezoid rule. Subinterval count `n` is config.
#[derive(Clone, Copy, Debug)]
pub struct TrapezoidRule {
    n: u64,
}

impl TrapezoidRule {
    pub fn new(n: u64) -> Self {
        TrapezoidRule { n }
    }
}

impl<F> Transform<Integrand1D<F>, QuadResult> for TrapezoidRule
where
    F: Fn(f64) -> f64,
{
    fn transform(&self, integrand: Integrand1D<F>) -> QuadResult {
        let Integrand1D { f, a, b } = integrand;
        let n = self.n;
        if n < 1 {
            panic!("trapezoidal: n must be ≥ 1, got {n}");
        }
        let h = (b - a) / n as f64;
        let mut s = 0.5 * (f(a) + f(b));
        for i in 1..n {
            s += f(a + i as f64 * h);
        }
        QuadResult {
            value: s * h,
            evaluations: n + 1,
            stderr: None,
        }
    }
}

// -----------------------------------------------------------------------------
// Composite Simpson 1/3.
//   ∫_a^b f(x) dx ≈ (h/3) · (f₀ + 4·Σ f_odd + 2·Σ f_even + f_n),  n even.
// -----------------------------------------------------------------------------

/// Composite Simpson 1/3 rule. Subinterval count `n` (even) is config.
#[derive(Clone, Copy, Debug)]
pub struct SimpsonRule {
    n: u64,
}

impl SimpsonRule {
    pub fn new(n: u64) -> Self {
        SimpsonRule { n }
    }
}

impl<F> Transform<Integrand1D<F>, QuadResult> for SimpsonRule
where
    F: Fn(f64) -> f64,
{
    fn transform(&self, integrand: Integrand1D<F>) -> QuadResult {
        let Integrand1D { f, a, b } = integrand;
        let n = self.n;
        if n < 2 || !n.is_multiple_of(2) {
            panic!("simpson: n must be even and ≥ 2, got {n}");
        }
        let h = (b - a) / n as f64;
        let mut s = f(a) + f(b);
        for i in 1..n {
            let x = a + i as f64 * h;
            let w = if i % 2 == 0 { 2.0 } else { 4.0 };
            s += w * f(x);
        }
        QuadResult {
            value: s * h / 3.0,
            evaluations: n + 1,
            stderr: None,
        }
    }
}

// -----------------------------------------------------------------------------
// Adaptive Simpson via recursive bisection. Splits the interval where the
// estimated error exceeds tol; uses the standard 15·(S − S_left − S_right)
// error gauge.
// -----------------------------------------------------------------------------

/// Adaptive Simpson via recursive bisection. CONFIG (error tolerance `tol`,
/// recursion depth cap `max_depth`) lives on the struct.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveSimpsonRule {
    tol: f64,
    max_depth: u32,
}

impl AdaptiveSimpsonRule {
    pub fn new(tol: f64, max_depth: u32) -> Self {
        AdaptiveSimpsonRule { tol, max_depth }
    }
}

impl Default for AdaptiveSimpsonRule {
    fn default() -> Self {
        AdaptiveSimpsonRule {
            tol: 1e-9,
            max_depth: 40,
        }
    }
}

/// Simpson estimate over `[a, b]` given endpoint and midpoint samples.
/// Argument order matches the TS `S(a, fa, fb, fm, b)`.
fn simpson_s(a: f64, fa: f64, fb: f64, fm: f64, b: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

#[allow(clippy::too_many_arguments)]
fn adaptive_recurse<F>(
    f: &F,
    a: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    b: f64,
    whole: f64,
    tol: f64,
    depth: u32,
    max_depth: u32,
    evals: &mut u64,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let m = (a + b) / 2.0;
    let lm = (a + m) / 2.0;
    let rm = (m + b) / 2.0;
    let flm = f(lm);
    let frm = f(rm);
    *evals += 2;
    let left = simpson_s(a, fa, fm, flm, m);
    let right = simpson_s(m, fm, fb, frm, b);
    let err = (left + right - whole) / 15.0;
    if depth >= max_depth && err.abs() > tol {
        eprintln!(
            "[quadrature.adaptiveSimpson] max recursion depth {max_depth} reached on [{a}, {b}] \
             with error gauge {:e} > tol {tol}; result may be inaccurate.",
            err.abs()
        );
    }
    if err.abs() <= tol || depth >= max_depth {
        return left + right + err;
    }
    adaptive_recurse(
        f,
        a,
        fa,
        fm,
        flm,
        m,
        left,
        tol / 2.0,
        depth + 1,
        max_depth,
        evals,
    ) + adaptive_recurse(
        f,
        m,
        fm,
        fb,
        frm,
        b,
        right,
        tol / 2.0,
        depth + 1,
        max_depth,
        evals,
    )
}

impl<F> Transform<Integrand1D<F>, QuadResult> for AdaptiveSimpsonRule
where
    F: Fn(f64) -> f64,
{
    fn transform(&self, integrand: Integrand1D<F>) -> QuadResult {
        let Integrand1D { f, a, b } = integrand;
        let mut evals: u64 = 0;
        let m = (a + b) / 2.0;
        let fa = f(a);
        let fb = f(b);
        let fm = f(m);
        evals += 3;
        let whole = simpson_s(a, fa, fb, fm, b);
        let value = adaptive_recurse(
            &f,
            a,
            fa,
            fb,
            fm,
            b,
            whole,
            self.tol,
            0,
            self.max_depth,
            &mut evals,
        );
        QuadResult {
            value,
            evaluations: evals,
            stderr: None,
        }
    }
}

// -----------------------------------------------------------------------------
// Gauss-Legendre quadrature. Nodes & weights for [−1, 1]; transformed to [a, b]
// via x = (b−a)/2·t + (b+a)/2, dx = (b−a)/2 dt.
// -----------------------------------------------------------------------------

/// Returns the `(nodes, weights)` for an `n`-point Gauss-Legendre rule, or
/// `None` if `n` is not supported (n ∈ {2,3,4,5,7,10}).
fn gl_nodes(n: u64) -> Option<(&'static [f64], &'static [f64])> {
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
                -0.906_179_845_938_664,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.906_179_845_938_664,
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
                0.219_086_362_515_982,
                0.2692667193099963,
                0.2955242247147529,
                0.2955242247147529,
                0.2692667193099963,
                0.219_086_362_515_982,
                0.1494513491505806,
                0.0666713443086881,
            ],
        )),
        _ => None,
    }
}

/// n-point Gauss-Legendre quadrature (n ∈ {2,3,4,5,7,10}). `n` is config.
#[derive(Clone, Copy, Debug)]
pub struct GaussLegendreRule {
    n: u64,
}

impl GaussLegendreRule {
    pub fn new(n: u64) -> Self {
        GaussLegendreRule { n }
    }
}

impl Default for GaussLegendreRule {
    fn default() -> Self {
        GaussLegendreRule { n: 5 }
    }
}

impl<F> Transform<Integrand1D<F>, QuadResult> for GaussLegendreRule
where
    F: Fn(f64) -> f64,
{
    fn transform(&self, integrand: Integrand1D<F>) -> QuadResult {
        let Integrand1D { f, a, b } = integrand;
        let n = self.n;
        let (x, w) = gl_nodes(n).unwrap_or_else(|| {
            panic!("gaussLegendre: only n ∈ {{2,3,4,5,7,10}} supported (got {n})")
        });
        let half = (b - a) / 2.0;
        let mid = (a + b) / 2.0;
        let mut s = 0.0;
        for i in 0..n as usize {
            s += w[i] * f(half * x[i] + mid);
        }
        QuadResult {
            value: s * half,
            evaluations: n,
            stderr: None,
        }
    }
}

// -----------------------------------------------------------------------------
// Monte Carlo integration. Returns an unbiased estimate plus stderr. Uses an
// injected seeded PRNG for reproducibility.
// -----------------------------------------------------------------------------

/// Monte Carlo integration over `[a, b]`. CONFIG (sample count `n`, injected
/// RNG) lives on the struct. Stateful because the RNG advances across calls
/// (per the migration note to inject a [`RandomSource`]).
#[derive(Clone, Debug)]
pub struct MonteCarloIntegrator<R = SeededRandom>
where
    R: RandomSource,
{
    n: u64,
    rng: R,
}

impl<R> MonteCarloIntegrator<R>
where
    R: RandomSource,
{
    pub fn new(n: u64, rng: R) -> Self {
        MonteCarloIntegrator { n, rng }
    }
}

impl MonteCarloIntegrator<SeededRandom> {
    /// Default integrator: 10,000 samples seeded with `mulberry32(1)`,
    /// matching the TS defaults.
    pub fn with_default_rng() -> Self {
        MonteCarloIntegrator {
            n: 10_000,
            rng: mulberry32(1),
        }
    }
}

impl<F, R> StatefulTransform<Integrand1D<F>, QuadResult> for MonteCarloIntegrator<R>
where
    F: Fn(f64) -> f64,
    R: RandomSource,
{
    fn transform(&mut self, integrand: Integrand1D<F>) -> QuadResult {
        let Integrand1D { f, a, b } = integrand;
        let n = self.n;
        let mut s = 0.0;
        let mut ss = 0.0;
        for _ in 0..n {
            let x = a + (b - a) * self.rng.next_float();
            let y = f(x);
            s += y;
            ss += y * y;
        }
        let nf = n as f64;
        let mean = s / nf;
        let variance = (ss / nf) - mean * mean;
        let value = mean * (b - a);
        let stderr = (variance.max(0.0) / nf).sqrt() * (b - a);
        QuadResult {
            value,
            evaluations: n,
            stderr: Some(stderr),
        }
    }
}

// -----------------------------------------------------------------------------
// Multidimensional Monte Carlo over a box.
// -----------------------------------------------------------------------------

/// Multidimensional Monte Carlo integration over the box `[lo, hi]`, drawing
/// `n` samples from the injected RNG.
pub fn monte_carlo_nd<F, R>(integrand: &IntegrandND<F>, n: u64, rng: &mut R) -> QuadResult
where
    F: Fn(&[f64]) -> f64,
    R: RandomSource,
{
    let IntegrandND { f, lo, hi } = integrand;
    let d = lo.len();
    let mut volume = 1.0;
    for k in 0..d {
        volume *= hi[k] - lo[k];
    }
    let mut s = 0.0;
    let mut ss = 0.0;
    let mut x = vec![0.0; d];
    for _ in 0..n {
        for k in 0..d {
            x[k] = lo[k] + (hi[k] - lo[k]) * rng.next_float();
        }
        let y = f(&x);
        s += y;
        ss += y * y;
    }
    let nf = n as f64;
    let mean = s / nf;
    let variance = (ss / nf) - mean * mean;
    let value = mean * volume;
    let stderr = (variance.max(0.0) / nf).sqrt() * volume;
    QuadResult {
        value,
        evaluations: n,
        stderr: Some(stderr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simpson_integrates_x_squared_exactly() {
        // Simpson 1/3 is exact for quadratics: ∫_0^1 x² dx = 1/3.
        let rule = SimpsonRule::new(2);
        let res = rule.transform(Integrand1D::new(|x: f64| x * x, 0.0, 1.0));
        assert!((res.value - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(res.evaluations, 3);
        assert_eq!(res.stderr, None);
    }

    #[test]
    fn trapezoid_and_gauss_legendre_approximate_x_squared() {
        let trap = TrapezoidRule::new(1000);
        let t = trap.transform(Integrand1D::new(|x: f64| x * x, 0.0, 1.0));
        assert!((t.value - 1.0 / 3.0).abs() < 1e-5);

        // Gauss-Legendre with 2 nodes is exact for degree ≤ 3 polynomials.
        let gl = GaussLegendreRule::new(2);
        let g = gl.transform(Integrand1D::new(|x: f64| x * x, 0.0, 1.0));
        assert!((g.value - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(g.evaluations, 2);
    }

    #[test]
    fn adaptive_simpson_matches_sine() {
        // ∫_0^π sin(x) dx = 2.
        let rule = AdaptiveSimpsonRule::default();
        let res = rule.transform(Integrand1D::new(f64::sin, 0.0, std::f64::consts::PI));
        assert!((res.value - 2.0).abs() < 1e-8);
    }

    #[test]
    fn monte_carlo_with_fixed_seed_is_close() {
        // ∫_0^1 x² dx = 1/3, loose tolerance for the stochastic estimate.
        let mut mc = MonteCarloIntegrator::new(200_000, mulberry32(1));
        let res = mc.transform(Integrand1D::new(|x: f64| x * x, 0.0, 1.0));
        assert!((res.value - 1.0 / 3.0).abs() < 5e-3);
        assert_eq!(res.evaluations, 200_000);
        assert!(res.stderr.is_some());
    }

    #[test]
    fn monte_carlo_nd_unit_square_volume() {
        // ∫∫_{[0,1]²} 1 dA = 1.
        let mut rng = mulberry32(7);
        let integrand = IntegrandND::new(|_x: &[f64]| 1.0, vec![0.0, 0.0], vec![1.0, 1.0]);
        let res = monte_carlo_nd(&integrand, 10_000, &mut rng);
        assert!((res.value - 1.0).abs() < 1e-9);
    }
}
