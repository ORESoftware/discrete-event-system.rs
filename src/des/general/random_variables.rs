//! Port of `src/des/general/random-variables.ts` (module `des::general::random_variables`).
//!
//! Random-variable toolkit: PMF algebra (convolution, Poisson-binomial,
//! competing risks, discretisation) plus distribution samplers. The PMF math is
//! pure and deterministic; the samplers are the consumer side of the
//! `prng.ts` ↔ capability port — each `sample_*` takes an injected
//! [`RandomSource`] (rather than the TS `rng: () => number` closure) so behaviour
//! stays reproducible.
//!
//! Conversion notes (per the TS "RUST MIGRATION" header):
//!   * `class … extends PureTransform` → struct + `impl Transform` (PMF math).
//!   * `PMF`s are `ReadonlyArray<number>` → `&[f64]`; results are fresh `Vec<f64>`.
//!   * `discretisePDF`'s `(x) => number` closure → generic `F: Fn(f64) -> f64`.
//!   * Every `sample*` injects a `&mut impl RandomSource` in place of `rng()`.
//!   * `throw` on bad mass / params is an invariant violation → `panic!`.
//!   * `@deprecated` free-fn shims around the PMF transforms are dropped; the
//!     public samplers are kept as free fns (their TS form was also `@deprecated`
//!     but they are the primary public API and are imported elsewhere).

use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::transform::Transform;

// -----------------------------------------------------------------------------
// PMF utilities
// -----------------------------------------------------------------------------

/// Total probability mass `Σ p_k` (for sanity checks).
pub fn pmf_total_mass(pmf: &[f64]) -> f64 {
    let mut s = 0.0;
    for &p in pmf {
        s += p;
    }
    s
}

/// Normalise a PMF by dividing by its total mass; tolerates tiny drift.
///
/// # Panics
/// Panics on a zero-mass (or negative-mass) PMF, mirroring the TS `throw`.
pub fn normalize_pmf(pmf: &[f64]) -> Vec<f64> {
    let m = pmf_total_mass(pmf);
    if m <= 0.0 {
        panic!("cannot normalise zero-mass PMF");
    }
    let mut out = vec![0.0; pmf.len()];
    for i in 0..pmf.len() {
        out[i] = pmf[i] / m;
    }
    out
}

/// Mean `Σ k · p_k`.
pub fn mean_from_pmf(pmf: &[f64]) -> f64 {
    let mut s = 0.0;
    for k in 0..pmf.len() {
        s += k as f64 * pmf[k];
    }
    s
}

/// Variance `E[X²] − (E[X])²`.
pub fn variance_from_pmf(pmf: &[f64]) -> f64 {
    let mut m = 0.0;
    let mut m2 = 0.0;
    for k in 0..pmf.len() {
        let kf = k as f64;
        m += kf * pmf[k];
        m2 += kf * kf * pmf[k];
    }
    m2 - m * m
}

// -----------------------------------------------------------------------------
// Convolution
// -----------------------------------------------------------------------------

/// Shared O(|p|·|q|) discrete linear convolution kernel used by the convolution
/// transforms below.
fn convolve(p: &[f64], q: &[f64]) -> Vec<f64> {
    if p.is_empty() || q.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0; p.len() + q.len() - 1];
    for i in 0..p.len() {
        if p[i] == 0.0 {
            continue;
        }
        for j in 0..q.len() {
            out[i + j] += p[i] * q[j];
        }
    }
    out
}

/// The two PMFs to convolve.
pub struct ConvolvePair<'a> {
    pub p: &'a [f64],
    pub q: &'a [f64],
}

/// Discrete linear convolution of two PMFs. If `P(X = i) = p[i]` and
/// `P(Y = j) = q[j]` with `X ⊥ Y`, the result is `P(X + Y = k)` for
/// `k = 0 .. (|p| + |q| − 2)`. Cost: `O(|p| · |q|)`.
pub struct DiscreteConvolve;

impl<'a> Transform<ConvolvePair<'a>, Vec<f64>> for DiscreteConvolve {
    fn transform(&self, ConvolvePair { p, q }: ConvolvePair<'a>) -> Vec<f64> {
        convolve(p, q)
    }
}

/// Iterative convolution: `P(Σ X_k = n)` given each `X_k`'s PMF.
pub struct DiscreteConvolveMany;

impl<'a> Transform<&'a [Vec<f64>], Vec<f64>> for DiscreteConvolveMany {
    fn transform(&self, pmfs: &'a [Vec<f64>]) -> Vec<f64> {
        if pmfs.is_empty() {
            return vec![1.0];
        }
        let mut acc: Vec<f64> = pmfs[0].clone();
        for i in 1..pmfs.len() {
            acc = convolve(&acc, &pmfs[i]);
        }
        acc
    }
}

/// `n`-fold self-convolution: PMF of `X_1 + … + X_n` where each `X_k` shares the
/// distribution `pmf`. Uses repeated squaring (`log₂ n` convolutions).
pub struct DiscreteConvolveSelf {
    n: u32,
}

impl DiscreteConvolveSelf {
    pub fn new(n: u32) -> Self {
        DiscreteConvolveSelf { n }
    }
}

impl<'a> Transform<&'a [f64], Vec<f64>> for DiscreteConvolveSelf {
    fn transform(&self, pmf: &'a [f64]) -> Vec<f64> {
        // The TS `n < 0 || !Number.isInteger(n)` guard is enforced by the `u32`
        // field type.
        let n = self.n;
        if n == 0 {
            return vec![1.0];
        }
        let mut result: Option<Vec<f64>> = None;
        let mut base: Vec<f64> = pmf.to_vec();
        let mut m = n;
        while m > 0 {
            if m & 1 == 1 {
                result = Some(match result {
                    None => base.clone(),
                    Some(r) => convolve(&r, &base),
                });
            }
            m >>= 1;
            if m > 0 {
                base = convolve(&base, &base);
            }
        }
        result.expect("self-convolution accumulated a result for n > 0")
    }
}

// -----------------------------------------------------------------------------
// Common PMFs
// -----------------------------------------------------------------------------

/// Bernoulli(p) PMF: `[1−p, p]`.
///
/// # Panics
/// Panics unless `0 ≤ p ≤ 1`.
pub fn bernoulli_pmf(p: f64) -> Vec<f64> {
    if !(0.0..=1.0).contains(&p) {
        panic!("bad p {p}");
    }
    vec![1.0 - p, p]
}

/// Binomial(n, p) closed-form PMF. Stable for `n ≤ ~1500` in `f64`. The success
/// probability `p` is the configuration; `n` (trial count) is the input.
pub struct BinomialPMF {
    p: f64,
}

impl BinomialPMF {
    pub fn new(p: f64) -> Self {
        BinomialPMF { p }
    }
}

impl Transform<u32, Vec<f64>> for BinomialPMF {
    fn transform(&self, n: u32) -> Vec<f64> {
        let p = self.p;
        if !(0.0..=1.0).contains(&p) {
            panic!("bad p {p}");
        }
        if n == 0 {
            return vec![1.0];
        }
        // Recursive formula: P(k+1) = P(k) · (n-k)/(k+1) · p/(1-p)
        let mut out = vec![0.0; (n + 1) as usize];
        if p == 0.0 {
            out[0] = 1.0;
            return out;
        }
        if p == 1.0 {
            out[n as usize] = 1.0;
            return out;
        }
        out[0] = (1.0 - p).powi(n as i32);
        let r = p / (1.0 - p);
        for k in 0..n {
            out[(k + 1) as usize] = out[k as usize] * (n - k) as f64 * r / (k + 1) as f64;
        }
        out
    }
}

/// Poisson-binomial PMF: `P(Σ Bᵢ = k)` where `Bᵢ ~ Bernoulli(probs[i])` are
/// independent with possibly different probabilities, computed exactly by
/// convolving the per-item Bernoulli PMFs. Delegates to the closed-form binomial
/// when all probabilities are equal (more numerically stable).
pub struct PoissonBinomialPMF;

impl<'a> Transform<&'a [f64], Vec<f64>> for PoissonBinomialPMF {
    fn transform(&self, probs: &'a [f64]) -> Vec<f64> {
        if probs.is_empty() {
            return vec![1.0];
        }
        // Detect uniform p — closed-form binomial is more numerically stable.
        let mut all_equal = true;
        for i in 1..probs.len() {
            if (probs[i] - probs[0]).abs() > 1e-15 {
                all_equal = false;
                break;
            }
        }
        if all_equal {
            return BinomialPMF::new(probs[0]).transform(probs.len() as u32);
        }
        // Iterative convolution of Bernoulli(p_i) PMFs.
        let mut pmf: Vec<f64> = vec![1.0];
        for i in 0..probs.len() {
            let p = probs[i];
            if !(0.0..=1.0).contains(&p) {
                panic!("bad p[{i}] {p}");
            }
            let mut next = vec![0.0; pmf.len() + 1];
            for k in 0..pmf.len() {
                next[k] += pmf[k] * (1.0 - p);
                next[k + 1] += pmf[k] * p;
            }
            pmf = next;
        }
        pmf
    }
}

// -----------------------------------------------------------------------------
// Competing-risks discrete-time transition probabilities
// -----------------------------------------------------------------------------

/// Given `K` independent continuous rates `λ₁ … λ_K` and a time step `dt`,
/// compute the exact discrete-time first-event probabilities:
///
/// ```text
///   p_no = exp(−Λ·dt)            where Λ = Σ λ_k
///   p_k  = (λ_k / Λ) · (1 − p_no)
/// ```
///
/// Returns `[p_no, p_1, …, p_K]`, summing to 1 modulo float drift.
pub struct CompetingRisks {
    dt: f64,
}

impl CompetingRisks {
    pub fn new(dt: f64) -> Self {
        CompetingRisks { dt }
    }
}

impl<'a> Transform<&'a [f64], Vec<f64>> for CompetingRisks {
    fn transform(&self, rates: &'a [f64]) -> Vec<f64> {
        let dt = self.dt;
        if dt < 0.0 {
            panic!("bad dt {dt}");
        }
        let mut total = 0.0;
        for i in 0..rates.len() {
            if rates[i] < 0.0 {
                panic!("bad rate[{i}] {}", rates[i]);
            }
            total += rates[i];
        }
        if total == 0.0 {
            let mut out = vec![0.0; rates.len() + 1];
            out[0] = 1.0;
            return out;
        }
        let p_no = (-total * dt).exp();
        let p_any = 1.0 - p_no;
        let mut out = vec![0.0; rates.len() + 1];
        out[0] = p_no;
        for i in 0..rates.len() {
            out[i + 1] = (rates[i] / total) * p_any;
        }
        out
    }
}

// -----------------------------------------------------------------------------
// Sampling
// -----------------------------------------------------------------------------

/// Categorical sampler: given `probs` summing to 1 (within float drift), draw an
/// index in `0..probs.len()` with the given probabilities. Linear search,
/// `O(K)`.
pub fn sample_categorical(rng: &mut impl RandomSource, probs: &[f64]) -> usize {
    let r = rng.next_float();
    let mut cum = 0.0;
    for i in 0..probs.len() {
        cum += probs[i];
        if r <= cum {
            return i;
        }
    }
    probs.len() - 1
}

/// Sample a single integer outcome from a PMF over `{0, …, |pmf|−1}`.
/// Equivalent to [`sample_categorical`]; the name is clearer when the array is a
/// numerical RV's PMF.
pub fn sample_from_pmf(rng: &mut impl RandomSource, pmf: &[f64]) -> usize {
    sample_categorical(rng, pmf)
}

// -----------------------------------------------------------------------------
// Continuous-distribution samplers (used by contact-based simulations).
// -----------------------------------------------------------------------------

/// Draw a Poisson-distributed integer (returned as `f64`) with mean `lambda`.
/// Uses Knuth's algorithm for `lambda < 30`, and a normal approximation with
/// continuity correction (Box–Muller draw) for `lambda ≥ 30`.
///
/// # Panics
/// Panics if `lambda < 0`.
pub fn sample_poisson(rng: &mut impl RandomSource, lambda: f64) -> f64 {
    if lambda < 0.0 {
        panic!("bad lambda {lambda}");
    }
    if lambda == 0.0 {
        return 0.0;
    }
    if lambda < 30.0 {
        let l = (-lambda).exp();
        let mut k = 0_i64;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.next_float();
            if p <= l {
                return (k - 1) as f64;
            }
        }
    }
    // Normal approximation with continuity correction; mean = variance = lambda.
    let u1 = 1.0 - rng.next_float();
    let u2 = rng.next_float();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let x = lambda + lambda.sqrt() * z + 0.5;
    x.floor().max(0.0)
}

/// Draw an Exponential(rate) sample with `X = −ln(U) / rate`, `U ~ (0, 1]`.
///
/// # Panics
/// Panics unless `rate > 0`.
pub fn sample_exponential(rng: &mut impl RandomSource, rate: f64) -> f64 {
    if !(rate > 0.0) {
        panic!("bad rate {rate}");
    }
    let u = 1.0 - rng.next_float(); // Uniform(0, 1] (avoid log(0))
    -u.ln() / rate
}

/// Draw a Gamma(shape, scale) sample using Marsaglia & Tsang's (2000) method.
/// Mean = `shape · scale`, variance = `shape · scale²`.
///
/// # Panics
/// Panics unless `shape > 0` and `scale > 0`.
pub fn sample_gamma(rng: &mut impl RandomSource, shape: f64, scale: f64) -> f64 {
    if !(shape > 0.0) || !(scale > 0.0) {
        panic!("bad shape/scale {shape}/{scale}");
    }
    if shape < 1.0 {
        // Boost: sample Gamma(shape+1, scale) and scale by U^{1/shape}.
        let g = sample_gamma(rng, shape + 1.0, scale);
        let u = 1.0 - rng.next_float();
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            let u1 = 1.0 - rng.next_float();
            let u2 = rng.next_float();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            x = z;
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = rng.next_float();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v * scale;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

// -----------------------------------------------------------------------------
// Continuous: simple grid-based discretisation for completeness.
// -----------------------------------------------------------------------------

/// Discretise a continuous PDF `f` on the regular grid `x = x0 + i·h`
/// (`i = 0..n−1`), returning approximate point masses `pmf[i] ≈ f(x0 + i·h) · h`.
/// The caller chooses a wide-enough grid and normalises.
pub struct DiscretisePDF {
    x0: f64,
    h: f64,
    n: usize,
}

impl DiscretisePDF {
    pub fn new(x0: f64, h: f64, n: usize) -> Self {
        DiscretisePDF { x0, h, n }
    }
}

impl<F> Transform<F, Vec<f64>> for DiscretisePDF
where
    F: Fn(f64) -> f64,
{
    fn transform(&self, f: F) -> Vec<f64> {
        let mut out = vec![0.0; self.n];
        for i in 0..self.n {
            out[i] = f(self.x0 + i as f64 * self.h) * self.h;
        }
        out
    }
}

// -----------------------------------------------------------------------------
// Additional distributions: continuous, discrete, and mixed continuous–discrete.
//
// These extend the sampler family above. Each `sample_*` injects a
// `&mut impl RandomSource` (so draws are reproducible under a seed) and follows
// the module convention of panicking on invalid parameters. Integer-valued
// draws are returned as `f64` (like [`sample_poisson`]) unless the support is a
// bounded integer range, where an `i64` is the natural type
// ([`sample_discrete_uniform`]).
//
// "Mixed continuous–discrete" means the distribution places positive
// probability on one or more discrete *atoms* and spreads the rest over a
// continuous density — see [`sample_zero_inflated_exponential`] and
// [`sample_censored_normal`].
// -----------------------------------------------------------------------------

// ---- Gaussian helpers (useful on their own; also back the samplers below). ---

/// Standard-normal PDF `φ(x) = e^{−x²/2} / √(2π)`.
pub fn std_normal_pdf(x: f64) -> f64 {
    (-(0.5 * x * x)).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard-normal CDF `Φ(x)`, via the Abramowitz–Stegun 7.1.26 `erf`
/// approximation (absolute error < 1.5e-7). Handy for the atom masses of the
/// censored-normal distribution below.
pub fn std_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// `erf(x)` — Abramowitz & Stegun 7.1.26 rational approximation.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-(ax * ax)).exp();
    sign * y
}

// ---- Continuous ------------------------------------------------------------

/// Normal(μ, σ) (Gaussian). Support ℝ; mean `μ`, variance `σ²`. Uses the
/// Box–Muller draw exposed by [`RandomSource::next_gaussian`].
///
/// # Panics
/// Panics if `sigma < 0`.
pub fn sample_normal(rng: &mut impl RandomSource, mu: f64, sigma: f64) -> f64 {
    if sigma < 0.0 {
        panic!("bad sigma {sigma}");
    }
    mu + sigma * rng.next_gaussian()
}

/// Continuous Uniform(a, b). Support `[a, b)`; mean `(a+b)/2`, variance
/// `(b−a)²/12`.
///
/// # Panics
/// Panics unless `b > a`.
pub fn sample_uniform(rng: &mut impl RandomSource, a: f64, b: f64) -> f64 {
    if !(b > a) {
        panic!("bad interval [{a}, {b})");
    }
    a + (b - a) * rng.next_float()
}

/// Weibull(shape `k`, scale `λ`) by inverse CDF: `λ · (−ln U)^{1/k}`,
/// `U ~ (0, 1]`. Support `[0, ∞)`; mean `λ · Γ(1 + 1/k)`. With `k = 1` this is
/// Exponential with mean `λ`.
///
/// # Panics
/// Panics unless `shape > 0` and `scale > 0`.
pub fn sample_weibull(rng: &mut impl RandomSource, shape: f64, scale: f64) -> f64 {
    if !(shape > 0.0) || !(scale > 0.0) {
        panic!("bad shape/scale {shape}/{scale}");
    }
    let u = 1.0 - rng.next_float(); // (0, 1] avoids ln(0)
    scale * (-u.ln()).powf(1.0 / shape)
}

/// Lognormal(μ, σ): `exp(Normal(μ, σ))`. Support `(0, ∞)`; mean
/// `exp(μ + σ²/2)`.
///
/// # Panics
/// Panics if `sigma < 0`.
pub fn sample_lognormal(rng: &mut impl RandomSource, mu: f64, sigma: f64) -> f64 {
    sample_normal(rng, mu, sigma).exp()
}

// ---- Discrete --------------------------------------------------------------

/// Geometric(p) — number of *failures before the first success*. Support
/// `{0, 1, 2, …}`; mean `(1−p)/p`, variance `(1−p)/p²`. Inverse-CDF:
/// `⌊ln(U) / ln(1−p)⌋`. Returned as `f64` (like [`sample_poisson`]).
///
/// # Panics
/// Panics unless `0 < p ≤ 1`.
pub fn sample_geometric(rng: &mut impl RandomSource, p: f64) -> f64 {
    if !(p > 0.0 && p <= 1.0) {
        panic!("bad p {p}");
    }
    if p >= 1.0 {
        return 0.0;
    }
    let u = 1.0 - rng.next_float(); // (0, 1]
    (u.ln() / (1.0 - p).ln()).floor()
}

/// Discrete Uniform over the inclusive integer range `{lo, …, hi}`. Mean
/// `(lo+hi)/2`.
///
/// # Panics
/// Panics unless `hi >= lo`.
pub fn sample_discrete_uniform(rng: &mut impl RandomSource, lo: i64, hi: i64) -> i64 {
    if hi < lo {
        panic!("bad range [{lo}, {hi}]");
    }
    rng.next_int(lo, hi + 1) // next_int is half-open [min, max)
}

/// Negative Binomial(r, p) — number of *failures before the r-th success*,
/// drawn as a Gamma–Poisson mixture: `λ ~ Gamma(r, (1−p)/p)`, then
/// `Poisson(λ)` (so `r` may be any positive real). Support `{0, 1, 2, …}`;
/// mean `r(1−p)/p`. Returned as `f64`.
///
/// # Panics
/// Panics unless `r > 0` and `0 < p ≤ 1`.
pub fn sample_negative_binomial(rng: &mut impl RandomSource, r: f64, p: f64) -> f64 {
    if !(r > 0.0) {
        panic!("bad r {r}");
    }
    if !(p > 0.0 && p <= 1.0) {
        panic!("bad p {p}");
    }
    if p >= 1.0 {
        return 0.0;
    }
    let lambda = sample_gamma(rng, r, (1.0 - p) / p);
    sample_poisson(rng, lambda)
}

// ---- Mixed continuous–discrete ---------------------------------------------

/// Zero-inflated Exponential — a **mixed continuous–discrete** distribution:
/// with probability `pi` the value is exactly `0` (a discrete atom), otherwise
/// it is a continuous `Exponential(rate)` draw on `(0, ∞)`. Mean
/// `(1 − pi)/rate`. Models e.g. service times where a fraction of jobs are
/// served instantly.
///
/// # Panics
/// Panics unless `0 ≤ pi ≤ 1` and `rate > 0`.
pub fn sample_zero_inflated_exponential(rng: &mut impl RandomSource, pi: f64, rate: f64) -> f64 {
    if !(0.0..=1.0).contains(&pi) {
        panic!("bad pi {pi}");
    }
    if !(rate > 0.0) {
        panic!("bad rate {rate}");
    }
    if rng.next_float() < pi {
        0.0 // discrete atom at 0
    } else {
        sample_exponential(rng, rate) // continuous part
    }
}

/// Censored (Tobit) Normal — a **mixed continuous–discrete** distribution:
/// `Normal(μ, σ)` clamped to `[lo, hi]`. Clamping puts a discrete atom at `lo`
/// (mass `Φ((lo−μ)/σ)`) and at `hi` (mass `1 − Φ((hi−μ)/σ)`), with a continuous
/// density in between. See [`std_normal_cdf`] for the atom masses. Models a
/// sensor that saturates outside its measurement range.
///
/// # Panics
/// Panics unless `sigma > 0` and `hi > lo`.
pub fn sample_censored_normal(
    rng: &mut impl RandomSource,
    mu: f64,
    sigma: f64,
    lo: f64,
    hi: f64,
) -> f64 {
    if !(sigma > 0.0) {
        panic!("bad sigma {sigma}");
    }
    if !(hi > lo) {
        panic!("bad bounds [{lo}, {hi}]");
    }
    sample_normal(rng, mu, sigma).clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::capabilities::SeededRandom;

    #[test]
    fn samplers_are_deterministic_for_a_fixed_seed() {
        let mut a = SeededRandom::new(1234);
        let mut b = SeededRandom::new(1234);
        let draws_a: Vec<f64> = (0..200).map(|_| sample_gamma(&mut a, 2.5, 1.5)).collect();
        let draws_b: Vec<f64> = (0..200).map(|_| sample_gamma(&mut b, 2.5, 1.5)).collect();
        assert_eq!(draws_a, draws_b);
    }

    #[test]
    fn gamma_empirical_mean_matches_theory() {
        // Mean of Gamma(shape, scale) = shape * scale.
        let (shape, scale) = (3.0, 2.0);
        let mut rng = SeededRandom::new(42);
        let n = 100_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += sample_gamma(&mut rng, shape, scale);
        }
        let mean = sum / n as f64;
        let theoretical = shape * scale; // 6.0
        assert!(
            (mean - theoretical).abs() < 0.15,
            "gamma empirical mean {mean} vs theory {theoretical}"
        );
    }

    #[test]
    fn exponential_and_poisson_means_match_theory() {
        let mut rng = SeededRandom::new(7);
        let n = 100_000;

        // Exponential mean = 1 / rate.
        let rate = 0.5;
        let mut esum = 0.0;
        for _ in 0..n {
            esum += sample_exponential(&mut rng, rate);
        }
        let emean = esum / n as f64;
        assert!((emean - 1.0 / rate).abs() < 0.1, "exp mean {emean} vs 2.0");

        // Poisson mean = lambda (uses Knuth branch for lambda < 30).
        let lambda = 4.0;
        let mut psum = 0.0;
        for _ in 0..n {
            psum += sample_poisson(&mut rng, lambda);
        }
        let pmean = psum / n as f64;
        assert!(
            (pmean - lambda).abs() < 0.1,
            "poisson mean {pmean} vs {lambda}"
        );
    }

    #[test]
    fn competing_risks_and_poisson_binomial_identities() {
        // Competing risks: probabilities sum to 1 and split by rate share.
        let out = CompetingRisks::new(0.3).transform(&[1.0, 3.0]);
        assert!((pmf_total_mass(&out) - 1.0).abs() < 1e-12);
        // Conditional split of the two events is 1:3.
        assert!((out[2] / out[1] - 3.0).abs() < 1e-9);

        // Poisson-binomial with equal probs equals the binomial closed form.
        let pb = PoissonBinomialPMF.transform(&[0.5, 0.5, 0.5]);
        let bin = BinomialPMF::new(0.5).transform(3);
        assert_eq!(pb.len(), bin.len());
        for (a, b) in pb.iter().zip(bin.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    // -------------------------------------------------------------------------
    // New distributions: continuous, discrete, and mixed continuous–discrete.
    // -------------------------------------------------------------------------

    /// Monte-Carlo mean of `f` over `n` draws from a fresh-seeded stream.
    fn mc_mean(seed: u32, n: usize, mut f: impl FnMut(&mut SeededRandom) -> f64) -> f64 {
        let mut rng = SeededRandom::new(seed);
        let mut sum = 0.0;
        for _ in 0..n {
            sum += f(&mut rng);
        }
        sum / n as f64
    }

    #[test]
    fn new_samplers_are_deterministic_for_a_fixed_seed() {
        let draw = |r: &mut SeededRandom| {
            (
                sample_normal(r, 1.0, 2.0),
                sample_uniform(r, -3.0, 5.0),
                sample_weibull(r, 1.5, 2.0),
                sample_lognormal(r, 0.0, 0.5),
                sample_geometric(r, 0.25),
                sample_discrete_uniform(r, 2, 9) as f64,
                sample_negative_binomial(r, 3.0, 0.4),
                sample_zero_inflated_exponential(r, 0.3, 0.5),
                sample_censored_normal(r, 0.0, 1.0, -1.0, 1.0),
            )
        };
        let mut a = SeededRandom::new(2024);
        let mut b = SeededRandom::new(2024);
        assert_eq!(draw(&mut a), draw(&mut b));
    }

    // ---- continuous ----

    #[test]
    fn normal_mean_and_variance_match_theory() {
        let (mu, sigma, n) = (3.0, 2.0, 200_000);
        let mut rng = SeededRandom::new(11);
        let (mut s, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let x = sample_normal(&mut rng, mu, sigma);
            s += x;
            s2 += x * x;
        }
        let m = s / n as f64;
        let var = s2 / n as f64 - m * m;
        assert!((m - mu).abs() < 0.03, "normal mean {m} vs {mu}");
        assert!((var - sigma * sigma).abs() < 0.15, "normal var {var} vs 4.0");
    }

    #[test]
    fn uniform_stays_in_range_with_correct_mean() {
        let (a, b, n) = (-2.0, 6.0, 100_000);
        let mut rng = SeededRandom::new(5);
        let mut s = 0.0;
        for _ in 0..n {
            let x = sample_uniform(&mut rng, a, b);
            assert!(x >= a && x < b, "uniform out of range: {x}");
            s += x;
        }
        assert!((s / n as f64 - (a + b) / 2.0).abs() < 0.05);
    }

    #[test]
    fn weibull_k1_reduces_to_exponential_mean() {
        // Weibull(k = 1, λ) is Exponential with mean λ.
        let lambda = 2.5;
        let m = mc_mean(99, 200_000, |r| sample_weibull(r, 1.0, lambda));
        assert!((m - lambda).abs() < 0.05, "weibull mean {m} vs {lambda}");
        // Support is non-negative for any shape.
        let mut r2 = SeededRandom::new(1);
        for _ in 0..1000 {
            assert!(sample_weibull(&mut r2, 2.0, 1.5) >= 0.0);
        }
    }

    #[test]
    fn lognormal_mean_matches_theory_and_is_positive() {
        let (mu, sigma) = (0.0_f64, 0.5_f64);
        let theory = (mu + 0.5 * sigma * sigma).exp();
        let m = mc_mean(77, 300_000, |r| sample_lognormal(r, mu, sigma));
        assert!((m - theory).abs() < 0.05, "lognormal mean {m} vs {theory}");
        let mut r2 = SeededRandom::new(2);
        for _ in 0..1000 {
            assert!(sample_lognormal(&mut r2, 0.0, 1.0) > 0.0);
        }
    }

    // ---- discrete ----

    #[test]
    fn geometric_is_nonneg_integer_with_correct_mean() {
        let p = 0.25;
        let m = mc_mean(3, 200_000, |r| sample_geometric(r, p));
        assert!((m - (1.0 - p) / p).abs() < 0.1, "geometric mean {m}");
        let mut r2 = SeededRandom::new(2);
        for _ in 0..1000 {
            let g = sample_geometric(&mut r2, 0.4);
            assert!(g >= 0.0 && g.fract() == 0.0, "geometric not a count: {g}");
        }
    }

    #[test]
    fn discrete_uniform_covers_inclusive_range() {
        let (lo, hi, n) = (2_i64, 9_i64, 100_000);
        let mut rng = SeededRandom::new(8);
        let (mut seen_lo, mut seen_hi, mut s) = (false, false, 0.0);
        for _ in 0..n {
            let x = sample_discrete_uniform(&mut rng, lo, hi);
            assert!(x >= lo && x <= hi, "discrete-uniform out of range: {x}");
            seen_lo |= x == lo;
            seen_hi |= x == hi;
            s += x as f64;
        }
        assert!(seen_lo && seen_hi, "both inclusive endpoints must be reachable");
        assert!((s / n as f64 - (lo + hi) as f64 / 2.0).abs() < 0.05);
    }

    #[test]
    fn negative_binomial_mean_matches_theory() {
        let (r, p) = (4.0, 0.4);
        let theory = r * (1.0 - p) / p;
        let m = mc_mean(444, 100_000, |g| sample_negative_binomial(g, r, p));
        assert!((m - theory).abs() < 0.2, "neg-binomial mean {m} vs {theory}");
    }

    // ---- mixed continuous–discrete ----

    #[test]
    fn zero_inflated_exponential_has_atom_and_correct_mean() {
        let (pi, rate, n) = (0.4, 0.5, 300_000);
        let mut rng = SeededRandom::new(20);
        let (mut zeros, mut s) = (0usize, 0.0);
        for _ in 0..n {
            let x = sample_zero_inflated_exponential(&mut rng, pi, rate);
            assert!(x >= 0.0);
            if x == 0.0 {
                zeros += 1;
            }
            s += x;
        }
        let zero_frac = zeros as f64 / n as f64;
        assert!((zero_frac - pi).abs() < 0.02, "atom mass {zero_frac} vs {pi}");
        assert!(
            (s / n as f64 - (1.0 - pi) / rate).abs() < 0.05,
            "mean vs (1-pi)/rate"
        );
    }

    #[test]
    fn censored_normal_is_bounded_with_atoms_at_both_ends() {
        let (mu, sigma, lo, hi, n) = (0.0, 1.0, -0.5, 0.5, 300_000);
        let mut rng = SeededRandom::new(31);
        let (mut at_lo, mut at_hi) = (0usize, 0usize);
        for _ in 0..n {
            let x = sample_censored_normal(&mut rng, mu, sigma, lo, hi);
            assert!(x >= lo && x <= hi, "censored-normal out of range: {x}");
            if x == lo {
                at_lo += 1;
            }
            if x == hi {
                at_hi += 1;
            }
        }
        // Atom masses: P(X = lo) = Φ((lo−μ)/σ); P(X = hi) = 1 − Φ((hi−μ)/σ).
        let mass_lo = std_normal_cdf((lo - mu) / sigma);
        let mass_hi = 1.0 - std_normal_cdf((hi - mu) / sigma);
        assert!(at_lo > 0 && at_hi > 0, "both atoms must be present");
        assert!((at_lo as f64 / n as f64 - mass_lo).abs() < 0.02, "lo atom mass");
        assert!((at_hi as f64 / n as f64 - mass_hi).abs() < 0.02, "hi atom mass");
    }

    #[test]
    fn std_normal_cdf_matches_known_values() {
        assert!((std_normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((std_normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((std_normal_cdf(-1.0) - 0.158_655).abs() < 1e-3);
        // PDF integrates (peak at 0) and is symmetric.
        assert!((std_normal_pdf(0.0) - 0.398_942).abs() < 1e-5);
        assert!((std_normal_pdf(1.0) - std_normal_pdf(-1.0)).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "bad pi")]
    fn zero_inflated_rejects_bad_pi() {
        let mut rng = SeededRandom::new(1);
        let _ = sample_zero_inflated_exponential(&mut rng, 1.5, 1.0);
    }
}
