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
        assert!((pmean - lambda).abs() < 0.1, "poisson mean {pmean} vs {lambda}");
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
}
