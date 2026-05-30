//! Port of `src/des/runners/stats.ts`.
//!
//! Tiny statistics helpers: mean, sample variance, Welch's t-test, and a
//! normal-approximation two-sided p-value good enough for n=30. These are leaf
//! math utilities, kept as plain `pub fn`s. Empty-input `NaN` sentinels become
//! `f64::NAN`.

#![allow(dead_code)]

/// Arithmetic mean. Empty input → `NaN` (matches the TS sentinel).
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut s = 0.0;
    for &x in xs {
        s += x;
    }
    s / xs.len() as f64
}

/// Unbiased sample variance (`/ (n-1)`). `< 2` samples → 0.
pub fn sample_variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let mut s = 0.0;
    for &x in xs {
        let d = x - m;
        s += d * d;
    }
    s / (xs.len() as f64 - 1.0)
}

/// Sample standard deviation.
pub fn stddev(xs: &[f64]) -> f64 {
    sample_variance(xs).sqrt()
}

/// Welch's t-test result. The two-sided p-value uses a normal CDF (erfc); for
/// df > 30 this is within a couple of percent of the true Student-t value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WelchResult {
    pub mean_a: f64,
    pub mean_b: f64,
    pub var_a: f64,
    pub var_b: f64,
    pub n_a: usize,
    pub n_b: usize,
    pub t: f64,
    pub df: f64,
    pub p_value_two_sided: f64,
    /// `|t| > 1.96` (normal approx).
    pub reject95: bool,
    /// `|t| > 2.58`.
    pub reject99: bool,
}

/// Welch's t-test.
pub fn welch(a: &[f64], b: &[f64]) -> WelchResult {
    let m_a = mean(a);
    let m_b = mean(b);
    let v_a = sample_variance(a);
    let v_b = sample_variance(b);
    let n_a = a.len();
    let n_b = b.len();
    let se_sq = v_a / n_a as f64 + v_b / n_b as f64;
    let t = if se_sq > 0.0 { (m_a - m_b) / se_sq.sqrt() } else { 0.0 };
    let df = if se_sq > 0.0 {
        (se_sq * se_sq)
            / ((v_a / n_a as f64).powi(2) / (1.0_f64).max(n_a as f64 - 1.0)
                + (v_b / n_b as f64).powi(2) / (1.0_f64).max(n_b as f64 - 1.0))
    } else {
        1.0
    };
    let p_value_two_sided = if se_sq > 0.0 { 2.0 * (1.0 - normal_cdf(t.abs())) } else { 1.0 };
    WelchResult {
        mean_a: m_a,
        mean_b: m_b,
        var_a: v_a,
        var_b: v_b,
        n_a,
        n_b,
        t,
        df,
        p_value_two_sided,
        reject95: t.abs() > 1.96,
        reject99: t.abs() > 2.58,
    }
}

/// Standard-normal CDF using Abramowitz-Stegun erfc approximation.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function — Abramowitz & Stegun 7.1.26, max abs error 1.5e-7.
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_variance() {
        assert!(mean(&[]).is_nan());
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(sample_variance(&[1.0]), 0.0);
        assert!((sample_variance(&[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn welch_identical_samples_no_reject() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let w = welch(&a, &a);
        assert_eq!(w.t, 0.0);
        assert!(!w.reject95);
        // p must be ~1 for identical samples; the bound reflects the A&S 7.1.26
        // erf approximation's accuracy (~1.5e-7), not exact arithmetic.
        assert!((w.p_value_two_sided - 1.0).abs() < 1e-6);
    }
}
