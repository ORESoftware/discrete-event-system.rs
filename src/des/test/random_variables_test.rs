//! Port of src/des/test/random-variables-test.ts
//!
//! Tests for `general/random_variables`. Each claim is pinned either by an
//! analytic identity or a seeded Monte Carlo cross-check (often both). The TS
//! free functions map onto the Rust `Transform` structs:
//!   discreteConvolve(p, q)      -> DiscreteConvolve.transform(ConvolvePair {p, q})
//!   discreteConvolveMany(arr)   -> DiscreteConvolveMany.transform(&arr)
//!   discreteConvolveSelf(p, n)  -> DiscreteConvolveSelf::new(n).transform(&p)
//!   binomialPMF(n, p)           -> BinomialPMF::new(p).transform(n)
//!   poissonBinomialPMF(probs)   -> PoissonBinomialPMF.transform(&probs)
//!   competingRisks(rates, dt)   -> CompetingRisks::new(dt).transform(&rates)
//! The samplers take an injected `&mut RandomSource` instead of a `() => number`.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::prng::mulberry32;
    use crate::des::general::random_variables::{
        bernoulli_pmf, mean_from_pmf, normalize_pmf, pmf_total_mass, sample_categorical,
        sample_exponential, sample_gamma, sample_poisson, variance_from_pmf, BinomialPMF,
        CompetingRisks, ConvolvePair, DiscreteConvolve, DiscreteConvolveMany,
        DiscreteConvolveSelf, PoissonBinomialPMF,
    };
    use crate::des::shared::capabilities::RandomSource;
    use crate::des::shared::transform::Transform;

    fn conv(p: &[f64], q: &[f64]) -> Vec<f64> {
        DiscreteConvolve.transform(ConvolvePair { p, q })
    }
    fn conv_self(pmf: &[f64], n: u32) -> Vec<f64> {
        DiscreteConvolveSelf::new(n).transform(pmf)
    }
    fn binomial(n: u32, p: f64) -> Vec<f64> {
        BinomialPMF::new(p).transform(n)
    }

    fn arr_approx(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    // T1  Convolution identities and total mass
    #[test]
    fn t1_convolution_identities() {
        let p = [0.2, 0.5, 0.3];
        let q = [0.1, 0.4, 0.5];
        let c = conv(&p, &q);
        assert_eq!(c.len(), 5, "|p| + |q| - 1");
        assert!((pmf_total_mass(&c) - 1.0).abs() < 1e-12);

        // Bernoulli(p) ⊕ Bernoulli(p) = Binomial(2, p).
        let b1 = bernoulli_pmf(0.3);
        let expected = binomial(2, 0.3);
        assert!(arr_approx(&conv(&b1, &b1), &expected, 1e-15));

        // n-fold self-convolution of Bernoulli(p) = Binomial(n, p).
        for n in [1u32, 5, 17, 32, 100] {
            let expected = binomial(n, 0.37);
            let got = conv_self(&bernoulli_pmf(0.37), n);
            assert!(arr_approx(&got, &expected, 1e-12), "Bernoulli^*{n}");
        }

        // Associativity.
        let p = [0.1, 0.4, 0.3, 0.2];
        let q = [0.5, 0.3, 0.2];
        let r = [0.25, 0.25, 0.25, 0.25];
        let left = conv(&conv(&p, &q), &r);
        let right = conv(&p, &conv(&q, &r));
        assert!(arr_approx(&left, &right, 1e-14));

        // Mean and variance add for independent sums.
        let p = binomial(10, 0.4);
        let q = binomial(10, 0.4);
        let c = conv(&p, &q);
        assert!((mean_from_pmf(&c) - (mean_from_pmf(&p) + mean_from_pmf(&q))).abs() < 1e-12);
        assert!(
            (variance_from_pmf(&c) - (variance_from_pmf(&p) + variance_from_pmf(&q))).abs() < 1e-10
        );
    }

    // T2  Poisson-binomial PMF
    #[test]
    fn t2_poisson_binomial() {
        let probs = vec![0.42; 20];
        let pb = PoissonBinomialPMF.transform(&probs);
        let bin = binomial(20, 0.42);
        assert!(arr_approx(&pb, &bin, 1e-13));

        let probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let expected_mean: f64 = probs.iter().sum();
        let expected_var: f64 = probs.iter().map(|p| p * (1.0 - p)).sum();
        let pb = PoissonBinomialPMF.transform(&probs);
        assert!((mean_from_pmf(&pb) - expected_mean).abs() < 1e-13);
        assert!((variance_from_pmf(&pb) - expected_var).abs() < 1e-12);
        assert!((pmf_total_mass(&pb) - 1.0).abs() < 1e-13);

        // Monte Carlo cross-check.
        let probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8];
        let pb = PoissonBinomialPMF.transform(&probs);
        let n = 200_000;
        let mut rng = mulberry32(0xBEEF);
        let mut empirical = vec![0.0; probs.len() + 1];
        for _ in 0..n {
            let mut k = 0usize;
            for &p in &probs {
                if rng.next_float() < p {
                    k += 1;
                }
            }
            empirical[k] += 1.0;
        }
        for e in empirical.iter_mut() {
            *e /= n as f64;
        }
        let max_abs_diff = pb
            .iter()
            .zip(empirical.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_abs_diff < 0.005, "max |Δ| = {max_abs_diff}");
    }

    // T3  Competing risks formula
    #[test]
    fn t3_competing_risks() {
        // K=1: collapses to {exp(−λdt), 1−exp(−λdt)}.
        let lambda = 0.7;
        let dt = 0.5;
        let out = CompetingRisks::new(dt).transform(&[lambda]);
        assert!((out[0] - (-lambda * dt).exp()).abs() < 1e-15);
        assert!((out[1] - (1.0 - (-lambda * dt).exp())).abs() < 1e-15);
        assert!((out[0] + out[1] - 1.0).abs() < 1e-15);

        // Two equal rates: each event has prob (1/2)(1 − exp(−2λdt)).
        let lambda = 0.4;
        let dt = 1.0;
        let out = CompetingRisks::new(dt).transform(&[lambda, lambda]);
        let big_lambda = 2.0 * lambda;
        let expected_any = 1.0 - (-big_lambda * dt).exp();
        assert!((out[1] - expected_any / 2.0).abs() < 1e-15);
        assert!((out[2] - expected_any / 2.0).abs() < 1e-15);
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-15);

        // Small Λ·dt: exact ≈ linear.
        let lambdas = [0.05, 0.05, 0.05];
        let dt = 0.1;
        let exact = CompetingRisks::new(dt).transform(&lambdas);
        for (i, &l) in lambdas.iter().enumerate() {
            assert!((exact[i + 1] - l * dt).abs() < 0.001);
        }

        // Monte Carlo: simulate competing exponentials.
        let lambdas = [0.5, 1.0, 0.3];
        let dt = 0.4;
        let exact = CompetingRisks::new(dt).transform(&lambdas);
        let n = 100_000;
        let mut rng = mulberry32(0xFEED);
        let mut cnt = vec![0.0; lambdas.len() + 1];
        for _ in 0..n {
            let mut min_t = f64::INFINITY;
            let mut who = -1i32;
            for (i, &l) in lambdas.iter().enumerate() {
                let u = 1.0 - rng.next_float();
                let time = -u.ln() / l;
                if time < min_t {
                    min_t = time;
                    who = i as i32;
                }
            }
            if min_t > dt {
                cnt[0] += 1.0;
            } else {
                cnt[(who + 1) as usize] += 1.0;
            }
        }
        let max_diff = cnt
            .iter()
            .zip(exact.iter())
            .map(|(c, e)| (c / n as f64 - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 0.005, "max |Δ| = {max_diff}");
    }

    // T4  PMF utilities
    #[test]
    fn t4_pmf_utilities() {
        let skewed = [0.5, 1.5, 2.0, 1.0, 0.0];
        let norm = normalize_pmf(&skewed);
        assert!((pmf_total_mass(&norm) - 1.0).abs() < 1e-15);
        assert!((norm[1] / norm[0] - skewed[1] / skewed[0]).abs() < 1e-15);

        // sampleCategorical histogram matches input probs.
        let probs = [0.1, 0.2, 0.3, 0.25, 0.15];
        let n = 50_000;
        let mut rng = mulberry32(0xC0DE);
        let mut cnt = vec![0.0; probs.len()];
        for _ in 0..n {
            cnt[sample_categorical(&mut rng, &probs)] += 1.0;
        }
        let max_diff = cnt
            .iter()
            .zip(probs.iter())
            .map(|(c, p)| (c / n as f64 - p).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 0.01, "max |Δ| = {max_diff}");
    }

    // T5  Self-convolution via repeated squaring matches iterative convolution
    #[test]
    fn t5_self_convolution_matches_iterative() {
        let base = vec![0.2, 0.5, 0.3];
        for n in [0u32, 1, 2, 5, 13, 64] {
            let expected_arr: Vec<Vec<f64>> = (0..n).map(|_| base.clone()).collect();
            let expected = DiscreteConvolveMany.transform(&expected_arr);
            let got = conv_self(&base, n);
            if n == 0 {
                assert!(arr_approx(&got, &[1.0], 1e-15), "self-conv n=0 = [1]");
            } else {
                assert!(arr_approx(&got, &expected, 1e-13), "self-conv n={n}");
            }
        }
    }

    // T6  Continuous samplers: Poisson, Exponential, Gamma
    #[test]
    fn t6_continuous_samplers() {
        let mut rng = mulberry32(0xCAFE);
        for lambda in [0.7, 5.0, 30.0, 100.0] {
            let n = 50_000;
            let (mut s, mut s2) = (0.0, 0.0);
            for _ in 0..n {
                let x = sample_poisson(&mut rng, lambda);
                s += x;
                s2 += x * x;
            }
            let m = s / n as f64;
            let v = s2 / n as f64 - m * m;
            let tol_mean = if lambda < 30.0 { 0.05 } else { 0.5 };
            let tol_var = lambda * 0.10;
            assert!((m - lambda).abs() <= tol_mean, "Poisson({lambda}) mean {m}");
            assert!((v - lambda).abs() <= tol_var, "Poisson({lambda}) var {v}");
        }

        // Exponential(rate): mean = 1/rate, variance = 1/rate².
        let mut rng = mulberry32(0xC001);
        let rate = 2.5;
        let n = 50_000;
        let (mut s, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let x = sample_exponential(&mut rng, rate);
            s += x;
            s2 += x * x;
        }
        let m = s / n as f64;
        let v = s2 / n as f64 - m * m;
        assert!((m - 1.0 / rate).abs() <= 0.01);
        assert!((v - 1.0 / (rate * rate)).abs() <= 0.05);

        // Gamma(shape, scale): mean = shape·scale, variance = shape·scale².
        let mut rng = mulberry32(0xB055);
        for (shape, scale) in [(2.0, 1.5), (0.5, 2.0), (10.0, 0.3)] {
            let n = 50_000;
            let (mut s, mut s2) = (0.0, 0.0);
            for _ in 0..n {
                let x = sample_gamma(&mut rng, shape, scale);
                s += x;
                s2 += x * x;
            }
            let expected_mean = shape * scale;
            let expected_var = shape * scale * scale;
            let m = s / n as f64;
            let v = s2 / n as f64 - m * m;
            assert!((m - expected_mean).abs() <= expected_mean * 0.03);
            assert!((v - expected_var).abs() <= expected_var * 0.10);
        }
    }
}
