//! TypeScript source: `src/des/test/random-variables-test.ts`
//! Rust target: `tests/random_variables_test.rs`

use discrete_event_system_rs::des::general::prng::Mulberry32;
use discrete_event_system_rs::des::general::random_variables::{
    bernoulli_pmf, binomial_pmf, competing_risks, discrete_convolve, discrete_convolve_many,
    discrete_convolve_self, mean_from_pmf, normalize_pmf, pmf_total_mass, poisson_binomial_pmf,
    sample_categorical, sample_exponential, sample_gamma, sample_poisson, variance_from_pmf,
};

fn approx(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

fn arr_approx(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(left, right)| (left - right).abs() <= tolerance)
}

fn rng(seed: u32) -> impl FnMut() -> f64 {
    let mut rng = Mulberry32::new(seed);
    move || rng.next_f64()
}

#[test]
fn convolution_preserves_mass_and_matches_binomial_identities() {
    let p = [0.2, 0.5, 0.3];
    let q = [0.1, 0.4, 0.5];
    let convolved = discrete_convolve(&p, &q);
    assert_eq!(convolved.len(), 5);
    assert!(approx(pmf_total_mass(&convolved), 1.0, 1e-12));

    let bernoulli = bernoulli_pmf(0.3);
    assert!(arr_approx(
        &discrete_convolve(&bernoulli, &bernoulli),
        &binomial_pmf(2, 0.3),
        1e-15
    ));

    for n in [1usize, 5, 17, 32, 100] {
        assert!(arr_approx(
            &discrete_convolve_self(&bernoulli_pmf(0.37), n),
            &binomial_pmf(n, 0.37),
            1e-12
        ));
    }
}

#[test]
fn convolution_is_associative_and_adds_moments() {
    let p = [0.1, 0.4, 0.3, 0.2];
    let q = [0.5, 0.3, 0.2];
    let r = [0.25, 0.25, 0.25, 0.25];
    let left = discrete_convolve(&discrete_convolve(&p, &q), &r);
    let right = discrete_convolve(&p, &discrete_convolve(&q, &r));
    assert!(arr_approx(&left, &right, 1e-14));

    let p = binomial_pmf(10, 0.4);
    let q = binomial_pmf(10, 0.4);
    let convolved = discrete_convolve(&p, &q);
    assert!(approx(
        mean_from_pmf(&convolved),
        mean_from_pmf(&p) + mean_from_pmf(&q),
        1e-12
    ));
    assert!(approx(
        variance_from_pmf(&convolved),
        variance_from_pmf(&p) + variance_from_pmf(&q),
        1e-10
    ));
}

#[test]
fn poisson_binomial_matches_binomial_and_moments() {
    let uniform = vec![0.42; 20];
    assert!(arr_approx(
        &poisson_binomial_pmf(&uniform),
        &binomial_pmf(20, 0.42),
        1e-13
    ));

    let probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let expected_mean: f64 = probs.iter().sum();
    let expected_variance: f64 = probs.iter().map(|p| p * (1.0 - p)).sum();
    let pmf = poisson_binomial_pmf(&probs);
    assert!(approx(mean_from_pmf(&pmf), expected_mean, 1e-13));
    assert!(approx(variance_from_pmf(&pmf), expected_variance, 1e-12));
    assert!(approx(pmf_total_mass(&pmf), 1.0, 1e-13));
}

#[test]
fn poisson_binomial_matches_monte_carlo() {
    let probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8];
    let pmf = poisson_binomial_pmf(&probs);
    let samples = 200_000;
    let mut next = rng(0xBEEF);
    let mut empirical = vec![0.0; probs.len() + 1];
    for _ in 0..samples {
        let mut k = 0usize;
        for p in probs {
            if next() < p {
                k += 1;
            }
        }
        empirical[k] += 1.0;
    }
    for value in &mut empirical {
        *value /= samples as f64;
    }
    let max_abs_diff = pmf
        .iter()
        .zip(empirical.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_abs_diff < 0.005, "max_abs_diff={max_abs_diff}");
}

#[test]
fn competing_risks_exact_probabilities_match_formulas() {
    let out = competing_risks(&[0.7], 0.5);
    assert!(approx(out[0], (-0.7_f64 * 0.5).exp(), 1e-15));
    assert!(approx(out[1], 1.0 - (-0.7_f64 * 0.5).exp(), 1e-15));
    assert!(approx(out[0] + out[1], 1.0, 1e-15));

    let lambda = 0.4;
    let dt = 1.0;
    let out = competing_risks(&[lambda, lambda], dt);
    let expected_any = 1.0 - (-(2.0 * lambda) * dt).exp();
    assert!(approx(out[1], expected_any / 2.0, 1e-15));
    assert!(approx(out[2], expected_any / 2.0, 1e-15));
    assert!(approx(out.iter().sum(), 1.0, 1e-15));

    let lambdas = [0.05, 0.05, 0.05];
    let exact = competing_risks(&lambdas, 0.1);
    for (index, lambda) in lambdas.iter().enumerate() {
        assert!(approx(exact[index + 1], lambda * 0.1, 0.001));
    }
}

#[test]
fn competing_risks_matches_first_event_monte_carlo() {
    let lambdas = [0.5, 1.0, 0.3];
    let dt = 0.4;
    let exact = competing_risks(&lambdas, dt);
    let samples = 100_000;
    let mut next = rng(0xFEED);
    let mut counts = vec![0.0; lambdas.len() + 1];

    for _ in 0..samples {
        let mut min_time = f64::INFINITY;
        let mut who = None;
        for (index, lambda) in lambdas.iter().copied().enumerate() {
            let u = 1.0 - next();
            let time = -u.ln() / lambda;
            if time < min_time {
                min_time = time;
                who = Some(index);
            }
        }
        if min_time > dt {
            counts[0] += 1.0;
        } else {
            counts[who.unwrap() + 1] += 1.0;
        }
    }

    let max_diff = counts
        .iter()
        .zip(exact.iter())
        .map(|(count, prob)| (count / samples as f64 - prob).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 0.005, "max_diff={max_diff}");
}

#[test]
fn pmf_utilities_normalize_and_sample() {
    let skewed = [0.5, 1.5, 2.0, 1.0, 0.0];
    let normalized = normalize_pmf(&skewed);
    assert!(approx(pmf_total_mass(&normalized), 1.0, 1e-15));
    assert!(approx(
        normalized[1] / normalized[0],
        skewed[1] / skewed[0],
        1e-15
    ));

    let probs = [0.1, 0.2, 0.3, 0.25, 0.15];
    let samples = 50_000;
    let mut next = rng(0xC0DE);
    let mut counts = vec![0.0; probs.len()];
    for _ in 0..samples {
        counts[sample_categorical(&probs, &mut next)] += 1.0;
    }
    let max_diff = counts
        .iter()
        .zip(probs.iter())
        .map(|(count, prob)| (count / samples as f64 - prob).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 0.01, "max_diff={max_diff}");
}

#[test]
fn self_convolution_matches_naive_iterative_convolution() {
    let base = [0.2, 0.5, 0.3];
    for n in [0usize, 1, 2, 5, 13, 64] {
        let pmfs: Vec<&[f64]> = (0..n).map(|_| base.as_slice()).collect();
        let expected = discrete_convolve_many(&pmfs);
        let got = discrete_convolve_self(&base, n);
        assert!(arr_approx(&got, &expected, 1e-13), "n={n}");
    }
}

#[test]
fn poisson_sampler_matches_mean_and_variance() {
    let mut next = rng(0xCAFE);
    for lambda in [0.7, 5.0, 30.0, 100.0] {
        let samples = 50_000;
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for _ in 0..samples {
            let x = sample_poisson(lambda, &mut next) as f64;
            sum += x;
            sum2 += x * x;
        }
        let mean = sum / samples as f64;
        let variance = sum2 / samples as f64 - mean * mean;
        let tolerance_mean = if lambda < 30.0 { 0.05 } else { 0.5 };
        let tolerance_variance = lambda * 0.10;
        assert!(
            approx(mean, lambda, tolerance_mean),
            "lambda={lambda}, mean={mean}"
        );
        assert!(
            approx(variance, lambda, tolerance_variance),
            "lambda={lambda}, variance={variance}"
        );
    }
}

#[test]
fn exponential_and_gamma_samplers_match_moments() {
    let mut next = rng(0xC001);
    let rate = 2.5;
    let samples = 50_000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..samples {
        let x = sample_exponential(rate, &mut next);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / samples as f64;
    let variance = sum2 / samples as f64 - mean * mean;
    assert!(approx(mean, 1.0 / rate, 0.01), "mean={mean}");
    assert!(
        approx(variance, 1.0 / (rate * rate), 0.05),
        "variance={variance}"
    );

    let mut next = rng(0xB055);
    for (shape, scale) in [(2.0, 1.5), (0.5, 2.0), (10.0, 0.3)] {
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for _ in 0..samples {
            let x = sample_gamma(shape, scale, &mut next);
            sum += x;
            sum2 += x * x;
        }
        let expected_mean = shape * scale;
        let expected_variance = shape * scale * scale;
        let mean = sum / samples as f64;
        let variance = sum2 / samples as f64 - mean * mean;
        assert!(approx(mean, expected_mean, expected_mean * 0.03));
        assert!(approx(
            variance,
            expected_variance,
            expected_variance * 0.10
        ));
    }
}
