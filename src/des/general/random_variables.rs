//! TypeScript source: `src/des/general/random-variables.ts`
//! Rust target: `src/des/general/random_variables.rs`
//!
//! Porting note: this file intentionally stays a pure helper module, matching
//! the TypeScript file. Sampling receives explicit RNG closures for
//! deterministic tests instead of using global randomness.

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/random-variables.ts",
    "src/des/general/random_variables.rs",
    &[
        "RUST MIGRATION: PMF math and sampling helpers remain free functions over slices and `Vec<f64>`.",
        "RUST MIGRATION: RNG callbacks are explicit `FnMut() -> f64` parameters.",
        "RUST MIGRATION: TypeScript throws are currently represented with assertions/panics on invalid parameters; a later hardening pass can add checked wrappers where callers need recovery.",
    ],
    &[
        "bernoulli_pmf",
        "binomial_pmf",
        "competing_risks",
        "discrete_convolve",
        "discrete_convolve_many",
        "discrete_convolve_self",
        "discretise_pdf",
        "mean_from_pmf",
        "normalize_pmf",
        "pmf_total_mass",
        "poisson_binomial_pmf",
        "sample_categorical",
        "sample_exponential",
        "sample_from_pmf",
        "sample_gamma",
        "sample_poisson",
        "variance_from_pmf",
    ],
);

pub fn pmf_total_mass(pmf: &[f64]) -> f64 {
    pmf.iter().sum()
}

pub fn normalize_pmf(pmf: &[f64]) -> Vec<f64> {
    let mass = pmf_total_mass(pmf);
    assert!(mass > 0.0, "cannot normalise zero-mass PMF");
    pmf.iter().map(|value| value / mass).collect()
}

pub fn mean_from_pmf(pmf: &[f64]) -> f64 {
    pmf.iter()
        .enumerate()
        .map(|(index, value)| index as f64 * value)
        .sum()
}

pub fn variance_from_pmf(pmf: &[f64]) -> f64 {
    let mut mean = 0.0;
    let mut second_moment = 0.0;
    for (index, value) in pmf.iter().copied().enumerate() {
        let k = index as f64;
        mean += k * value;
        second_moment += k * k * value;
    }
    second_moment - mean * mean
}

pub fn discrete_convolve(p: &[f64], q: &[f64]) -> Vec<f64> {
    if p.is_empty() || q.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0; p.len() + q.len() - 1];
    for (i, p_value) in p.iter().copied().enumerate() {
        if p_value == 0.0 {
            continue;
        }
        for (j, q_value) in q.iter().copied().enumerate() {
            out[i + j] += p_value * q_value;
        }
    }
    out
}

pub fn discrete_convolve_many(pmfs: &[&[f64]]) -> Vec<f64> {
    if pmfs.is_empty() {
        return vec![1.0];
    }
    let mut acc = pmfs[0].to_vec();
    for pmf in pmfs.iter().skip(1) {
        acc = discrete_convolve(&acc, pmf);
    }
    acc
}

pub fn discrete_convolve_self(pmf: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![1.0];
    }
    let mut result: Option<Vec<f64>> = None;
    let mut base = pmf.to_vec();
    let mut remaining = n;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = Some(match result {
                Some(current) => discrete_convolve(&current, &base),
                None => base.clone(),
            });
        }
        remaining >>= 1;
        if remaining > 0 {
            base = discrete_convolve(&base, &base);
        }
    }
    result.expect("n > 0 produces a PMF")
}

pub fn bernoulli_pmf(p: f64) -> Vec<f64> {
    assert!((0.0..=1.0).contains(&p), "bad p {p}");
    vec![1.0 - p, p]
}

pub fn binomial_pmf(n: usize, p: f64) -> Vec<f64> {
    assert!((0.0..=1.0).contains(&p), "bad p {p}");
    if n == 0 {
        return vec![1.0];
    }
    let mut out = vec![0.0; n + 1];
    if p == 0.0 {
        out[0] = 1.0;
        return out;
    }
    if p == 1.0 {
        out[n] = 1.0;
        return out;
    }
    out[0] = (1.0 - p).powi(n as i32);
    let ratio = p / (1.0 - p);
    for k in 0..n {
        out[k + 1] = out[k] * (n - k) as f64 * ratio / (k + 1) as f64;
    }
    out
}

pub fn poisson_binomial_pmf(probs: &[f64]) -> Vec<f64> {
    if probs.is_empty() {
        return vec![1.0];
    }
    let all_equal = probs
        .iter()
        .skip(1)
        .all(|prob| (*prob - probs[0]).abs() <= 1e-15);
    if all_equal {
        return binomial_pmf(probs.len(), probs[0]);
    }

    let mut pmf = vec![1.0];
    for (index, p) in probs.iter().copied().enumerate() {
        assert!((0.0..=1.0).contains(&p), "bad p[{index}] {p}");
        let mut next = vec![0.0; pmf.len() + 1];
        for (k, mass) in pmf.iter().copied().enumerate() {
            next[k] += mass * (1.0 - p);
            next[k + 1] += mass * p;
        }
        pmf = next;
    }
    pmf
}

pub fn competing_risks(rates: &[f64], dt: f64) -> Vec<f64> {
    assert!(dt >= 0.0, "bad dt {dt}");
    let mut total = 0.0;
    for (index, rate) in rates.iter().copied().enumerate() {
        assert!(rate >= 0.0, "bad rate[{index}] {rate}");
        total += rate;
    }
    if total == 0.0 {
        let mut out = vec![0.0; rates.len() + 1];
        out[0] = 1.0;
        return out;
    }
    let p_no = (-total * dt).exp();
    let p_any = 1.0 - p_no;
    let mut out = Vec::with_capacity(rates.len() + 1);
    out.push(p_no);
    out.extend(rates.iter().map(|rate| (rate / total) * p_any));
    out
}

pub fn sample_categorical(probs: &[f64], rng: &mut impl FnMut() -> f64) -> usize {
    let draw = rng();
    let mut cumulative = 0.0;
    for (index, prob) in probs.iter().copied().enumerate() {
        cumulative += prob;
        if draw <= cumulative {
            return index;
        }
    }
    probs.len().saturating_sub(1)
}

pub fn sample_from_pmf(pmf: &[f64], rng: &mut impl FnMut() -> f64) -> usize {
    sample_categorical(pmf, rng)
}

pub fn sample_poisson(lambda: f64, rng: &mut impl FnMut() -> f64) -> usize {
    assert!(lambda >= 0.0, "bad lambda {lambda}");
    if lambda == 0.0 {
        return 0;
    }
    if lambda < 30.0 {
        let limit = (-lambda).exp();
        let mut k = 0usize;
        let mut product = 1.0;
        loop {
            k += 1;
            product *= rng();
            if product <= limit {
                return k - 1;
            }
        }
    }

    let u1 = 1.0 - rng();
    let u2 = rng();
    let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let x = lambda + lambda.sqrt() * z + 0.5;
    x.floor().max(0.0) as usize
}

pub fn sample_exponential(rate: f64, rng: &mut impl FnMut() -> f64) -> f64 {
    assert!(rate > 0.0, "bad rate {rate}");
    let u = 1.0 - rng();
    -u.ln() / rate
}

pub fn sample_gamma(shape: f64, scale: f64, rng: &mut impl FnMut() -> f64) -> f64 {
    assert!(
        shape > 0.0 && scale > 0.0,
        "bad shape/scale {shape}/{scale}"
    );
    if shape < 1.0 {
        let g = sample_gamma(shape + 1.0, scale, rng);
        let u = 1.0 - rng();
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, mut v) = loop {
            let u1 = 1.0 - rng();
            let u2 = rng();
            let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = 1.0 + c * z;
            if v > 0.0 {
                break (z, v);
            }
        };
        v = v * v * v;
        let u = rng();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v * scale;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

pub fn discretise_pdf(f: impl Fn(f64) -> f64, x0: f64, h: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| f(x0 + i as f64 * h) * h).collect()
}
