//! Port of src/des/test/float-bias-test.ts
//!
//! Floating-point / decimal bias tests for the operations the engine relies on.
//! Both plain f64 arithmetic and the exact-decimal bookkeeping have known
//! failure modes (0.1 + 0.2 != 0.3, summation drift, coercion error); these
//! tests bound the individual contributions. The `mathjs.BigNumber` usage maps
//! onto `crate::des::shared::precision` (`bgn` / `to_f64` / `Decimal`).

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::des::general::prng::mulberry32;
    use crate::des::shared::capabilities::RandomSource;
    use crate::des::shared::precision::{bgn, kahan_sum, to_f64, Decimal};

    /// ~4-sigma tolerance per assertion (combined failure budget < 1e-4).
    const K: f64 = 4.0;

    // -------------------------------------------------------------------------
    // F1: U(a, b) sample mean / variance bias from `a + (b-a) * mulberry32()`.
    // -------------------------------------------------------------------------
    fn f1_uniform_samples(a: f64, b: f64, n: usize, seed: u32) {
        let mut rng = mulberry32(seed);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let x = a + (b - a) * rng.next_float();
            sum += x;
            sum_sq += x * x;
        }
        let nf = n as f64;
        let obs_mean = sum / nf;
        let obs_var = sum_sq / nf - obs_mean * obs_mean;

        let exp_mean = (a + b) / 2.0;
        let exp_var = (b - a).powi(2) / 12.0;

        let se_mean = (exp_var / nf).sqrt();
        let se_var = ((4.0 / 5.0) * exp_var.powi(2) / nf).sqrt();

        assert!(
            (obs_mean - exp_mean).abs() < K * se_mean,
            "mean deviation {} exceeds {}",
            obs_mean - exp_mean,
            K * se_mean
        );
        assert!(
            (obs_var - exp_var).abs() < K * se_var,
            "var deviation {} exceeds {}",
            obs_var - exp_var,
            K * se_var
        );
    }

    #[test]
    fn f1_uniform_sample_statistics() {
        f1_uniform_samples(0.7, 1.3, 1_000_000, 0xC001A);
        f1_uniform_samples(0.20, 0.40, 1_000_000, 0xBADC0DE);
        f1_uniform_samples(1.50, 2.50, 1_000_000, 0xDEADC0DE);
    }

    // -------------------------------------------------------------------------
    // F2: step-accumulator drift over a long horizon.
    // -------------------------------------------------------------------------
    fn f2_step_accumulator(step_size: f64, n_steps: usize) {
        let expected = step_size * n_steps as f64;

        // 1. plain f64 summation.
        let mut plain = 0.0_f64;
        for _ in 0..n_steps {
            plain += step_size;
        }
        // 2. Kahan compensated summation.
        let kahan = kahan_sum(&vec![step_size; n_steps]);
        // 3. exact-decimal summation (the `mathjs.BigNumber` analog).
        let bn_step = bgn(step_size);
        let mut bn = Decimal::ZERO;
        for _ in 0..n_steps {
            bn += bn_step;
        }
        let bn_number = to_f64(bn);

        let ulp = 2f64.powi(-52) * 1.0_f64.max(expected);

        assert!(
            (plain - expected).abs() < 1e-4,
            "plain Number drift {} not < 1e-4",
            plain - expected
        );
        assert!(
            (kahan - expected).abs() < 100.0 * ulp,
            "Kahan drift {} not < 100 ULP",
            kahan - expected
        );
        assert!(
            (bn_number - expected).abs() <= ulp,
            "Decimal drift {} not <= 1 ULP",
            bn_number - expected
        );
    }

    #[test]
    fn f2_step_accumulator_drift() {
        f2_step_accumulator(0.05, 1_000_000);
        f2_step_accumulator(0.1, 1_000_000);
    }

    // -------------------------------------------------------------------------
    // F3: Decimal <-> f64 round-trip for the values the engine uses.
    // -------------------------------------------------------------------------
    #[test]
    fn f3_decimal_round_trip() {
        let cases = [0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 1.3, 1.5, 2.5, 1200.0, 800.0];
        for &x in &cases {
            let d = bgn(x);
            let num = to_f64(d);
            assert_eq!(num, x, "round-trip drifted for {x}");
        }
        // Repeated coercion (per-tick conversion) must not drift.
        let bn = bgn(0.05);
        let mut drift = false;
        for _ in 0..1_000_000 {
            if to_f64(bn) != 0.05 {
                drift = true;
                break;
            }
        }
        assert!(!drift, "to_f64(bgn(0.05)) is unstable across 1M coercions");
    }

    // -------------------------------------------------------------------------
    // F4: floor((t + epsilon) / stepSize) at exact step boundaries.
    // -------------------------------------------------------------------------
    fn f4_bucket_boundary(step_size: f64, k_max: i64) {
        let mut bad_k: i64 = -1;
        for k in 0..k_max {
            let t = k as f64 * step_size;
            let bucket = (t / step_size).floor() as i64;
            if bucket != k && bucket != k - 1 {
                bad_k = k;
                break;
            }
        }
        assert_eq!(bad_k, -1, "bucket assignment out of {{k, k-1}} at k={bad_k}");

        // Accumulator variant (matches the engine's per-tick accumulation).
        let mut t = 0.0_f64;
        let mut first_drift: i64 = -1;
        for k in 0..k_max {
            let bucket = (t / step_size).floor() as i64;
            if bucket != k && bucket != k - 1 {
                first_drift = k;
                break;
            }
            t += step_size;
        }
        assert_eq!(first_drift, -1, "accumulator bucket drift at k={first_drift}");
    }

    #[test]
    fn f4_bucket_boundaries() {
        f4_bucket_boundary(0.05, 100_000);
        f4_bucket_boundary(0.1, 100_000);
    }

    // -------------------------------------------------------------------------
    // F5: probability-decision Bernoulli bias.
    // -------------------------------------------------------------------------
    fn f5_decision_bias(p: f64, n: usize, seed: u32) {
        let mut rng = mulberry32(seed);
        let mut true_count = 0usize;
        for _ in 0..n {
            if rng.next_float() < p {
                true_count += 1;
            }
        }
        let obs_p = true_count as f64 / n as f64;
        let se = (p * (1.0 - p) / n as f64).sqrt();
        assert!(
            (obs_p - p).abs() < K * se,
            "obsP {} deviates {} from p (> {})",
            obs_p,
            obs_p - p,
            K * se
        );
    }

    #[test]
    fn f5_decision_bias_bounded() {
        f5_decision_bias(0.40, 1_000_000, 0xFADE);
        f5_decision_bias(0.20, 1_000_000, 0xC0DE);
        f5_decision_bias(0.12, 1_000_000, 0xBABE);
    }

    // -------------------------------------------------------------------------
    // F6: mulberry32 period + uniformity.
    // -------------------------------------------------------------------------
    fn f6_prng_period_and_uniformity(n: usize, buckets: usize, seed: u32) {
        let mut rng = mulberry32(seed);
        // Test 1: first 4096 outputs are all distinct.
        let first_k = 4096;
        let mut seen: HashSet<u64> = HashSet::new();
        let mut dup_at: i64 = -1;
        for i in 0..first_k {
            let r = rng.next_float();
            let bits = r.to_bits();
            if seen.contains(&bits) {
                dup_at = i as i64;
                break;
            }
            seen.insert(bits);
        }
        assert_eq!(dup_at, -1, "duplicate output at i={dup_at}");

        // Test 2: chi-square uniformity over N draws into B buckets.
        let mut rng2 = mulberry32(seed);
        let mut counts = vec![0usize; buckets];
        for _ in 0..n {
            let r = rng2.next_float();
            let idx = ((r * buckets as f64).floor() as usize).min(buckets - 1);
            counts[idx] += 1;
        }
        let expected = n as f64 / buckets as f64;
        let mut chi2 = 0.0;
        for &c in &counts {
            chi2 += (c as f64 - expected).powi(2) / expected;
        }
        // For B=100, df=99, crit_9999 ≈ 159.7 (alpha=0.0001).
        let crit_9999 = 159.7;
        assert!(chi2 < crit_9999, "chi-square {chi2} >= {crit_9999}");
    }

    #[test]
    fn f6_prng_period_and_uniformity_test() {
        f6_prng_period_and_uniformity(1_000_000, 100, 0xFEED);
    }
}
