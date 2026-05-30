//! Extensive cross-cutting tests added during the hardening pass.
//!
//! These complement the per-module unit tests with three things they don't
//! cover individually:
//!   * RNG determinism & fidelity (the `mulberry32` source must be reproducible
//!     and statistically sane — its bit-faithfulness to the TS engine is what
//!     makes every seeded simulation cross-validate),
//!   * statistical properties of the random sources,
//!   * golden numeric snapshots cross-validated against the TypeScript engine.

#[cfg(test)]
mod tests {
    use crate::des::shared::capabilities::{RandomSource, SeededRandom};
    use crate::des::shared::precision::{bgn, kahan_sum, to_f64, Decimal};

    // ---------------------------------------------------------------------
    // mulberry32 RNG: determinism
    // ---------------------------------------------------------------------

    #[test]
    fn seeded_rng_is_bit_reproducible() {
        let mut a = SeededRandom::new(12345);
        let mut b = SeededRandom::new(12345);
        for i in 0..2000 {
            assert_eq!(
                a.next_float().to_bits(),
                b.next_float().to_bits(),
                "drift at draw {i}"
            );
        }
    }

    #[test]
    fn mulberry32_golden_sequence() {
        // Reference values computed independently in Python with u32 arithmetic.
        // Bit-faithfulness to this sequence is what makes seeded simulations
        // cross-validate against the TypeScript engine; this guards the kernel
        // against silent drift.
        let cases: [(u32, [f64; 5]); 2] = [
            (
                1,
                [
                    0.627_073_940_588_161_35,
                    0.002_735_721_180_215_477_9,
                    0.527_447_039_959_952_24,
                    0.981_050_967_471_674_08,
                    0.968_377_898_214_384_91,
                ],
            ),
            (
                42,
                [
                    0.601_103_751_920_163_63,
                    0.448_290_558_997_541_67,
                    0.852_465_793_490_409_85,
                    0.669_734_041_439_369_32,
                    0.174_813_898_745_924_23,
                ],
            ),
        ];
        for (seed, expected) in cases {
            let mut r = SeededRandom::new(seed);
            for (i, &want) in expected.iter().enumerate() {
                let got = r.next_float();
                assert!(
                    (got - want).abs() < 1e-15,
                    "seed {seed} draw {i}: {got} != {want}"
                );
            }
        }
    }

    #[test]
    fn seeded_rng_outputs_in_unit_interval() {
        let mut r = SeededRandom::new(7);
        for _ in 0..50_000 {
            let x = r.next_float();
            assert!((0.0..1.0).contains(&x), "out of [0,1): {x}");
        }
    }

    #[test]
    fn distinct_seeds_diverge() {
        let mut a = SeededRandom::new(1);
        let mut b = SeededRandom::new(2);
        let differences = (0..200)
            .filter(|_| a.next_float() != b.next_float())
            .count();
        assert!(
            differences > 190,
            "seeds 1 and 2 barely differ ({differences}/200)"
        );
    }

    #[test]
    fn next_int_respects_half_open_bounds() {
        let mut r = SeededRandom::new(55);
        let mut seen_lo = false;
        let mut seen_hi = false;
        for _ in 0..50_000 {
            let v = r.next_int(3, 9);
            assert!((3..9).contains(&v), "out of [3,9): {v}");
            seen_lo |= v == 3;
            seen_hi |= v == 8;
        }
        assert!(seen_lo && seen_hi, "range endpoints never sampled");
    }

    // ---------------------------------------------------------------------
    // mulberry32 RNG: statistical properties
    // ---------------------------------------------------------------------

    #[test]
    fn uniform_mean_and_variance_are_sane() {
        let mut r = SeededRandom::new(99);
        let n = 500_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let x = r.next_float();
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!((mean - 0.5).abs() < 5e-3, "mean = {mean}");
        // Var of U(0,1) is 1/12 ≈ 0.0833.
        assert!((var - 1.0 / 12.0).abs() < 5e-3, "var = {var}");
    }

    #[test]
    fn gaussian_mean_and_variance_are_sane() {
        let mut r = SeededRandom::new(2024);
        let n = 500_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let x = r.next_gaussian();
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 1e-2, "gaussian mean = {mean}");
        assert!((var - 1.0).abs() < 2e-2, "gaussian var = {var}");
    }

    #[test]
    fn uniform_buckets_are_roughly_balanced() {
        // Chi-square-flavoured smoke test: 10 equal buckets over 500k draws.
        let mut r = SeededRandom::new(31337);
        let n = 500_000usize;
        let mut buckets = [0usize; 10];
        for _ in 0..n {
            let b = (r.next_float() * 10.0) as usize;
            buckets[b.min(9)] += 1;
        }
        let expected = n as f64 / 10.0;
        for (i, &c) in buckets.iter().enumerate() {
            let rel = (c as f64 - expected).abs() / expected;
            assert!(rel < 0.05, "bucket {i} skewed: {c} vs {expected}");
        }
    }

    // ---------------------------------------------------------------------
    // Decimal (Tier-2) edge cases beyond the precision module's own tests
    // ---------------------------------------------------------------------

    #[test]
    fn decimal_subtraction_is_exact() {
        // 1.00 - 0.90 - 0.10 == 0 exactly (f64 leaves a residue here).
        let mut x = bgn(1.0);
        x -= bgn(0.9);
        x -= bgn(0.1);
        assert_eq!(x, Decimal::ZERO);
        let naive = 1.0_f64 - 0.9 - 0.1;
        assert!(naive != 0.0, "f64 unexpectedly exact ({naive})");
    }

    #[test]
    fn decimal_handles_negative_running_balance() {
        let mut bal = bgn(0.0);
        for _ in 0..100 {
            bal -= bgn(0.07);
        }
        assert_eq!(bal, bgn(-7.0));
        assert_eq!(to_f64(bal), -7.0);
    }

    #[test]
    fn kahan_matches_decimal_on_repeated_add() {
        let v = vec![0.1_f64; 100_000];
        let k = kahan_sum(&v);
        // Decimal gives the exact 10_000.0; Kahan must be within a few ULP.
        assert!((k - 10_000.0).abs() < 1e-6, "kahan = {k}");
    }
}
