//! Canonical use path: `crate::des::random_variables::generate::*`
//!
//! Port of `src/des/random-variables/generate.ts` — a demo / CLI script that
//! samples uniform and exponential draws and prints empirical moments.
//!
//! PORT NOTES:
//!   * This was a script guarded by `if (require.main === module) runExponential()`.
//!     That entry-point guard maps to a `[[bin]]` / `examples/` `main`, not library
//!     code; here the two routines are plain functions taking an injected
//!     `RandomSource` (no ambient `Math.random()`).
//!   * `runUniform` in the TS printed all 100000 samples to stdout; here it
//!     returns the samples instead of flooding stdout.

#![allow(dead_code)]

use crate::des::shared::capabilities::RandomSource;

/// `runUniform` — draw `count` uniforms on `[a, b)`; returns the samples.
pub fn run_uniform(rng: &mut dyn RandomSource) -> Vec<f64> {
    let a = 5.0_f64;
    let b = 8.0_f64;
    let count = 100_000usize;

    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(a + rng.next_float() * (b - a));
    }
    values
}

/// Empirical vs. theoretical moments returned by [`run_exponential`].
#[derive(Clone, Copy, Debug)]
pub struct ExponentialMoments {
    pub theoretical: [f64; 5],
    pub empirical: [f64; 5],
}

/// `runExponential` — sample `count` exponential(lambda=1) draws via inverse-CDF
/// and compare empirical raw moments to the theoretical `k! / lambda^k`.
pub fn run_exponential(rng: &mut dyn RandomSource) -> ExponentialMoments {
    let lambda = 1.0_f64;
    let count = 1000usize;

    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        // (-1/lambda) * ln(1 - u)
        values.push((-1.0 / lambda) * (1.0 - rng.next_float()).ln());
    }

    let pow_sum = |p: i32| -> f64 { values.iter().map(|v| v.powi(p)).sum() };

    // Raw-moment sums (1st..5th). PORT NOTE: the TS chained `mapToSquare`, so its
    // "squared/cubed/quad/quint" arrays were actually v^2, v^4, v^8, v^16 — and it
    // then printed the standard `k!/lambda^k` orders beside them. We report the
    // standard raw moments (v^1..v^5), which is what the printed orders describe.
    let s1 = pow_sum(1);
    let s2 = pow_sum(2);
    let s3 = pow_sum(3);
    let s4 = pow_sum(4);
    let s5 = pow_sum(5);

    let n = count as f64;
    let empirical = [s1 / n, s2 / n, s3 / n, s4 / n, s5 / n];

    let first_order = 1.0 / lambda.powi(1);
    let second_order = 2.0 / lambda.powi(2);
    let third_order = (3.0 * 2.0) / lambda.powi(3);
    let fourth_order = (4.0 * 3.0 * 2.0) / lambda.powi(4);
    let fifth_order = (5.0 * 4.0 * 3.0 * 2.0) / lambda.powi(5);
    let theoretical = [
        first_order,
        second_order,
        third_order,
        fourth_order,
        fifth_order,
    ];

    println!("{first_order} {second_order} {third_order} {fourth_order} {fifth_order}");
    println!(
        "{} {} {} {} {}",
        empirical[0], empirical[1], empirical[2], empirical[3], empirical[4]
    );

    ExponentialMoments {
        theoretical,
        empirical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::capabilities::SeededRandom;

    #[test]
    fn uniform_samples_in_range() {
        let mut rng = SeededRandom::new(11);
        let v = run_uniform(&mut rng);
        assert_eq!(v.len(), 100_000);
        assert!(v.iter().all(|&x| (5.0..8.0).contains(&x)));
    }

    #[test]
    fn exponential_first_moment_is_near_one() {
        let mut rng = SeededRandom::new(11);
        let m = run_exponential(&mut rng);
        assert_eq!(m.theoretical[0], 1.0);
        // Mean of exp(1) is 1; with 1000 samples it should be in a loose band.
        assert!((m.empirical[0] - 1.0).abs() < 0.5);
    }
}
