//! Engine-wide numeric-precision policy and helpers.
//!
//! The TypeScript engine ran a deliberate TWO-TIER precision strategy, and this
//! module makes it explicit in Rust. (See `src/des/test/float-bias-test.ts`,
//! which bounds the exact failure modes below.)
//!
//! ## Tier 1 — numerical kernels stay `f64`
//!
//! Linear algebra, optimization, ODE integration, LP/simplex, eigen-solvers,
//! quadrature, statistics and RL value functions use plain `f64`. This is
//! intentional, not lazy:
//!
//!   * The engine cross-validates every model against five other
//!     implementations (Python `float64` / numpy / scipy / R / Octave). Those
//!     references are all `float64`, so the engine must be too — swapping in
//!     decimals would make the numbers stop matching.
//!   * These algorithms are *iterative approximations* (Newton, RK45, power
//!     iteration, simplex pivots). Their truncation error dwarfs `f64`
//!     round-off, so exact arithmetic buys nothing and costs ~100x.
//!   * `float-bias-test.ts` F2 measures plain-`f64` accumulation drift over
//!     1,000,000 ticks at ~1e-6 (relative ~2e-10) — negligible for simulation.
//!
//! The one `f64` hazard worth guarding is COMPOUND ACCUMULATION of many small
//! values in a hot loop. For that use [`KahanAccumulator`] / [`kahan_sum`]
//! (compensated summation, accurate to a few ULP). Never compare `f64` with
//! `==`; use [`approx_eq`] / [`approx_eq_rel`] / [`almost_zero`].
//!
//! ## Tier 2 — exact base-10 decimals for compound bookkeeping
//!
//! Where the TS code used `mathjs.BigNumber` it did so to keep a *running total*
//! exact across millions of updates, then coerced back to a `Number` on read.
//! Those domains are:
//!
//!   * the DES simulation clock / time accrual (`time-accrued.ts`, the entity
//!     framework's elapsed-time bookkeeping),
//!   * money / prediction-market balances (`factmachine` markets),
//!   * routing probabilities that must sum to exactly 1 (`probability-decision`),
//!   * the random-variable algebra (`random-variables/rv.ts`).
//!
//! These map to [`Decimal`] (re-exported from `rust_decimal`): a 96-bit base-10
//! decimal that represents values like `0.05` and `0.1` EXACTLY, so summing
//! `0.05` a million times yields exactly `50000`, and round-tripping to `f64`
//! and back is lossless. Construct decimals with [`bgn`] (from an `f64`, matching
//! TS `math.bignumber(String(x))`), [`dec`]-style integer helpers, or the
//! re-exported [`dec!`] macro for literals. Coerce to `f64` on read-out with
//! [`to_f64`] (the analog of TS `Number(bn.toString())`).
//!
//! ## Tier 3 — exact rationals
//!
//! For the rare spots needing exact `p/q` (not terminating decimals), use the
//! re-exported [`Rational64`] / [`BigRational`] and the [`frac`] helper.

pub use num_rational::{BigRational, Rational64};
pub use rust_decimal::Decimal;
pub use rust_decimal_macros::dec;

use core::str::FromStr;

// -----------------------------------------------------------------------------
// Tier 2 — exact decimals (the `mathjs.BigNumber` analog)
// -----------------------------------------------------------------------------

/// Build an exact [`Decimal`] from an `f64`, matching the TypeScript engine's
/// `bgn` / `math.bignumber(String(x))`.
///
/// The subtlety: `Decimal::from_f64_retain(0.05)` captures the binary
/// representation error of `0.05` (`0.05000000000000000277…`). TS avoided this
/// by stringifying first (`String(0.05) === "0.05"`). We do the same — format
/// with the shortest round-tripping decimal (Rust's `{}` uses Ryū/Grisu, just
/// like JS `String`) and parse that — so `bgn(0.05)` is exactly `0.05`.
///
/// Non-finite inputs (`NaN`, `±∞`) have no decimal representation and panic
/// (an invariant violation — decimals are for exact bookkeeping, never for
/// the output of a numerical kernel that might diverge).
pub fn bgn(x: f64) -> Decimal {
    if !x.is_finite() {
        panic!("bgn: cannot represent non-finite value {x} as an exact Decimal");
    }
    // `x.to_string()` is the shortest string that round-trips to `x`, identical
    // to JS `String(x)`; parsing it yields the exact decimal the literal names.
    Decimal::from_str(&x.to_string())
        .or_else(|_| Decimal::from_scientific(&x.to_string()))
        .unwrap_or_else(|e| panic!("bgn: failed to parse {x} as Decimal: {e}"))
}

/// Exact [`Decimal`] from an integer (no rounding, no `f64` round-trip).
#[inline]
pub fn bgn_int(i: i64) -> Decimal {
    Decimal::from(i)
}

/// Coerce a [`Decimal`] back to `f64` for read-out / interop, matching TS
/// `Number(bn.toString())`. For the terminating decimals the engine actually
/// stores (clock ticks, prices, probabilities) this is lossless.
#[inline]
pub fn to_f64(d: Decimal) -> f64 {
    // String round-trip mirrors `Number(bn.toString())` exactly; the parse
    // cannot fail for a value produced by Decimal's own `Display`.
    d.to_string().parse::<f64>().unwrap_or_else(|_| {
        use num_traits::ToPrimitive;
        d.to_f64().unwrap_or(f64::NAN)
    })
}

// -----------------------------------------------------------------------------
// Tier 3 — exact rationals
// -----------------------------------------------------------------------------

/// Exact rational `n/d`. Panics on a zero denominator (invariant violation),
/// matching `Rational64`'s own contract.
#[inline]
pub fn frac(n: i64, d: i64) -> Rational64 {
    Rational64::new(n, d)
}

// -----------------------------------------------------------------------------
// Tier 1 — `f64` accumulation & comparison guards
// -----------------------------------------------------------------------------

/// Kahan–Babuška compensated summation: sum `f64`s while tracking the running
/// round-off in a correction term, keeping the result accurate to a few ULP
/// even over millions of additions. Use this instead of a naive `+=` fold for
/// any long-horizon `f64` accumulation in a hot kernel.
///
/// The compensation steps are kept as separate statements on purpose: Rust does
/// NOT auto-contract them into an FMA (there is no implicit `-ffast-math`), so
/// the algebraic cancellation that makes Kahan work is preserved.
#[derive(Clone, Copy, Debug, Default)]
pub struct KahanAccumulator {
    sum: f64,
    /// Running compensation (the low-order bits lost on each add).
    c: f64,
}

impl KahanAccumulator {
    #[inline]
    pub fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    /// Add one value, folding in the previous round-off.
    #[inline]
    pub fn add(&mut self, value: f64) {
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    /// The compensated running total.
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }
}

/// Compensated sum of a slice (see [`KahanAccumulator`]).
pub fn kahan_sum(values: &[f64]) -> f64 {
    let mut acc = KahanAccumulator::new();
    for &v in values {
        acc.add(v);
    }
    acc.sum()
}

/// Absolute-tolerance float equality: `|a - b| <= eps`. Prefer this over `==`.
#[inline]
pub fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

/// Relative-tolerance float equality: `|a - b| <= eps * max(1, |a|, |b|)`.
/// Robust across magnitudes; use for comparing computed quantities.
#[inline]
pub fn approx_eq_rel(a: f64, b: f64, eps: f64) -> bool {
    let scale = 1.0_f64.max(a.abs()).max(b.abs());
    (a - b).abs() <= eps * scale
}

/// Whether `x` is within `eps` of zero.
#[inline]
pub fn almost_zero(x: f64, eps: f64) -> bool {
    x.abs() <= eps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgn_is_exact_for_engine_decimals() {
        // F3 of float-bias-test: these values must round-trip exactly.
        for &x in &[0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 1.3, 1.5, 2.5, 1200.0, 800.0] {
            let d = bgn(x);
            assert_eq!(to_f64(d), x, "round-trip drifted for {x}");
        }
        // The whole point: bgn(0.1) is EXACTLY 0.1, not 0.1000000000000000055.
        assert_eq!(bgn(0.1), dec!(0.1));
        assert_eq!(bgn(0.05), dec!(0.05));
    }

    #[test]
    fn decimal_accumulation_does_not_drift() {
        // F2: adding 0.05 a million times must land EXACTLY on 50000 in decimal,
        // whereas naive f64 would drift by ~1e-6.
        let step = bgn(0.05);
        let mut clock = Decimal::ZERO;
        for _ in 0..1_000_000 {
            clock += step;
        }
        assert_eq!(clock, dec!(50000));
        assert_eq!(to_f64(clock), 50000.0);

        // Contrast: naive f64 drifts (documenting WHY we use Decimal here).
        let mut naive = 0.0_f64;
        for _ in 0..1_000_000 {
            naive += 0.05;
        }
        assert!((naive - 50000.0).abs() > 1e-9, "naive f64 should visibly drift");
    }

    #[test]
    fn kahan_beats_naive_summation() {
        // F2: Kahan summation of 0.05 x 1e6 stays within a few ULP of 50000.
        let v = vec![0.05_f64; 1_000_000];
        let k = kahan_sum(&v);
        let ulp = 2f64.powi(-52) * 50000.0_f64.max(1.0);
        assert!((k - 50000.0).abs() < 100.0 * ulp, "kahan drift {}", k - 50000.0);
    }

    #[test]
    fn rationals_are_exact() {
        // 1/3 + 1/3 + 1/3 == 1 exactly (impossible in f64).
        let third = frac(1, 3);
        assert_eq!(third + third + third, Rational64::from_integer(1));
    }

    #[test]
    fn comparison_helpers() {
        assert!(approx_eq(0.1 + 0.2, 0.3, 1e-12));
        assert!(!approx_eq(0.1 + 0.2, 0.3, 0.0)); // exact == would (correctly) fail
        assert!(approx_eq_rel(1e9, 1e9 + 1.0, 1e-9));
        assert!(almost_zero(1e-15, 1e-12));
    }
}
