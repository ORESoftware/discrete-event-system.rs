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
use core::sync::atomic::{AtomicU64, Ordering};

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

// -----------------------------------------------------------------------------
// Tier 1 — divide-by-zero, domain, and non-finite guards
// -----------------------------------------------------------------------------
//
// The numerical kernels stay on raw IEEE-754 `f64` (see the module header), so
// every arithmetic hazard the platform inherits — division by zero, `0.0/0.0`,
// `sqrt(-1)`, `ln(0)`, `exp(10000)`, `0.0.powf(-1.0)` — silently produces a
// `NaN` or `±∞` that then propagates through an entire simulation trace, an LP
// objective, or a rendered frame before anyone notices. These helpers turn each
// hazard into a *bounded, finite, logged* value so a single degenerate input
// (a zero mass, an empty population, a collapsed denominator) regularizes the
// step instead of poisoning the run.
//
// They are deliberately identity functions on well-posed inputs: a non-zero
// finite denominator, a non-negative `sqrt` argument, a positive `ln` argument
// are passed straight through, so the engine still matches its `float64`
// reference implementations bit-for-bit on every healthy model. Only the
// degenerate path is altered (and logged), never the healthy one.

/// Sign-preserving epsilon used to regularize a vanishing denominator. Tiny
/// enough that `x / PERTURB_EPS` is large (`~1e12·x`) yet finite, so the step is
/// dominated by — but does not diverge from — the singularity.
pub const PERTURB_EPS: f64 = 1e-12;

/// Process-wide count of numeric-guard interventions (saturating). Surfaced to
/// callers via [`numeric_event_count`] so a runner can assert "this model never
/// hit a guard" or attach the count to its result summary.
static NUMERIC_EVENTS: AtomicU64 = AtomicU64::new(0);

/// After this many emitted lines, further events are counted but no longer
/// printed, so a hot ODE/LP loop that hits a singularity every tick cannot flood
/// stderr. The running total is still available via [`numeric_event_count`].
const NUMERIC_LOG_CAP: u64 = 64;

/// Emit one structured, rate-limited numeric-guard record to stderr and bump the
/// global counter. The line is OpenTelemetry-style `key=value` attributes
/// (`numeric.guard kind=… detail=… seq=…`) so a log scraper / OTel stderr
/// receiver can parse it without a logging dependency being pulled into this
/// otherwise `std`-only kernel. Set `DES_NUMERIC_QUIET=1` to suppress printing
/// entirely (the counter still advances).
pub fn numeric_event(kind: &str, detail: impl core::fmt::Display) {
    let seq = NUMERIC_EVENTS.fetch_add(1, Ordering::Relaxed);
    if seq >= NUMERIC_LOG_CAP || std::env::var_os("DES_NUMERIC_QUIET").is_some() {
        return;
    }
    eprintln!("numeric.guard kind={kind} detail=\"{detail}\" seq={seq}");
    if seq + 1 == NUMERIC_LOG_CAP {
        eprintln!(
            "numeric.guard kind=rate_limited detail=\"further numeric-guard events suppressed; \
             see numeric_event_count()\" seq={}",
            seq + 1
        );
    }
}

/// Total number of numeric-guard interventions since process start.
#[inline]
pub fn numeric_event_count() -> u64 {
    NUMERIC_EVENTS.load(Ordering::Relaxed)
}

/// Replace a non-finite value with a bounded finite sentinel, logging the
/// intervention: `NaN → 0`, `+∞ → f64::MAX`, `−∞ → f64::MIN`. Finite values are
/// returned untouched (and unlogged). This is the last line of defense applied
/// to the *result* of every guarded operation, so a `NaN`/`∞` produced by an
/// unforeseen combination still cannot escape the kernel.
#[inline]
pub fn clamp_finite(x: f64, ctx: impl core::fmt::Display) -> f64 {
    if x.is_finite() {
        return x;
    }
    let repl = if x.is_nan() {
        0.0
    } else if x > 0.0 {
        f64::MAX
    } else {
        f64::MIN
    };
    numeric_event("non_finite_clamped", format_args!("{ctx} -> {repl}"));
    repl
}

/// Division that never yields `±∞` or `NaN` from a zero/non-finite denominator.
///
/// * A denominator of exactly `±0.0` is perturbed by a sign-preserving
///   [`PERTURB_EPS`] (so `1/0 → ~1e12`, `−1/0 → ~−1e12`, `0/0 → 0`), turning a
///   pole into a large-but-finite, sign-correct value.
/// * A non-finite denominator falls through to IEEE semantics and is then
///   [`clamp_finite`]d (`a/±∞ → 0`, `a/NaN → NaN → 0`).
/// * A non-zero finite denominator is used as-is — the healthy path is exact.
#[inline]
pub fn safe_div(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        let eps = if b.is_sign_negative() {
            -PERTURB_EPS
        } else {
            PERTURB_EPS
        };
        numeric_event("divide_by_zero", format_args!("{a} / 0 -> {a} / {eps}"));
        return clamp_finite(a / eps, format_args!("safe_div({a}, 0)"));
    }
    clamp_finite(a / b, format_args!("safe_div({a}, {b})"))
}

/// `a^b` guarded against the overflow / domain pitfalls of [`f64::powf`]
/// (`exp`-style overflow → `∞`, `(−2)^0.5` → `NaN`, `0^−1` → `∞`). Result is
/// [`clamp_finite`]d; healthy bases/exponents are exact.
#[inline]
pub fn safe_powf(a: f64, b: f64) -> f64 {
    clamp_finite(a.powf(b), format_args!("{a}^{b}"))
}

/// `sqrt` clamped to its domain: a negative argument is treated as `0` (the
/// nearest in-domain point) rather than returning `NaN`. Matches the common
/// regularization `sqrt(max(x, 0))` used to keep variances/norms real.
#[inline]
pub fn safe_sqrt(x: f64) -> f64 {
    if x < 0.0 {
        numeric_event("sqrt_negative", format_args!("sqrt({x}) -> 0"));
        return 0.0;
    }
    x.sqrt()
}

/// Natural log clamped to its domain: a non-positive argument is pulled to the
/// nearest representable positive magnitude (`ln(0) → ln(PERTURB_EPS)`, a large
/// finite negative; `ln(−x) → ln(|x|)`) instead of returning `−∞`/`NaN`.
#[inline]
pub fn safe_ln(x: f64) -> f64 {
    if x <= 0.0 {
        numeric_event("log_domain", format_args!("ln({x}) -> ln(|x| clamped)"));
        return x.abs().max(PERTURB_EPS).ln();
    }
    x.ln()
}

/// `exp` with overflow clamped to a finite range (`exp(10000) → f64::MAX`
/// instead of `+∞`).
#[inline]
pub fn safe_exp(x: f64) -> f64 {
    clamp_finite(x.exp(), format_args!("exp({x})"))
}

/// `asin`/`acos` argument clamped to `[-1, 1]` so floating-point drift just past
/// the boundary (`acos(1.0000000002)`) yields the endpoint value rather than
/// `NaN`. Returns the clamped argument; the caller applies the actual function.
#[inline]
pub fn clamp_unit_interval(x: f64) -> f64 {
    if x < -1.0 || x > 1.0 {
        let c = x.clamp(-1.0, 1.0);
        numeric_event("asin_acos_domain", format_args!("{x} -> {c}"));
        return c;
    }
    x
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
        assert!(
            (naive - 50000.0).abs() > 1e-9,
            "naive f64 should visibly drift"
        );
    }

    #[test]
    fn kahan_beats_naive_summation() {
        // F2: Kahan summation of 0.05 x 1e6 stays within a few ULP of 50000.
        let v = vec![0.05_f64; 1_000_000];
        let k = kahan_sum(&v);
        let ulp = 2f64.powi(-52) * 50000.0_f64.max(1.0);
        assert!(
            (k - 50000.0).abs() < 100.0 * ulp,
            "kahan drift {}",
            k - 50000.0
        );
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

    #[test]
    fn safe_div_is_identity_on_healthy_inputs() {
        // The whole correctness contract: well-posed division is bit-for-bit the
        // same as `/`, so reference cross-validation is unaffected.
        assert_eq!(safe_div(6.0, 3.0), 2.0);
        assert_eq!(safe_div(-1.0, 4.0), -0.25);
        assert_eq!(safe_div(0.0, 5.0), 0.0);
        assert_eq!(safe_div(7.5, 2.5), 3.0);
    }

    #[test]
    fn guard_counter_advances_on_intervention() {
        // The counter is process-global; assert it strictly increases across a
        // guarded op rather than a specific value (other tests share it).
        let before = numeric_event_count();
        let _ = safe_div(1.0, 0.0);
        assert!(numeric_event_count() > before);
    }

    #[test]
    fn safe_div_regularizes_zero_denominator() {
        // 1/0 -> large finite positive; -1/0 -> large finite negative; 0/0 -> 0.
        assert!(safe_div(1.0, 0.0).is_finite());
        assert!(safe_div(1.0, 0.0) > 1e9);
        assert!(safe_div(-1.0, 0.0) < -1e9);
        assert_eq!(safe_div(0.0, 0.0), 0.0);
        // Sign of a negative zero denominator is preserved.
        assert!(safe_div(1.0, -0.0) < 0.0);
    }

    #[test]
    fn clamp_finite_maps_nonfinite_to_sentinels() {
        assert_eq!(clamp_finite(3.0, "ok"), 3.0);
        assert_eq!(clamp_finite(f64::NAN, "nan"), 0.0);
        assert_eq!(clamp_finite(f64::INFINITY, "+inf"), f64::MAX);
        assert_eq!(clamp_finite(f64::NEG_INFINITY, "-inf"), f64::MIN);
    }

    #[test]
    fn domain_guards_stay_finite() {
        assert_eq!(safe_sqrt(-5.0), 0.0); // domain clamp, not NaN
        assert_eq!(safe_sqrt(9.0), 3.0); // healthy path exact
        assert!(safe_ln(0.0).is_finite()); // ln(0) -> large finite negative
        assert!(safe_ln(0.0) < -20.0);
        assert!(safe_ln(-3.0).is_finite()); // ln(-3) -> ln(3)
        assert!((safe_ln(std::f64::consts::E) - 1.0).abs() < 1e-12);
        assert_eq!(safe_exp(10000.0), f64::MAX); // overflow clamp, not +inf
        assert!((safe_exp(0.0) - 1.0).abs() < 1e-12);
        assert!(safe_powf(0.0, -1.0).is_finite()); // 0^-1 -> clamped
        assert!(safe_powf(-2.0, 0.5).is_finite()); // NaN -> clamped
        assert_eq!(clamp_unit_interval(1.0000000002), 1.0);
        assert_eq!(clamp_unit_interval(-1.5), -1.0);
        assert_eq!(clamp_unit_interval(0.5), 0.5);
    }
}
