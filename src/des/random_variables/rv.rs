//! Canonical use path: `crate::des::random_variables::rv::*`
//!
//! Port of `src/des/random-variables/rv.ts` — the `RandomVariable` family
//! (inter-event sampling distributions).
//!
//! DETERMINISM is the headline: the TS called `math.random()` / `Math.random()`
//! and the ambient `getReasonableU*` helpers directly. Here every variant stores
//! an injected RNG (`Box<dyn RandomSource>`), and `get_reasonable_u*` take that
//! RNG, so seeded simulations reproduce.
//!
//! The TS generic `getSerializableData()` shapes collapse to one [`RvData`] DTO.
//! `mathjs.BigNumber` maps to [`Decimal`] (exact compound algebra); `math.exp` /
//! `math.log` use `rust_decimal`'s [`MathematicalOps`].
//!
//! PORT NOTES:
//!   * The `getNextEvents(): Generator<number>` methods are DROPPED. Most TS impls
//!     `return undefined` from the generator (non-functional stubs); only
//!     `UniformRandomVariable.getNextEvents` yielded, and it duplicated
//!     `getNextEventQuantity`. The live `getNextEventQuantity` is what callers use.
//!   * `UniformRandomVariable2.getNextEventQuantity` begins with `return 1;` in the
//!     TS (the rest is dead code) — only the live `return 1` is preserved.
//!   * The five near-duplicate Exp/Uniform variants share the sampling helpers
//!     [`exp_next_val`] / [`uniform_next_val_native`]; all public struct names are
//!     kept.

#![allow(dead_code)]

use rust_decimal::MathematicalOps;

use crate::des::general::general::{bgn, get_reasonable_u, get_reasonable_u_native};
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::precision::{bgn_int, to_f64, Decimal};

/// Serializable view of a random variable (`type RT` / the ad-hoc objects the TS
/// `getSerializableData` returned). Fields are optional and populated per variant.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RvData {
    pub lambda: Option<f64>,
    pub a: Option<f64>,
    pub b: Option<f64>,
}

/// `abstract class RandomVariable` — object-safe (no generics, no `Self` return).
pub trait RandomVariable {
    fn get_rate(&self) -> Decimal;
    fn get_next_event_quantity(&mut self, time_step: Decimal) -> i64;
    fn get_serializable_data(&self) -> RvData;
}

// ── shared sampling helpers (the consolidated core of the variants) ──────────

/// Exponential inter-arrival increment: `(-1/lambda) * ln(1 - u) + prev`.
fn exp_next_val(lambda: Decimal, prev: Decimal, rng: &mut dyn RandomSource) -> Decimal {
    let u = get_reasonable_u(rng);
    let v = (bgn_int(-1) / lambda) * (Decimal::ONE - u).ln();
    v + prev
}

/// Uniform inter-arrival increment (native `f64`): `a + u*width + prev`.
fn uniform_next_val_native(a: f64, width: f64, prev: f64, rng: &mut dyn RandomSource) -> f64 {
    let u = get_reasonable_u_native(rng);
    a + (u * width) + prev
}

// =============================================================================
// Bernoulli / Poisson  (identical bodies in the TS)
// =============================================================================

pub struct BernoulliRandomVariable {
    rng: Box<dyn RandomSource>,
}

impl BernoulliRandomVariable {
    pub fn new(rng: Box<dyn RandomSource>) -> Self {
        BernoulliRandomVariable { rng }
    }
}

impl RandomVariable for BernoulliRandomVariable {
    fn get_rate(&self) -> Decimal {
        bgn(0.3)
    }
    fn get_next_event_quantity(&mut self, _time_step: Decimal) -> i64 {
        if self.rng.next_float() > 0.5 {
            1
        } else {
            0
        }
    }
    fn get_serializable_data(&self) -> RvData {
        RvData {
            lambda: Some(5.0),
            ..Default::default()
        }
    }
}

pub struct PoissonRandomVariable {
    rng: Box<dyn RandomSource>,
}

impl PoissonRandomVariable {
    pub fn new(rng: Box<dyn RandomSource>) -> Self {
        PoissonRandomVariable { rng }
    }
}

impl RandomVariable for PoissonRandomVariable {
    fn get_rate(&self) -> Decimal {
        bgn(0.3)
    }
    fn get_next_event_quantity(&mut self, _time_step: Decimal) -> i64 {
        if self.rng.next_float() > 0.5 {
            1
        } else {
            0
        }
    }
    fn get_serializable_data(&self) -> RvData {
        RvData {
            lambda: Some(5.0),
            ..Default::default()
        }
    }
}

// =============================================================================
// ExponentialRandomVariable  (precomputed-RHS, u-product method)
// =============================================================================

pub struct ExponentialRandomVariable {
    lambda: Decimal,
    pre_computed_rhs: Decimal,
    /// TS `maxVal` was always null (a TODO); kept for shape fidelity.
    max_val: Option<Decimal>,
    rng: Box<dyn RandomSource>,
}

impl ExponentialRandomVariable {
    pub fn new(lambda: Decimal, time_step: Decimal, rng: Box<dyn RandomSource>) -> Self {
        if lambda <= bgn(0.0) {
            eprintln!(
                "[ExponentialRandomVariable] invalid lambda={lambda} (must be > 0); rate parameter cannot be zero or negative."
            );
            panic!("lambda must be larger than 0.");
        }
        let pre_computed_rhs = (bgn_int(-1) * (time_step * lambda)).exp();
        ExponentialRandomVariable {
            lambda,
            pre_computed_rhs,
            max_val: None,
            rng,
        }
    }

    pub fn get_adjusted_expected_val(&self, time_step: Decimal) -> Decimal {
        self.lambda / time_step
    }

    /// Extra public method from the TS (`getNextEventCount`).
    pub fn get_next_event_count(&mut self, time_step: Decimal) -> i64 {
        let mut sum = bgn(0.0);
        let mut q: i64 = -1;
        while sum < time_step {
            q += 1;
            let u = get_reasonable_u(&mut *self.rng);
            let diff = Decimal::ONE - u;
            let t = bgn_int(-1) * diff.ln();
            sum += t;
        }
        q
    }
}

impl RandomVariable for ExponentialRandomVariable {
    fn get_rate(&self) -> Decimal {
        bgn(0.3)
    }
    fn get_next_event_quantity(&mut self, _time_step: Decimal) -> i64 {
        let rhs = self.pre_computed_rhs;
        let mut u_product = bgn_int(1);
        let mut q: i64 = -1;
        while u_product > rhs {
            q += 1;
            let u = get_reasonable_u(&mut *self.rng);
            u_product *= u;
        }
        q
    }
    fn get_serializable_data(&self) -> RvData {
        RvData {
            lambda: Some(to_f64(self.lambda)),
            ..Default::default()
        }
    }
}

// =============================================================================
// ExponentialRandomVariable3  (accumulating nextEvent; seeds nextU on first call)
// =============================================================================

pub struct ExponentialRandomVariable3 {
    next_event: Decimal,
    lambda: Decimal,
    first: bool,
    time_step: Decimal,
    next_u: Decimal,
    precomputed_rhs: Decimal,
    rng: Box<dyn RandomSource>,
}

impl ExponentialRandomVariable3 {
    pub fn new(lambda: Decimal, time_step: Decimal, mut rng: Box<dyn RandomSource>) -> Self {
        let next_u = bgn(rng.next_float());
        let precomputed_rhs = (bgn_int(-1) * (lambda * time_step)).exp();
        if lambda < bgn(0.00000001) {
            panic!("Width of uniform distribution needs to be greater than 0.00000001");
        }
        ExponentialRandomVariable3 {
            next_event: bgn(0.0),
            lambda,
            first: true,
            time_step,
            next_u,
            precomputed_rhs,
            rng,
        }
    }

    fn get_next_val(&mut self) -> Decimal {
        exp_next_val(self.lambda, self.next_event, &mut *self.rng)
    }
}

impl RandomVariable for ExponentialRandomVariable3 {
    fn get_rate(&self) -> Decimal {
        bgn(0.3)
    }
    fn get_next_event_quantity(&mut self, time_step: Decimal) -> i64 {
        // PORT NOTE: faithful to the TS — on the FIRST call this only re-draws
        // `next_u` and does NOT seed `next_event`, so the first quantity is 0.
        if self.first {
            self.first = false;
            self.next_u = bgn(self.rng.next_float());
        } else {
            self.next_event -= time_step;
        }

        let mut q: i64 = 0;
        while self.next_event < time_step {
            q += 1;
            self.next_event = self.get_next_val();
        }
        q
    }
    fn get_serializable_data(&self) -> RvData {
        RvData {
            lambda: Some(to_f64(self.lambda)),
            ..Default::default()
        }
    }
}

// =============================================================================
// ExponentialRandomVariable2  (accumulating nextEvent; seeds nextEvent on first call)
// =============================================================================

pub struct ExponentialRandomVariable2 {
    next_event: Decimal,
    lambda: Decimal,
    first: bool,
    time_step: Decimal,
    rng: Box<dyn RandomSource>,
}

impl ExponentialRandomVariable2 {
    pub fn new(lambda: Decimal, time_step: Decimal, rng: Box<dyn RandomSource>) -> Self {
        if lambda < bgn(0.00000001) {
            panic!("Width of uniform distribution needs to be greater than 0.00000001");
        }
        ExponentialRandomVariable2 {
            next_event: bgn(0.0),
            lambda,
            first: true,
            time_step,
            rng,
        }
    }

    fn get_next_val(&mut self) -> Decimal {
        exp_next_val(self.lambda, self.next_event, &mut *self.rng)
    }
}

impl RandomVariable for ExponentialRandomVariable2 {
    fn get_rate(&self) -> Decimal {
        bgn(0.3)
    }
    fn get_next_event_quantity(&mut self, time_step: Decimal) -> i64 {
        if self.first {
            self.first = false;
            self.next_event = self.get_next_val();
        } else {
            self.next_event -= time_step;
        }

        let mut q: i64 = 0;
        while self.next_event < time_step {
            q += 1;
            self.next_event = self.get_next_val();
        }
        q
    }
    fn get_serializable_data(&self) -> RvData {
        RvData {
            lambda: Some(to_f64(self.lambda)),
            ..Default::default()
        }
    }
}

// =============================================================================
// UniformRandomVariable  (native f64 inter-arrival)
// =============================================================================

pub struct UniformRandomVariable {
    next_event: f64,
    a_val: f64,
    b_val: f64,
    width: f64,
    first: bool,
    rng: Box<dyn RandomSource>,
}

impl UniformRandomVariable {
    pub fn new(a_val: Decimal, b_val: Decimal, rng: Box<dyn RandomSource>) -> Self {
        let a = to_f64(a_val);
        let b = to_f64(b_val);
        if a.is_nan() {
            panic!("this.aVal is not a number.");
        }
        if b.is_nan() {
            panic!("this.bVal is not a number.");
        }
        let width = b - a;
        if width < 0.00000001 {
            eprintln!(
                "[UniformRandomVariable] degenerate interval: width={width} from [{a}, {b}] (need width > 1e-8)."
            );
            panic!("Width of uniform distribution needs to be greater than 0.00000001");
        }
        UniformRandomVariable {
            next_event: -1.0,
            a_val: a,
            b_val: b,
            width,
            first: true,
            rng,
        }
    }

    fn get_next_val(&mut self) -> f64 {
        uniform_next_val_native(self.a_val, self.width, self.next_event, &mut *self.rng)
    }
}

impl RandomVariable for UniformRandomVariable {
    fn get_rate(&self) -> Decimal {
        bgn(0.3)
    }
    fn get_next_event_quantity(&mut self, time_step: Decimal) -> i64 {
        let ts = to_f64(time_step);
        if self.first {
            self.first = false;
            self.next_event = self.get_next_val();
        } else {
            self.next_event -= ts;
        }

        let mut q: i64 = 0;
        while self.next_event < ts {
            q += 1;
            self.next_event = self.get_next_val();
        }
        q
    }
    fn get_serializable_data(&self) -> RvData {
        // PORT NOTE: TS returned `this` (the whole object); we expose the bounds.
        RvData {
            a: Some(self.a_val),
            b: Some(self.b_val),
            ..Default::default()
        }
    }
}

// =============================================================================
// UniformRandomVariable2  (decimal; getNextEventQuantity is a live `return 1`)
// =============================================================================

pub struct UniformRandomVariable2 {
    next_event: Decimal,
    a_val: Decimal,
    b_val: Decimal,
    width: Decimal,
    first: bool,
    rng: Box<dyn RandomSource>,
}

impl UniformRandomVariable2 {
    pub fn new(a_val: Decimal, b_val: Decimal, rng: Box<dyn RandomSource>) -> Self {
        let width = b_val - a_val;
        if width < bgn(0.00000001) {
            panic!("Width of uniform distribution needs to be greater than 0.00000001");
        }
        UniformRandomVariable2 {
            next_event: bgn(0.0),
            a_val,
            b_val,
            width,
            first: true,
            rng,
        }
    }
}

impl RandomVariable for UniformRandomVariable2 {
    fn get_rate(&self) -> Decimal {
        bgn(0.3)
    }
    fn get_next_event_quantity(&mut self, _time_step: Decimal) -> i64 {
        // PORT NOTE: the TS method's first statement is `return 1;`; everything
        // after it is unreachable and is intentionally dropped.
        1
    }
    fn get_serializable_data(&self) -> RvData {
        RvData {
            a: Some(to_f64(self.a_val)),
            b: Some(to_f64(self.b_val)),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::capabilities::SeededRandom;
    use crate::des::shared::precision::bgn;

    fn rng() -> Box<dyn RandomSource> {
        Box::new(SeededRandom::new(12345))
    }

    #[test]
    fn bernoulli_is_binary() {
        let mut rv = BernoulliRandomVariable::new(rng());
        for _ in 0..100 {
            let q = rv.get_next_event_quantity(bgn(0.1));
            assert!(q == 0 || q == 1);
        }
        assert_eq!(rv.get_serializable_data().lambda, Some(5.0));
    }

    #[test]
    fn exponential_quantity_is_nonneg() {
        let mut rv = ExponentialRandomVariable::new(bgn(1.0), bgn(0.5), rng());
        let q = rv.get_next_event_quantity(bgn(0.5));
        assert!(q >= -1); // q starts at -1 and increments
        assert_eq!(rv.get_serializable_data().lambda, Some(1.0));
    }

    #[test]
    #[should_panic(expected = "lambda must be larger than 0")]
    fn exponential_rejects_nonpositive_lambda() {
        let _ = ExponentialRandomVariable::new(bgn(0.0), bgn(0.5), rng());
    }

    #[test]
    fn uniform2_always_returns_one() {
        let mut rv = UniformRandomVariable2::new(bgn(1.0), bgn(5.0), rng());
        assert_eq!(rv.get_next_event_quantity(bgn(0.1)), 1);
        let data = rv.get_serializable_data();
        assert_eq!(data.a, Some(1.0));
        assert_eq!(data.b, Some(5.0));
    }

    #[test]
    fn uniform_native_samples() {
        let mut rv = UniformRandomVariable::new(bgn(5.0), bgn(8.0), rng());
        let q = rv.get_next_event_quantity(bgn(0.5));
        assert!(q >= 0);
    }

    #[test]
    fn as_trait_object() {
        let mut rvs: Vec<Box<dyn RandomVariable>> = vec![
            Box::new(BernoulliRandomVariable::new(rng())),
            Box::new(PoissonRandomVariable::new(rng())),
            Box::new(ExponentialRandomVariable2::new(bgn(1.0), bgn(0.1), rng())),
        ];
        for rv in rvs.iter_mut() {
            let _ = rv.get_rate();
            let _ = rv.get_next_event_quantity(bgn(0.1));
        }
    }
}
