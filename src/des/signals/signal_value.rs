//! Canonical use path: `crate::des::signals::signal_value::*`
//!
//! Port of `src/des/signals/signal-value.ts` — the signal "sample" carried
//! through the signal graph, plus the 0/1 constant specializations.
//!
//! Inheritance chain `SignalEntity -> AbstractSignalValue -> SignalValue ->
//! {SignalValueUnity, SignalValueZero}` is flattened: shared moving-entity state
//! lives in a [`MovingCore`]; the abstract layer becomes the
//! [`AbstractSignalValue`] trait; the concrete sample is [`SignalValue`].
//! `Unity`/`Zero` specialized the generic `V` to `BigNumber` — here they are
//! plain constructors ([`SignalValue::unity`] / [`SignalValue::zero`]).
//!
//! Conversion notes (file-specific):
//!   * `value = <unknown>null as V` placeholder + `math.isNumber(val)` guard ->
//!     `value: Option<Decimal>`; only `Some` when a numeric sample is supplied.
//!     The `math.BigNumber` sample domain maps to [`Decimal`] (exact bookkeeping).
//!   * PORT NOTE: the running-total path elsewhere does `new SignalValue({val:
//!     bigNumber})`, but `math.isNumber(BigNumber)` is `false`, so the TS leaves
//!     `value` unset there (a latent bug). [`SignalValue::from_value`] carries
//!     the decimal explicitly so the signal graph is actually testable.
//!   * `getMatrixValue(): Array<Array<BigNumber>>` -> `Vec<Vec<f64>>` (empty).
//!   * `getShortUUID()` default id -> [`get_short_uuid`].
//!   * `runFinish()`/`doValidation()` `throw new Error('not implemented')` ->
//!     `panic!` (invariant violation per the migration policy).
//!   * PORT NOTE: TS `AbstractSignalValue.runTimeStep` calls `doTimeStep`, which
//!     (via `SignalEntity`) calls `runTimeStep` — a mutual recursion that would
//!     stack-overflow if ever driven on a leaf value. A `SignalValue` is a leaf
//!     and never ticked, so `run_time_step_signal` is a documented no-op here.

#![allow(dead_code)]

use crate::des::entity_moving::moving::{MovingCore, MovingEntity, MovingValue};
use crate::des::general::general::get_short_uuid;
use crate::des::r#abstract::interfaces::EntityGraphData;
use crate::des::r#abstract::r#abstract::{Entity, EntityCore};
use crate::des::shared::precision::{bgn, to_f64, Decimal};

use crate::des::signals::r#abstract::{SignalEntity, SignalTimeStepOpts};

/// `abstract class AbstractSignalValue<E,V> extends SignalEntity<E,V>` — adds the
/// abstract sample accessor `getValue(): V`.
pub trait AbstractSignalValue: SignalEntity {
    /// `abstract getValue(): V` — the carried sample (erased generic `V` ->
    /// optional [`Decimal`]).
    fn signal_value(&self) -> Option<Decimal>;
}

/// `class SignalValue<E,V>` — a single signal sample flowing as a moving-entity.
pub struct SignalValue {
    pub core: MovingCore,
    /// `value = <unknown>null as V` -> `None`; set when a numeric sample exists.
    pub value: Option<Decimal>,
}

/// `constructor({id, val})` argument bag.
#[derive(Clone, Debug, Default)]
pub struct SignalValueArgs {
    pub id: Option<String>,
    pub val: Option<f64>,
}

impl SignalValue {
    /// `new SignalValue({id, val})`. The TS `math.isNumber(val)` guard only sets
    /// the value when `val` is numeric — represented here by `Option`.
    pub fn new(args: SignalValueArgs) -> Self {
        let id = args.id.unwrap_or_else(get_short_uuid);
        SignalValue {
            core: MovingCore::new(id),
            value: args.val.map(bgn),
        }
    }

    /// Build a sample from an exact [`Decimal`] (the running-total / diff path).
    /// See the module PORT NOTE on the TS `math.isNumber` quirk.
    pub fn from_value(val: Decimal) -> Self {
        SignalValue {
            core: MovingCore::new(get_short_uuid()),
            value: Some(val),
        }
    }

    /// `class SignalValueUnity` — `value = bgn(1)`.
    pub fn unity() -> Self {
        SignalValue {
            core: MovingCore::new(get_short_uuid()),
            value: Some(bgn(1.0)),
        }
    }

    /// `class SignalValueZero` — `value = bgn(0)`.
    pub fn zero() -> Self {
        SignalValue {
            core: MovingCore::new(get_short_uuid()),
            value: Some(bgn(0.0)),
        }
    }

    /// `setValue(m)`.
    pub fn set_value(&mut self, m: Decimal) {
        self.value = Some(m);
    }

    /// `getMatrixValue(): Array<Array<BigNumber>>` -> empty `Vec<Vec<f64>>`.
    pub fn get_matrix_value(&self) -> Vec<Vec<f64>> {
        Vec::new()
    }
}

impl Entity for SignalValue {
    fn core(&self) -> &EntityCore {
        &self.core.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.entity
    }
    /// `doValidation() {}` (the base `SignalValue` is a no-op; the `Unity`/`Zero`
    /// specializations `throw` — see the dedicated constructors' note).
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        // PORT NOTE: `SignalEntity.getGraphData()` returned `null`; there is no
        // graph view for a bare sample, so the framework-required hook yields an
        // empty payload (`get_signal_graph_data` returns `None`).
        EntityGraphData::default()
    }
    fn run_time_step(&mut self, step_size: Decimal) {
        self.run_time_step_signal(step_size, None);
    }
}

impl MovingEntity for SignalValue {
    fn moving_core(&self) -> &MovingCore {
        &self.core
    }
    fn moving_core_mut(&mut self) -> &mut MovingCore {
        &mut self.core
    }
    /// The framework `getValue()` returns the erased [`MovingValue`]; the sample
    /// is surfaced through its `value` field (and via [`AbstractSignalValue`]).
    fn get_value(&self) -> MovingValue {
        MovingValue {
            id: Some(self.core.entity.id.clone()),
            value: self.value.map(to_f64),
            q: None,
        }
    }
    /// `runFinish(): void { throw new Error('not implemented.'); }`.
    fn run_finish(&mut self) {
        panic!("not implemented.");
    }
}

impl SignalEntity for SignalValue {
    /// See module PORT NOTE: TS `runTimeStep`/`doTimeStep` mutually recurse; a
    /// leaf sample is never ticked, so this is a documented no-op.
    fn run_time_step_signal(&mut self, _step_size: Decimal, _opts: Option<SignalTimeStepOpts>) {}
}

impl AbstractSignalValue for SignalValue {
    fn signal_value(&self) -> Option<Decimal> {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_with_numeric_value() {
        let sv = SignalValue::new(SignalValueArgs {
            id: Some("s1".into()),
            val: Some(3.5),
        });
        assert_eq!(sv.signal_value(), Some(bgn(3.5)));
        assert_eq!(sv.get_value().value, Some(3.5));
        assert_eq!(sv.id(), "s1");
    }

    #[test]
    fn missing_value_is_none() {
        let sv = SignalValue::new(SignalValueArgs::default());
        assert_eq!(sv.signal_value(), None);
        assert_eq!(sv.get_value().value, None);
        // default id is a 10-char short uuid
        assert_eq!(sv.id().len(), 10);
    }

    #[test]
    fn unity_and_zero_constants() {
        assert_eq!(SignalValue::unity().signal_value(), Some(bgn(1.0)));
        assert_eq!(SignalValue::zero().signal_value(), Some(bgn(0.0)));
    }

    #[test]
    fn set_value_and_matrix_value() {
        let mut sv = SignalValue::zero();
        sv.set_value(bgn(9.0));
        assert_eq!(sv.signal_value(), Some(bgn(9.0)));
        assert!(sv.get_matrix_value().is_empty());
    }

    #[test]
    #[should_panic(expected = "not implemented.")]
    fn run_finish_panics() {
        let mut sv = SignalValue::zero();
        sv.run_finish();
    }
}
