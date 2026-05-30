//! Canonical use path: `crate::des::signals::r#abstract::*`
//!
//! Port of `src/des/signals/abstract.ts` — the base of the signal-processing
//! entity family. Signals flow through the graph AS moving-entities, so a
//! `SignalEntity` is a specialization of the framework `MovingEntity`.
//!
//! PORT NOTE on the module name: the file is `abstract.ts` (a filename in
//! `signals/`, NOT the keyword directory `abstract/`). `abstract` is a Rust
//! keyword, so this module is addressed with the raw identifier
//! `crate::des::signals::r#abstract` while the file stays `abstract.rs`.
//!
//! Shape changes vs. the TS source:
//!   * `const SignalMarker = Symbol('signal')` was used as a COMPUTED PROPERTY
//!     KEY (`[SignalMarker]: true`) to brand signal types. Rust has no symbol
//!     keys; the brand becomes the marker trait [`SignalMarker`] (implemented by
//!     the branded structs) — see [`SignalTimeStepOpts`] / [`SignalEntityGraphData`].
//!   * `abstract class SignalEntity<E,V> extends AbstractMovingEntity` does not
//!     map to `extends`; it becomes the [`SignalEntity`] trait, a sub-trait of
//!     [`MovingEntity`]. Concrete signal nodes compose a `MovingCore` and `impl`
//!     both traits.
//!   * `getGraphData()`/`getWithComputedProperties()` returned `null as any` and
//!     `getSerializableData()` returned `undefined` -> `Option<_>`/`None`.
//!   * `doTimeStep`/`runTimeStep` take an optional [`SignalTimeStepOpts`]; the
//!     `math.BigNumber` step size -> [`Decimal`].

#![allow(dead_code)]

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::r#abstract::interfaces::EntityGraphData;
use crate::des::r#abstract::r#abstract::TimeStepOpts;
use crate::des::shared::precision::Decimal;

/// `const SignalMarker = Symbol('signal')` — the `[SignalMarker]: true` brand on
/// signal data shapes. Modelled as a marker trait rather than a symbol key.
pub trait SignalMarker {}

/// `interface SignalTimeStepOpts extends TimeStepOpts { [SignalMarker]: true }`.
/// The symbol-keyed brand becomes the [`SignalMarker`] impl below.
#[derive(Clone, Copy, Debug, Default)]
pub struct SignalTimeStepOpts {
    pub base: TimeStepOpts,
}

impl SignalMarker for SignalTimeStepOpts {}

/// `interface SignalEntityGraphData extends EntityGraphData { [SignalMarker]: true }`.
#[derive(Clone, Debug, Default)]
pub struct SignalEntityGraphData {
    pub base: EntityGraphData,
}

impl SignalMarker for SignalEntityGraphData {}

/// `abstract class SignalEntity<E,V> extends AbstractMovingEntity<E,V>`.
///
/// Object-safe behavioural contract layered on top of [`MovingEntity`]. The
/// generic `<E,V>` were pervasive `any` and are erased.
pub trait SignalEntity: MovingEntity {
    /// `abstract runTimeStep(stepSize, opts?)` — every signal node implements the
    /// per-tick behaviour (no default).
    fn run_time_step_signal(&mut self, step_size: Decimal, opts: Option<SignalTimeStepOpts>);

    /// `doTimeStep(stepSize, opts?) { return this.runTimeStep(stepSize, opts); }`.
    fn do_time_step_signal(&mut self, step_size: Decimal, opts: Option<SignalTimeStepOpts>) {
        self.run_time_step_signal(step_size, opts);
    }

    /// `getGraphData(): SignalEntityGraphData { return null as any; }`.
    fn get_signal_graph_data(&self) -> Option<SignalEntityGraphData> {
        None
    }

    /// `getWithComputedProperties(): any { return null as any; }`.
    fn get_with_computed_properties_signal(&self) -> Option<SignalEntityGraphData> {
        None
    }

    /// `getSerializableData(): any { return undefined; }`.
    fn get_serializable_data_signal(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::{MovingCore, MovingValue};
    use crate::des::r#abstract::r#abstract::{Entity, EntityCore};

    /// Minimal concrete signal entity exercising the trait defaults.
    struct TestSignal {
        core: MovingCore,
        last_step: Decimal,
        ticks: u32,
    }

    impl Entity for TestSignal {
        fn core(&self) -> &EntityCore {
            &self.core.entity
        }
        fn core_mut(&mut self) -> &mut EntityCore {
            &mut self.core.entity
        }
        fn do_validation(&mut self) {}
        fn do_validation_before_run(&mut self) -> bool {
            true
        }
        fn get_graph_data(&self) -> EntityGraphData {
            EntityGraphData::default()
        }
        fn run_time_step(&mut self, step_size: Decimal) {
            self.run_time_step_signal(step_size, None);
        }
    }

    impl MovingEntity for TestSignal {
        fn moving_core(&self) -> &MovingCore {
            &self.core
        }
        fn moving_core_mut(&mut self) -> &mut MovingCore {
            &mut self.core
        }
        fn get_value(&self) -> MovingValue {
            MovingValue::default()
        }
        fn run_finish(&mut self) {}
    }

    impl SignalEntity for TestSignal {
        fn run_time_step_signal(&mut self, step_size: Decimal, _opts: Option<SignalTimeStepOpts>) {
            self.last_step = step_size;
            self.ticks += 1;
        }
    }

    #[test]
    fn do_time_step_delegates_to_run_time_step() {
        use crate::des::shared::precision::bgn;
        let mut s = TestSignal {
            core: MovingCore::new("sig".into()),
            last_step: Decimal::ZERO,
            ticks: 0,
        };
        s.do_time_step_signal(bgn(0.25), None);
        assert_eq!(s.ticks, 1);
        assert_eq!(s.last_step, bgn(0.25));
        assert!(s.get_signal_graph_data().is_none());
        assert!(s.get_serializable_data_signal().is_none());
    }
}
