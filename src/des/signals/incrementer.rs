//! Canonical use path: `crate::des::signals::incrementer::*`
//!
//! Port of `src/des/signals/incrementer.ts` — a signal incrementer node that is
//! currently an unimplemented stub.
//!
//! Conversion notes (file-specific):
//!   * PORT NOTE: the ctor called `super(null as any)` (a null id). A null id has
//!     no Rust analogue, so [`SignalIncrementor::new`] generates a real short
//!     uuid instead.
//!   * `runTimeStep` is empty, `acceptItem` returns `false`, `takeItem` is a
//!     no-op — a not-yet-implemented node, ported as faithful no-ops.
//!   * `getValue()`/`runFinish()` `throw` -> `panic!`.
//!   * `runningTotal: BigNumber` -> [`Decimal`]; `queue: LinkedQueue` lives in the
//!     composed [`MultiDirectionalSignalCore`].

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::{MovingCore, MovingEntity, MovingValue};
use crate::des::general::general::get_short_uuid;
use crate::des::r#abstract::interfaces::EntityGraphData;
use crate::des::r#abstract::r#abstract::{Entity, EntityCore, TimeStepOpts};
use crate::des::shared::precision::{bgn, Decimal};

use crate::des::signals::multi_directional_signal_entity::{
    MultiDirectionalSignalCore, MultiDirectionalSignalEntity,
};
use crate::des::signals::r#abstract::{SignalEntity, SignalTimeStepOpts};

/// `interface IncrementorTimeStepOpts extends TimeStepOpts {}` (empty).
#[derive(Clone, Copy, Debug, Default)]
pub struct IncrementorTimeStepOpts {
    pub base: TimeStepOpts,
}

/// `class SignalIncrementor<E,V> extends MultiDirectionalSignalEntity<E,V>`.
pub struct SignalIncrementor {
    pub core: MultiDirectionalSignalCore,
    /// `runningTotal = bgn(0)`.
    pub running_total: Decimal,
}

impl Default for SignalIncrementor {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalIncrementor {
    /// `constructor() { super(null as any); }` — see PORT NOTE (real id minted).
    pub fn new() -> Self {
        SignalIncrementor {
            core: MultiDirectionalSignalCore::new(get_short_uuid()),
            running_total: bgn(0.0),
        }
    }
}

impl Entity for SignalIncrementor {
    fn core(&self) -> &EntityCore {
        &self.core.moving.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.moving.entity
    }
    /// `doValidation() {}` (no-op).
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

impl MovingEntity for SignalIncrementor {
    fn moving_core(&self) -> &MovingCore {
        &self.core.moving
    }
    fn moving_core_mut(&mut self) -> &mut MovingCore {
        &mut self.core.moving
    }
    /// `getValue(): V { throw new Error("Method not implemented."); }`.
    fn get_value(&self) -> MovingValue {
        panic!("Method not implemented.");
    }
    /// `runFinish(): void { throw new Error('not implemented.'); }`.
    fn run_finish(&mut self) {
        panic!("not implemented.");
    }
}

impl SignalEntity for SignalIncrementor {
    /// `runTimeStep(...) {}` — empty in the TS source (not-yet-implemented).
    fn run_time_step_signal(&mut self, _step_size: Decimal, _opts: Option<SignalTimeStepOpts>) {}
}

impl MultiDirectionalSignalEntity for SignalIncrementor {
    fn md_core(&self) -> &MultiDirectionalSignalCore {
        &self.core
    }
    fn md_core_mut(&mut self) -> &mut MultiDirectionalSignalCore {
        &mut self.core
    }
    /// `acceptItem(m): boolean { return false; }`.
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        false
    }
    /// `takeItem(m): void {}` — no-op in the TS source.
    fn take_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::signals::signal_value::SignalValue;

    #[test]
    fn stub_behaviour_is_faithful() {
        let mut inc = SignalIncrementor::new();
        assert_eq!(inc.id().len(), 10); // generated short uuid
        let sv: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(SignalValue::unity()));
        // acceptItem returns false; takeItem is a no-op (nothing enqueued).
        assert!(!inc.accept_item(sv.clone()));
        inc.take_item(sv);
        assert!(inc.is_empty());
        // empty runTimeStep leaves the running total untouched.
        inc.run_time_step_signal(bgn(0.1), None);
        assert_eq!(inc.running_total, bgn(0.0));
    }
}
