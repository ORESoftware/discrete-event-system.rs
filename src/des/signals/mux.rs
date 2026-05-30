//! Canonical use path: `crate::des::signals::mux::*`
//!
//! Port of `src/des/signals/mux.ts` — a signal multiplexer node that is
//! currently an unimplemented stub.
//!
//! Conversion notes (file-specific):
//!   * `runTimeStep` is empty, `acceptItem` returns `false`, `takeItem` is a
//!     no-op -> faithful no-ops.
//!   * `getValue()` returns `<unknown>undefined as V` -> [`MovingValue::default`]
//!     (all-`None`).
//!   * `notifySources()`/`notifyTargets()`/`runFinish()` `throw` -> `panic!`
//!     (these override the base trait's no-op `notify_*` defaults).
//!   * `doSetupAfterInputConn()`/`doSetupAfterOutputConn()` return `true` here
//!     (overriding the base `false`).
//!   * `runningTotal: BigNumber` is unused here -> [`Decimal`] field kept for parity.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::{MovingCore, MovingEntity, MovingValue};
use crate::des::r#abstract::interfaces::EntityGraphData;
use crate::des::r#abstract::r#abstract::{Entity, EntityCore, TimeStepOpts};
use crate::des::shared::precision::{bgn, Decimal};

use crate::des::signals::multi_directional_signal_entity::{
    MultiDirectionalSignalCore, MultiDirectionalSignalEntity,
};
use crate::des::signals::r#abstract::{SignalEntity, SignalTimeStepOpts};

/// `interface MultiplexerTimeStepOpts extends TimeStepOpts {}` (empty).
#[derive(Clone, Copy, Debug, Default)]
pub struct MultiplexerTimeStepOpts {
    pub base: TimeStepOpts,
}

/// `class Multiplexer<E,V> extends MultiDirectionalSignalEntity<E,V>`.
pub struct Multiplexer {
    pub core: MultiDirectionalSignalCore,
    /// `runningTotal = bgn(0)` (unused by the stub body).
    pub running_total: Decimal,
}

impl Multiplexer {
    pub fn new(id: String) -> Self {
        Multiplexer {
            core: MultiDirectionalSignalCore::new(id),
            running_total: bgn(0.0),
        }
    }
}

impl Entity for Multiplexer {
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

impl MovingEntity for Multiplexer {
    fn moving_core(&self) -> &MovingCore {
        &self.core.moving
    }
    fn moving_core_mut(&mut self) -> &mut MovingCore {
        &mut self.core.moving
    }
    /// `getValue(): V { return <unknown>undefined as V; }`.
    fn get_value(&self) -> MovingValue {
        MovingValue::default()
    }
    /// `runFinish(): void { throw new Error('not implemented.'); }`.
    fn run_finish(&mut self) {
        panic!("not implemented.");
    }
}

impl SignalEntity for Multiplexer {
    /// `runTimeStep(...) {}` — empty in the TS source.
    fn run_time_step_signal(&mut self, _step_size: Decimal, _opts: Option<SignalTimeStepOpts>) {}
}

impl MultiDirectionalSignalEntity for Multiplexer {
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
    /// `takeItem(m): void {}` — no-op.
    fn take_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) {}

    /// `notifySources(): void { throw new Error("Method not implemented."); }`.
    fn notify_sources(&mut self) {
        panic!("Method not implemented.");
    }
    /// `notifyTargets(): void { throw new Error("Method not implemented."); }`.
    fn notify_targets(&mut self) {
        panic!("Method not implemented.");
    }
    /// `doSetupAfterInputConn(): boolean { return true; }`.
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    /// `doSetupAfterOutputConn(): boolean { return true; }`.
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::signals::signal_value::SignalValue;

    #[test]
    fn stub_behaviour_is_faithful() {
        let mut mux = Multiplexer::new("mux".into());
        let sv: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(SignalValue::unity()));
        assert!(!mux.accept_item(sv.clone()));
        mux.take_item(sv);
        assert!(mux.is_empty());
        assert!(mux.do_setup_after_input_conn());
        assert!(mux.do_setup_after_output_conn());
        let v = mux.get_value();
        assert!(v.value.is_none() && v.q.is_none());
    }

    #[test]
    #[should_panic(expected = "Method not implemented.")]
    fn notify_sources_panics() {
        let mut mux = Multiplexer::new("mux".into());
        mux.notify_sources();
    }
}
