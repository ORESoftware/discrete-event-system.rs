//! Canonical use path: `crate::des::signals::integral::*`
//!
//! Port of `src/des/signals/integral.ts` — a signal integrator node that
//! accumulates incoming sample values and broadcasts the running total.
//!
//! Conversion notes (file-specific):
//!   * DUPLICATE of [`Adder`](crate::des::signals::adder::Adder): identical
//!     dequeue-and-sum body. Kept as its own type for a faithful 1:1 move.
//!   * PORT NOTE: it is a RUNNING SUM, NOT a true integral — `stepSize` is unused
//!     (a real integral would multiply the sample by `dt`). Behaviour preserved.
//!   * `runningTotal: BigNumber` -> [`Decimal`]; `math.add` -> `+=`.
//!   * `queue.dequeueIterator()` -> repeated `dequeue()`; samples read via
//!     `MovingEntity::get_value().value` (see the `adder.rs` PORT NOTE).
//!   * `getValue()` returns `<unknown>undefined as any` -> [`MovingValue::default`]
//!     (all-`None`); `runFinish()` `throw` -> `panic!`.
//!   * The dead `IntegratorTimeStepOpts` + `marker` symbol -> documented no-op /
//!     dropped.

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
use crate::des::signals::signal_value::SignalValue;

/// `interface IntegratorTimeStepOpts extends TimeStepOpts {}` (empty brand).
#[derive(Clone, Copy, Debug, Default)]
pub struct IntegratorTimeStepOpts {
    pub base: TimeStepOpts,
}

/// `class Integrator<E,V> extends MultiDirectionalSignalEntity<E,V>`.
pub struct Integrator {
    pub core: MultiDirectionalSignalCore,
    /// `runningTotal = bgn(0)`.
    pub running_total: Decimal,
}

impl Integrator {
    pub fn new(id: String) -> Self {
        Integrator {
            core: MultiDirectionalSignalCore::new(id),
            running_total: bgn(0.0),
        }
    }

    fn broadcast(&self, sv: Rc<RefCell<dyn MovingEntity>>) {
        for conn in self.core.get_out_connections() {
            let target = conn.borrow().get_target();
            if let Some(target) = target {
                let accepted = target.borrow_mut().accept_item(sv.clone());
                if accepted {
                    target.borrow_mut().take_item(sv.clone());
                }
            }
        }
    }
}

impl Entity for Integrator {
    fn core(&self) -> &EntityCore {
        &self.core.moving.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.moving.entity
    }
    /// `doValidation() {}` (no-op in the TS source).
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

impl MovingEntity for Integrator {
    fn moving_core(&self) -> &MovingCore {
        &self.core.moving
    }
    fn moving_core_mut(&mut self) -> &mut MovingCore {
        &mut self.core.moving
    }
    /// `getValue() { return <unknown>undefined as any; }`.
    fn get_value(&self) -> MovingValue {
        MovingValue::default()
    }
    /// `runFinish(): void { throw new Error('not yet implemented.'); }`.
    fn run_finish(&mut self) {
        panic!("not yet implemented.");
    }
}

impl SignalEntity for Integrator {
    fn run_time_step_signal(&mut self, _step_size: Decimal, _opts: Option<SignalTimeStepOpts>) {
        while let Some((_id, d)) = self.core.queue.dequeue() {
            if let Some(v) = d.borrow().get_value().value {
                self.running_total += bgn(v);
            }
        }

        let sv: Rc<RefCell<dyn MovingEntity>> =
            Rc::new(RefCell::new(SignalValue::from_value(self.running_total)));
        self.broadcast(sv);
    }
}

impl MultiDirectionalSignalEntity for Integrator {
    fn md_core(&self) -> &MultiDirectionalSignalCore {
        &self.core
    }
    fn md_core_mut(&mut self) -> &mut MultiDirectionalSignalCore {
        &mut self.core
    }
    /// `acceptItem(m): boolean { return true; }`.
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        true
    }
    /// `takeItem(m): void { this.queue.enqueue(m); }`.
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        self.core.take_item(m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::signals::signal_value::{SignalValue, SignalValueArgs};

    #[test]
    fn accumulates_running_sum() {
        let mut integ = Integrator::new("integ".into());
        for x in [0.5, 0.5, 0.5, 0.5] {
            let sv: Rc<RefCell<dyn MovingEntity>> =
                Rc::new(RefCell::new(SignalValue::new(SignalValueArgs {
                    id: None,
                    val: Some(x),
                })));
            integ.take_item(sv);
        }
        integ.run_time_step_signal(bgn(0.1), None);
        assert_eq!(integ.running_total, bgn(2.0));

        // A second tick keeps accumulating onto the prior total.
        let sv: Rc<RefCell<dyn MovingEntity>> =
            Rc::new(RefCell::new(SignalValue::new(SignalValueArgs {
                id: None,
                val: Some(3.0),
            })));
        integ.take_item(sv);
        integ.run_time_step_signal(bgn(0.1), None);
        assert_eq!(integ.running_total, bgn(5.0));
    }

    #[test]
    fn get_value_is_empty() {
        let integ = Integrator::new("integ".into());
        let v = integ.get_value();
        assert!(v.value.is_none() && v.q.is_none() && v.id.is_none());
    }
}
