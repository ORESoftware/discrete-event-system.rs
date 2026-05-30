//! Canonical use path: `crate::des::signals::adder::*`
//!
//! Port of `src/des/signals/adder.ts` — a signal node that sums incoming sample
//! values into a running total and broadcasts that total downstream every tick.
//!
//! Conversion notes (file-specific):
//!   * `runningTotal: BigNumber` accumulator -> [`Decimal`]; `math.add` -> `+=`.
//!   * `queue.dequeueIterator()` yielding each enqueued sample -> drain the
//!     framework [`LinkedQueue`](crate::des::shared::linked_queue) by repeated
//!     `dequeue()`.
//!   * PORT NOTE: the TS read `d.getValue()` (a `BigNumber`); here the dequeued
//!     token is an erased `dyn MovingEntity`, so the sample is read back via
//!     `MovingEntity::get_value().value` (`f64`) and re-wrapped with `bgn` —
//!     exact for the terminating decimals signals carry.
//!   * PORT NOTE: `new SignalValue({val: runningTotal})` would (in TS) drop the
//!     value, because `math.isNumber(BigNumber)` is `false`; we use
//!     [`SignalValue::from_value`] so the broadcast actually carries the total.
//!   * DUPLICATE LOGIC: identical to `Integrator` (`integral.rs`). The dead
//!     `IntegratorTimeStepOpts` (misnamed copy) + `marker` symbol are kept only
//!     as a documented no-op type / dropped, respectively.
//!   * `getValue()`/`doValidation()`/`runFinish()` `throw` -> `panic!`.

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

/// `interface IntegratorTimeStepOpts extends TimeStepOpts {}` — a misnamed,
/// empty copy from `integral.ts`. Kept verbatim; the brand carries no fields.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntegratorTimeStepOpts {
    pub base: TimeStepOpts,
}

/// `class Adder<E,V> extends MultiDirectionalSignalEntity<E,V>`.
pub struct Adder {
    pub core: MultiDirectionalSignalCore,
    /// `runningTotal = bgn(0)`.
    pub running_total: Decimal,
}

impl Adder {
    pub fn new(id: String) -> Self {
        Adder {
            core: MultiDirectionalSignalCore::new(id),
            running_total: bgn(0.0),
        }
    }

    /// Offer `sv` to every outbound target, handing it off to each that accepts
    /// (`for (const v of connectionsOut) { if (v.target.acceptItem(sv)) v.target.takeItem(sv); }`).
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

impl Entity for Adder {
    fn core(&self) -> &EntityCore {
        &self.core.moving.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.moving.entity
    }
    /// `doValidation(): void { throw new Error("Method not implemented."); }`.
    fn do_validation(&mut self) {
        panic!("Method not implemented.");
    }
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

impl MovingEntity for Adder {
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
    /// `runFinish(): void { throw new Error('not yet implemented.'); }`.
    fn run_finish(&mut self) {
        panic!("not yet implemented.");
    }
}

impl SignalEntity for Adder {
    fn run_time_step_signal(&mut self, _step_size: Decimal, _opts: Option<SignalTimeStepOpts>) {
        // Drain the queue, summing each carried sample into the running total.
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

impl MultiDirectionalSignalEntity for Adder {
    fn md_core(&self) -> &MultiDirectionalSignalCore {
        &self.core
    }
    fn md_core_mut(&mut self) -> &mut MultiDirectionalSignalCore {
        &mut self.core
    }
    /// `acceptItem(m): boolean { return true; }` (TODO: reject when full).
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
    use crate::des::r#abstract::interfaces::{HasInput, HasOutput};
    use crate::des::r#abstract::r#abstract::EntityConnection;
    use crate::des::shared::precision::to_f64;
    use crate::des::signals::signal_value::{SignalValue, SignalValueArgs};

    #[test]
    fn sums_enqueued_samples_into_running_total() {
        let mut adder = Adder::new("adder".into());
        for x in [1.0, 2.5, 4.0] {
            let sv: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(SignalValue::new(
                SignalValueArgs {
                    id: None,
                    val: Some(x),
                },
            )));
            adder.take_item(sv);
        }
        adder.run_time_step_signal(bgn(0.1), None);
        assert_eq!(adder.running_total, bgn(7.5));
        // queue drained
        assert!(adder.is_empty());
    }

    /// A downstream node that records every token it is handed (acts as both the
    /// connection's `HasInput` target and `HasOutput` source for wiring).
    struct Collector {
        id: String,
        received: usize,
    }
    impl HasInput for Collector {
        fn id(&self) -> String {
            self.id.clone()
        }
        fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
            true
        }
        fn take_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) {
            self.received += 1;
        }
        fn do_setup_after_input_conn(&mut self) -> bool {
            true
        }
        fn notify_sources(&mut self) {}
        fn do_setup_after_output_conn(&mut self) -> bool {
            true
        }
        fn add_in_connection(
            &mut self,
            _source: Rc<RefCell<dyn crate::des::r#abstract::interfaces::HasManyOutputConnections>>,
        ) -> Option<Rc<RefCell<EntityConnection>>> {
            None
        }
    }
    impl HasOutput for Collector {
        fn id(&self) -> String {
            self.id.clone()
        }
        fn add_out_connection(
            &mut self,
            _target: Rc<RefCell<dyn HasInput>>,
        ) -> Option<Rc<RefCell<EntityConnection>>> {
            None
        }
        fn do_setup_after_input_conn(&mut self) -> bool {
            true
        }
        fn notify_targets(&mut self) {}
        fn do_setup_after_output_conn(&mut self) -> bool {
            true
        }
    }

    #[test]
    fn broadcasts_total_to_connected_targets() {
        let mut adder = Adder::new("adder".into());

        let collector = Rc::new(RefCell::new(Collector {
            id: "sink".into(),
            received: 0,
        }));
        // Wire an outbound connection: source = the adder (as HasOutput, via a
        // standalone Collector here for the Weak), target = the collector.
        let src: Rc<RefCell<dyn HasOutput>> = collector.clone();
        let tgt: Rc<RefCell<dyn HasInput>> = collector.clone();
        let conn = EntityConnection::new(Rc::downgrade(&src), tgt);
        adder.add_out_connection_built(Rc::new(RefCell::new(conn)));

        let sv: Rc<RefCell<dyn MovingEntity>> =
            Rc::new(RefCell::new(SignalValue::new(SignalValueArgs {
                id: None,
                val: Some(3.0),
            })));
        adder.take_item(sv);
        adder.run_time_step_signal(bgn(0.1), None);

        assert_eq!(adder.running_total, bgn(3.0));
        assert_eq!(to_f64(adder.running_total), 3.0);
        assert_eq!(collector.borrow().received, 1);
    }
}
