//! Canonical use path: `crate::des::signals::differential::*`
//!
//! Port of `src/des/signals/differential.ts` — a signal differentiator node that
//! emits successive-sample differences.
//!
//! Conversion notes (file-specific):
//!   * PORT NOTE: `previousValue = <unknown>null as { [marker]: SignalValue }`
//!     used a SYMBOL-KEYED wrapper object to stash the previous sample. That is a
//!     TS hack with no Rust analogue; here it is just `previous_value:
//!     Option<Decimal>` (the previous sample's value).
//!   * PORT NOTE: the TS faithfully has two quirks, preserved here:
//!       1. On the FIRST tick (no previous yet) it records the first sample and
//!          `break`s out of the drain loop (the rest of that tick's queue is
//!          left unprocessed).
//!       2. After that, `previousValue` is NEVER reassigned, so every emitted
//!          difference is `current - firstSample` rather than a true first
//!          difference. A real differentiator would update the previous each step.
//!   * `(v as any).getValue()` dynamic access -> the erased
//!     `MovingEntity::get_value().value` (see `adder.rs` PORT NOTE on the f64
//!     round-trip).
//!   * `queue.dequeue()` `[k,v]` + `IsVoid.check(k)` + `console.error` -> repeated
//!     `dequeue()` over the framework `LinkedQueue` (no void sentinel; the option
//!     is the emptiness signal).
//!   * `stepSize` is unused (a true derivative divides the diff by `dt`).
//!   * `acceptItem(m: SignalValue)` narrows the trait's `acceptItem(m:
//!     AbstractMovingEntity)` param in the TS; the Rust trait keeps the erased
//!     `dyn MovingEntity` signature.
//!   * `math.subtract` -> `-`; `getValue()`/`runFinish()` `throw` -> `panic!`.

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

/// `interface DifferentialTimeStepOpts extends TimeStepOpts {}` (empty).
#[derive(Clone, Copy, Debug, Default)]
pub struct DifferentialTimeStepOpts {
    pub base: TimeStepOpts,
}

/// `class Differentiator<E,V> extends MultiDirectionalSignalEntity<E,V>`.
pub struct Differentiator {
    pub core: MultiDirectionalSignalCore,
    /// `runningTotal = bgn(0)` (carried for parity; unused by the body).
    pub running_total: Decimal,
    /// `previousValue` — the previously seen sample value (see PORT NOTE).
    pub previous_value: Option<Decimal>,
}

impl Differentiator {
    pub fn new(id: String) -> Self {
        Differentiator {
            core: MultiDirectionalSignalCore::new(id),
            running_total: bgn(0.0),
            previous_value: None,
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

impl Entity for Differentiator {
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

impl MovingEntity for Differentiator {
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

impl SignalEntity for Differentiator {
    fn run_time_step_signal(&mut self, _step_size: Decimal, _opts: Option<SignalTimeStepOpts>) {
        while self.core.queue.size() > 0 {
            let (_k, v) = match self.core.queue.dequeue() {
                Some(pair) => pair,
                None => break,
            };

            let current = v.borrow().get_value().value;

            let prev_opt = self.previous_value;
            match prev_opt {
                None => {
                    // First sample: record it and break (faithful TS quirk #1).
                    self.previous_value = current.map(bgn);
                    break;
                }
                Some(prev) => {
                    if let Some(cur) = current {
                        // `previous_value` is intentionally NOT updated (quirk #2).
                        let diff = bgn(cur) - prev;
                        let sv: Rc<RefCell<dyn MovingEntity>> =
                            Rc::new(RefCell::new(SignalValue::from_value(diff)));
                        self.broadcast(sv);
                    }
                }
            }
        }
    }
}

impl MultiDirectionalSignalEntity for Differentiator {
    fn md_core(&self) -> &MultiDirectionalSignalCore {
        &self.core
    }
    fn md_core_mut(&mut self) -> &mut MultiDirectionalSignalCore {
        &mut self.core
    }
    /// `acceptItem(m): boolean { return true; }` (TS narrows the param to
    /// `SignalValue`; the trait keeps the erased `dyn MovingEntity`).
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

    fn sample(x: f64) -> Rc<RefCell<dyn MovingEntity>> {
        Rc::new(RefCell::new(SignalValue::new(SignalValueArgs {
            id: None,
            val: Some(x),
        })))
    }

    #[test]
    fn first_tick_records_previous_and_breaks() {
        let mut diff = Differentiator::new("diff".into());
        diff.take_item(sample(2.0));
        diff.take_item(sample(5.0));
        diff.run_time_step_signal(bgn(0.1), None);
        // First sample recorded; the rest of the queue is left unprocessed.
        assert_eq!(diff.previous_value, Some(bgn(2.0)));
        assert_eq!(diff.core.queue.size(), 1);
    }

    #[test]
    fn subsequent_ticks_diff_against_fixed_first_sample() {
        let mut diff = Differentiator::new("diff".into());
        // Tick 1: records first sample (3.0), breaks.
        diff.take_item(sample(3.0));
        diff.run_time_step_signal(bgn(0.1), None);
        assert_eq!(diff.previous_value, Some(bgn(3.0)));

        // Tick 2: 10 - 3 = 7, then previous stays 3 (faithful quirk #2).
        diff.take_item(sample(10.0));
        diff.run_time_step_signal(bgn(0.1), None);
        assert_eq!(diff.previous_value, Some(bgn(3.0)));
        assert!(diff.is_empty());
    }
}
