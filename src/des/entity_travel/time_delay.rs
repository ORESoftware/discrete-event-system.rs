//! Canonical use path: `crate::des::entity_travel::time_delay::*`
//!
//! Port of `src/des/entity-travel/time-delay.ts` — a travel / time-delay node,
//! currently a mostly-unimplemented stub.
//!
//! TS `class TimeDelayOrTravelEntity extends AbstractBidirectionalEntity
//! implements HasComputedProperties, HasInternalQueue`. Modelled as a struct
//! embedding [`BidirectionalCore`] plus an injected `RandomVariable`.
//!
//! PORT NOTES:
//!   * `doValidation()` and `takeItem()` are `throw new Error("Method not
//!     implemented.")` stubs -> `panic!` (faithful to the thrown invariant).
//!   * `rv: RandomVariable` carries an injected `RandomSource` (no `Math.random`);
//!     it is stored as `Box<dyn RandomVariable>`.
//!   * `getWithComputedProperties(): this` (a shallow `Object.assign({}, this)`
//!     typed as `Self`) becomes an `EntityGraphData` payload, not `Self`.
//!   * `getGraphData()` returns the hardcoded `{ processedCount: 3 }` stub.
//!   * The `<S, T>` generics were `any` and are erased.
//!
//! `TimeDelayEntityGraphData` is re-used by the `entity_decision::*` and
//! `entity_routing::entity_splitter` modules (they import it from here).

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_queue::queue::{build_in_conn, build_out_conn};
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasInternalQueue, HasManyInputConnections, HasManyOutputConnections,
    HasOutput,
};
use crate::des::r#abstract::r#abstract::{
    BidirectionalCore, Entity, EntityCore, EntityConnection, TimeStepOpts,
};
use crate::des::random_variables::rv::RandomVariable;
use crate::des::shared::precision::Decimal;

/// `interface TimeDelayEntityGraphData extends EntityGraphData { processedCount }`.
#[derive(Clone, Debug, Default)]
pub struct TimeDelayEntityGraphData {
    pub processed_count: i64,
}

/// `interface DelayTimeStepOpts extends TimeStepOpts` (currently empty).
///
/// PORT NOTE: TS added no fields; we alias the framework's [`TimeStepOpts`].
pub type DelayTimeStepOpts = TimeStepOpts;

/// `class TimeDelayOrTravelEntity`.
pub struct TimeDelayOrTravelEntity {
    pub bi: BidirectionalCore,
    pub queue: VecDeque<Rc<RefCell<dyn MovingEntity>>>,
    pub rv: Box<dyn RandomVariable>,
    /// `-1` sentinel = unbounded.
    pub max_queue_size: i64,
}

impl TimeDelayOrTravelEntity {
    /// `new(id, { rv })`.
    pub fn new(id: String, rv: Box<dyn RandomVariable>) -> Self {
        TimeDelayOrTravelEntity {
            bi: BidirectionalCore::new(id),
            queue: VecDeque::new(),
            rv,
            max_queue_size: -1,
        }
    }

    /// `getGraphData(): TimeDelayEntityGraphData`.
    pub fn get_graph_data_typed(&self) -> TimeDelayEntityGraphData {
        TimeDelayEntityGraphData { processed_count: 3 }
    }
}

impl Entity for TimeDelayOrTravelEntity {
    fn core(&self) -> &EntityCore {
        &self.bi.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.bi.entity
    }
    fn do_validation(&mut self) {
        eprintln!(
            "[time-delay:{}] doValidation() is not implemented for TimeDelayOrTravelEntity.",
            self.bi.entity.id
        );
        panic!("Method not implemented.");
    }
    fn do_validation_before_run(&mut self) -> bool {
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default().with("processedCount", 3.0)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default().with("queue.size", self.queue.len() as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        // TS body is empty (noop stub).
    }
}

impl HasInternalQueue for TimeDelayOrTravelEntity {
    fn max_queue_size(&self) -> usize {
        if self.max_queue_size < 0 {
            usize::MAX
        } else {
            self.max_queue_size as usize
        }
    }
    fn is_full(&self) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        self.queue.len() < 1
    }
}

impl HasInput for TimeDelayOrTravelEntity {
    fn id(&self) -> String {
        self.bi.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        self.bi.accept_item()
    }
    fn take_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) {
        eprintln!(
            "[time-delay:{}] takeItem() is not implemented — entity cannot enter this travel/delay node.",
            self.bi.entity.id
        );
        panic!("Method not implemented.");
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_sources(&mut self) {
        self.bi.notify_sources();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
    fn add_in_connection(
        &mut self,
        _source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        let conn = build_in_conn();
        self.bi.add_in_connection(conn.clone());
        Some(conn)
    }
}

impl HasManyInputConnections for TimeDelayOrTravelEntity {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_in_connections()
    }
}

impl HasOutput for TimeDelayOrTravelEntity {
    fn id(&self) -> String {
        self.bi.entity.id.clone()
    }
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        let conn = build_out_conn(target);
        self.bi.add_out_connection(conn.clone());
        Some(conn)
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_targets(&mut self) {
        self.bi.notify_targets();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl HasManyOutputConnections for TimeDelayOrTravelEntity {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_out_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::BasicMovingEntity;
    use crate::des::random_variables::rv::BernoulliRandomVariable;
    use crate::des::shared::capabilities::SeededRandom;
    use crate::des::shared::precision::bgn;

    fn rv() -> Box<dyn RandomVariable> {
        Box::new(BernoulliRandomVariable::new(Box::new(SeededRandom::new(1))))
    }

    #[test]
    fn constructs_and_runs_noop_step() {
        let mut td = TimeDelayOrTravelEntity::new("td1".to_string(), rv());
        assert!(td.is_empty());
        // runTimeStep is a noop stub — must not panic.
        td.run_time_step(bgn(0.1));
        assert_eq!(td.get_graph_data_typed().processed_count, 3);
    }

    #[test]
    #[should_panic(expected = "Method not implemented.")]
    fn take_item_is_unimplemented() {
        let mut td = TimeDelayOrTravelEntity::new("td2".to_string(), rv());
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(BasicMovingEntity::new()));
        td.take_item(m);
    }
}
