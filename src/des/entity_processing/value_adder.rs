//! Canonical use path: `crate::des::entity_processing::value_adder::*`
//!
//! Port of `src/des/entity-processing/value-adder.ts` — a processor that pops two
//! queued tokens and emits a new token carrying their numeric sum.
//!
//! TS `class EntityNumericProcessor extends QueueEntity implements
//! HasEntityValidation`. COMPOSES a [`QueueEntity`] (`base`); the FIFO is
//! `base.queue`.
//!
//! PORT NOTES:
//!   * BRANDING: the `processorSymbol` / `isProcessor` symbol brand becomes the
//!     [`ProcessorTag`] marker trait (defined once in `processing`, reused here).
//!   * `doesFanOut = new DoesFanOut({entity: this})` cannot be built in the
//!     constructor (a struct cannot hand an `Rc<RefCell<Self>>` to the helper),
//!     so the fan-out is INLINED in `runTimeStep` — it replicates
//!     `composers::DoesFanOut::do_fan_out` exactly (offer the token to each
//!     out-connection in turn; the first acceptor wins).
//!   * `k.value + p.value` is plain `number` (NOT BigNumber) addition -> `f64`,
//!     read via the token's `get_value().q`.
//!   * `throw new Error(..)` / `throw makeError(..)` invariants -> `panic!`;
//!     `LinkedQueue.dequeue()/peek()` + `IsVoid.check` -> `VecDeque` `pop_front` /
//!     `front` with `Option`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::{BasicQuantityMovingEntity, MovingEntity};
use crate::des::entity_processing::processing::ProcessorTag;
use crate::des::entity_queue::queue::{QueueEntity, QueueEntityGraphData, QueueOpts};
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasEntityValidation, HasInput, HasInternalQueue, HasManyInputConnections,
    HasManyOutputConnections, HasOutput,
};
use crate::des::r#abstract::r#abstract::{Entity, EntityCore, EntityConnection};
use crate::des::shared::precision::Decimal;

/// `interface GraphData extends QueueEntityGraphData`.
#[derive(Clone, Debug, Default)]
pub struct GraphData {
    pub base: QueueEntityGraphData,
}

/// `class EntityNumericProcessor`.
pub struct EntityNumericProcessor {
    pub base: QueueEntity,
    pub processed_count: i64,
}

impl ProcessorTag for EntityNumericProcessor {}

impl EntityNumericProcessor {
    /// `new(id)` — the TS ctor forwarded `{ xx: true }` to the `QueueEntity` base.
    pub fn new(id: String) -> Self {
        EntityNumericProcessor {
            base: QueueEntity::new(id, QueueOpts { xx: Some(true) }),
            processed_count: 0,
        }
    }

    pub fn get_queue_size(&self) -> usize {
        self.base.queue.len()
    }

    /// `doAudit()` -> `{ totalSize }`.
    pub fn do_audit(&self) -> usize {
        self.base.queue.len()
    }
}

impl HasEntityValidation for EntityNumericProcessor {
    fn validate(&self) -> bool {
        true
    }
}

impl Entity for EntityNumericProcessor {
    fn core(&self) -> &EntityCore {
        &self.base.bi.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.base.bi.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        false
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("processedCount", self.processed_count as f64)
            .with("timeStepCount", self.base.bi.entity.time_step_count as f64)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default().with("queue.size", self.base.queue.len() as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        self.base.bi.entity.time_step_count += 1;

        let gd = self.get_graph_data();
        self.send_update_to_subs("GRAPH_DATA:PROCESSING", &gd);

        if self.base.queue.len() < 2 {
            return;
        }

        if self.base.queue.len() > 4 {
            panic!("Queue length should never be");
        }

        let k = match self.base.queue.pop_front() {
            Some(x) => x,
            None => panic!("void item in front of queue."),
        };

        let p = match self.base.queue.front() {
            Some(x) => x.clone(),
            None => panic!("queue item should not be void:"),
        };

        // TS read `.value` directly off the BasicQuantityMovingEntity; via the
        // object-safe trait that is `get_value().q`.
        let k_val = k
            .borrow()
            .get_value()
            .q
            .expect("value-adder: dequeued token has no numeric quantity");
        let p_val = p
            .borrow()
            .get_value()
            .q
            .expect("value-adder: peeked token has no numeric quantity");
        let sum = k_val + p_val;

        let mut e = BasicQuantityMovingEntity::new(sum);
        e.init();
        let ame: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(e));

        // Inlined DoesFanOut: first accepting out-connection wins.
        let connections = self.base.bi.get_out_connections();
        let mut accepted = false;
        for conn in &connections {
            let target = conn.borrow().get_target();
            let target = match target {
                Some(t) => t,
                None => {
                    eprintln!("warning: could not find target.");
                    continue;
                }
            };
            accepted = target.borrow_mut().accept_item(ame.clone());
            if accepted {
                target.borrow_mut().take_item(ame.clone());
                break;
            }
        }

        if !accepted {
            panic!("moving entity was not accepted but it must be accepted:");
        }

        self.processed_count += 1;
    }
}

impl HasInternalQueue for EntityNumericProcessor {
    fn max_queue_size(&self) -> usize {
        HasInternalQueue::max_queue_size(&self.base)
    }
    fn is_full(&self) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        self.base.queue.len() < 1
    }
}

impl HasInput for EntityNumericProcessor {
    fn id(&self) -> String {
        self.base.bi.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        true
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        let id = self.base.bi.entity.id.clone();
        {
            let mut mb = m.borrow_mut();
            mb.moving_core_mut().stations_visited_count += 1;
            mb.add_visited_station(&id);
        }
        self.base.queue.push_back(m);
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_sources(&mut self) {
        self.base.bi.notify_sources();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
    fn add_in_connection(
        &mut self,
        source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        HasInput::add_in_connection(&mut self.base, source)
    }
}

impl HasManyInputConnections for EntityNumericProcessor {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.base.bi.get_in_connections()
    }
}

impl HasOutput for EntityNumericProcessor {
    fn id(&self) -> String {
        self.base.bi.entity.id.clone()
    }
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        HasOutput::add_out_connection(&mut self.base, target)
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_targets(&mut self) {
        self.base.bi.notify_targets();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl HasManyOutputConnections for EntityNumericProcessor {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.base.bi.get_out_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::BasicQuantityMovingEntity;
    use crate::des::shared::precision::bgn;

    fn q(value: f64) -> Rc<RefCell<dyn MovingEntity>> {
        Rc::new(RefCell::new(BasicQuantityMovingEntity::new(value)))
    }

    #[test]
    fn fewer_than_two_items_is_noop() {
        let mut p = EntityNumericProcessor::new("va1".to_string());
        p.take_item(q(1.0));
        p.run_time_step(bgn(0.1)); // queue len 1 -> early return, no panic
        assert_eq!(p.base.bi.entity.time_step_count, 1);
        assert_eq!(p.get_queue_size(), 1);
    }

    #[test]
    #[should_panic(expected = "moving entity was not accepted")]
    fn sum_with_no_downstream_panics() {
        let mut p = EntityNumericProcessor::new("va2".to_string());
        p.take_item(q(2.0));
        p.take_item(q(3.0));
        // two items, sum produced, but no out-connection accepts -> invariant panic.
        p.run_time_step(bgn(0.1));
    }

    #[test]
    fn take_item_records_visit() {
        let mut p = EntityNumericProcessor::new("va3".to_string());
        let m = q(4.0);
        p.take_item(m.clone());
        assert_eq!(m.borrow().moving_core().stations_visited_count, 1);
        assert!(p.validate());
    }
}
