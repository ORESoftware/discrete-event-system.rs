//! Canonical use path: `crate::des::entity_decision::decision::*`
//!
//! Port of `src/des/entity-decision/decision.ts` — the base decision / branching
//! node (mostly a stub here).
//!
//! TS `class DecisionEntity extends AbstractBidirectionalEntity implements
//! HasComputedProperties, HasInternalQueue`. Modelled as a struct embedding
//! [`BidirectionalCore`] plus a `VecDeque` of moving entities.
//!
//! PORT NOTES:
//!   * `getWithComputedProperties()` was `throw new Error("Method not
//!     implemented.")` -> `panic!` (here via the overridden `Entity` default).
//!   * `runTimeStep` is a noop stub; `getGraphData()` is the hardcoded
//!     `{ processedCount: 3 }` (reusing [`TimeDelayEntityGraphData`]).
//!   * `DecisionEntityGraph` (an empty marker) is defined ONCE here and reused by
//!     `binary_decision` / `probability_decision` (it was duplicated across all
//!     three TS files).
//!   * `opts: { xx: boolean }` placeholder -> a small config struct.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_queue::queue::{build_in_conn, build_out_conn};
use crate::des::entity_travel::time_delay::TimeDelayEntityGraphData;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasInternalQueue, HasManyInputConnections, HasManyOutputConnections,
    HasOutput,
};
use crate::des::r#abstract::r#abstract::{BidirectionalCore, Entity, EntityConnection, EntityCore};
use crate::des::shared::precision::Decimal;

/// `interface DecisionEntityGraph` — empty marker, shared across the decision
/// modules.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecisionEntityGraph;

/// `opts: { xx: boolean }` placeholder.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecisionOpts {
    pub xx: bool,
}

/// `class DecisionEntity`.
pub struct DecisionEntity {
    pub bi: BidirectionalCore,
    pub queue: VecDeque<Rc<RefCell<dyn MovingEntity>>>,
    pub max_queue_size: i64,
    pub opts: DecisionOpts,
}

impl DecisionEntity {
    pub fn new(id: String, opts: DecisionOpts) -> Self {
        DecisionEntity {
            bi: BidirectionalCore::new(id),
            queue: VecDeque::new(),
            max_queue_size: -1,
            opts,
        }
    }

    pub fn get_graph_data_typed(&self) -> TimeDelayEntityGraphData {
        TimeDelayEntityGraphData { processed_count: 3 }
    }
}

impl Entity for DecisionEntity {
    fn core(&self) -> &EntityCore {
        &self.bi.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.bi.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default().with("processedCount", 3.0)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        // TS: `throw new Error("Method not implemented.")`.
        panic!("Method not implemented.");
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        // TS body is empty (noop stub).
    }
}

impl HasInternalQueue for DecisionEntity {
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

impl HasInput for DecisionEntity {
    fn id(&self) -> String {
        self.bi.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        self.bi.accept_item()
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        self.queue.push_back(m);
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

impl HasManyInputConnections for DecisionEntity {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_in_connections()
    }
}

impl HasOutput for DecisionEntity {
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

impl HasManyOutputConnections for DecisionEntity {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_out_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::BasicMovingEntity;
    use crate::des::shared::precision::bgn;

    #[test]
    fn noop_step_and_take_item() {
        let mut d = DecisionEntity::new("dec1".to_string(), DecisionOpts::default());
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(BasicMovingEntity::new()));
        d.take_item(m);
        assert!(!d.is_empty());
        d.run_time_step(bgn(0.1)); // noop
        assert_eq!(d.get_graph_data_typed().processed_count, 3);
    }

    #[test]
    #[should_panic(expected = "Method not implemented.")]
    fn computed_properties_unimplemented() {
        let d = DecisionEntity::new("dec2".to_string(), DecisionOpts::default());
        let _ = d.get_with_computed_properties();
    }
}
