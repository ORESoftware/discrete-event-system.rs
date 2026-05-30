//! Canonical use path: `crate::des::entity_decision::binary_decision::*`
//!
//! Port of `src/des/entity-decision/binary-decision.ts` — a two-way decision
//! node (routing logic currently a stub).
//!
//! TS `class BinaryDecisionEntity extends AbstractBidirectionalEntity implements
//! HasComputedProperties, HasInternalQueue`.
//!
//! PORT NOTES:
//!   * the constructor's `rv: RandomVariable` was never stored/used -> dropped.
//!   * `doValidation()` enforces EXACTLY 2 out-connections, else `throw` -> `panic!`.
//!   * `doValidationBeforeRun()` wrapped `doValidation()` in a try/catch. Rust
//!     does not use exceptions for control flow, so it instead performs the same
//!     non-panicking check directly (`connections_out.len() == 2`) — the
//!     migration header's recommended `Result`-style mapping.
//!   * `getWithComputedProperties()` was `throw new Error("Method not
//!     implemented.")` -> `panic!`; `runTimeStep` is a noop stub.
//!   * reuses [`DecisionEntityGraph`] and [`TimeDelayEntityGraphData`].

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::des::entity_decision::decision::DecisionEntityGraph;
use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_queue::queue::{build_in_conn, build_out_conn};
use crate::des::entity_travel::time_delay::TimeDelayEntityGraphData;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasInternalQueue, HasManyInputConnections, HasManyOutputConnections,
    HasOutput,
};
use crate::des::r#abstract::r#abstract::{
    BidirectionalCore, Entity, EntityCore, EntityConnection,
};
use crate::des::shared::precision::Decimal;

/// `opts: { xx: boolean }` placeholder.
#[derive(Clone, Copy, Debug, Default)]
pub struct BinaryDecisionOpts {
    pub xx: bool,
}

/// `class BinaryDecisionEntity`.
pub struct BinaryDecisionEntity {
    pub bi: BidirectionalCore,
    pub queue: VecDeque<Rc<RefCell<dyn MovingEntity>>>,
    pub max_queue_size: i64,
    pub opts: BinaryDecisionOpts,
}

impl BinaryDecisionEntity {
    /// `new(id, opts)` — the TS `rv` param was unused and is dropped.
    pub fn new(id: String, opts: BinaryDecisionOpts) -> Self {
        BinaryDecisionEntity {
            bi: BidirectionalCore::new(id),
            queue: VecDeque::new(),
            max_queue_size: -1,
            opts,
        }
    }

    pub fn get_graph_data_typed(&self) -> TimeDelayEntityGraphData {
        TimeDelayEntityGraphData { processed_count: 3 }
    }

    /// Marker view (empty in the TS).
    pub fn decision_graph(&self) -> DecisionEntityGraph {
        DecisionEntityGraph
    }
}

impl Entity for BinaryDecisionEntity {
    fn core(&self) -> &EntityCore {
        &self.bi.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.bi.entity
    }
    fn do_validation(&mut self) {
        if self.bi.connections_out.len() != 2 {
            eprintln!(
                "[binary-decision:{}] expected exactly 2 out-connections, found {}.",
                self.bi.entity.id,
                self.bi.connections_out.len()
            );
            panic!("Binary decision node needs two connections out.");
        }
    }
    fn do_validation_before_run(&mut self) -> bool {
        // PORT NOTE: TS caught the `doValidation` throw; here we check directly.
        if self.bi.connections_out.len() != 2 {
            eprintln!(
                "[binary-decision:{}] pre-run validation failed: needs exactly 2 out-connections (found {}).",
                self.bi.entity.id,
                self.bi.connections_out.len()
            );
            return false;
        }
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default().with("processedCount", 3.0)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        panic!("Method not implemented.");
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        // TS body is empty (noop stub).
    }
}

impl HasInternalQueue for BinaryDecisionEntity {
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

impl HasInput for BinaryDecisionEntity {
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

impl HasManyInputConnections for BinaryDecisionEntity {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_in_connections()
    }
}

impl HasOutput for BinaryDecisionEntity {
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

impl HasManyOutputConnections for BinaryDecisionEntity {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_out_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::precision::bgn;

    #[test]
    fn validation_requires_two_out_connections() {
        let mut d = BinaryDecisionEntity::new("bin1".to_string(), BinaryDecisionOpts::default());
        assert!(!d.do_validation_before_run()); // zero out-connections
        d.run_time_step(bgn(0.1)); // noop, no panic
        assert_eq!(d.get_graph_data_typed().processed_count, 3);
    }

    #[test]
    #[should_panic(expected = "Binary decision node needs two connections out.")]
    fn do_validation_panics_without_two_connections() {
        let mut d = BinaryDecisionEntity::new("bin2".to_string(), BinaryDecisionOpts::default());
        d.do_validation();
    }
}
