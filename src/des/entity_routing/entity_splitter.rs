//! Canonical use path: `crate::des::entity_routing::entity_splitter::*`
//!
//! Port of `src/des/entity-routing/entity-splitter.ts` — a node that BROADCASTS
//! each queued item to EVERY out-connection.
//!
//! TS `class EntitySplitter extends AbstractBidirectionalEntity implements
//! HasComputedProperties, HasInternalQueue`. Modelled as a struct embedding
//! [`BidirectionalCore`] plus a `VecDeque` of moving entities.
//!
//! PORT NOTES:
//!   * broadcast semantics: every out-connection MUST accept; a refusal was
//!     `throw makeError('must accept item:', k)` -> `panic!`.
//!   * `opts.replayItemsIfNotFirstAccepted` was typed as the literal `false`
//!     (a quirk) -> a plain `bool` field.
//!   * `getGraphData()` returns the hardcoded `{ processedCount: 3 }` stub
//!     (reusing [`TimeDelayEntityGraphData`] from `entity_travel::time_delay`).
//!   * `queue.dequeueIterator()` -> draining the `VecDeque`; no void sentinel.
//!   * a `None` (unresolvable) target is skipped rather than dereferenced.

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

/// `interface DecisionEntityGraph` — empty marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecisionEntityGraph;

/// `opts: { xx?: boolean, replayItemsIfNotFirstAccepted?: false }`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SplitterOpts {
    pub xx: Option<bool>,
    pub replay_items_if_not_first_accepted: bool,
}

/// `class EntitySplitter`.
pub struct EntitySplitter {
    pub bi: BidirectionalCore,
    pub queue: VecDeque<Rc<RefCell<dyn MovingEntity>>>,
    pub max_queue_size: i64,
    pub opts: SplitterOpts,
}

impl EntitySplitter {
    pub fn new(id: String, opts: SplitterOpts) -> Self {
        EntitySplitter {
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

impl Entity for EntitySplitter {
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
        EntityGraphData::default().with("queue.size", self.queue.len() as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        self.bi.entity.time_step_count += 1;

        let conns = self.bi.get_out_connections();
        if conns.len() < 1 {
            eprintln!(
                "[splitter:{}] has no out-connections; queued items cannot be broadcast downstream.",
                self.bi.entity.id
            );
        }

        while let Some(item) = self.queue.pop_front() {
            // Broadcast: send the item to EACH out-connection.
            for conn in &conns {
                let target = conn.borrow().get_target();
                let target = match target {
                    Some(t) => t,
                    None => continue,
                };
                if target.borrow_mut().accept_item(item.clone()) {
                    target.borrow_mut().take_item(item.clone());
                } else {
                    eprintln!(
                        "[splitter:{}] downstream refused item; splitter requires all targets to accept (broadcast semantics).",
                        self.bi.entity.id
                    );
                    panic!("must accept item:");
                }
            }
        }
    }
}

impl HasInternalQueue for EntitySplitter {
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

impl HasInput for EntitySplitter {
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

impl HasManyInputConnections for EntitySplitter {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_in_connections()
    }
}

impl HasOutput for EntitySplitter {
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

impl HasManyOutputConnections for EntitySplitter {
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
    fn empty_splitter_runs_without_panicking() {
        let mut s = EntitySplitter::new("split1".to_string(), SplitterOpts::default());
        // No items queued -> nothing to broadcast -> no panic.
        s.run_time_step(bgn(0.1));
        assert_eq!(s.bi.entity.time_step_count, 1);
    }

    #[test]
    fn take_item_enqueues() {
        let mut s = EntitySplitter::new("split2".to_string(), SplitterOpts::default());
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(BasicMovingEntity::new()));
        s.take_item(m);
        assert!(!s.is_empty());
    }
}
