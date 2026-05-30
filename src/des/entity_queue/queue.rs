//! Canonical use path: `crate::des::entity_queue::queue::*`
//!
//! Port of `src/des/entity-queue/queue.ts` — the base buffering queue entity
//! (the parent that `EntityProcessor` / `EntityNumericProcessor` `extend`).
//!
//! TS `class QueueEntity extends AbstractBidirectionalEntity implements
//! HasInternalQueue`. Rust has no inheritance, so this is modelled as a struct
//! [`QueueEntity`] embedding the framework's [`BidirectionalCore`] field-bag plus
//! the internal FIFO of moving entities. The processors COMPOSE a `QueueEntity`
//! (`base`) rather than extending it.
//!
//! PORT NOTES:
//!   * the generic `<S, T>` was pervasive `any` and is erased; moving entities
//!     are held as `Rc<RefCell<dyn MovingEntity>>` trait objects in a `VecDeque`.
//!   * `maxQueueSize = -1` was a "-1 = unbounded" sentinel `number`; the
//!     object-safe `HasInternalQueue::max_queue_size` returns `usize`, so `-1`
//!     maps to `usize::MAX` (unbounded).
//!   * `getSerializableData()` logged a warning then `throw makeError('this is
//!     wrong.')` BEFORE its (dead) return — preserved as `panic!`.
//!   * `getWithComputedProperties()` (returning `{ 'queue.size' }`) and
//!     `getGraphData()` (hardcoded `processedCount: 3`) are kept as
//!     `EntityGraphData` payloads.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::general::general::get_short_uuid;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasInternalQueue, HasManyInputConnections, HasManyOutputConnections,
    HasOutput,
};
use crate::des::r#abstract::r#abstract::{
    BidirectionalCore, ConnectionOpts, Entity, EntityCore, EntityConnection,
};
use crate::des::shared::precision::Decimal;

/// Build an out-edge connection holding `target`. The source back-edge is left
/// unset (`None`): a `&mut self` trait method cannot mint a `Weak` to itself, so
/// the source `Weak` is deferred to whoever owns the `Rc<RefCell<Self>>`.
pub(crate) fn build_out_conn(target: Rc<RefCell<dyn HasInput>>) -> Rc<RefCell<EntityConnection>> {
    Rc::new(RefCell::new(EntityConnection {
        id: get_short_uuid(),
        source: None,
        target: Some(target),
        opts: ConnectionOpts::default(),
    }))
}

/// Build an in-edge placeholder connection. Both ends are `None` because the
/// edge references `this` (the target) and the upstream `dyn HasOutput`
/// back-edge, neither of which a `&mut self` method can store; the integrator
/// rewires real ends from the owning `Rc`s.
pub(crate) fn build_in_conn() -> Rc<RefCell<EntityConnection>> {
    Rc::new(RefCell::new(EntityConnection {
        id: get_short_uuid(),
        source: None,
        target: None,
        opts: ConnectionOpts::default(),
    }))
}

/// `interface QueueEntityGraphData extends EntityGraphData { processedCount }`.
#[derive(Clone, Debug, Default)]
pub struct QueueEntityGraphData {
    pub base: EntityGraphData,
    pub processed_count: i64,
}

/// `opts: { xx?: boolean }` — a near-empty marker config.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueueOpts {
    pub xx: Option<bool>,
}

/// `class QueueEntity` — buffering queue entity, base for the processors.
pub struct QueueEntity {
    pub bi: BidirectionalCore,
    pub queue: VecDeque<Rc<RefCell<dyn MovingEntity>>>,
    /// `-1` sentinel = unbounded (see module note).
    pub max_queue_size: i64,
    pub opts: QueueOpts,
}

impl QueueEntity {
    pub fn new(id: String, opts: QueueOpts) -> Self {
        QueueEntity {
            bi: BidirectionalCore::new(id),
            queue: VecDeque::new(),
            max_queue_size: -1,
            opts,
        }
    }

    /// `getGraphData(): QueueEntityGraphData` (the TS hardcoded stub).
    pub fn get_graph_data_typed(&self) -> QueueEntityGraphData {
        QueueEntityGraphData {
            base: EntityGraphData::default().with("queue.size", self.queue.len() as f64),
            processed_count: 3,
        }
    }
}

impl Entity for QueueEntity {
    fn core(&self) -> &EntityCore {
        &self.bi.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.bi.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        false
    }
    fn get_graph_data(&self) -> EntityGraphData {
        // Object.assign(getWithComputedProperties(), { processedCount: 3 })
        EntityGraphData::default()
            .with("queue.size", self.queue.len() as f64)
            .with("processedCount", 3.0)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default().with("queue.size", self.queue.len() as f64)
    }
    fn get_serializable_data(&self) -> String {
        eprintln!(
            "[queue:{}] getSerializableData() called on base QueueEntity — this path is not supported and will throw.",
            self.bi.entity.id
        );
        panic!("this is wrong.");
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        self.bi.entity.time_step_count += 1;
    }
}

impl HasInternalQueue for QueueEntity {
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

impl HasInput for QueueEntity {
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

impl HasManyInputConnections for QueueEntity {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_in_connections()
    }
}

impl HasOutput for QueueEntity {
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

impl HasManyOutputConnections for QueueEntity {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_out_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::BasicMovingEntity;
    use crate::des::shared::precision::bgn;

    fn moving() -> Rc<RefCell<dyn MovingEntity>> {
        Rc::new(RefCell::new(BasicMovingEntity::new()))
    }

    #[test]
    fn take_item_buffers_and_runs_step() {
        let mut q = QueueEntity::new("q1".to_string(), QueueOpts::default());
        assert!(q.is_empty());
        q.take_item(moving());
        assert!(!q.is_empty());
        assert_eq!(q.queue.len(), 1);
        q.run_time_step(bgn(0.1));
        assert_eq!(q.bi.entity.time_step_count, 1);
    }

    #[test]
    fn unbounded_max_queue_size_is_usize_max() {
        let q = QueueEntity::new("q2".to_string(), QueueOpts::default());
        assert_eq!(HasInternalQueue::max_queue_size(&q), usize::MAX);
        assert!(!q.is_full());
    }

    #[test]
    #[should_panic(expected = "this is wrong.")]
    fn serializable_data_panics() {
        let q = QueueEntity::new("q3".to_string(), QueueOpts::default());
        let _ = q.get_serializable_data();
    }
}
