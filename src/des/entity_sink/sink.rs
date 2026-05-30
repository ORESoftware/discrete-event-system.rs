//! Canonical use path: `crate::des::entity_sink::sink::*`
//!
//! Port of `src/des/entity-sink/sink.ts` — terminal entities that absorb
//! (destroy) moving-entities.
//!
//! TS chain: `AbstractSinkEntity extends Entity implements HasComputedProperties,
//! HasManyInputConnections`; concrete `EntitySink`. Sinks are INPUT-ONLY nodes,
//! so they hold `connections_in` and implement `HasInput` /
//! `HasManyInputConnections` but not `HasOutput`.
//!
//! PORT NOTES:
//!   * the `[entityType] = 'Sink'` + `['entity.type'] = 'Sink'` symbol brand
//!     becomes the [`SinkKind`] tag (shared with `generic_sink`).
//!   * the abstract `acceptItem()` took NO args in the TS but the `HasInput`
//!     trait's `accept_item(m)` takes one — reconciled by ignoring `m`.
//!   * the constructor's `rv: RandomVariable` parameter was never used -> dropped;
//!     `opts: {}` was empty -> dropped.
//!   * `reg.registerSink(this)` is NOT done in the constructor (no self-`Rc`); the
//!     integrator registers after wrapping in an `Rc`.
//!   * `m.doFinish()` is the absorb side effect (a `MovingEntity` trait method).
//!   * `[util.inspect.custom]` / `getCleanVersion()` collapse into
//!     `get_with_computed_properties` returning an `EntityGraphData` payload.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_queue::queue::build_in_conn;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasManyInputConnections, HasManyOutputConnections,
};
use crate::des::r#abstract::r#abstract::{Entity, EntityCore, EntityConnection};
use crate::des::shared::precision::Decimal;

/// The sink brand (`[entityType] = 'Sink'`), shared with `generic_sink`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkKind {
    Sink,
}

/// Field-bag for `abstract class AbstractSinkEntity` — a sink's inbound edge set
/// atop the shared [`EntityCore`].
pub struct SinkCore {
    pub entity: EntityCore,
    pub connections_in: Vec<Rc<RefCell<EntityConnection>>>,
}

impl SinkCore {
    pub fn new(id: String) -> Self {
        SinkCore {
            entity: EntityCore::new(id),
            connections_in: Vec::new(),
        }
    }

    /// `addInConnection(source)` — store an inbound placeholder edge. See
    /// `build_in_conn`: the real ends reference `this`/the upstream node, which a
    /// `&mut self` method cannot store, so they are deferred.
    pub fn add_in_connection_from(
        &mut self,
        _source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Rc<RefCell<EntityConnection>> {
        let conn = build_in_conn();
        self.connections_in.push(conn.clone());
        conn
    }

    pub fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.connections_in.clone()
    }
}

/// `abstract class AbstractSinkEntity` — behaviour marker shared by the sinks.
pub trait AbstractSinkEntity: Entity {}

// =============================================================================
// EntitySink
// =============================================================================

/// `class EntitySink` — counts and destroys every token handed to it.
pub struct EntitySink {
    pub core: SinkCore,
    pub kind: SinkKind,
    pub destroyed_count: i64,
}

impl EntitySink {
    /// `new(id)` — the TS `rv` / `opts` constructor params were unused and are
    /// dropped.
    pub fn new(id: String) -> Self {
        EntitySink {
            core: SinkCore::new(id),
            kind: SinkKind::Sink,
            destroyed_count: 0,
        }
    }

    /// `doAudit()` -> `{ totalSize }`.
    pub fn do_audit(&self) -> i64 {
        self.destroyed_count
    }
}

impl AbstractSinkEntity for EntitySink {}

impl Entity for EntitySink {
    fn core(&self) -> &EntityCore {
        &self.core.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        false
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("destroyedCount", self.destroyed_count as f64)
            .with("timeStepCount", self.core.entity.time_step_count as f64)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default().with("destroyedCount", self.destroyed_count as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        self.core.entity.time_step_count += 1;
        let gd = self.get_graph_data();
        self.send_update_to_subs("SINK", &gd);
    }
}

impl HasInput for EntitySink {
    fn id(&self) -> String {
        self.core.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        // TODO (from TS): should reject items if full?
        true
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        self.destroyed_count += 1;
        m.borrow_mut().do_finish();
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        false
    }
    fn notify_sources(&mut self) {}
    fn do_setup_after_output_conn(&mut self) -> bool {
        false
    }
    fn add_in_connection(
        &mut self,
        source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        Some(self.core.add_in_connection_from(source))
    }
}

impl HasManyInputConnections for EntitySink {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.core.get_in_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::BasicMovingEntity;
    use crate::des::shared::precision::bgn;

    #[test]
    fn sink_absorbs_and_counts() {
        let mut sink = EntitySink::new("sink1".to_string());
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(BasicMovingEntity::new()));
        assert!(sink.accept_item(m.clone()));
        sink.take_item(m.clone());
        assert_eq!(sink.destroyed_count, 1);
        assert!(m.borrow().moving_core().has_exited_system);
        sink.run_time_step(bgn(0.1));
        assert_eq!(sink.core.entity.time_step_count, 1);
        assert_eq!(sink.do_audit(), 1);
    }
}
