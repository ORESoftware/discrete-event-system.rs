//! Canonical use path: `crate::des::observers::program_observer::*`
//!
//! Port of `src/des/observers/program-observer.ts` — an [`EntityObserver`] that
//! tracks every moving-entity created during a run.
//!
//! Conversion notes (file-specific):
//!   * `doUpdate` dispatches on `type === 'NEW_BASIC_MOVING_ENTITY'` (a string).
//!     The framework `EntityObserver::do_update(type_: &str, payload: &dyn Any)`
//!     surface is string-typed, so the event name is matched against the
//!     [`NEW_BASIC_MOVING_ENTITY`] constant (the conversion note's "event enum"
//!     would require changing the framework trait, which is off-limits).
//!   * `movingEntities: Set<AbstractMovingEntity>` -> `Vec<Rc<RefCell<dyn
//!     MovingEntity>>>` with identity (`Rc::ptr_eq`) dedup to mirror `Set`.
//!   * `m as any` payload casts -> `&dyn Any` downcast to the concrete
//!     `Rc<RefCell<dyn MovingEntity>>` handle the sender passes.
//!   * `getStatus()` `console.log` loop -> [`ProgramObserver::get_status`] returns
//!     the collected graph-data snapshots (printing is left to the caller).

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::r#abstract::interfaces::EntityGraphData;
use crate::des::r#abstract::r#abstract::EntityObserver;

/// The event name the observer reacts to (`type === 'NEW_BASIC_MOVING_ENTITY'`).
pub const NEW_BASIC_MOVING_ENTITY: &str = "NEW_BASIC_MOVING_ENTITY";

/// `class ProgramObserver extends EntityObserver<any>`.
#[derive(Default)]
pub struct ProgramObserver {
    /// `movingEntities = new Set<AbstractMovingEntity<any>>()`.
    pub moving_entities: Vec<Rc<RefCell<dyn MovingEntity>>>,
}

impl ProgramObserver {
    pub fn new() -> Self {
        Self::default()
    }

    /// `addMovingEntityRef(m)` — add with `Set`-style identity dedup.
    pub fn add_moving_entity_ref(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        if !self.moving_entities.iter().any(|e| Rc::ptr_eq(e, &m)) {
            self.moving_entities.push(m);
        }
    }

    /// `getStatus()` — snapshot each tracked entity's graph data (the TS logged
    /// `v.getGraphData()` per entity).
    pub fn get_status(&self) -> Vec<EntityGraphData> {
        self.moving_entities
            .iter()
            .map(|v| v.borrow().get_graph_data())
            .collect()
    }
}

impl EntityObserver for ProgramObserver {
    /// `doUpdate<T>(type, m)` — on `NEW_BASIC_MOVING_ENTITY`, record the entity.
    fn do_update(&mut self, type_: &str, payload: &dyn Any) {
        if type_ == NEW_BASIC_MOVING_ENTITY {
            if let Some(m) = payload.downcast_ref::<Rc<RefCell<dyn MovingEntity>>>() {
                self.add_moving_entity_ref(m.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::BasicMovingEntity;

    #[test]
    fn tracks_new_basic_moving_entities() {
        let mut obs = ProgramObserver::new();
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(BasicMovingEntity::new()));

        // Matching event records it; the payload is the erased entity handle.
        let payload: &dyn Any = &m;
        obs.do_update(NEW_BASIC_MOVING_ENTITY, payload);
        assert_eq!(obs.moving_entities.len(), 1);

        // Re-sending the same handle does not duplicate (Set semantics).
        let payload: &dyn Any = &m;
        obs.do_update(NEW_BASIC_MOVING_ENTITY, payload);
        assert_eq!(obs.moving_entities.len(), 1);

        // Unrelated events are ignored.
        let payload: &dyn Any = &m;
        obs.do_update("SOME_OTHER_EVENT", payload);
        assert_eq!(obs.moving_entities.len(), 1);
    }

    #[test]
    fn get_status_snapshots_graph_data() {
        let mut obs = ProgramObserver::new();
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(BasicMovingEntity::new()));
        obs.add_moving_entity_ref(m);
        let status = obs.get_status();
        assert_eq!(status.len(), 1);
        // BasicMovingEntity populates a `timeInSystem` key in its graph data.
        assert!(status[0].data.contains_key("timeInSystem"));
    }
}
