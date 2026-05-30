//! Canonical use path: `crate::des::entity_sink::generic_sink::*`
//!
//! Port of `src/des/entity-sink/generic-sink.ts` — a sink that logs each
//! absorbed entity's value before destroying it.
//!
//! TS `class GenericEntitySink extends AbstractSinkEntity implements
//! StationaryEntity, HasManyInputConnections`. Reuses the shared [`SinkCore`]
//! field-bag, [`SinkKind`] tag and [`AbstractSinkEntity`] marker from `sink`
//! (the TS `EntitySinkGraphData` + `entityType` Symbol were duplicated across the
//! two files; here they are defined once in `sink` and imported).
//!
//! PORT NOTES:
//!   * the ONLY behavioural difference from `EntitySink` is
//!     `console.log('generic sink value:', m.getValue())` in `takeItem` -> a
//!     `println!` of the token's [`MovingValue`].
//!   * `opts: {}` empty -> dropped; `[util.inspect.custom]` -> the
//!     `get_with_computed_properties` payload.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_sink::sink::{AbstractSinkEntity, SinkCore, SinkKind};
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasManyInputConnections, HasManyOutputConnections,
};
use crate::des::r#abstract::r#abstract::{Entity, EntityConnection, EntityCore};
use crate::des::shared::precision::Decimal;

/// `class GenericEntitySink` — like `EntitySink` but logs each token's value.
pub struct GenericEntitySink {
    pub core: SinkCore,
    pub kind: SinkKind,
    pub destroyed_count: i64,
}

impl GenericEntitySink {
    pub fn new(id: String) -> Self {
        GenericEntitySink {
            core: SinkCore::new(id),
            kind: SinkKind::Sink,
            destroyed_count: 0,
        }
    }
}

impl AbstractSinkEntity for GenericEntitySink {}

impl Entity for GenericEntitySink {
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

impl HasInput for GenericEntitySink {
    fn id(&self) -> String {
        self.core.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        true
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        self.destroyed_count += 1;
        println!("generic sink value: {:?}", m.borrow().get_value());
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

impl HasManyInputConnections for GenericEntitySink {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.core.get_in_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::BasicQuantityMovingEntity;
    use crate::des::shared::precision::bgn;

    #[test]
    fn generic_sink_absorbs_and_logs() {
        let mut sink = GenericEntitySink::new("gsink1".to_string());
        let m: Rc<RefCell<dyn MovingEntity>> =
            Rc::new(RefCell::new(BasicQuantityMovingEntity::new(7.0)));
        assert!(sink.accept_item(m.clone()));
        sink.take_item(m.clone());
        assert_eq!(sink.destroyed_count, 1);
        assert!(m.borrow().moving_core().has_exited_system);
        sink.run_time_step(bgn(0.1));
        assert_eq!(sink.core.entity.time_step_count, 1);
    }
}
