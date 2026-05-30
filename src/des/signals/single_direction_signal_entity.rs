//! Canonical use path: `crate::des::signals::single_direction_signal_entity::*`
//!
//! Port of `src/des/signals/single-direction-signal-entity.ts` — the signal node
//! base with ONE inbound connection and MANY outbound connections.
//!
//! `abstract class SingleInManyOutSignalEntity<E,V> extends SignalEntity`
//! becomes the [`SingleInManyOutSignalEntity`] trait plus the composable
//! field-bag [`SingleInManyOutSignalCore`] (same pattern as the multi-directional
//! base and the framework's `BidirectionalCore`).
//!
//! Conversion notes (file-specific):
//!   * `maxQueueSize = <unknown>null as number` -> `Option<usize>`.
//!   * `connectionIn = <unknown>null as EntityConnection` -> `Option<Rc<RefCell<EntityConnection>>>`.
//!   * `connectionsOut: Set<EntityConnection>` -> `Vec<Rc<RefCell<EntityConnection>>>`;
//!     `queue: LinkedQueue<SignalValue>` -> `LinkedQueue<u64, Rc<RefCell<dyn MovingEntity>>>`.
//!   * `addInConnection` does `return this.connectionIn = conn` (assignment-as-
//!     expression); Rust has no assignment-expressions, so set the field then
//!     return the handle. As with the multi-directional base, the
//!     `EntityConnection(source, this)` self-reference forces deferred
//!     construction — `add_in_connection_built` stores a pre-built connection.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::{MovingCore, MovingEntity};
use crate::des::r#abstract::r#abstract::EntityConnection;

use crate::des::signals::r#abstract::SignalEntity;

/// Composable field-bag for `abstract class SingleInManyOutSignalEntity`.
pub struct SingleInManyOutSignalCore {
    pub moving: MovingCore,
    /// `maxQueueSize = null as number`.
    pub max_queue_size: Option<usize>,
    /// `queue = new LinkedQueue<SignalValue>()`.
    pub queue: crate::des::shared::linked_queue::LinkedQueue<u64, Rc<RefCell<dyn MovingEntity>>>,
    /// `connectionIn = null as EntityConnection` (single inbound edge).
    pub connection_in: Option<Rc<RefCell<EntityConnection>>>,
    /// `connectionsOut = new Set<EntityConnection>()`.
    pub connections_out: Vec<Rc<RefCell<EntityConnection>>>,
}

impl SingleInManyOutSignalCore {
    pub fn new(id: String) -> Self {
        SingleInManyOutSignalCore {
            moving: MovingCore::new(id),
            max_queue_size: None,
            queue: crate::des::shared::linked_queue::LinkedQueue::new(),
            connection_in: None,
            connections_out: Vec::new(),
        }
    }

    /// `addInConnection(source)` — set the single inbound edge and return it
    /// (the TS `return this.connectionIn = conn` assignment-expression).
    pub fn add_in_connection_built(
        &mut self,
        conn: Rc<RefCell<EntityConnection>>,
    ) -> Rc<RefCell<EntityConnection>> {
        self.connection_in = Some(conn.clone());
        conn
    }

    /// `addOutConnection(target)` (deferred construction).
    pub fn add_out_connection_built(&mut self, conn: Rc<RefCell<EntityConnection>>) {
        self.connections_out.push(conn);
    }

    /// `getOutConnections()`.
    pub fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.connections_out.clone()
    }

    /// `getInConnection()`.
    pub fn get_in_connection(&self) -> Option<Rc<RefCell<EntityConnection>>> {
        self.connection_in.clone()
    }

    /// `isEmpty(): boolean { return this.queue.size < 1; }`.
    pub fn is_empty(&self) -> bool {
        self.queue.size() < 1
    }

    /// `isFull()` — same `null`-max-queue-size quirk as the multi-directional base.
    pub fn is_full(&self) -> bool {
        self.queue.size() >= self.max_queue_size.unwrap_or(0)
    }

    /// Enqueue an accepted token.
    pub fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        self.queue.enqueue(m);
    }
}

/// `abstract class SingleInManyOutSignalEntity<E,V>` (implements
/// `HasInternalQueue`, `HasSingleInputConnection`, `HasManyOutputConnections`).
pub trait SingleInManyOutSignalEntity: SignalEntity {
    fn si_core(&self) -> &SingleInManyOutSignalCore;
    fn si_core_mut(&mut self) -> &mut SingleInManyOutSignalCore;

    /// `abstract acceptItem(m): boolean`.
    fn accept_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) -> bool;
    /// `abstract takeItem(m): void`.
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>);

    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.si_core().get_out_connections()
    }
    fn get_in_connection(&self) -> Option<Rc<RefCell<EntityConnection>>> {
        self.si_core().get_in_connection()
    }
    fn is_empty(&self) -> bool {
        self.si_core().is_empty()
    }
    fn is_full(&self) -> bool {
        self.si_core().is_full()
    }
    /// `doSetupAfterInputConn(): boolean { return false; }`.
    fn do_setup_after_input_conn(&mut self) -> bool {
        false
    }
    /// `doSetupAfterOutputConn(): boolean { return false; }`.
    fn do_setup_after_output_conn(&mut self) -> bool {
        false
    }
    fn notify_sources(&mut self) {}
    fn notify_targets(&mut self) {}

    fn add_in_connection_built(
        &mut self,
        conn: Rc<RefCell<EntityConnection>>,
    ) -> Rc<RefCell<EntityConnection>> {
        self.si_core_mut().add_in_connection_built(conn)
    }
    fn add_out_connection_built(&mut self, conn: Rc<RefCell<EntityConnection>>) {
        self.si_core_mut().add_out_connection_built(conn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::MovingValue;
    use crate::des::r#abstract::interfaces::EntityGraphData;
    use crate::des::r#abstract::r#abstract::{Entity, EntityCore};
    use crate::des::shared::precision::Decimal;
    use crate::des::signals::r#abstract::{SignalEntity, SignalTimeStepOpts};
    use crate::des::signals::signal_value::SignalValue;

    struct StubNode {
        core: SingleInManyOutSignalCore,
    }

    impl Entity for StubNode {
        fn core(&self) -> &EntityCore {
            &self.core.moving.entity
        }
        fn core_mut(&mut self) -> &mut EntityCore {
            &mut self.core.moving.entity
        }
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

    impl MovingEntity for StubNode {
        fn moving_core(&self) -> &MovingCore {
            &self.core.moving
        }
        fn moving_core_mut(&mut self) -> &mut MovingCore {
            &mut self.core.moving
        }
        fn get_value(&self) -> MovingValue {
            MovingValue::default()
        }
        fn run_finish(&mut self) {}
    }

    impl SignalEntity for StubNode {
        fn run_time_step_signal(&mut self, _step: Decimal, _opts: Option<SignalTimeStepOpts>) {}
    }

    impl SingleInManyOutSignalEntity for StubNode {
        fn si_core(&self) -> &SingleInManyOutSignalCore {
            &self.core
        }
        fn si_core_mut(&mut self) -> &mut SingleInManyOutSignalCore {
            &mut self.core
        }
        fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
            true
        }
        fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
            self.core.take_item(m);
        }
    }

    #[test]
    fn single_input_and_fifo() {
        let mut node = StubNode {
            core: SingleInManyOutSignalCore::new("si".into()),
        };
        assert!(node.is_empty());
        assert!(node.get_in_connection().is_none());
        let sv: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(SignalValue::unity()));
        node.take_item(sv);
        assert!(!node.is_empty());
    }
}
