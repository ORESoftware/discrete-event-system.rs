//! Canonical use path: `crate::des::signals::multi_directional_signal_entity::*`
//!
//! Port of `src/des/signals/multi-directional-signal-entity.ts` — the signal
//! node base with MANY inbound and MANY outbound connections (parent of
//! `Adder` / `Multiplexer` / `Integrator` / `Differentiator` / `SignalIncrementor`).
//!
//! `abstract class MultiDirectionalSignalEntity<E,V> extends SignalEntity`
//! becomes the [`MultiDirectionalSignalEntity`] trait (a sub-trait of
//! [`SignalEntity`]) plus the composable field-bag [`MultiDirectionalSignalCore`].
//! Concrete nodes embed a core and `impl` the trait (mirroring the framework's
//! `BidirectionalCore` pattern in `abstract/abstract.rs`).
//!
//! Conversion notes (file-specific):
//!   * `maxQueueSize = <unknown>null as number` -> `Option<usize>`.
//!   * `queue: LinkedQueue<SignalValue>` of `acceptItem`'d tokens ->
//!     `LinkedQueue<u64, Rc<RefCell<dyn MovingEntity>>>` (unkeyed FIFO). Signals
//!     flow as moving-entities, so the queue holds erased `dyn MovingEntity`
//!     handles; the carried sample is read back via `MovingEntity::get_value`.
//!   * `connectionsIn/Out: Set<EntityConnection>` -> `Vec<Rc<RefCell<EntityConnection>>>`.
//!   * PORT NOTE: `addInConnection(source)`/`addOutConnection(target)` built an
//!     `EntityConnection(source, this)` referencing `this`. A struct cannot mint
//!     an `Rc` to itself, so (exactly like the framework `BidirectionalCore`)
//!     connection CONSTRUCTION is deferred to the wiring site and these helpers
//!     push the already-built `Rc<RefCell<EntityConnection>>`.
//!   * PORT NOTE: `isFull()` is `queue.size >= maxQueueSize` with `maxQueueSize`
//!     left `null`; in JS `n >= null` coerces `null -> 0`, so it is effectively
//!     `size >= 0` (always true). Preserved via `unwrap_or(0)`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::{MovingCore, MovingEntity};
use crate::des::r#abstract::r#abstract::EntityConnection;

use crate::des::signals::r#abstract::SignalEntity;

/// Composable field-bag for `abstract class MultiDirectionalSignalEntity` — the
/// data half (moving-entity state + internal queue + both connection sets).
pub struct MultiDirectionalSignalCore {
    pub moving: MovingCore,
    /// `maxQueueSize = null as number` placeholder.
    pub max_queue_size: Option<usize>,
    /// `queue = new LinkedQueue<SignalValue>()`.
    pub queue: crate::des::shared::linked_queue::LinkedQueue<u64, Rc<RefCell<dyn MovingEntity>>>,
    /// `connectionsIn = new Set<EntityConnection>()`.
    pub connections_in: Vec<Rc<RefCell<EntityConnection>>>,
    /// `connectionsOut = new Set<EntityConnection>()`.
    pub connections_out: Vec<Rc<RefCell<EntityConnection>>>,
}

impl MultiDirectionalSignalCore {
    pub fn new(id: String) -> Self {
        MultiDirectionalSignalCore {
            moving: MovingCore::new(id),
            max_queue_size: None,
            queue: crate::des::shared::linked_queue::LinkedQueue::new(),
            connections_in: Vec::new(),
            connections_out: Vec::new(),
        }
    }

    /// Push a pre-built inbound connection (see the deferred-construction note).
    pub fn add_in_connection_built(&mut self, conn: Rc<RefCell<EntityConnection>>) {
        self.connections_in.push(conn);
    }

    /// Push a pre-built outbound connection (see the deferred-construction note).
    pub fn add_out_connection_built(&mut self, conn: Rc<RefCell<EntityConnection>>) {
        self.connections_out.push(conn);
    }

    /// `getOutConnections()`.
    pub fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.connections_out.clone()
    }

    /// `getInConnections()`.
    pub fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.connections_in.clone()
    }

    /// `isEmpty(): boolean { return this.queue.size < 1; }`.
    pub fn is_empty(&self) -> bool {
        self.queue.size() < 1
    }

    /// `isFull()` — see the module PORT NOTE on the `null` max-queue-size quirk.
    pub fn is_full(&self) -> bool {
        self.queue.size() >= self.max_queue_size.unwrap_or(0)
    }

    /// `takeItem(m) { this.queue.enqueue(m); }`.
    pub fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        self.queue.enqueue(m);
    }
}

/// `abstract class MultiDirectionalSignalEntity<E,V>` (implements
/// `HasInternalQueue`, `HasManyInputConnections`, `HasManyOutputConnections`).
///
/// PORT NOTE: rather than `impl`-ing those framework graph traits literally
/// (their `add_*_connection` methods need an `Rc`-to-self and `take_item` would
/// need to downcast `dyn MovingEntity`), the structural surface is provided as
/// trait methods delegating to [`MultiDirectionalSignalCore`] — the same
/// field-bag-with-inherent-methods approach the framework uses for
/// `BidirectionalCore`.
pub trait MultiDirectionalSignalEntity: SignalEntity {
    fn md_core(&self) -> &MultiDirectionalSignalCore;
    fn md_core_mut(&mut self) -> &mut MultiDirectionalSignalCore;

    /// `abstract acceptItem(m): boolean`.
    fn accept_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) -> bool;
    /// `abstract takeItem(m): void`.
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>);

    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.md_core().get_out_connections()
    }
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.md_core().get_in_connections()
    }
    fn is_empty(&self) -> bool {
        self.md_core().is_empty()
    }
    fn is_full(&self) -> bool {
        self.md_core().is_full()
    }
    /// `doSetupAfterInputConn(): boolean { return false; }`.
    fn do_setup_after_input_conn(&mut self) -> bool {
        false
    }
    /// `doSetupAfterOutputConn(): boolean { return false; }`.
    fn do_setup_after_output_conn(&mut self) -> bool {
        false
    }
    /// `notifySources()` — TODO in the TS source.
    fn notify_sources(&mut self) {}
    /// `notifyTargets()` — TODO in the TS source.
    fn notify_targets(&mut self) {}

    /// `addInConnection(source)` (deferred construction — see PORT NOTE).
    fn add_in_connection_built(&mut self, conn: Rc<RefCell<EntityConnection>>) {
        self.md_core_mut().add_in_connection_built(conn);
    }
    /// `addOutConnection(target)` (deferred construction — see PORT NOTE).
    fn add_out_connection_built(&mut self, conn: Rc<RefCell<EntityConnection>>) {
        self.md_core_mut().add_out_connection_built(conn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::MovingValue;
    use crate::des::r#abstract::interfaces::EntityGraphData;
    use crate::des::r#abstract::r#abstract::EntityCore;
    use crate::des::shared::precision::Decimal;
    use crate::des::signals::r#abstract::{SignalEntity, SignalTimeStepOpts};
    use crate::des::signals::signal_value::SignalValue;

    /// Minimal concrete multi-directional node for the base-trait smoke test.
    struct StubNode {
        core: MultiDirectionalSignalCore,
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
    use crate::des::r#abstract::r#abstract::Entity;

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

    impl MultiDirectionalSignalEntity for StubNode {
        fn md_core(&self) -> &MultiDirectionalSignalCore {
            &self.core
        }
        fn md_core_mut(&mut self) -> &mut MultiDirectionalSignalCore {
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
    fn queue_and_connection_helpers() {
        let mut node = StubNode {
            core: MultiDirectionalSignalCore::new("md".into()),
        };
        assert!(node.is_empty());
        let sv: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(SignalValue::unity()));
        assert!(node.accept_item(sv.clone()));
        node.take_item(sv);
        assert!(!node.is_empty());
        assert_eq!(node.get_out_connections().len(), 0);
        assert!(!node.do_setup_after_input_conn());
    }
}
