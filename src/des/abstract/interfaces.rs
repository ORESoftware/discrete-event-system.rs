//! Canonical use path: `crate::des::r#abstract::interfaces::*`
//!
//! Port of `src/des/abstract/interfaces.ts` — the capability traits and
//! graph-data shapes of the queueing-network entity model.
//!
//! The TypeScript interfaces were heavily generic (`<S, T>` / `<any>`). Those
//! generics were pervasive `any`, so they are ERASED here: every trait is
//! NON-generic and object-safe, so entities can be held as
//! `Rc<RefCell<dyn Trait>>` trait objects. Methods that returned `this`
//! (builder style) return `()` to preserve object-safety.
//!
//! Graph edges are modelled with `Rc<RefCell<..>>`; the back-edge from a
//! connection to its source is a `std::rc::Weak` (see `EntityConnection` in
//! `abstract.rs`) to avoid reference cycles.

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::r#abstract::r#abstract::{EntityConnection, EntityObserver};

/// `interface EntityGraphData` — in TS an empty structural marker that concrete
/// `getGraphData()` results widen. Kept as a small open key→value payload so
/// entities can attach derived numbers (e.g. `timeInSystem`) without per-entity
/// DTO types.
///
/// PORT NOTE: the TS interface was literally empty (`{}`); the optional `data`
/// map is an additive convenience and defaults empty, so it still behaves as a
/// marker.
#[derive(Clone, Debug, Default)]
pub struct EntityGraphData {
    pub data: std::collections::HashMap<String, f64>,
}

impl EntityGraphData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, key: &str, value: f64) -> Self {
        self.data.insert(key.to_string(), value);
        self
    }
}

/// `enum EventNames`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventNames {
    Foo,
}

/// `interface HasId`.
pub trait HasId {
    fn id(&self) -> String;
}

/// `interface HasEntityValidation`.
pub trait HasEntityValidation {
    fn validate(&self) -> bool;
}

/// `interface IsObservable`.
///
/// The TS interface also declared `subscribers` / `subscribersByEvent` *fields*;
/// Rust traits cannot hold fields, so the storage lives in `EntityCore`
/// (see `abstract.rs`) and the `Entity` trait provides default implementations.
/// This trait is the object-safe behavioural contract.
pub trait IsObservable {
    fn subscribe_to(&mut self, name: &str, o: Rc<RefCell<dyn EntityObserver>>);
    fn subscribe(&mut self, o: Rc<RefCell<dyn EntityObserver>>);
    /// TODO (from TS): observers only get updates every N timesteps.
    fn subscribe_with_frequency(&mut self, count: i64, o: Rc<RefCell<dyn EntityObserver>>);
    /// Returns `true` if the subject actually had the observer.
    fn unsubscribe(&mut self, o: &Rc<RefCell<dyn EntityObserver>>) -> bool;
    fn send_update_to_subs(&mut self, type_: &str, v: &dyn Any);
}

/// `interface HasOutput<S, T>` — a node that can feed downstream targets.
pub trait HasOutput {
    fn id(&self) -> String;
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>>;
    fn do_setup_after_input_conn(&mut self) -> bool;
    fn notify_targets(&mut self);
    fn do_setup_after_output_conn(&mut self) -> bool;
}

/// `interface HasSingleOutputConnection<S, T> extends HasOutput`.
pub trait HasSingleOutputConnection: HasOutput {
    fn get_out_connection(&self) -> Rc<RefCell<EntityConnection>>;
}

/// `interface HasManyOutputConnections<S, T> extends HasOutput`.
pub trait HasManyOutputConnections: HasOutput {
    /// TS returned `Set<EntityConnection>`; here a `Vec` of `Rc` clones, which is
    /// both object-safe and borrow-checker friendly for the fan-out loop.
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>>;
}

/// `interface HasInternalQueue<T>`.
///
/// The TS interface exposed a `queue: LinkedQueue<T>` field plus `maxQueueSize`;
/// the queue itself lives on the concrete entity and is reached via accessors.
pub trait HasInternalQueue {
    fn max_queue_size(&self) -> usize;
    fn is_full(&self) -> bool;
    fn is_empty(&self) -> bool;
}

/// `interface HasInput<S, T>` — a node that can accept moving entities.
pub trait HasInput {
    fn id(&self) -> String;
    fn accept_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) -> bool;
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>);
    fn do_setup_after_input_conn(&mut self) -> bool;
    fn notify_sources(&mut self);
    fn do_setup_after_output_conn(&mut self) -> bool;
    fn add_in_connection(
        &mut self,
        source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>>;
}

/// `interface HasSingleInputConnection<S, T> extends HasInput`.
pub trait HasSingleInputConnection: HasInput {
    fn get_in_connection(&self) -> Rc<RefCell<EntityConnection>>;
}

/// `interface HasManyInputConnections<S, T> extends HasInput`.
pub trait HasManyInputConnections: HasInput {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Node {
        id: String,
    }
    impl HasId for Node {
        fn id(&self) -> String {
            self.id.clone()
        }
    }
    impl HasEntityValidation for Node {
        fn validate(&self) -> bool {
            !self.id.is_empty()
        }
    }

    #[test]
    fn graph_data_marker_carries_payload() {
        let g = EntityGraphData::new().with("timeInSystem", 1.5);
        assert_eq!(g.data.get("timeInSystem"), Some(&1.5));
    }

    #[test]
    fn has_id_and_validation() {
        let n = Node { id: "x".into() };
        assert_eq!(n.id(), "x");
        assert!(n.validate());
        assert_eq!(EventNames::Foo, EventNames::Foo);
    }
}
