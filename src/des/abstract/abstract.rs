//! Canonical use path: `crate::des::r#abstract::r#abstract::*`
//!
//! Port of `src/des/abstract/abstract.ts` — the root entity hierarchy for the
//! queueing-network model. (`abstract` is a Rust keyword, so the module is
//! addressed with the raw identifier `r#abstract`; the file stays `abstract.rs`.)
//!
//! The TS inheritance chain `Entity -> StationaryEntity ->
//! AbstractBidirectionalEntity` does NOT map to `extends`. Instead:
//!   * shared state lives in a field-bag struct ([`EntityCore`], [`BidirectionalCore`]),
//!   * behaviour lives in an object-safe [`Entity`] trait with default methods,
//!   * concrete entities compose a core and `impl Entity` (and the `Has*` traits
//!     from `interfaces.rs`).
//!
//! Generics `<E, V>` / `<S, T>` were pervasive `any` and are erased. The
//! placeholder `<unknown>null as ...` casts become `Option`. `EntityObserver`'s
//! payload generic is erased to `&dyn Any`.

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::des::r#abstract::interfaces::{EntityGraphData, HasId, HasInput, HasOutput};
use crate::des::shared::precision::Decimal;

/// `abstract class EntityObserver<T>` — erased to a non-generic observer that
/// receives an event `type_` plus a type-erased payload.
pub trait EntityObserver {
    fn do_update(&mut self, type_: &str, payload: &dyn Any);
}

/// `interface IsSerializable<T>` / `abstract class Serializable<T>`.
///
/// PORT NOTE: there is no `serde` dependency in the foundation, so the typed
/// `getSerializableData(): T` collapses to a JSON-ish `String`. Re-introduce a
/// real `#[derive(Serialize)]` DTO per entity when serialization is ported.
pub trait Serializable {
    fn get_serializable_data(&self) -> String;

    fn serialize(&self) -> String {
        self.get_serializable_data()
    }

    fn serialize_pretty(&self) -> String {
        self.get_serializable_data()
    }
}

/// `interface TimeStepOpts`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TimeStepOpts {
    pub is_final_time_step: bool,
}

/// `interface HasNumericValue` — modelled as a `Default`-able value struct.
#[derive(Clone, Copy, Debug, Default)]
pub struct HasNumericValue {
    pub value: f64,
}

/// Shared field-bag for every entity (the data half of `abstract class Entity`).
pub struct EntityCore {
    pub id: String,
    pub time_step_count: u64,
    pub short_uuid: Option<String>,
    pub subscribers: Vec<Rc<RefCell<dyn EntityObserver>>>,
    pub subscribers_by_event:
        std::collections::HashMap<String, Vec<Rc<RefCell<dyn EntityObserver>>>>,
}

impl EntityCore {
    pub fn new(id: String) -> Self {
        EntityCore {
            id,
            time_step_count: 0,
            short_uuid: None,
            subscribers: Vec::new(),
            subscribers_by_event: std::collections::HashMap::new(),
        }
    }
}

/// `abstract class Entity` — the behaviour half. Object-safe: no generics, no
/// `Self`-returning methods (the TS builder methods that returned `this` return
/// `()` here).
pub trait Entity {
    /// Access to the shared field-bag (required of every implementor).
    fn core(&self) -> &EntityCore;
    fn core_mut(&mut self) -> &mut EntityCore;

    // ── abstract hooks ────────────────────────────────────────────────────
    fn do_validation(&mut self);
    fn do_validation_before_run(&mut self) -> bool;
    fn get_graph_data(&self) -> EntityGraphData;
    fn run_time_step(&mut self, step_size: Decimal);

    // ── hooks with defaults (override as needed) ──────────────────────────
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default()
    }

    fn get_serializable_data(&self) -> String {
        "{}".to_string()
    }

    fn do_time_step(&mut self, step_size: Decimal) {
        self.run_time_step(step_size);
    }

    fn get_initial_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default().with("initialGraphData", 1.0)
    }

    // ── default implementations of the observable surface ─────────────────
    fn id(&self) -> String {
        self.core().id.clone()
    }

    fn short_uuid(&self) -> Option<String> {
        self.core().short_uuid.clone()
    }

    fn set_short_uuid(&mut self, value: String) {
        if self.core().short_uuid.is_some() {
            panic!("Should not be re-setting uuid on entity");
        }
        self.core_mut().short_uuid = Some(value);
    }

    fn subscribe(&mut self, o: Rc<RefCell<dyn EntityObserver>>) {
        self.core_mut().subscribers.push(o);
    }

    fn subscribe_to(&mut self, name: &str, o: Rc<RefCell<dyn EntityObserver>>) {
        self.core_mut()
            .subscribers_by_event
            .entry(name.to_string())
            .or_default()
            .push(o);
    }

    fn subscribe_with_frequency(&mut self, _count: i64, _o: Rc<RefCell<dyn EntityObserver>>) {
        // TS returned `this`; here a no-op (frequency batching unimplemented).
    }

    fn unsubscribe(&mut self, o: &Rc<RefCell<dyn EntityObserver>>) -> bool {
        let subs = &mut self.core_mut().subscribers;
        if let Some(pos) = subs.iter().position(|s| Rc::ptr_eq(s, o)) {
            subs.remove(pos);
            true
        } else {
            false
        }
    }

    fn send_update_to_subs(&mut self, type_: &str, v: &dyn Any) {
        // Clone the Rc handles so we are not borrowing `self.core()` while each
        // observer is mutated.
        let subs = self.core().subscribers.clone();
        for s in subs {
            s.borrow_mut().do_update(type_, v);
        }
    }
}

/// `class EntityConnection<S, T>` — a directed edge between a source `HasOutput`
/// and a target `HasInput`.
///
/// PORT NOTE: in TS this `extends Entity`. Here it is a lightweight edge struct
/// (no `EntityCore`/`Entity` impl) because the graph traits hand it back as a
/// concrete `Rc<RefCell<EntityConnection>>`, never as `dyn Entity`. The back-edge
/// to the source is a `Weak` to avoid a reference cycle; the `<unknown>null`
/// placeholders become `Option`.
pub struct EntityConnection {
    pub id: String,
    pub source: Option<Weak<RefCell<dyn HasOutput>>>,
    pub target: Option<Rc<RefCell<dyn HasInput>>>,
    pub opts: ConnectionOpts,
}

/// `opts` bag for a connection (currently empty; defaults only).
#[derive(Clone, Debug, Default)]
pub struct ConnectionOpts {}

impl EntityConnection {
    /// `new EntityConnection(source, target, opts?)`. The id is the last 10 chars
    /// of a fresh v4 UUID (`uuid.v4().slice(-10)`).
    pub fn new(source: Weak<RefCell<dyn HasOutput>>, target: Rc<RefCell<dyn HasInput>>) -> Self {
        let full = uuid::Uuid::new_v4().to_string();
        let id = full[full.len() - 10..].to_string();
        EntityConnection {
            id,
            source: Some(source),
            target: Some(target),
            opts: ConnectionOpts::default(),
        }
    }

    /// `getTarget()` — warns (to stderr) when the edge has no wired target.
    pub fn get_target(&self) -> Option<Rc<RefCell<dyn HasInput>>> {
        if self.target.is_none() {
            eprintln!(
                "[connection:{}] getTarget() returned null — connection has no target wired; downstream routing will skip it.",
                self.id
            );
        }
        self.target.clone()
    }

    /// `getSerializableData()` — the source/target ids, with `opts` dropped.
    pub fn get_serializable_data(&self) -> String {
        let source_id = self
            .source
            .as_ref()
            .and_then(|w| w.upgrade())
            .map(|rc| rc.borrow().id())
            .unwrap_or_default();
        let target_id = self
            .target
            .as_ref()
            .map(|rc| rc.borrow().id())
            .unwrap_or_default();
        format!(
            "{{\"id\":\"{}\",\"source\":\"{}\",\"target\":\"{}\"}}",
            self.id, source_id, target_id
        )
    }

    pub fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default()
    }

    /// `runTimeStep` — noop (a connection carries no state per tick).
    pub fn run_time_step(&mut self, _step_size: Decimal) {}

    pub fn do_time_step(&mut self, step_size: Decimal) {
        self.run_time_step(step_size);
    }

    pub fn do_validation(&mut self) {}

    pub fn do_validation_before_run(&mut self) -> bool {
        false
    }
}

impl HasId for EntityConnection {
    fn id(&self) -> String {
        self.id.clone()
    }
}

/// `abstract class StationaryEntity<E> extends Entity` — a fixed node in the
/// network. Adds post-wiring setup hooks on top of [`Entity`].
pub trait StationaryEntity: Entity {
    fn do_setup_after_input_conn(&mut self) -> bool;
    fn do_setup_after_output_conn(&mut self) -> bool;
}

/// Field-bag for `abstract class AbstractBidirectionalEntity` — a stationary
/// node holding both inbound and outbound connections.
///
/// PORT NOTE: the TS `addInConnection`/`addOutConnection` built an
/// `EntityConnection(source, this)` referencing `this`. A struct cannot mint an
/// `Rc` to itself, so connection *construction* is deferred to the concrete
/// entity (which owns its `Rc<RefCell<Self>>`); these helpers just push/clone the
/// resulting connection handles.
pub struct BidirectionalCore {
    pub entity: EntityCore,
    pub connections_in: Vec<Rc<RefCell<EntityConnection>>>,
    pub connections_out: Vec<Rc<RefCell<EntityConnection>>>,
}

impl BidirectionalCore {
    pub fn new(id: String) -> Self {
        BidirectionalCore {
            entity: EntityCore::new(id),
            connections_in: Vec::new(),
            connections_out: Vec::new(),
        }
    }

    pub fn add_in_connection(&mut self, conn: Rc<RefCell<EntityConnection>>) {
        self.connections_in.push(conn);
    }

    pub fn add_out_connection(&mut self, conn: Rc<RefCell<EntityConnection>>) {
        self.connections_out.push(conn);
    }

    pub fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.connections_out.clone()
    }

    pub fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.connections_in.clone()
    }

    /// `acceptItem` — TS always accepted (TODO: reject when full).
    pub fn accept_item(&self) -> bool {
        true
    }

    pub fn notify_sources(&self) {}
    pub fn notify_targets(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::precision::bgn;

    /// A minimal concrete entity to exercise the `Entity` default methods.
    struct Dummy {
        core: EntityCore,
        validated: bool,
    }

    impl Dummy {
        fn new(id: &str) -> Self {
            Dummy {
                core: EntityCore::new(id.to_string()),
                validated: false,
            }
        }
    }

    impl Entity for Dummy {
        fn core(&self) -> &EntityCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut EntityCore {
            &mut self.core
        }
        fn do_validation(&mut self) {
            self.validated = true;
        }
        fn do_validation_before_run(&mut self) -> bool {
            true
        }
        fn get_graph_data(&self) -> EntityGraphData {
            EntityGraphData::default()
        }
        fn run_time_step(&mut self, _step_size: Decimal) {
            self.core.time_step_count += 1;
        }
    }

    struct CountingObserver {
        count: usize,
    }
    impl EntityObserver for CountingObserver {
        fn do_update(&mut self, _type_: &str, _payload: &dyn Any) {
            self.count += 1;
        }
    }

    #[test]
    fn entity_defaults_work() {
        let mut d = Dummy::new("e1");
        assert_eq!(d.id(), "e1");
        assert_eq!(d.short_uuid(), None);
        d.set_short_uuid("abc".into());
        assert_eq!(d.short_uuid(), Some("abc".to_string()));
        d.do_time_step(bgn(0.1));
        assert_eq!(d.core().time_step_count, 1);
        assert!(d.do_validation_before_run());
    }

    #[test]
    #[should_panic(expected = "re-setting uuid")]
    fn short_uuid_cannot_be_reset() {
        let mut d = Dummy::new("e2");
        d.set_short_uuid("first".into());
        d.set_short_uuid("second".into());
    }

    #[test]
    fn subscribe_and_notify() {
        let mut d = Dummy::new("e3");
        let obs = Rc::new(RefCell::new(CountingObserver { count: 0 }));
        let obs_dyn: Rc<RefCell<dyn EntityObserver>> = obs.clone();
        d.subscribe(obs_dyn.clone());
        d.send_update_to_subs("tick", &1u32);
        assert_eq!(obs.borrow().count, 1);
        assert!(d.unsubscribe(&obs_dyn));
    }
}
