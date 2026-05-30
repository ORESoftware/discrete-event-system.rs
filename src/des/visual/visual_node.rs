//! Canonical use path: `crate::des::visual::visual_node::*`
//!
//! Port of `src/des/visual/visual-node.ts` — visualization wrapper nodes that
//! mirror the entity graph for the UI layer.
//!
//! Conversion notes (file-specific):
//!   * `VisualNodeEvents` is a STRING enum (`FOO = 'FOO'`). With no `serde`
//!     dependency in the foundation (matching `abstract/abstract.rs`), it is a
//!     plain enum exposing [`VisualNodeEvents::as_str`]; reinstate
//!     `#[serde(rename_all)]` when serialization is ported.
//!   * PORT NOTE: the ctor wired `subscription` to a CLOSURE that captures `this`
//!     and fans out to `this.subscribers`. A closure that borrows the owning node
//!     back is a self-referential `Rc` cycle in Rust. The fan-out INTENT is
//!     preserved as the inherent method [`VisualNode::fan_out`]; the stored
//!     `subscription` keeps a standalone (non-capturing) [`VisualNodeObserver`].
//!   * PORT NOTE: `VisualNodeObserver`'s ctor did `e.subscribe(this)` (subscribe
//!     self to the entity) — an `Rc`-to-self; that auto-subscription is deferred.
//!   * `fn: ((type,m)=>void|null) = null as any` -> `Option<Box<dyn FnMut(..)>>`.
//!   * PORT NOTE: `connectionsOut/In: Map<VisualNode, VisualConnection>` keyed by
//!     NODE IDENTITY -> `HashMap<String, VisualConnection>` keyed by a node id
//!     (each [`VisualNode`] mints a short uuid `id`); `VisualConnection` stores
//!     the endpoint ids rather than node refs, since `addVisualConnection*` uses
//!     `this` as the source (an `Rc`-to-self) and a whole struct is the wrong
//!     map key.
//!   * `sub()` built an anonymous `new class extends EntityObserver {..}` -> the
//!     closure-backed adapter [`FnObserver`].
//!   * `subscribeTo`/`subscribeWithFrequency`/`sendUpdateToSubs` `throw` ->
//!     `panic!` (the `IsObservable` impl mirrors the TS throws).
//!   * `subscribers: Set<EntityObserver>` / `subscribersByEvent: Map<..>` ->
//!     `Vec<Rc<RefCell<dyn EntityObserver>>>` / `HashMap<String, Vec<..>>`.
//!   * The empty `OneInOneOut`/… subclasses are arity markers -> newtype wrappers
//!     around [`VisualNode`]; `ManyInManyOut` is an independent empty struct.

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::general::general::get_short_uuid;
use crate::des::r#abstract::interfaces::IsObservable;
use crate::des::r#abstract::r#abstract::{EntityObserver, StationaryEntity};

/// `class VisualNodeObserver extends EntityObserver<any>` — adapts a callback
/// into an [`EntityObserver`].
pub struct VisualNodeObserver {
    /// `entity: StationaryEntity<any>`.
    pub entity: Rc<RefCell<dyn StationaryEntity>>,
    /// `fn: ((type, m) => void) = null as any`.
    pub fn_: Option<Box<dyn FnMut(&str, &dyn Any)>>,
}

impl VisualNodeObserver {
    /// `constructor(e, fn) { super(); e.subscribe(this); entity = e; fn = fn; }`.
    /// See module PORT NOTE: the `e.subscribe(this)` self-registration is deferred.
    pub fn new(entity: Rc<RefCell<dyn StationaryEntity>>, fn_: Box<dyn FnMut(&str, &dyn Any)>) -> Self {
        VisualNodeObserver {
            entity,
            fn_: Some(fn_),
        }
    }
}

impl EntityObserver for VisualNodeObserver {
    /// `doUpdate(type, m) { this.fn(type, m); }`.
    fn do_update(&mut self, type_: &str, payload: &dyn Any) {
        if let Some(f) = self.fn_.as_mut() {
            f(type_, payload);
        }
    }
}

/// `class FnObserver` — the anonymous `EntityObserver` `sub()` builds inline.
pub struct FnObserver {
    pub f: Box<dyn FnMut(&str, &dyn Any)>,
}

impl EntityObserver for FnObserver {
    fn do_update(&mut self, type_: &str, payload: &dyn Any) {
        (self.f)(type_, payload);
    }
}

/// `class VisualConnection` — a directed edge between two visual nodes.
///
/// PORT NOTE: the TS held `source`/`target` as `VisualNode` refs; here the
/// endpoints are node ids (see module note) to avoid an `Rc`-to-self.
#[derive(Clone, Debug)]
pub struct VisualConnection {
    pub source: String,
    pub target: String,
}

impl VisualConnection {
    pub fn new(source: String, target: String) -> Self {
        VisualConnection { source, target }
    }
}

/// `enum VisualNodeEvents { FOO = 'FOO' }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VisualNodeEvents {
    Foo,
}

impl VisualNodeEvents {
    /// The string value of the enum (`FOO`).
    pub fn as_str(&self) -> &'static str {
        match self {
            VisualNodeEvents::Foo => "FOO",
        }
    }
}

/// `class VisualNode<T> implements IsObservable`.
pub struct VisualNode {
    /// Stable identity for connection map keys (see module PORT NOTE).
    pub id: String,
    /// `entity: StationaryEntity<any>`.
    pub entity: Rc<RefCell<dyn StationaryEntity>>,
    pub label: String,
    /// `iconUrl` (a URL).
    pub icon_url: String,
    /// `subscription: EntityObserver<any>` — standalone observer (see PORT NOTE).
    pub subscription: VisualNodeObserver,
    /// `subscribersByEvent = new Map()`.
    pub subscribers_by_event: HashMap<String, Vec<Rc<RefCell<dyn EntityObserver>>>>,
    /// `subscribers = new Set<EntityObserver<any>>()`.
    pub subscribers: Vec<Rc<RefCell<dyn EntityObserver>>>,
    /// `connectionsOut = new Map<VisualNode, VisualConnection>()`.
    pub connections_out: HashMap<String, VisualConnection>,
    /// `connectionsIn = new Map<VisualNode, VisualConnection>()`.
    pub connections_in: HashMap<String, VisualConnection>,
}

/// `constructor(v: { label, iconUrl, entity })` argument bag.
pub struct VisualNodeArgs {
    pub label: String,
    pub icon_url: String,
    pub entity: Rc<RefCell<dyn StationaryEntity>>,
}

impl VisualNode {
    pub fn new(v: VisualNodeArgs) -> Self {
        // The TS subscription's callback fanned out to `this.subscribers`; that
        // self-reference is replaced by `fan_out` (see PORT NOTE). The stored
        // observer keeps a non-capturing no-op callback.
        let subscription = VisualNodeObserver::new(v.entity.clone(), Box::new(|_t, _m| {}));
        VisualNode {
            id: get_short_uuid(),
            entity: v.entity,
            label: v.label,
            icon_url: v.icon_url,
            subscription,
            subscribers_by_event: HashMap::new(),
            subscribers: Vec::new(),
            connections_out: HashMap::new(),
            connections_in: HashMap::new(),
        }
    }

    /// `doValidationBeforeRun(): boolean { return true; }`.
    pub fn do_validation_before_run(&self) -> bool {
        true
    }

    /// The fan-out the TS subscription closure performed: deliver `(type, m)` to
    /// every subscriber. Subscriber handles are snapshotted so no `RefCell`
    /// borrow is held across `do_update`.
    pub fn fan_out(&mut self, type_: &str, m: &dyn Any) {
        let subs = self.subscribers.clone();
        for s in subs {
            s.borrow_mut().do_update(type_, m);
        }
    }

    /// `sub(fn)` — wrap a callback in an [`FnObserver`], subscribe it, and return
    /// the handle (TS returned `this`; the handle is more useful for `unsubscribe`).
    pub fn sub(&mut self, f: Box<dyn FnMut(&str, &dyn Any)>) -> Rc<RefCell<dyn EntityObserver>> {
        let o: Rc<RefCell<dyn EntityObserver>> = Rc::new(RefCell::new(FnObserver { f }));
        self.subscribers.push(o.clone());
        o
    }

    /// `addVisualConnectionOut(target)`.
    pub fn add_visual_connection_out(&mut self, target: &VisualNode) {
        self.connections_out.insert(
            target.id.clone(),
            VisualConnection::new(self.id.clone(), target.id.clone()),
        );
    }

    /// `addVisualConnectionIn(target)`.
    pub fn add_visual_connection_in(&mut self, target: &VisualNode) {
        self.connections_in.insert(
            target.id.clone(),
            VisualConnection::new(target.id.clone(), self.id.clone()),
        );
    }
}

impl IsObservable for VisualNode {
    /// `subscribeTo(name, o): this { throw new Error("Method not implemented."); }`.
    fn subscribe_to(&mut self, _name: &str, _o: Rc<RefCell<dyn EntityObserver>>) {
        panic!("Method not implemented.");
    }

    /// `subscribe(o): this { this.subscribers.add(o); return this; }`.
    fn subscribe(&mut self, o: Rc<RefCell<dyn EntityObserver>>) {
        if !self.subscribers.iter().any(|s| Rc::ptr_eq(s, &o)) {
            self.subscribers.push(o);
        }
    }

    /// `subscribeWithFrequency(count, o): this { throw new Error(...); }`.
    fn subscribe_with_frequency(&mut self, _count: i64, _o: Rc<RefCell<dyn EntityObserver>>) {
        panic!("Method not implemented.");
    }

    /// `unsubscribe(o): boolean` — remove by identity.
    fn unsubscribe(&mut self, o: &Rc<RefCell<dyn EntityObserver>>) -> bool {
        if let Some(pos) = self.subscribers.iter().position(|s| Rc::ptr_eq(s, o)) {
            self.subscribers.remove(pos);
            true
        } else {
            false
        }
    }

    /// `sendUpdateToSubs(type, v): void { throw new Error("Method not implemented."); }`.
    fn send_update_to_subs(&mut self, _type_: &str, _v: &dyn Any) {
        panic!("Method not implemented.");
    }
}

// =============================================================================
// Arity-marker subtypes (`extends VisualNode`) — newtype wrappers.
// =============================================================================

/// `class OneInOneOut extends VisualNode {}`.
pub struct OneInOneOut(pub VisualNode);
/// `class OneInManyOut extends VisualNode {}`.
pub struct OneInManyOut(pub VisualNode);
/// `class ZeroInManyOut extends VisualNode {}`.
pub struct ZeroInManyOut(pub VisualNode);
/// `class ZeroOutManyIn extends VisualNode {}`.
pub struct ZeroOutManyIn(pub VisualNode);

/// `class ManyInManyOut {}` — an independent empty class (no `extends`).
#[derive(Clone, Copy, Debug, Default)]
pub struct ManyInManyOut;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::r#abstract::interfaces::EntityGraphData;
    use crate::des::r#abstract::r#abstract::{Entity, EntityCore};
    use crate::des::shared::precision::Decimal;

    /// Minimal stationary entity to back a visual node.
    struct TestStation {
        core: EntityCore,
    }
    impl Entity for TestStation {
        fn core(&self) -> &EntityCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut EntityCore {
            &mut self.core
        }
        fn do_validation(&mut self) {}
        fn do_validation_before_run(&mut self) -> bool {
            true
        }
        fn get_graph_data(&self) -> EntityGraphData {
            EntityGraphData::default()
        }
        fn run_time_step(&mut self, _step_size: Decimal) {}
    }
    impl StationaryEntity for TestStation {
        fn do_setup_after_input_conn(&mut self) -> bool {
            false
        }
        fn do_setup_after_output_conn(&mut self) -> bool {
            false
        }
    }

    /// An observer that counts the updates it receives.
    struct CountingObserver {
        count: usize,
    }
    impl EntityObserver for CountingObserver {
        fn do_update(&mut self, _type_: &str, _payload: &dyn Any) {
            self.count += 1;
        }
    }

    fn make_node(label: &str) -> VisualNode {
        let entity: Rc<RefCell<dyn StationaryEntity>> = Rc::new(RefCell::new(TestStation {
            core: EntityCore::new(label.to_string()),
        }));
        VisualNode::new(VisualNodeArgs {
            label: label.to_string(),
            icon_url: "icon.png".to_string(),
            entity,
        })
    }

    #[test]
    fn subscribe_fan_out_and_unsubscribe() {
        let mut node = make_node("n1");
        assert!(node.do_validation_before_run());

        let obs = Rc::new(RefCell::new(CountingObserver { count: 0 }));
        let obs_dyn: Rc<RefCell<dyn EntityObserver>> = obs.clone();
        node.subscribe(obs_dyn.clone());
        // duplicate subscribe is ignored (Set semantics)
        node.subscribe(obs_dyn.clone());

        node.fan_out("FOO", &1u32);
        assert_eq!(obs.borrow().count, 1);

        assert!(node.unsubscribe(&obs_dyn));
        node.fan_out("FOO", &1u32);
        assert_eq!(obs.borrow().count, 1); // no longer notified
    }

    #[test]
    fn visual_connections_keyed_by_node_id() {
        let mut a = make_node("a");
        let b = make_node("b");
        a.add_visual_connection_out(&b);
        a.add_visual_connection_in(&b);
        assert!(a.connections_out.contains_key(&b.id));
        assert!(a.connections_in.contains_key(&b.id));
        let out = &a.connections_out[&b.id];
        assert_eq!(out.source, a.id);
        assert_eq!(out.target, b.id);
    }

    #[test]
    fn enum_string_value() {
        assert_eq!(VisualNodeEvents::Foo.as_str(), "FOO");
    }

    #[test]
    #[should_panic(expected = "Method not implemented.")]
    fn subscribe_to_panics() {
        let mut node = make_node("n2");
        let obs: Rc<RefCell<dyn EntityObserver>> = Rc::new(RefCell::new(CountingObserver { count: 0 }));
        node.subscribe_to("FOO", obs);
    }
}
