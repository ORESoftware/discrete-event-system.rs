//! Canonical use path: `crate::des::general::entity_registration::*`
//!
//! Port of `src/des/general/entity-registration.ts` — the process-wide registry
//! of all sources / sinks / processors / decision nodes.
//!
//! The TS `reg` was a module-level mutable singleton of four `Set<EntityX>`s. We
//! mirror that singleton with a `thread_local!` [`Registry`] and free functions
//! (`register_*` / `get_all_*`) that operate on it. Code that wants an isolated
//! registry can also own a [`Registry`] directly.
//!
//! PORT NOTE: the concrete entity types (`EntitySource`/`EntitySink`/
//! `EntityProcessor`/`ProbabilityDecisionEntity`) are not ported yet, so each
//! category stores type-erased `Rc<RefCell<dyn HasId>>` handles. Narrow these to
//! the concrete entity traits once those modules exist. `get_all_*` returns a
//! snapshot clone of the `Rc` handles (cheap pointer clones) to avoid handing out
//! a borrow of the thread-local.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::r#abstract::interfaces::HasId;

/// A registered entity handle (type-erased to its identity).
pub type EntityHandle = Rc<RefCell<dyn HasId>>;

/// The registry of all registered network entities, by category.
#[derive(Default)]
pub struct Registry {
    all_processors: Vec<EntityHandle>,
    all_sources: Vec<EntityHandle>,
    all_sinks: Vec<EntityHandle>,
    all_decision: Vec<EntityHandle>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_all_decision_nodes(&self) -> Vec<EntityHandle> {
        self.all_decision.clone()
    }
    pub fn get_all_sources(&self) -> Vec<EntityHandle> {
        self.all_sources.clone()
    }
    pub fn get_all_sinks(&self) -> Vec<EntityHandle> {
        self.all_sinks.clone()
    }
    pub fn get_all_processors(&self) -> Vec<EntityHandle> {
        self.all_processors.clone()
    }

    /// Register if no handle with the same id is present (`Set.add` semantics).
    fn add_unique(list: &mut Vec<EntityHandle>, v: EntityHandle) {
        let id = v.borrow().id();
        if !list.iter().any(|e| e.borrow().id() == id) {
            list.push(v);
        }
    }

    pub fn register_source(&mut self, v: EntityHandle) {
        Self::add_unique(&mut self.all_sources, v);
    }
    pub fn register_sink(&mut self, v: EntityHandle) {
        Self::add_unique(&mut self.all_sinks, v);
    }
    pub fn register_processor(&mut self, v: EntityHandle) {
        Self::add_unique(&mut self.all_processors, v);
    }
    pub fn register_decision(&mut self, v: EntityHandle) {
        Self::add_unique(&mut self.all_decision, v);
    }
}

thread_local! {
    /// The `reg` singleton (faithful 1:1 of the TS module-level `vals` + `reg`).
    static REGISTRY: RefCell<Registry> = RefCell::new(Registry::new());
}

// ── free-function facade mirroring `reg.<method>(...)` ───────────────────────

pub fn register_source(v: EntityHandle) {
    REGISTRY.with(|r| r.borrow_mut().register_source(v));
}
pub fn register_sink(v: EntityHandle) {
    REGISTRY.with(|r| r.borrow_mut().register_sink(v));
}
pub fn register_processor(v: EntityHandle) {
    REGISTRY.with(|r| r.borrow_mut().register_processor(v));
}
pub fn register_decision(v: EntityHandle) {
    REGISTRY.with(|r| r.borrow_mut().register_decision(v));
}

pub fn get_all_sources() -> Vec<EntityHandle> {
    REGISTRY.with(|r| r.borrow().get_all_sources())
}
pub fn get_all_sinks() -> Vec<EntityHandle> {
    REGISTRY.with(|r| r.borrow().get_all_sinks())
}
pub fn get_all_processors() -> Vec<EntityHandle> {
    REGISTRY.with(|r| r.borrow().get_all_processors())
}
pub fn get_all_decision_nodes() -> Vec<EntityHandle> {
    REGISTRY.with(|r| r.borrow().get_all_decision_nodes())
}

/// Clear the global registry (test/runner setup; the TS relied on module
/// re-evaluation between runs).
pub fn reset_registry() {
    REGISTRY.with(|r| *r.borrow_mut() = Registry::new());
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

    #[test]
    fn owned_registry_dedups() {
        let mut reg = Registry::new();
        let a: EntityHandle = Rc::new(RefCell::new(Node { id: "a".into() }));
        let a2: EntityHandle = Rc::new(RefCell::new(Node { id: "a".into() }));
        let b: EntityHandle = Rc::new(RefCell::new(Node { id: "b".into() }));
        reg.register_source(a);
        reg.register_source(a2); // same id -> not added
        reg.register_source(b);
        assert_eq!(reg.get_all_sources().len(), 2);
    }

    #[test]
    fn thread_local_registry_roundtrip() {
        reset_registry();
        let p: EntityHandle = Rc::new(RefCell::new(Node { id: "proc".into() }));
        register_processor(p);
        assert_eq!(get_all_processors().len(), 1);
        assert_eq!(get_all_processors()[0].borrow().id(), "proc");
        reset_registry();
        assert_eq!(get_all_processors().len(), 0);
    }
}
