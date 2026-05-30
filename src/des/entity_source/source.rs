//! Canonical use path: `crate::des::entity_source::source::*`
//!
//! Port of `src/des/entity-source/source.ts` — source entities that inject new
//! moving-entities into the network.
//!
//! TS chain: `AbstractSourceEntity extends Entity implements
//! HasComputedProperties, HasManyOutputConnections`; concrete `EntitySource`
//! (Poisson-ish, draws an RV per tick) and `DefiniteFiniteSource` (drains a
//! finite list of `initialValues`). Sources are OUTPUT-ONLY nodes, so they hold
//! `connections_out` and implement `HasOutput` / `HasManyOutputConnections` but
//! not `HasInput`.
//!
//! PORT NOTES:
//!   * `<S, T>` generics were `any` and are erased; the `addOutConnection<T>`
//!     method-generic that SHADOWED the class `T` is dropped.
//!   * `(global as any).turnOffSources` ambient flag is DROPPED — there is no
//!     process global; only the per-source `turn_off_after_count` guard remains.
//!   * `reg.registerSource(this)` is NOT done in the constructor: a constructor
//!     returns `Self` and has no `Rc<RefCell<Self>>` to register. The integrator
//!     registers the source after wrapping it in an `Rc`.
//!   * `rv.getNextEventQuantity(stepSize)` draws from an injected
//!     `RandomVariable` (which itself carries a `RandomSource`); no `Math.random`.
//!   * `[util.inspect.custom]` debug hook and `getCleanVersion()` collapse into
//!     `get_with_computed_properties` returning an `EntityGraphData` payload.
//!   * `LinkedQueue`/`Array` buffers become `VecDeque`/`Vec` of
//!     `Rc<RefCell<dyn MovingEntity>>`; the `DefiniteFiniteSource` `getNextValue`
//!     `[V,V]` tuple becomes `Option<HasNumericValue>`.
//!   * `throw new Error('needs initial values.')` -> `panic!` in the constructor.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::des::entity_moving::moving::{
    BasicQuantityMovingEntity, MovingEntity, ProcessableMovingEntity,
};
use crate::des::entity_queue::queue::build_out_conn;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasManyOutputConnections, HasOutput,
};
use crate::des::r#abstract::r#abstract::{
    Entity, EntityCore, EntityConnection, HasNumericValue,
};
use crate::des::random_variables::rv::RandomVariable;
use crate::des::shared::precision::Decimal;

/// Field-bag for `abstract class AbstractSourceEntity` — a source's outbound
/// edge set, atop the shared [`EntityCore`].
pub struct SourceCore {
    pub entity: EntityCore,
    pub connections_out: Vec<Rc<RefCell<EntityConnection>>>,
}

impl SourceCore {
    pub fn new(id: String) -> Self {
        SourceCore {
            entity: EntityCore::new(id),
            connections_out: Vec::new(),
        }
    }

    /// `addOutConnection(target)` — build an out-edge to `target` and store it.
    pub fn add_out_connection_to(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Rc<RefCell<EntityConnection>> {
        let conn = build_out_conn(target);
        self.connections_out.push(conn.clone());
        conn
    }

    pub fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.connections_out.clone()
    }
}

/// `abstract class AbstractSourceEntity` — behaviour marker (imported by the sink
/// modules as the connection's source type).
pub trait AbstractSourceEntity: Entity {}

// =============================================================================
// EntitySource
// =============================================================================

/// `class EntitySource` — draws `rv.getNextEventQuantity` new tokens per tick.
pub struct EntitySource {
    pub core: SourceCore,
    pub rv: Box<dyn RandomVariable>,
    pub created_count: i64,
    /// `opts.turnOffAfterCount` (default `-1` = never turn off).
    pub turn_off_after_count: i64,
    /// Backpressure buffer for tokens no downstream accepted this tick.
    pub queue: Vec<Rc<RefCell<dyn MovingEntity>>>,
}

impl EntitySource {
    /// `new(id, { rv, turnOffAfterCount = -1 })`.
    pub fn new(id: String, rv: Box<dyn RandomVariable>, turn_off_after_count: i64) -> Self {
        EntitySource {
            core: SourceCore::new(id),
            rv,
            created_count: 0,
            turn_off_after_count,
            queue: Vec::new(),
        }
    }

    /// `checkIfSourcesOff()` — only the per-source count guard (see module note).
    pub fn check_if_sources_off(&self) -> bool {
        if self.turn_off_after_count > 0 && self.created_count > self.turn_off_after_count {
            return true;
        }
        false
    }
}

impl AbstractSourceEntity for EntitySource {}

impl Entity for EntitySource {
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
        // TS returned { id, createdCount }; the id stays on the entity, the
        // numeric payload carries createdCount.
        EntityGraphData::default().with("createdCount", self.created_count as f64)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("createdCount", self.created_count as f64)
            .with("connectionsOut.size", self.core.connections_out.len() as f64)
            .with("queue.size", self.queue.len() as f64)
    }
    fn run_time_step(&mut self, step_size: Decimal) {
        self.core.entity.time_step_count += 1;

        if self.check_if_sources_off() {
            return;
        }

        let num_events = self.rv.get_next_event_quantity(step_size);
        let conns = self.core.connections_out.clone();

        for _ in 0..num_events {
            if self.check_if_sources_off() {
                break;
            }

            let mut e = ProcessableMovingEntity::new();
            e.init();
            let next: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(e));

            let mid = next.borrow().id();
            self.send_update_to_subs("NEW_BASIC_MOVING_ENTITY", &mid);
            self.created_count += 1;

            let mut accepted = false;
            for conn in &conns {
                let target = conn.borrow().get_target();
                let target = match target {
                    Some(t) => t,
                    None => {
                        eprintln!(
                            "[source:{}] out-connection has no resolvable target; cannot route newly-created entity.",
                            self.core.entity.id
                        );
                        continue;
                    }
                };
                accepted = target.borrow_mut().accept_item(next.clone());
                if accepted {
                    target.borrow_mut().take_item(next.clone());
                    break;
                }
            }

            if !accepted {
                self.queue.push(next);
            }
        }

        let gd = self.get_graph_data();
        self.send_update_to_subs("GRAPH_DATA:SOURCE", &gd);
    }
}

impl HasOutput for EntitySource {
    fn id(&self) -> String {
        self.core.entity.id.clone()
    }
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        Some(self.core.add_out_connection_to(target))
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_targets(&mut self) {}
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl HasManyOutputConnections for EntitySource {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.core.get_out_connections()
    }
}

// =============================================================================
// DefiniteFiniteSource
// =============================================================================

/// `class DefiniteFiniteSource` — emits a fixed list of numeric values, one per
/// tick, until drained.
pub struct DefiniteFiniteSource {
    pub core: SourceCore,
    pub created_count: i64,
    pub turn_off_after_count: i64,
    /// Remaining input values, front = next to emit.
    pub queue: VecDeque<HasNumericValue>,
    /// Backpressure buffer of already-minted tokens awaiting a downstream.
    pub out_queue: VecDeque<Rc<RefCell<dyn MovingEntity>>>,
}

impl DefiniteFiniteSource {
    /// `new(id, { initialValues, turnOffAfterCount = -1 })`.
    ///
    /// Panics if `initial_values` is empty (TS `throw new Error('needs initial
    /// values.')`). Values are pushed front-first (TS popped from the back and
    /// `addToFront`-ed) to preserve emission order.
    pub fn new(
        id: String,
        mut initial_values: Vec<HasNumericValue>,
        turn_off_after_count: i64,
    ) -> Self {
        if initial_values.is_empty() {
            panic!("needs initial values.");
        }
        let mut queue: VecDeque<HasNumericValue> = VecDeque::new();
        while let Some(val) = initial_values.pop() {
            queue.push_front(val);
        }
        DefiniteFiniteSource {
            core: SourceCore::new(id),
            created_count: 0,
            turn_off_after_count,
            queue,
            out_queue: VecDeque::new(),
        }
    }

    pub fn is_done_emitting(&self) -> bool {
        self.queue.is_empty()
    }

    /// `getNextValue()` — dequeue the next input value.
    pub fn get_next_value(&mut self) -> Option<HasNumericValue> {
        self.queue.pop_front()
    }

    pub fn check_if_sources_off(&self) -> bool {
        if self.turn_off_after_count > 0 && self.created_count > self.turn_off_after_count {
            return true;
        }
        false
    }
}

impl AbstractSourceEntity for DefiniteFiniteSource {}

impl Entity for DefiniteFiniteSource {
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
        EntityGraphData::default().with("createdCount", self.created_count as f64)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("createdCount", self.created_count as f64)
            .with("connectionsOut.size", self.core.connections_out.len() as f64)
            .with("queue.size", self.queue.len() as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        self.core.entity.time_step_count += 1;

        if self.check_if_sources_off() {
            return;
        }

        let conns = self.core.connections_out.clone();

        // Flush previously-untaken items. `while (count-- >= 0)` -> size+1 passes.
        let mut count = self.out_queue.len() as i64;
        while count >= 0 {
            count -= 1;
            let item = match self.out_queue.pop_front() {
                Some(x) => x,
                None => continue,
            };
            // PORT NOTE: TS tried only the FIRST out-connection then `break`-ed;
            // preserved here (re-enqueue at back on refusal).
            if let Some(conn) = conns.first() {
                let target = conn.borrow().get_target();
                if let Some(t) = target {
                    if t.borrow_mut().accept_item(item.clone()) {
                        t.borrow_mut().take_item(item.clone());
                    } else {
                        self.out_queue.push_back(item.clone());
                    }
                }
            }
        }

        if self.queue.is_empty() {
            return;
        }

        let k = match self.get_next_value() {
            Some(v) => v,
            None => {
                eprintln!(
                    "[finite-source:{}] getNextValue() returned void at step {}; initial values appear drained.",
                    self.core.entity.id, self.core.entity.time_step_count
                );
                return;
            }
        };

        let mut e = BasicQuantityMovingEntity::new(k.value);
        e.init();
        let next: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(e));

        let mid = next.borrow().id();
        self.send_update_to_subs("NEW_BASIC_MOVING_ENTITY", &mid);
        self.created_count += 1;

        let mut accepted = false;
        for conn in &conns {
            let target = conn.borrow().get_target();
            let target = match target {
                Some(t) => t,
                None => {
                    eprintln!(
                        "[finite-source:{}] out-connection has no resolvable target; cannot route emitted value.",
                        self.core.entity.id
                    );
                    continue;
                }
            };
            accepted = target.borrow_mut().accept_item(next.clone());
            if accepted {
                target.borrow_mut().take_item(next.clone());
                break;
            }
        }

        if !accepted {
            self.out_queue.push_back(next);
        }

        let gd = self.get_graph_data();
        self.send_update_to_subs("GRAPH_DATA:SOURCE", &gd);
    }
}

impl HasOutput for DefiniteFiniteSource {
    fn id(&self) -> String {
        self.core.entity.id.clone()
    }
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        Some(self.core.add_out_connection_to(target))
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_targets(&mut self) {}
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl HasManyOutputConnections for DefiniteFiniteSource {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.core.get_out_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::random_variables::rv::BernoulliRandomVariable;
    use crate::des::shared::capabilities::SeededRandom;
    use crate::des::shared::precision::bgn;

    fn rv() -> Box<dyn RandomVariable> {
        Box::new(BernoulliRandomVariable::new(Box::new(SeededRandom::new(9))))
    }

    #[test]
    fn entity_source_runs_a_step_and_buffers_unaccepted() {
        let mut s = EntitySource::new("src1".to_string(), rv(), -1);
        // No out-connections: any created tokens go to the backpressure queue.
        s.run_time_step(bgn(0.1));
        assert_eq!(s.core.entity.time_step_count, 1);
        // created_count == queue length when nothing downstream accepts.
        assert_eq!(s.created_count as usize, s.queue.len());
    }

    #[test]
    fn finite_source_drains_initial_values() {
        let vals = vec![
            HasNumericValue { value: 1.0 },
            HasNumericValue { value: 2.0 },
        ];
        let mut s = DefiniteFiniteSource::new("fin1".to_string(), vals, -1);
        assert!(!s.is_done_emitting());
        s.run_time_step(bgn(0.1)); // emits the first value (front = 1.0)
        assert_eq!(s.created_count, 1);
        s.run_time_step(bgn(0.1)); // emits the second
        assert!(s.is_done_emitting());
    }

    #[test]
    #[should_panic(expected = "needs initial values.")]
    fn finite_source_requires_initial_values() {
        let _ = DefiniteFiniteSource::new("fin2".to_string(), Vec::new(), -1);
    }
}
