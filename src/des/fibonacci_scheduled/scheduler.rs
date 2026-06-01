//! Canonical use path: `crate::des::fibonacci_scheduled::scheduler::*`
//!
//! A small **deterministic scheduler** — the coordination "brain" / *enforcer*
//! for order-sensitive, time-stepped DES graphs.
//!
//! ## Why this exists
//!
//! The original `main_fibonacci_recursion` stepped its nodes by iterating a
//! hand-written `Vec<dyn Entity>` in *insertion order* (`A, B, C, D`). That
//! works, but the correctness of the whole simulation rests on an **implicit,
//! unchecked invariant**: the splitter `C` must be stepped *after* the processor
//! `B` so the `C → B` feedback lands in the same tick. Reorder the vector, insert
//! a node, or add an edge and the recurrence silently breaks — nothing complains.
//!
//! This scheduler makes that ordering **explicit, derived, and validated**:
//!
//! * The caller declares the graph as nodes plus two edge kinds —
//!   [`wire_forward`](DeterministicScheduler::wire_forward) (intra-tick dataflow)
//!   and [`wire_feedback`](DeterministicScheduler::wire_feedback) (cross-tick).
//! * [`freeze`](DeterministicScheduler::freeze) computes a **topological order**
//!   of the forward edges (deterministic tie-break by registration index), proves
//!   that the forward graph is a DAG, and proves every feedback edge is a genuine
//!   *back* edge. The resulting order is frozen as the per-tick execution order.
//! * [`step`](DeterministicScheduler::step) advances every node exactly once per
//!   tick, in that frozen order — no statistics, no RNG, fully reproducible.
//!
//! Wiring goes *through* the scheduler, so the topology used to derive the order
//! is the same topology that is physically connected on the entities — the two
//! cannot drift apart.

#![allow(dead_code)]

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::rc::Rc;

use crate::des::r#abstract::interfaces::{HasInput, HasOutput};
use crate::des::r#abstract::r#abstract::Entity;
use crate::des::shared::precision::Decimal;

/// One node under the scheduler's control: its stable id, the steppable trait
/// object, and the execution `rank` assigned by [`DeterministicScheduler::freeze`].
struct ScheduledNode {
    id: String,
    entity: Rc<RefCell<dyn Entity>>,
    rank: usize,
}

/// A declared edge (by node id). The role (forward vs. feedback) is implied by
/// which list it lives in.
#[derive(Clone, Debug)]
struct Edge {
    from: String,
    to: String,
}

/// The deterministic enforcer. Build it, register nodes, declare edges, then
/// [`freeze`](Self::freeze) (validate + lock the order) and [`run`](Self::run).
pub struct DeterministicScheduler {
    /// Registered nodes. After `freeze`, re-sorted into execution (rank) order.
    nodes: Vec<ScheduledNode>,
    /// `id -> position in nodes`.
    index: HashMap<String, usize>,
    /// Intra-tick dataflow edges. MUST form a DAG (data flows "downhill" within a
    /// single tick).
    forward: Vec<Edge>,
    /// Cross-tick edges. These are the *only* legal cycles: a value emitted now
    /// is consumed on the next tick.
    feedback: Vec<Edge>,
    /// Fixed simulation step handed to every `do_time_step`.
    step_size: Decimal,
    /// Set by `freeze`; gates `step`/`run` and forbids further mutation.
    frozen: bool,
    /// Completed-tick counter.
    tick: u64,
    /// The frozen execution order (ids), exposed for inspection / debugging.
    order: Vec<String>,
}

impl DeterministicScheduler {
    /// New empty scheduler with a fixed per-tick step size.
    pub fn new(step_size: Decimal) -> Self {
        DeterministicScheduler {
            nodes: Vec::new(),
            index: HashMap::new(),
            forward: Vec::new(),
            feedback: Vec::new(),
            step_size,
            frozen: false,
            tick: 0,
            order: Vec::new(),
        }
    }

    /// Register a steppable node. The id is read from the entity itself; ids must
    /// be unique. Panics if called after [`freeze`](Self::freeze) or on a
    /// duplicate id (an order-sensitive system must never silently merge nodes).
    pub fn register(&mut self, entity: Rc<RefCell<dyn Entity>>) {
        assert!(!self.frozen, "cannot register nodes after freeze()");
        let id = entity.borrow().id();
        if self.index.contains_key(&id) {
            panic!("duplicate node id registered: {id}");
        }
        let idx = self.nodes.len();
        self.index.insert(id.clone(), idx);
        self.nodes.push(ScheduledNode {
            id,
            entity,
            rank: usize::MAX,
        });
    }

    /// Declare an **intra-tick** dataflow edge `from → to` AND physically connect
    /// the entities (`from.add_out_connection(to)`). Both endpoints must already
    /// be registered. Forward edges must remain acyclic (checked at `freeze`).
    pub fn wire_forward(
        &mut self,
        from: &Rc<RefCell<dyn HasOutput>>,
        to: &Rc<RefCell<dyn HasInput>>,
    ) {
        self.wire(from, to, false);
    }

    /// Declare a **cross-tick** feedback edge `from → to` AND physically connect
    /// the entities. The value travels in the same `add_out_connection` channel,
    /// but the scheduler records it as feedback so the cycle it introduces does
    /// not violate the forward-DAG requirement.
    pub fn wire_feedback(
        &mut self,
        from: &Rc<RefCell<dyn HasOutput>>,
        to: &Rc<RefCell<dyn HasInput>>,
    ) {
        self.wire(from, to, true);
    }

    fn wire(
        &mut self,
        from: &Rc<RefCell<dyn HasOutput>>,
        to: &Rc<RefCell<dyn HasInput>>,
        feedback: bool,
    ) {
        assert!(!self.frozen, "cannot wire edges after freeze()");
        let from_id = from.borrow().id();
        let to_id = to.borrow().id();
        assert!(
            self.index.contains_key(&from_id),
            "wire source is not a registered node: {from_id}"
        );
        assert!(
            self.index.contains_key(&to_id),
            "wire target is not a registered node: {to_id}"
        );
        assert!(
            from_id != to_id,
            "self-edges are not allowed ({from_id} -> {from_id})"
        );

        // Physically connect the entities, so the recorded topology and the real
        // wiring are the same object graph.
        from.borrow_mut().add_out_connection(to.clone());

        let edge = Edge {
            from: from_id,
            to: to_id,
        };
        if feedback {
            self.feedback.push(edge);
        } else {
            self.forward.push(edge);
        }
    }

    /// Validate the declared graph and lock in the execution order. Panics with a
    /// descriptive message on any violation (fail fast — an order-sensitive model
    /// must never run with a broken schedule). Use [`try_freeze`](Self::try_freeze)
    /// for a recoverable variant.
    pub fn freeze(&mut self) {
        if let Err(e) = self.try_freeze() {
            panic!("scheduler validation failed: {e}");
        }
    }

    /// Like [`freeze`](Self::freeze) but returns the error instead of panicking.
    ///
    /// Validates, in order:
    /// 1. at least one node is registered;
    /// 2. the forward edges form a DAG (topological sort succeeds) — otherwise an
    ///    intra-tick edge would require a node to run before itself;
    /// 3. every feedback edge is a true back edge (`rank(from) >= rank(to)`),
    ///    i.e. it was correctly classified as cross-tick, not forward.
    pub fn try_freeze(&mut self) -> Result<(), String> {
        if self.frozen {
            return Err("scheduler is already frozen".to_string());
        }
        if self.nodes.is_empty() {
            return Err("no nodes registered".to_string());
        }

        let order = self.topological_order()?;
        let rank_of: HashMap<String, usize> = order
            .iter()
            .enumerate()
            .map(|(rank, id)| (id.clone(), rank))
            .collect();

        for e in &self.feedback {
            let rf = rank_of[&e.from];
            let rt = rank_of[&e.to];
            if rf < rt {
                return Err(format!(
                    "edge {}->{} was declared feedback but is actually a forward edge \
                     (rank {rf} < {rt}); declare it with wire_forward",
                    e.from, e.to
                ));
            }
        }

        for node in &mut self.nodes {
            node.rank = rank_of[&node.id];
        }
        self.nodes.sort_by_key(|n| n.rank);
        self.index = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.clone(), i))
            .collect();
        self.order = order;
        self.frozen = true;
        Ok(())
    }

    /// Kahn's algorithm over the forward edges, with a deterministic tie-break:
    /// when several nodes are ready (in-degree 0) the one with the smallest
    /// registration index is emitted first. This makes the order a pure function
    /// of the declared graph — never of hash iteration or insertion timing.
    fn topological_order(&self) -> Result<Vec<String>, String> {
        let n = self.nodes.len();
        let mut indegree = vec![0usize; n];
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];

        for e in &self.forward {
            let fi = self.index[&e.from];
            let ti = self.index[&e.to];
            successors[fi].push(ti);
            indegree[ti] += 1;
        }

        // Min-heap on registration index for a deterministic ready-set order.
        let mut ready: BinaryHeap<Reverse<usize>> =
            (0..n).filter(|&i| indegree[i] == 0).map(Reverse).collect();

        let mut order: Vec<String> = Vec::with_capacity(n);
        while let Some(Reverse(i)) = ready.pop() {
            order.push(self.nodes[i].id.clone());
            let mut succ = successors[i].clone();
            succ.sort_unstable();
            for j in succ {
                indegree[j] -= 1;
                if indegree[j] == 0 {
                    ready.push(Reverse(j));
                }
            }
        }

        if order.len() != n {
            let stuck: Vec<String> = (0..n)
                .filter(|&i| indegree[i] > 0)
                .map(|i| self.nodes[i].id.clone())
                .collect();
            return Err(format!(
                "forward edges contain a cycle among {stuck:?}; intra-tick dataflow must be \
                 acyclic — declare cross-tick edges with wire_feedback"
            ));
        }
        Ok(order)
    }

    /// Advance the whole network by exactly one tick: step every node once, in the
    /// frozen execution order. Panics if not yet frozen.
    pub fn step(&mut self) {
        assert!(self.frozen, "freeze() must be called before stepping");
        let step_size = self.step_size;
        for node in &self.nodes {
            node.entity.borrow_mut().do_time_step(step_size);
        }
        self.tick += 1;
    }

    /// Run `ticks` ticks.
    pub fn run(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.step();
        }
    }

    /// The frozen per-tick execution order (node ids). Empty until `freeze`.
    pub fn execution_order(&self) -> Vec<String> {
        self.order.clone()
    }

    /// Number of completed ticks.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Whether the schedule has been validated and locked.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::MovingEntity;
    use crate::des::r#abstract::interfaces::{EntityGraphData, HasManyOutputConnections};
    use crate::des::r#abstract::r#abstract::{EntityConnection, EntityCore};
    use crate::des::shared::precision::bgn;

    /// Minimal inert node: enough trait surface to be registered and wired. It
    /// performs no dataflow, so these tests isolate the *ordering* logic.
    struct Dot {
        core: EntityCore,
    }
    impl Dot {
        fn new(id: &str) -> Rc<RefCell<Dot>> {
            Rc::new(RefCell::new(Dot {
                core: EntityCore::new(id.to_string()),
            }))
        }
    }
    impl Entity for Dot {
        fn core(&self) -> &EntityCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut EntityCore {
            &mut self.core
        }
        fn do_validation(&mut self) {}
        fn do_validation_before_run(&mut self) -> bool {
            false
        }
        fn get_graph_data(&self) -> EntityGraphData {
            EntityGraphData::default()
        }
        fn run_time_step(&mut self, _step_size: Decimal) {
            self.core.time_step_count += 1;
        }
    }
    impl HasOutput for Dot {
        fn id(&self) -> String {
            self.core.id.clone()
        }
        fn add_out_connection(
            &mut self,
            _target: Rc<RefCell<dyn HasInput>>,
        ) -> Option<Rc<RefCell<EntityConnection>>> {
            None
        }
        fn do_setup_after_input_conn(&mut self) -> bool {
            true
        }
        fn notify_targets(&mut self) {}
        fn do_setup_after_output_conn(&mut self) -> bool {
            true
        }
    }
    impl HasManyOutputConnections for Dot {
        fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
            Vec::new()
        }
    }
    impl HasInput for Dot {
        fn id(&self) -> String {
            self.core.id.clone()
        }
        fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
            true
        }
        fn take_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) {}
        fn do_setup_after_input_conn(&mut self) -> bool {
            true
        }
        fn notify_sources(&mut self) {}
        fn do_setup_after_output_conn(&mut self) -> bool {
            true
        }
        fn add_in_connection(
            &mut self,
            _source: Rc<RefCell<dyn HasManyOutputConnections>>,
        ) -> Option<Rc<RefCell<EntityConnection>>> {
            None
        }
    }

    fn views(
        n: &Rc<RefCell<Dot>>,
    ) -> (
        Rc<RefCell<dyn Entity>>,
        Rc<RefCell<dyn HasOutput>>,
        Rc<RefCell<dyn HasInput>>,
    ) {
        (n.clone(), n.clone(), n.clone())
    }

    #[test]
    fn linear_chain_orders_topologically() {
        let mut s = DeterministicScheduler::new(bgn(1.0));
        let a = Dot::new("a");
        let b = Dot::new("b");
        let c = Dot::new("c");
        // Register out of dependency order to prove the order is DERIVED.
        let (ce, _co, ci) = views(&c);
        let (be, bo, bi) = views(&b);
        let (ae, ao, _ai) = views(&a);
        s.register(ce);
        s.register(ae);
        s.register(be);
        s.wire_forward(&ao, &bi);
        s.wire_forward(&bo, &ci);
        s.freeze();
        assert_eq!(s.execution_order(), vec!["a", "b", "c"]);
    }

    #[test]
    fn diamond_tie_breaks_on_registration_index() {
        // a -> b, a -> c, b -> d, c -> d. b and c are both ready after a; the
        // smaller registration index (b) must come first, deterministically.
        let mut s = DeterministicScheduler::new(bgn(1.0));
        let a = Dot::new("a");
        let b = Dot::new("b");
        let c = Dot::new("c");
        let d = Dot::new("d");
        let (ae, ao, _) = views(&a);
        let (be, bo, bi) = views(&b);
        let (ce, co, ci) = views(&c);
        let (de, _do, di) = views(&d);
        s.register(ae);
        s.register(be);
        s.register(ce);
        s.register(de);
        s.wire_forward(&ao, &bi);
        s.wire_forward(&ao, &ci);
        s.wire_forward(&bo, &di);
        s.wire_forward(&co, &di);
        s.freeze();
        assert_eq!(s.execution_order(), vec!["a", "b", "c", "d"]);
        s.run(3);
        assert_eq!(s.tick(), 3);
    }

    #[test]
    fn feedback_back_edge_is_accepted() {
        // a -> b forward, b -> a feedback: legal (the cycle is cross-tick).
        let mut s = DeterministicScheduler::new(bgn(1.0));
        let a = Dot::new("a");
        let b = Dot::new("b");
        let (ae, ao, ai) = views(&a);
        let (be, bo, bi) = views(&b);
        s.register(ae);
        s.register(be);
        s.wire_forward(&ao, &bi);
        s.wire_feedback(&bo, &ai);
        assert!(s.try_freeze().is_ok());
        assert_eq!(s.execution_order(), vec!["a", "b"]);
    }

    #[test]
    fn forward_cycle_is_rejected() {
        // a -> b and b -> a both forward: no valid intra-tick order exists.
        let mut s = DeterministicScheduler::new(bgn(1.0));
        let a = Dot::new("a");
        let b = Dot::new("b");
        let (ae, ao, ai) = views(&a);
        let (be, bo, bi) = views(&b);
        s.register(ae);
        s.register(be);
        s.wire_forward(&ao, &bi);
        s.wire_forward(&bo, &ai);
        let err = s.try_freeze().unwrap_err();
        assert!(err.contains("cycle"), "unexpected error: {err}");
    }

    #[test]
    #[should_panic(expected = "duplicate node id")]
    fn duplicate_ids_panic() {
        let mut s = DeterministicScheduler::new(bgn(1.0));
        let a1 = Dot::new("a");
        let a2 = Dot::new("a");
        s.register(a1);
        s.register(a2);
    }

    #[test]
    #[should_panic(expected = "freeze() must be called")]
    fn stepping_before_freeze_panics() {
        let mut s = DeterministicScheduler::new(bgn(1.0));
        let a = Dot::new("a");
        s.register(a);
        s.step();
    }
}
