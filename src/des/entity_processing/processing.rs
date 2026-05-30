//! Canonical use path: `crate::des::entity_processing::processing::*`
//!
//! Port of `src/des/entity-processing/processing.ts` — the multi-server queueing
//! processor with input / processing / output stages.
//!
//! TS `class EntityProcessor extends QueueEntity implements
//! HasEntityValidation`. Rust has no inheritance, so this COMPOSES a
//! [`QueueEntity`] (`base`); the input queue IS `base.queue` (the TS aliased
//! `inputQueue = this.queue`), and the processing / out queues are separate
//! `VecDeque`s.
//!
//! PORT NOTES:
//!   * BRANDING: the `processorSymbol` / `isProcessor` duck-typed symbol becomes
//!     the [`ProcessorTag`] marker trait (shared with `value_adder`).
//!   * `reg.registerProcessor(this)` is omitted in the constructor (no self-`Rc`);
//!     the integrator registers after wrapping in an `Rc`.
//!   * `rv.getNextEventQuantity(stepSize)` draws from an injected
//!     `RandomVariable`; output routing uses an [`OutputConnectionRouter`].
//!   * The keyed `LinkedQueue.remove(k)` flush of the outQueue is replaced by a
//!     drain-and-rebuild over a `VecDeque` (the std `VecDeque` has no
//!     remove-by-key). Routing order/markAccepted go through the router by INDEX
//!     (the `Rc<RefCell<EntityConnection>>` element has no `PartialEq`).
//!   * BIG SIMPLIFICATION: the per-entity timing calls
//!     (`startNewStation`, `setTimeInInputQueue`, `setStartTimeInProcessQueue`,
//!     `setTimeInProcessingQueue`, `setStartTimeInOutputQueue`,
//!     `setTimeInOutputQueue`, `bumpTotalWaitTime`, `bumpOutQueueWaitTime`,
//!     `bumpTotalProcessingTime`) live on the concrete `ProcessableMovingEntity`,
//!     NOT on the object-safe `MovingEntity` trait, and the framework provides no
//!     downcast. They are therefore OMITTED here, which also empties the
//!     `*QueueTimeHistogram`s. The trait-available `bump_time_in_system` /
//!     `add_visited_station` and `stations_visited_count` (via `moving_core_mut`)
//!     ARE applied. The size histograms, server busy/idle accounting, evq-driven
//!     service, routing and input→processing promotion are all preserved.
//!   * histograms use plain `HashMap<i64, Decimal>` (the migration header's
//!     `DESMap<number, BigNumber>` mapping); `getWithComputedProperties` /
//!     `getGraphData` return `EntityGraphData` payloads.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_queue::queue::{QueueEntity, QueueEntityGraphData, QueueOpts};
use crate::des::entity_routing::output_routing_policy::{
    OutputConnectionRouter, OutputRoutingPolicy,
};
use crate::des::general::time_accrued::get_step_size;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasEntityValidation, HasInput, HasInternalQueue, HasManyInputConnections,
    HasManyOutputConnections, HasOutput,
};
use crate::des::r#abstract::r#abstract::{Entity, EntityCore, EntityConnection};
use crate::des::random_variables::rv::RandomVariable;
use crate::des::shared::iterable_int::IterableInt;
use crate::des::shared::precision::{bgn, to_f64, Decimal};

/// Marker for "this entity is a processor" (replaces the TS `processorSymbol`
/// duck-type brand). Shared with `value_adder::EntityNumericProcessor`.
pub trait ProcessorTag {}

/// `interface ProcessorEntityGraphData extends QueueEntityGraphData`.
#[derive(Clone, Debug, Default)]
pub struct ProcessorEntityGraphData {
    pub base: QueueEntityGraphData,
}

/// `bumpHistogram(key, m)` — `+1` at `key` (creating the bucket at `1`).
fn bump_histogram(key: i64, m: &mut HashMap<i64, Decimal>) {
    match m.get(&key).copied() {
        None => {
            m.insert(key, bgn(1.0));
        }
        Some(v) => {
            m.insert(key, v + bgn(1.0));
        }
    }
}

/// `class EntityProcessor`.
pub struct EntityProcessor {
    pub base: QueueEntity,
    pub concurrency: usize,
    pub processing_queue: VecDeque<Rc<RefCell<dyn MovingEntity>>>,
    pub out_queue: VecDeque<Rc<RefCell<dyn MovingEntity>>>,
    pub total_server_busy_time: Decimal,
    pub total_server_idle_time: Decimal,
    pub processed_count: i64,
    pub input_queue_histogram: HashMap<i64, Decimal>,
    pub processing_queue_histogram: HashMap<i64, Decimal>,
    pub output_queue_histogram: HashMap<i64, Decimal>,
    pub input_queue_time_histogram: HashMap<i64, Decimal>,
    pub processing_queue_time_histogram: HashMap<i64, Decimal>,
    pub output_queue_time_histogram: HashMap<i64, Decimal>,
    pub rv: Box<dyn RandomVariable>,
    pub output_router: OutputConnectionRouter,
}

impl ProcessorTag for EntityProcessor {}

impl EntityProcessor {
    /// `new(id, { rv, outputRouting = 'random' })`.
    pub fn new(id: String, rv: Box<dyn RandomVariable>, output_routing: OutputRoutingPolicy) -> Self {
        EntityProcessor {
            base: QueueEntity::new(id, QueueOpts { xx: Some(true) }),
            concurrency: 5,
            processing_queue: VecDeque::new(),
            out_queue: VecDeque::new(),
            total_server_busy_time: bgn(0.0),
            total_server_idle_time: bgn(0.0),
            processed_count: 0,
            input_queue_histogram: HashMap::new(),
            processing_queue_histogram: HashMap::new(),
            output_queue_histogram: HashMap::new(),
            input_queue_time_histogram: HashMap::new(),
            processing_queue_time_histogram: HashMap::new(),
            output_queue_time_histogram: HashMap::new(),
            rv,
            output_router: OutputConnectionRouter::new(output_routing),
        }
    }

    /// `getServerUtilization()` — busy / (busy + idle), guarding divide-by-zero.
    pub fn get_server_utilization(&self) -> Decimal {
        let total = self.total_server_busy_time + self.total_server_idle_time;
        if total == Decimal::ZERO {
            Decimal::ZERO
        } else {
            self.total_server_busy_time / total
        }
    }

    fn add_to_input_histogram(&mut self, time_step: Decimal) {
        let size = self.base.queue.len() as i64;
        let e = self.input_queue_histogram.entry(size).or_insert(bgn(0.0));
        *e += time_step;
    }

    fn add_to_processing_histogram(&mut self, time_step: Decimal) {
        let size = self.processing_queue.len() as i64;
        let e = self.processing_queue_histogram.entry(size).or_insert(bgn(0.0));
        *e += time_step;
    }

    fn add_to_output_histogram(&mut self, time_step: Decimal) {
        let size = self.out_queue.len() as i64;
        let e = self.output_queue_histogram.entry(size).or_insert(bgn(0.0));
        *e += time_step;
    }

    fn bump_total_server_busy_time(&mut self, step_size: Decimal) {
        self.total_server_busy_time += step_size;
    }

    fn bump_total_server_idle_time(&mut self, step_size: Decimal) {
        self.total_server_idle_time += step_size;
    }

    /// `getKeyForHistogram(t)` — `floor((t + 1) / stepSize)`.
    ///
    /// PORT NOTE: only meaningful when a per-entity time `t` is available, which
    /// requires the omitted `ProcessableMovingEntity` timing calls; kept for
    /// API parity.
    pub fn get_key_for_histogram(&self, t: Decimal) -> i64 {
        let step = get_step_size();
        if step == Decimal::ZERO {
            return 0;
        }
        to_f64((t + bgn(1.0)) / step).floor() as i64
    }

    pub fn get_queue_sizes(&self) -> (usize, usize, usize) {
        (
            self.base.queue.len(),
            self.out_queue.len(),
            self.processing_queue.len(),
        )
    }

    /// `doAudit()` -> total across all internal queues.
    pub fn do_audit(&self) -> usize {
        let (i, o, p) = self.get_queue_sizes();
        i + o + p
    }
}

impl HasEntityValidation for EntityProcessor {
    fn validate(&self) -> bool {
        true
    }
}

impl Entity for EntityProcessor {
    fn core(&self) -> &EntityCore {
        &self.base.bi.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.base.bi.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        false
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("processedCount", self.processed_count as f64)
            .with("timeStepCount", self.base.bi.entity.time_step_count as f64)
            .with("serverUtilization", to_f64(self.get_server_utilization()))
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("serverUtilization", to_f64(self.get_server_utilization()))
            .with("inputQueue.size", self.base.queue.len() as f64)
            .with("processingQueue.size", self.processing_queue.len() as f64)
            .with("outQueue.size", self.out_queue.len() as f64)
            .with("processedCount", self.processed_count as f64)
    }
    fn run_time_step(&mut self, step_size: Decimal) {
        self.base.bi.entity.time_step_count += 1;
        self.add_to_input_histogram(step_size);
        self.add_to_processing_histogram(step_size);
        self.add_to_output_histogram(step_size);

        let gd = self.get_graph_data();
        self.send_update_to_subs("GRAPH_DATA:PROCESSING", &gd);

        // Idle / busy server-time accounting. `s` may be negative (more in
        // service than servers); IterableInt(0, s<=0) yields nothing.
        let s = self.concurrency as i64 - self.processing_queue.len() as i64;
        for _ in IterableInt::new(0, s) {
            self.bump_total_server_idle_time(step_size);
        }
        let busy = self.processing_queue.len() as i64;
        for _ in IterableInt::new(0, busy) {
            self.bump_total_server_busy_time(step_size);
        }

        // ── flush the outQueue ──────────────────────────────────────────────
        let out_items: Vec<Rc<RefCell<dyn MovingEntity>>> = self.out_queue.drain(..).collect();
        let mut out_remaining: VecDeque<Rc<RefCell<dyn MovingEntity>>> = VecDeque::new();
        for v in out_items {
            v.borrow_mut().bump_time_in_system(step_size);
            // bumpTotalWaitTime / bumpOutQueueWaitTime: ProcessableMovingEntity-only -> omitted.

            let connections = self.base.bi.get_out_connections();
            if connections.is_empty() {
                eprintln!(
                    "[processor:{}] outQueue flush: entity has no out-connections; item will remain stuck.",
                    self.base.bi.entity.id
                );
            }
            let ordered = self.output_router.order(&connections);
            let mut routed = false;
            for conn in &ordered {
                let target = conn.borrow().get_target();
                let target = match target {
                    Some(t) => t,
                    None => continue,
                };
                if target.borrow_mut().accept_item(v.clone()) {
                    if let Some(ix) = connections.iter().position(|c| Rc::ptr_eq(c, conn)) {
                        self.output_router.mark_accepted_index(connections.len(), ix);
                    }
                    // setTimeInOutputQueue + bumpHistogram: omitted (see module note).
                    target.borrow_mut().take_item(v.clone());
                    routed = true;
                    break;
                }
            }
            if !routed {
                out_remaining.push_back(v);
            }
        }
        self.out_queue = out_remaining;

        // ── advance time on items in service ────────────────────────────────
        for v in self.processing_queue.iter() {
            v.borrow_mut().bump_time_in_system(step_size);
            // bumpTotalProcessingTime: ProcessableMovingEntity-only -> omitted.
        }

        // ── service completions for this tick ───────────────────────────────
        let evq = self.rv.get_next_event_quantity(step_size);
        for _i in 0..evq {
            let next = match self.processing_queue.pop_front() {
                Some(x) => x,
                None => break,
            };
            // setTimeInProcessingQueue + bumpHistogram: omitted (see module note).
            self.processed_count += 1;

            let connections = self.base.bi.get_out_connections();
            if connections.is_empty() {
                eprintln!(
                    "[processor:{}] processed item but has zero out-connections — buffering in outQueue.",
                    self.base.bi.entity.id
                );
            }
            let ordered = self.output_router.order(&connections);
            let mut handled = false;
            for conn in &ordered {
                let target = conn.borrow().get_target();
                let target = match target {
                    Some(t) => t,
                    None => continue,
                };
                if target.borrow_mut().accept_item(next.clone()) {
                    if let Some(ix) = connections.iter().position(|c| Rc::ptr_eq(c, conn)) {
                        self.output_router.mark_accepted_index(connections.len(), ix);
                    }
                    target.borrow_mut().take_item(next.clone());
                } else {
                    // first resolvable target rejected -> buffer in outQueue.
                    // setStartTimeInOutputQueue: omitted (see module note).
                    self.out_queue.push_back(next.clone());
                }
                handled = true;
                break;
            }
            if !handled {
                // zero (resolvable) connections — buffer to preserve the token.
                self.out_queue.push_back(next);
            }
        }

        // ── promote input -> processing while capacity remains ──────────────
        let in_items: Vec<Rc<RefCell<dyn MovingEntity>>> = self.base.queue.drain(..).collect();
        let mut in_kept: VecDeque<Rc<RefCell<dyn MovingEntity>>> = VecDeque::new();
        for z in in_items {
            z.borrow_mut().bump_time_in_system(step_size);
            // bumpTotalWaitTime: ProcessableMovingEntity-only -> omitted.
            if self.processing_queue.len() < self.concurrency {
                // setTimeInInputQueue + bumpHistogram + setStartTimeInProcessQueue: omitted.
                self.processing_queue.push_back(z);
            } else {
                in_kept.push_back(z);
            }
        }
        self.base.queue = in_kept;
    }
}

impl HasInternalQueue for EntityProcessor {
    fn max_queue_size(&self) -> usize {
        HasInternalQueue::max_queue_size(&self.base)
    }
    fn is_full(&self) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        false
    }
}

impl HasInput for EntityProcessor {
    fn id(&self) -> String {
        self.base.bi.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        true
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        let id = self.base.bi.entity.id.clone();
        {
            let mut mb = m.borrow_mut();
            mb.moving_core_mut().stations_visited_count += 1;
            // startNewStation: ProcessableMovingEntity-only -> omitted (see module note).
            mb.add_visited_station(&id);
        }
        if self.processing_queue.len() < self.concurrency {
            // setTimeInInputQueue + bumpHistogram + setStartTimeInProcessQueue: omitted.
            self.processing_queue.push_back(m);
            return;
        }
        self.base.queue.push_back(m);
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_sources(&mut self) {
        self.base.bi.notify_sources();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
    fn add_in_connection(
        &mut self,
        source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        HasInput::add_in_connection(&mut self.base, source)
    }
}

impl HasManyInputConnections for EntityProcessor {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.base.bi.get_in_connections()
    }
}

impl HasOutput for EntityProcessor {
    fn id(&self) -> String {
        self.base.bi.entity.id.clone()
    }
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        HasOutput::add_out_connection(&mut self.base, target)
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_targets(&mut self) {
        self.base.bi.notify_targets();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl HasManyOutputConnections for EntityProcessor {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.base.bi.get_out_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::ProcessableMovingEntity;
    use crate::des::random_variables::rv::BernoulliRandomVariable;
    use crate::des::shared::capabilities::SeededRandom;
    use crate::des::general::time_accrued::{reset_global_clock, set_step_size};

    fn rv() -> Box<dyn RandomVariable> {
        Box::new(BernoulliRandomVariable::new(Box::new(SeededRandom::new(5))))
    }

    #[test]
    fn take_item_and_run_step() {
        reset_global_clock();
        set_step_size(bgn(0.1));
        let mut p = EntityProcessor::new("proc1".to_string(), rv(), OutputRoutingPolicy::Random);
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(ProcessableMovingEntity::new()));
        p.take_item(m.clone());
        // under concurrency, the item promotes straight into the processing queue.
        assert_eq!(p.processing_queue.len(), 1);
        assert_eq!(m.borrow().moving_core().stations_visited_count, 1);

        p.run_time_step(bgn(0.1));
        assert_eq!(p.base.bi.entity.time_step_count, 1);
        // idle/busy accounting accumulated some time this tick.
        assert!(p.total_server_busy_time + p.total_server_idle_time > bgn(0.0));
    }

    #[test]
    fn empty_processor_runs_without_panic() {
        reset_global_clock();
        set_step_size(bgn(0.1));
        let mut p = EntityProcessor::new("proc2".to_string(), rv(), OutputRoutingPolicy::Ordered);
        p.run_time_step(bgn(0.1));
        // all 5 servers idle, none busy.
        assert!(p.total_server_idle_time > bgn(0.0));
        assert_eq!(p.total_server_busy_time, bgn(0.0));
        assert!(p.validate());
    }
}
