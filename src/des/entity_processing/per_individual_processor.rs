//! Canonical use path: `crate::des::entity_processing::per_individual_processor::*`
//!
//! Port of `src/des/entity-processing/per-individual-processor.ts` — a FEL-style
//! per-individual service-time processor (M/M/inf-ish): each token draws its own
//! residence time on arrival, counts down each tick, and routes downstream when
//! it reaches zero.
//!
//! TS `class PerIndividualProcessor extends AbstractBidirectionalEntity
//! implements HasComputedProperties, HasInternalQueue`. Modelled as a struct
//! embedding [`BidirectionalCore`].
//!
//! PORT NOTES:
//!   * `drawDuration: () => number` is a CLOSURE -> `Box<dyn FnMut() -> f64>`.
//!   * the real store is the parallel `items: Vec<QueuedItem>`; the TS `queue`
//!     (a `LinkedQueue` facade for `HasInternalQueue`) is dropped — `is_empty`
//!     reports on `items`. `items.unshift(item)` retry -> `Vec::insert(0, ..)`.
//!   * `m instanceof ProcessableMovingEntity || (m as any).startNewStation` was a
//!     runtime type test + dynamic method probe; Rust has no `instanceof`, so the
//!     trait-available visit bookkeeping (`stations_visited_count`,
//!     `add_visited_station`) is applied unconditionally. `startNewStation` lives
//!     on the concrete `ProcessableMovingEntity` only and is omitted.
//!   * `['isProcessor'] = true` brand -> the [`ProcessorTag`] marker trait.
//!   * `reg.registerProcessor(this)` is omitted in the constructor (no self-`Rc`).
//!   * `Number(stepSize)` BigNumber->f64 becomes `to_f64`; output routing uses an
//!     [`OutputConnectionRouter`] by INDEX (no `PartialEq` on the edge handle).

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_processing::processing::ProcessorTag;
use crate::des::entity_queue::queue::{build_in_conn, build_out_conn};
use crate::des::entity_routing::output_routing_policy::{
    OutputConnectionRouter, OutputRoutingPolicy,
};
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasInternalQueue, HasManyInputConnections, HasManyOutputConnections,
    HasOutput,
};
use crate::des::r#abstract::r#abstract::{
    BidirectionalCore, Entity, EntityCore, EntityConnection,
};
use crate::des::random_variables::rv::RandomVariable;
use crate::des::shared::precision::{to_f64, Decimal};

/// `interface QueuedItem` — a token plus its remaining residence time.
pub struct QueuedItem {
    pub entity: Rc<RefCell<dyn MovingEntity>>,
    pub remaining_time: f64,
}

/// `interface PerIndividualProcessorOpts`.
pub struct PerIndividualProcessorOpts {
    /// Per-individual residence-time draw (independent sample each call).
    pub draw_duration: Box<dyn FnMut() -> f64>,
    /// Optional RV shim (for registry compatibility; unused by the kernel).
    pub rv: Option<Box<dyn RandomVariable>>,
    pub output_routing: Option<OutputRoutingPolicy>,
}

/// `class PerIndividualProcessor`.
pub struct PerIndividualProcessor {
    pub bi: BidirectionalCore,
    pub items: Vec<QueuedItem>,
    pub max_queue_size: i64,
    pub opts: PerIndividualProcessorOpts,
    pub output_router: OutputConnectionRouter,
}

impl ProcessorTag for PerIndividualProcessor {}

impl PerIndividualProcessor {
    /// `new(id, opts)`.
    pub fn new(id: String, opts: PerIndividualProcessorOpts) -> Self {
        let policy = opts.output_routing.unwrap_or_default();
        PerIndividualProcessor {
            bi: BidirectionalCore::new(id),
            items: Vec::new(),
            max_queue_size: -1,
            opts,
            output_router: OutputConnectionRouter::new(policy),
        }
    }

    /// `doAudit()` -> `{ totalSize }`.
    pub fn do_audit(&self) -> usize {
        self.items.len()
    }
}

impl Entity for PerIndividualProcessor {
    fn core(&self) -> &EntityCore {
        &self.bi.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.bi.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default().with("processedCount", self.items.len() as f64)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default().with("items.size", self.items.len() as f64)
    }
    fn run_time_step(&mut self, step_size: Decimal) {
        self.bi.entity.time_step_count += 1;
        let dt = to_f64(step_size);

        // Decrement remaining times; partition into ready / still-waiting.
        let mut ready: Vec<QueuedItem> = Vec::new();
        let mut still: Vec<QueuedItem> = Vec::new();
        for mut item in std::mem::take(&mut self.items) {
            item.remaining_time -= dt;
            if item.remaining_time <= 0.0 {
                ready.push(item);
            } else {
                still.push(item);
            }
        }
        self.items = still;

        // Route the finished tokens via the configured policy.
        for item in ready {
            let connections = self.bi.get_out_connections();
            if connections.is_empty() {
                eprintln!(
                    "[per-individual:{}] entity finished service but station has no out-connections; it will loop back.",
                    self.bi.entity.id
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
                if target.borrow_mut().accept_item(item.entity.clone()) {
                    if let Some(ix) = connections.iter().position(|c| Rc::ptr_eq(c, conn)) {
                        self.output_router.mark_accepted_index(connections.len(), ix);
                    }
                    target.borrow_mut().take_item(item.entity.clone());
                    routed = true;
                    break;
                }
            }
            if !routed {
                // Nobody accepted — retry next step (front of the queue).
                let mut it = item;
                it.remaining_time = 0.0;
                self.items.insert(0, it);
            }
        }
    }
}

impl HasInternalQueue for PerIndividualProcessor {
    fn max_queue_size(&self) -> usize {
        if self.max_queue_size < 0 {
            usize::MAX
        } else {
            self.max_queue_size as usize
        }
    }
    fn is_full(&self) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl HasInput for PerIndividualProcessor {
    fn id(&self) -> String {
        self.bi.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        true
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        let id = self.bi.entity.id.clone();
        {
            let mut mb = m.borrow_mut();
            mb.moving_core_mut().stations_visited_count += 1;
            // startNewStation: ProcessableMovingEntity-only -> omitted (see module note).
            mb.add_visited_station(&id);
        }
        let d = (self.opts.draw_duration)();
        self.items.push(QueuedItem {
            entity: m,
            remaining_time: d,
        });
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_sources(&mut self) {
        self.bi.notify_sources();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
    fn add_in_connection(
        &mut self,
        _source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        let conn = build_in_conn();
        self.bi.add_in_connection(conn.clone());
        Some(conn)
    }
}

impl HasManyInputConnections for PerIndividualProcessor {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_in_connections()
    }
}

impl HasOutput for PerIndividualProcessor {
    fn id(&self) -> String {
        self.bi.entity.id.clone()
    }
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        let conn = build_out_conn(target);
        self.bi.add_out_connection(conn.clone());
        Some(conn)
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_targets(&mut self) {
        self.bi.notify_targets();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl HasManyOutputConnections for PerIndividualProcessor {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_out_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::ProcessableMovingEntity;
    use crate::des::shared::precision::bgn;

    fn opts(duration: f64) -> PerIndividualProcessorOpts {
        PerIndividualProcessorOpts {
            draw_duration: Box::new(move || duration),
            rv: None,
            output_routing: None,
        }
    }

    #[test]
    fn take_item_draws_a_duration() {
        let mut p = PerIndividualProcessor::new("pi1".to_string(), opts(2.0));
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(ProcessableMovingEntity::new()));
        p.take_item(m.clone());
        assert_eq!(p.items.len(), 1);
        assert_eq!(p.items[0].remaining_time, 2.0);
        assert_eq!(m.borrow().moving_core().stations_visited_count, 1);
    }

    #[test]
    fn run_step_counts_down_and_retries_when_unrouted() {
        let mut p = PerIndividualProcessor::new("pi2".to_string(), opts(0.05));
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(ProcessableMovingEntity::new()));
        p.take_item(m);
        // dt = 0.1 > 0.05 -> ready, but no out-connections -> re-queued at remaining 0.
        p.run_time_step(bgn(0.1));
        assert_eq!(p.items.len(), 1);
        assert_eq!(p.items[0].remaining_time, 0.0);
        assert_eq!(p.bi.entity.time_step_count, 1);
    }
}
