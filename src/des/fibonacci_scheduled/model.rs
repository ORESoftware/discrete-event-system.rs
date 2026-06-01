//! Canonical use path: `crate::des::fibonacci_scheduled::model::*`
//!
//! A **scheduler-enforced** Fibonacci variant — independent of, and alongside,
//! `crate::des::main_fibonacci_recursion`. It reuses the same recognizable
//! building blocks (a finite source, the numeric processor, the broadcasting
//! splitter) but the per-tick execution order is no longer a hand-ordered `Vec`:
//! it is declared as a graph and **derived + validated** by the
//! [`DeterministicScheduler`].
//!
//! Graph (identical shape to the original):
//!
//! ```text
//!   A(source: 0,1) --fwd--> B(processor: pop+peek, emit sum) --fwd--> C(splitter)
//!                                ^                                       |
//!                                |                                  fwd  | --> D(sink, records)
//!                                +------------- feedback ----------------+
//! ```
//!
//! Forward edges (`A→B→C→D`) are intra-tick dataflow; the single `C→B` edge is
//! declared *feedback* (cross-tick). The scheduler topologically sorts the
//! forward DAG to get the order `A, B, C, D`, and verifies `C→B` is a genuine
//! back edge. The values absorbed at the sink are the Fibonacci numbers.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_processing::value_adder::EntityNumericProcessor;
use crate::des::entity_routing::entity_splitter::{EntitySplitter, SplitterOpts};
use crate::des::entity_sink::sink::{AbstractSinkEntity, SinkCore, SinkKind};
use crate::des::entity_source::source::DefiniteFiniteSource;
use crate::des::fibonacci_scheduled::scheduler::DeterministicScheduler;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasManyInputConnections, HasManyOutputConnections, HasOutput,
};
use crate::des::r#abstract::r#abstract::{Entity, EntityConnection, EntityCore, HasNumericValue};
use crate::des::shared::precision::{bgn, Decimal};

// =============================================================================
// RecordingSink — a sink that REMEMBERS each absorbed value.
// =============================================================================

/// Like [`crate::des::entity_sink::generic_sink::GenericEntitySink`], but instead
/// of only counting + printing it appends each token's numeric value to
/// `recorded`. This lets the model assert the produced sequence exactly — the
/// proof that the schedule is deterministic.
pub struct RecordingSink {
    pub core: SinkCore,
    pub kind: SinkKind,
    pub recorded: Vec<f64>,
}

impl RecordingSink {
    pub fn new(id: String) -> Self {
        RecordingSink {
            core: SinkCore::new(id),
            kind: SinkKind::Sink,
            recorded: Vec::new(),
        }
    }
}

impl AbstractSinkEntity for RecordingSink {}

impl Entity for RecordingSink {
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
        EntityGraphData::default()
            .with("recordedCount", self.recorded.len() as f64)
            .with("timeStepCount", self.core.entity.time_step_count as f64)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default().with("recordedCount", self.recorded.len() as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        self.core.entity.time_step_count += 1;
    }
}

impl HasInput for RecordingSink {
    fn id(&self) -> String {
        self.core.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        true
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        let q = m.borrow().get_value().q;
        if let Some(v) = q {
            self.recorded.push(v);
        }
        m.borrow_mut().do_finish();
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        false
    }
    fn notify_sources(&mut self) {}
    fn do_setup_after_output_conn(&mut self) -> bool {
        false
    }
    fn add_in_connection(
        &mut self,
        source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        Some(self.core.add_in_connection_from(source))
    }
}

impl HasManyInputConnections for RecordingSink {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.core.get_in_connections()
    }
}

// =============================================================================
// Model builder + result
// =============================================================================

/// Result of one scheduled run, returned for inspection / assertions.
#[derive(Clone, Debug)]
pub struct FibonacciRun {
    /// The frozen per-tick execution order the scheduler derived.
    pub order: Vec<String>,
    /// The Fibonacci values absorbed by the sink, in order.
    pub sequence: Vec<f64>,
    /// The processor's queue length at the end of the run (its retained state).
    pub processor_queue_len: usize,
    /// Number of ticks executed.
    pub ticks: u64,
}

/// Number of ticks kept small enough that every Fibonacci value stays an exact
/// `f64` integer (well under 2^53), so the produced sequence is bit-for-bit
/// reproducible.
pub const DEFAULT_TICKS: u64 = 40;

/// Build the scheduled Fibonacci graph, validate the schedule, and run `ticks`
/// ticks. All ordering is enforced by [`DeterministicScheduler`]; there is no
/// hand-ordered node list anywhere.
pub fn build_and_run(ticks: u64) -> FibonacciRun {
    let step_size = bgn(500.0);
    let mut sched = DeterministicScheduler::new(step_size);

    // A: emits 0 then 1, then goes inert (the Fibonacci seeds).
    let a = Rc::new(RefCell::new(DefiniteFiniteSource::new(
        "A".to_string(),
        vec![
            HasNumericValue { value: 0.0 },
            HasNumericValue { value: 1.0 },
        ],
        -1,
    )));
    // B: pops one token, peeks (retains) the next, emits their sum.
    let b = Rc::new(RefCell::new(EntityNumericProcessor::new("B".to_string())));
    // C: broadcasts each result to every out-connection (the sink AND back to B).
    let c = Rc::new(RefCell::new(EntitySplitter::new(
        "C".to_string(),
        SplitterOpts {
            xx: None,
            replay_items_if_not_first_accepted: false,
        },
    )));
    // D: records every absorbed value.
    let d = Rc::new(RefCell::new(RecordingSink::new("D".to_string())));

    // Register steppable views. Registration order is irrelevant — the execution
    // order is derived from the topology, not from this sequence.
    sched.register(a.clone() as Rc<RefCell<dyn Entity>>);
    sched.register(b.clone() as Rc<RefCell<dyn Entity>>);
    sched.register(c.clone() as Rc<RefCell<dyn Entity>>);
    sched.register(d.clone() as Rc<RefCell<dyn Entity>>);

    // Connection views.
    let a_out: Rc<RefCell<dyn HasOutput>> = a.clone();
    let b_out: Rc<RefCell<dyn HasOutput>> = b.clone();
    let c_out: Rc<RefCell<dyn HasOutput>> = c.clone();
    let b_in: Rc<RefCell<dyn HasInput>> = b.clone();
    let c_in: Rc<RefCell<dyn HasInput>> = c.clone();
    let d_in: Rc<RefCell<dyn HasInput>> = d.clone();

    // Forward (intra-tick) dataflow: A -> B -> C -> D.
    sched.wire_forward(&a_out, &b_in);
    sched.wire_forward(&b_out, &c_in);
    sched.wire_forward(&c_out, &d_in);
    // Feedback (cross-tick): C -> B. This is the recurrence; declaring it as
    // feedback is what lets the forward graph stay a DAG.
    sched.wire_feedback(&c_out, &b_in);

    // Validate + lock the order, then run. freeze() panics loudly if the wiring
    // ever stops being a valid order-enforced schedule.
    sched.freeze();
    sched.run(ticks);

    let order = sched.execution_order();
    let sequence = d.borrow().recorded.clone();
    let processor_queue_len = b.borrow().get_queue_size();

    FibonacciRun {
        order,
        sequence,
        processor_queue_len,
        ticks,
    }
}

/// Entry point (mirrors the other `main_*::run` demos). Builds the scheduled
/// graph, runs it, and prints the enforced order and the produced sequence.
pub fn run() {
    let result = build_and_run(DEFAULT_TICKS);

    println!("# Fibonacci (scheduler-enforced variant)");
    println!("# enforced per-tick execution order: {:?}", result.order);
    println!(
        "# {} ticks -> {} values absorbed at the sink",
        result.ticks,
        result.sequence.len()
    );
    let preview: Vec<i64> = result.sequence.iter().take(16).map(|v| *v as i64).collect();
    println!("# sequence (first {}): {:?}", preview.len(), preview);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference Fibonacci sequence the sink should observe: it starts `1, 2`
    /// (`0+1`, then `1+1`) and each later term is the sum of the previous two.
    fn reference(count: usize) -> Vec<f64> {
        let mut out = vec![1.0_f64, 2.0_f64];
        while out.len() < count {
            let n = out.len();
            out.push(out[n - 1] + out[n - 2]);
        }
        out.truncate(count);
        out
    }

    #[test]
    fn enforced_order_is_topological() {
        let r = build_and_run(DEFAULT_TICKS);
        assert_eq!(r.order, vec!["A", "B", "C", "D"]);
    }

    #[test]
    fn produces_exact_fibonacci_sequence() {
        let r = build_and_run(DEFAULT_TICKS);
        // The sink starts receiving on tick 2, so we get ticks-1 values.
        assert_eq!(r.sequence.len(), (DEFAULT_TICKS - 1) as usize);
        assert_eq!(r.sequence, reference(r.sequence.len()));
        // Spot-check the canonical head.
        assert_eq!(
            &r.sequence[..6],
            &[1.0, 2.0, 3.0, 5.0, 8.0, 13.0],
            "sink sequence head must be Fibonacci"
        );
    }

    #[test]
    fn run_is_deterministic_across_runs() {
        // No RNG, no fuzz: identical inputs -> identical outputs, every time.
        let a = build_and_run(DEFAULT_TICKS);
        let b = build_and_run(DEFAULT_TICKS);
        assert_eq!(a.order, b.order);
        assert_eq!(a.sequence, b.sequence);
    }

    #[test]
    fn processor_queue_stays_bounded() {
        // After warmup the processor retains exactly two tokens each tick; it must
        // never approach its own >4 overflow guard.
        let r = build_and_run(DEFAULT_TICKS);
        assert_eq!(r.processor_queue_len, 2);
    }

    /// If the recurrence edge `C->B` were (mis)declared as forward instead of
    /// feedback, the forward graph would contain the cycle `B->C->B` and the
    /// scheduler must refuse to freeze — the failure the original implicit
    /// `Vec` ordering could never catch.
    #[test]
    fn misdeclaring_feedback_as_forward_is_rejected() {
        let mut sched = DeterministicScheduler::new(bgn(500.0));
        let a = Rc::new(RefCell::new(DefiniteFiniteSource::new(
            "A".to_string(),
            vec![HasNumericValue { value: 0.0 }],
            -1,
        )));
        let b = Rc::new(RefCell::new(EntityNumericProcessor::new("B".to_string())));
        let c = Rc::new(RefCell::new(EntitySplitter::new(
            "C".to_string(),
            SplitterOpts::default(),
        )));
        let d = Rc::new(RefCell::new(RecordingSink::new("D".to_string())));

        sched.register(a.clone() as Rc<RefCell<dyn Entity>>);
        sched.register(b.clone() as Rc<RefCell<dyn Entity>>);
        sched.register(c.clone() as Rc<RefCell<dyn Entity>>);
        sched.register(d.clone() as Rc<RefCell<dyn Entity>>);

        let a_out: Rc<RefCell<dyn HasOutput>> = a.clone();
        let b_out: Rc<RefCell<dyn HasOutput>> = b.clone();
        let c_out: Rc<RefCell<dyn HasOutput>> = c.clone();
        let b_in: Rc<RefCell<dyn HasInput>> = b.clone();
        let c_in: Rc<RefCell<dyn HasInput>> = c.clone();
        let d_in: Rc<RefCell<dyn HasInput>> = d.clone();

        sched.wire_forward(&a_out, &b_in);
        sched.wire_forward(&b_out, &c_in);
        sched.wire_forward(&c_out, &d_in);
        // BUG on purpose: forward instead of feedback.
        sched.wire_forward(&c_out, &b_in);

        let err = sched.try_freeze().unwrap_err();
        assert!(err.contains("cycle"), "unexpected error: {err}");
    }
}
