//! Canonical use path: `crate::des::checkpoint_precedence::model::*`
//!
//! A demo of **token-level ordering**: tokens are emitted into the network in
//! their natural sequence, but each declares (by UUID reference) which other
//! tokens must clear a checkpoint before it may pass. The [`CheckpointGate`]
//! holds and re-orders them so the order through the checkpoint honors the
//! declared constraints, with the `seq` stamp as the deterministic tie-break.
//!
//! Graph: `source(emits T1..T5) → gate "C" → sink(records payloads)`.
//!
//! Constraints (a partial order, not a total one):
//! * `T1` must clear `C` only after `T4`;
//! * `T2` must clear `C` only after `T5`.
//!
//! Even though the tokens *arrive* in order `T1, T2, T3, T4, T5`, the gate
//! releases them as `T3, T4, T1, T5, T2` — the unique deterministic order that
//! satisfies the constraints and otherwise prefers the lowest `seq`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::checkpoint_precedence::entities::{
    CheckpointGate, OrderedTokenSource, PreparedToken, RecordingSink,
};
use crate::des::checkpoint_precedence::ledger::{PrecedenceLedger, Requirement};
use crate::des::r#abstract::interfaces::{HasInput, HasOutput};
use crate::des::r#abstract::r#abstract::Entity;
use crate::des::shared::precision::bgn;

/// Result of one checkpoint-ordered run, returned for inspection / assertions.
#[derive(Clone, Debug)]
pub struct CheckpointRun {
    /// UUIDs in the order the source emitted them (arrival order).
    pub emitted_order: Vec<String>,
    /// UUIDs in the order the gate released them (enforced order).
    pub released_order: Vec<String>,
    /// Payloads absorbed at the sink, in release order.
    pub payload_sequence: Vec<f64>,
    pub ticks: u64,
}

/// The checkpoint name used throughout the demo.
pub const CHECKPOINT: &str = "C";

/// Build the `source → gate → sink` network with the demo's precedence
/// constraints, validate the precedence graph, and run `ticks` ticks.
pub fn build_and_run(ticks: u64) -> CheckpointRun {
    let step = bgn(500.0);
    let ledger = Rc::new(RefCell::new(PrecedenceLedger::new()));

    // Tokens emitted in natural order T1..T5, payloads 10..50. The constraints
    // reference predecessors by UUID — exactly the "stamp + reference" idea.
    let tokens = vec![
        PreparedToken::new("T1", 10.0, vec![Requirement::new("T4", CHECKPOINT)]),
        PreparedToken::new("T2", 20.0, vec![Requirement::new("T5", CHECKPOINT)]),
        PreparedToken::new("T3", 30.0, vec![]),
        PreparedToken::new("T4", 40.0, vec![]),
        PreparedToken::new("T5", 50.0, vec![]),
    ];

    let src = Rc::new(RefCell::new(OrderedTokenSource::new(
        "SRC".to_string(),
        ledger.clone(),
        tokens,
    )));
    let gate = Rc::new(RefCell::new(CheckpointGate::new(
        CHECKPOINT.to_string(),
        ledger.clone(),
    )));
    let sink = Rc::new(RefCell::new(RecordingSink::new("SINK".to_string())));

    // Validate the precedence graph BEFORE running — fail fast on an unsatisfiable
    // (cyclic) or dangling constraint.
    if let Err(e) = ledger.borrow().validate() {
        panic!("precedence validation failed: {e}");
    }

    // Wire source -> gate -> sink.
    let gate_in: Rc<RefCell<dyn HasInput>> = gate.clone();
    let sink_in: Rc<RefCell<dyn HasInput>> = sink.clone();
    src.borrow_mut().add_out_connection(gate_in);
    gate.borrow_mut().add_out_connection(sink_in);

    // Node order is a trivial linear chain (source before gate before sink), so a
    // simple ordered loop suffices; the interesting ordering happens *inside* the
    // gate, at the token level.
    let nodes: Vec<Rc<RefCell<dyn Entity>>> = vec![src.clone(), gate.clone(), sink.clone()];
    for _ in 0..ticks {
        for n in &nodes {
            n.borrow_mut().do_time_step(step);
        }
    }

    let emitted_order = src.borrow().emitted_order().to_vec();
    let released_order = gate.borrow().released_order().to_vec();
    let payload_sequence = sink.borrow().recorded.clone();

    CheckpointRun {
        emitted_order,
        released_order,
        payload_sequence,
        ticks,
    }
}

/// Entry point (mirrors the other `main_*::run` demos).
pub fn run() {
    let result = build_and_run(6);

    println!("# Checkpoint-precedence ordering (token-level enforcer)");
    println!("# tokens arrive in order : {:?}", result.emitted_order);
    println!("# gate releases in order : {:?}", result.released_order);
    let payloads: Vec<i64> = result.payload_sequence.iter().map(|v| *v as i64).collect();
    println!("# payloads at the sink   : {payloads:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrival_order_is_natural() {
        let r = build_and_run(6);
        assert_eq!(r.emitted_order, vec!["T1", "T2", "T3", "T4", "T5"]);
    }

    #[test]
    fn gate_releases_in_constraint_order() {
        let r = build_and_run(6);
        assert_eq!(r.released_order, vec!["T3", "T4", "T1", "T5", "T2"]);
        assert_eq!(r.payload_sequence, vec![30.0, 40.0, 10.0, 50.0, 20.0]);
    }

    #[test]
    fn every_token_clears_exactly_once() {
        let r = build_and_run(6);
        assert_eq!(r.released_order.len(), 5);
        assert_eq!(r.payload_sequence.len(), 5);
    }

    #[test]
    fn run_is_deterministic_across_runs() {
        let a = build_and_run(6);
        let b = build_and_run(6);
        assert_eq!(a.released_order, b.released_order);
        assert_eq!(a.payload_sequence, b.payload_sequence);
    }
}
