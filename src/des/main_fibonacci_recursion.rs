//! Port of `src/des/main-fibonacci-recursion.ts`.
//!
//! Wires a small entity graph (source -> value-adder -> splitter -> sink) that
//! computes Fibonacci by recurrent feedback (the splitter feeds the processor
//! back its own running sums).
//!
//! The TS `const run = () => {...}` closure + top-level invocation becomes
//! [`run`]. `bgn(500)` step size → `crate::des::general::general::bgn`. The
//! heterogeneous `Map<string, Entity<any>>` + uniform `addOutConnection` becomes
//! concrete `Rc<RefCell<…>>` handles coerced to `dyn Entity` (for the step loop)
//! and `dyn HasInput` (as connection targets).

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_processing::value_adder::EntityNumericProcessor;
use crate::des::entity_routing::entity_splitter::{EntitySplitter, SplitterOpts};
use crate::des::entity_sink::generic_sink::GenericEntitySink;
use crate::des::entity_source::source::DefiniteFiniteSource;
use crate::des::general::general::bgn;
use crate::des::observers::program_observer::ProgramObserver;
use crate::des::r#abstract::interfaces::{HasInput, HasOutput};
use crate::des::r#abstract::r#abstract::{Entity, EntityObserver, HasNumericValue};

/// Entry point (TS top-level `run()` closure + invocation).
pub fn run() {
    let step_size_millis = bgn(500.0);
    let obs: Rc<RefCell<dyn EntityObserver>> = Rc::new(RefCell::new(ProgramObserver::new()));

    // input(0,1) -> recurse (processor) -> splitter -> sink, with splitter -> processor feedback.
    let a = Rc::new(RefCell::new(DefiniteFiniteSource::new(
        "A".to_string(),
        vec![HasNumericValue { value: 0.0 }, HasNumericValue { value: 1.0 }],
        -1,
    )));
    let b = Rc::new(RefCell::new(EntityNumericProcessor::new("B".to_string())));
    let c = Rc::new(RefCell::new(EntitySplitter::new(
        "C".to_string(),
        SplitterOpts { xx: None, replay_items_if_not_first_accepted: false },
    )));
    let d = Rc::new(RefCell::new(GenericEntitySink::new("D".to_string())));

    a.borrow_mut().subscribe(obs.clone());
    b.borrow_mut().subscribe(obs.clone());
    c.borrow_mut().subscribe(obs.clone());
    d.borrow_mut().subscribe(obs.clone());

    // `HasInput` views used as connection targets.
    let b_in: Rc<RefCell<dyn HasInput>> = b.clone();
    let c_in: Rc<RefCell<dyn HasInput>> = c.clone();
    let d_in: Rc<RefCell<dyn HasInput>> = d.clone();

    // Edges: A->B, B->C, C->D, C->B (feedback). `addOutConnection` is `HasOutput`.
    let _ = a.borrow_mut().add_out_connection(b_in.clone());
    let _ = b.borrow_mut().add_out_connection(c_in.clone());
    let _ = c.borrow_mut().add_out_connection(d_in.clone());
    let _ = c.borrow_mut().add_out_connection(b_in.clone());

    // `Array.from(programEntities)` preserves insertion order A,B,C,D.
    let program: Vec<Rc<RefCell<dyn Entity>>> = vec![a.clone(), b.clone(), c.clone(), d.clone()];

    for _ in 0..100 {
        for v in &program {
            v.borrow_mut().do_time_step(step_size_millis);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_graph_runs_without_panicking() {
        run();
    }
}
