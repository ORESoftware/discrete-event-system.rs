//! Canonical use path: `crate::des::checkpoint_precedence::task_dag::*`
//!
//! A **real computation that benefits from token-level precedence**:
//! dependency-ordered task execution (a build / job scheduler).
//!
//! Topological ordering is a fundamental computation — "run every task only after
//! its dependencies have finished." That is *exactly* what the
//! [`CheckpointGate`](crate::des::checkpoint_precedence::entities::CheckpointGate)
//! enforces: each task is a token; "task X depends on task Y" is the constraint
//! "Y must clear the `BUILD` checkpoint before X." The gate then releases tasks in
//! a deterministic, dependency-respecting order, and the precedence ledger rejects
//! a circular dependency before anything runs.
//!
//! This reuses the checkpoint-precedence machinery verbatim (source → gate →
//! sink); only the graph is different. It shows the mechanism generalizing from
//! the toy demo to a recognizable computation.
//!
//! Graph (a small multi-target build):
//!
//! ```text
//!   fetch ─┬─> compile-core ─┬─> compile-cli ─┐
//!          │                 ├─> compile-gui ─┼─> link ─┐
//!          └─> gen-proto ────┘                │         ├─> integration-test ─> package
//!                            compile-core ───────> unit-test ──────────────────┘
//! ```

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::des::checkpoint_precedence::entities::{
    CheckpointGate, OrderedTokenSource, PreparedToken, RecordingSink,
};
use crate::des::checkpoint_precedence::ledger::{PrecedenceLedger, Requirement};
use crate::des::r#abstract::interfaces::{HasInput, HasOutput};
use crate::des::r#abstract::r#abstract::Entity;
use crate::des::shared::precision::bgn;

/// The single checkpoint every task must clear ("a task is done").
pub const CHECKPOINT: &str = "BUILD";

/// Result of one scheduled build.
#[derive(Clone, Debug)]
pub struct TaskDagRun {
    /// Tasks in the (deliberately non-topological) order they were submitted.
    pub submitted_order: Vec<String>,
    /// Tasks in the order the gate executed them — a valid topological order.
    pub execution_order: Vec<String>,
    pub ticks: u64,
}

/// The demo build graph as `(task, dependencies)`. Declared in **alphabetical**
/// order on purpose — a naive "just run them in listed order" would violate
/// dependencies (e.g. `compile-cli` before `fetch`). The gate fixes that.
fn build_graph() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("compile-cli", vec!["compile-core", "gen-proto"]),
        ("compile-core", vec!["fetch"]),
        ("compile-gui", vec!["compile-core"]),
        ("fetch", vec![]),
        ("gen-proto", vec!["fetch"]),
        ("integration-test", vec!["link", "unit-test"]),
        ("link", vec!["compile-cli", "compile-gui"]),
        ("package", vec!["integration-test"]),
        ("unit-test", vec!["compile-core"]),
    ]
}

/// Build the `source → gate → sink` network from `graph`, validate the
/// dependency DAG, and run until the gate drains.
pub fn build_and_run_graph(graph: &[(&str, Vec<&str>)], ticks: u64) -> TaskDagRun {
    let step = bgn(500.0);
    let ledger = Rc::new(RefCell::new(PrecedenceLedger::new()));

    let tokens: Vec<PreparedToken> = graph
        .iter()
        .enumerate()
        .map(|(i, (name, deps))| {
            let requirements = deps
                .iter()
                .map(|d| Requirement::new(d, CHECKPOINT))
                .collect();
            PreparedToken::new(name, (i as f64) + 1.0, requirements)
        })
        .collect();

    let src = Rc::new(RefCell::new(OrderedTokenSource::new(
        "SUBMIT".to_string(),
        ledger.clone(),
        tokens,
    )));
    let gate = Rc::new(RefCell::new(CheckpointGate::new(
        CHECKPOINT.to_string(),
        ledger.clone(),
    )));
    let sink = Rc::new(RefCell::new(RecordingSink::new("DONE".to_string())));

    // Reject a circular / dangling dependency before running a single task.
    if let Err(e) = ledger.borrow().validate() {
        panic!("dependency validation failed: {e}");
    }

    let gate_in: Rc<RefCell<dyn HasInput>> = gate.clone();
    let sink_in: Rc<RefCell<dyn HasInput>> = sink.clone();
    src.borrow_mut().add_out_connection(gate_in);
    gate.borrow_mut().add_out_connection(sink_in);

    let nodes: Vec<Rc<RefCell<dyn Entity>>> = vec![src.clone(), gate.clone(), sink.clone()];
    for _ in 0..ticks {
        for n in &nodes {
            n.borrow_mut().do_time_step(step);
        }
    }

    let submitted_order = src.borrow().emitted_order().to_vec();
    let execution_order = gate.borrow().released_order().to_vec();

    TaskDagRun {
        submitted_order,
        execution_order,
        ticks,
    }
}

/// Build + run the default demo graph.
pub fn build_and_run(ticks: u64) -> TaskDagRun {
    let graph = build_graph();
    build_and_run_graph(&graph, ticks)
}

/// `true` if `order` lists every task only after all of its dependencies — i.e.
/// it is a valid topological order of `graph`.
pub fn is_topological_order(order: &[String], graph: &[(&str, Vec<&str>)]) -> bool {
    let deps: HashMap<&str, &Vec<&str>> = graph.iter().map(|(n, d)| (*n, d)).collect();
    let mut done: HashSet<String> = HashSet::new();
    for task in order {
        if let Some(task_deps) = deps.get(task.as_str()) {
            for d in task_deps.iter() {
                if !done.contains(*d) {
                    return false;
                }
            }
        }
        done.insert(task.clone());
    }
    true
}

/// Entry point (mirrors the other `main_*::run` demos).
pub fn run() {
    let result = build_and_run(12);

    println!("# Dependency-ordered task scheduler (checkpoint-precedence applied)");
    println!("# submitted (alphabetical) : {:?}", result.submitted_order);
    println!("# executed  (topological)  : {:?}", result.execution_order);
    let ok = is_topological_order(&result.execution_order, &build_graph());
    println!("# is a valid build order   : {ok}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submitted_order_is_alphabetical_not_topological() {
        let r = build_and_run(12);
        // Submitted order is the alphabetical declaration order...
        assert_eq!(r.submitted_order[0], "compile-cli");
        // ...which is NOT a valid execution order (compile-cli before fetch).
        assert!(!is_topological_order(&r.submitted_order, &build_graph()));
    }

    #[test]
    fn gate_executes_in_valid_topological_order() {
        let r = build_and_run(12);
        assert_eq!(r.execution_order.len(), 9);
        assert!(
            is_topological_order(&r.execution_order, &build_graph()),
            "gate must release tasks in dependency order: {:?}",
            r.execution_order
        );
    }

    #[test]
    fn execution_order_is_deterministic() {
        let a = build_and_run(12);
        let b = build_and_run(12);
        assert_eq!(a.execution_order, b.execution_order);
        // Exact, reproducible order (topological with lowest-seq tie-break).
        assert_eq!(
            a.execution_order,
            vec![
                "fetch",
                "compile-core",
                "compile-gui",
                "gen-proto",
                "compile-cli",
                "link",
                "unit-test",
                "integration-test",
                "package",
            ]
        );
    }

    #[test]
    #[should_panic(expected = "dependency validation failed")]
    fn circular_dependency_is_rejected() {
        // a -> b -> a is an impossible build order; validation must refuse it.
        let graph: Vec<(&str, Vec<&str>)> = vec![("a", vec!["b"]), ("b", vec!["a"])];
        let _ = build_and_run_graph(&graph, 8);
    }
}
