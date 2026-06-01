//! Binary entrypoint for `main_task_build_order`.
//!
//! Dependency-ordered task scheduler (a build/job DAG) built on the
//! checkpoint-precedence gate: tasks are submitted in a non-topological order and
//! executed in a valid dependency order. Delegates to
//! `des_engine::des::checkpoint_precedence::task_dag`. Run with
//! `cargo run --bin main_task_build_order`.

fn main() {
    des_engine::des::checkpoint_precedence::task_dag::run();
}
