//! Binary entrypoint for `main_knapsack_problem` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_knapsack_problem`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_knapsack_problem`.

fn main() {
    des_engine::des::main_knapsack_problem::run();
}
