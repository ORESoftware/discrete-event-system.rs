//! Binary entrypoint for `main_optimization_as_des` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_optimization_as_des`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_optimization_as_des`.

fn main() {
    des_engine::des::main_optimization_as_des::run();
}
