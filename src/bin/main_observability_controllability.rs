//! Binary entrypoint for `main_observability_controllability` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_observability_controllability`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_observability_controllability`.

fn main() {
    des_engine::des::main_observability_controllability::run();
}
