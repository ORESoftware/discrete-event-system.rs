//! Binary entrypoint for `main_backpropagation` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_backpropagation`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_backpropagation`.

fn main() {
    des_engine::des::main_backpropagation::run();
}
