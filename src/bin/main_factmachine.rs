//! Binary entrypoint for `main_factmachine` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_factmachine`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_factmachine`.

fn main() {
    des_engine::des::main_factmachine::run();
}
