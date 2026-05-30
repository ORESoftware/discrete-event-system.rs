//! Binary entrypoint for `main_factmachine_markets` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_factmachine_markets`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_factmachine_markets`.

fn main() {
    des_engine::des::main_factmachine_markets::run();
}
