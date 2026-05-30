//! Binary entrypoint for `main_elevator` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_elevator`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_elevator`.

fn main() {
    des_engine::des::main_elevator::run();
}
