//! Binary entrypoint for `main_calculus` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_calculus`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_calculus`.

fn main() {
    des_engine::des::main_calculus::run();
}
