//! Binary entrypoint for `main_newsvendor` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_newsvendor`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_newsvendor`.

fn main() {
    des_engine::des::main_newsvendor::run();
}
