//! Binary entrypoint for `main_plumbing_flow` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_plumbing_flow`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_plumbing_flow`.

fn main() {
    des_engine::des::main_plumbing_flow::run();
}
