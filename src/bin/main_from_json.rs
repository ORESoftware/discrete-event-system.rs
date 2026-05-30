//! Binary entrypoint for `main_from_json` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_from_json`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_from_json`.

fn main() {
    des_engine::des::main_from_json::run();
}
