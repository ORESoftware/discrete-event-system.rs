//! Binary entrypoint for `main_shadow_eval` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module
//! `des_engine::des::main_shadow_eval`. The real logic lives in the module
//! (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_shadow_eval`.

fn main() {
    des_engine::des::main_shadow_eval::run();
}
