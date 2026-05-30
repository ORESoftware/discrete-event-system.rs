//! Binary entrypoint for `main_dispatch_combo` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_dispatch_combo`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_dispatch_combo`.

fn main() {
    des_engine::des::main_dispatch_combo::run();
}
