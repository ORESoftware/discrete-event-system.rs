//! Binary entrypoint for `main_epidemic_improved` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_epidemic_improved`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_epidemic_improved`.

fn main() {
    des_engine::des::main_epidemic_improved::run();
}
