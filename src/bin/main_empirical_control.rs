//! Binary entrypoint for `main_empirical_control` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_empirical_control`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_empirical_control`.

fn main() {
    des_engine::des::main_empirical_control::run();
}
