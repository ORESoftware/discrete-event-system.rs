//! Binary entrypoint for `main_lp_factory` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_lp_factory`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_lp_factory`.

fn main() {
    des_engine::des::main_lp_factory::run();
}
