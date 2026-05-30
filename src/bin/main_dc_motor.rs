//! Binary entrypoint for `main_dc_motor` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_dc_motor`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_dc_motor`.

fn main() {
    des_engine::des::main_dc_motor::run();
}
