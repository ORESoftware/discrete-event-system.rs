//! Binary entrypoint for `main_temp_control_anim` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_temp_control_anim`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_temp_control_anim`.

fn main() {
    des_engine::des::main_temp_control_anim::run();
}
