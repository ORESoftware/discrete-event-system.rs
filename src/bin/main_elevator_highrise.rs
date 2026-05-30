//! Binary entrypoint for `main_elevator_highrise` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_elevator_highrise`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_elevator_highrise`.

fn main() {
    des_engine::des::main_elevator_highrise::run();
}
