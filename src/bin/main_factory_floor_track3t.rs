//! Binary entrypoint for `main_factory_floor_track3t` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_factory_floor_track3t`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_factory_floor_track3t`.

fn main() {
    des_engine::des::main_factory_floor_track3t::run();
}
