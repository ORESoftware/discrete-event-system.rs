//! Binary entrypoint for `compare_elevator_dispatch` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::compare_elevator_dispatch`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin compare_elevator_dispatch`.

fn main() {
    des_engine::des::runners::compare_elevator_dispatch::run();
}
