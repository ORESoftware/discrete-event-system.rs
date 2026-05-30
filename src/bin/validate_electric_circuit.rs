//! Binary entrypoint for `validate_electric_circuit` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_electric_circuit`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_electric_circuit`.

fn main() {
    des_engine::des::runners::validate_electric_circuit::run();
}
