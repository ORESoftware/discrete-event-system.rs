//! Binary entrypoint for `validate_neural_network` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_neural_network`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_neural_network`.

fn main() {
    des_engine::des::runners::validate_neural_network::run();
}
