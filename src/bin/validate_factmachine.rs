//! Binary entrypoint for `validate_factmachine` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_factmachine`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_factmachine`.

fn main() {
    des_engine::des::runners::validate_factmachine::run();
}
