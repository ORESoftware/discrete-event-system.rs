//! Binary entrypoint for `validate_factmachine_math` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_factmachine_math`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_factmachine_math`.

fn main() {
    des_engine::des::runners::validate_factmachine_math::run();
}
