//! Binary entrypoint for `validate_two_disease` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_two_disease`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_two_disease`.

fn main() {
    des_engine::des::runners::validate_two_disease::run();
}
