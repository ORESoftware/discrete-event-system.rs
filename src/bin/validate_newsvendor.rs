//! Binary entrypoint for `validate_newsvendor` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_newsvendor`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_newsvendor`.

fn main() {
    des_engine::des::runners::validate_newsvendor::run();
}
