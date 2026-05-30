//! Binary entrypoint for `validate_with_externals` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_with_externals`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_with_externals`.

fn main() {
    des_engine::des::runners::validate_with_externals::run();
}
