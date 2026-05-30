//! Binary entrypoint for `validate_backpropagation` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_backpropagation`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_backpropagation`.

fn main() {
    std::process::exit(des_engine::des::runners::validate_backpropagation::run());
}
