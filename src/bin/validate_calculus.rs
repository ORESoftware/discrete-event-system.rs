//! Binary entrypoint for `validate_calculus` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_calculus`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_calculus`.

fn main() {
    std::process::exit(des_engine::des::runners::validate_calculus::run());
}
