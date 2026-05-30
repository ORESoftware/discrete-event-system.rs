//! Binary entrypoint for `compare_external_fel_models` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::compare_external_fel_models`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin compare_external_fel_models`.

fn main() {
    std::process::exit(des_engine::des::runners::compare_external_fel_models::run());
}
