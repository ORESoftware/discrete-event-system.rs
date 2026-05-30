//! Binary entrypoint for `run_external_module` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::run_external_module`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin run_external_module`.

fn main() {
    std::process::exit(des_engine::des::runners::run_external_module::run());
}
