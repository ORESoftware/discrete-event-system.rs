//! Binary entrypoint for `main_build_site` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_build_site`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_build_site`.

fn main() {
    des_engine::des::main_build_site::run();
}
