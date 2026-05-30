//! Binary entrypoint for `main` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main`.

fn main() {
    des_engine::des::main::run();
}
