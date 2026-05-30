//! Binary entrypoint for `replicate` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::replicate`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin replicate`.

fn main() {
    des_engine::des::runners::replicate::run();
}
