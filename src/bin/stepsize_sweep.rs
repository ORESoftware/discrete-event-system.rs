//! Binary entrypoint for `stepsize_sweep` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::stepsize_sweep`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin stepsize_sweep`.

fn main() {
    des_engine::des::runners::stepsize_sweep::run();
}
