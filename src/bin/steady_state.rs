//! Binary entrypoint for `steady_state` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::steady_state`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin steady_state`.

fn main() {
    des_engine::des::runners::steady_state::run();
}
