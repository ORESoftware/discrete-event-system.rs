//! Binary entrypoint for `main_network_mutex` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_network_mutex`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_network_mutex`.

fn main() {
    des_engine::des::main_network_mutex::run();
}
