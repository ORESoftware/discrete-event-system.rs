//! Binary entrypoint for `main_computer_network` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_computer_network`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_computer_network`.

fn main() {
    des_engine::des::main_computer_network::run();
}
