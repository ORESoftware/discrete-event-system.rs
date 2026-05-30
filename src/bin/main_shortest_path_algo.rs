//! Binary entrypoint for `main_shortest_path_algo` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_shortest_path_algo`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_shortest_path_algo`.

fn main() {
    des_engine::des::main_shortest_path_algo::run();
}
