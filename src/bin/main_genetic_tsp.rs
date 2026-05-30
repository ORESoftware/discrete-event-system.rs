//! Binary entrypoint for `main_genetic_tsp` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_genetic_tsp`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_genetic_tsp`.

fn main() {
    des_engine::des::main_genetic_tsp::run();
}
