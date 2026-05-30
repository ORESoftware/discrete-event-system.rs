//! Binary entrypoint for `main_monte_carlo_sim` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_monte_carlo_sim`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_monte_carlo_sim`.

fn main() {
    des_engine::des::main_monte_carlo_sim::run();
}
