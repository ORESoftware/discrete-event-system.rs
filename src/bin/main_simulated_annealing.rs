//! Binary entrypoint for `main_simulated_annealing` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_simulated_annealing`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_simulated_annealing`.

fn main() {
    des_engine::des::main_simulated_annealing::run();
}
