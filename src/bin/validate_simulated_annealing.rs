//! Binary entrypoint for `validate_simulated_annealing` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_simulated_annealing`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_simulated_annealing`.

fn main() {
    des_engine::des::runners::validate_simulated_annealing::run();
}
