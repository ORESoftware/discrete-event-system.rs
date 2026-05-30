//! Binary entrypoint for `main_stochastic_sde` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_stochastic_sde`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_stochastic_sde`.

fn main() {
    des_engine::des::main_stochastic_sde::run();
}
