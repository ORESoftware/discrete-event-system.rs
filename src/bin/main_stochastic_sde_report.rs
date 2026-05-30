//! Binary entrypoint for `main_stochastic_sde_report` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_stochastic_sde_report`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_stochastic_sde_report`.

fn main() {
    des_engine::des::main_stochastic_sde_report::run();
}
