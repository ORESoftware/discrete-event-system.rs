//! Binary entrypoint for `main_hazard_function_survival_analysis` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_hazard_function_survival_analysis`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_hazard_function_survival_analysis`.

fn main() {
    des_engine::des::main_hazard_function_survival_analysis::run();
}
