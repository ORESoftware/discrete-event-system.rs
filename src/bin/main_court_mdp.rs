//! Binary entrypoint for `main_court_mdp` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_court_mdp`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_court_mdp`.

fn main() {
    des_engine::des::main_court_mdp::run();
}
