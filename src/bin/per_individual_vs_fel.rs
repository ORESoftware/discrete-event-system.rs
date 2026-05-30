//! Binary entrypoint for `per_individual_vs_fel` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::per_individual_vs_fel`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin per_individual_vs_fel`.

fn main() {
    des_engine::des::runners::per_individual_vs_fel::run();
}
