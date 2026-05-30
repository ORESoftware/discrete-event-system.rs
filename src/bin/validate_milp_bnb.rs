//! Binary entrypoint for `validate_milp_bnb` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_milp_bnb`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_milp_bnb`.

fn main() {
    des_engine::des::runners::validate_milp_bnb::run();
}
