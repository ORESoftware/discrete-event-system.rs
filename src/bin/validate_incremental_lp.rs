//! Binary entrypoint for `validate_incremental_lp` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_incremental_lp`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_incremental_lp`.

fn main() {
    des_engine::des::runners::validate_incremental_lp::run();
}
