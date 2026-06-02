//! Binary entrypoint for `validate_optimization_scale`.
//!
//! Thin wrapper around the in-tree runner module so the scale envelope can be
//! executed with `cargo run --bin validate_optimization_scale`.

fn main() {
    des_engine::des::runners::validate_optimization_scale::run();
}
