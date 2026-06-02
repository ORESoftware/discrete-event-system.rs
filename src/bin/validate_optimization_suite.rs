//! Binary entrypoint for `validate_optimization_suite`.
//!
//! Thin wrapper that delegates to the library module
//! `des_engine::des::runners::validate_optimization_suite`.

fn main() {
    des_engine::des::runners::validate_optimization_suite::run();
}
