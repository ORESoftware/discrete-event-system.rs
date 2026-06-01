//! Binary entrypoint for `main_fibonacci_scheduled`.
//!
//! Thin wrapper that delegates to the library module
//! `des_engine::des::fibonacci_scheduled`. The real logic (the deterministic
//! scheduler / enforcer and the order-enforced Fibonacci graph) lives in the
//! module; this binary just exposes it as
//! `cargo run --bin main_fibonacci_scheduled`.

fn main() {
    des_engine::des::fibonacci_scheduled::run();
}
