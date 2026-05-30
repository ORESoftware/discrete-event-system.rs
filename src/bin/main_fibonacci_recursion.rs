//! Binary entrypoint for `main_fibonacci_recursion` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_fibonacci_recursion`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_fibonacci_recursion`.

fn main() {
    des_engine::des::main_fibonacci_recursion::run();
}
