//! Binary entrypoint for `validate_computer_network` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_computer_network`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_computer_network`.

fn main() {
    std::process::exit(des_engine::des::runners::validate_computer_network::run());
}
