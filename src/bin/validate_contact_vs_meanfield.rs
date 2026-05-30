//! Binary entrypoint for `validate_contact_vs_meanfield` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_contact_vs_meanfield`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_contact_vs_meanfield`.

fn main() {
    std::process::exit(des_engine::des::runners::validate_contact_vs_meanfield::run());
}
