//! Binary entrypoint for `validate_ip_mip_external` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_ip_mip_external`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_ip_mip_external`.

fn main() {
    des_engine::des::runners::validate_ip_mip_external::run();
}
