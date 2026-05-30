//! Binary entrypoint for `validate_smart_traffic_external` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::validate_smart_traffic_external`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_smart_traffic_external`.

fn main() {
    des_engine::des::runners::validate_smart_traffic_external::run();
}
