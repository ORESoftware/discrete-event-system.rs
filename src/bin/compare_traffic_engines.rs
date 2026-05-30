//! Binary entrypoint for `compare_traffic_engines` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::runners::compare_traffic_engines`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin compare_traffic_engines`.

fn main() {
    des_engine::des::runners::compare_traffic_engines::run();
}
