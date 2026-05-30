//! Binary entrypoint for `main_ip_mip_des` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_ip_mip_des`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_ip_mip_des`.

fn main() {
    des_engine::des::main_ip_mip_des::run();
}
