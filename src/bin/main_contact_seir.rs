//! Binary entrypoint for `main_contact_seir` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `des_engine::des::main_contact_seir`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_contact_seir`.

fn main() {
    des_engine::des::main_contact_seir::run();
}
