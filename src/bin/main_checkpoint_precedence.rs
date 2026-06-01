//! Binary entrypoint for `main_checkpoint_precedence`.
//!
//! Thin wrapper that delegates to the library module
//! `des_engine::des::checkpoint_precedence`. The real logic (the precedence
//! ledger, the BST-backed checkpoint gate, and the token-level ordering demo)
//! lives in the module; this binary just exposes it as
//! `cargo run --bin main_checkpoint_precedence`.

fn main() {
    des_engine::des::checkpoint_precedence::run();
}
