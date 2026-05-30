# Merge Reconciliation

This branch manually reconciles `origin/main` and `main-lb`.

- Mainline basis: `origin/main` at `2d29f1ddf37354108c71236975274963e4f33b85`.
- Other parent: `main-lb` at `183ec49ee2ecf69d0caf8b64b722e840488843e8`.
- Decision rule: when the two branches disagreed ambiguously, keep the mainline implementation.

## Semantic Choices

- Kept `origin/main` for the behavioral Rust port. It contains the complete library modules, runner implementations, ported TS test suite, extensive tests, integration smoke tests, shared precision utilities, and the serial `run_all_simulations` binary.
- Rejected scaffold-heavy replacements from local `main`/`main-lb` where they would replace implemented modules with placeholders or move 1:1 `src/des/*.rs` modules into thin Cargo binaries.
- Kept the existing mainline 1:1 mapping for TypeScript files under `src/des`, including `src/des/main_*.rs`, `src/des/runners/*.rs`, `src/des/test/*.rs`, and `src/des/shared/*.rs`.
- Accepted the low-risk hardening ideas from `main-lb`: clippy module-inception annotations, enum default derivation, iterator-based matrix/graph loops, Simpson parity checking via `is_multiple_of`, and safer Gaussian-elimination row borrowing.
- Kept mainline numeric policy in `src/des/shared/precision.rs`; it already uses `rust_decimal` for exact base-10 bookkeeping, exact rationals for fractions, and compensated `f64` summation for numerical kernels.

## Verification

Baseline before hardening on the mainline tree:

- `cargo check --all-targets`
- `cargo test --all-targets -- --test-threads=1`

Final verification should be rerun after formatting and hardening:

- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --all-targets -- --test-threads=1`
- `cargo run --bin run_all_simulations`
