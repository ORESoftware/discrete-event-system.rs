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

## File-by-file Audit

The full branch comparison is `git diff --name-status origin/main main-lb`.
It contains 592 path-level differences:

- 259 same-path modifications: audited against `origin/main`; mainline behavior was kept except for the hardening/exposure fixes listed below.
- 168 paths deleted by `main-lb`: kept from `origin/main` because they are behavioral ports, tests, shared precision utilities, or the serial simulation driver.
- 165 paths added by `main-lb`: 155 are `src/bin/*` or root `tests/*` relocations that duplicate the mainline 1:1 `src/des/*` layout, 9 are scaffold/documentation/root-helper additions superseded by the mainline architecture, and 1 (`src/des/ws_server/mod.rs`) was accepted.

Accepted from the `main-lb` audit because they improve coverage without breaking the 1:1 layout:

- Exposed and repaired `src/des/general/math_blocks.rs`.
- Exposed and repaired `src/des/general/des_base/control_blocks.rs`.
- Exposed `src/des/general/des_base/visual_block.rs`.
- Exposed `src/des/general/adapters/math_blocks_adapter.rs`.
- Exposed `src/des/general/adapters/mdp_adjacent_adapters.rs`.
- Exposed and repaired `src/des/general/adapters/network_flow_adapter.rs`.
- Exposed and repaired `src/des/general/adapters/optimal_control_adapters.rs`.
- Exposed and repaired `src/des/general/adapters/statistical_optimization_adapter.rs`.
- Exposed `src/des/http_server/mod.rs` and `src/des/ws_server/ws_server.rs`; added `src/des/ws_server/mod.rs`.

Rejected after audit:

- `src/bin/*` and root `tests/*` relocation scaffolds, because the project goal is a file-for-file port under `src/des`, and `origin/main` already has compiled behavioral modules/tests in that layout.
- `src/core.rs`, `src/migration.rs`, and `src/numeric.rs`, because their concepts are already represented by the mainline `des::shared` and in-file migration notes without adding a second root framework.
- `src/des/entity_conn_ts/*`, because current `src/des/mod.rs` preserves the literal TS directory mapping with `#[path = "entity_conn.ts/conn.rs"] pub mod entity_conn;`.
- `README.md`, `MIGRATION_MANIFEST.md`, `MIGRATION_STATUS.md`, and `tools/migrate_ts_tree.js`, because they describe the older scaffold/relocation pass and understate the completed mainline behavioral port.

## Verification

Baseline before hardening on the mainline tree:

- `cargo check --all-targets`
- `cargo test --all-targets -- --test-threads=1`

Final verification should be rerun after formatting and hardening:

- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --all-targets -- --test-threads=1`
- `cargo run --bin run_all_simulations`
