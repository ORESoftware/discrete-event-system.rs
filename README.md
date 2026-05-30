# discrete-event-system.rs

Rust migration target for `/Users/alexandermills/codes/ores/discrete-event-system`.

This crate preserves the TypeScript repository's migration headers as the source
of truth for the first file-for-file port:

- `src/des/**/*.ts` library files map to `src/des/**/*.rs`.
- runner/main entrypoints map to `src/bin/*.rs`.
- TypeScript test files map to `tests/*.rs`.

The generated scaffold keeps every mapped file present, while the core DES
foundation has been manually ported first: entity state/traits, graph data,
queues, moving entities, pure transforms, output routing, decisions, signal
values, and shared utility helpers.

`cargo`, `rustc`, and `rustfmt` are not currently on base PATH in this
environment. Verification was run through Nix instead:

```bash
nix-shell -p cargo rustc rustfmt --run 'cargo fmt --check'
nix-shell -p cargo rustc rustfmt --run 'cargo check --all-targets'
nix-shell -p cargo rustc rustfmt --run 'cargo test --all-targets'
```

With a local Rust toolchain installed, the equivalent direct commands are:

```bash
cargo fmt
cargo check --all-targets
cargo test --all-targets
```

The scaffolding script is at `tools/migrate_ts_tree.js`; it is idempotent for
files marked `MigrationFile::ported_core`.
