# Merge: `main` + `main-lb` → `main-merged`

The two branches are **unrelated histories** (git refuses a normal merge), so this
was reconciled manually, file-by-file, per the rule **"when in doubt, use `main`."**

## What each branch is

| Branch | Character |
|---|---|
| `main` (`2d29f1d`) | Complete, tested port: 391/391 TS files ported, **973 passing tests**, 59/59 simulations run in series, cross-validated numerically against the TypeScript engine. |
| `main-lb` (`183ec49`) | Earlier **stub scaffold** (2 commits). Only the core DES foundation was hand-ported; most library files are 0–14 line stubs (e.g. `lp.rs` 14 vs `main`'s 1447; `precision.rs` empty vs 245). Its distinctive ideas are an idiomatic layout (`src/bin/*.rs` mains, `tests/*.rs`) and a typed numeric/error policy. |

## Strategy (chosen with the user)

- **Layout: hybrid.** Keep `main`'s tested module layout (`src/des/main_*.rs`, wired
  `src/des/test/*.rs`) as the base, **and** add thin `src/bin/*.rs` wrappers so every
  entrypoint is runnable via `cargo run --bin <name>` (adopting `main-lb`'s best
  structural idea without disturbing working/tested code).
- **Content: `main` only.** `main` is the basis everywhere; `main-lb`'s stub
  foundation, docs, and duplicate tests are not pulled in.

## File-by-file reconciliation (592 differing files)

### 1. Files only in `main` — 168 files → **KEPT**
`main` is the superset. These are full implementations, `main_*` modules, and wired
`src/des/test/*.rs` suites that `main-lb` never ported.

### 2. Files in both but differing — 259 files → **KEPT `main`'s version**
`main-lb`'s overlapping files are stubs or partial foundation; `main`'s complete,
tested implementations win (reinforced by "when in doubt, use `main`").

### 3. Files only in `main-lb` — 165 files → reconciled as follows

- **`src/bin/*.rs` (95): IDEA ADOPTED, content rewritten.** `main-lb`'s bins were
  empty scaffolds (`fn main() -> Result<()> { Ok(()) }`). Replaced with **95 real thin
  wrappers** that delegate to the corresponding library module
  (`des_engine::des::main_*::run` / `des_engine::des::runners::*::run`; the 7 runners
  that return `i32` are wrapped with `std::process::exit`). Plus a bare `main` bin →
  `des_engine::des::main::run`. The existing `run_all_simulations` bin is retained
  (96 bins total).
- **`tests/*.rs` (60): NOT pulled.** Equivalent — and fixed/wired — versions already
  live in `main` as `src/des/test/*.rs` (part of the 973-test suite). `main-lb`'s are
  duplicates/stubs of the same tests against stub modules.
- **Foundation / docs / tooling: NOT pulled** (mine-only): `src/numeric.rs`,
  `src/core.rs`, `src/migration.rs`, `src/des/entity_conn_ts/*`,
  `src/des/ws_server/mod.rs`, `MIGRATION_MANIFEST.md`, `MIGRATION_STATUS.md`,
  `README.md`, `tools/migrate_ts_tree.js`. `main` keeps its own live numeric layer
  (`des::shared::precision`, `rust_decimal` + `num_rational` + Kahan) and structure.

## Result

`main-merged` = `main` (complete + tested) + 95 `src/bin/*.rs` entrypoint wrappers.
Verified: `cargo build --bins` clean (96 binaries, 0 warnings); `cargo test --lib`
= **973 passed, 0 failed, 1 ignored**.

To inspect the raw differences:

```bash
git diff --name-status main origin/main-lb   # A = only in main-lb, D = only in main, M = differing
```
