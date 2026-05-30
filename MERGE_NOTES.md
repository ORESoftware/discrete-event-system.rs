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

**Every differing file has an explicit, recorded decision in
[`RECONCILIATION.tsv`](RECONCILIATION.tsv)** (one row per file: `status`, `path`,
`lb_added`, `main_removed`, `decision`, `basis`). The categories below summarise it.

### 1. Files only in `main` — 168 files → **KEPT** (`KEEP`)
`main` is the superset. These are full implementations, `main_*` modules, and wired
`src/des/test/*.rs` suites that `main-lb` never ported.

### 2. Files in both but differing — 259 files → **KEPT `main`'s version** (`KEEP-main`)
Split by how much unique content `main-lb` actually carried (via `git diff --numstat`):

- **226 files: `main-lb` side is a stub / alt-phrasing** of the same logic → keep `main`.
- **33 files: `main-lb` side is *substantive*** (≥ 40 unique lines — the hand-ported
  DES foundation: `validation.rs`, `network_flow.rs`, `ode.rs`, `quadrature.rs`,
  `random_variables.rs`, `station.rs`, `transform_entity.rs`, `runner.rs`, `prng.rs`,
  …). **These were genuinely reviewed file-by-file** (5 parallel diff reviews of
  `git show main:… vs origin/main-lb:…`). Verdict: **all 33 → keep `main`.** `main-lb`'s
  extra lines are an *alternate architecture* (`crate::core`, `serde`, `Result`/
  `thiserror`, `DesDecimal`, free functions) — not additive improvements, and adopting
  them would replace `main`'s tested, cross-validated behaviour. Marginal `main-lb`-only
  ideas that were explicitly considered and rejected as unsafe/architecture-bound:
  `Preconditions::equal` (unused), `ground_truth`/`external_reference_validator`
  (intentionally deferred, untested), `validate_args`/`validate_graph` (bundled with a
  `Result` API + `DesDecimal`), `BTreeMap` control ordering in `html_player`.

### 3. Files only in `main-lb` — 165 files → reconciled as follows

- **`src/bin/*.rs` (95): IDEA ADOPTED, content rewritten** (`ADOPT-idea`). `main-lb`'s
  bins were empty scaffolds (`fn main() -> Result<()> { Ok(()) }`). Replaced with **95
  real thin wrappers** that delegate to the corresponding library module
  (`des_engine::des::main_*::run` / `des_engine::des::runners::*::run`; the 7 runners
  that return `i32` are wrapped with `std::process::exit`). Plus a bare `main` bin →
  `des_engine::des::main::run`. The existing `run_all_simulations` bin is retained
  (96 bins total).
- **`tests/*.rs` (60): reconciled by content.**
  - **49 are 8–11 line placeholders** (`SKIP`) — real, fixed/wired equivalents already
    live in `main` as `src/des/test/*.rs` (part of the test suite).
  - **11 are substantive** (`HARVEST`) — reviewed scenario-by-scenario against my
    counterparts. Most were already met or exceeded by `main`. Genuinely-unique,
    high-value cases were **ported into `main`'s suite, adapted to `main`'s API**:
    - `src/des/test/output_routing_policy_test.rs` (was empty) → 5 tests: RR rotation
      `ABCABCA`, default policy, unknown-accept cursor, and `PerIndividualProcessor`
      round-robin (2/2/2) vs ordered (6/0/0) distribution through real `EntitySink`s.
    - `src/des/test/probability_decision_test.rs` (new) → deterministic 25/75 branch
      selection at RNG boundaries (draw 0.20→branch 0, 0.90→branch 1).
    - `src/des/test/preconditions_test.rs` → `PreconditionError` structured-field
      assertions (`model`/`param`/`condition`/`observed`).
- **Foundation / docs / tooling (10): NOT pulled** (`SKIP`, mine-only): `src/numeric.rs`,
  `src/core.rs`, `src/migration.rs`, `src/des/entity_conn_ts/*`,
  `src/des/ws_server/mod.rs`, `MIGRATION_MANIFEST.md`, `MIGRATION_STATUS.md`,
  `README.md`, `tools/migrate_ts_tree.js`. `main` keeps its own live numeric layer
  (`des::shared::precision`, `rust_decimal` + `num_rational` + Kahan) and structure.

## Result

`main-merged` = `main` (complete + tested) + 95 `src/bin/*.rs` entrypoint wrappers
+ 7 harvested tests from `main-lb`'s substantive suites.
Verified: `cargo build --bins` clean (96 binaries); `cargo test --lib`
= **980 passed, 0 failed, 1 ignored**.

To inspect the raw differences and the per-file decisions:

```bash
git diff --name-status main origin/main-lb   # A = only in main-lb, D = only in main, M = differing
column -t -s $'\t' RECONCILIATION.tsv | less  # the full 592-file decision manifest
```
