# Functional programming conformance

This repository is checked against the house functional-programming guidelines.
Functional programming here means nine specific things:

- **explicit inputs** — what a function needs arrives through its parameters
- **explicit outputs** — what a function produces leaves through its return type
- **immutable values** — bindings and fields do not change after construction
- **pure transformations** — same input, same output, no observable effect
- **typed errors** — failure is a value in the signature, not an escape
- **explicit state transitions** — state changes are named and returned
- **composition** — small named steps combined, rather than one long body
- **effects pushed outward** — I/O, clocks, randomness and logging live at the edge
- **illegal states excluded by types** — the compiler rejects what must not happen

Stateful code is not exempt from all of this. Websocket handlers, TCP
connections, stateful clients and actor loops legitimately hold mutable state, and
the scanner relaxes the mutability rules for modules whose path marks them as such
(`ws/`, `socket/`, `conn/`, `session/`, `pool/`, `cache/`, `stream/`, `actor/`,
`fsm/`, `state_machine/`). Everything else — typed errors, exhaustive matching,
composition, effects at the edge — still applies there. Likewise, modules that
*are* the outward edge (`main`, `bin/`, `effects/`, `io/`, `adapters/`, `infra/`,
`transport/`, `handlers/`, `routes/`, `db/`, `telemetry/`) are allowed to perform
effects: that is the point of pushing effects outward.

## Running the check

```sh
python3 tools/fp-conformance/fp_conformance.py .                    # report
python3 tools/fp-conformance/fp_conformance.py . --limit 200        # more detail
python3 tools/fp-conformance/fp_conformance.py . --json /tmp/fp.json
```

Stdlib Python 3 only — no toolchain, no dependencies, no network — so it runs
identically on a laptop and on a CI runner.

## The budget, and why CI is not red today

`tools/fp-conformance/budget.json` records the per-rule counts at the moment this
check was introduced: **13,913 findings across 661 files
and 374,266 lines**. CI compares against that budget and fails only when a
rule's count *increases*. The existing backlog blocks nobody; new violations do.

The budget is a ratchet. It should only ever move down. When you clear a class of
violation, re-baseline in the same commit as the fix:

```sh
python3 tools/fp-conformance/fp_conformance.py . \
    --write-budget tools/fp-conformance/budget.json
```

Raising the budget to turn CI green defeats the whole mechanism. Fix the code.

## Baseline for this repository

| rule | count | severity | principle | what it flags |
|---|---:|---|---|---|
| `RS001` | 6334 | warn | immutable values | mutable local binding (`let mut`) |
| `RS003` | 3098 | error | typed errors | panic-based control flow (`unwrap`/`expect`/`panic!`) |
| `RS007` | 1872 | warn | effects pushed outward | direct stdout/stderr effect in library code |
| `RS008` | 941 | warn | explicit outputs | `&mut self` method returning unit |
| `RS006` | 524 | warn | pure transformations | interior mutability outside a stateful module |
| `XX002` | 504 | warn | explicit outputs | long function body |
| `RS004` | 430 | warn | illegal states excluded by types | wildcard match arm defeats exhaustiveness |
| `XX001` | 182 | warn | composition | oversized module |
| `RS009` | 15 | error | illegal states excluded by types | `unsafe` block |
| `RS002` | 7 | error | explicit state transitions | module-level mutable / shared-interior global |
| `RS005` | 6 | warn | typed errors | untyped/erased error in a signature |

## How to clear the top offenders

### `RS001` — mutable local binding (`let mut`)

*immutable values* · 6334 occurrences at baseline

Rebind with `let`, fold with an iterator, or build the value with `collect()`/`fold()` instead of mutating in place.

### `RS003` — panic-based control flow (`unwrap`/`expect`/`panic!`)

*typed errors* · 3098 occurrences at baseline

Return `Result<T, E>` with a domain error enum and propagate with `?`; reserve panics for genuinely unreachable invariants proven by types.

### `RS007` — direct stdout/stderr effect in library code

*effects pushed outward* · 1872 occurrences at baseline

Emit through the ores-otel tracing layer so the effect lives at the outward edge and stays observable.

### `RS008` — `&mut self` method returning unit

*explicit outputs* · 941 occurrences at baseline

Return the new state (or a typed transition) instead of mutating in place and returning `()`.

### `RS006` — interior mutability outside a stateful module

*pure transformations* · 524 occurrences at baseline

Thread the value through as a parameter and return the updated value, rather than hiding mutation behind `RefCell`/`Cell`.

### `XX002` — long function body

*explicit outputs* · 504 occurrences at baseline

A body over 60 lines is usually several transformations. Extract named, individually testable steps and compose them.

### `RS004` — wildcard match arm defeats exhaustiveness

*illegal states excluded by types* · 430 occurrences at baseline

Enumerate the remaining variants explicitly so adding a variant becomes a compile error.

### `XX001` — oversized module

*composition* · 182 occurrences at baseline

Split into focused modules; the guidelines call for modularisation rather than a single large entry point.

### `RS009` — `unsafe` block

*illegal states excluded by types* · 15 occurrences at baseline

Replace with a safe abstraction, or isolate behind a reviewed, documented module at the effect boundary.

### `RS002` — module-level mutable / shared-interior global

*explicit state transitions* · 7 occurrences at baseline

Move the value into an explicit state struct threaded through call sites, or confine it to an effects module at the outward edge.

## Language-native enforcement

The Python scanner is the portable floor — it runs everywhere and costs nothing.
The real type-level enforcement belongs to each toolchain, and those configs ship
in this tree:

- **Rust** — `[lints.clippy]` in `Cargo.toml`. Run `cargo clippy --all-targets`.
- **TypeScript** — `eslint.fp.config.mjs`. Run `npx eslint -c eslint.fp.config.mjs .`
  (needs `eslint`, `typescript-eslint` and `eslint-plugin-functional` as devDependencies).
- **Dart** — `analysis_options.fp.yaml`. Add `include: analysis_options.fp.yaml`
  to `analysis_options.yaml`, then run `dart analyze`.

Those steps are deliberately **not** in the CI job. A toolchain install costs far
more Actions minutes than the Python pass, and we are budget-conscious about
runner time. Run them locally, and in the nightly job on the sibling `-test` org.
