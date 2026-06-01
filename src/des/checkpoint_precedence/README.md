# Checkpoint-precedence ordering (token-level enforcer)

A second, complementary way to make a DES deterministic in *order*. The
`fibonacci_scheduled` `DeterministicScheduler` enforces **node order** (which
station runs when). This module enforces **token order**: which movable may pass
a given point in the network, relative to other movables — expressed as
happens-before constraints between tokens, referenced by UUID.

- `ledger.rs` — `PrecedenceLedger`: token stamps (UUID + `seq`), per-checkpoint
  clearances, and `validate()` (the precedence graph must be acyclic).
- `entities.rs` — `LabeledToken`, `OrderedTokenSource`, `CheckpointGate` (the
  BST-backed waiting room), and a `RecordingSink`.
- `model.rs` — the toy demo + tests.
- `task_dag.rs` — a **real computation** built on the same gate: dependency-ordered
  task execution (a build/job scheduler). See the section below.
- `mod.rs` — module wiring (`run`, `build_and_run`, `CheckpointRun`).

Run it:

```bash
cargo run --bin main_checkpoint_precedence   # the toy ordering demo
cargo run --bin main_task_build_order        # dependency-ordered task scheduler
cargo test checkpoint_precedence
```

---

## The idea

Sources stamp each token and let it declare, *by reference to other tokens*, what
must happen before it. A token says:

> "I may not pass checkpoint **C** until token **X** has cleared **C**."

A **checkpoint** is a gate station. It holds arriving tokens and only releases one
when all of its declared predecessors have already cleared that checkpoint —
releasing them in a deterministic order. No statistics, no fuzz: the same inputs
always produce the same order.

This reuses infrastructure that already exists on every movable:

- `moving_uuid` — the stamp tokens are referenced by.
- `stations_visited` / `add_visited_station(name)` — the record of which
  checkpoints a token has cleared.

So "stamp a UUID and reference it" and "did predecessor X clear checkpoint C?" are
both first-class already.

## The pieces

### 1. Stamp + constraints (`PrecedenceLedger`)

Each token has a `TokenSpec { uuid, seq, payload, requirements }`:

- `uuid` — a stable, caller-chosen label (e.g. `"T1"`) so it can be referenced.
- `seq` — a monotonic stamp assigned at the source; the **deterministic tie-break**
  when several tokens are simultaneously eligible.
- `requirements` — a list of `Requirement { predecessor, checkpoint }`: pairwise
  happens-before references to other tokens.

The ledger is shared (`Rc<RefCell<…>>`): the source registers specs; the gate
reads them and records clearances (`checkpoint -> {cleared uuids}`).

### 2. Validate before running

`PrecedenceLedger::validate()` fails fast on:

1. a token referencing **itself**;
2. a reference to an **unregistered** predecessor UUID;
3. a **cycle** in the happens-before graph at any checkpoint — e.g. `A` after `B`
   *and* `B` after `A`. Such an order is unsatisfiable, so it is rejected. This is
   the token-level analog of the node scheduler's forward-cycle check, and it uses
   the same deterministic Kahn topological sort.

### 3. The gate is a balanced BST keyed by `seq`

`CheckpointGate` parks arriving tokens in a `BTreeMap<seq, token>` — a balanced
BST. Ordered iteration yields the lowest-`seq` token first, and insert/remove are
`O(log n)`. That is exactly the "a BST might help" structure: it provides the
deterministic, ordered release with cheap updates.

Each tick the gate runs a release loop:

1. Scan the BST in ascending `seq` and pick the **first token whose constraints
   are satisfied** (all of its predecessors at this checkpoint have cleared).
2. Release it: stamp `add_visited_station(C)`, record the clearance in the ledger,
   forward it downstream.
3. Releasing a token can unblock a *lower*-`seq` successor, so re-scan and repeat
   until nothing is eligible.

Predecessor lookups by UUID are `O(1)` via the ledger's hash map; ordered release
is `O(log n)` via the BST.

## What the demo shows

`source(emits T1..T5) → gate "C" → sink`. Payloads `10..50`. Constraints (a
*partial* order, not a total one):

- `T1` may clear `C` only after `T4`;
- `T2` may clear `C` only after `T5`.

Tokens **arrive** in their natural order, but the gate **releases** them in the
unique order that satisfies the constraints (lowest `seq` otherwise):

```text
arrive : T1, T2, T3, T4, T5
release: T3, T4, T1, T5, T2      (payloads: 30, 40, 10, 50, 20)
```

Walkthrough (the gate runs right after the source each tick, so a token can be
released the same tick it arrives):

| tick | arrives | waiting (by seq) | released this tick | why |
| ---- | ------- | ---------------- | ------------------ | --- |
| 1 | T1 | {T1} | — | T1 blocked (needs T4) |
| 2 | T2 | {T1, T2} | — | T1 needs T4, T2 needs T5 |
| 3 | T3 | {T1, T2, T3} | **T3** | T3 unconstrained; T1, T2 still blocked |
| 4 | T4 | {T1, T2, T4} | **T4, T1** | T4 clears → unblocks T1 (lower seq) |
| 5 | T5 | {T2, T5} | **T5, T2** | T5 clears → unblocks T2 |

The tests assert this order exactly and that two runs are identical
(`run_is_deterministic_across_runs`).

## Failure modes (the point of an enforcer)

- **Cyclic constraints** → `validate()` returns an error before any tick runs.
- **Reference to an unknown / self token** → rejected at `validate()`.
- **A token reaches the gate without being registered** → the gate panics, rather
  than silently passing it in arrival order.

## Applied to a real computation: dependency-ordered task execution

The toy demo above is artificial. `task_dag.rs` points the exact same gate at a
genuine computation: **topological scheduling** — "run every task only after its
dependencies finish." That is a fundamental computation (build systems, job
schedulers, instruction scheduling, spreadsheet recalculation, PERT/CPM project
plans), and it *is* what the gate already does:

- each task is a token (its name is the UUID);
- "task X depends on task Y" is the constraint "Y must clear the `BUILD`
  checkpoint before X";
- the gate releases tasks in a deterministic, dependency-respecting order;
- the ledger's `validate()` rejects a **circular dependency** before anything runs.

Nothing in `entities.rs`/`ledger.rs` changed — only the graph. The demo submits a
small multi-target build in **alphabetical** order (which is *not* a runnable
order — `compile-cli` is listed before `fetch`), and the gate executes it
correctly:

```text
submitted (alphabetical): compile-cli, compile-core, compile-gui, fetch,
                          gen-proto, integration-test, link, package, unit-test
executed  (topological) : fetch, compile-core, compile-gui, gen-proto,
                          compile-cli, link, unit-test, integration-test, package
```

The build graph:

```text
  fetch ─┬─> compile-core ─┬─> compile-cli ─┐
         │                 ├─> compile-gui ─┼─> link ─┐
         └─> gen-proto ────┘                │         ├─> integration-test ─> package
                           compile-core ───────> unit-test ──────────────────┘
```

Tests (`task_dag.rs`):

- `submitted_order_is_alphabetical_not_topological` — the input order is invalid;
- `gate_executes_in_valid_topological_order` — the output respects every edge;
- `execution_order_is_deterministic` — exact, reproducible order across runs;
- `circular_dependency_is_rejected` — `a → b → a` is refused by `validate()`.

`build_and_run_graph(graph, ticks)` is public, so any `(task, deps)` list can be
scheduled this way.

## Other computations that could benefit

The same token-precedence idea fits any computation whose correctness depends on a
*relative order* of items, not on sampled time:

- **Instruction / operation scheduling** — respect data hazards (a read after a
  write must follow the write).
- **Spreadsheet / dataflow recalculation** — recompute a cell only after its
  precedents.
- **Distributed event ordering** — Lamport happens-before between messages.
- **Manufacturing / assembly lines** — parts that must be processed in a fixed
  relative order at a station.

Computations that are *order-insensitive for correctness* (e.g. Bellman-Ford
relaxation, Monte-Carlo sampling) do **not** need this; and where a single global
priority already suffices (e.g. Dijkstra's min-distance settling), a plain
priority queue — which the engine already uses in `shortest_path_des` — is enough.
The gate earns its keep when the ordering is a **partial order over many tokens**
expressed by reference.

## Relationship to the node scheduler

The two layers compose:

- `fibonacci_scheduled::DeterministicScheduler` — *node* order (topological,
  validated, frozen).
- `checkpoint_precedence::CheckpointGate` — *token* order (per-checkpoint
  happens-before, validated, BST-ordered release).

A model that needs both deterministic station execution *and* deterministic
token traversal can use a `DeterministicScheduler` to drive ticks and place one or
more `CheckpointGate`s in the network. This module keeps the node graph a trivial
linear chain on purpose, so the spotlight is on the token-level mechanism.
