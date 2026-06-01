# Scheduler-enforced Fibonacci

An independent variant of `src/des/main_fibonacci_recursion.rs`. Same recurrence,
same building blocks — but the per-tick **execution order is derived from the
graph and validated by a scheduler**, instead of resting on an implicit,
hand-ordered list of nodes.

- `scheduler.rs` — `DeterministicScheduler`, the reusable "enforcer" (brain).
- `model.rs` — the Fibonacci graph wired through the scheduler, plus a
  `RecordingSink` so the produced sequence can be asserted exactly.
- `mod.rs` — module wiring (`run`, `build_and_run`, `FibonacciRun`).

Run it:

```bash
cargo run --bin main_fibonacci_scheduled
cargo test fibonacci_scheduled
```

---

## The problem this fixes

The original model steps its nodes by iterating a `Vec<dyn Entity>` in insertion
order:

```text
program = [A, B, C, D]
for _ in 0..100 { for v in &program { v.do_time_step(dt) } }
```

The graph is `A(source) → B(processor) → C(splitter) → D(sink)` plus a feedback
edge `C → B`. Fibonacci only comes out **if `C` is stepped after `B`** in the
same tick: `B` emits a sum into `C`, and because `C` runs later in the same pass
it broadcasts that sum back into `B`'s queue, ready for the next tick.

That ordering is an **implicit, unchecked invariant**. Reorder the vector, insert
a node, or add an edge, and the recurrence silently produces wrong numbers — no
error, no panic, no signal. In a system where "order really matters and cannot be
fuzzy," that is exactly the failure mode you want a machine to catch.

## The fix: declare the graph, derive the order, validate, then freeze

The scheduler turns the implicit invariant into an explicit, enforced contract.
You declare nodes and two kinds of edges; it computes and locks the order.

### 1. Two edge kinds

| Edge kind     | Meaning                                   | Declared with    |
| ------------- | ----------------------------------------- | ---------------- |
| **forward**   | intra-tick dataflow (runs "downhill" now) | `wire_forward`   |
| **feedback**  | cross-tick edge (value consumed next tick)| `wire_feedback`  |

For Fibonacci:

```text
forward:  A → B,  B → C,  C → D
feedback: C → B          (the recurrence)
```

Wiring goes **through the scheduler**, which both records the edge *and* performs
the physical `add_out_connection`. The topology used to compute the order is
therefore the same object graph that is actually connected — they cannot drift.

### 2. Derive the order (deterministic topological sort)

`freeze()` runs **Kahn's algorithm** over the *forward* edges only:

1. Compute each node's in-degree from forward edges.
2. Repeatedly emit a node whose in-degree is 0, decrementing its successors.
3. **Tie-break deterministically:** when several nodes are simultaneously ready,
   emit the one with the smallest registration index (implemented with a
   min-heap). The order is a pure function of the declared graph — never of hash
   iteration order or insertion timing.

For the Fibonacci graph this yields the unique order `A, B, C, D`, with ranks
`0, 1, 2, 3`. Registration order is irrelevant; `model.rs` could register the
nodes in any order and still get `A, B, C, D`.

### 3. Validate (fail fast, loudly)

Before any tick runs, `freeze()` proves three things and **panics with a
descriptive message** otherwise (or returns `Err` via `try_freeze`):

1. **At least one node** is registered.
2. **The forward graph is a DAG.** If the topological sort cannot place every
   node, there is a forward cycle — meaning some node would have to run before
   itself within a tick. That is unsatisfiable, so it is rejected. (This is the
   check that catches "I accidentally made the recurrence a forward edge": declare
   `C → B` with `wire_forward` and `freeze()` refuses, because `B → C → B` is a
   cycle.)
3. **Every feedback edge is a true back edge** (`rank(from) >= rank(to)`). If an
   edge marked feedback actually points "downhill," it was misclassified and
   should have been forward — also rejected.

Additional guards: duplicate node ids panic at `register`; self-edges are
rejected at `wire`; stepping before `freeze` panics.

### 4. Step (exactly once per tick, in the frozen order)

`step()` advances **every node exactly once, in rank order**, then increments the
tick counter. `run(n)` repeats it `n` times. There is no RNG and no variable
ordering, so the run is bit-for-bit reproducible (`run_is_deterministic_across_runs`
asserts this).

## Why this is enough to guarantee the recurrence

The correctness argument is now structural rather than incidental:

- Within a tick, data only flows along **forward** edges, and the execution order
  is a topological order of those edges. So whenever a node runs, every upstream
  node has *already* run this tick — its inputs for this tick are ready.
- The single cycle in the system, `C → B`, is a **feedback** edge. The scheduler
  has proven `rank(C) ≥ rank(B)`, so `C` runs *after* `B` every tick. `B` emits
  the sum, `C` (running later, same tick) feeds it back, and it sits in `B`'s
  queue for the next tick. That is precisely the recurrence.

If anyone later breaks the wiring — reorders, inserts a node that creates a
forward cycle, or mislabels the feedback edge — `freeze()` fails before the
simulation produces a single (wrong) number.

## What the model proves at runtime

`model.rs` runs the graph and the tests assert:

- `enforced_order_is_topological` → derived order is exactly `["A","B","C","D"]`.
- `produces_exact_fibonacci_sequence` → the sink records `1, 2, 3, 5, 8, 13, …`
  (each term the sum of the previous two; the seeds are the source's `0, 1`).
- `processor_queue_stays_bounded` → after warmup `B` retains exactly two tokens
  per tick and never approaches its own overflow guard.
- `misdeclaring_feedback_as_forward_is_rejected` → declaring `C → B` as forward
  makes `freeze()` reject the cycle — the failure the original implicit ordering
  could never detect.

## Scope / relationship to the rest of the engine

This is the **time-stepped** paradigm (a fixed `Δt` tick, like the main entity
network), hardened with an explicit order enforcer. It is deliberately *not* the
`src/des/fel/` future-event-list engine: there is no stochastic inter-event
timing here — order, not sampled time, is the thing being controlled. The
scheduler is generic (`DeterministicScheduler` knows nothing about Fibonacci), so
any order-sensitive, time-stepped DES graph can reuse it.
