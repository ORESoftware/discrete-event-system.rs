# Todos

## Explore real-time soccer embeddings and vector search

Problem framing: capture 22 on-field players plus the ball, including position,
velocity, and acceleration, then use similarity search to recognize states that
look like prior goal-producing moments:

- a great pass that leads to a goal
- a great dribble that leads to a goal
- a great shot that leads to a goal

The training/data goal is to run simulation after simulation after simulation,
learn what actually led to goals, and store vector snapshots only for goals that
were scored cleanly through strategy rather than luck. The saved vector should
encode the actionable state/action pattern that led to the goal, not merely the
fact that the ball eventually crossed the line.

Best current read: this is likely feasible inside a 250 ms soft deadline if the
embedding is computed locally, the vector index is in memory, and the main game
loop never waits for the result. It is not feasible if "embedding" means calling
a remote model/API during live play, and it is risky if every tick performs an
external vector-database round trip. Treat this as an asynchronous, best-effort
decision hint system.

Data-mining loop:

- Run many simulated matches and persist event traces, not just final scores.
- When a goal happens, rewind the rolling trace and extract the sequence of
  decision/action snapshots that plausibly caused the goal.
- Classify the goal as strategic or luck-driven before adding it to the useful
  vector corpus. Strategic examples are things like a planned through ball, a
  defender-beating dribble, a cutback into space, or a shot created by spacing
  and pressure. Luck-driven examples are things like random deflections, keeper
  mistakes, own goals, loose-ball chaos, or low-quality shots that scored anyway.
- Store the clean strategic examples as labeled rows: event type, vector,
  ball-location bucket, possession phase, action actor, target actor if any,
  time-to-goal, score delta, and replay pointer.
- Keep rejected/lucky goals too, but in a separate analysis set. They are useful
  for debugging and for teaching the model what not to imitate.
- Periodically rebuild or refresh the in-memory search index from the accepted
  strategic-goal corpus.

Recommended architecture:

- Keep a rolling ring buffer of recent match snapshots, probably 5-15 seconds.
- On every decision-relevant tick, write a compact normalized state vector:
  23 entities * position/velocity/acceleration. In 2D this is 23 * 6 = 138 f32s
  before extra context.
- Normalize coordinates around the opponent's goal as the canonical orientation.
  The distance to goal, angle to goal, goal-facing velocity, and ball movement
  toward/away from goal are probably more meaningful than raw absolute pitch
  coordinates. Always rotate/flip examples so "toward the opponent's goal" means
  the same thing for both teams and every half.
- Decide whether player ordering should be identity-based, role-based, or
  nearest-to-ball based. For tactical similarity, role/nearest ordering is often
  more useful than fixed player identity.
- Store raw snapshot data and computed vectors side by side with an embedder
  version. This lets the embedding scheme change without losing replay data.
- When a goal occurs, rewind through the ring buffer and extract event windows,
  not just isolated frames. Good key moments:
  - pass: passer receive/scan, lane opens, pass release, receiver first touch
  - dribble: possession start, defender engagement, acceleration/cut, defender
    beaten, final action
  - shot: pre-shot body/ball state, defensive pressure, keeper position, strike
- Insert newly labeled goal examples into the searchable index asynchronously,
  outside the frame-critical path.

Embedding/search strategy:

- Start with handcrafted numeric vectors, not learned embeddings. The first goal
  is to prove latency and usefulness.
- A 1536-element embedding-style vector has plenty of room for the 22 players'
  position/velocity/acceleration plus the ball's position/velocity/acceleration,
  and also leaves room for derived tactical features. The key is that it should
  be a local vector/projection for game physics, not a remote text-embedding call
  in the live loop.
- Do not blindly pad 138 raw features out to 1536 dimensions. Either keep the
  compact vector for exact search, or learn/project into 1536 dimensions only if
  the larger space improves retrieval quality enough to justify the cost.
- Keep separate indexes or filters for pass, dribble, and shot examples.
- Use a cascade: cheap feature checks first, then vector search for plausible
  candidates only.
- Use the ball location as the first lookup key. In opponent-goal-oriented
  coordinates, quantize the ball into spatial buckets such as pitch grid cells,
  distance-to-goal bands, and angle-to-goal sectors. Search the current bucket
  plus nearby buckets, then run vector similarity only over that much smaller
  candidate set.
- If the full corpus is about 50k rows, keep those rows in memory for the
  real-time worker. Use the database as the durable store, not the per-decision
  lookup path. The in-memory index can map ball-location buckets to candidate
  row ids, then exact-scan only those candidates.
- A realistic ball-location seed might reduce 50k rows to roughly 5k candidates.
  That changes the live query from "scan the whole corpus" to "scan a small,
  tactically relevant slice."
- Include ball velocity/acceleration in the prefilter when useful: a ball moving
  toward goal, wide-to-central, or backward-to-reset should seed different
  candidate pools even if the current ball position is similar.
- Keep the ball-location filter soft, not absolute. Great passes and dribbles
  can start in one bucket and finish in another, so the lookup should include
  neighboring cells and temporal-window metadata.
- Query on decision moments, possession changes, pass/shot consideration, or at
  5-10 Hz. Do not query every render frame unless benchmarks prove it is cheap.
- Run queries on a worker thread through a bounded/lock-free channel. Attach a
  request id and simulation timestamp; if the result comes back stale or after
  the 250 ms deadline, ignore it.
- Prefer in-process search for the real-time path. A ts-vector-style prototype
  is fine for exploration, but the Rust game loop should benchmark an in-memory
  exact scan and an ANN index before depending on an external service.

Compute and storage estimates:

- A simple 2D vector is about 138 f32s, or 552 bytes per snapshot before
  metadata. 100k examples is roughly 55 MB of vector data; 1M examples is
  roughly 552 MB plus index overhead.
- A 1536 f32 vector is about 6 KB per snapshot. 100k examples is roughly 614 MB;
  1M examples is roughly 6.1 GB before index overhead. That is still possible on
  a workstation/server, but it pushes the design toward ANN, filtering, or a
  smaller first-stage vector.
- The 138-float representation is the better first implementation target. For
  50k rows, the raw vector data is only about 27.6 MB, which is small enough to
  keep memory-resident. A quick local exact top-k scan over 50k x 138 f32s took
  about 5 ms p95 in plain Node, so the math itself should fit easily inside a
  250 ms budget.
- If ball-location bucketing narrows the candidate set from 50k to 5k rows, the
  raw vector data being scanned is only about 2.76 MB. A quick local exact top-k
  scan over 5k x 138 f32s took about 0.5 ms p95 in plain Node, which is
  comfortably below the 250 ms deadline.
- If those 50k rows live in a database, the answer depends on layout and query
  path. A local, indexed, memory-warm vector table should still be plausible
  under 250 ms. A cold scan from disk, JSON/row-by-row decoding, or a remote DB
  round trip could burn the budget even though the distance calculation is cheap.
- Exact kNN may be fast enough for 10k-100k vectors and may even pass at 1M on
  good hardware with SIMD for compact vectors, but exact search over 1M x 1536
  floats is much less likely to stay comfortably under the real-time budget.
- ANN/HNSW-style indexing should be considered once the corpus grows or if p95
  latency is too high. Index updates can happen in the background; live queries
  can use the latest immutable snapshot of the index.
- The useful target should be stricter than 250 ms. Aim for p95 under 25-50 ms
  and p99 under 100 ms so the result still matters tactically.

Modeling concerns:

- A single frame may not distinguish "great pass" from "ordinary pass". Temporal
  windows are probably the real embedding unit.
- Goal-only positives will bias the system, and lucky goals will poison the
  retrieval set if treated as good strategy. Capture negative and neutral
  examples too: similar pass/dribble/shot setups that did not lead to goals.
- The classifier for "clean strategic goal" can start as rules plus manual
  review and later become learned. Useful signals include defender pressure,
  pass lane quality, shot quality, xG/xT improvement, number of controlled
  touches before the goal, and whether the scorer's team intentionally possessed
  the ball through the sequence.
- Rank matches by more than vector distance: event type, pitch zone, possession
  phase, pressure, expected-threat/xG delta, time-to-goal, and whether the same
  action was actually available.
- Keep the online system explanatory where possible: "similar to prior cutback
  goal pattern" is more useful than just a nearest-neighbor id.

Benchmark plan:

- Add a small soccer-vector benchmark that generates replay-derived and random
  vectors at dimensions 96, 138, 256, and 1536.
- Test corpus sizes: 10k, 100k, 1M, and 5M vectors.
- Measure exact scan, approximate search, index insertion, worker-channel
  overhead, and result staleness.
- Measure ball-location prefilter quality: average candidate count, missed
  relevant examples, and latency improvement for each bucket/radius scheme.
- Report p50/p95/p99 latency for k=5, k=20, and k=50 while the soccer sim is
  running.
- Declare the real-time path acceptable only if main-thread blocking is zero and
  p99 worker response time stays below the stale-result threshold.

Likely conclusion: vector search itself is probably not the hard part if vectors
are compact, local, and indexed in memory. The harder parts are mining enough
simulations, separating clean strategic goals from luck, defining the right
temporal embedding, collecting negative examples, and making sure the worker
result is useful before the game state has moved on.
