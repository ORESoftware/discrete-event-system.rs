# Todos

## Explore Real-Time Soccer Embeddings And Vector Search

Build a replay-mining subsystem for great attacking moments: passes, dribbles,
and shots that lead to goals. The soccer simulation already exposes useful
state for this because each player tracks position, velocity, acceleration,
jerk, role/team context, and recent position history.

Core direction:

- Keep a rolling 5-15 second ring buffer of compact match snapshots with event
  markers for pass release/reception, dribble start/end, shot release/contact,
  save/block/goal, possession changes, and pressure events.
- When a goal occurs, rewind the buffer and extract temporal windows, not just
  isolated frames. Label useful windows as great pass, great dribble, or great
  shot examples.
- Separate clean strategic goals from lucky goals before adding examples to the
  online retrieval corpus. Keep rejected/lucky goals in a separate analysis set.
- Store raw snapshot windows beside computed vectors and an embedder version so
  the feature scheme can evolve without losing replay data.

Embedding and indexing plan:

- Do not call remote embeddings or vector databases inside the live match loop.
  Use deterministic local numeric vectors for real-time search; remote models
  can be used offline for metadata or analysis.
- A direct 2D vector for 22 players plus the ball is `23 * 3 * 2 = 138` f32s
  for position, velocity, and acceleration. Add derived tactical features only
  when benchmarks show they improve retrieval.
- Canonicalize every vector so the attacking team moves toward the same
  opponent-goal orientation. Include distance/angle to goal, pressure, passing
  lane openness, shot lane openness, receiver/defender time-to-ball, possession
  phase, action kind, and previous action kind.
- Use ball location as the first-stage lookup key: pitch grid cell,
  distance-to-goal bucket, side/central lane, possession phase, and action
  label. Search nearby buckets, then run vector similarity on the narrowed
  candidate set.
- Keep separate indexes or filters for pass, dribble, and shot examples, plus a
  combined fallback index.

Real-time constraints:

- Never block the simulation tick. Run searches on a worker thread with bounded
  channels, request ids, simulation timestamps, and a hard stale-result policy.
- Treat 250 ms as a soft upper bound, but aim for p95 under 25-50 ms and p99
  under 100 ms so recommendations still matter tactically.
- Use immutable/read-mostly index snapshots in live play. Queue new goal moments
  and merge them between matches, during stoppages, or by atomically swapping in
  a rebuilt index.
- Start with bucketed brute-force scans over contiguous f32 vectors before
  adopting heavier ANN structures. Add HNSW/IVF-style indexes only when corpus
  size or p95 latency requires it.

Benchmark plan:

- Generate replay-derived and random vectors at dimensions 96, 138, 256, and
  1536.
- Test corpus sizes of 10k, 100k, 1M, and 5M vectors.
- Measure exact scan, approximate search, index insertion, channel overhead,
  stale-result rate, and search quality after ball-location filtering.
- Report p50/p95/p99 latency for k=5, k=20, and k=50 while the soccer sim is
  running.

Likely conclusion: vector search is probably feasible if vectors are compact,
local, bucketed, and memory-resident. The harder work is mining enough
simulations, identifying clean strategic goals, collecting negative examples,
choosing the right temporal embedding, and making the returned hint useful
before the match state has moved on.
