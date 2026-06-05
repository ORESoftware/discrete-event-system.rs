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

## 2026-06-05 Repo Update: Existing Moment Vectors And Correlation Path

The newer repo is much closer to this idea than the original sketch. There is
already a local soccer moment system in `src/des/general/soccer.rs`:

- `SoccerRealtimeSession::capture_moments_for_events` rewinds on `goal` events.
- It labels windows as `great_shot_to_goal`, `great_pass_to_goal`, and
  `great_dribble_to_goal` based on recent learning-history actions.
- `SoccerMomentWindow` stores summary metadata, action markers, tracking frames,
  `embedderVersion`, and a local `featureVector`.
- `SoccerMomentVectorIndex` builds bucketed in-memory search over those windows.
- `MatchConfig` already has `adversarial_embedding_exploitation_enabled` and an
  adversarial moment-memory limit, so the concept is already part of the match
  runtime rather than just a separate research note.

Important correction: the active repo vector is not only 138 elements. The
current embedder version, `soccer-moment-local-v4`, uses 8 sampled frames, with
`23 entities * 6 motion features = 138` features per frame, so the stored vector
length is `8 * 138 = 1104` f32s. That is probably the right direction: 138 is
the compact single-frame basis, while 1104 encodes the short action window that
distinguishes an ordinary pass/dribble/shot from a goal-producing sequence.

Performance read:

- A 50k corpus of 1104-float vectors is about 220 MB of raw f32 data, which is
  reasonable to keep resident in a worker-side cache.
- A quick local exact top-k scan over `50k x 1104` f32s was about 38 ms p95 in
  plain Node, still comfortably under the 250 ms soft deadline before game-load
  overhead.
- If ball-location, phase, label, and bucket filtering reduce 50k rows to about
  5k candidates, the scan is about 22 MB of vector data and measured around
  4 ms p95 locally.
- Therefore the calculation is likely real-time safe if the vectors are already
  memory-resident and queried from a worker thread. The risky path is reading
  JSONL or database rows on every decision.

Recommended next step is not to invent a new embedding layer. Extend the
existing moment system:

- Add a "clean strategic goal" score to moment capture. Today the capture path
  labels goal windows as great moments when the action type is present; it should
  also decide whether the goal was strategically earned or mostly luck.
- Store rejected/lucky goal windows too, but mark them separately so they do not
  pollute the positive retrieval corpus.
- Add negative windows: similar pass/dribble/shot sequences that did not become
  goals. Correlation quality depends on comparing clean positives against near
  misses, failed attacks, keeper saves, blocked shots, turnovers, and chaotic
  lucky goals.
- Add a Postgres analytical table for moment windows, separate from policy
  entries and completed runs. The current Postgres strategy has normalized
  learning runs, deltas, set-play episodes, and neural metrics, but moment
  windows still look JSONL-oriented.
- Keep Postgres as the durable analytical store and rebuild an in-memory
  `SoccerMomentVectorIndex` or equivalent contiguous-vector cache for live
  search. Do not query Postgres row-by-row inside the match loop.
- Periodically compute correlations by label, bucket, phase, action sequence,
  actor role, target role, pressure, pass/shot lane openness, possession
  continuity, and goal-delta ticks. Promote only patterns with good lift against
  negative examples.

Possible clean-strategy signals:

- Possession was controlled by the scoring team through the window.
- The action chain includes an intentional pass, dribble, or shot marker with
  positive tactical reward.
- Ball movement improves opponent-goal orientation instead of coming from a
  random rebound.
- Shot/pass lane openness improves before the decisive action.
- Defender pressure is beaten or moved, not merely absent due to simulation
  noise.
- The scorer/receiver was a plausible target before the goal.
- Reject or down-weight own goals, uncontrolled loose-ball chaos, keeper errors,
  random deflections, and very low-quality shots that happened to score.

Implementation shape:

- Use the existing goal-window capture as the hook.
- Version the vector schema separately from the clean-strategy classifier.
- Add moment rows with fields like run id, episode, label, team, player, target,
  start/end/event ticks, bucket fields, clean-strategy score, luck score,
  correlation tags, action markers, replay pointer, vector dimension, embedder
  version, and packed f32 vector bytes.
- Keep the live cache grouped by the existing bucket partition key: phase,
  ball macro cell, tactical cell, yards-to-goal bin, and central-lane bin.
- Search same-label/same-bucket first, then nearby buckets, then fallback wider
  only if candidates are sparse.
- Measure searched records, candidate records, p50/p95/p99 latency, hit quality,
  and stale-result rate while simulations run in parallel.

Bottom line: vector search/correlation is now a very natural fit for the repo
because the moment-window infrastructure already exists. The main work is data
quality: distinguish clean strategy from luck, collect negatives, persist moment
rows in Postgres, and keep a memory-resident worker index for live use.
