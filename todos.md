# Todos

## Evaluate soccer moment embeddings and real-time vector search

Explore a replay-mining subsystem for great attacking moments: passes, dribbles,
and shots that lead to goals. The current soccer simulation already has a useful
shape for this because each `PlayerAgent` tracks `position`, `velocity`,
`acceleration`, `jerk`, and recent position history, and the match tick is
`DEFAULT_DT_SECONDS = 0.1`.

Core idea:

- Keep a rolling ring buffer of compact world snapshots for the last N seconds
  of match time. A 10-15 second buffer is probably enough to capture the buildup
  to most goals, with explicit event markers for pass release, pass reception,
  dribble start/end, shot release, shot contact, save/block/goal, possession
  changes, and pressure events.
- When a goal occurs, rewind through that buffer and emit labeled training
  moments:
  - `great_pass_to_goal`
  - `great_dribble_to_goal`
  - `great_shot_to_goal`
- Store both the raw snapshot window and one or more numeric vectors derived
  from it. The raw snapshot is important because the vector will evolve as we
  learn what features matter.

Feature encoding thoughts:

- Do not start with a remote OpenAI embedding call in the real-time loop. It is
  too slow, non-deterministic, and network-dependent for a 250 ms budget. If
  OpenAI embeddings are useful, use them offline for descriptive metadata or
  post-match analysis.
- For match-time search, use a deterministic local numeric embedding. A single
  frame with 22 players plus ball has 23 entities. Position, velocity, and
  acceleration are 2D, so the direct vector is `23 * 3 * 2 = 138` floats. If we
  add jerk, it becomes `184` floats. A short temporal window of 8-10 keyframes
  still fits inside a 1536-float embedding budget (`138 * 10 = 1380`).
- Normalize orientation so every query attacks "upfield" toward the opponent
  goal. For the away team, mirror/rotate the pitch so the opponent goal is at the
  same canonical side as home. Store positions as distances relative to the
  opponent goal, ball, and ball carrier, not just absolute pitch coordinates.
- Sort or group players in a stable tactical order before vectorizing:
  ball/carrier first, nearest 3 teammates, nearest 5 opponents, goalkeeper,
  remaining teammates/opponents by role and distance to goal. Raw player ids make
  vectors brittle because the same tactical situation can be produced by
  different players.
- Include derived tactical features alongside raw p/v/a:
  - ball grid cell and distance/angle to goal
  - carrier distance/angle to goal
  - nearest defender distance and closing velocity
  - passing lane openness
  - shot lane openness and blocker distance
  - off-ball runner depth/width
  - expected time-to-ball for likely receiver and nearest defender
  - possession phase, action kind, and previous action kind

Indexing and search plan:

- Treat the user's "seed by ball location" idea as the first-stage index, not as
  the whole similarity search. Store every moment in coarse spatial buckets:
  attacking direction, ball macro/tactical/fine grid cell, ball distance-to-goal
  bucket, ball side/central lane, possession phase, and action label. At query
  time, fetch only moments from nearby buckets before vector search.
- Suggested lookup pipeline:
  1. Canonicalize current state to attacking-toward-opponent-goal coordinates.
  2. Compute cheap bucket keys from ball location and phase.
  3. Pull candidate ids from the same cell, neighboring cells, and nearby
     distance-to-goal buckets.
  4. Run ANN or brute-force cosine/L2 over only those candidates.
  5. Return the top K hints only if the worker finishes before the deadline.
- This two-stage shape matters because a full vector scan gets expensive as the
  moment library grows, but a grid seed can cut the candidate set by orders of
  magnitude. The ball location is especially strong because "great pass from own
  half", "cutback from the end line", and "shot at the top of the box" are
  different search neighborhoods even before looking at player arrangement.
- Keep separate indexes by label (`great_pass_to_goal`, `great_dribble_to_goal`,
  `great_shot_to_goal`) plus one combined index. Different action types should
  not compete too early because their similarity features have different weights.
- If this lives beside a TypeScript UI/service, a `ts-vector`-style in-memory
  index is a reasonable prototype target. If it lives inside this Rust simulator,
  keep the same data contract but use a local Rust index over contiguous `f32`
  vectors so live play does not cross a process or network boundary.

Real-time feasibility:

- The hard rule should be: never block the simulation tick. At `dt = 0.1s`, a
  250 ms search is already 2.5 ticks, so vector search belongs on a worker thread
  with a deadline. If the result arrives late, drop it.
- A 138-1536 float vector search over a few hundred or a few thousand candidates
  should be plausible inside 250 ms on a local worker if vectors are already
  computed, normalized, and resident in memory. The risky part is not math for
  one query; it is allocation, locking, cache misses, index updates, and
  accidentally involving IO/network calls.
- Runtime queries should use precomputed vectors and an immutable/read-mostly
  index snapshot. New goal moments can be appended to a write queue and merged
  into the searchable index between matches, during stoppages, or by swapping in
  a rebuilt index atomically.
- Start simple: benchmark brute-force over bucketed candidates before adopting a
  heavier ANN library. With 1536 dimensions, brute force against 1000 candidates
  is about 1.5 million multiply-adds, which is tiny compared with the 250 ms
  budget if implemented as tight contiguous `f32` arrays. If bucketed candidate
  counts stay low, brute force may be simpler and more predictable than HNSW.
- If the library grows large enough that bucketed brute force misses the budget,
  add ANN per bucket or per action label. HNSW/IVF-style indexes make sense once
  candidate counts are consistently in the tens or hundreds of thousands.

Implementation sketch:

- Add a `SoccerMomentSnapshot`/`SoccerMomentWindow` structure that captures:
  tick, clock, score, possession team, action label, ball p/v/a, and 22 player
  p/v/a/role/team/facing.
- Add `SoccerMomentEmbedding` with:
  `label`, `canonical_team`, `bucket_key`, `vector: Vec<f32>`, `raw_window_id`,
  `goal_delta_ticks`, and quality metadata.
- Add a background `SoccerSimilarityWorker`:
  - receives current query embeddings over a bounded channel
  - searches an immutable index snapshot
  - returns top K matches with elapsed time and deadline status
  - drops stale requests when a newer tick has superseded them
- Add microbenchmarks before gameplay integration:
  - vector build time per tick
  - bucket lookup time
  - brute-force search over 100, 1000, 10000, and 100000 candidates
  - end-to-end worker latency with channels and deadline handling
  - p50/p95/p99 latency, not just average latency

Initial conclusion:

- Yes, this can likely be made real-time if the runtime path is local,
  precomputed, bucketed by ball location/phase, and run on a deadline-bound
  worker thread.
- No, it should not depend on generating OpenAI embeddings during live play.
- The ball-location seed is a strong idea. Use it as a coarse tactical/spatial
  index, then run vector similarity only over a narrowed candidate set.
