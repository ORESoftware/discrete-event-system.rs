# Soccer Learning Postgres Strategy

The durable learning authority should be Postgres, not filesystem JSON. JSON is
still useful as JSONB for high-dimensional state/config payloads and as an
optional export format, but queue state, simulation results, deltas, and merged
policy versions should be rows. Migrations remain declarative: commit the
pg-defs contract and generated adapters, then generate any RDS diff SQL on
demand for review. Do not check generated migration `.sql` files into version
control.

## Storage Model

- `des_soccer_learning_experiments`: one row per training program.
- `des_soccer_learning_policy_versions`: immutable policy snapshots. A version
  records lineage, source kind (`seed`, `merge`, `mutation`, `crossover`,
  `import`, `replay`), generation, status, and aggregate fitness.
- `des_soccer_learning_policy_entries`: one row per team/state/action or
  team/state/action/target value. `state_key` is JSONB, `state_hash` is the
  lookup key, and values/weights use fixed-point micros.
- `des_soccer_learning_jobs`: leaseable queue rows for local or distributed
  runners. Claim with `FOR UPDATE SKIP LOCKED`, set `lease_owner` and
  `lease_expires_at`, then complete/fail the row.
- `des_soccer_learning_runs`: one row per finished simulation, including score,
  outcome, team-specific merge weights, elapsed time, and summary/stats JSONB.
- `des_soccer_learning_run_deltas`: immutable learned deltas from a run, keyed
  like policy entries and weighted by that team’s match outcome.
- `des_soccer_learning_merge_events`: records which strategy produced a new
  policy version from prior deltas or elite parents.
- `des_soccer_learning_set_play_runs`: typed metrics for restart-learning runs,
  including primary restart, spot, duration, goals, and windowed goal rates.
- `des_soccer_learning_set_play_restart_mix`: one row per restart type trained
  in a run, preserving indirect/direct ordering without packing it into JSONB.
- `des_soccer_learning_set_play_episode_metrics`: one row per repeated restart
  episode with restart, routine, scoring, policy-entry, and neural-step facts.
- `des_soccer_learning_neural_run_metrics`: one row per learning run with
  bounded neural gradient stats, replay stats, parameter count, and losses.

JSONB on the main run and policy-version rows remains a compatibility snapshot,
not the primary analytical surface for restart or neural learning metrics.

## Merge Method

Each simulation starts from a policy version and emits deltas by comparing the
post-game policy to the pre-game policy. A delta is stored only when visits
increase.

For each team:

```text
goal_diff = goals_for - goals_against
win weight  = 1.0 + 0.22 * min(goal_diff, 6)
draw weight = 0.55
loss weight = 0.20 / (1.0 + 0.35 * abs(goal_diff))
```

Small offensive and defensive terms adjust that base, then the result is
clamped. This means a 3-4 loss contributes far less than a 3-1 win even though
both runs contain goals.

Policy merge is an outcome-weighted visit average:

```text
effective_visits = visit_delta * team_merge_weight
merged_q = sum(after_q * effective_visits) / sum(effective_visits)
```

Existing policy entries can be seeded into the merge with a prior/decay weight,
so old knowledge fades only when enough better evidence arrives.

## Queue Runner

The Rust queue runner keeps `N` worker slots full, with `N` clamped to `1..=100`.
When a game finishes:

1. score the match from both teams’ perspectives,
2. extract policy deltas,
3. merge deltas into the current policy,
4. prune if configured,
5. start the next game from the newest policy.

This differs from batch execution: the runner does not wait for the slowest game
in a batch before starting replacements.

## Evolutionary / Genetic Spawning

Policy versions form a lineage graph. New jobs can spawn from:

- `latest`: current active policy,
- `elite`: highest-fitness recent policies,
- `mutation`: one elite policy plus small value perturbations,
- `crossover`: weighted average of multiple elite parents,
- `random`: blank/exploratory seed,
- `replay`: moment-window or tracking-data replay.

The Rust module includes deterministic crossover/mutation over policy entries.
Persist those generated children as `des_soccer_learning_policy_versions` with
`source_kind = 'mutation'` or `source_kind = 'crossover'`.

## Reward Timing

The simulator already rewards and penalizes the action chain around goals,
including attacking credit and defensive blame over recent actions. The
cross-run layer should not duplicate that low-level reward. Its job is to
decide how much to trust a run overall after the final result is known.
