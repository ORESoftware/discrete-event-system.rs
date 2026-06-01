---
name: des-general
description: >-
  Build and modify optimization, learning, numerical, and control models in
  `src/des/general` as discrete-event station graphs. Use when implementing an
  iterative algorithm (gradient/evolutionary/MCMC/DP/graph/inference/etc.) with
  the `DESStation` + `run_time_step` pattern, wiring `Transform`/`PureTransform`
  functions, or hooking a model into `VisualBlock` visual specs.
---

# general

> Skill for AI agents working in `src/des/general` — the "iterative algorithm as
> a discrete-event system" framework plus its model collections.

## Overview

`src/des/general` turns numerical/optimization algorithms into explicit DES
station graphs: state moves as typed tokens between stations over named
channels, and each station advances one algorithm iteration per tick from its
`DESStation::run_time_step` hook. The driver `run_iterative_des` ticks every
station until the system goes quiescent, a stop predicate fires, or a tick cap
is hit.

Two layers:

- **`des_base/`** — foundations: `station.rs` (`DESStation`/`StationCore`),
  `runner.rs` (`run_iterative_des`), `visual_block.rs` (`VisualBlock`, a visual
  wrapper over `CompositeDESStation` that renders SVG specs), `transform_entity.rs`
  (`PureTransformEntity`/`MemoryTransformEntity` — functions as graph nodes),
  `visual_solver.rs` (**shared `IterativeSolver` trait** + `source → solver → sink`
  scaffold), and the template-method optimizer bases `single_state_optimizer.rs`,
  `population_optimizer.rs`, `learning_optimization.rs`.
- **Model collections** (files in this folder) — each exposes `run_*` entry
  points returning a result with a trace + station topology (and often
  `visual_blocks`): `classical_optimization_models.rs`,
  `nonlinear_optimization_models.rs`, `advanced_optimization_models.rs`,
  `learning_optimization_models.rs`, `math_blocks.rs` (calculus/control block
  diagrams as `VisualBlock`s), and `numerical_solver_models.rs`.

`numerical_solver_models.rs` implements eight concrete solvers on the shared
[`IterativeSolver`](des_base/visual_solver.rs) base in `des_base/visual_solver.rs`:
a `VisualBlock`-composing `source → solver → sink` pipeline, one model per family —
`run_lbfgs` (L-BFGS), `run_sequence_alignment` (Needleman–Wunsch DP),
`run_metropolis_hastings` (MCMC), `run_differential_evolution`, `run_prim_mst`
(graph MST), `run_backprop_mlp`, `run_gaussian_mixture_em` (EM), and
`run_mean_field_vi` (variational inference). Animations and index cards are
registered in `src/des/html_index.rs`; run `cargo run --bin main_build_site`.

## When to use

- Adding or changing a model/simulation that is an iterative algorithm
  (optimization, learning, sampling, dynamic programming, graph search,
  probabilistic inference, control/ODE block diagrams).
- Wiring vanilla math as a pure `Transform`/`PureTransform`, or stateful logic
  (accumulators, queues, solver memory) as `StatefulTransform`/
  `MemoryTransformEntity`.
- Representing a model's stations as `VisualBlock`s for the visual editor.

## Key files & entry points

- `des_base/station.rs` — `DESStation` trait, `StationCore` (inboxes, `pipe`,
  `emit`, `drain::<T>`), `StationRef`.
- `des_base/runner.rs` — `run_iterative_des`, `IterativeRunOptions`, `RunReason`.
- `des_base/visual_block.rs` — `VisualBlock`, `visual_block_specs`, `VisualBlockSpec`.
- `des_base/transform_entity.rs`, `../shared/transform.rs` — function-as-node.
- `des_base/visual_solver.rs` — `IterativeSolver`, `SolverStation`, `run_visual_solver`,
  `VisualSolverRun` (shared base for one-iter-per-tick algorithms).
- `des_base/single_state_optimizer.rs`, `des_base/population_optimizer.rs`,
  `des_base/learning_optimization.rs` — reusable solver bases.
- `numerical_solver_models.rs`, `*_optimization_models.rs`, `math_blocks.rs` —
  the `run_*` model collections.
- `prng.rs` / `../shared/capabilities.rs` — injected `RandomSource` (mulberry32);
  never reach for a global RNG.

## How to run & test

```sh
# Test one model collection (fast — lib only)
cargo test --lib numerical_solver_models
# Test the whole general subsystem
cargo test --lib des::general
# Build check
cargo build --lib
```

## Conventions & gotchas

- One algorithm iteration per `run_time_step`; gate progress with `has_work`.
- Pure math → `Transform`/`PureTransform`; cross-iteration state lives inside the
  station (the `StatefulTransform` analogue).
- Compose a `VisualBlock` to make a station a visual node; collect specs with
  `visual_block_specs(&[..])` at the end of a run (see `math_blocks.rs` and
  `numerical_solver_models.rs`).
- Validate inputs with `des_base::preconditions::Preconditions` and `panic!` on
  invariant violations; recoverable validation goes through the runner's
  validator path.
- Inject RNG (`mulberry32(seed)`) for determinism; default runner order is
  shuffled, so pass `shuffle: false` for reproducible single-pipeline runs.

## Related skills

- `des_base/.skills/SKILL.md` — the station/runner/visual-block foundations.
- `control_systems/.skills/SKILL.md`, `adapters/.skills/SKILL.md` — sibling
  general submodules.
