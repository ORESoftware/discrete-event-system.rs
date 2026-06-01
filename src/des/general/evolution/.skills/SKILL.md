---
name: des-general-evolution
description: >-
  Build and modify evolutionary optimization code in `src/des/general/evolution`,
  including generic GA traits, DES-station GA execution, genetic programming,
  curve fitting, GPU-shaped batch evaluation, and toy bio-design problems.
---

# evolution

> Skill for AI agents working in `src/des/general/evolution`.

## Overview

This module collects evolutionary search tools for the DES engine:

- `ga_core.rs` defines shared GA traits (`PopulationInitializer`,
  `FitnessEvaluator`, `GeneticOperators`, `GaProblem`), standalone GA flavors,
  and `EvolutionGaStation` for one-generation-per-DES-tick execution.
- `genetic_programming.rs` evolves `Expr` trees for symbolic regression.
- `curve_fitting.rs` uses GA/GP to search model structure, then solves fixed
  linear coefficients with ridge-normal equations.
- `gpu_batch.rs` provides CPU batch residual evaluation shaped for future GPU
  backends through `evolution-gpu`.
- `bio_design.rs` contains demonstration search spaces for HP lattice folding
  and ligand scaffold design.

## When to use

- Adding a new evolutionary algorithm, GA flavor, chromosome type, fitness
  evaluator, or genetic operator.
- Wiring a population-based optimizer into DES ticks for animation or station
  graph reporting.
- Improving curve fitting, symbolic regression, or batch fitness evaluation.
- Extending the toy biology/design examples without introducing heavyweight
  chemistry or molecular-dynamics dependencies.

## Key files & entry points

- `mod.rs` — public exports for the evolution package.
- `ga_core.rs` — shared traits, standalone `run_ga`, `EvolutionGaStation`, and
  `run_ga_as_des`.
- `curve_fitting.rs` — `ParametricCurveProblem`, `run_curve_fit_ga`,
  `run_curve_fit_gp`, `run_piecewise_ga`, and synthetic datasets.
- `genetic_programming.rs` — tree generation, subtree crossover, mutation, and
  `run_gp`.
- `gpu_batch.rs` — `CpuBatchEvaluator`, `GpuBatchBackend`, and backend-dispatch
  residual helpers.
- `bio_design.rs` — `run_hp_protein_ga` and `run_ligand_design_ga`.
- `src/des/main_evolution_lab.rs` / `src/bin/main_evolution_lab.rs` — CLI demo.

## How to run & test

```sh
cargo test --lib evolution
cargo run --bin main_evolution_lab
cargo check --lib
```

## Conventions & gotchas

- Fitness is minimized.
- Keep stochastic behavior deterministic through `RandomSource` / `mulberry32`;
  do not reach for a global RNG.
- Prefer the shared GA traits before adding problem-specific loops.
- Override `FitnessEvaluator::evaluate_population` when population scoring can
  share matrix work or use a GPU backend.
- `EvolutionGaStation` is intentionally generational; standalone `run_ga`
  carries the broader flavor set (`SteadyState`, `MuPlusLambda`, `Island`).
- Bio-design models here are demonstration landscapes, not validated
  physicochemical models.

## Related skills

- `../.skills/SKILL.md` — general iterative algorithm and station graph patterns.
- `../des_base/.skills/SKILL.md` — station, runner, and population optimizer base.
- `../../test/.skills/SKILL.md` — test module conventions.
