# Evolution: GA, GP, curve fitting, and bio-design

Additive module on top of `population_optimizer` / `expr` / `linalg`. Generic
GA problems can run either through the standalone flavor driver or through
`EvolutionGaStation`, where each generational update is a DES tick and emits
`GaGenerationInfo` snapshots for animation/reporting.

## Flavors

| Family | Variants | Individual |
|--------|----------|------------|
| GA | `Generational`, `SteadyState`, `MuPlusLambda`, `Island` | Any `Clone` type via [`GaProblem`] |
| GP | `Standard`, `ParsimonyPressure` | [`Expr`] trees (`genetic_programming`) |
| Curve fit | Parametric GA, GP symbolic, piecewise knots, hybrid ridge | [`ParametricChromosome`], [`PiecewiseChromosome`], `Expr` |
| Bio | HP lattice protein, ligand scaffold | [`HpGenome`], [`LigandGenome`] |

## Curve fitting + linear algebra

1. **GA/GP** proposes structure (terms, knots, or expression shape).
2. [`curve_fitting::hybrid_refine`] builds a design matrix and solves ridge-normal
   equations via `shared::linalg::LinearSystem` (coefficients for fixed structure).
3. [`gpu_batch::CpuBatchEvaluator`] evaluates population residuals through shared
   batch hooks. Fixed-design populations use one BLAS-shaped `X * B` multiply;
   variable-design populations use a one-call batched-design hook that GPU
   backends can override.

## GPU

The crate stays dependency-free by default. With `--features evolution-gpu`,
[`GpuBatchBackend`] is available for out-of-tree GPU plugins; the built-in
[`CpuBatchEvaluator`] is always used unless you register a backend.

## Entry points

- Library: `des::general::evolution::*`
- CLI: `cargo run --bin main_evolution_lab`
