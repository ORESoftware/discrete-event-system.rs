//! Re-exports of the shared DES model "base class" trait families.
//!
//! Models compose by embedding a core struct and implementing one of these
//! template-method traits. Not every family connects to every other — ports and
//! token types must match — but all families share [`DESStation`] +
//! [`run_iterative_des`](super::runner::run_iterative_des) as the execution kernel.
//!
//! | Family | Trait | Typical topology |
//! |--------|-------|------------------|
//! | Iterative visual solver | [`IterativeSolver`] | `source → solver → sink` |
//! | Single-state optimizer | [`SingleStateOptimizer`](super::single_state_optimizer::SingleStateOptimizer) | evaluator + walker |
//! | Population optimizer | [`PopulationOptimizer`](super::population_optimizer::PopulationOptimizer) | evaluator + population |
//! | Gradient learning | [`GradientOptimizerHook`](super::learning_optimization::GradientOptimizerHook) | source → batch → optimizer → sink |
//! | Tree search | [`TreeSearchStation`](super::tree_search::TreeSearchStation) | branch-and-bound tree |
//! | Fixed-point iteration | [`FixedPointIterationStation`](super::fixed_point::FixedPointIterationStation) | iterate until convergence |
//! | Function-as-node | [`Transform`](crate::des::shared::transform::Transform) / [`PureTransformEntity`](super::transform_entity::PureTransformEntity) | inline or queued transform |
//! | Entity network | [`Entity`](crate::des::r#abstract::abstract::Entity) | queueing / spatial graph |

pub use super::visual_solver::{
    run_visual_solver, IterativeSolver, SolverStation, VisualSolverRun, CH_RESULT, CH_START,
};
