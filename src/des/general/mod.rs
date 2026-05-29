//! `des::general` — port of `src/des/general/` (algorithms, models, solvers).
//!
//! Ported in dependency order; foundation-only modules come first.

pub mod des_base;
pub mod prng;

pub mod expr;
pub mod optim;

pub mod ode;
pub mod quadrature;
pub mod random_variables;
pub mod hungarian;
pub mod factmachine_math;
pub mod value_iteration;

pub mod lp;
pub mod incremental_lp;
pub mod statistical_optimization;
pub mod genetic_tsp;
