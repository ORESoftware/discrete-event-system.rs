//! `des::general::control_systems` — control-theory models (port of
//! `src/des/general/control-systems/`): linear algebra, numerical/SDE solvers,
//! observability/controllability, DC motor, empirical control, wind MPPT.

pub mod dc_motor;
pub mod empirical_control;
pub mod information_theory;
pub mod linear_algebra;
pub mod numerical_solvers;
pub mod observability_controllability;
pub mod sde_learning;
pub mod shadow_eval;
pub mod stochastic_sde;
pub mod transform_methods;
// (ordering: shadow_eval is the dual/shadow obs-ctrl evaluator built on the
// dc_motor, empirical_control, and observability_controllability modules.)
pub mod wind_mppt;
