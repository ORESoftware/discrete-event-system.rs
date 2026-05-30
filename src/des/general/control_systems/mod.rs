//! `des::general::control_systems` — control-theory models (port of
//! `src/des/general/control-systems/`): linear algebra, numerical/SDE solvers,
//! observability/controllability, DC motor, empirical control, wind MPPT.

pub mod linear_algebra;
pub mod numerical_solvers;
pub mod observability_controllability;
pub mod stochastic_sde;
pub mod sde_learning;
pub mod dc_motor;
pub mod empirical_control;
pub mod wind_mppt;
