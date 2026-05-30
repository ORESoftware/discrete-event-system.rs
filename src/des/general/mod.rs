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

pub mod belief;
pub mod des_spec;
pub mod cartesian_state_space;
pub mod pomdp;
pub mod kalman_filter;
pub mod double_integrator_lqr;
pub mod sliding_mode_control;
pub mod mrac;
pub mod feedback_linearization;
pub mod mpc_double_integrator;
pub mod pontryagin_bang_bang;
pub mod advanced_control_models;

pub mod rl_environments;
pub mod time_stepped_station;
pub mod shortest_path_des;
pub mod des_lp_bridge;
pub mod root;
pub mod run;

// Intra-general dependency leaves (unblock dispatch, ip-mip, internal-solver, etc.).
pub mod time_accrued;
pub mod mcts;
pub mod field_station;
pub mod max_flow;
pub mod network_flow;
pub mod lp_des;
pub mod ga_des;
pub mod sa_des;
