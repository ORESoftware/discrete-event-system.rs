//! `des::general` — port of `src/des/general/` (algorithms, models, solvers).
//!
//! Ported in dependency order; foundation-only modules come first.

pub mod des_base;
pub mod prng;

// Grab-bag utilities (DESSet/DESMap, bgn, histograms, shuffle, uuid) + the
// entity registry — foundations of the entity framework.
pub mod entity_registration;
#[allow(clippy::module_inception)]
pub mod general;

pub mod expr;
pub mod optim;

pub mod factmachine_math;
pub mod hungarian;
pub mod ode;
pub mod quadrature;
pub mod random_variables;
pub mod value_iteration;

pub mod genetic_tsp;
pub mod incremental_lp;
pub mod lp;
pub mod statistical_optimization;

pub mod advanced_control_models;
pub mod belief;
pub mod cartesian_state_space;
pub mod des_spec;
pub mod double_integrator_lqr;
pub mod feedback_linearization;
pub mod kalman_filter;
pub mod mpc_double_integrator;
pub mod mrac;
pub mod pomdp;
pub mod pontryagin_bang_bang;
pub mod sliding_mode_control;

pub mod des_lp_bridge;
pub mod rl_environments;
pub mod root;
pub mod run;
pub mod shortest_path_des;
pub mod time_stepped_station;

// Intra-general dependency leaves (unblock dispatch, ip-mip, internal-solver, etc.).
pub mod field_station;
pub mod ga_des;
pub mod lp_des;
pub mod max_flow;
pub mod mcts;
pub mod network_flow;
pub mod sa_des;
pub mod time_accrued;

// Model/algorithm consumers (built on des-base + general kernels).
pub mod actor_critic_gridworld;
pub mod blackjack;
pub mod classical_optimization_models;
pub mod four_rooms;
pub mod grid_localization_pomdp;
pub mod inventory_dp;
pub mod iterative_learning_control;
pub mod learning_optimization_models;
pub mod milp_bnb;
pub mod mountain_car;
pub mod network_mutex;
pub mod neural_network;
pub mod nonlinear_optimization_models;
pub mod numerical_solver_models;
pub mod ppo_des;
pub mod qlearning_des;
pub mod rl_learning_models;
pub mod simulated_annealing;
pub mod stag_hunt;
pub mod temp_control;
pub mod tiger_pomdp;

// Control-theory models (self-contained numeric cluster on shared/linalg).
pub mod control_systems;

// Standalone optimisation/forecasting/LP models built on des-base + general kernels.
pub mod do_audit;
pub mod equation_to_stations;
pub mod feasibility_pipeline;
pub mod internal_solver_network;
pub mod ip_mip_des;
pub mod multistage_stochastic;
pub mod nonlinear_forecasting_model;
pub mod signal_transforms;
pub mod soccer_rotation;
pub mod stochastic_flow_mdp;
pub mod stochastic_lp;

// Domain/simulation models + dispatch (port wave G).
pub mod advanced_optimization_models;
pub mod collaborative_inference;
pub mod computer_network;
pub mod dispatch;
pub mod factory_floor_track3t;
pub mod math_blocks;
pub mod math_equation_input;
pub mod smart_traffic_flow;
pub mod traffic_flow;

// Universal-model spec, the model registry, and domain application models.
pub mod des_registry;
pub mod domain_application_models;
pub mod universal_model_spec;

// Adapters: wrap solver/model modules as DES stations / universal-model specs.
pub mod adapters;
