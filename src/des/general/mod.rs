//! `des::general` — port of `src/des/general/` (algorithms, models, solvers).
//!
//! Ported in dependency order; foundation-only modules come first.

pub mod des_base;
pub mod prng;

// Grab-bag utilities (DESSet/DESMap, bgn, histograms, shuffle, uuid) + the
// entity registry — foundations of the entity framework.
pub mod general;
pub mod entity_registration;

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

// Model/algorithm consumers (built on des-base + general kernels).
pub mod blackjack;
pub mod four_rooms;
pub mod mountain_car;
pub mod qlearning_des;
pub mod ppo_des;
pub mod rl_learning_models;
pub mod actor_critic_gridworld;
pub mod stag_hunt;
pub mod tiger_pomdp;
pub mod grid_localization_pomdp;
pub mod inventory_dp;
pub mod iterative_learning_control;
pub mod temp_control;
pub mod network_mutex;
pub mod nonlinear_optimization_models;
pub mod classical_optimization_models;
pub mod learning_optimization_models;
pub mod simulated_annealing;
pub mod milp_bnb;
pub mod neural_network;

// Control-theory models (self-contained numeric cluster on shared/linalg).
pub mod control_systems;

// Standalone optimisation/forecasting/LP models built on des-base + general kernels.
pub mod internal_solver_network;
pub mod signal_transforms;
pub mod stochastic_flow_mdp;
pub mod multistage_stochastic;
pub mod stochastic_lp;
pub mod ip_mip_des;
pub mod feasibility_pipeline;
pub mod nonlinear_forecasting_model;
pub mod soccer_rotation;
pub mod do_audit;
pub mod equation_to_stations;

// Domain/simulation models + dispatch (port wave G).
pub mod advanced_optimization_models;
pub mod collaborative_inference;
pub mod computer_network;
pub mod dispatch;
pub mod factory_floor_track3t;
pub mod math_equation_input;
pub mod smart_traffic_flow;
pub mod traffic_flow;

// Universal-model spec, the model registry, and domain application models.
// (`math_blocks` waits on des_base::visual_block, which needs the unported
// `general::ln`, so it stays unwired for now.)
pub mod des_registry;
pub mod domain_application_models;
pub mod universal_model_spec;

// Adapters: wrap solver/model modules as DES stations / universal-model specs.
pub mod adapters;
