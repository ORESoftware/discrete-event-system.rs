//! `des::runners` — port of the TypeScript `src/des/runners/` harnesses.
//!
//! These are the engine's validation / comparison drivers (cross-checking model
//! outputs against external references, comparing FEL vs per-individual engines,
//! sweeping step sizes, etc.) plus their shared infrastructure (`types`,
//! `shared`, `stats`).

pub mod shared;
pub mod stats;
pub mod types;

pub mod compare_elevator_dispatch;
pub mod compare_external_fel_models;
pub mod compare_traffic_engines;
pub mod difference_runner;
pub mod external_modules;
pub mod external_program;
pub mod fel_runner;
pub mod framework_runner;
pub mod gillespie_runner;
pub mod ode_runner;
pub mod per_individual_runner;
pub mod per_individual_vs_fel;
pub mod replicate;
pub mod run_external_module;
pub mod steady_state;
pub mod stepsize_sweep;

pub mod validate_backpropagation;
pub mod validate_calculus;
pub mod validate_computer_network;
pub mod validate_contact_vs_meanfield;
pub mod validate_convolution;
pub mod validate_court_mdp;
pub mod validate_dispatch;
pub mod validate_electric_circuit;
pub mod validate_elevator;
pub mod validate_external_fel_models;
pub mod validate_factmachine;
pub mod validate_factmachine_math;
pub mod validate_genetic_tsp;
pub mod validate_incremental_lp;
pub mod validate_ip_mip_external;
pub mod validate_lp;
pub mod validate_milp_bnb;
pub mod validate_neural_network;
pub mod validate_newsvendor;
pub mod validate_optimization_as_des;
pub mod validate_optimization_suite;
pub mod validate_references;
pub mod validate_shortest_path;
pub mod validate_simulated_annealing;
pub mod validate_smart_traffic_external;
pub mod validate_soccer;
pub mod validate_stochastic_lp;
pub mod validate_temp_control;
pub mod validate_two_disease;
pub mod validate_with_externals;
