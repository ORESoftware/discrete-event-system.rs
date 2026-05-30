//! `des::general::adapters` — adapters wrapping solver/model modules as DES
//! stations / universal-model specs (port of `src/des/general/adapters/`).

pub mod adapter_utils;
pub mod advanced_optimization_control_adapter;
pub mod classical_optimization_adapter;
pub mod collaborative_inference_adapter;
pub mod computer_network_adapter;
pub mod domain_application_adapter;
pub mod feasibility_pipeline_adapter;
pub mod internal_solver_network_adapter;
pub mod learning_optimization_adapter;
pub mod milp_bnb_adapter;
pub mod multistage_sddp_adapter;
pub mod network_flow_adapters;
pub mod neural_network_adapters;
pub mod nonlinear_forecasting_adapter;
pub mod nonlinear_optimization_adapter;
pub mod shortest_path_adapter;
pub mod signal_transforms_adapter;
pub mod simulated_annealing_adapter;
pub mod stochastic_flow_mdp_adapter;
pub mod stochastic_optimization_adapters;
pub mod temp_control_adapter;
