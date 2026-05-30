//! `des::test` — port of the TypeScript `src/des/test/` suite.
//!
//! Each TS `*-test.ts` becomes a Rust module here. The whole bucket is compiled
//! only under `cfg(test)` (wired from `des/mod.rs`), so it adds no weight to a
//! normal build. Modules whose subject is not yet ported are left commented out
//! with a note.

pub mod advanced_optimization_control_test;
pub mod animation_test;
pub mod argmax_tiebreak_test;
pub mod calculus_test;
pub mod classical_optimization_test;
pub mod collaborative_inference_test;
pub mod computer_network_test;
pub mod dc_motor_test;
pub mod dispatch_test;
// pub mod domain_application_test; // waits on general::domain_application_models
pub mod elevator_invariants_test;
pub mod empirical_control_test;
pub mod external_module_test;
// pub mod factmachine_markets_test; // waits on main_factmachine_markets
pub mod factmachine_math_test;
pub mod factory_floor_track3t_test;
pub mod feasibility_pipeline_test;
pub mod float_bias_test;
pub mod genetic_tsp_test;
pub mod incremental_lp_test;
pub mod internal_solver_network_test;
pub mod ip_mip_des_test;
pub mod iterator_test;
pub mod learning_optimization_test;
pub mod lp_test;
// pub mod math_blocks_test; // waits on general::math_blocks
pub mod mdp_adjacent_test;
pub mod milp_bnb_test;
pub mod multistage_stochastic_test;
pub mod network_flow_test;
pub mod network_mutex_test;
pub mod neural_animation_test;
pub mod neural_network_test;
pub mod newsvendor_test;
pub mod nonlinear_forecasting_test;
pub mod nonlinear_optimization_test;
pub mod observability_controllability_test;
pub mod optimal_control_test;
pub mod optimization_as_des_test;
pub mod output_routing_policy_test;
pub mod preconditions_test;
pub mod queue_bias_test;
pub mod random_variables_test;
pub mod shortest_path_test;
pub mod signal_transforms_test;
pub mod simulated_annealing_test;
pub mod soccer_test;
pub mod statistical_optimization_test;
pub mod stochastic_lp_test;
pub mod stochastic_sde_test;
pub mod temp_control_test;
pub mod test;
pub mod transform_entity_test;
pub mod ts_test;
pub mod universal_model_spec_test;
pub mod validation_test;
pub mod visual_block_test;
pub mod wind_mppt_test;
