//! `des::general::des_base` — port of `src/des/general/des-base/`.
//!
//! Base classes / shared protocol for the discrete-event station framework.
//! Ported incrementally; foundation guards (`preconditions`) come first.

pub mod preconditions;

// Foundation of the station framework (tier 0 + station core).
pub mod argmax;
pub mod episode_accounting;
pub mod model_topology;
pub mod station;
pub mod validation;
pub mod visual_block;

// Tier 2: directly on the station core.
pub mod belief_state;
pub mod composite_station;
pub mod control_blocks;
pub mod controller;
pub mod cut_pool;
pub mod fixed_point;
pub mod neural_network;
pub mod population_optimizer;
pub mod rl_tokens;
pub mod runner;
pub mod single_state_optimizer;
pub mod stateful_token;
pub mod transform_entity;
pub mod tree_search;
pub mod visual_solver;

// Tier 3: depend on tier-2 bases.
pub mod advanced_optimization;
pub mod adversarial_control;
pub mod environment;
pub mod finite_horizon_dp;
pub mod learning_optimization;
pub mod lqr_controller;
pub mod model_families;
pub mod policy_gradient_agent;
pub mod rl_agent;
pub mod smart_movable;

// Tier 4: depend on rl_agent.
pub mod actor_critic;
pub mod linear_vfa;
pub mod monte_carlo_rl;
pub mod multi_agent;
pub mod semi_mdp;
