//! `des::general::des_base` — port of `src/des/general/des-base/`.
//!
//! Base classes / shared protocol for the discrete-event station framework.
//! Ported incrementally; foundation guards (`preconditions`) come first.

pub mod preconditions;

// Foundation of the station framework (tier 0 + station core).
pub mod argmax;
pub mod episode_accounting;
pub mod model_topology;
pub mod validation;
pub mod station;

// Tier 2: directly on the station core.
pub mod rl_tokens;
pub mod stateful_token;
pub mod cut_pool;
pub mod neural_network;
pub mod tree_search;
pub mod single_state_optimizer;
pub mod population_optimizer;
pub mod fixed_point;
pub mod controller;
pub mod belief_state;
pub mod runner;
pub mod composite_station;
pub mod transform_entity;

// Tier 3: depend on tier-2 bases.
pub mod rl_agent;
pub mod environment;
pub mod policy_gradient_agent;
pub mod finite_horizon_dp;
pub mod advanced_optimization;
pub mod lqr_controller;
pub mod adversarial_control;
pub mod learning_optimization;
pub mod smart_movable;

// Tier 4: depend on rl_agent.
pub mod actor_critic;
pub mod linear_vfa;
pub mod monte_carlo_rl;
pub mod semi_mdp;
pub mod multi_agent;
