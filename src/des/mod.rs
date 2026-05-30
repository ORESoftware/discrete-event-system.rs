//! `des` — root of the engine, mirroring the TypeScript `src/des/` tree.
//!
//! Modules are added here as they are ported from TypeScript, in dependency
//! order (foundation first).

pub mod general;
pub mod shared;

// Entity framework (queueing-network model). `abstract` is a reserved keyword,
// so the module is reached via the raw identifier `r#abstract`.
pub mod r#abstract;
pub mod entity_moving;
pub mod random_variables;

// Concrete entity stations (built on entity_moving + abstract framework).
pub mod entity_decision;
pub mod entity_processing;
pub mod entity_queue;
pub mod entity_routing;
pub mod entity_sink;
pub mod entity_source;
pub mod entity_travel;
// `entity-conn.ts/` is a directory whose name literally ends in `.ts`; reach
// its single `conn` module via an explicit path so the file mapping is 1:1.
#[path = "entity_conn.ts/conn.rs"]
pub mod entity_conn;

// Standalone infrastructure clusters.
pub mod mdp;
pub mod observability;
pub mod reference;

// Signal-flow entities + observers + visual (build on the entity framework).
pub mod observers;
pub mod signals;
pub mod visual;

// Animation framework (frame recording, rendering, scene builders).
pub mod animation;

// WebSocket server: live-connection registry for broadcasting (e.g. animation
// frames). Faithful port; the network binding is stubbed (see module docs). The
// directory holds a single `ws_server.rs`, reached via an explicit path.
#[path = "ws_server/ws_server.rs"]
pub mod ws_server;

// Cross-subsystem integration smoke tests (compiled only under cfg(test)).
#[cfg(test)]
mod integration_smoke;

// Extensive cross-cutting tests: RNG determinism, statistical properties,
// Decimal edge cases (compiled only under cfg(test)).
#[cfg(test)]
mod extensive_tests;

// Ported TypeScript test suite (`src/des/test/`), compiled only for tests.
#[cfg(test)]
mod test;

// Validation / comparison harnesses (cross-checking models against references).
pub mod runners;

// Serial driver that runs every simulation entry point one after another.
pub mod simulations;

// Top-level simulation entry points (each TS `main-*.ts` becomes a module with
// a `pub fn run()`), plus the cluster/process scaffolding scripts. These mirror
// the executable scripts under the TS `src/des/` root.
pub mod child;
pub mod main;
pub mod main_backpropagation;
pub mod main_build_site;
pub mod main_calculus;
pub mod main_computer_network;
pub mod main_contact_seir;
pub mod main_convolution;
pub mod main_court_mdp;
pub mod main_dc_motor;
pub mod main_dc_motor_anim;
pub mod main_dispatch_combo;
pub mod main_electric_circuit;
pub mod main_elevator;
pub mod main_elevator_highrise;
pub mod main_empirical_control;
pub mod main_empirical_control_report;
pub mod main_epidemic;
pub mod main_epidemic_improved;
pub mod main_factmachine;
pub mod main_factmachine_markets;
pub mod main_factory_floor_track3t;
pub mod main_fibonacci_recursion;
pub mod main_from_json;
pub mod main_genetic_tsp;
pub mod main_hazard_function_survival_analysis;
pub mod main_incremental_lp;
pub mod main_inventory_mdp;
pub mod main_ip_mip_des;
pub mod main_knapsack_problem;
pub mod main_lp_des;
pub mod main_lp_factory;
pub mod main_markov;
pub mod main_mdp_lp;
pub mod main_milp_bnb;
pub mod main_monte_carlo_sim;
pub mod main_network_mutex;
pub mod main_neural_net;
pub mod main_newsvendor;
pub mod main_observability_controllability;
pub mod main_observability_controllability_anim;
pub mod main_optimization_as_des;
pub mod main_plumbing_flow;
pub mod main_shortest_path;
pub mod main_shortest_path_algo;
pub mod main_signal_processing;
pub mod main_simulated_annealing;
pub mod main_snowball;
pub mod main_soccer_rotation;
pub mod main_stochastic_flow_mdp;
pub mod main_stochastic_lp;
pub mod main_stochastic_sde;
pub mod main_stochastic_sde_report;
pub mod main_temp_control;
pub mod main_temp_control_anim;
pub mod main_traffic;
pub mod main_two_disease;
pub mod main_wind_mppt;
pub mod main_wind_mppt_anim;
pub mod max_flow;
pub mod parent;
pub mod program;
