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
pub mod http_server;
pub mod mdp;
pub mod observability;
pub mod reference;
pub mod ws_server;

// Signal-flow entities + observers + visual (build on the entity framework).
pub mod observers;
pub mod signals;
pub mod visual;

// Animation framework (frame recording, rendering, scene builders).
pub mod animation;

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

// Service self-description, plugin/extension system, and machine discovery for
// HTTP servers that embed this engine as a library (JSON-first contract).
pub mod service;

// JSONL streaming contract for the iterative solvers (LP, MILP/MIP/IP, MDP,
// POMDP). A stream of problem edits in, a stream of solution frames out, with a
// name-keyed model registry (`build_streaming_model` / `run_named_jsonl`) and
// per-command panic isolation. Additive: wraps the existing solvers without
// modifying them.
pub mod streaming;

// Interactive 11-a-side soccer rotation planner: roster constraints, chemistry
// rules, IP/MIP re-solve, pitch + solver tabbed UI.
pub mod soccer_planner;

// Future-event-list (next-event) discrete-event engine, plus an M/M/1 harness
// that compares it against the existing fixed-time-step entity network.
// Additive: wraps the existing engine without modifying it.
pub mod fel;

// External-program plugin system: run third-party programs (Rust for now) that
// emit JSON / JSONL and render the result as an interactive HTML player (sim or
// results), with plugin-declared UI controls. Bridges into `service` for
// discovery. Additive: spawns external processes, never modifies the engine.
pub mod plugin;

// Hybrid block-diagram engine — the Simulink-style spine: one connectable
// `Block` type (typed ports, continuous + discrete state, zero-crossings) and a
// single executive that integrates continuous dynamics, runs discrete blocks at
// their own multirate sample times, and bisects zero-crossing state events.
// Additive: a new unified modeling layer that does not modify the existing
// engine.
pub mod hybrid;

// First-class-model contract — the paradigm-neutral seam every modeling kind
// plugs into as a peer (descriptor + validate-and-run-from-JSON + uniform
// `RunArtifact`), plus a registry of built-in citizens. Additive: composes
// `plugin`, `decision` and `hybrid` without modifying them.
pub mod model;

// Decision processes (MDP / POMDP) as a first-class model under `model`: a
// canonical serde spec, unified solve/rollout reusing the existing solvers, and
// an animated artifact. Additive.
pub mod decision;

// The two-layer visual-block + runtime core: a flat, non-nesting VisualBlock
// graph (Layer 1) over runtime cells of one-or-more `Transform` elements
// (Layer 2), with a dataflow executive. Additive.
pub mod studio;

// The executive-selection seam: a small `Executive` trait the studio and hybrid
// engines implement, plus capability profiles and a `select` that routes a
// VisualBlock subgraph to the executive it needs. Additive.
pub mod exec;

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
pub mod main_music_production;
pub mod main_network_mutex;
pub mod main_neural_net;
pub mod main_newsvendor;
pub mod main_observability_controllability;
pub mod main_observability_controllability_anim;
pub mod main_optimization_as_des;
pub mod main_plumbing_flow;
pub mod main_shadow_eval;
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
