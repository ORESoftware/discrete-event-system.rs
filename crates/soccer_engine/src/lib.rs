//! # soccer_engine — agnostic 2D soccer simulation + RL game engine
//!
//! The soccer domain code extracted from `des_engine`. It depends on
//! [`des_engine`] for the generic optimization/learning primitives (LP/IP-MIP,
//! the neural-network MLP + policy-gradient, MDP/POMDP, PRNG, the animation
//! framework) but owns all soccer-specific simulation, rules, agents, planner,
//! rotation, and reinforcement learning (tabular Q-learning + neural value head,
//! actor-critic, PFSP league, world model).
//!
//! ## Transport-agnostic by default
//! This crate ships **no HTTP server** in its default build, so it can be
//! embedded directly in a desktop game. Web concerns are opt-in:
//! - `web-bridge` — the typed request→reply bridge + HTML generators the axum
//!   servers wrap (no sockets).
//! - `embedded-http-server` — the legacy standalone `TcpListener` dev server.
//!
//! During the extraction this is a workspace member of `discrete-event-system.rs`;
//! the soccer modules are relocated here phase by phase. This skeleton compiles
//! green so the workspace stays buildable from the first step.

// Re-export the engine so downstream crates can reach generic primitives through
// a single dependency if they wish.
pub use des_engine;
