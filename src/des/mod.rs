//! `des` — root of the engine, mirroring the TypeScript `src/des/` tree.
//!
//! Modules are added here as they are ported from TypeScript, in dependency
//! order (foundation first).

pub mod shared;
pub mod general;

// Entity framework (queueing-network model). `abstract` is a reserved keyword,
// so the module is reached via the raw identifier `r#abstract`.
pub mod r#abstract;
pub mod entity_moving;
pub mod random_variables;

// Concrete entity stations (built on entity_moving + abstract framework).
pub mod entity_queue;
pub mod entity_source;
pub mod entity_sink;
pub mod entity_travel;
pub mod entity_routing;
pub mod entity_decision;
pub mod entity_processing;
// `entity-conn.ts/` is a directory whose name literally ends in `.ts`; reach
// its single `conn` module via an explicit path so the file mapping is 1:1.
#[path = "entity_conn.ts/conn.rs"]
pub mod entity_conn;

// Standalone infrastructure clusters.
pub mod mdp;
pub mod observability;
pub mod reference;

// Signal-flow entities + observers + visual (build on the entity framework).
pub mod signals;
pub mod observers;
pub mod visual;
