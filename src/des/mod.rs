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

// Standalone infrastructure clusters.
pub mod mdp;
pub mod observability;
pub mod reference;

// Signal-flow entities + observers + visual (build on the entity framework).
pub mod signals;
pub mod observers;
pub mod visual;
