//! `des::model` — the paradigm-neutral **first-class-citizen contract**.
//!
//! The platform's promise is "describe a model in English, get a runnable model
//! with a UI". That requires one seam every modeling paradigm plugs into as a
//! *peer* — acausal equations, MDP, POMDP, hybrid block diagrams, DES networks,
//! optimization — with none privileged. This module is that seam:
//!
//! * [`ModelDescriptor`] — self-describing discovery metadata (kind, `$schema`,
//!   methods, an example spec the LLM/UI targets).
//! * [`ModelCitizen`] — validate-and-run a JSON spec into a uniform artifact.
//! * [`RunArtifact`] — the uniform output: an animated frame stream *and* a
//!   results document, rendered through the existing plugin player.
//! * [`CitizenRegistry`] — discovery + run-from-JSON across all citizens.
//!
//! [`with_builtins`] returns a registry pre-loaded with acausal, MDP, POMDP,
//! authoring, hybrid, equation, and studio citizens, demonstrating the contract
//! across paradigms.
//!
//! Purely additive: it composes [`crate::des::plugin`],
//! [`crate::des::acausal`], [`crate::des::decision`], and
//! [`crate::des::hybrid`] without modifying any of them.

pub mod artifact;
pub mod authoring;
pub mod builtins;
pub mod codegen;
pub mod graph;
pub mod registry;

pub use crate::des::equation::{EquationCitizen, EQUATION_SCHEMA};
pub use artifact::RunArtifact;
pub use authoring::{model_authoring_json_schema, ModelAuthoringSpec, MODEL_AUTHORING_SCHEMA};
pub use builtins::{with_builtins, HybridCitizen, HYBRID_DEMO_SCHEMA, HYBRID_SCHEMA};
pub use graph::{
    generate_model_graph_rust_code, model_graph_json_schema, ModelGraphKind, ModelGraphSpec,
    MODEL_GRAPH_SCHEMA,
};
pub use registry::{CitizenError, CitizenRegistry, ModelCitizen, ModelDescriptor};
