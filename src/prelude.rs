//! The curated SDK surface — `use des_engine::prelude::*;`.
//!
//! Everything in the engine is reachable under [`crate::des`], but that tree is
//! large and deep. This prelude lifts the **first-class seams** a server,
//! desktop app, or CLI actually builds against into one stable, shallow import,
//! so consumers don't have to memorize module paths:
//!
//! * **Run a model from a JSON spec** — [`with_builtins`], [`CitizenRegistry`],
//!   [`ModelCitizen`], [`ModelDescriptor`], [`RunArtifact`].
//! * **Compile equation-based models** — [`compile_acausal_model`],
//!   [`simulate_acausal_model`], [`acausal_workbench_descriptor`].
//! * **Pick an executive for a graph** — [`Executive`], [`ExecCapabilities`],
//!   [`select`], [`requirements_for_studio`], [`StudioExecutive`],
//!   [`HybridExecutive`].
//! * **Build a visual block graph** — [`StudioGraph`], [`RuntimeCell`],
//!   [`RuntimeOp`], [`Composite`].
//! * **Stream a solver (LP/MILP/MDP/POMDP)** — [`StreamingModel`],
//!   [`StreamContract`], [`build_streaming_model`], [`run_named_jsonl`],
//!   [`run_jsonl`].
//! * **Run external plugins / render players** — [`PluginManifest`],
//!   [`run_and_render`], [`PluginTransport`], [`ProcessTransport`],
//!   [`PluginRegistry`].
//! * **Describe a service for HTTP discovery** — [`ServiceBuilder`],
//!   [`ServiceDescriptor`], [`DesExtension`].
//!
//! Deeper or specialized APIs (the entity framework, individual solvers, the
//! `main_*` simulation demos) stay under [`crate::des`] on purpose.

// First-class model contract (English → JSON spec → run → artifact).
pub use crate::des::model::{
    generate_model_graph_rust_code, model_authoring_json_schema, model_graph_json_schema,
    with_builtins, CitizenError, CitizenRegistry, ModelAuthoringSpec, ModelCitizen,
    ModelDescriptor, ModelGraphKind, ModelGraphSpec, RunArtifact, MODEL_GRAPH_SCHEMA,
};

// Acausal/equation-based modeling surface.
pub use crate::des::acausal::{
    acausal_palette, acausal_workbench_descriptor, compile_acausal_model, run_acausal_model,
    simulate_acausal_model, starter_acausal_model_spec, AcausalCitizen, AcausalEquationKind,
    AcausalEquationSpec, AcausalError, AcausalModelSpec, AcausalPaletteItem, AcausalVariableKind,
    AcausalVariableSpec, AcausalWorkbenchDescriptor, CompiledAcausalModel, StructuralDiagnostics,
    ACAUSAL_SCHEMA,
};

// Executive-selection seam.
pub use crate::des::exec::{
    requirements_for_studio, select, ExecCapabilities, Executive, HybridExecutive, StudioExecutive,
};

// Two-layer visual-block + runtime core.
pub use crate::des::hybrid::{
    hybrid_model_json_schema, HybridBlockSpec, HybridModelSpec, HybridWireSpec, HYBRID_GRAPH_SCHEMA,
};
pub use crate::des::studio::{
    studio_model_json_schema, CompiledStudio, Composite, RuntimeCell, RuntimeOp, StudioGraph,
    StudioModelSpec, VisualNode, Wire, STUDIO_GRAPH_SCHEMA,
};

// JSONL streaming solver contract + registry.
pub use crate::des::streaming::{
    build_streaming_model, run_jsonl, run_named_jsonl, streaming_contracts, SolverKind,
    StreamContract, StreamingModel,
};

// External-program plugin system.
pub use crate::des::plugin::{
    run_and_render, run_plugin, run_plugin_with_input, PluginError, PluginManifest, PluginRegistry,
    PluginTransport, ProcessTransport, RunSpec, PLUGIN_PROTOCOL_SCHEMA,
};

// Service self-description for embedding servers.
pub use crate::des::service::{DesExtension, ServiceBuilder, ServiceDescriptor};
