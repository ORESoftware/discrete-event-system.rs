//! `des::studio` — the two-layer **visual block + runtime** core.
//!
//! This realizes the platform's modeling spine as the two layers the design
//! calls for, with their constraints encoded in the types:
//!
//! * **Layer 1 — visual blocks ([`graph`])**: a *flat* graph of [`graph::VisualNode`]s
//!   wired by typed scalar ports. Blocks **cannot nest** — a node owns a
//!   [`cell::RuntimeCell`] (Layer-2 only), never another node/graph — so visual
//!   blocks stay visually separate. Composition is *inside* a block.
//! * **Layer 2 — runtime cells ([`cell`])**: each block runs a pipeline of **one
//!   or more** [`cell::RuntimeOp`] elements built from the `Transform` /
//!   `StatefulTransform` primitives. This is where the calculations happen.
//!
//! The two nesting rules are deliberate and complementary:
//!
//! * **Visual blocks cannot nest** — they are visual, so they stay visually
//!   separate; a node holds a cell, never another node/graph.
//! * **Runtime elements *can* nest** — a [`cell::Composite`] op wraps a whole
//!   sub-cell of further ops, so the runtime tree inside a single block composes
//!   recursively to any depth without ever nesting a visual block.
//!
//! [`run`] is the dataflow executive (acyclic signal flow), a peer of the DES
//! run-loop and the hybrid signal-flow executive. [`citizen::StudioCitizen`]
//! exposes a studio graph through the [`crate::des::model`] contract so it
//! renders and is discoverable like every other first-class kind.
//!
//! Purely additive: it composes the shared `transform` primitives, `plugin`
//! player and `model` contract without modifying them.

pub mod analysis;
pub mod cell;
pub mod citizen;
pub mod demos;
pub mod design;
pub mod editor;
pub mod graph;
pub mod players;
pub mod run;
pub mod spec;
pub mod sweep;
pub mod ui;
pub mod workbench;

pub use analysis::{
    analyze_model_spec, analyze_model_spec_artifact, StudioAnalysis, StudioComponentAnalysis,
    StudioConnectionAnalysis, StudioN2Cell, StudioValidationAnalysis,
};
pub use cell::{
    Affine, Composite, Gain, Integrator, Map, Probe, Queue, RuntimeCell, RuntimeOp, Saturation,
    Scalar, Source, SourceKind, Sum, TransportDelay,
};
pub use citizen::{StudioCitizen, STUDIO_DEMO_SCHEMA, STUDIO_SCHEMA};
pub use demos::{mixer, queue_line, signal_chain, StudioDemo};
pub use design::{
    run_design_study, StudioDesignDriver, StudioDesignObjective, StudioDesignRun,
    StudioDesignStudy, StudioDesignVariable,
};
pub use editor::{studio_editor_html, write_studio_editor_html, STUDIO_EDITOR_REL_PATH};
pub use graph::{CompiledStudio, NodeRole, StudioError, StudioGraph, VisualNode, Wire};
pub use players::{
    design_sweep_artifact, first_design_sweep_artifact, n2_analysis_artifact, studio_run_artifact,
    write_studio_player_html, StudioPlayerError,
};
pub use run::{run, StudioRun};
pub use spec::{
    compile_model_spec, demo_from_spec, example_spec, generate_rust_code, starter_model_spec,
    studio_block_io, studio_model_json_schema, studio_palette, PaletteItem, PaletteParam,
    PaletteParamKind, StudioBlockIo, StudioBlockKind, StudioBlockSpec, StudioConstraintSpec,
    StudioDesignVariableSpec, StudioModelSpec, StudioObjectiveSense, StudioObjectiveSpec,
    StudioSpecError, StudioWireSpec, MAX_SWEEP_SAMPLES, STUDIO_GRAPH_SCHEMA, STUDIO_SPEC_SCHEMA,
};
pub use sweep::{
    run_design_sweep, run_design_sweep_artifact, run_first_design_sweep,
    run_first_design_sweep_artifact, StudioConstraintValue, StudioObjectiveValue, StudioSweepCase,
    StudioSweepError, StudioSweepResult,
};
pub use ui::{render_starter_workbench_html, render_workbench_html, write_workbench_html};
pub use workbench::{workbench_html, workbench_html_for_spec, write_workbench};
