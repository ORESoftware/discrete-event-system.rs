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

pub mod cell;
pub mod citizen;
pub mod demos;
pub mod graph;
pub mod run;

pub use cell::{
    Affine, Composite, Gain, Integrator, Map, Queue, RuntimeCell, RuntimeOp, Saturation, Scalar,
    Source, SourceKind, Sum, TransportDelay,
};
pub use citizen::{StudioCitizen, STUDIO_SCHEMA};
pub use demos::{mixer, queue_line, signal_chain, StudioDemo};
pub use graph::{CompiledStudio, NodeRole, StudioError, StudioGraph, VisualNode, Wire};
pub use run::{run, StudioRun};
