//! Hybrid block-diagram engine — the unified Simulink-style spine.
//!
//! Where the rest of the crate offers many *separate* modeling stacks (the DES
//! entity network, control-block frameworks, math-block diagrams, FEL events,
//! and standalone control scripts), this module provides a single connectable
//! [`block::Block`] type and one executive that mixes paradigms in the same
//! run:
//!
//! * **Continuous** dynamics integrated with RK4 ([`blocks::Integrator`],
//!   [`blocks::StateSpace`], [`blocks::BouncingBall`]).
//! * **Discrete** blocks at their own [`block::SampleTime`], with zero-order
//!   hold and true multirate scheduling ([`blocks::DiscretePi`],
//!   [`blocks::Counter`]).
//! * **Events**: zero-crossing detection with bisection and state-reset
//!   handlers (the bouncing ball reflecting at the floor).
//!
//! Build a [`diagram::Diagram`], wire typed ports (widths are checked and
//! direct-feedthrough cycles are rejected as algebraic loops), `build()` it, and
//! [`executive::simulate`] it into a [`executive::Trace`] that exports CSV/JSONL.
//! The JSONL frames are exactly what the [`crate::des::plugin`] sim-player
//! renders, so a hybrid run visualizes with no extra glue.
//!
//! This module is purely additive — it does not modify or depend on the
//! existing engine; it only reuses the shared `serde_json` value type for trace
//! export.
//!
//! ## Example
//!
//! ```
//! use des_engine::des::hybrid::{demos, executive::simulate};
//!
//! let (compiled, opts) = demos::bouncing_ball().unwrap();
//! let trace = simulate(&compiled, &opts);
//! assert!(trace.events >= 1); // the ball bounced
//! ```

pub mod block;
pub mod blocks;
pub mod demos;
pub mod diagram;
pub mod executive;

pub use block::{Block, PortSpec, SampleTime, Signal};
pub use diagram::{BlockHandle, Compiled, Diagram, HybridError, Wire};
pub use executive::{simulate, SimOptions, Trace};
