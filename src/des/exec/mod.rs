//! `des::exec` — the **executive-selection seam**.
//!
//! The platform has several runtime *executives* — engines that advance a model
//! through time and produce frames:
//!
//! * the **studio** dataflow executive ([`crate::des::studio::run`]): acyclic
//!   signal flow over Layer-2 cells;
//! * the **hybrid** signal-flow executive ([`crate::des::hybrid::executive`]):
//!   continuous RK4 + multirate discrete + zero-crossing events with feedback;
//! * the **DES run-loop** (the station/movable token network): discrete-event
//!   agents with feedback.
//!
//! This module is the small seam that lets a single VisualBlock graph target
//! *whichever executive a block subgraph needs*. Each executive advertises an
//! [`ExecCapabilities`] profile and implements the object-safe [`Executive`]
//! trait (advance → uniform [`RunArtifact`]). [`select`] then chooses the
//! simplest executive whose capabilities satisfy a subgraph's
//! [`ExecCapabilities`] *requirements* — so a pure-dataflow subgraph routes to
//! `studio`, while one needing continuous integration or events/feedback routes
//! to `hybrid`, and a token-agent subgraph routes to the DES run-loop.
//!
//! Purely additive: it composes `studio`, `hybrid` and `model` without modifying
//! them.

mod adapters;
pub use adapters::{HybridExecutive, StudioExecutive};

use crate::des::model::RunArtifact;

/// The dynamics an executive supports (advertised) or a workload requires
/// (requested). A profile *satisfies* a requirement when it has every capability
/// the requirement flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecCapabilities {
    /// Continuous-time dynamics (ODE integration).
    pub continuous: bool,
    /// Discrete / stepped updates (multirate, per-tick stateful ops).
    pub discrete: bool,
    /// Event handling (zero-crossings, state resets).
    pub events: bool,
    /// Closed loops carrying state (feedback).
    pub feedback: bool,
    /// Acyclic signal dataflow.
    pub dataflow: bool,
    /// Decision agents / flowing tokens (DES entities).
    pub agents: bool,
}

impl ExecCapabilities {
    /// Number of capabilities present (used to prefer the simplest sufficient
    /// executive).
    pub fn count(&self) -> usize {
        [
            self.continuous,
            self.discrete,
            self.events,
            self.feedback,
            self.dataflow,
            self.agents,
        ]
        .iter()
        .filter(|b| **b)
        .count()
    }

    /// True when `self` (an executive's profile) provides every capability that
    /// `req` (a workload's requirement) demands.
    pub fn satisfies(&self, req: ExecCapabilities) -> bool {
        (!req.continuous || self.continuous)
            && (!req.discrete || self.discrete)
            && (!req.events || self.events)
            && (!req.feedback || self.feedback)
            && (!req.dataflow || self.dataflow)
            && (!req.agents || self.agents)
    }

    /// A pure stateless dataflow requirement.
    pub fn dataflow_only() -> Self {
        ExecCapabilities {
            dataflow: true,
            ..Default::default()
        }
    }
}

/// A runtime executive: advance a prepared model in time, producing a uniform
/// artifact. Constructed from its *native* workload (a studio graph, a hybrid
/// diagram, …) so the trait stays object-safe and paradigm-neutral.
pub trait Executive {
    /// Stable kind id matching the executive's [`ExecProfile::kind`].
    fn kind(&self) -> &'static str;
    /// What this executive can do.
    fn capabilities(&self) -> ExecCapabilities;
    /// Run to completion and render as a uniform artifact.
    fn run(&mut self) -> RunArtifact;
}

/// Self-describing metadata for an executive (for discovery + selection).
#[derive(Clone, Copy, Debug)]
pub struct ExecProfile {
    pub kind: &'static str,
    pub title: &'static str,
    pub caps: ExecCapabilities,
    pub summary: &'static str,
}

const T: bool = true;
const F: bool = false;

/// The executives the platform knows about, with their capability profiles.
///
/// `studio` and `hybrid` have concrete [`Executive`] adapters in this module;
/// `des-run-loop` is advertised for selection — its adapter over the existing
/// station/movable network is the next integration.
static PROFILES: [ExecProfile; 3] = [
    ExecProfile {
        kind: "studio",
        title: "Visual-block dataflow",
        caps: ExecCapabilities {
            continuous: F,
            discrete: T,
            events: F,
            feedback: F,
            dataflow: T,
            agents: F,
        },
        summary: "Acyclic signal dataflow over Layer-2 cells; per-tick stateful ops.",
    },
    ExecProfile {
        kind: "hybrid",
        title: "Hybrid signal-flow",
        caps: ExecCapabilities {
            continuous: T,
            discrete: T,
            events: T,
            feedback: T,
            dataflow: T,
            agents: F,
        },
        summary: "Continuous RK4 + multirate discrete + zero-crossing events, with feedback.",
    },
    ExecProfile {
        kind: "des-run-loop",
        title: "DES station/movable run-loop",
        caps: ExecCapabilities {
            continuous: F,
            discrete: T,
            events: F,
            feedback: T,
            dataflow: F,
            agents: T,
        },
        summary: "Discrete-event token network of stations and movables (adapter pending).",
    },
];

/// All known executive profiles.
pub fn profiles() -> &'static [ExecProfile] {
    &PROFILES
}

/// Choose the simplest executive whose capabilities satisfy `req`, or `None` if
/// nothing can. "Simplest" = fewest extra capabilities, so a pure-dataflow
/// requirement prefers `studio` over the more capable `hybrid`.
pub fn select(req: ExecCapabilities) -> Option<&'static ExecProfile> {
    profiles()
        .iter()
        .filter(|p| p.caps.satisfies(req))
        .min_by_key(|p| p.caps.count())
}

/// The capabilities a compiled studio graph requires: dataflow always, plus
/// discrete stepping if any block carries state. This is how a VisualBlock
/// subgraph asks the seam which executive it needs.
pub fn requirements_for_studio(c: &crate::des::studio::CompiledStudio) -> ExecCapabilities {
    ExecCapabilities {
        dataflow: true,
        discrete: c.has_state(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_routes_by_capability() {
        // Pure dataflow → the simplest capable executive, studio.
        assert_eq!(
            select(ExecCapabilities::dataflow_only()).unwrap().kind,
            "studio"
        );
        // Continuous dynamics → only hybrid offers it.
        assert_eq!(
            select(ExecCapabilities {
                continuous: true,
                ..Default::default()
            })
            .unwrap()
            .kind,
            "hybrid"
        );
        // Events + feedback → hybrid.
        assert_eq!(
            select(ExecCapabilities {
                events: true,
                feedback: true,
                ..Default::default()
            })
            .unwrap()
            .kind,
            "hybrid"
        );
        // Token agents → the DES run-loop.
        assert_eq!(
            select(ExecCapabilities {
                agents: true,
                ..Default::default()
            })
            .unwrap()
            .kind,
            "des-run-loop"
        );
    }

    #[test]
    fn unsatisfiable_requirement_returns_none() {
        // Continuous + agents: no single executive offers both today.
        assert!(select(ExecCapabilities {
            continuous: true,
            agents: true,
            ..Default::default()
        })
        .is_none());
    }

    #[test]
    fn studio_graph_requirements_route_to_studio() {
        // A stateful studio graph (queue + delay) requires dataflow + discrete.
        let demo = crate::des::studio::queue_line().unwrap();
        let req = requirements_for_studio(&demo.compiled);
        assert!(req.dataflow && req.discrete);
        assert_eq!(select(req).unwrap().kind, "studio");
    }
}
