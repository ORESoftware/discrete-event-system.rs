//! `des::decision` — MDP / POMDP as a **first-class model** under the
//! [`crate::des::model`] contract.
//!
//! MDP/POMDP is one first-class citizen, not a privileged special case: this
//! module is the reference implementation of the cross-paradigm pattern every
//! modeling kind follows —
//!
//! 1. a canonical, `serde`, JSON-first **spec** ([`spec`]) that an LLM/UI emits
//!    and the library validates with repairable error messages;
//! 2. a unified **solve** ([`solve`]) that reuses the crate's existing solvers
//!    (`value_iteration`, the POMDP solvers) verbatim;
//! 3. a unified **rollout** ([`rollout`]) producing one [`rollout::EpisodeTrace`];
//! 4. a **viz** layer ([`viz`]) emitting a uniform [`crate::des::model::RunArtifact`]
//!    (animated state graph / belief bars + a solved results document);
//! 5. a **citizen** ([`citizen`]) implementing [`crate::des::model::ModelCitizen`]
//!    so the platform can discover it and run it from JSON.
//!
//! Purely additive: it composes existing solvers behind a new canonical surface
//! and does not modify them.

pub mod citizen;
pub mod demos;
pub mod rollout;
pub mod solve;
pub mod spec;
pub mod viz;

pub use citizen::{MdpCitizen, PomdpCitizen};
pub use demos::{machine_maintenance_mdp, tiger_pomdp};
pub use rollout::{rollout_mdp, rollout_pomdp, EpisodeTrace, Prng};
pub use solve::{
    solve_mdp, solve_pomdp, solve_pomdp_underlying, MdpMethod, MdpSolution, PomdpMethod, PomdpPlan,
    PomdpSolution,
};
pub use spec::{MdpSpec, MdpTransition, PomdpSpec, TerminalState, MDP_SCHEMA, POMDP_SCHEMA};
pub use viz::{mdp_artifact, pomdp_artifact};
