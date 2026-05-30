//! `des::general::des_base` — port of `src/des/general/des-base/`.
//!
//! Base classes / shared protocol for the discrete-event station framework.
//! Ported incrementally; foundation guards (`preconditions`) come first.

pub mod preconditions;

// Foundation of the station framework (tier 0 + station core).
pub mod argmax;
pub mod episode_accounting;
pub mod model_topology;
pub mod validation;
pub mod station;
