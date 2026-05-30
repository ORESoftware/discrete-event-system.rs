//! `des::shared` — dependency-free foundation (port of `src/des/shared/`).
//!
//! Translated first because nothing here depends on a feature module. Encodes
//! the engine-wide conventions: the `Transform` trait, `Result`/`Option`
//! helpers, capability ports (RNG/clock), and dense linear algebra.

pub mod result;
pub mod transform;
pub mod capabilities;
pub mod linalg;
pub mod precision;
// Shims for npm packages with no crate equivalent (used by the entity framework).
pub mod linked_queue;
pub mod iterable_int;
