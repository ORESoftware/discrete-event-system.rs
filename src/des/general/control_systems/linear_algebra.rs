//! Port of `src/des/general/control-systems/linear-algebra.ts`.
//!
//! Compatibility re-export shim. The dense linear-algebra toolkit (LinAlg,
//! VecOps, MatrixInverse, LinearSystem, MatrixRank, SymmetricEigen) now lives in
//! `crate::des::shared::linalg`, a dependency-free leaf module. This module
//! declares no items of its own; it simply re-exports that toolkit so legacy
//! `control-systems/linear-algebra` imports keep resolving. The TypeScript
//! `Mat` / `Vec` aliases are exposed by `shared::linalg` as `Matrix` / `Vector`
//! (to avoid clashing with the Rust standard `Vec`).
//!
//! New code should depend on `crate::des::shared::linalg` directly; this shim
//! can be deleted once all call sites are repointed.

pub use crate::des::shared::linalg::*;
