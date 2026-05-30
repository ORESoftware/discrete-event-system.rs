//! Rust migration target for the TypeScript discrete event system.
//!
//! The first migration pass preserves the TypeScript module topology under
//! `src/des` while core DES abstractions are lifted into reusable Rust traits
//! and state structs. Files generated from the TypeScript headers retain their
//! original source path and target mapping so the port can proceed file by file.

pub mod core;
pub mod des;
pub mod migration;
pub mod numeric;

pub use crate::core::*;
