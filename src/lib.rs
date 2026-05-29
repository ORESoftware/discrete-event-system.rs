//! des_engine — Rust port of the TypeScript discrete-event-system engine.
//!
//! The TypeScript tree under `src/des/` maps 1:1 onto this crate's `des`
//! module. Foundation modules in `des::shared` are dependency-free and are the
//! reference for the engine-wide conventions (the `Transform` trait, `Result`/
//! `Option` helpers, capability ports, and linear algebra).

pub mod des;
