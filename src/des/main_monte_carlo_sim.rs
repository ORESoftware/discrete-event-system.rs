//! Port of `src/des/main-monte-carlo-sim.ts`.
//!
//! 1:1 file move. The TypeScript source is an EMPTY PLACEHOLDER (shebang +
//! `'use strict'` only). As a library crate this is a no-op `run()` entry point
//! rather than a `fn main`.
//!
//! PORT NOTE: any future Monte-Carlo sampling must inject a
//! `crate::des::shared::capabilities::RandomSource` (e.g. `SeededRandom`)
//! rather than calling an ambient RNG.

/// Entry point. Empty placeholder, matching the empty TypeScript source.
pub fn run() {}
