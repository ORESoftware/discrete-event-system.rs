//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/pomdp.ts`
//! Rust target: `src/des/general/pomdp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/pomdp.ts",
    "src/des/general/pomdp.rs",
    &["RUST MIGRATION: Target module `src/des/general/pomdp.rs`.", "RUST MIGRATION: Convert `POMDPSpec`, value-iteration/lookahead options, action values, and exact-result shapes to generic `serde` structs where type bounds permit serialization.", "RUST MIGRATION: Port solver classes (`QMDPSolver`, `BeliefLookaheadSolver`, `MostLikelyStateSolver`) as structs implementing solver traits with explicit generic bounds on state/action/observation keys.", "RUST MIGRATION: Use nominal enums/structs instead of structural tuples for leaves and alpha vectors; prefer `HashMap` indexes when generic states/actions need lookup.", "RUST MIGRATION: Make invalid probability, horizon, and missing-index cases return `Result`; keep pure solver helpers as free functions unless a DES-visible `PureTransform` wrapper is added."],
    &["BeliefActionValue", "BeliefLookaheadLeaf", "BeliefLookaheadOptions", "BeliefLookaheadSolver", "MDPVIOptions", "MDPVIResult", "MostLikelyStateSolver", "POMDPExactResult", "POMDPSpec", "QMDPSolver", "beliefUpdate", "expectedBeliefReward", "mdpValueIteration", "observationDistribution", "pomdpExactFiniteHorizon"],
);
