//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/stochastic-flow-mdp-adapter.ts`
//! Rust target: `src/des/general/adapters/stochastic_flow_mdp_adapter.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/stochastic-flow-mdp-adapter.ts",
    "src/des/general/adapters/stochastic_flow_mdp_adapter.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/stochastic_flow_mdp_adapter.rs`.", "RUST MIGRATION: Convert stochastic-flow MDP adapter registration into Rust adapter structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map graph edges, stochastic transition params, policies, and results to `serde` config/result structs; output paths become `PathBuf`.", "RUST MIGRATION: Return `Result<_, ValidationError>` for invalid capacities, probabilities, states, and action definitions."],
    &[],
);
