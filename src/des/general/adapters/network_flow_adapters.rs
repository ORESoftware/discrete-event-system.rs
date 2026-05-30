//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/network-flow-adapters.ts`
//! Rust target: `src/des/general/adapters/network_flow_adapters.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/network-flow-adapters.ts",
    "src/des/general/adapters/network_flow_adapters.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/network_flow_adapters.rs`.", "RUST MIGRATION: Convert the grouped max-flow, stochastic-flow MDP, and traffic adapters into structs/functions around `DESModelSpec`.", "RUST MIGRATION: Map flow graphs, MDP transition payloads, traffic params, and results to `serde` config/result structs; output paths become `PathBuf`.", "RUST MIGRATION: Return `Result<_, ValidationError>` for malformed capacities, probabilities, and network topology."],
    &[],
);
