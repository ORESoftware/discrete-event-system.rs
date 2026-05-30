//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/model-topology.ts`
//! Rust target: `src/des/general/des_base/model_topology.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/model-topology.ts",
    "src/des/general/des_base/model_topology.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/model_topology.rs",
        "- Keep file-for-file. StationGraphTopology becomes a data struct with station",
        "- stationGraphTopology can remain a pure module function over ids; if it is",
        "- Prefer explicit string newtypes/enums for ids if this grows beyond",
        "- Return Result only if future validation is added; today this stays",
    ],
    &["StationGraphTopology", "stationGraphTopology"],
);
