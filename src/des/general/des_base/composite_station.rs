//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/composite-station.ts`
//! Rust target: `src/des/general/des_base/composite_station.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/composite-station.ts",
    "src/des/general/des_base/composite_station.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/composite_station.rs",
        "- Keep file-for-file. CompositeInputPort, CompositeOutputPort, and",
        "- CompositeDESStation becomes a state-owning struct implementing DESStation",
        "- Keep bridge/port helpers private unless needed by mod.rs re-exports; graph",
        "- Convert duplicate-port and invalid-connection failures to Result.",
    ],
    &["CompositeDESStation", "CompositeStationSnapshot"],
);
