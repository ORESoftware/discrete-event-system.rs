//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/environment.ts`
//! Rust target: `src/des/general/des_base/environment.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/environment.ts",
    "src/des/general/des_base/environment.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/environment.rs",
        "- Keep file-for-file. PureEnvironment is a behavior trait returning next-state,",
        "- EnvironmentStation becomes a concrete wrapper struct implementing DESStation",
        "- Keep pure environment stepping as trait methods; if a pure transition model",
        "- Convert invalid actions, terminal-step misuse, and validation failures to",
    ],
    &[
        "EnvironmentStation",
        "EnvironmentStationOptions",
        "PureEnvironment",
    ],
);
