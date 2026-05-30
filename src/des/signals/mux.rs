//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/signals/mux.ts`
//! Rust target: `src/des/signals/mux.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/signals/mux.ts",
    "src/des/signals/mux.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/signals/mux.rs",
        "- MultiplexerTimeStepOpts becomes a struct; Multiplexer<E,V> becomes a signal",
        "- Multiplexing should be modeled as a PureTransform once selection semantics",
        "- Current selection/runTimeStep behavior is intentionally open; port the",
        "- Queue intake mirrors other signal transforms; Rust should use",
    ],
    &["Multiplexer", "MultiplexerTimeStepOpts"],
);
