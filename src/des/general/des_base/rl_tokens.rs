//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/rl-tokens.ts`
//! Rust target: `src/des/general/des_base/rl_tokens.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/rl-tokens.ts",
    "src/des/general/des_base/rl_tokens.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/rl_tokens.rs",
        "- Keep file-for-file. StateToken, ActionToken, TransitionToken,",
        "- Generic state/action payloads should become type parameters where the owning",
        "- Empty marker token classes map to unit structs implementing the Token marker",
        "- No throws here; constructors should stay infallible unless future validation",
    ],
    &[
        "ActionToken",
        "ResumeToken",
        "StateToken",
        "TrainTriggerToken",
        "TransitionToken",
    ],
);
