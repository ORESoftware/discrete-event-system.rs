//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/multi-agent.ts`
//! Rust target: `src/des/general/des_base/multi_agent.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/multi-agent.ts",
    "src/des/general/des_base/multi_agent.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/multi_agent.rs",
        "- Keep file-for-file. JointEnvironment becomes a behavior trait and",
        "- JointEnvStation and MultiAgentSystem become state-owning structs; per-agent",
        "- Joint step/policy helpers can stay associated methods; if a coordination",
        "- Convert agent-count and missing-action errors to Result.",
    ],
    &[
        "JointEnvStation",
        "JointEnvironment",
        "MultiAgentSystem",
        "MultiAgentSystemOpts",
    ],
);
