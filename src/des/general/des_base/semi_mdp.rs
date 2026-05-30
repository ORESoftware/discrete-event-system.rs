//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/semi-mdp.ts`
//! Rust target: `src/des/general/des_base/semi_mdp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/semi-mdp.ts",
    "src/des/general/des_base/semi_mdp.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/semi_mdp.rs",
        "- Keep file-for-file. Option and SemiMDPOptions become data/config structs or",
        "- SemiMDPAgentStation becomes a trait plus shared RL-agent state struct for",
        "- Pure option policies/termination predicates can stay trait methods; if",
        "- Convert missing legal-option and invalid option-duration throws to Result.",
    ],
    &["Option", "SemiMDPAgentStation", "SemiMDPOptions"],
);
