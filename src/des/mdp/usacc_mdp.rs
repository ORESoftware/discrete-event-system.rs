//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/mdp/usacc-mdp.ts`
//! Rust target: `src/des/mdp/usacc_mdp.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/mdp/usacc-mdp.ts",
    "src/des/mdp/usacc_mdp.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/mdp/usacc_mdp.rs",
        "- The const arrays/types should become Rust enums plus `TryFrom<usize>` or",
        "- encode/decode/quality/reward/outcomes/sampleInitialState are pure functions;",
        "- Replace TS union-string action types, Record<Action, number>, non-null",
        "- Inject RNG for sampleInitialState instead of accepting an untyped closure.",
    ],
    &[
        "ACCEPTED",
        "ACTIONS",
        "Action",
        "CLOSED",
        "CONFLICT",
        "CORROBORATION",
        "CaseState",
        "Conflict",
        "Corroboration",
        "EVIDENCE",
        "EXHAUSTED",
        "Evidence",
        "FUNDING",
        "FUND_ACTIVE",
        "FUND_ESCROWED",
        "FUND_EXHAUSTED",
        "FUND_UNFUNDED",
        "Funding",
        "MANIPULATION",
        "Manipulation",
        "N_ACTIONS",
        "N_STATES",
        "Outcome",
        "STAGES",
        "Stage",
        "decode",
        "encode",
        "isTerminal",
        "outcomes",
        "quality",
        "rewardOfAccept",
        "rewardOfClose",
        "sampleInitialState",
        "terminalReward",
    ],
);
