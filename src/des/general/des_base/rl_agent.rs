//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/rl-agent.ts`
//! Rust target: `src/des/general/des_base/rl_agent.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/rl-agent.ts",
    "src/des/general/des_base/rl_agent.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/rl_agent.rs",
        "- Keep file-for-file. RLAgentStation becomes the core RL-agent trait plus a",
        "- pickAction/update/endOfEpisode hooks should become trait methods, with",
        "- State/action/transition tokens come from rl_tokens.rs as structs; channel",
        "- Pure policy/value functions used as graph nodes should implement",
        "- Q-learning:    Q[s,a] ← Q[s,a] + α(r + γ max_a' Q[s',a'] − Q[s,a])",
        "- SARSA:         Q[s,a] ← Q[s,a] + α(r + γ Q[s',a'] − Q[s,a])",
        "- Expected SARSA: Q[s,a] ← Q[s,a] + α(r + γ Σ_a' π(a'|s') Q[s',a'] − Q[s,a])",
        "- REINFORCE:     θ ← θ + α (Σ_t γ^t r_t) ∇log π(a|s)",
    ],
    &["RLAgentStation"],
);
