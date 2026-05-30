//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/tree-search.ts`
//! Rust target: `src/des/general/des_base/tree_search.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/tree-search.ts",
    "src/des/general/des_base/tree_search.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/tree_search.rs",
        "- Keep file-for-file. SearchObjective becomes an enum and NodeEvaluation",
        "- TreeSearchStation becomes a trait plus shared station-state struct for the",
        "- pickNext/evaluate/expand and optional pruning hooks map to trait methods;",
        "- Convert invalid objective, empty-frontier, and evaluation failures to Result.",
        "- SELECT ordering   (DFS, BFS, best-first by bound, UCT)",
        "- EVAL semantics    (LP solve, simulation rollout, heuristic h(n))",
        "- PRUNE rule        (bound ≤ incumbent for MILP; nothing for plain DFS)",
        "- EXPAND rule       (split on fractional var; one untried child for MCTS)",
    ],
    &["NodeEvaluation", "SearchObjective", "TreeSearchStation"],
);
