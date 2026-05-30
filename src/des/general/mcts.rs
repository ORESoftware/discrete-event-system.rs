//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/mcts.ts`
//! Rust target: `src/des/general/mcts.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/mcts.ts",
    "src/des/general/mcts.rs",
    &["RUST MIGRATION: target module src/des/general/mcts.rs.", "RUST MIGRATION: MCTSEnv<S> is behavior and should become a trait with associated State/Action types or explicit generic bounds.", "RUST MIGRATION: MCTSOptions and private Node<S> become structs; MCTSStation<S> becomes a struct implementing TreeSearchStation<Node<S>> behavior through traits.", "RUST MIGRATION: mcts<S> is a generic solver helper and can remain a free function; wrap it in PureTransform only for registered DES graph use.", "RUST MIGRATION: Tree nodes need explicit ownership indexes or Rc/RefCell alternatives; prefer arena Vec<Node> plus node indexes for Rust friendliness.", "RUST MIGRATION: Rollouts must use injected rand::Rng and invalid terminal/action states should return Result or domain-specific status."],
    &["MCTSEnv", "MCTSOptions", "MCTSStation", "mcts"],
);
