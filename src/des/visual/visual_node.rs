//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/visual/visual-node.ts`
//! Rust target: `src/des/visual/visual_node.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/visual/visual-node.ts",
    "src/des/visual/visual_node.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/visual/visual_node.rs",
        "- VisualNodeObserver, VisualConnection, VisualNodeEvents, VisualNode, and the",
        "- IsObservable should be a nominal Observable trait with typed event payloads;",
        "- Replace anonymous inner EntityObserver classes, `any` subscribers, and icon",
    ],
    &[
        "ManyInManyOut",
        "OneInManyOut",
        "OneInOneOut",
        "VisualConnection",
        "VisualNode",
        "VisualNodeEvents",
        "VisualNodeObserver",
        "ZeroInManyOut",
        "ZeroOutManyIn",
    ],
);
