//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/stateful-token.ts`
//! Rust target: `src/des/general/des_base/stateful_token.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/stateful-token.ts",
    "src/des/general/des_base/stateful_token.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/stateful_token.rs",
        "- Keep file-for-file. TokenStateMode becomes an enum; lineage, transition,",
        "- Factory functions can become constructors or associated functions on token",
        "- Map usage in the registry maps to HashMap/BTreeMap keyed by token id; type",
        "- Pure lineage/transition helpers can remain module functions, or become",
    ],
    &[
        "PayloadStatefulToken",
        "StatefulToken",
        "StatefulTokenRegistry",
        "StatefulTokenRegistryStats",
        "TokenLineage",
        "TokenStateMode",
        "TokenStateTransition",
        "childLineage",
        "isStatefulToken",
        "makeStatefulToken",
        "makeStatelessToken",
        "spawnStatefulChildToken",
        "transitionToken",
    ],
);
