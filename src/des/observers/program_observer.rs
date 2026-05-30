//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/observers/program-observer.ts`
//! Rust target: `src/des/observers/program_observer.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/observers/program-observer.ts",
    "src/des/observers/program_observer.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/observers/program_observer.rs",
        "- ProgramObserver becomes a concrete Observer trait implementation that owns a",
        "- `Set<AbstractMovingEntity<any>>` needs an ownership decision: Rc<RefCell<_>>,",
        "- doUpdate should take a typed event enum/payload instead of string +",
    ],
    &["ProgramObserver"],
);
