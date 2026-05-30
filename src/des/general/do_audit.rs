//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/do-audit.ts`
//! Rust target: `src/des/general/do_audit.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/do-audit.ts",
    "src/des/general/do_audit.rs",
    &["RUST MIGRATION: target module src/des/general/do_audit.rs.", "RUST MIGRATION: doAudit is a small diagnostic command and can be a free function or src/bin/do_audit.rs wrapper around the registry.", "RUST MIGRATION: Replace thrown makeError paths with Result<(), Error>; avoid any-typed entity values by using the Rust entity trait hierarchy."],
    &["doAudit"],
);
