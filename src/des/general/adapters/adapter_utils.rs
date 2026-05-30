//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/adapters/adapter-utils.ts`
//! Rust target: `src/des/general/adapters/adapter_utils.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/adapters/adapter-utils.ts",
    "src/des/general/adapters/adapter_utils.rs",
    &["RUST MIGRATION: Target module `src/des/general/adapters/adapter_utils.rs`.", "RUST MIGRATION: Convert these shared adapter helpers to free functions around `DESModelSpec`/registration modules; use `serde` structs for any config/result records.", "RUST MIGRATION: Represent CSV/output paths as `PathBuf`, filesystem writes as `std::fs`/`std::io::Result`, and validation failures as `Result<_, ValidationError>`."],
    &["csvCell", "csvRow", "defaultFramesPath", "framesPath", "jsonCsvCell", "jsonCsvRow", "numberPair", "optionalNumberPair", "validationLine", "writeCsvLines"],
);
