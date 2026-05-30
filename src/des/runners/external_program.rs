//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/runners/external-program.ts`
//! Rust target: `src/des/runners/external_program.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/runners/external-program.ts",
    "src/des/runners/external_program.rs",
    &["RUST MIGRATION:", "- Target: src/des/runners/external_program.rs.", "- Keep this as the process-adapter module; interfaces become serde-friendly structs/enums plus an ExternalProgramRunner trait.", "- Replace spawnSync with std::process::Command or tokio::process::Command and return Result<ExternalProgramResult, ExternalProgramError>.", "- Use PathBuf canonicalization for repo-root/script guards and keep external parameter values explicit instead of structural Record types."],
    &["ExternalInterpreterSpec", "ExternalModuleContext", "ExternalModuleKind", "ExternalModuleParams", "ExternalParamValue", "ExternalProgramModule", "ExternalProgramResult", "getExternalModule", "listExternalModules", "registerExternalModule", "repoRootFromRunner", "resolveExternalScript", "runExternalModule", "runExternalProgram", "runPythonReference"],
);
