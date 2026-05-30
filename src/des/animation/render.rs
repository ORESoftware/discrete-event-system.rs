//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/render.ts`
//! Rust target: `src/des/animation/render.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/render.ts",
    "src/des/animation/render.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/render.rs", "- Keep file-for-file as the post-hoc renderer module; a thin src/bin wrapper can call main_result() if this becomes a CLI binary.", "- Convert process.argv/process.exit flow to a Result-returning main_result(args: impl Iterator<Item=String>) and map errors to exit codes at the boundary.", "- Filesystem paths should use PathBuf/std::fs; preserve the .frames.jsonl -> .html suffix rule with typed path helpers.", "- readAnimation/buildHTML become frame_recorder::read_animation and html_player::build_html returning Results."],
    &[],
);
