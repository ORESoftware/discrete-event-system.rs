//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/frame-recorder.ts`
//! Rust target: `src/des/animation/frame_recorder.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/frame-recorder.ts",
    "src/des/animation/frame_recorder.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/frame_recorder.rs", "- Keep FrameRecorderOpts as a config struct and FrameRecorder as a stateful writer around PathBuf + std::fs::File/BufWriter.", "- frame(...) can stay an inherent method accepting an FnOnce builder; if graph-visible later, split a PureTransform adapter out.", "- readAnimation should return Result<Animation, AnimationReadError>; replace throw/any JSON events with serde-tagged structs/enums.", "- HTML output remains delegated to html_player::build_html and filesystem work should use std::fs plus PathBuf."],
    &["FrameRecorder", "FrameRecorderOpts", "readAnimation"],
);
