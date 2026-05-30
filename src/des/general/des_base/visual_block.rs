//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/des-base/visual-block.ts`
//! Rust target: `src/des/general/des_base/visual_block.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/des-base/visual-block.ts",
    "src/des/general/des_base/visual_block.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/des_base/visual_block.rs",
        "- Keep file-for-file. Visual role/direction unions become enums; port,",
        "- VisualBlock becomes a struct implementing CompositeDESStation/DESStation",
        "- Helper functions such as renderVisualBlocks, visualBlockSpecs, port",
        "- Set usage maps to HashSet/BTreeSet. Convert duplicate port, missing port,",
    ],
    &[
        "VisualBlock",
        "VisualBlockConnectionOptions",
        "VisualBlockConnectionSpec",
        "VisualBlockLayout",
        "VisualBlockMember",
        "VisualBlockOptions",
        "VisualBlockPort",
        "VisualBlockPortSpec",
        "VisualBlockRenderContext",
        "VisualBlockRenderable",
        "VisualBlockRole",
        "VisualBlockSpec",
        "VisualBlockStyle",
        "VisualPortDirection",
        "VisualPortInput",
        "VisualPortOptions",
        "isVisualBlock",
        "renderVisualBlockSpec",
        "renderVisualBlocks",
        "visualBlockSpecs",
    ],
);
