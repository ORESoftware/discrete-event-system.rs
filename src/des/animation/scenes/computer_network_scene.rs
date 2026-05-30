//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/scenes/computer-network-scene.ts`
//! Rust target: `src/des/animation/scenes/computer_network_scene.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/scenes/computer-network-scene.ts",
    "src/des/animation/scenes/computer_network_scene.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/scenes/computer_network_scene.rs", "- buildComputerNetworkAnimation can remain a module helper returning Animation with Vec<Frame> and ChartSpec serde data.", "- Point and network layout records become private structs; Map<string, Point> should become HashMap<String, Point> or BTreeMap for stable order.", "- Protocol/color dictionaries should become enums plus match expressions where the domain model allows it.", "- If packet rendering becomes graph-visible, wrap NetworkTimeSample -> Frame in a PureTransform implementor."],
    &["COMPUTER_NETWORK_STAGE_H", "COMPUTER_NETWORK_STAGE_W", "buildComputerNetworkAnimation"],
);
