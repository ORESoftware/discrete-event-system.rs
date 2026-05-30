//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/collaborative-inference.ts`
//! Rust target: `src/des/general/collaborative_inference.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/collaborative-inference.ts",
    "src/des/general/collaborative_inference.rs",
    &["RUST MIGRATION: target module src/des/general/collaborative_inference.rs.", "RUST MIGRATION: CollaborativeInferenceScenario becomes an enum; all public params/items/responses/results become serde structs.", "RUST MIGRATION: Internal preset/config/stat interfaces should become private structs, with Record-like respondent/item tables mapped to HashMap<String, _>.", "RUST MIGRATION: Respondent/evidence/ranking tokens and station classes become Token and Station trait impls; runCollaborativeInference should be a PureTransform entry struct.", "RUST MIGRATION: Convert validation throws to Result and keep scoring helpers as private free functions."],
    &["CollaborativeInferenceCoverage", "CollaborativeInferenceItem", "CollaborativeInferenceParams", "CollaborativeInferenceResponse", "CollaborativeInferenceResult", "CollaborativeInferenceScenario", "CollaborativeItemScore", "CredibilityWeightSummary", "runCollaborativeInference"],
);
