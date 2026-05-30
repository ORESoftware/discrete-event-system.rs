//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/universal-model-spec.ts`
//! Rust target: `src/des/general/universal_model_spec.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/universal-model-spec.ts",
    "src/des/general/universal_model_spec.rs",
    &["RUST MIGRATION: Target module `src/des/general/universal_model_spec.rs`.", "RUST MIGRATION: Convert model/input kind unions to enums and every universal spec/interface to `serde` structs with `#[serde(rename_all = \"...\")]` matching JSON.", "RUST MIGRATION: Replace `Record<string, unknown>` with typed maps (`HashMap<String, serde_json::Value>`) only at true JSON extension boundaries.", "RUST MIGRATION: Keep conversion functions as free functions returning `Result`; `assertUniversalDESModelSpec` becomes a validator that maps checks into errors.", "RUST MIGRATION: Preserve one-to-one field names for portable snapshots, and use typed helper structs for endpoints, edges, variables, conditions, and solver intent."],
    &["UniversalDESModelSpec", "UniversalDESNetworkSpec", "UniversalEndpointSpec", "UniversalGraphEdge", "UniversalInputFormat", "UniversalMathCondition", "UniversalMathEquation", "UniversalMathParameter", "UniversalMathSpec", "UniversalMathVariable", "UniversalModelKind", "UniversalMovingEntity", "UniversalNormalizedMath", "UniversalNumericsSpec", "UniversalOriginalInput", "UniversalPortRef", "UniversalSolverSpec", "UniversalStationaryEntity", "assertUniversalDESModelSpec", "isUniversalDESModelSpec", "universalFromMathEquationResult", "universalToDESModelSpec", "universalToMathEquationInput", "validateUniversalDESModelSpec"],
);
