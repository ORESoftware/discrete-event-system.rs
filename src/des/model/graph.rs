//! Unified graph-spec envelope for embedders that accept either Studio or
//! Hybrid graph JSON.

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::des::{hybrid, studio};

pub const MODEL_GRAPH_SCHEMA: &str = "des/model-graph/v1";

fn model_graph_schema() -> String {
    MODEL_GRAPH_SCHEMA.to_string()
}

/// Unified graph envelope for the specs that can be authored today.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelGraphSpec {
    #[serde(rename = "$schema", default = "model_graph_schema")]
    pub schema: String,
    #[serde(flatten)]
    pub graph: ModelGraphKind,
}

/// A typed sum of concrete graph specs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case", tag = "kind", content = "spec")]
pub enum ModelGraphKind {
    Studio(studio::StudioModelSpec),
    Hybrid(hybrid::HybridModelSpec),
}

/// JSON Schema for the unified graph envelope.
pub fn model_graph_json_schema() -> Value {
    serde_json::to_value(schema_for!(ModelGraphSpec)).expect("ModelGraphSpec schema serializes")
}

/// Generate a Rust runner for either graph kind.
pub fn generate_model_graph_rust_code(spec: &ModelGraphSpec) -> String {
    match &spec.graph {
        ModelGraphKind::Studio(spec) => studio::generate_rust_code(spec),
        ModelGraphKind::Hybrid(spec) => hybrid::generate_rust_code(spec),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_graph_schema_covers_studio_and_hybrid() {
        let schema = model_graph_json_schema();
        let schema_text = serde_json::to_string(&schema).unwrap();
        assert!(schema_text.contains("StudioModelSpec"));
        assert!(schema_text.contains("HybridModelSpec"));
    }

    #[test]
    fn graph_codegen_dispatches_by_kind() {
        let spec = ModelGraphSpec {
            schema: MODEL_GRAPH_SCHEMA.to_string(),
            graph: ModelGraphKind::Hybrid(hybrid::starter_hybrid_model_spec()),
        };
        let code = generate_model_graph_rust_code(&spec);
        assert!(code.contains("HybridModelSpec"));
    }
}
