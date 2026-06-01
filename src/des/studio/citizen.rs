//! The studio (visual block graph) as a first-class [`ModelCitizen`], so a flat
//! VisualBlock diagram is discoverable and runnable-from-JSON like every other
//! paradigm, rendering through the same uniform artifact.

use serde_json::{json, Value};

use crate::des::model::{CitizenError, ModelCitizen, ModelDescriptor, RunArtifact};

use super::analysis::analyze_model_spec;
use super::demos::{mixer, queue_line, signal_chain, StudioDemo};
use super::run::run;
use super::spec::{
    compile_model_spec, generate_rust_code, starter_model_spec, studio_model_json_schema,
    StudioModelSpec, STUDIO_GRAPH_SCHEMA,
};

pub const STUDIO_SCHEMA: &str = STUDIO_GRAPH_SCHEMA;
pub const STUDIO_DEMO_SCHEMA: &str = "des/studio-demo/v1";

/// Visual-block dataflow studio citizen.
pub struct StudioCitizen;

impl ModelCitizen for StudioCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "studio".to_string(),
            title: "Visual Block Studio".to_string(),
            description:
                "A JSON-authored graph of visual blocks (Layer 1) wired by typed ports; each \
                          block runs a cell of one or more Layer-2 runtime elements (Layer 2). \
                          Blocks never nest. Saved specs derive JSON Schema from Rust types and \
                          can emit a Rust runner."
                    .to_string(),
            spec_schema: STUDIO_SCHEMA.to_string(),
            methods: vec![
                "model-spec".to_string(),
                "signal-chain".to_string(),
                "mixer".to_string(),
                "queue-line".to_string(),
            ],
            example_spec: {
                serde_json::to_value(starter_model_spec()).unwrap_or_else(
                    |_| json!({ "$schema": STUDIO_DEMO_SCHEMA, "demo": "signal-chain" }),
                )
            },
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        if spec.get("blocks").is_some() {
            let model: StudioModelSpec = serde_json::from_value(spec.clone()).map_err(|e| {
                CitizenError::InvalidSpec(format!("invalid studio model spec: {e}"))
            })?;
            let mut compiled =
                compile_model_spec(&model).map_err(|e| CitizenError::InvalidSpec(e.to_string()))?;
            let blocks = super::demos::blocks_doc(&compiled);
            let run_out = run(&mut compiled, model.steps, model.dt);
            let mut artifact = run_out.to_artifact(
                "studio",
                &model.name,
                "A saved Studio model spec rendered as a live wiring diagram.",
                blocks,
            );
            if let Value::Object(map) = &mut artifact.results {
                map.insert(
                    "model".to_string(),
                    serde_json::to_value(&model).unwrap_or(Value::Null),
                );
                map.insert(
                    "analysis".to_string(),
                    serde_json::to_value(analyze_model_spec(&model)).unwrap_or(Value::Null),
                );
                map.insert(
                    "generatedRust".to_string(),
                    Value::from(generate_rust_code(&model)),
                );
                map.insert("jsonSchema".to_string(), studio_model_json_schema());
            }
            return Ok(artifact);
        }

        let demo_name = spec
            .get("demo")
            .and_then(Value::as_str)
            .unwrap_or("signal-chain");
        let demo: StudioDemo = match demo_name {
            "signal-chain" => signal_chain().map_err(|e| CitizenError::Run(e.to_string())),
            "mixer" => mixer().map_err(|e| CitizenError::Run(e.to_string())),
            "queue-line" => queue_line().map_err(|e| CitizenError::Run(e.to_string())),
            other => Err(CitizenError::InvalidSpec(format!(
                "unknown studio demo `{other}` (expected `signal-chain`, `mixer` or `queue-line`)"
            ))),
        }?;

        let mut demo = demo;
        let run_out = run(&mut demo.compiled, demo.steps, demo.dt);
        Ok(run_out.to_artifact("studio", &demo.title, &demo.description, demo.blocks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_citizen_runs_its_example() {
        let c = StudioCitizen;
        let art = c.run_json(&c.descriptor().example_spec).unwrap();
        assert_eq!(art.kind, "studio");
        assert!(!art.frames.is_empty());
        assert!(art.results["blocks"].is_array());
        assert!(art.results["analysis"]["validation"]["ok"]
            .as_bool()
            .unwrap());
    }

    #[test]
    fn unknown_demo_is_invalid_spec() {
        let c = StudioCitizen;
        match c.run_json(&json!({ "demo": "nope" })) {
            Err(CitizenError::InvalidSpec(_)) => {}
            other => panic!("expected InvalidSpec, got {other:?}"),
        }
    }

    #[test]
    fn studio_citizen_runs_saved_model_spec() {
        let c = StudioCitizen;
        let spec = serde_json::to_value(starter_model_spec()).unwrap();
        let art = c.run_json(&spec).unwrap();
        assert_eq!(art.title, "ramp-gain-sink");
        assert_eq!(art.results["model"]["name"], "ramp-gain-sink");
    }
}
