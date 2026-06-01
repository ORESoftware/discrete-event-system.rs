//! The studio (visual block graph) as a first-class [`ModelCitizen`], so a flat
//! VisualBlock diagram is discoverable and runnable-from-JSON like every other
//! paradigm, rendering through the same uniform artifact.

use serde_json::Value;

use crate::des::model::{CitizenError, ModelCitizen, ModelDescriptor, RunArtifact};

use super::demos::{mixer, queue_line, signal_chain, StudioDemo};
use super::design::run_design_study;
use super::run::run;
use super::spec::{demo_from_spec, example_spec, STUDIO_SPEC_SCHEMA};

pub const STUDIO_SCHEMA: &str = "des/studio-demo/v1";

/// Visual-block dataflow studio citizen.
pub struct StudioCitizen;

impl ModelCitizen for StudioCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "studio".to_string(),
            title: "Visual Block Studio".to_string(),
            description: "A flat graph of visual blocks (Layer 1) wired by typed ports; each \
                          block runs a cell of one or more Layer-2 runtime elements (Layer 2). \
                          Blocks never nest. Selects a demo and renders the live wiring diagram."
                .to_string(),
            spec_schema: STUDIO_SPEC_SCHEMA.to_string(),
            methods: vec![
                "json-spec".to_string(),
                "signal-chain".to_string(),
                "mixer".to_string(),
                "queue-line".to_string(),
            ],
            example_spec: example_spec(),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        let schema = spec.get("$schema").and_then(Value::as_str);
        if schema == Some(STUDIO_SPEC_SCHEMA) || spec.get("blocks").is_some() {
            let design_run =
                run_design_study(spec).map_err(|e| CitizenError::InvalidSpec(e.to_string()))?;
            let runnable_spec = design_run.as_ref().map(|d| &d.final_spec).unwrap_or(spec);
            let mut demo = demo_from_spec(runnable_spec)
                .map_err(|e| CitizenError::InvalidSpec(e.to_string()))?;
            let run_out = run(&mut demo.compiled, demo.steps, demo.dt);
            let mut artifact =
                run_out.to_artifact("studio", &demo.title, &demo.description, demo.blocks);
            if let Some(design) = design_run {
                if let Value::Object(results) = &mut artifact.results {
                    results.insert("designStudy".to_string(), design.to_json());
                    results.insert("finalSpec".to_string(), design.final_spec.clone());
                }
                artifact.summary = format!(
                    "{} Optimized objective {:.6} -> {:.6}.",
                    artifact.summary, design.initial_objective, design.final_objective
                );
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
                "unknown studio demo `{other}` (expected `signal-chain`, `mixer`, `queue-line`, or a `{STUDIO_SPEC_SCHEMA}` JSON block spec)"
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
    use serde_json::json;

    #[test]
    fn studio_citizen_runs_its_example() {
        let c = StudioCitizen;
        let art = c.run_json(&c.descriptor().example_spec).unwrap();
        assert_eq!(art.kind, "studio");
        assert!(!art.frames.is_empty());
        assert!(art.results["blocks"].is_array());
    }

    #[test]
    fn unknown_demo_is_invalid_spec() {
        let c = StudioCitizen;
        match c.run_json(&json!({ "demo": "nope" })) {
            Err(CitizenError::InvalidSpec(_)) => {}
            other => panic!("expected InvalidSpec, got {other:?}"),
        }
    }
}
