//! Built-in first-class citizens and a default registry.
//!
//! This proves the [`ModelCitizen`] contract is paradigm-neutral: acausal
//! equations, MDP/POMDP citizens (from [`crate::des::decision`]), the hybrid
//! block-diagram citizen, and Studio graphs are registered side by side, each
//! advertising a `$schema` and rendering through the same uniform artifact.

use serde_json::{json, Value};

use crate::des::acausal::AcausalCitizen;
use crate::des::decision::{MdpCitizen, PomdpCitizen};
use crate::des::hybrid::{
    demos as hybrid_demos, executive::simulate, spec as hybrid_spec, HYBRID_GRAPH_SCHEMA,
};
use crate::des::plugin::UiControl;
use crate::des::studio::StudioCitizen;

use super::artifact::RunArtifact;
use super::registry::{CitizenError, CitizenRegistry, ModelCitizen, ModelDescriptor};

pub const HYBRID_SCHEMA: &str = HYBRID_GRAPH_SCHEMA;
pub const HYBRID_DEMO_SCHEMA: &str = "des/hybrid-demo/v1";

/// Hybrid block-diagram engine as a first-class citizen.
///
/// The citizen accepts the schema-backed JSON graph spec and still supports the
/// older demo selector for smoke tests and examples.
pub struct HybridCitizen;

impl ModelCitizen for HybridCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "hybrid".to_string(),
            title: "Hybrid Block Diagram".to_string(),
            description: "Continuous + discrete + event-driven block diagram (the Simulink-style \
                          executive). Runs typed JSON graph specs generated from Rust JSON Schema \
                          and can emit a Rust runner."
                .to_string(),
            spec_schema: HYBRID_SCHEMA.to_string(),
            methods: vec![
                "simulate".to_string(),
                "rust-codegen".to_string(),
                "closed-loop".to_string(),
                "bouncing-ball".to_string(),
            ],
            example_spec: serde_json::to_value(hybrid_spec::starter_hybrid_model_spec())
                .unwrap_or_else(
                    |_| json!({ "$schema": HYBRID_DEMO_SCHEMA, "demo": "bouncing-ball" }),
                ),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        if spec.get("blocks").is_some()
            || spec
                .get("$schema")
                .and_then(Value::as_str)
                .map(|schema| schema == HYBRID_SCHEMA)
                .unwrap_or(false)
        {
            let model: hybrid_spec::HybridModelSpec = serde_json::from_value(spec.clone())
                .map_err(|e| {
                    CitizenError::InvalidSpec(format!("invalid hybrid model spec: {e}"))
                })?;
            let (compiled, opts) = hybrid_spec::compile_hybrid_spec(&model)
                .map_err(|e| CitizenError::InvalidSpec(e.to_string()))?;
            let trace = simulate(&compiled, &opts);
            let frames = trace.to_jsonl_frames();
            let results = json!({
                "kind": "hybrid",
                "model": model,
                "events": trace.events,
                "columns": trace.columns,
                "samples": trace.times.len(),
                "generatedRust": hybrid_spec::generate_rust_code(&model),
                "jsonSchema": hybrid_spec::hybrid_model_json_schema(),
            });
            let summary = format!(
                "Hybrid `{}` run: {} samples, {} event(s).",
                model.name,
                trace.times.len(),
                trace.events
            );
            return Ok(RunArtifact::sim(
                "hybrid",
                &model.name,
                "Schema-backed hybrid JSON graph rendered as an animated scope.",
                frames,
                results,
                vec![UiControl::range(
                    "speed",
                    "Speed (fps)",
                    1.0,
                    60.0,
                    1.0,
                    20.0,
                )],
                &summary,
            ));
        }

        let demo = spec
            .get("demo")
            .and_then(Value::as_str)
            .unwrap_or("closed-loop");
        let (compiled, opts) = match demo {
            "bouncing-ball" => hybrid_demos::bouncing_ball(),
            "closed-loop" => hybrid_demos::closed_loop(),
            other => {
                return Err(CitizenError::InvalidSpec(format!(
                    "unknown hybrid demo `{other}` (expected `closed-loop` or `bouncing-ball`)"
                )))
            }
        }
        .map_err(|e| CitizenError::Run(format!("{e:?}")))?;

        let trace = simulate(&compiled, &opts);
        let frames = trace.to_jsonl_frames();
        let results = json!({
            "kind": "hybrid",
            "demo": demo,
            "events": trace.events,
            "columns": trace.columns,
            "samples": trace.times.len(),
        });
        let summary = format!(
            "Hybrid `{demo}` run: {} samples, {} event(s).",
            trace.times.len(),
            trace.events
        );
        Ok(RunArtifact::sim(
            "hybrid",
            "Hybrid Block Diagram",
            "Mixed continuous/discrete/event simulation rendered as an animated scope.",
            frames,
            results,
            vec![UiControl::range(
                "speed",
                "Speed (fps)",
                1.0,
                60.0,
                1.0,
                20.0,
            )],
            &summary,
        ))
    }
}

/// A registry pre-loaded with the built-in first-class citizens: acausal
/// equation models, MDP, POMDP, the hybrid block-diagram engine, and the
/// visual-block studio - peers under one contract.
pub fn with_builtins() -> CitizenRegistry {
    let mut reg = CitizenRegistry::new();
    reg.register(Box::new(AcausalCitizen));
    reg.register(Box::new(MdpCitizen));
    reg.register(Box::new(PomdpCitizen));
    reg.register(Box::new(HybridCitizen));
    reg.register(Box::new(StudioCitizen));
    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lists_all_peer_kinds() {
        let reg = with_builtins();
        let kinds = reg.kinds();
        assert!(kinds.contains(&"acausal".to_string()));
        assert!(kinds.contains(&"mdp".to_string()));
        assert!(kinds.contains(&"pomdp".to_string()));
        assert!(kinds.contains(&"hybrid".to_string()));
        assert!(kinds.contains(&"studio".to_string()));
    }

    #[test]
    fn registry_runs_each_citizen_from_its_example_spec() {
        let reg = with_builtins();
        for desc in reg.descriptors() {
            let art = reg
                .run(&desc.kind, &desc.example_spec)
                .unwrap_or_else(|e| panic!("kind {} failed: {e}", desc.kind));
            assert_eq!(art.kind, desc.kind);
            assert!(
                !art.frames.is_empty(),
                "kind {} produced no frames",
                desc.kind
            );
            // Every artifact renders to a non-trivial HTML page.
            let html = art.to_player_html();
            assert!(
                html.contains("<html") || html.contains("<!DOCTYPE"),
                "kind {} html",
                desc.kind
            );
        }
    }

    #[test]
    fn hybrid_rejects_unknown_demo() {
        let c = HybridCitizen;
        match c.run_json(&json!({ "demo": "nope" })) {
            Err(CitizenError::InvalidSpec(_)) => {}
            other => panic!("expected InvalidSpec, got {other:?}"),
        }
    }
}
