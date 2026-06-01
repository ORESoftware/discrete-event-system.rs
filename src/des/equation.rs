//! First-class equation-based modeling citizen.
//!
//! This is the open ModelingToolkit-style seam over the equation machinery that
//! already exists in `general::math_equation_input`: JSON/LaTeX/XML specs are
//! normalized into a block network, simulated, and returned as the same
//! [`RunArtifact`] used by the visual studio, hybrid diagrams, and decision
//! models.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::des::general::des_spec::{JsonObject, JsonValue};
use crate::des::general::math_equation_input::{
    normalize_math_equation_problem, run_math_equation_problem, EquationInputFormat,
    EquationProblemKind, EquationValidationCheck, Heat1DBlockResult, IntegratorMethod,
    MathEquationInputParams, MathEquationNetwork, MathEquationResult, Normalized,
    ODEBlockSystemResult,
};
use crate::des::model::{CitizenError, ModelCitizen, ModelDescriptor, RunArtifact};
use crate::des::plugin::UiControl;

/// Schema id for equation-based model specs.
pub const EQUATION_SCHEMA: &str = "des/equation/v1";
const MAX_ODE_STATES: usize = 64;
const MAX_TRACE_STEPS: usize = 50_000;
const MAX_HEAT_CELLS: usize = 1_000;
const MAX_HEAT_CELL_STEPS: usize = 250_000;

/// Equation-based ODE / PDE citizen.
pub struct EquationCitizen;

impl ModelCitizen for EquationCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "equation".to_string(),
            title: "Equation-Based Model".to_string(),
            description: "Symbolic-numeric equation input for ODE and heat-equation models: \
                          JSON, LaTeX, or XML in; normalized block network, simulation trace, \
                          validation, and artifact out."
                .to_string(),
            spec_schema: EQUATION_SCHEMA.to_string(),
            methods: vec![
                "json-ode".to_string(),
                "latex-ode".to_string(),
                "xml-ode".to_string(),
                "heat1d".to_string(),
            ],
            example_spec: example_spec(),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        let params = params_from_spec(spec)?;
        let normalized = normalize_math_equation_problem(&params)
            .map_err(|e| CitizenError::InvalidSpec(e.to_string()))?;
        validate_normalized(&normalized)?;
        let title = spec
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("Equation-Based Model");
        let description = spec
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("Normalized equation model compiled into a runnable simulation trace.");

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_math_equation_problem(&params, None)
        }))
        .map_err(|_| CitizenError::Run("equation solver panicked while parsing or running".into()))?
        .map_err(|e| CitizenError::InvalidSpec(e.to_string()))?;

        let frames = frames_from_result(&result);
        let results = results_json(&result);
        let summary = summary(&result);

        Ok(RunArtifact::sim(
            "equation",
            title,
            description,
            frames,
            results,
            vec![
                UiControl::range("speed", "Speed (fps)", 1.0, 60.0, 1.0, 12.0),
                UiControl::toggle(
                    "show_validations",
                    "Show validations",
                    true,
                    Some("validation"),
                ),
            ],
            &summary,
        ))
    }
}

/// Compact ODE example: exponential decay.
pub fn example_spec() -> Value {
    json!({
        "$schema": EQUATION_SCHEMA,
        "title": "Equation Spec: Exponential Decay",
        "description": "A JSON-authored ODE model normalized into an equation network and simulated.",
        "format": "json",
        "kind": "ode",
        "simulation": { "t0": 0.0, "t1": 4.0, "dt": 0.05, "method": "trapezoid" },
        "constants": { "k": 0.7 },
        "states": [
            { "name": "x", "initial": 1.0, "derivative": "-k*x" }
        ]
    })
}

fn params_from_spec(spec: &Value) -> Result<MathEquationInputParams, CitizenError> {
    if !spec.is_object() {
        return Err(CitizenError::InvalidSpec(
            "equation spec must be a JSON object".into(),
        ));
    }
    if let Some(schema) = read_str(spec, "$schema") {
        if schema != EQUATION_SCHEMA {
            return Err(CitizenError::InvalidSpec(format!(
                "unsupported equation schema `{schema}` (expected `{EQUATION_SCHEMA}`)"
            )));
        }
    }
    require_object_field(spec, "simulation")?;
    require_object_field(spec, "constants")?;
    require_object_field(spec, "initial")?;
    require_array_field(spec, "states")?;
    require_object_field(spec, "ode")?;
    require_object_field(spec, "heat1d")?;

    let format = match read_str(spec, "format").unwrap_or("json") {
        "json" => EquationInputFormat::Json,
        "latex" => EquationInputFormat::Latex,
        "xml" => EquationInputFormat::Xml,
        other => {
            return Err(CitizenError::InvalidSpec(format!(
                "unknown equation format `{other}` (expected `json`, `latex`, or `xml`)"
            )))
        }
    };
    let kind = match read_str(spec, "kind") {
        Some("ode") => Some(EquationProblemKind::Ode),
        Some("heat1d") => Some(EquationProblemKind::Heat1d),
        Some(other) => {
            return Err(CitizenError::InvalidSpec(format!(
                "unknown equation kind `{other}` (expected `ode` or `heat1d`)"
            )))
        }
        None => None,
    };
    let simulation = spec.get("simulation").unwrap_or(&Value::Null);

    let mut params = MathEquationInputParams {
        format,
        kind,
        equation: read_str(spec, "equation").map(str::to_string),
        states: json_array(spec.get("states")),
        constants: json_object(spec.get("constants")),
        initial: json_object(spec.get("initial")),
        t0: read_f64(spec, "t0").or_else(|| read_f64(simulation, "t0")),
        t1: read_f64(spec, "t1").or_else(|| read_f64(simulation, "t1")),
        dt: read_f64(spec, "dt").or_else(|| read_f64(simulation, "dt")),
        method: read_str(spec, "method")
            .or_else(|| read_str(simulation, "method"))
            .map(parse_method)
            .transpose()?,
        cells: read_f64(spec, "cells"),
        length: read_f64(spec, "length"),
        alpha: read_f64(spec, "alpha"),
        initial_expression: read_str(spec, "initialExpression").map(str::to_string),
        initial_values: json_number_array(spec.get("initialValues"))?,
        left_boundary: read_f64(spec, "leftBoundary"),
        right_boundary: read_f64(spec, "rightBoundary"),
        ..Default::default()
    };

    if let Some(ode) = spec.get("ode") {
        params.ode = json_object(Some(ode));
    }
    if let Some(heat) = spec.get("heat1d") {
        params.heat1d = json_object(Some(heat));
        params.cells = params.cells.or_else(|| read_f64(heat, "cells"));
        params.length = params.length.or_else(|| read_f64(heat, "length"));
        params.alpha = params.alpha.or_else(|| read_f64(heat, "alpha"));
        params.initial_expression = params
            .initial_expression
            .or_else(|| read_str(heat, "initialExpression").map(str::to_string));
        params.initial_values = params
            .initial_values
            .or(json_number_array(heat.get("initialValues"))?);
        params.left_boundary = params
            .left_boundary
            .or_else(|| read_f64(heat, "leftBoundary"));
        params.right_boundary = params
            .right_boundary
            .or_else(|| read_f64(heat, "rightBoundary"));
    }

    Ok(params)
}

fn validate_normalized(normalized: &Normalized) -> Result<(), CitizenError> {
    match normalized {
        Normalized::Ode(p) => {
            if p.states.len() > MAX_ODE_STATES {
                return Err(CitizenError::InvalidSpec(format!(
                    "equation ODE state count {} exceeds the artifact cap of {MAX_ODE_STATES}",
                    p.states.len()
                )));
            }
            checked_steps("equation ODE", p.t0, p.t1, p.dt, MAX_TRACE_STEPS)?;
        }
        Normalized::Heat1d(p) => {
            let steps = checked_steps("equation heat1d", p.t0, p.t1, p.dt, MAX_TRACE_STEPS)?;
            if p.cells > MAX_HEAT_CELLS as f64 {
                return Err(CitizenError::InvalidSpec(format!(
                    "equation heat1d cell count {} exceeds the artifact cap of {MAX_HEAT_CELLS}",
                    p.cells as usize
                )));
            }
            let cells = p.cells.max(0.0) as usize;
            let samples = cells.saturating_mul(steps.saturating_add(1));
            if samples > MAX_HEAT_CELL_STEPS {
                return Err(CitizenError::InvalidSpec(format!(
                    "equation heat1d grid has {samples} cell-step samples, above the artifact cap of {MAX_HEAT_CELL_STEPS}"
                )));
            }
        }
    }
    Ok(())
}

fn checked_steps(
    label: &str,
    t0: f64,
    t1: f64,
    dt: f64,
    max_steps: usize,
) -> Result<usize, CitizenError> {
    if !t0.is_finite() || !t1.is_finite() || !dt.is_finite() {
        return Err(CitizenError::InvalidSpec(format!(
            "{label} time grid must be finite"
        )));
    }
    if dt <= 0.0 {
        return Err(CitizenError::InvalidSpec(format!(
            "{label} dt must be positive"
        )));
    }
    if t1 <= t0 {
        return Err(CitizenError::InvalidSpec(format!(
            "{label} must satisfy t1 > t0"
        )));
    }
    let exact = (t1 - t0) / dt;
    if !exact.is_finite() {
        return Err(CitizenError::InvalidSpec(format!(
            "{label} step count must be finite"
        )));
    }
    let steps = exact.round();
    let tolerance = 1e-9 * exact.abs().max(1.0);
    if (exact - steps).abs() > tolerance {
        return Err(CitizenError::InvalidSpec(format!(
            "{label} duration/dt must be an integer number of steps"
        )));
    }
    if steps < 1.0 || steps > max_steps as f64 {
        return Err(CitizenError::InvalidSpec(format!(
            "{label} step count {} is outside the supported range 1..={max_steps}",
            steps as usize
        )));
    }
    Ok(steps as usize)
}

fn parse_method(value: &str) -> Result<IntegratorMethod, CitizenError> {
    match value {
        "euler" => Ok(IntegratorMethod::Euler),
        "trapezoid" => Ok(IntegratorMethod::Trapezoid),
        other => Err(CitizenError::InvalidSpec(format!(
            "unknown integration method `{other}` (expected `euler` or `trapezoid`)"
        ))),
    }
}

fn frames_from_result(result: &MathEquationResult) -> Vec<Value> {
    match (&result.ode, &result.heat1d) {
        (Some(ode), _) => ode_frames(ode),
        (_, Some(heat)) => heat_frames(heat),
        _ => Vec::new(),
    }
}

fn ode_frames(ode: &ODEBlockSystemResult) -> Vec<Value> {
    ode.trace
        .iter()
        .map(|row| {
            let mut obj = serde_json::Map::new();
            obj.insert("t".to_string(), json!(row.time));
            obj.insert("tick".to_string(), json!(row.tick));
            obj.insert("caption".to_string(), json!(format!("t={:.3}", row.time)));
            for state in &ode.params.states {
                if let Some(v) = row.state.get(&state.name) {
                    obj.insert(state.name.clone(), json!(v));
                }
                if let Some(dv) = row.derivatives.get(&state.name) {
                    obj.insert(format!("d_{}", state.name), json!(dv));
                }
            }
            Value::Object(obj)
        })
        .collect()
}

fn heat_frames(heat: &Heat1DBlockResult) -> Vec<Value> {
    let center = heat.params.cells as usize / 2;
    heat.trace
        .iter()
        .map(|row| {
            json!({
                "t": row.time,
                "tick": row.tick,
                "min": row.min,
                "max": row.max,
                "mean": row.mean,
                "center": row.values.get(center).copied().unwrap_or(0.0),
                "caption": format!("t={:.3}; heat grid min={:.3}, max={:.3}", row.time, row.min, row.max),
            })
        })
        .collect()
}

fn results_json(result: &MathEquationResult) -> Value {
    json!({
        "kind": "equation",
        "format": format_str(result.input_format),
        "problemKind": kind_str(result.kind),
        "equation": result.equation,
        "normalized": normalized_json(&result.normalized),
        "network": network_json(&result.network),
        "validation": validation_json(&result.validation),
        "ode": result.ode.as_ref().map(ode_json),
        "heat1d": result.heat1d.as_ref().map(heat_json),
        "workflow": workflow_json(),
    })
}

fn normalized_json(normalized: &Normalized) -> Value {
    match normalized {
        Normalized::Ode(p) => json!({
            "kind": "ode",
            "states": p.states.iter().map(|s| json!({
                "name": s.name,
                "initial": s.initial,
                "derivative": s.derivative,
            })).collect::<Vec<_>>(),
            "constants": string_map_json(&p.constants),
            "t0": p.t0,
            "t1": p.t1,
            "dt": p.dt,
            "method": method_str(p.method),
        }),
        Normalized::Heat1d(p) => json!({
            "kind": "heat1d",
            "cells": p.cells,
            "length": p.length,
            "alpha": p.alpha,
            "t0": p.t0,
            "t1": p.t1,
            "dt": p.dt,
            "initialExpression": p.initial_expression,
            "leftBoundary": p.left_boundary,
            "rightBoundary": p.right_boundary,
            "constants": string_map_json(&p.constants),
        }),
    }
}

fn ode_json(ode: &ODEBlockSystemResult) -> Value {
    json!({
        "steps": ode.steps,
        "finalState": ode.final_state.iter().map(|(name, value)| {
            json!({ "state": name, "value": value })
        }).collect::<Vec<_>>(),
        "traceRows": ode.trace.len(),
    })
}

fn heat_json(heat: &Heat1DBlockResult) -> Value {
    let final_values = heat
        .trace
        .last()
        .map(|row| row.values.clone())
        .unwrap_or_default();
    json!({
        "steps": heat.steps,
        "dx": heat.dx,
        "cfl": heat.cfl,
        "x": heat.x,
        "finalValues": final_values,
        "traceRows": heat.trace.len(),
    })
}

fn network_json(network: &MathEquationNetwork) -> Value {
    json!({
        "nodes": network.nodes.iter().map(|n| json!({
            "id": n.id,
            "kind": n.kind,
            "inputs": n.inputs,
            "output": n.output,
            "expression": n.expression,
        })).collect::<Vec<_>>(),
        "edges": network.edges.iter().map(|e| json!({
            "from": e.from,
            "to": e.to,
            "fromChannel": e.from_channel,
            "toChannel": e.to_channel,
            "signal": e.signal,
        })).collect::<Vec<_>>(),
    })
}

fn validation_json(checks: &[EquationValidationCheck]) -> Value {
    json!(checks
        .iter()
        .map(|c| json!({
            "name": c.name,
            "passed": c.passed,
            "group": c.group,
        }))
        .collect::<Vec<_>>())
}

fn workflow_json() -> Value {
    json!([
        { "stage": "author", "status": "available", "detail": "JSON, LaTeX, and XML equation input." },
        { "stage": "normalize", "status": "available", "detail": "Equations normalize into ODE/heat parameters and block-network nodes." },
        { "stage": "simulate", "status": "available", "detail": "ODE and heat traces stream into the common artifact player." },
        { "stage": "structural-analysis", "status": "partial", "detail": "Validation and graph extraction exist; alias elimination, tearing, and DAE index reduction are next." },
        { "stage": "calibration-uq-surrogate-control", "status": "planned", "detail": "The platform has optimization/control primitives; this citizen is the contract to attach JuliaSim-like analyses." }
    ])
}

fn summary(result: &MathEquationResult) -> String {
    match (&result.ode, &result.heat1d) {
        (Some(ode), _) => {
            let states = ode.params.states.len();
            let final_bits = ode
                .final_state
                .iter()
                .map(|(n, v)| format!("{n}={v:.4}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "Equation ODE run: {states} state(s), {} steps, final {final_bits}.",
                ode.steps
            )
        }
        (_, Some(heat)) => format!(
            "Equation heat1d run: {} cells, {} steps, CFL {:.4}.",
            heat.params.cells as usize, heat.steps, heat.cfl
        ),
        _ => "Equation run produced no trace.".to_string(),
    }
}

fn read_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.as_object()?.get(key)?.as_str()
}

fn read_f64(value: &Value, key: &str) -> Option<f64> {
    value.as_object()?.get(key)?.as_f64()
}

fn json_object(value: Option<&Value>) -> Option<JsonObject> {
    match value.cloned().map(JsonValue::from) {
        Some(JsonValue::Object(o)) => Some(o),
        _ => None,
    }
}

fn json_array(value: Option<&Value>) -> Option<Vec<JsonValue>> {
    match value.cloned().map(JsonValue::from) {
        Some(JsonValue::Array(a)) => Some(a),
        _ => None,
    }
}

fn json_number_array(value: Option<&Value>) -> Result<Option<Vec<f64>>, CitizenError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for (idx, item) in items.iter().enumerate() {
                out.push(item.as_f64().ok_or_else(|| {
                    CitizenError::InvalidSpec(format!("initialValues[{idx}] must be numeric"))
                })?);
            }
            Ok(Some(out))
        }
        Some(_) => Err(CitizenError::InvalidSpec(
            "initialValues must be an array of numbers".into(),
        )),
    }
}

fn string_map_json(map: &HashMap<String, f64>) -> Value {
    let mut obj = serde_json::Map::new();
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    for key in keys {
        obj.insert(key.clone(), json!(map[key]));
    }
    Value::Object(obj)
}

fn format_str(format: EquationInputFormat) -> &'static str {
    match format {
        EquationInputFormat::Json => "json",
        EquationInputFormat::Latex => "latex",
        EquationInputFormat::Xml => "xml",
    }
}

fn kind_str(kind: EquationProblemKind) -> &'static str {
    match kind {
        EquationProblemKind::Ode => "ode",
        EquationProblemKind::Heat1d => "heat1d",
    }
}

fn method_str(method: IntegratorMethod) -> &'static str {
    match method {
        IntegratorMethod::Euler => "euler",
        IntegratorMethod::Trapezoid => "trapezoid",
    }
}

fn require_object_field(spec: &Value, key: &str) -> Result<(), CitizenError> {
    if let Some(value) = spec.get(key) {
        if !value.is_object() && !value.is_null() {
            return Err(CitizenError::InvalidSpec(format!(
                "equation field `{key}` must be an object"
            )));
        }
    }
    Ok(())
}

fn require_array_field(spec: &Value, key: &str) -> Result<(), CitizenError> {
    if let Some(value) = spec.get(key) {
        if !value.is_array() && !value.is_null() {
            return Err(CitizenError::InvalidSpec(format!(
                "equation field `{key}` must be an array"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equation_citizen_runs_example_ode() {
        let c = EquationCitizen;
        let art = c.run_json(&example_spec()).unwrap();
        assert_eq!(art.kind, "equation");
        assert!(!art.frames.is_empty());
        assert_eq!(art.results["problemKind"], "ode");
        let final_state = art.results["ode"]["finalState"].as_array().unwrap();
        let x = final_state[0]["value"].as_f64().unwrap();
        assert!(x > 0.0 && x < 1.0);
    }

    #[test]
    fn equation_citizen_runs_latex_ode() {
        let c = EquationCitizen;
        let spec = json!({
            "$schema": EQUATION_SCHEMA,
            "format": "latex",
            "kind": "ode",
            "equation": "\\frac{dx}{dt} = -k x",
            "constants": { "k": 0.5 },
            "initial": { "x": 1.0 },
            "simulation": { "t0": 0.0, "t1": 1.0, "dt": 0.1 }
        });
        let art = c.run_json(&spec).unwrap();
        assert_eq!(art.results["format"], "latex");
        assert!(!art.frames.is_empty());
    }

    #[test]
    fn unknown_format_is_invalid_spec() {
        let c = EquationCitizen;
        match c.run_json(&json!({ "format": "spreadsheet" })) {
            Err(CitizenError::InvalidSpec(_)) => {}
            other => panic!("expected InvalidSpec, got {other:?}"),
        }
    }

    #[test]
    fn wrong_schema_is_invalid_spec() {
        let c = EquationCitizen;
        match c.run_json(&json!({ "$schema": "des/other/v1" })) {
            Err(CitizenError::InvalidSpec(msg)) => assert!(msg.contains("unsupported")),
            other => panic!("expected InvalidSpec, got {other:?}"),
        }
    }

    #[test]
    fn oversized_trace_is_rejected_before_run() {
        let c = EquationCitizen;
        let spec = json!({
            "$schema": EQUATION_SCHEMA,
            "format": "json",
            "kind": "ode",
            "simulation": { "t0": 0.0, "t1": 100.0, "dt": 0.001 },
            "states": [{ "name": "x", "initial": 1.0, "derivative": "-x" }]
        });
        match c.run_json(&spec) {
            Err(CitizenError::InvalidSpec(msg)) => assert!(msg.contains("step count")),
            other => panic!("expected InvalidSpec, got {other:?}"),
        }
    }

    #[test]
    fn malformed_sections_are_invalid_spec() {
        let c = EquationCitizen;
        match c.run_json(&json!({ "kind": "ode", "constants": ["k"], "states": [] })) {
            Err(CitizenError::InvalidSpec(msg)) => assert!(msg.contains("constants")),
            other => panic!("expected InvalidSpec, got {other:?}"),
        }
    }
}
