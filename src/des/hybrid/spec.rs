//! JSON graph specification for the hybrid block-diagram engine.
//!
//! This is the runnable Simulink-style authoring surface: typed block variants,
//! id-addressed wires, solver options, shared authoring metadata, JSON Schema
//! generation from Rust types, compilation into [`super::diagram::Diagram`],
//! and Rust runner generation.

use std::collections::HashMap;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::des::model::{
    authoring::ModelAuthoringSpec,
    codegen::{rust_ident, rust_raw_string_literal},
};

use super::blocks::{
    BouncingBall, Constant, Counter, DiscretePi, Gain, Integrator, Saturation, StateSpace, Sum,
};
use super::diagram::{BlockHandle, Compiled, Diagram, HybridError};
use super::executive::SimOptions;

pub const HYBRID_GRAPH_SCHEMA: &str = "des/hybrid-graph/v1";

fn hybrid_graph_schema() -> String {
    HYBRID_GRAPH_SCHEMA.to_string()
}

fn default_t_end() -> f64 {
    10.0
}

fn default_max_step() -> f64 {
    0.01
}

fn default_zc_tol() -> f64 {
    1e-9
}

fn one_usize() -> usize {
    1
}

/// JSON Schema for saved Hybrid graph documents.
pub fn hybrid_model_json_schema() -> Value {
    serde_json::to_value(schema_for!(HybridModelSpec)).expect("HybridModelSpec schema serializes")
}

/// Top-level hybrid graph document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HybridModelSpec {
    #[serde(rename = "$schema", default = "hybrid_graph_schema")]
    pub schema: String,
    pub name: String,
    #[serde(default = "default_t_end")]
    pub t_end: f64,
    #[serde(default = "default_max_step")]
    pub max_step: f64,
    #[serde(default = "default_zc_tol")]
    pub zc_tol: f64,
    pub blocks: Vec<HybridBlockSpec>,
    #[serde(default)]
    pub wires: Vec<HybridWireSpec>,
    #[serde(default)]
    pub authoring: ModelAuthoringSpec,
}

/// A typed hybrid block. The `kind` tag makes the JSON Schema a real `oneOf`
/// surface instead of an untyped map of arbitrary parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum HybridBlockSpec {
    Constant {
        id: String,
        value: Vec<f64>,
    },
    Gain {
        id: String,
        #[serde(default = "one_usize")]
        width: usize,
        k: f64,
    },
    Sum {
        id: String,
        #[serde(default = "one_usize")]
        width: usize,
        signs: Vec<f64>,
    },
    Saturation {
        id: String,
        lo: f64,
        hi: f64,
    },
    Integrator {
        id: String,
        initial: Vec<f64>,
    },
    StateSpace {
        id: String,
        a: Vec<Vec<f64>>,
        b: Vec<Vec<f64>>,
        c: Vec<Vec<f64>>,
        d: Vec<Vec<f64>>,
        #[serde(default)]
        x0: Option<Vec<f64>>,
    },
    DiscretePi {
        id: String,
        kp: f64,
        ki: f64,
        period: f64,
    },
    Counter {
        id: String,
        period: f64,
    },
    BouncingBall {
        id: String,
        h0: f64,
        #[serde(default)]
        v0: f64,
        restitution: f64,
    },
}

impl HybridBlockSpec {
    pub fn id(&self) -> &str {
        match self {
            HybridBlockSpec::Constant { id, .. }
            | HybridBlockSpec::Gain { id, .. }
            | HybridBlockSpec::Sum { id, .. }
            | HybridBlockSpec::Saturation { id, .. }
            | HybridBlockSpec::Integrator { id, .. }
            | HybridBlockSpec::StateSpace { id, .. }
            | HybridBlockSpec::DiscretePi { id, .. }
            | HybridBlockSpec::Counter { id, .. }
            | HybridBlockSpec::BouncingBall { id, .. } => id,
        }
    }
}

/// A directed connection from one block output port to another block input port.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HybridWireSpec {
    pub from: String,
    #[serde(default)]
    pub from_port: usize,
    pub to: String,
    #[serde(default)]
    pub to_port: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HybridSpecError {
    InvalidParam {
        block: String,
        param: String,
        message: String,
    },
    DuplicateBlock(String),
    UnknownBlock(String),
    Diagram(HybridError),
}

impl std::fmt::Display for HybridSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HybridSpecError::InvalidParam {
                block,
                param,
                message,
            } => write!(
                f,
                "block `{block}` parameter `{param}` is invalid: {message}"
            ),
            HybridSpecError::DuplicateBlock(id) => write!(f, "duplicate block id `{id}`"),
            HybridSpecError::UnknownBlock(id) => write!(f, "wire references unknown block `{id}`"),
            HybridSpecError::Diagram(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for HybridSpecError {}

impl From<HybridError> for HybridSpecError {
    fn from(value: HybridError) -> Self {
        HybridSpecError::Diagram(value)
    }
}

fn invalid(block: &str, param: &str, message: impl Into<String>) -> HybridSpecError {
    HybridSpecError::InvalidParam {
        block: block.to_string(),
        param: param.to_string(),
        message: message.into(),
    }
}

fn finite(block: &str, param: &str, value: f64) -> Result<f64, HybridSpecError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(block, param, "expected a finite number"))
    }
}

fn finite_vec(block: &str, param: &str, values: &[f64]) -> Result<(), HybridSpecError> {
    if values.is_empty() {
        return Err(invalid(block, param, "expected at least one value"));
    }
    for value in values {
        finite(block, param, *value)?;
    }
    Ok(())
}

fn matrix_dims(
    block: &str,
    param: &str,
    matrix: &[Vec<f64>],
) -> Result<(usize, usize), HybridSpecError> {
    if matrix.is_empty() {
        return Err(invalid(block, param, "expected a non-empty matrix"));
    }
    let cols = matrix[0].len();
    for row in matrix {
        if row.len() != cols {
            return Err(invalid(block, param, "expected a rectangular matrix"));
        }
        finite_vec(block, param, row)?;
    }
    Ok((matrix.len(), cols))
}

fn validate_state_space(
    id: &str,
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: &[Vec<f64>],
    d: &[Vec<f64>],
    x0: &Option<Vec<f64>>,
) -> Result<(), HybridSpecError> {
    let (an, am) = matrix_dims(id, "a", a)?;
    if an != am {
        return Err(invalid(id, "a", "expected a square n x n matrix"));
    }
    let (bn, bm) = matrix_dims(id, "b", b)?;
    if bn != an {
        return Err(invalid(id, "b", "expected n rows to match a"));
    }
    let (cp, cn) = matrix_dims(id, "c", c)?;
    if cn != an {
        return Err(invalid(id, "c", "expected n columns to match a"));
    }
    let (dp, dm) = matrix_dims(id, "d", d)?;
    if dp != cp || dm != bm {
        return Err(invalid(
            id,
            "d",
            "expected p x m dimensions to match c rows and b columns",
        ));
    }
    if let Some(x0) = x0 {
        if x0.len() != an {
            return Err(invalid(id, "x0", "expected n initial states"));
        }
        finite_vec(id, "x0", x0)?;
    }
    Ok(())
}

fn block_to_runtime(block: &HybridBlockSpec) -> Result<Box<dyn super::Block>, HybridSpecError> {
    let id = block.id();
    match block {
        HybridBlockSpec::Constant { value, .. } => {
            finite_vec(id, "value", value)?;
            Ok(Box::new(Constant::new(id, value.clone())))
        }
        HybridBlockSpec::Gain { width, k, .. } => {
            if *width == 0 {
                return Err(invalid(id, "width", "expected width > 0"));
            }
            Ok(Box::new(Gain::new(id, *width, finite(id, "k", *k)?)))
        }
        HybridBlockSpec::Sum { width, signs, .. } => {
            if *width == 0 {
                return Err(invalid(id, "width", "expected width > 0"));
            }
            finite_vec(id, "signs", signs)?;
            Ok(Box::new(Sum::new(id, *width, signs.clone())))
        }
        HybridBlockSpec::Saturation { lo, hi, .. } => {
            let lo = finite(id, "lo", *lo)?;
            let hi = finite(id, "hi", *hi)?;
            if lo > hi {
                return Err(invalid(id, "lo", "expected lo <= hi"));
            }
            Ok(Box::new(Saturation::new(id, lo, hi)))
        }
        HybridBlockSpec::Integrator { initial, .. } => {
            finite_vec(id, "initial", initial)?;
            Ok(Box::new(Integrator::new(id, initial.clone())))
        }
        HybridBlockSpec::StateSpace { a, b, c, d, x0, .. } => {
            validate_state_space(id, a, b, c, d, x0)?;
            let block = StateSpace::new(id, a.clone(), b.clone(), c.clone(), d.clone());
            Ok(match x0 {
                Some(x0) => Box::new(block.with_x0(x0.clone())),
                None => Box::new(block),
            })
        }
        HybridBlockSpec::DiscretePi { kp, ki, period, .. } => {
            let period = finite(id, "period", *period)?;
            if period <= 0.0 {
                return Err(invalid(id, "period", "expected period > 0"));
            }
            Ok(Box::new(DiscretePi::new(
                id,
                finite(id, "kp", *kp)?,
                finite(id, "ki", *ki)?,
                period,
            )))
        }
        HybridBlockSpec::Counter { period, .. } => {
            let period = finite(id, "period", *period)?;
            if period <= 0.0 {
                return Err(invalid(id, "period", "expected period > 0"));
            }
            Ok(Box::new(Counter::new(id, period)))
        }
        HybridBlockSpec::BouncingBall {
            h0,
            v0,
            restitution,
            ..
        } => {
            let restitution = finite(id, "restitution", *restitution)?;
            if restitution < 0.0 {
                return Err(invalid(id, "restitution", "expected restitution >= 0"));
            }
            Ok(Box::new(BouncingBall::new(
                id,
                finite(id, "h0", *h0)?,
                finite(id, "v0", *v0)?,
                restitution,
            )))
        }
    }
}

/// Compile a saved hybrid model into a runnable diagram and simulation options.
pub fn compile_hybrid_spec(
    spec: &HybridModelSpec,
) -> Result<(Compiled, SimOptions), HybridSpecError> {
    if !spec.t_end.is_finite() || spec.t_end <= 0.0 {
        return Err(invalid(
            &spec.name,
            "tEnd",
            "expected a positive finite time",
        ));
    }
    if !spec.max_step.is_finite() || spec.max_step <= 0.0 {
        return Err(invalid(
            &spec.name,
            "maxStep",
            "expected a positive finite step",
        ));
    }
    if !spec.zc_tol.is_finite() || spec.zc_tol <= 0.0 {
        return Err(invalid(
            &spec.name,
            "zcTol",
            "expected a positive finite tolerance",
        ));
    }

    let mut diagram = Diagram::new();
    let mut ids: HashMap<String, BlockHandle> = HashMap::new();
    for block in &spec.blocks {
        let id = block.id().to_string();
        if ids.contains_key(&id) {
            return Err(HybridSpecError::DuplicateBlock(id));
        }
        let handle = diagram.add(block_to_runtime(block)?);
        ids.insert(id, handle);
    }

    for wire in &spec.wires {
        let from = *ids
            .get(&wire.from)
            .ok_or_else(|| HybridSpecError::UnknownBlock(wire.from.clone()))?;
        let to = *ids
            .get(&wire.to)
            .ok_or_else(|| HybridSpecError::UnknownBlock(wire.to.clone()))?;
        diagram.connect((from, wire.from_port), (to, wire.to_port))?;
    }

    let opts = SimOptions {
        t_end: spec.t_end,
        max_step: spec.max_step,
        zc_tol: spec.zc_tol,
    };
    Ok((diagram.build()?, opts))
}

/// Generate a Rust runner for this hybrid graph.
pub fn generate_rust_code(spec: &HybridModelSpec) -> String {
    let spec_json = serde_json::to_string_pretty(spec).expect("HybridModelSpec serializes");
    let spec_lit = rust_raw_string_literal(&spec_json);
    let fn_name = rust_ident(
        &spec.authoring.codegen.rust.function_name,
        "run_generated_model",
    );
    format!(
        r#"//! Generated from {schema}. Do not hand-edit without updating the source model.

use des_engine::des::hybrid::spec::{{compile_hybrid_spec, HybridModelSpec}};
use des_engine::des::hybrid::simulate;
use des_engine::des::model::RunArtifact;
use serde_json::json;

pub fn {fn_name}() -> RunArtifact {{
    let spec: HybridModelSpec =
        serde_json::from_str({spec_lit}).expect("generated HybridModelSpec is valid JSON");
    let (compiled, opts) = compile_hybrid_spec(&spec).expect("generated HybridModelSpec compiles");
    let trace = simulate(&compiled, &opts);
    let frames = trace.to_jsonl_frames();
    let results = json!({{
        "kind": "hybrid",
        "model": spec.name,
        "events": trace.events,
        "columns": trace.columns,
        "samples": trace.times.len()
    }});
    RunArtifact::sim(
        "hybrid",
        "Generated Hybrid Graph",
        "Generated Rust runner for a des/hybrid-graph/v1 model.",
        frames,
        results,
        Vec::new(),
        "Generated hybrid graph run complete.",
    )
}}
"#,
        schema = HYBRID_GRAPH_SCHEMA,
        fn_name = fn_name,
        spec_lit = spec_lit
    )
}

/// Small starter diagram: `one -> integrator`, producing `x(t) = t`.
pub fn starter_hybrid_model_spec() -> HybridModelSpec {
    HybridModelSpec {
        schema: HYBRID_GRAPH_SCHEMA.to_string(),
        name: "integrator-ramp".to_string(),
        t_end: 2.0,
        max_step: 0.01,
        zc_tol: 1e-9,
        blocks: vec![
            HybridBlockSpec::Constant {
                id: "one".to_string(),
                value: vec![1.0],
            },
            HybridBlockSpec::Integrator {
                id: "x".to_string(),
                initial: vec![0.0],
            },
        ],
        wires: vec![HybridWireSpec {
            from: "one".to_string(),
            from_port: 0,
            to: "x".to_string(),
            to_port: 0,
        }],
        authoring: ModelAuthoringSpec::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::hybrid::simulate;

    #[test]
    fn starter_spec_round_trips_compiles_and_runs() {
        let spec = starter_hybrid_model_spec();
        let json = serde_json::to_string(&spec).unwrap();
        let decoded: HybridModelSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.schema, HYBRID_GRAPH_SCHEMA);
        let (compiled, opts) = compile_hybrid_spec(&decoded).unwrap();
        assert_eq!(compiled.block_count(), 2);
        let trace = simulate(&compiled, &opts);
        let (_t, x) = trace.series("x.p0").unwrap();
        assert!((x.last().copied().unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn compile_rejects_unknown_wire_reference() {
        let mut spec = starter_hybrid_model_spec();
        spec.wires[0].from = "missing".to_string();
        assert!(matches!(
            compile_hybrid_spec(&spec),
            Err(HybridSpecError::UnknownBlock(id)) if id == "missing"
        ));
    }

    #[test]
    fn compile_rejects_bad_state_space_dimensions() {
        let spec = HybridModelSpec {
            schema: HYBRID_GRAPH_SCHEMA.to_string(),
            name: "bad-ss".to_string(),
            t_end: 1.0,
            max_step: 0.1,
            zc_tol: 1e-9,
            blocks: vec![HybridBlockSpec::StateSpace {
                id: "plant".to_string(),
                a: vec![vec![1.0, 0.0]],
                b: vec![vec![1.0]],
                c: vec![vec![1.0]],
                d: vec![vec![0.0]],
                x0: None,
            }],
            wires: vec![],
            authoring: ModelAuthoringSpec::default(),
        };
        assert!(matches!(
            compile_hybrid_spec(&spec),
            Err(HybridSpecError::InvalidParam { param, .. }) if param == "a"
        ));
    }

    #[test]
    fn schema_is_generated_from_hybrid_spec_types() {
        let schema = hybrid_model_json_schema();
        assert_eq!(schema["title"], "HybridModelSpec");
        assert!(schema["properties"]["blocks"].is_object());
        assert!(schema["properties"]["authoring"].is_object());
    }

    #[test]
    fn rust_codegen_embeds_checked_hybrid_graph() {
        let code = generate_rust_code(&starter_hybrid_model_spec());
        assert!(code.contains("pub fn run_generated_model()"));
        assert!(code.contains("HybridModelSpec"));
        assert!(code.contains(HYBRID_GRAPH_SCHEMA));
    }
}
