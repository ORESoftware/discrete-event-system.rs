//! Serializable Modeling Studio surface: palette metadata, diagram specs, and
//! compilation into the existing flat [`StudioGraph`](super::graph::StudioGraph).
//!
//! This is the UI-facing entry point for a Simulink-like editor. The editor can render
//! [`studio_palette`] as a block palette + property inspector, persist a
//! [`StudioModelSpec`] as JSON, and call [`compile_model_spec`] before running.

use std::collections::HashMap;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::des::model::{
    authoring::ModelAuthoringSpec,
    codegen::{rust_ident, rust_raw_string_literal},
};

use super::cell::{
    Affine, Gain, Integrator, Probe, Queue, RuntimeCell, RuntimeOp, Saturation, Source, SourceKind,
    Sum, TransportDelay,
};
use super::graph::{CompiledStudio, NodeRole, StudioError, StudioGraph, VisualNode};

pub const STUDIO_GRAPH_SCHEMA: &str = "des/studio-graph/v1";

fn studio_graph_schema() -> String {
    STUDIO_GRAPH_SCHEMA.to_string()
}

/// JSON Schema for saved Studio graph documents.
pub fn studio_model_json_schema() -> Value {
    serde_json::to_value(schema_for!(StudioModelSpec)).expect("StudioModelSpec schema serializes")
}

/// Stable block kinds exposed to the graphical editor palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum StudioBlockKind {
    Constant,
    Step,
    Ramp,
    Sine,
    Gain,
    Sum,
    Saturation,
    Affine,
    Integrator,
    Queue,
    TransportDelay,
    Sink,
}

impl StudioBlockKind {
    pub fn as_str(self) -> &'static str {
        match self {
            StudioBlockKind::Constant => "constant",
            StudioBlockKind::Step => "step",
            StudioBlockKind::Ramp => "ramp",
            StudioBlockKind::Sine => "sine",
            StudioBlockKind::Gain => "gain",
            StudioBlockKind::Sum => "sum",
            StudioBlockKind::Saturation => "saturation",
            StudioBlockKind::Affine => "affine",
            StudioBlockKind::Integrator => "integrator",
            StudioBlockKind::Queue => "queue",
            StudioBlockKind::TransportDelay => "transport-delay",
            StudioBlockKind::Sink => "sink",
        }
    }

    pub fn role(self) -> NodeRole {
        match self {
            StudioBlockKind::Constant
            | StudioBlockKind::Step
            | StudioBlockKind::Ramp
            | StudioBlockKind::Sine => NodeRole::Source,
            StudioBlockKind::Sink => NodeRole::Sink,
            _ => NodeRole::Transform,
        }
    }

    /// VisualBlock renderer kind used by the animation/icon layer.
    pub fn visual_kind(self) -> &'static str {
        match self {
            StudioBlockKind::Constant => "constant-source",
            StudioBlockKind::Step | StudioBlockKind::Ramp | StudioBlockKind::Sine => {
                "function-source"
            }
            StudioBlockKind::Gain => "gain",
            StudioBlockKind::Sum => "sum",
            StudioBlockKind::Saturation => "saturation",
            StudioBlockKind::Affine => "gain",
            StudioBlockKind::Integrator => "integrator",
            StudioBlockKind::Queue => "station",
            StudioBlockKind::TransportDelay => "first-order-filter",
            StudioBlockKind::Sink => "sink",
        }
    }
}

/// Parameter control type for a generated inspector panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum PaletteParamKind {
    Number,
    Integer,
    NumberArray,
}

/// One editable parameter in the block inspector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PaletteParam {
    pub name: String,
    pub label: String,
    pub kind: PaletteParamKind,
    pub default_value: Value,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
}

/// Palette metadata consumed directly by a UI.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PaletteItem {
    pub kind: StudioBlockKind,
    pub label: String,
    pub category: String,
    pub description: String,
    pub inputs: usize,
    pub outputs: usize,
    pub stateful: bool,
    pub visual_kind: String,
    pub params: Vec<PaletteParam>,
}

fn p_num(name: &str, label: &str, default_value: f64) -> PaletteParam {
    PaletteParam {
        name: name.to_string(),
        label: label.to_string(),
        kind: PaletteParamKind::Number,
        default_value: Value::from(default_value),
        min: None,
        max: None,
        step: Some(0.1),
    }
}

fn p_int(name: &str, label: &str, default_value: u64, min: f64) -> PaletteParam {
    PaletteParam {
        name: name.to_string(),
        label: label.to_string(),
        kind: PaletteParamKind::Integer,
        default_value: Value::from(default_value),
        min: Some(min),
        max: None,
        step: Some(1.0),
    }
}

fn p_arr(name: &str, label: &str, default_value: Vec<f64>) -> PaletteParam {
    PaletteParam {
        name: name.to_string(),
        label: label.to_string(),
        kind: PaletteParamKind::NumberArray,
        default_value: Value::Array(default_value.into_iter().map(Value::from).collect()),
        min: None,
        max: None,
        step: None,
    }
}

fn item(
    kind: StudioBlockKind,
    label: &str,
    category: &str,
    description: &str,
    inputs: usize,
    outputs: usize,
    stateful: bool,
    params: Vec<PaletteParam>,
) -> PaletteItem {
    PaletteItem {
        kind,
        label: label.to_string(),
        category: category.to_string(),
        description: description.to_string(),
        inputs,
        outputs,
        stateful,
        visual_kind: kind.visual_kind().to_string(),
        params,
    }
}

/// Built-in block palette for the first UI editor.
pub fn studio_palette() -> Vec<PaletteItem> {
    vec![
        item(
            StudioBlockKind::Constant,
            "Constant",
            "Sources",
            "Emit a fixed scalar value.",
            0,
            1,
            false,
            vec![p_num("value", "Value", 1.0)],
        ),
        item(
            StudioBlockKind::Step,
            "Step",
            "Sources",
            "Switch from one scalar value to another at time t0.",
            0,
            1,
            false,
            vec![
                p_num("t0", "Step time", 1.0),
                p_num("before", "Before", 0.0),
                p_num("after", "After", 1.0),
            ],
        ),
        item(
            StudioBlockKind::Ramp,
            "Ramp",
            "Sources",
            "Emit slope * t + intercept.",
            0,
            1,
            false,
            vec![
                p_num("slope", "Slope", 1.0),
                p_num("intercept", "Intercept", 0.0),
            ],
        ),
        item(
            StudioBlockKind::Sine,
            "Sine",
            "Sources",
            "Emit amp * sin(2*pi*freq*t) + bias.",
            0,
            1,
            false,
            vec![
                p_num("amp", "Amplitude", 1.0),
                p_num("freq", "Frequency", 1.0),
                p_num("bias", "Bias", 0.0),
            ],
        ),
        item(
            StudioBlockKind::Gain,
            "Gain",
            "Math",
            "Scale one input by k.",
            1,
            1,
            false,
            vec![p_num("k", "Gain", 1.0)],
        ),
        item(
            StudioBlockKind::Sum,
            "Sum",
            "Math",
            "Weighted sum over N input ports.",
            2,
            1,
            false,
            vec![p_arr("weights", "Weights", vec![1.0, 1.0])],
        ),
        item(
            StudioBlockKind::Saturation,
            "Saturation",
            "Math",
            "Clamp one input into [lo, hi].",
            1,
            1,
            false,
            vec![
                p_num("lo", "Lower bound", -1.0),
                p_num("hi", "Upper bound", 1.0),
            ],
        ),
        item(
            StudioBlockKind::Affine,
            "Affine",
            "Math",
            "Apply m*x + b.",
            1,
            1,
            false,
            vec![p_num("m", "Multiplier", 1.0), p_num("b", "Bias", 0.0)],
        ),
        item(
            StudioBlockKind::Integrator,
            "Integrator",
            "Continuous/Discrete",
            "Forward-Euler scalar integrator.",
            1,
            1,
            true,
            vec![p_num("initial", "Initial state", 0.0)],
        ),
        item(
            StudioBlockKind::Queue,
            "Queue",
            "Discrete Events",
            "Single-server queue with a per-tick service rate.",
            1,
            1,
            true,
            vec![p_num("serviceRate", "Service rate", 1.0)],
        ),
        item(
            StudioBlockKind::TransportDelay,
            "Transport Delay",
            "Discrete Events",
            "Emit the input received N ticks ago.",
            1,
            1,
            true,
            vec![p_int("delay", "Delay ticks", 1, 1.0)],
        ),
        item(
            StudioBlockKind::Sink,
            "Sink",
            "Sinks",
            "Probe and record one input signal.",
            1,
            0,
            false,
            vec![],
        ),
    ]
}

/// One block in a saved studio diagram.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StudioBlockSpec {
    pub id: String,
    pub kind: StudioBlockKind,
    pub label: Option<String>,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
}

/// One wire in a saved studio diagram.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StudioWireSpec {
    pub from: String,
    #[serde(default)]
    pub from_port: usize,
    pub to: String,
    #[serde(default)]
    pub to_port: usize,
}

/// A tunable scalar parameter, mirroring OpenMDAO-style design variables.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StudioDesignVariableSpec {
    pub name: String,
    pub block: String,
    pub param: String,
    pub lower: f64,
    pub upper: f64,
    #[serde(default = "default_sweep_samples")]
    pub samples: usize,
    pub units: Option<String>,
}

/// Optimization / exploration direction for a recorded signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum StudioObjectiveSense {
    Minimize,
    Maximize,
    Track,
}

fn default_objective_sense() -> StudioObjectiveSense {
    StudioObjectiveSense::Minimize
}

/// A final signal metric to optimize, record, or track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StudioObjectiveSpec {
    pub name: String,
    pub block: String,
    #[serde(default)]
    pub port: usize,
    #[serde(default = "default_objective_sense")]
    pub sense: StudioObjectiveSense,
    pub target: Option<f64>,
}

/// A bound check over a block's final recorded signal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StudioConstraintSpec {
    pub name: String,
    pub block: String,
    #[serde(default)]
    pub port: usize,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

/// Top-level UI model document. This is the format to save/load from the
/// browser editor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StudioModelSpec {
    #[serde(rename = "$schema", default = "studio_graph_schema")]
    pub schema: String,
    pub name: String,
    #[serde(default = "default_dt")]
    pub dt: f64,
    #[serde(default = "default_steps")]
    pub steps: usize,
    pub blocks: Vec<StudioBlockSpec>,
    #[serde(default)]
    pub wires: Vec<StudioWireSpec>,
    #[serde(default)]
    pub design_variables: Vec<StudioDesignVariableSpec>,
    #[serde(default)]
    pub objectives: Vec<StudioObjectiveSpec>,
    #[serde(default)]
    pub constraints: Vec<StudioConstraintSpec>,
    #[serde(default)]
    pub authoring: ModelAuthoringSpec,
}

fn default_dt() -> f64 {
    0.1
}

fn default_steps() -> usize {
    100
}

fn default_sweep_samples() -> usize {
    9
}

/// User-facing spec/compile errors.
#[derive(Clone, Debug, PartialEq)]
pub enum StudioSpecError {
    InvalidParam {
        block: String,
        param: String,
        message: String,
    },
    UnknownBlock(String),
    Graph(StudioError),
}

impl std::fmt::Display for StudioSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StudioSpecError::InvalidParam {
                block,
                param,
                message,
            } => {
                write!(
                    f,
                    "block `{block}` parameter `{param}` is invalid: {message}"
                )
            }
            StudioSpecError::UnknownBlock(id) => write!(f, "wire references unknown block `{id}`"),
            StudioSpecError::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StudioSpecError {}

impl From<StudioError> for StudioSpecError {
    fn from(value: StudioError) -> Self {
        StudioSpecError::Graph(value)
    }
}

fn param_f64(block: &StudioBlockSpec, name: &str, default: f64) -> Result<f64, StudioSpecError> {
    match block.params.get(name) {
        None => Ok(default),
        Some(Value::Number(n)) => {
            n.as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| StudioSpecError::InvalidParam {
                    block: block.id.clone(),
                    param: name.to_string(),
                    message: "expected a finite number".to_string(),
                })
        }
        Some(_) => Err(StudioSpecError::InvalidParam {
            block: block.id.clone(),
            param: name.to_string(),
            message: "expected a number".to_string(),
        }),
    }
}

fn param_usize(
    block: &StudioBlockSpec,
    name: &str,
    default: usize,
) -> Result<usize, StudioSpecError> {
    match block.params.get(name) {
        None => Ok(default),
        Some(Value::Number(n)) => n
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or_else(|| StudioSpecError::InvalidParam {
                block: block.id.clone(),
                param: name.to_string(),
                message: "expected a positive integer".to_string(),
            }),
        Some(_) => Err(StudioSpecError::InvalidParam {
            block: block.id.clone(),
            param: name.to_string(),
            message: "expected an integer".to_string(),
        }),
    }
}

fn param_vec(
    block: &StudioBlockSpec,
    name: &str,
    default: &[f64],
) -> Result<Vec<f64>, StudioSpecError> {
    match block.params.get(name) {
        None => Ok(default.to_vec()),
        Some(Value::Array(xs)) => {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                let Some(v) = x.as_f64().filter(|v| v.is_finite()) else {
                    return Err(StudioSpecError::InvalidParam {
                        block: block.id.clone(),
                        param: name.to_string(),
                        message: "expected an array of finite numbers".to_string(),
                    });
                };
                out.push(v);
            }
            if out.is_empty() {
                return Err(StudioSpecError::InvalidParam {
                    block: block.id.clone(),
                    param: name.to_string(),
                    message: "expected at least one weight".to_string(),
                });
            }
            Ok(out)
        }
        Some(_) => Err(StudioSpecError::InvalidParam {
            block: block.id.clone(),
            param: name.to_string(),
            message: "expected an array".to_string(),
        }),
    }
}

fn runtime_cell(block: &StudioBlockSpec) -> Result<RuntimeCell, StudioSpecError> {
    let op: Box<dyn RuntimeOp> = match block.kind {
        StudioBlockKind::Constant => Box::new(Source::new(
            "constant",
            SourceKind::Const(param_f64(block, "value", 1.0)?),
        )),
        StudioBlockKind::Step => Box::new(Source::new(
            "step",
            SourceKind::Step {
                t0: param_f64(block, "t0", 1.0)?,
                before: param_f64(block, "before", 0.0)?,
                after: param_f64(block, "after", 1.0)?,
            },
        )),
        StudioBlockKind::Ramp => Box::new(Source::new(
            "ramp",
            SourceKind::Ramp {
                slope: param_f64(block, "slope", 1.0)?,
                intercept: param_f64(block, "intercept", 0.0)?,
            },
        )),
        StudioBlockKind::Sine => Box::new(Source::new(
            "sine",
            SourceKind::Sine {
                amp: param_f64(block, "amp", 1.0)?,
                freq: param_f64(block, "freq", 1.0)?,
                bias: param_f64(block, "bias", 0.0)?,
            },
        )),
        StudioBlockKind::Gain => Box::new(Gain::new("gain", param_f64(block, "k", 1.0)?)),
        StudioBlockKind::Sum => {
            Box::new(Sum::new("sum", param_vec(block, "weights", &[1.0, 1.0])?))
        }
        StudioBlockKind::Saturation => Box::new(Saturation::new(
            "saturation",
            param_f64(block, "lo", -1.0)?,
            param_f64(block, "hi", 1.0)?,
        )),
        StudioBlockKind::Affine => Box::new(Affine::new(
            "affine",
            param_f64(block, "m", 1.0)?,
            param_f64(block, "b", 0.0)?,
        )),
        StudioBlockKind::Integrator => Box::new(Integrator::new(
            "integrator",
            param_f64(block, "initial", 0.0)?,
        )),
        StudioBlockKind::Queue => {
            Box::new(Queue::new("queue", param_f64(block, "serviceRate", 1.0)?))
        }
        StudioBlockKind::TransportDelay => Box::new(TransportDelay::new(
            "transport-delay",
            param_usize(block, "delay", 1)?,
        )),
        StudioBlockKind::Sink => Box::new(Probe::new("probe")),
    };
    Ok(RuntimeCell::single(op))
}

/// Port/state metadata for a block after its parameters are applied.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StudioBlockIo {
    pub inputs: usize,
    pub outputs: usize,
    pub stateful: bool,
    pub elements: Vec<String>,
}

/// Resolve a block's runtime-facing I/O without compiling the full graph.
pub fn studio_block_io(block: &StudioBlockSpec) -> Result<StudioBlockIo, StudioSpecError> {
    let cell = runtime_cell(block)?;
    Ok(StudioBlockIo {
        inputs: cell.n_in(),
        outputs: cell.n_out(),
        stateful: cell.has_state(),
        elements: cell.element_names(),
    })
}

/// Compile a saved UI model into the runnable Studio graph.
pub fn compile_model_spec(spec: &StudioModelSpec) -> Result<CompiledStudio, StudioSpecError> {
    if !spec.dt.is_finite() || spec.dt <= 0.0 {
        return Err(StudioSpecError::InvalidParam {
            block: spec.name.clone(),
            param: "dt".to_string(),
            message: "expected a positive finite timestep".to_string(),
        });
    }
    if spec.steps == 0 {
        return Err(StudioSpecError::InvalidParam {
            block: spec.name.clone(),
            param: "steps".to_string(),
            message: "expected at least one step".to_string(),
        });
    }

    let mut graph = StudioGraph::new();
    let mut ids: HashMap<String, usize> = HashMap::new();

    for block in &spec.blocks {
        let label = block.label.as_deref().unwrap_or(&block.id);
        let node = VisualNode::new(&block.id, block.kind.role(), runtime_cell(block)?)
            .with_label(label)
            .with_kind(block.kind.as_str())
            .at(block.x, block.y);
        let idx = graph.add(node)?;
        ids.insert(block.id.clone(), idx);
    }

    for wire in &spec.wires {
        let from = *ids
            .get(&wire.from)
            .ok_or_else(|| StudioSpecError::UnknownBlock(wire.from.clone()))?;
        let to = *ids
            .get(&wire.to)
            .ok_or_else(|| StudioSpecError::UnknownBlock(wire.to.clone()))?;
        graph.connect(from, wire.from_port, to, wire.to_port)?;
    }

    graph.build().map_err(Into::into)
}

/// Generate a small Rust runner for this Studio graph.
///
/// The first generator intentionally targets Rust, not C/C++: it embeds the
/// checked JSON graph, deserializes it into [`StudioModelSpec`], compiles it
/// through the same validator the UI uses, and returns a uniform run artifact.
pub fn generate_rust_code(spec: &StudioModelSpec) -> String {
    let spec_json = serde_json::to_string_pretty(spec).expect("StudioModelSpec serializes");
    let spec_lit = rust_raw_string_literal(&spec_json);
    let fn_name = rust_ident(
        &spec.authoring.codegen.rust.function_name,
        "run_generated_model",
    );
    format!(
        r#"//! Generated from {schema}. Do not hand-edit without updating the source model.

use des_engine::des::model::RunArtifact;
use des_engine::des::studio::demos::blocks_doc;
use des_engine::des::studio::{{compile_model_spec, run, StudioModelSpec}};

pub fn {fn_name}() -> RunArtifact {{
    let spec: StudioModelSpec =
        serde_json::from_str({spec_lit}).expect("generated StudioModelSpec is valid JSON");
    let mut compiled = compile_model_spec(&spec).expect("generated StudioModelSpec compiles");
    let blocks = blocks_doc(&compiled);
    let run_out = run(&mut compiled, spec.steps, spec.dt);
    run_out.to_artifact(
        "studio",
        &spec.name,
        "Generated Rust runner for a des/studio-graph/v1 model.",
        blocks,
    )
}}
"#,
        schema = STUDIO_GRAPH_SCHEMA,
        fn_name = fn_name,
        spec_lit = spec_lit
    )
}

/// Small starter diagram for a first-load editor canvas.
pub fn starter_model_spec() -> StudioModelSpec {
    StudioModelSpec {
        schema: STUDIO_GRAPH_SCHEMA.to_string(),
        name: "ramp-gain-sink".to_string(),
        dt: 0.1,
        steps: 80,
        blocks: vec![
            StudioBlockSpec {
                id: "input".to_string(),
                kind: StudioBlockKind::Ramp,
                label: Some("Input".to_string()),
                params: Map::from_iter([
                    ("slope".to_string(), Value::from(1.0)),
                    ("intercept".to_string(), Value::from(0.0)),
                ]),
                x: 40.0,
                y: 120.0,
            },
            StudioBlockSpec {
                id: "gain".to_string(),
                kind: StudioBlockKind::Gain,
                label: Some("Gain".to_string()),
                params: Map::from_iter([("k".to_string(), Value::from(0.5))]),
                x: 250.0,
                y: 120.0,
            },
            StudioBlockSpec {
                id: "out".to_string(),
                kind: StudioBlockKind::Sink,
                label: Some("Output".to_string()),
                params: Map::new(),
                x: 460.0,
                y: 120.0,
            },
        ],
        wires: vec![
            StudioWireSpec {
                from: "input".to_string(),
                from_port: 0,
                to: "gain".to_string(),
                to_port: 0,
            },
            StudioWireSpec {
                from: "gain".to_string(),
                from_port: 0,
                to: "out".to_string(),
                to_port: 0,
            },
        ],
        design_variables: vec![StudioDesignVariableSpec {
            name: "gain.k".to_string(),
            block: "gain".to_string(),
            param: "k".to_string(),
            lower: 0.0,
            upper: 2.0,
            samples: 9,
            units: None,
        }],
        objectives: vec![StudioObjectiveSpec {
            name: "final output".to_string(),
            block: "out".to_string(),
            port: 0,
            sense: StudioObjectiveSense::Track,
            target: Some(4.0),
        }],
        constraints: vec![StudioConstraintSpec {
            name: "output ceiling".to_string(),
            block: "out".to_string(),
            port: 0,
            lower: None,
            upper: Some(8.0),
        }],
        authoring: ModelAuthoringSpec::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::studio::run::run;

    #[test]
    fn palette_exposes_editor_metadata() {
        let palette = studio_palette();
        assert!(palette.iter().any(|p| {
            p.kind == StudioBlockKind::Integrator && p.stateful && p.visual_kind == "integrator"
        }));
        assert!(palette
            .iter()
            .any(|p| p.kind == StudioBlockKind::Sum && p.inputs == 2));
    }

    #[test]
    fn starter_spec_round_trips_through_json_and_runs() {
        let spec = starter_model_spec();
        let json = serde_json::to_string(&spec).expect("serialize studio model");
        let decoded: StudioModelSpec =
            serde_json::from_str(&json).expect("deserialize studio model");
        assert_eq!(decoded.schema, STUDIO_GRAPH_SCHEMA);
        let mut compiled = compile_model_spec(&decoded).expect("compile studio model");
        assert_eq!(compiled.node_count(), 3);
        assert_eq!(compiled.nodes()[1].kind, "gain");

        let out = run(&mut compiled, decoded.steps, decoded.dt);
        assert!((out.final_value("out").unwrap() - 3.95).abs() < 1e-9);
    }

    #[test]
    fn compile_rejects_bad_wire_reference() {
        let mut spec = starter_model_spec();
        spec.wires[0].from = "missing".to_string();
        assert!(matches!(
            compile_model_spec(&spec),
            Err(StudioSpecError::UnknownBlock(id)) if id == "missing"
        ));
    }

    #[test]
    fn compile_rejects_bad_sum_weights() {
        let mut params = Map::new();
        params.insert("weights".to_string(), Value::Array(vec![]));
        let spec = StudioModelSpec {
            schema: STUDIO_GRAPH_SCHEMA.to_string(),
            name: "bad".to_string(),
            dt: 0.1,
            steps: 1,
            blocks: vec![StudioBlockSpec {
                id: "sum".to_string(),
                kind: StudioBlockKind::Sum,
                label: None,
                params,
                x: 0.0,
                y: 0.0,
            }],
            wires: vec![],
            design_variables: vec![],
            objectives: vec![],
            constraints: vec![],
            authoring: ModelAuthoringSpec::default(),
        };
        assert!(matches!(
            compile_model_spec(&spec),
            Err(StudioSpecError::InvalidParam { param, .. }) if param == "weights"
        ));
    }

    #[test]
    fn compile_rejects_non_positive_dt() {
        let mut spec = starter_model_spec();
        spec.dt = 0.0;
        assert!(matches!(
            compile_model_spec(&spec),
            Err(StudioSpecError::InvalidParam { param, .. }) if param == "dt"
        ));
    }

    #[test]
    fn block_io_reflects_parameterized_sum_width() {
        let mut block = StudioBlockSpec {
            id: "sum".to_string(),
            kind: StudioBlockKind::Sum,
            label: None,
            params: Map::new(),
            x: 0.0,
            y: 0.0,
        };
        block.params.insert(
            "weights".to_string(),
            Value::Array(vec![Value::from(1.0), Value::from(-1.0), Value::from(0.5)]),
        );
        let io = studio_block_io(&block).unwrap();
        assert_eq!(io.inputs, 3);
        assert_eq!(io.outputs, 1);
        assert!(!io.stateful);
    }

    #[test]
    fn schema_is_generated_from_studio_spec_types() {
        let schema = studio_model_json_schema();
        assert_eq!(schema["title"], "StudioModelSpec");
        assert!(schema["properties"]["blocks"].is_object());
        assert!(schema["properties"]["authoring"].is_object());
    }

    #[test]
    fn rust_codegen_embeds_checked_studio_graph() {
        let code = generate_rust_code(&starter_model_spec());
        assert!(code.contains("pub fn run_generated_model()"));
        assert!(code.contains("StudioModelSpec"));
        assert!(code.contains(STUDIO_GRAPH_SCHEMA));
    }
}
