//! Serializable Modeling Studio surface: palette metadata, diagram specs, and
//! compilation into the existing flat [`StudioGraph`](super::graph::StudioGraph).
//!
//! This is the UI-facing entry point for a Simulink-like editor. The editor can render
//! [`studio_palette`] as a block palette + property inspector, persist a
//! [`StudioModelSpec`] as JSON, and call [`compile_model_spec`] before running.

use std::collections::{HashMap, HashSet};

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::des::model::{
    authoring::ModelAuthoringSpec,
    codegen::{rust_ident, rust_raw_string_literal},
};

use super::cell::{
    Affine, Composite, Gain, Integrator, Probe, Queue, RuntimeCell, RuntimeOp, Saturation, Source,
    SourceKind, Sum, TransportDelay,
};
use super::demos::{blocks_doc, StudioDemo};
use super::graph::{CompiledStudio, NodeRole, StudioError, StudioGraph, VisualNode};

pub const STUDIO_GRAPH_SCHEMA: &str = "des/studio-graph/v1";
pub const STUDIO_SPEC_SCHEMA: &str = "des/studio/v1";
pub const MAX_MODEL_BLOCKS: usize = 1_024;
pub const MAX_MODEL_WIRES: usize = 4_096;
pub const MAX_RUN_STEPS: usize = 100_000;
pub const MAX_BLOCK_ID_LEN: usize = 96;
pub const MAX_LABEL_LEN: usize = 160;
pub const MAX_PARAM_VECTOR_LEN: usize = 256;
pub const MAX_RUNTIME_CELL_OPS: usize = 128;
pub const MAX_RUNTIME_NESTING: usize = 16;
pub const MAX_SWEEP_SAMPLES: usize = 10_000;

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

fn block_param_kind(kind: StudioBlockKind, param: &str) -> Option<PaletteParamKind> {
    studio_palette()
        .into_iter()
        .find(|item| item.kind == kind)
        .and_then(|item| {
            item.params
                .into_iter()
                .find(|p| p.name == param)
                .map(|p| p.kind)
        })
}

fn allowed_param_names(kind: StudioBlockKind) -> HashSet<String> {
    studio_palette()
        .into_iter()
        .find(|item| item.kind == kind)
        .map(|item| item.params.into_iter().map(|p| p.name).collect())
        .unwrap_or_default()
}

fn validate_user_text(
    value: &str,
    field: &str,
    max_len: usize,
    allow_empty: bool,
) -> Result<(), StudioSpecError> {
    if !allow_empty && value.trim().is_empty() {
        return Err(StudioSpecError::new(format!("{field} must be non-empty")));
    }
    if value.len() > max_len {
        return Err(StudioSpecError::new(format!(
            "{field} is too long; limit is {max_len} bytes"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(StudioSpecError::new(format!(
            "{field} must not contain control characters"
        )));
    }
    Ok(())
}

fn invalid_param(
    block: &StudioBlockSpec,
    param: &str,
    message: impl Into<String>,
) -> StudioSpecError {
    StudioSpecError::InvalidParam {
        block: block.id.clone(),
        param: param.to_string(),
        message: message.into(),
    }
}

/// User-facing spec/compile errors.
#[derive(Clone, Debug, PartialEq)]
pub enum StudioSpecError {
    Parse(String),
    InvalidParam {
        block: String,
        param: String,
        message: String,
    },
    UnknownBlock(String),
    Graph(StudioError),
}

impl StudioSpecError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        StudioSpecError::Parse(message.into())
    }
}

impl std::fmt::Display for StudioSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StudioSpecError::Parse(message) => write!(f, "{message}"),
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
        Some(Value::Number(n)) => n
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| invalid_param(block, name, "expected a finite number")),
        Some(_) => Err(invalid_param(block, name, "expected a number")),
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
            .ok_or_else(|| invalid_param(block, name, "expected a positive integer")),
        Some(_) => Err(invalid_param(block, name, "expected an integer")),
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
                    return Err(invalid_param(
                        block,
                        name,
                        "expected an array of finite numbers",
                    ));
                };
                out.push(v);
            }
            if out.is_empty() {
                return Err(invalid_param(block, name, "expected at least one weight"));
            }
            if out.len() > MAX_PARAM_VECTOR_LEN {
                return Err(invalid_param(
                    block,
                    name,
                    format!("expected at most {MAX_PARAM_VECTOR_LEN} entries"),
                ));
            }
            Ok(out)
        }
        Some(_) => Err(invalid_param(block, name, "expected an array")),
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
        StudioBlockKind::Saturation => {
            let lo = param_f64(block, "lo", -1.0)?;
            let hi = param_f64(block, "hi", 1.0)?;
            if lo > hi {
                return Err(invalid_param(
                    block,
                    "lo",
                    "expected lower bound to be <= upper bound",
                ));
            }
            Box::new(Saturation::new("saturation", lo, hi))
        }
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
            let service_rate = param_f64(block, "serviceRate", 1.0)?;
            if service_rate < 0.0 {
                return Err(invalid_param(
                    block,
                    "serviceRate",
                    "expected a non-negative service rate",
                ));
            }
            Box::new(Queue::new("queue", service_rate))
        }
        StudioBlockKind::TransportDelay => Box::new(TransportDelay::new(
            "transport-delay",
            param_usize(block, "delay", 1)?,
        )),
        StudioBlockKind::Sink => Box::new(Probe::new("probe")),
    };
    Ok(RuntimeCell::single(op))
}

fn validate_scalar_design_param(
    block: &StudioBlockSpec,
    param: &str,
) -> Result<(), StudioSpecError> {
    match block_param_kind(block.kind, param) {
        Some(PaletteParamKind::Number) => {}
        Some(PaletteParamKind::Integer) => {
            return Err(StudioSpecError::InvalidParam {
                block: block.id.clone(),
                param: param.to_string(),
                message: "integer parameters are not sweepable by the scalar driver yet"
                    .to_string(),
            });
        }
        Some(PaletteParamKind::NumberArray) => {
            return Err(StudioSpecError::InvalidParam {
                block: block.id.clone(),
                param: param.to_string(),
                message: "expected a scalar number parameter".to_string(),
            });
        }
        None => {
            return Err(StudioSpecError::InvalidParam {
                block: block.id.clone(),
                param: param.to_string(),
                message: "unknown parameter for this block kind".to_string(),
            });
        }
    }

    if let Some(value) = block.params.get(param) {
        value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| StudioSpecError::InvalidParam {
                block: block.id.clone(),
                param: param.to_string(),
                message: "expected an existing finite number before sweeping".to_string(),
            })?;
    }
    Ok(())
}

fn validate_model_metadata_pre_graph(spec: &StudioModelSpec) -> Result<(), StudioSpecError> {
    if spec.schema != STUDIO_GRAPH_SCHEMA {
        return Err(StudioSpecError::new(format!(
            "unsupported studio schema `{}`; expected `{STUDIO_GRAPH_SCHEMA}`",
            spec.schema
        )));
    }
    validate_user_text(&spec.name, "model name", MAX_LABEL_LEN, false)?;
    if spec.steps == 0 || spec.steps > MAX_RUN_STEPS {
        return Err(StudioSpecError::InvalidParam {
            block: spec.name.clone(),
            param: "steps".to_string(),
            message: format!("expected steps in 1..={MAX_RUN_STEPS}"),
        });
    }
    if spec.blocks.is_empty() {
        return Err(StudioSpecError::new(
            "studio model requires at least one block",
        ));
    }
    if spec.blocks.len() > MAX_MODEL_BLOCKS {
        return Err(StudioSpecError::new(format!(
            "studio model is limited to {MAX_MODEL_BLOCKS} blocks"
        )));
    }
    if spec.wires.len() > MAX_MODEL_WIRES {
        return Err(StudioSpecError::new(format!(
            "studio model is limited to {MAX_MODEL_WIRES} wires"
        )));
    }

    let mut block_ids = HashSet::new();
    for block in &spec.blocks {
        validate_user_text(&block.id, "block id", MAX_BLOCK_ID_LEN, false)?;
        if !block_ids.insert(block.id.as_str()) {
            return Err(StudioSpecError::new(format!(
                "duplicate block id `{}`",
                block.id
            )));
        }
        if let Some(label) = &block.label {
            validate_user_text(label, "block label", MAX_LABEL_LEN, true)?;
        }
        if !block.x.is_finite() || !block.y.is_finite() {
            return Err(StudioSpecError::InvalidParam {
                block: block.id.clone(),
                param: "position".to_string(),
                message: "expected finite x/y coordinates".to_string(),
            });
        }
        let allowed = allowed_param_names(block.kind);
        for param in block.params.keys() {
            validate_user_text(param, "parameter name", MAX_BLOCK_ID_LEN, false)?;
            if !allowed.contains(param) {
                return Err(StudioSpecError::InvalidParam {
                    block: block.id.clone(),
                    param: param.clone(),
                    message: "unknown parameter for this block kind".to_string(),
                });
            }
        }
    }

    let mut design_names = HashSet::new();
    for dv in &spec.design_variables {
        if dv.name.trim().is_empty() {
            return Err(StudioSpecError::new(
                "design variable names must be non-empty",
            ));
        }
        if !design_names.insert(dv.name.as_str()) {
            return Err(StudioSpecError::new(format!(
                "duplicate design variable `{}`",
                dv.name
            )));
        }
        if dv.block.trim().is_empty() || dv.param.trim().is_empty() {
            return Err(StudioSpecError::new(format!(
                "design variable `{}` requires block and param",
                dv.name
            )));
        }
        let block = spec
            .blocks
            .iter()
            .find(|block| block.id == dv.block)
            .ok_or_else(|| {
                StudioSpecError::new(format!(
                    "design variable `{}` references unknown block `{}`",
                    dv.name, dv.block
                ))
            })?;
        validate_scalar_design_param(block, &dv.param)?;
        if !dv.lower.is_finite() || !dv.upper.is_finite() {
            return Err(StudioSpecError::new(format!(
                "design variable `{}` bounds must be finite",
                dv.name
            )));
        }
        if dv.lower > dv.upper {
            return Err(StudioSpecError::new(format!(
                "design variable `{}` lower bound exceeds upper bound",
                dv.name
            )));
        }
        if dv.samples == 0 || dv.samples > MAX_SWEEP_SAMPLES {
            return Err(StudioSpecError::new(format!(
                "design variable `{}` samples must be in 1..={}",
                dv.name, MAX_SWEEP_SAMPLES
            )));
        }
    }

    let mut objective_names = HashSet::new();
    for objective in &spec.objectives {
        if objective.name.trim().is_empty() {
            return Err(StudioSpecError::new("objective names must be non-empty"));
        }
        if !objective_names.insert(objective.name.as_str()) {
            return Err(StudioSpecError::new(format!(
                "duplicate objective `{}`",
                objective.name
            )));
        }
        if !block_ids.contains(objective.block.as_str()) {
            return Err(StudioSpecError::new(format!(
                "objective `{}` references unknown block `{}`",
                objective.name, objective.block
            )));
        }
        if objective.port != 0 {
            return Err(StudioSpecError::new(format!(
                "objective `{}` uses unsupported port {}; only primary port 0 is recorded",
                objective.name, objective.port
            )));
        }
        if objective.target.is_some_and(|target| !target.is_finite()) {
            return Err(StudioSpecError::new(format!(
                "objective `{}` target must be finite",
                objective.name
            )));
        }
    }

    let mut constraint_names = HashSet::new();
    for constraint in &spec.constraints {
        if constraint.name.trim().is_empty() {
            return Err(StudioSpecError::new("constraint names must be non-empty"));
        }
        if !constraint_names.insert(constraint.name.as_str()) {
            return Err(StudioSpecError::new(format!(
                "duplicate constraint `{}`",
                constraint.name
            )));
        }
        if !block_ids.contains(constraint.block.as_str()) {
            return Err(StudioSpecError::new(format!(
                "constraint `{}` references unknown block `{}`",
                constraint.name, constraint.block
            )));
        }
        if constraint.port != 0 {
            return Err(StudioSpecError::new(format!(
                "constraint `{}` uses unsupported port {}; only primary port 0 is recorded",
                constraint.name, constraint.port
            )));
        }
        if constraint.lower.is_none() && constraint.upper.is_none() {
            return Err(StudioSpecError::new(format!(
                "constraint `{}` requires at least one finite bound",
                constraint.name
            )));
        }
        if constraint.lower.is_some_and(|lower| !lower.is_finite())
            || constraint.upper.is_some_and(|upper| !upper.is_finite())
        {
            return Err(StudioSpecError::new(format!(
                "constraint `{}` bounds must be finite",
                constraint.name
            )));
        }
        if let (Some(lower), Some(upper)) = (constraint.lower, constraint.upper) {
            if lower > upper {
                return Err(StudioSpecError::new(format!(
                    "constraint `{}` lower bound exceeds upper bound",
                    constraint.name
                )));
            }
        }
    }

    Ok(())
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
    validate_model_metadata_pre_graph(spec)?;

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

/// A compact, runnable starter spec: a control-style error shaper with a nested
/// runtime cell inside one visual block.
pub fn example_spec() -> Value {
    json!({
        "$schema": STUDIO_SPEC_SCHEMA,
        "title": "Studio Spec: Saturated Control Loop",
        "description": "A flat block diagram authored as JSON: setpoint and measurement feed an error block, then one controller block with nested runtime elements.",
        "simulation": { "steps": 120, "dt": 0.05 },
        "blocks": [
            {
                "id": "setpoint",
                "label": "setpoint",
                "role": "source",
                "x": 40.0,
                "y": 65.0,
                "cell": [{ "op": "source", "signal": "const", "value": 1.0 }]
            },
            {
                "id": "measurement",
                "label": "measurement",
                "role": "source",
                "x": 40.0,
                "y": 210.0,
                "cell": [{ "op": "source", "signal": "sine", "amp": 0.35, "freq": 0.4, "bias": 0.35 }]
            },
            {
                "id": "error",
                "label": "error = r - y",
                "role": "transform",
                "x": 255.0,
                "y": 137.0,
                "cell": [{ "op": "sum", "weights": [1.0, -1.0] }]
            },
            {
                "id": "controller",
                "label": "controller",
                "role": "transform",
                "x": 470.0,
                "y": 137.0,
                "cell": [
                    { "op": "gain", "k": 1.0, "name": "Kp" },
                    {
                        "op": "composite",
                        "name": "limit",
                        "cell": [{ "op": "saturation", "lo": -1.0, "hi": 1.0 }]
                    }
                ]
            },
            {
                "id": "command",
                "label": "command",
                "role": "sink",
                "x": 690.0,
                "y": 147.0,
                "cell": [{ "op": "gain", "k": 1.0, "name": "probe" }]
            }
        ],
        "wires": [
            { "from": "setpoint", "out": 0, "to": "error", "in": 0 },
            { "from": "measurement", "out": 0, "to": "error", "in": 1 },
            { "from": "error", "out": 0, "to": "controller", "in": 0 },
            { "from": "controller", "out": 0, "to": "command", "in": 0 }
        ],
        "design": {
            "variables": [
                {
                    "id": "controller_gain",
                    "block": "controller",
                    "op": 0,
                    "field": "k",
                    "lower": 0.0,
                    "upper": 6.0,
                    "initial": 1.0
                }
            ],
            "objectives": [
                { "id": "track_command", "block": "command", "target": 0.5, "weight": 1.0 }
            ],
            "driver": {
                "kind": "finite-difference-gradient-descent",
                "iterations": 24,
                "step": 0.2,
                "eps": 0.0001
            }
        }
    })
}

/// Parse a JSON studio spec into a runnable demo payload.
pub fn demo_from_spec(spec: &Value) -> Result<StudioDemo, StudioSpecError> {
    let root = spec
        .as_object()
        .ok_or_else(|| StudioSpecError::new("studio spec must be a JSON object"))?;
    let blocks = root
        .get("blocks")
        .and_then(Value::as_array)
        .ok_or_else(|| StudioSpecError::new("studio spec requires a non-empty `blocks` array"))?;
    if blocks.is_empty() {
        return Err(StudioSpecError::new(
            "studio spec requires at least one block in `blocks`",
        ));
    }
    if blocks.len() > MAX_MODEL_BLOCKS {
        return Err(StudioSpecError::new(format!(
            "studio spec is limited to {MAX_MODEL_BLOCKS} blocks"
        )));
    }

    let sim = root.get("simulation").unwrap_or(&Value::Null);
    let steps = read_usize(sim, "steps").unwrap_or(80);
    if steps == 0 || steps > MAX_RUN_STEPS {
        return Err(StudioSpecError::new(format!(
            "simulation.steps must be in 1..={MAX_RUN_STEPS}"
        )));
    }
    let dt = read_f64(sim, "dt").unwrap_or(0.1);
    if !dt.is_finite() || dt <= 0.0 {
        return Err(StudioSpecError::new(
            "simulation.dt must be a positive finite number",
        ));
    }
    let title = read_str(root, "title").unwrap_or("Studio Block Diagram");
    let description = read_str(root, "description").unwrap_or(
        "A JSON-authored flat visual block diagram running on the studio dataflow executive.",
    );
    validate_user_text(title, "studio spec title", MAX_LABEL_LEN, false)?;
    validate_user_text(
        description,
        "studio spec description",
        MAX_LABEL_LEN * 4,
        true,
    )?;

    let mut graph = StudioGraph::new();
    let mut ids: HashMap<String, usize> = HashMap::new();

    for (idx, block) in blocks.iter().enumerate() {
        let obj = block
            .as_object()
            .ok_or_else(|| StudioSpecError::new(format!("blocks[{idx}] must be an object")))?;
        let id = read_str(obj, "id")
            .ok_or_else(|| StudioSpecError::new(format!("blocks[{idx}] requires string `id`")))?;
        validate_user_text(id, "block id", MAX_BLOCK_ID_LEN, false)?;
        let role = parse_role(read_str(obj, "role").unwrap_or("transform"))?;
        let cell_value = obj
            .get("cell")
            .ok_or_else(|| StudioSpecError::new(format!("block `{id}` requires `cell`")))?;
        let cell = parse_cell(cell_value, &format!("block `{id}` cell"))?;
        let label = read_str(obj, "label").unwrap_or(id);
        validate_user_text(label, "block label", MAX_LABEL_LEN, true)?;

        let mut node = VisualNode::new(id, role, cell).with_label(label).at(
            read_f64_obj(obj, "x").unwrap_or(40.0 + idx as f64 * 190.0),
            read_f64_obj(obj, "y").unwrap_or(120.0),
        );
        if let Some(w) = read_f64_obj(obj, "w") {
            node.w = w.max(56.0);
        }
        if let Some(h) = read_f64_obj(obj, "h") {
            node.h = h.max(44.0);
        }

        let handle = graph.add(node)?;
        ids.insert(id.to_string(), handle);
    }

    let wires = root
        .get("wires")
        .and_then(Value::as_array)
        .ok_or_else(|| StudioSpecError::new("studio spec requires a `wires` array"))?;
    if wires.len() > MAX_MODEL_WIRES {
        return Err(StudioSpecError::new(format!(
            "studio spec is limited to {MAX_MODEL_WIRES} wires"
        )));
    }
    for (idx, wire) in wires.iter().enumerate() {
        let obj = wire
            .as_object()
            .ok_or_else(|| StudioSpecError::new(format!("wires[{idx}] must be an object")))?;
        let from = read_str(obj, "from")
            .ok_or_else(|| StudioSpecError::new(format!("wires[{idx}] requires string `from`")))?;
        let to = read_str(obj, "to")
            .ok_or_else(|| StudioSpecError::new(format!("wires[{idx}] requires string `to`")))?;
        let from_handle = ids.get(from).copied().ok_or_else(|| {
            StudioSpecError::new(format!(
                "wires[{idx}] references unknown `from` block `{from}`"
            ))
        })?;
        let to_handle = ids.get(to).copied().ok_or_else(|| {
            StudioSpecError::new(format!("wires[{idx}] references unknown `to` block `{to}`"))
        })?;
        graph.connect(
            from_handle,
            read_usize_obj(obj, "out").unwrap_or(0),
            to_handle,
            read_usize_obj(obj, "in").unwrap_or(0),
        )?;
    }

    let compiled = graph.build()?;
    let blocks = blocks_doc(&compiled);
    Ok(StudioDemo {
        compiled,
        steps,
        dt,
        title: title.to_string(),
        description: description.to_string(),
        blocks,
    })
}

fn parse_cell(value: &Value, path: &str) -> Result<RuntimeCell, StudioSpecError> {
    parse_cell_at_depth(value, path, 0)
}

fn parse_cell_at_depth(
    value: &Value,
    path: &str,
    depth: usize,
) -> Result<RuntimeCell, StudioSpecError> {
    if depth > MAX_RUNTIME_NESTING {
        return Err(StudioSpecError::new(format!(
            "{path} exceeds maximum runtime nesting depth {MAX_RUNTIME_NESTING}"
        )));
    }
    let stages = value
        .as_array()
        .ok_or_else(|| StudioSpecError::new(format!("{path} must be an array of ops")))?;
    if stages.is_empty() {
        return Err(StudioSpecError::new(format!(
            "{path} must contain at least one op"
        )));
    }
    if stages.len() > MAX_RUNTIME_CELL_OPS {
        return Err(StudioSpecError::new(format!(
            "{path} is limited to {MAX_RUNTIME_CELL_OPS} ops"
        )));
    }
    let mut ops: Vec<Box<dyn RuntimeOp>> = Vec::with_capacity(stages.len());
    for (idx, stage) in stages.iter().enumerate() {
        ops.push(parse_op(stage, &format!("{path}[{idx}]"), depth)?);
    }
    RuntimeCell::new(ops).map_err(StudioSpecError::from)
}

fn parse_op(
    value: &Value,
    path: &str,
    depth: usize,
) -> Result<Box<dyn RuntimeOp>, StudioSpecError> {
    let obj = value
        .as_object()
        .ok_or_else(|| StudioSpecError::new(format!("{path} must be an object")))?;
    let op = read_str(obj, "op")
        .ok_or_else(|| StudioSpecError::new(format!("{path} requires string `op`")))?;
    let name = read_str(obj, "name").unwrap_or(op);
    match op {
        "source" => Ok(Box::new(Source::new(name, parse_source_kind(obj, path)?))),
        "gain" => Ok(Box::new(Gain::new(
            name,
            read_f64_obj(obj, "k")
                .ok_or_else(|| StudioSpecError::new(format!("{path} gain requires numeric `k`")))?,
        ))),
        "saturation" => {
            let lo = read_f64_obj(obj, "lo").ok_or_else(|| {
                StudioSpecError::new(format!("{path} saturation requires numeric `lo`"))
            })?;
            let hi = read_f64_obj(obj, "hi").ok_or_else(|| {
                StudioSpecError::new(format!("{path} saturation requires numeric `hi`"))
            })?;
            if lo > hi {
                return Err(StudioSpecError::new(format!(
                    "{path} saturation lower bound exceeds upper bound"
                )));
            }
            Ok(Box::new(Saturation::new(name, lo, hi)))
        }
        "affine" => Ok(Box::new(Affine::new(
            name,
            read_f64_obj(obj, "m").unwrap_or(1.0),
            read_f64_obj(obj, "b").unwrap_or(0.0),
        ))),
        "sum" => {
            let weights = obj
                .get("weights")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    StudioSpecError::new(format!("{path} sum requires `weights` array"))
                })?
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                        StudioSpecError::new(format!("{path} weights[{i}] must be numeric"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if weights.is_empty() {
                return Err(StudioSpecError::new(format!(
                    "{path} sum requires at least one weight"
                )));
            }
            if weights.len() > MAX_PARAM_VECTOR_LEN {
                return Err(StudioSpecError::new(format!(
                    "{path} sum is limited to {MAX_PARAM_VECTOR_LEN} weights"
                )));
            }
            Ok(Box::new(Sum::new(name, weights)))
        }
        "integrator" => Ok(Box::new(Integrator::new(
            name,
            read_f64_obj(obj, "initial").unwrap_or(0.0),
        ))),
        "queue" => {
            let service_rate = read_f64_obj(obj, "serviceRate").ok_or_else(|| {
                StudioSpecError::new(format!("{path} queue requires numeric `serviceRate`"))
            })?;
            if service_rate < 0.0 {
                return Err(StudioSpecError::new(format!(
                    "{path} queue requires non-negative `serviceRate`"
                )));
            }
            Ok(Box::new(Queue::new(name, service_rate)))
        }
        "delay" => {
            let ticks = read_usize_obj(obj, "ticks").ok_or_else(|| {
                StudioSpecError::new(format!("{path} delay requires integer `ticks`"))
            })?;
            if ticks == 0 {
                return Err(StudioSpecError::new(format!(
                    "{path} delay requires positive integer `ticks`"
                )));
            }
            Ok(Box::new(TransportDelay::new(name, ticks)))
        }
        "composite" => {
            let inner = obj
                .get("cell")
                .ok_or_else(|| StudioSpecError::new(format!("{path} composite requires `cell`")))?;
            Ok(Box::new(Composite::new(
                name,
                parse_cell_at_depth(inner, path, depth + 1)?,
            )))
        }
        other => Err(StudioSpecError::new(format!(
            "{path} has unknown op `{other}`"
        ))),
    }
}

fn parse_source_kind(
    obj: &serde_json::Map<String, Value>,
    path: &str,
) -> Result<SourceKind, StudioSpecError> {
    let signal = read_str(obj, "signal").unwrap_or("const");
    match signal {
        "const" => Ok(SourceKind::Const(read_f64_obj(obj, "value").unwrap_or(0.0))),
        "step" => Ok(SourceKind::Step {
            t0: read_f64_obj(obj, "t0").unwrap_or(0.0),
            before: read_f64_obj(obj, "before").unwrap_or(0.0),
            after: read_f64_obj(obj, "after").unwrap_or(1.0),
        }),
        "ramp" => Ok(SourceKind::Ramp {
            slope: read_f64_obj(obj, "slope").unwrap_or(1.0),
            intercept: read_f64_obj(obj, "intercept").unwrap_or(0.0),
        }),
        "sine" => Ok(SourceKind::Sine {
            amp: read_f64_obj(obj, "amp").unwrap_or(1.0),
            freq: read_f64_obj(obj, "freq").unwrap_or(1.0),
            bias: read_f64_obj(obj, "bias").unwrap_or(0.0),
        }),
        other => Err(StudioSpecError::new(format!(
            "{path} source has unknown signal `{other}`"
        ))),
    }
}

fn parse_role(role: &str) -> Result<NodeRole, StudioSpecError> {
    match role {
        "source" => Ok(NodeRole::Source),
        "transform" => Ok(NodeRole::Transform),
        "sink" => Ok(NodeRole::Sink),
        other => Err(StudioSpecError::new(format!(
            "unknown block role `{other}`"
        ))),
    }
}

fn read_str<'a>(obj: &'a serde_json::Map<String, Value>, key: &str) -> Option<&'a str> {
    obj.get(key).and_then(Value::as_str)
}

fn read_f64(value: &Value, key: &str) -> Option<f64> {
    value.as_object().and_then(|o| read_f64_obj(o, key))
}

fn read_f64_obj(obj: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
    obj.get(key)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
}

fn read_usize(value: &Value, key: &str) -> Option<usize> {
    value.as_object().and_then(|o| read_usize_obj(o, key))
}

fn read_usize_obj(obj: &serde_json::Map<String, Value>, key: &str) -> Option<usize> {
    obj.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::studio::run::run;
    use serde_json::json;

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
    fn compile_rejects_wrong_schema_and_excessive_steps() {
        let mut spec = starter_model_spec();
        spec.schema = "des/studio-graph/v0".to_string();
        let err = compile_model_spec(&spec).unwrap_err();
        assert!(err.to_string().contains("unsupported studio schema"));

        let mut spec = starter_model_spec();
        spec.steps = MAX_RUN_STEPS + 1;
        assert!(matches!(
            compile_model_spec(&spec),
            Err(StudioSpecError::InvalidParam { param, .. }) if param == "steps"
        ));
    }

    #[test]
    fn compile_rejects_unknown_and_unsafe_block_params() {
        let mut spec = starter_model_spec();
        spec.blocks[1]
            .params
            .insert("surprise".to_string(), Value::from(1.0));
        assert!(matches!(
            compile_model_spec(&spec),
            Err(StudioSpecError::InvalidParam { param, .. }) if param == "surprise"
        ));

        let mut spec = starter_model_spec();
        spec.design_variables.clear();
        spec.blocks[1].kind = StudioBlockKind::Queue;
        spec.blocks[1].params = Map::from_iter([("serviceRate".to_string(), Value::from(-1.0))]);
        assert!(matches!(
            compile_model_spec(&spec),
            Err(StudioSpecError::InvalidParam { param, .. }) if param == "serviceRate"
        ));

        let mut spec = starter_model_spec();
        spec.design_variables.clear();
        spec.blocks[1].kind = StudioBlockKind::Saturation;
        spec.blocks[1].params = Map::from_iter([
            ("lo".to_string(), Value::from(2.0)),
            ("hi".to_string(), Value::from(1.0)),
        ]);
        assert!(matches!(
            compile_model_spec(&spec),
            Err(StudioSpecError::InvalidParam { param, .. }) if param == "lo"
        ));
    }

    #[test]
    fn compile_rejects_bad_design_variable_reference() {
        let mut spec = starter_model_spec();
        spec.design_variables[0].block = "missing".to_string();
        let err = compile_model_spec(&spec).unwrap_err();
        assert!(err
            .to_string()
            .contains("design variable `gain.k` references unknown block `missing`"));
    }

    #[test]
    fn compile_rejects_non_scalar_design_variable_param() {
        let mut spec = starter_model_spec();
        spec.blocks.push(StudioBlockSpec {
            id: "sum".to_string(),
            kind: StudioBlockKind::Sum,
            label: None,
            params: Map::from_iter([(
                "weights".to_string(),
                Value::Array(vec![Value::from(1.0), Value::from(-1.0)]),
            )]),
            x: 0.0,
            y: 260.0,
        });
        spec.design_variables[0].block = "sum".to_string();
        spec.design_variables[0].param = "weights".to_string();
        let err = compile_model_spec(&spec).unwrap_err();
        assert!(err
            .to_string()
            .contains("expected a scalar number parameter"));
    }

    #[test]
    fn compile_rejects_bad_metric_metadata() {
        let mut spec = starter_model_spec();
        spec.objectives[0].port = 1;
        let err = compile_model_spec(&spec).unwrap_err();
        assert!(err
            .to_string()
            .contains("objective `final output` uses unsupported port 1"));

        spec.objectives[0].port = 0;
        spec.constraints[0].lower = Some(9.0);
        spec.constraints[0].upper = Some(8.0);
        let err = compile_model_spec(&spec).unwrap_err();
        assert!(err
            .to_string()
            .contains("constraint `output ceiling` lower bound exceeds upper bound"));
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

    #[test]
    fn example_spec_runs_to_command_signal() {
        let mut demo = demo_from_spec(&example_spec()).unwrap();
        let out = run(&mut demo.compiled, demo.steps, demo.dt);
        assert_eq!(out.steps, 120);
        assert!(out.series("command").is_some());
        assert!(out.final_value("command").unwrap().abs() <= 1.0 + 1e-9);
    }

    #[test]
    fn parser_reports_unknown_wire_endpoint() {
        let mut spec = example_spec();
        let wires = spec.get_mut("wires").unwrap().as_array_mut().unwrap();
        wires[0]["from"] = json!("missing");
        let err = match demo_from_spec(&spec) {
            Ok(_) => panic!("expected unknown wire endpoint error"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("unknown `from` block `missing`"));
    }

    #[test]
    fn parser_rejects_unbounded_legacy_specs() {
        let spec = json!({
            "$schema": STUDIO_SPEC_SCHEMA,
            "simulation": { "steps": MAX_RUN_STEPS + 1, "dt": 1.0 },
            "blocks": [
                { "id": "a", "role": "source", "cell": [{ "op": "source", "value": 1.0 }] }
            ],
            "wires": []
        });
        let err = match demo_from_spec(&spec) {
            Ok(_) => panic!("expected excessive steps error"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("simulation.steps"));

        let spec = json!({
            "$schema": STUDIO_SPEC_SCHEMA,
            "simulation": { "steps": 1, "dt": 1.0 },
            "blocks": [
                { "id": "a", "role": "source", "cell": [{ "op": "queue", "serviceRate": -1.0 }] }
            ],
            "wires": []
        });
        let err = match demo_from_spec(&spec) {
            Ok(_) => panic!("expected negative queue service rate error"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("non-negative"));
    }

    #[test]
    fn parser_supports_queue_and_delay_ops() {
        let spec = json!({
            "$schema": STUDIO_SPEC_SCHEMA,
            "simulation": { "steps": 8, "dt": 1.0 },
            "blocks": [
                { "id": "arrivals", "role": "source", "cell": [{ "op": "source", "signal": "step", "after": 4.0 }] },
                { "id": "server", "role": "transform", "cell": [{ "op": "queue", "serviceRate": 2.0 }] },
                { "id": "belt", "role": "transform", "cell": [{ "op": "delay", "ticks": 2 }] },
                { "id": "out", "role": "sink", "cell": [{ "op": "gain", "k": 1.0 }] }
            ],
            "wires": [
                { "from": "arrivals", "to": "server" },
                { "from": "server", "to": "belt" },
                { "from": "belt", "to": "out" }
            ]
        });
        let mut demo = demo_from_spec(&spec).unwrap();
        let out = run(&mut demo.compiled, demo.steps, demo.dt);
        assert_eq!(out.series("server").unwrap()[4], 2.0);
        assert_eq!(out.series("out").unwrap()[6], 2.0);
    }
}
