//! JSON specification support for the visual-block studio.
//!
//! The demo citizen is useful for screenshots and regression tests, but an
//! open Simulink/Modelica-style tool needs a real model document users and
//! tools can author. This module parses a small, explicit JSON block-diagram
//! contract into the existing flat [`StudioGraph`] runtime.

use std::collections::HashMap;

use serde_json::{json, Value};

use super::cell::{
    Affine, Composite, Gain, Integrator, Queue, RuntimeCell, RuntimeOp, Saturation, Source,
    SourceKind, Sum, TransportDelay,
};
use super::demos::{blocks_doc, StudioDemo};
use super::graph::{NodeRole, StudioError, StudioGraph, VisualNode};

/// Schema id for arbitrary studio block-diagram specs.
pub const STUDIO_SPEC_SCHEMA: &str = "des/studio/v1";

/// Recoverable spec parsing error, phrased for a user or LLM to fix.
#[derive(Clone, Debug, PartialEq)]
pub struct StudioSpecError {
    message: String,
}

impl StudioSpecError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        StudioSpecError {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for StudioSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for StudioSpecError {}

impl From<StudioError> for StudioSpecError {
    fn from(value: StudioError) -> Self {
        StudioSpecError::new(value.to_string())
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

    let sim = root.get("simulation").unwrap_or(&Value::Null);
    let steps = read_usize(sim, "steps").unwrap_or(80).max(1);
    let dt = read_f64(sim, "dt").unwrap_or(0.1).max(f64::EPSILON);
    let title = read_str(root, "title").unwrap_or("Studio Block Diagram");
    let description = read_str(root, "description").unwrap_or(
        "A JSON-authored flat visual block diagram running on the studio dataflow executive.",
    );

    let mut graph = StudioGraph::new();
    let mut ids: HashMap<String, usize> = HashMap::new();

    for (idx, block) in blocks.iter().enumerate() {
        let obj = block
            .as_object()
            .ok_or_else(|| StudioSpecError::new(format!("blocks[{idx}] must be an object")))?;
        let id = read_str(obj, "id")
            .ok_or_else(|| StudioSpecError::new(format!("blocks[{idx}] requires string `id`")))?;
        let role = parse_role(read_str(obj, "role").unwrap_or("transform"))?;
        let cell_value = obj
            .get("cell")
            .ok_or_else(|| StudioSpecError::new(format!("block `{id}` requires `cell`")))?;
        let cell = parse_cell(cell_value, &format!("block `{id}` cell"))?;

        let mut node = VisualNode::new(id, role, cell)
            .with_label(read_str(obj, "label").unwrap_or(id))
            .at(
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
    let stages = value
        .as_array()
        .ok_or_else(|| StudioSpecError::new(format!("{path} must be an array of ops")))?;
    if stages.is_empty() {
        return Err(StudioSpecError::new(format!(
            "{path} must contain at least one op"
        )));
    }
    let mut ops: Vec<Box<dyn RuntimeOp>> = Vec::with_capacity(stages.len());
    for (idx, stage) in stages.iter().enumerate() {
        ops.push(parse_op(stage, &format!("{path}[{idx}]"))?);
    }
    RuntimeCell::new(ops).map_err(StudioSpecError::from)
}

fn parse_op(value: &Value, path: &str) -> Result<Box<dyn RuntimeOp>, StudioSpecError> {
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
        "saturation" => Ok(Box::new(Saturation::new(
            name,
            read_f64_obj(obj, "lo").ok_or_else(|| {
                StudioSpecError::new(format!("{path} saturation requires numeric `lo`"))
            })?,
            read_f64_obj(obj, "hi").ok_or_else(|| {
                StudioSpecError::new(format!("{path} saturation requires numeric `hi`"))
            })?,
        ))),
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
                    v.as_f64().ok_or_else(|| {
                        StudioSpecError::new(format!("{path} weights[{i}] must be numeric"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if weights.is_empty() {
                return Err(StudioSpecError::new(format!(
                    "{path} sum requires at least one weight"
                )));
            }
            Ok(Box::new(Sum::new(name, weights)))
        }
        "integrator" => Ok(Box::new(Integrator::new(
            name,
            read_f64_obj(obj, "initial").unwrap_or(0.0),
        ))),
        "queue" => Ok(Box::new(Queue::new(
            name,
            read_f64_obj(obj, "serviceRate").ok_or_else(|| {
                StudioSpecError::new(format!("{path} queue requires numeric `serviceRate`"))
            })?,
        ))),
        "delay" => Ok(Box::new(TransportDelay::new(
            name,
            read_usize_obj(obj, "ticks").ok_or_else(|| {
                StudioSpecError::new(format!("{path} delay requires integer `ticks`"))
            })?,
        ))),
        "composite" => {
            let inner = obj
                .get("cell")
                .ok_or_else(|| StudioSpecError::new(format!("{path} composite requires `cell`")))?;
            Ok(Box::new(Composite::new(name, parse_cell(inner, path)?)))
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
    obj.get(key).and_then(Value::as_f64)
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
    use super::super::run::run;
    use super::*;

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
