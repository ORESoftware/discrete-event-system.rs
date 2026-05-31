//! Streaming (JSONL) contract for the engine's *solvers*.
//!
//! ## Why this exists — three kinds of "runnable" thing
//!
//! Not everything in this engine is a discrete-event simulation. There are
//! three distinct shapes, and only the third benefits from a streaming I/O
//! contract:
//!
//! 1. **Time-stepped DES** ([`SolverKind::TimeSteppedDes`]) — a discrete-event
//!    simulation advanced by a fixed time step (not a future-event list).
//!    Driven forward by *time*; the "input" is the model parameters, the output
//!    is a trajectory. Examples: epidemics, elevator, traffic, temp-control.
//! 2. **Time-stepped numeric solver** ([`SolverKind::TimeSteppedNumeric`]) — an
//!    ODE / control / filtering integrator advanced by a time step. Also driven
//!    by *time*. Examples: DC motor, LQR/MPC, Kalman filter, SDEs.
//! 3. **Iterative algorithmic solver** ([`SolverKind::IterativeSolver`]) — LP,
//!    MIP/MILP, MDP/POMDP, graph algorithms. These are *not* time-driven: they
//!    iterate toward a solution of a problem you describe. The problem itself
//!    can change over time (add/remove a variable or constraint, re-weight an
//!    objective, add a transition), and the solver can re-converge and emit an
//!    updated solution. That is exactly a *stream in → stream out* contract.
//!
//! This module defines that contract as [`StreamingModel`] (JSONL in, JSONL
//! out) and ships concrete streamers for the iterative solvers:
//!
//! - [`lp::StreamingLp`] — warm-started incremental LP (the live-edit case:
//!   add/remove variable, add/remove constraint, change objective weights).
//! - [`milp::StreamingMilp`] — MILP built/edited by a stream, re-solved on
//!   demand, streaming the branch-and-bound node trace.
//! - [`mdp::StreamingMdp`] — tabular MDP built by a stream, solved by value
//!   iteration on demand.
//! - [`pomdp::StreamingPomdp`] — tabular POMDP built by a stream, solved on
//!   demand (qmdp / lookahead / exact), recommending an action under the belief.
//!
//! It is *additive*: it wraps the existing solvers (`incremental_lp`,
//! `milp_bnb`, `value_iteration`) through their public APIs and changes none of
//! them. JSON is the boundary, so `serde_json` is used here (never in the
//! engine core), mirroring [`crate::des::service`].

use serde::Serialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

pub mod lp;
pub mod mdp;
pub mod milp;
pub mod pomdp;

/// Newline-delimited JSON media type the streaming contract speaks in both
/// directions (a.k.a. JSON Lines / `jsonl`).
pub const JSONL_MEDIA_TYPE: &str = "application/x-ndjson";

/// Which of the three "runnable" shapes a model is. Only
/// [`SolverKind::IterativeSolver`] models implement [`StreamingModel`] today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SolverKind {
    /// Discrete-event simulation advanced by a fixed time step (not an FEL).
    TimeSteppedDes,
    /// Numerical solver advanced by a time step (ODE / control / filtering).
    TimeSteppedNumeric,
    /// Iterative algorithmic solver (LP / MIP / MDP / POMDP / graph).
    IterativeSolver,
}

/// One input command the model accepts, for the self-describing contract.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamOp {
    pub op: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub example: Option<Value>,
}

impl StreamOp {
    pub fn new(op: &str, description: &str, example: Value) -> Self {
        StreamOp {
            op: op.to_string(),
            description: description.to_string(),
            example: Some(example),
        }
    }
}

/// One output frame type the model emits, for the self-describing contract.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamEvent {
    pub event: String,
    pub description: String,
}

impl StreamEvent {
    pub fn new(event: &str, description: &str) -> Self {
        StreamEvent {
            event: event.to_string(),
            description: description.to_string(),
        }
    }
}

/// Machine-readable description of a model's JSONL streaming contract. JSON-first
/// so a server can advertise it (e.g. alongside the [`crate::des::service`]
/// descriptor) without the client probing.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamContract {
    pub model: String,
    pub kind: SolverKind,
    pub input_media_type: String,
    pub output_media_type: String,
    pub description: String,
    pub input_ops: Vec<StreamOp>,
    pub output_events: Vec<StreamEvent>,
}

impl StreamContract {
    /// Build a contract with the standard JSONL media types filled in.
    pub fn new(
        model: &str,
        kind: SolverKind,
        description: &str,
        input_ops: Vec<StreamOp>,
        output_events: Vec<StreamEvent>,
    ) -> Self {
        StreamContract {
            model: model.to_string(),
            kind,
            input_media_type: JSONL_MEDIA_TYPE.to_string(),
            output_media_type: JSONL_MEDIA_TYPE.to_string(),
            description: description.to_string(),
            input_ops,
            output_events,
        }
    }

    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// A solver that consumes a stream of JSON commands and emits a stream of JSON
/// frames. Each call to [`StreamingModel::apply`] handles exactly one input
/// document (one JSONL line) and returns zero or more output documents (lines).
///
/// Object-safe so [`run_jsonl`] can drive a `&mut dyn StreamingModel`.
pub trait StreamingModel {
    /// Which taxonomy bucket this model belongs to.
    fn kind(&self) -> SolverKind;

    /// The self-describing JSONL contract (accepted ops, emitted events).
    fn contract(&self) -> StreamContract;

    /// Apply one parsed command; return the frames to emit. Recoverable input
    /// problems should be returned as an [`error_frame`] rather than panicking,
    /// so a single bad command never tears down a long-lived stream.
    fn apply(&mut self, command: &Value) -> Vec<Value>;
}

/// A standard error frame (does not terminate the stream).
pub fn error_frame(message: impl Into<String>) -> Value {
    json!({ "event": "error", "message": message.into() })
}

/// Apply a batch of commands and collect every emitted frame. Pure (no I/O) —
/// handy for embedding a streaming model in another program and for tests.
pub fn drive(model: &mut dyn StreamingModel, commands: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for command in commands {
        out.extend(model.apply(command));
    }
    out
}

/// Drive a model over real JSONL: read one command per line from `input`, write
/// one frame per line to `output`. Blank lines are skipped; a malformed line
/// yields an [`error_frame`] and the stream continues.
pub fn run_jsonl<R: BufRead, W: Write>(
    model: &mut dyn StreamingModel,
    input: R,
    output: &mut W,
) -> io::Result<()> {
    for line in input.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let frames = match serde_json::from_str::<Value>(trimmed) {
            Ok(command) => model.apply(&command),
            Err(err) => vec![error_frame(format!("invalid JSON line: {err}"))],
        };
        for frame in frames {
            let line = serde_json::to_string(&frame).unwrap_or_else(|_| "{}".to_string());
            writeln!(output, "{line}")?;
        }
        output.flush()?;
    }
    Ok(())
}

// =============================================================================
// Shared command-parsing helpers (used by the concrete streamers below).
// =============================================================================

pub(crate) fn op_of(command: &Value) -> &str {
    command.get("op").and_then(Value::as_str).unwrap_or("")
}

pub(crate) fn f64_at(command: &Value, key: &str, default: f64) -> f64 {
    command.get(key).and_then(Value::as_f64).unwrap_or(default)
}

pub(crate) fn usize_at(command: &Value, key: &str) -> Option<usize> {
    command.get(key).and_then(Value::as_u64).map(|n| n as usize)
}

pub(crate) fn bool_at(command: &Value, key: &str, default: bool) -> bool {
    command.get(key).and_then(Value::as_bool).unwrap_or(default)
}

pub(crate) fn str_at(command: &Value, key: &str) -> Option<String> {
    command
        .get(key)
        .and_then(Value::as_str)
        .map(|s| s.to_string())
}

pub(crate) fn vec_f64(command: &Value, key: &str) -> Option<Vec<f64>> {
    command.get(key)?.as_array().map(|items| {
        items
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0))
            .collect::<Vec<f64>>()
    })
}

pub(crate) fn vec_vec_f64(command: &Value, key: &str) -> Option<Vec<Vec<f64>>> {
    command.get(key)?.as_array().map(|rows| {
        rows.iter()
            .map(|row| {
                row.as_array()
                    .map(|cells| cells.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default()
            })
            .collect::<Vec<Vec<f64>>>()
    })
}

pub(crate) fn vec_str(command: &Value, key: &str) -> Option<Vec<String>> {
    command.get(key)?.as_array().map(|items| {
        items
            .iter()
            .filter_map(|s| s.as_str().map(String::from))
            .collect::<Vec<String>>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial echo model exercising the driver, framing, and error handling.
    struct EchoModel;
    impl StreamingModel for EchoModel {
        fn kind(&self) -> SolverKind {
            SolverKind::IterativeSolver
        }
        fn contract(&self) -> StreamContract {
            StreamContract::new(
                "echo",
                SolverKind::IterativeSolver,
                "echoes the `value` of each command",
                vec![StreamOp::new("echo", "echo a value", json!({"op":"echo","value":1}))],
                vec![StreamEvent::new("echo", "the echoed value")],
            )
        }
        fn apply(&mut self, command: &Value) -> Vec<Value> {
            match op_of(command) {
                "echo" => vec![json!({"event":"echo","value": command.get("value").cloned()})],
                other => vec![error_frame(format!("unknown op `{other}`"))],
            }
        }
    }

    #[test]
    fn drive_collects_frames_in_order() {
        let mut m = EchoModel;
        let frames = drive(
            &mut m,
            &[json!({"op":"echo","value":1}), json!({"op":"echo","value":2})],
        );
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0]["value"], json!(1));
        assert_eq!(frames[1]["value"], json!(2));
    }

    #[test]
    fn unknown_op_yields_error_frame_not_panic() {
        let mut m = EchoModel;
        let frames = drive(&mut m, &[json!({"op":"nope"})]);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0]["event"], json!("error"));
    }

    #[test]
    fn run_jsonl_reads_and_writes_lines_and_tolerates_bad_json() {
        let mut m = EchoModel;
        let input = "{\"op\":\"echo\",\"value\":7}\n\nnot json\n{\"op\":\"echo\",\"value\":8}\n";
        let mut out: Vec<u8> = Vec::new();
        run_jsonl(&mut m, input.as_bytes(), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        // echo 7, error for "not json", echo 8 — blank line skipped.
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("\"value\":7"));
        assert!(lines[1].contains("\"event\":\"error\""));
        assert!(lines[2].contains("\"value\":8"));
    }

    #[test]
    fn contract_serializes_with_media_types() {
        let c = EchoModel.contract();
        let v: Value = serde_json::from_str(&c.to_json_string()).unwrap();
        assert_eq!(v["model"], json!("echo"));
        assert_eq!(v["kind"], json!("iterative-solver"));
        assert_eq!(v["inputMediaType"], json!(JSONL_MEDIA_TYPE));
    }
}
