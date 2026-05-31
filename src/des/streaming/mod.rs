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

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub mod lp;
pub mod mdp;
pub mod milp;
pub mod pomdp;

pub use lp::StreamingLp;
pub use mdp::StreamingMdp;
pub use milp::{StreamingIp, StreamingMilp, StreamingMip};
pub use pomdp::StreamingPomdp;

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

/// The concrete optimization/decision model family carried by a stream.
///
/// This is intentionally narrower than [`SolverKind`]: all current streaming
/// models are iterative solvers, but a server/desktop app/IPC bridge still needs
/// to know whether a frame belongs to LP, integer programming, MDP, or POMDP.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelStreamKind {
    Lp,
    Ip,
    Mip,
    Mdp,
    Pomdp,
    GenericIterative,
}

impl ModelStreamKind {
    pub fn from_model_name(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        if m.contains("pomdp") {
            ModelStreamKind::Pomdp
        } else if m.contains("mdp") {
            ModelStreamKind::Mdp
        } else if m.contains("milp") || m.contains("mip") {
            ModelStreamKind::Mip
        } else if m.contains("integer") || m.ends_with("-ip") || m.contains("ip-") {
            ModelStreamKind::Ip
        } else if m.contains("lp") {
            ModelStreamKind::Lp
        } else {
            ModelStreamKind::GenericIterative
        }
    }
}

/// Direction of a framed streaming document at the SDK/IPC boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StreamDirection {
    In,
    Out,
}

/// A typed JSONL envelope for IPC/server/desktop transports.
///
/// Existing streamers still accept their compact command objects directly. This
/// envelope is the hardened SDK boundary: transport code can sequence,
/// authenticate, multiplex, and route frames without inspecting model-specific
/// payloads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamFrame {
    pub seq: u64,
    pub model: ModelStreamKind,
    pub direction: StreamDirection,
    pub payload: Value,
}

impl StreamFrame {
    pub fn input(seq: u64, model: ModelStreamKind, payload: Value) -> Self {
        StreamFrame {
            seq,
            model,
            direction: StreamDirection::In,
            payload,
        }
    }

    pub fn output(seq: u64, model: ModelStreamKind, payload: Value) -> Self {
        StreamFrame {
            seq,
            model,
            direction: StreamDirection::Out,
            payload,
        }
    }
}

/// Minimal state for validating and producing sequential stream envelopes.
#[derive(Clone, Debug)]
pub struct StreamSession {
    model: ModelStreamKind,
    next_in_seq: u64,
    next_out_seq: u64,
}

impl StreamSession {
    pub fn new(model: ModelStreamKind) -> Self {
        StreamSession {
            model,
            next_in_seq: 0,
            next_out_seq: 0,
        }
    }

    pub fn accept(&mut self, frame: &StreamFrame) -> Result<(), String> {
        if frame.model != self.model {
            return Err(format!(
                "frame model {:?} does not match session model {:?}",
                frame.model, self.model
            ));
        }
        if frame.direction != StreamDirection::In {
            return Err("input session only accepts inbound frames".to_string());
        }
        if frame.seq != self.next_in_seq {
            return Err(format!(
                "expected inbound seq {}, got {}",
                self.next_in_seq, frame.seq
            ));
        }
        self.next_in_seq += 1;
        Ok(())
    }

    pub fn emit(&mut self, payload: Value) -> StreamFrame {
        let frame = StreamFrame::output(self.next_out_seq, self.model, payload);
        self.next_out_seq += 1;
        frame
    }
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
    pub model_stream_kind: ModelStreamKind,
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
        Self::new_for_stream_kind(
            model,
            kind,
            ModelStreamKind::from_model_name(model),
            description,
            input_ops,
            output_events,
        )
    }

    /// Build a contract with an explicit concrete model family.
    pub fn new_for_stream_kind(
        model: &str,
        kind: SolverKind,
        model_stream_kind: ModelStreamKind,
        description: &str,
        input_ops: Vec<StreamOp>,
        output_events: Vec<StreamEvent>,
    ) -> Self {
        StreamContract {
            model: model.to_string(),
            kind,
            model_stream_kind,
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

/// Apply one command with panic isolation: if the underlying solver `panic!`s on
/// malformed input (several numeric backends do), the panic is caught and turned
/// into an [`error_frame`] so one bad command never tears down a long-lived
/// stream. Well-behaved models that already return error frames are unaffected.
pub fn apply_safe(model: &mut dyn StreamingModel, command: &Value) -> Vec<Value> {
    match catch_unwind(AssertUnwindSafe(|| model.apply(command))) {
        Ok(frames) => frames,
        Err(_) => vec![error_frame(
            "the solver rejected this command (internal error); the stream continues",
        )],
    }
}

/// Apply a batch of commands and collect every emitted frame. Pure (no I/O) —
/// handy for embedding a streaming model in another program and for tests.
/// Panic-isolated per command (see [`apply_safe`]).
pub fn drive(model: &mut dyn StreamingModel, commands: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for command in commands {
        out.extend(apply_safe(model, command));
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
            Ok(command) => apply_safe(model, &command),
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
// Model registry — pick a streaming model by name (the seam a CLI / HTTP layer
// uses to route a request to the right solver).
// =============================================================================

/// The streaming models the engine ships, by stable name.
pub fn streaming_model_names() -> &'static [&'static str] {
    &["lp", "milp", "mdp", "pomdp"]
}

/// Construct a boxed streaming model by name. Aliases `mip`/`ip` to the MILP
/// branch-and-bound streamer (integer/mixed-integer programs are solved there).
/// Returns `None` for an unknown name so the caller can report it cleanly.
pub fn build_streaming_model(name: &str) -> Option<Box<dyn StreamingModel>> {
    match name {
        "lp" => Some(Box::new(lp::StreamingLp::new())),
        "milp" | "mip" | "ip" => Some(Box::new(milp::StreamingMilp::new())),
        "mdp" => Some(Box::new(mdp::StreamingMdp::new())),
        "pomdp" => Some(Box::new(pomdp::StreamingPomdp::new())),
        _ => None,
    }
}

/// Every shipped model's self-describing contract — for a server to advertise
/// the streaming catalogue (e.g. in its `/api/docs.json`) without instantiating.
pub fn streaming_contracts() -> Vec<StreamContract> {
    streaming_model_names()
        .iter()
        .filter_map(|name| build_streaming_model(name).map(|m| m.contract()))
        .collect()
}

/// Run a named streaming model over JSONL (the end-to-end "pick model + stream"
/// entry point a CLI or request handler calls). Returns `Ok(false)` if `name` is
/// unknown (so the caller can 404), `Ok(true)` after a completed stream. Per-line
/// JSON and per-command solver panics are isolated (see [`run_jsonl`]).
pub fn run_named_jsonl<R: BufRead, W: Write>(
    name: &str,
    input: R,
    output: &mut W,
) -> io::Result<bool> {
    match build_streaming_model(name) {
        Some(mut model) => {
            run_jsonl(model.as_mut(), input, output)?;
            Ok(true)
        }
        None => Ok(false),
    }
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

pub(crate) fn probability_error(probs: &[f64], label: &str) -> Option<Value> {
    if probs.is_empty() {
        return Some(error_frame(format!("`{label}` must not be empty")));
    }
    if probs.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return Some(error_frame(format!(
            "`{label}` must contain only finite, non-negative probabilities"
        )));
    }
    let sum: f64 = probs.iter().sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Some(error_frame(format!(
            "`{label}` probabilities must sum to 1.0 (got {sum})"
        )));
    }
    None
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
                vec![StreamOp::new(
                    "echo",
                    "echo a value",
                    json!({"op":"echo","value":1}),
                )],
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
            &[
                json!({"op":"echo","value":1}),
                json!({"op":"echo","value":2}),
            ],
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
        assert_eq!(v["modelStreamKind"], json!("generic-iterative"));
        assert_eq!(v["inputMediaType"], json!(JSONL_MEDIA_TYPE));
    }

    /// A model whose `apply` panics — used to prove panic isolation.
    struct PanicModel;
    impl StreamingModel for PanicModel {
        fn kind(&self) -> SolverKind {
            SolverKind::IterativeSolver
        }
        fn contract(&self) -> StreamContract {
            StreamContract::new("panic", SolverKind::IterativeSolver, "panics", vec![], vec![])
        }
        fn apply(&mut self, _command: &Value) -> Vec<Value> {
            panic!("boom");
        }
    }

    #[test]
    fn apply_safe_converts_a_solver_panic_into_an_error_frame() {
        // Silence the default panic hook so the caught panic doesn't spam stderr.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let mut m = PanicModel;
        let frames = drive(&mut m, &[json!({ "op": "anything" })]);
        std::panic::set_hook(prev);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0]["event"], json!("error"));
    }

    #[test]
    fn registry_builds_every_shipped_model_and_round_trips_its_contract() {
        for name in streaming_model_names() {
            let model = build_streaming_model(name).expect("shipped model builds");
            let contract = model.contract();
            // Round-trip the contract through JSON and back.
            let json_str = contract.to_json_string();
            let back: Value = serde_json::from_str(&json_str).expect("contract round-trips");
            assert_eq!(back["kind"], json!("iterative-solver"));
            assert!(back["model"].as_str().is_some());
            assert!(back["inputOps"].as_array().map(|a| !a.is_empty()).unwrap_or(false));
        }
        // Aliases route integer/mixed-integer programs to the MILP streamer.
        assert!(build_streaming_model("mip").is_some());
        assert!(build_streaming_model("ip").is_some());
        assert!(build_streaming_model("nope").is_none());
        assert_eq!(streaming_contracts().len(), streaming_model_names().len());
    }

    #[test]
    fn run_named_jsonl_routes_known_models_and_rejects_unknown() {
        // Known model: initialise an MDP and confirm a frame comes back.
        let input = "{\"op\":\"init\",\"numStates\":2,\"gamma\":0.9}\n";
        let mut out: Vec<u8> = Vec::new();
        let handled = run_named_jsonl("mdp", input.as_bytes(), &mut out).unwrap();
        assert!(handled, "mdp is a known model");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("\"event\""), "got a frame: {text}");

        // Unknown model: Ok(false), nothing written.
        let mut out2: Vec<u8> = Vec::new();
        let handled = run_named_jsonl("does-not-exist", "{}\n".as_bytes(), &mut out2).unwrap();
        assert!(!handled);
        assert!(out2.is_empty());
    }

    #[test]
    fn stream_session_rejects_wrong_sequence_or_model() {
        let mut session = StreamSession::new(ModelStreamKind::Mip);
        session
            .accept(&StreamFrame::input(
                0,
                ModelStreamKind::Mip,
                json!({"op":"init"}),
            ))
            .unwrap();
        assert!(session
            .accept(&StreamFrame::input(
                0,
                ModelStreamKind::Mip,
                json!({"op":"solve"})
            ))
            .unwrap_err()
            .contains("expected inbound seq 1"));
        assert!(StreamSession::new(ModelStreamKind::Mdp)
            .accept(&StreamFrame::input(0, ModelStreamKind::Pomdp, json!({})))
            .unwrap_err()
            .contains("does not match"));
    }

    #[test]
    fn stream_session_emits_sequential_output_frames() {
        let mut session = StreamSession::new(ModelStreamKind::Lp);
        let a = session.emit(json!({"event":"solution"}));
        let b = session.emit(json!({"event":"trace"}));
        assert_eq!(a.seq, 0);
        assert_eq!(b.seq, 1);
        assert_eq!(a.direction, StreamDirection::Out);
        assert_eq!(a.model, ModelStreamKind::Lp);
    }
}
