//! Run an external plugin program and capture its JSON / JSONL output.
//!
//! The runner is **language-agnostic** — it spawns whatever `run.command`
//! names and reads stdout. "Rust for now" only means the example plugins are
//! Rust binaries; the host neither knows nor cares. Parsing is split out
//! ([`parse_output`]) so it can be tested without spawning a process.
//!
//! ## Transport seam ([`PluginTransport`])
//!
//! Execution is abstracted behind the [`PluginTransport`] trait so the *how*
//! (spawn a child, talk over a socket, call across a C ABI) is decoupled from
//! the *what* (a [`PluginManifest`] describing a program that speaks JSON/JSONL).
//! [`ProcessTransport`] is the built-in implementation: it spawns a child
//! process, optionally streams a JSON spec to the child's stdin, enforces
//! `run.timeout_ms`, and captures stdout/stderr. Future cross-language plugins
//! (Python/C++ over IPC, or a C-ABI shared library) are additional `impl`s of
//! this same trait — the manifest and the output contract stay identical.

use std::io::{Read, Write};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::Value;

use super::manifest::{OutputKind, PluginManifest};

/// Why a plugin run failed. Recoverable (returned, never panicked) and
/// `Serialize` so an HTTP layer can return a machine-readable error body.
#[derive(Debug, Serialize)]
#[serde(tag = "error", content = "detail", rename_all = "camelCase")]
pub enum PluginError {
    /// The process could not be spawned (command missing, not executable, …).
    Spawn(String),
    /// The process ran but exited non-zero.
    #[serde(rename_all = "camelCase")]
    NonZeroExit { code: Option<i32>, stderr: String },
    /// stdout was not valid JSON / JSONL.
    Parse(String),
    /// The program produced no output.
    Empty,
    /// The plugin exceeded its `run.timeout_ms` budget and was killed.
    #[serde(rename_all = "camelCase")]
    Timeout { after_ms: u64 },
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginError::Spawn(m) => write!(f, "failed to spawn plugin: {m}"),
            PluginError::NonZeroExit { code, stderr } => write!(
                f,
                "plugin exited with {} — stderr: {}",
                code.map(|c| c.to_string()).unwrap_or_else(|| "signal".to_string()),
                stderr.trim()
            ),
            PluginError::Parse(m) => write!(f, "could not parse plugin output: {m}"),
            PluginError::Empty => write!(f, "plugin produced no output"),
            PluginError::Timeout { after_ms } => {
                write!(f, "plugin timed out after {after_ms} ms and was killed")
            }
        }
    }
}

impl std::error::Error for PluginError {}

/// Parsed plugin output.
#[derive(Clone, Debug)]
pub enum PluginOutput {
    /// A single JSON document.
    Json(Value),
    /// A stream of JSON documents (one per JSONL line) = frames.
    Jsonl(Vec<Value>),
}

impl PluginOutput {
    /// Number of frames (1 for a single JSON document).
    pub fn frame_count(&self) -> usize {
        match self {
            PluginOutput::Json(_) => 1,
            PluginOutput::Jsonl(v) => v.len(),
        }
    }
}

/// The result of running a plugin.
#[derive(Clone, Debug)]
pub struct PluginRun {
    pub plugin_id: String,
    pub output: PluginOutput,
    pub exit_code: Option<i32>,
    pub stderr: String,
}

/// Parse captured stdout per the declared [`OutputKind`]. Pure — no process.
pub fn parse_output(kind: OutputKind, stdout: &str) -> Result<PluginOutput, PluginError> {
    match kind {
        OutputKind::Json => {
            let trimmed = stdout.trim();
            if trimmed.is_empty() {
                return Err(PluginError::Empty);
            }
            serde_json::from_str::<Value>(trimmed)
                .map(PluginOutput::Json)
                .map_err(|e| PluginError::Parse(e.to_string()))
        }
        OutputKind::Jsonl => {
            let mut frames = Vec::new();
            for (i, line) in stdout.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let value = serde_json::from_str::<Value>(trimmed)
                    .map_err(|e| PluginError::Parse(format!("line {}: {e}", i + 1)))?;
                frames.push(value);
            }
            if frames.is_empty() {
                return Err(PluginError::Empty);
            }
            Ok(PluginOutput::Jsonl(frames))
        }
    }
}

/// How a plugin is executed. Decouples the launch/transport mechanism from the
/// manifest + output contract so additional transports (cross-language IPC, a
/// C-ABI shared library, an already-running service) can be added as new `impl`s
/// without changing callers, the manifest, or the player.
pub trait PluginTransport {
    /// Stable transport id (for discovery / logging), e.g. `"process"`.
    fn id(&self) -> &str;

    /// Run the plugin, optionally handing it a JSON `input` spec, and return the
    /// parsed output. Implementations must not panic on plugin misbehavior —
    /// every failure is a [`PluginError`].
    fn run(&self, manifest: &PluginManifest, input: Option<&Value>)
        -> Result<PluginRun, PluginError>;
}

/// The built-in transport: spawn a child process. Streams an optional JSON spec
/// to the child's stdin (closing the pipe to signal EOF), drains stdout/stderr
/// on threads (so a full pipe can't deadlock), and enforces `run.timeout_ms`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessTransport;

impl PluginTransport for ProcessTransport {
    fn id(&self) -> &str {
        "process"
    }

    fn run(
        &self,
        manifest: &PluginManifest,
        input: Option<&Value>,
    ) -> Result<PluginRun, PluginError> {
        run_process(manifest, input)
    }
}

/// Wait for `child` for at most `dur`; kill it and return `Ok(None)` on timeout.
fn wait_with_timeout(child: &mut Child, dur: Duration) -> Result<Option<ExitStatus>, PluginError> {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) => {}
            Err(e) => return Err(PluginError::Spawn(e.to_string())),
        }
        if start.elapsed() >= dur {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn run_process(manifest: &PluginManifest, input: Option<&Value>) -> Result<PluginRun, PluginError> {
    let mut command = Command::new(&manifest.run.command);
    command.args(&manifest.run.args);
    if let Some(cwd) = &manifest.run.cwd {
        command.current_dir(cwd);
    }
    for (key, value) in &manifest.run.env {
        command.env(key, value);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    command.stdin(if input.is_some() { Stdio::piped() } else { Stdio::null() });

    let mut child = command
        .spawn()
        .map_err(|e| PluginError::Spawn(format!("{}: {e}", manifest.run.command)))?;

    // Hand the plugin its JSON spec on stdin, then close the pipe (EOF).
    if let Some(value) = input {
        if let Some(mut stdin) = child.stdin.take() {
            let mut bytes = serde_json::to_vec(value).unwrap_or_default();
            bytes.push(b'\n');
            let _ = stdin.write_all(&bytes);
        }
    }

    // Drain stdout/stderr on threads so a full pipe buffer can't deadlock us.
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let out_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(p) = stdout_pipe.as_mut() {
            let _ = p.read_to_end(&mut buf);
        }
        buf
    });
    let err_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(p) = stderr_pipe.as_mut() {
            let _ = p.read_to_end(&mut buf);
        }
        buf
    });

    let status = match manifest.run.timeout_ms.map(Duration::from_millis) {
        Some(dur) => match wait_with_timeout(&mut child, dur)? {
            Some(status) => status,
            None => {
                // The child was killed. We deliberately do NOT join the reader
                // threads: a grandchild (e.g. a `sleep` under `sh -c`) can inherit
                // the stdout pipe and keep it open after the parent dies, which
                // would block `read_to_end` until that grandchild exits. Detach
                // them so the host returns immediately on a hung plugin.
                drop(out_handle);
                drop(err_handle);
                return Err(PluginError::Timeout { after_ms: dur.as_millis() as u64 });
            }
        },
        None => child.wait().map_err(|e| PluginError::Spawn(e.to_string()))?,
    };

    let stdout_bytes = out_handle.join().unwrap_or_default();
    let stderr_bytes = err_handle.join().unwrap_or_default();
    let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();

    if !status.success() {
        return Err(PluginError::NonZeroExit { code: status.code(), stderr });
    }

    let stdout = String::from_utf8_lossy(&stdout_bytes);
    let parsed = parse_output(manifest.output, &stdout)?;
    Ok(PluginRun {
        plugin_id: manifest.id.clone(),
        output: parsed,
        exit_code: status.code(),
        stderr,
    })
}

/// Spawn the plugin program (via [`ProcessTransport`]), wait for it (enforcing
/// `run.timeout_ms` if set), and parse stdout.
pub fn run_plugin(manifest: &PluginManifest) -> Result<PluginRun, PluginError> {
    ProcessTransport.run(manifest, None)
}

/// Like [`run_plugin`], but hands the plugin a JSON `input` spec on stdin — the
/// host-→plugin parameter channel (e.g. the model spec to run).
pub fn run_plugin_with_input(
    manifest: &PluginManifest,
    input: &Value,
) -> Result<PluginRun, PluginError> {
    ProcessTransport.run(manifest, Some(input))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::plugin::manifest::{PlayerKind, RunSpec};

    #[test]
    fn parses_single_json_document() {
        let out = parse_output(OutputKind::Json, "  {\"objective\": 42, \"x\": [1,2]}  ").unwrap();
        match out {
            PluginOutput::Json(v) => assert_eq!(v["objective"], serde_json::json!(42)),
            _ => panic!("expected Json"),
        }
    }

    #[test]
    fn parses_jsonl_frames_skipping_blank_lines() {
        let text = "{\"t\":0,\"n\":1}\n\n{\"t\":1,\"n\":2}\n";
        let out = parse_output(OutputKind::Jsonl, text).unwrap();
        assert_eq!(out.frame_count(), 2);
    }

    #[test]
    fn reports_parse_error_with_line_number() {
        let err = parse_output(OutputKind::Jsonl, "{\"ok\":1}\nnot json\n").unwrap_err();
        match err {
            PluginError::Parse(m) => assert!(m.contains("line 2"), "{m}"),
            _ => panic!("expected Parse error"),
        }
    }

    #[test]
    fn empty_output_is_an_error() {
        assert!(matches!(parse_output(OutputKind::Json, "   "), Err(PluginError::Empty)));
        assert!(matches!(parse_output(OutputKind::Jsonl, "\n\n"), Err(PluginError::Empty)));
    }

    #[cfg(unix)]
    #[test]
    fn spawns_an_external_program_and_captures_jsonl() {
        // The runner is language-agnostic; a Rust plugin is the intended use,
        // but any program that writes JSONL works. Use `sh -c printf` here so
        // the spawn→capture→parse path is exercised without a build step.
        let manifest = PluginManifest {
            id: "echo".to_string(),
            name: "Echo".to_string(),
            version: String::new(),
            description: String::new(),
            run: RunSpec::new(
                "sh",
                &["-c", "printf '{\"t\":0,\"n\":1}\\n{\"t\":1,\"n\":3}\\n'"],
            ),
            output: OutputKind::Jsonl,
            player: PlayerKind::Sim,
            controls: Vec::new(),
            title: None,
        };
        let run = run_plugin(&manifest).expect("plugin should run");
        assert_eq!(run.output.frame_count(), 2);
        assert_eq!(run.exit_code, Some(0));
    }

    #[test]
    fn plugin_error_is_serializable_for_http() {
        let e = PluginError::Timeout { after_ms: 250 };
        let v: serde_json::Value = serde_json::to_value(&e).unwrap();
        assert_eq!(v["error"], serde_json::json!("timeout"));
        assert_eq!(v["detail"]["afterMs"], serde_json::json!(250));
        // The newtype variants must serialize too (adjacent tagging supports all
        // variant shapes; an internal tag would fail on these at runtime).
        let s = serde_json::to_value(PluginError::Spawn("x".into())).unwrap();
        assert_eq!(s["error"], serde_json::json!("spawn"));
    }

    #[cfg(unix)]
    #[test]
    fn enforces_timeout_and_kills_a_hung_plugin() {
        let mut manifest = PluginManifest {
            id: "hang".to_string(),
            name: "Hang".to_string(),
            version: String::new(),
            description: String::new(),
            run: RunSpec::new("sh", &["-c", "sleep 5; echo '{}'"]),
            output: OutputKind::Json,
            player: PlayerKind::Results,
            controls: Vec::new(),
            title: None,
        };
        manifest.run.timeout_ms = Some(150);
        let start = std::time::Instant::now();
        match run_plugin(&manifest) {
            Err(PluginError::Timeout { after_ms }) => assert_eq!(after_ms, 150),
            other => panic!("expected Timeout, got {other:?}"),
        }
        // It returned promptly (killed at ~150ms), not after the 5s sleep.
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
    }

    #[cfg(unix)]
    #[test]
    fn passes_an_input_spec_to_the_plugin_stdin() {
        // `cat` echoes stdin → stdout; prove the host→plugin param channel works.
        let manifest = PluginManifest {
            id: "cat".to_string(),
            name: "Cat".to_string(),
            version: String::new(),
            description: String::new(),
            run: RunSpec::new("cat", &[]),
            output: OutputKind::Json,
            player: PlayerKind::Results,
            controls: Vec::new(),
            title: None,
        };
        let run = run_plugin_with_input(&manifest, &serde_json::json!({ "hello": 42 }))
            .expect("cat echoes the spec");
        match run.output {
            PluginOutput::Json(v) => assert_eq!(v["hello"], serde_json::json!(42)),
            other => panic!("expected Json, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn nonzero_exit_is_surfaced() {
        let manifest = PluginManifest {
            id: "fail".to_string(),
            name: "Fail".to_string(),
            version: String::new(),
            description: String::new(),
            run: RunSpec::new("sh", &["-c", "echo oops >&2; exit 3"]),
            output: OutputKind::Json,
            player: PlayerKind::Results,
            controls: Vec::new(),
            title: None,
        };
        match run_plugin(&manifest) {
            Err(PluginError::NonZeroExit { code, stderr }) => {
                assert_eq!(code, Some(3));
                assert!(stderr.contains("oops"));
            }
            other => panic!("expected NonZeroExit, got {other:?}"),
        }
    }
}
