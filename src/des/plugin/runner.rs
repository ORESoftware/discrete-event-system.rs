//! Run an external plugin program and capture its JSON / JSONL output.
//!
//! The runner is **language-agnostic** — it spawns whatever `run.command`
//! names and reads stdout. "Rust for now" only means the example plugins are
//! Rust binaries; the host neither knows nor cares. Parsing is split out
//! ([`parse_output`]) so it can be tested without spawning a process.

use std::process::Command;

use serde_json::Value;

use super::manifest::{OutputKind, PluginManifest};

/// Why a plugin run failed. Recoverable (returned, never panicked).
#[derive(Debug)]
pub enum PluginError {
    /// The process could not be spawned (command missing, not executable, …).
    Spawn(String),
    /// The process ran but exited non-zero.
    NonZeroExit { code: Option<i32>, stderr: String },
    /// stdout was not valid JSON / JSONL.
    Parse(String),
    /// The program produced no output.
    Empty,
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

/// Spawn the plugin program, wait for it, and parse stdout.
///
/// Note: synchronous (`Command::output`) — `run.timeout_ms` is advisory and not
/// yet enforced. stderr is captured and surfaced on non-zero exit.
pub fn run_plugin(manifest: &PluginManifest) -> Result<PluginRun, PluginError> {
    let mut command = Command::new(&manifest.run.command);
    command.args(&manifest.run.args);
    if let Some(cwd) = &manifest.run.cwd {
        command.current_dir(cwd);
    }
    for (key, value) in &manifest.run.env {
        command.env(key, value);
    }

    let output = command
        .output()
        .map_err(|e| PluginError::Spawn(format!("{}: {e}", manifest.run.command)))?;

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        return Err(PluginError::NonZeroExit {
            code: output.status.code(),
            stderr,
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = parse_output(manifest.output, &stdout)?;
    Ok(PluginRun {
        plugin_id: manifest.id.clone(),
        output: parsed,
        exit_code: output.status.code(),
        stderr,
    })
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
