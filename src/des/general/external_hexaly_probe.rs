//! Small readiness probe for a local Hexaly/LocalSolver installation.
//!
//! Hexaly is license-gated, so finding a `hexaly` binary is weaker evidence
//! than it is for most open-source command-line solvers. This probe runs a
//! tiny HXM model and only reports ready when the executable can actually load
//! the model and solve under the current license configuration.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const HEXALY_SMOKE_MODEL: &str = r#"
function model() {
    x[0..2] <- bool();
    constraint x[0] + x[1] + x[2] >= 1;
    minimize x[0] + 2 * x[1] + 3 * x[2];
}

function param() {
    hxTimeLimit = 1;
}
"#;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalHexalyProbe {
    pub ready: bool,
    pub command: PathBuf,
    pub message: String,
}

pub fn probe_external_hexaly_command(command: &Path, timeout_ms: u64) -> ExternalHexalyProbe {
    let probe_dir = unique_hexaly_probe_dir();
    let model_path = probe_dir.join("des_rs_hexaly_smoke.hxm");
    if let Err(err) =
        fs::create_dir_all(&probe_dir).and_then(|_| fs::write(&model_path, HEXALY_SMOKE_MODEL))
    {
        return ExternalHexalyProbe {
            ready: false,
            command: command.to_path_buf(),
            message: format!("could not create Hexaly smoke model: {err}"),
        };
    }

    let child = Command::new(command)
        .arg(&model_path)
        .arg("hxTimeLimit=1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    let probe = match child {
        Ok(child) => match wait_for_hexaly_output(child, timeout_ms) {
            Ok((output, timed_out)) => {
                classify_hexaly_output(command, output, timed_out, timeout_ms)
            }
            Err(err) => ExternalHexalyProbe {
                ready: false,
                command: command.to_path_buf(),
                message: err,
            },
        },
        Err(err) => ExternalHexalyProbe {
            ready: false,
            command: command.to_path_buf(),
            message: format!(
                "could not launch Hexaly command {}: {err}",
                command.display()
            ),
        },
    };

    let _ = fs::remove_dir_all(&probe_dir);
    probe
}

fn classify_hexaly_output(
    command: &Path,
    output: Output,
    timed_out: bool,
    timeout_ms: u64,
) -> ExternalHexalyProbe {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    let lower = combined.to_ascii_lowercase();
    let solved = lower.contains("optimal solution")
        || lower.contains("feasible solution")
        || lower.contains("solution status")
        || lower.lines().any(|line| {
            let line = line.trim();
            line.starts_with("obj") && line.contains('=')
        });

    if output.status.success() && !timed_out && solved {
        return ExternalHexalyProbe {
            ready: true,
            command: command.to_path_buf(),
            message: format!(
                "Hexaly solved the local HXM smoke model via {}",
                command.display()
            ),
        };
    }

    let mut reason = if timed_out {
        format!("Hexaly smoke model timed out after {timeout_ms}ms")
    } else {
        format!(
            "Hexaly smoke model exited with status {}",
            output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "terminated by signal".to_string())
        )
    };
    let excerpt = short_output_excerpt(&combined);
    if !excerpt.is_empty() {
        reason.push_str(": ");
        reason.push_str(&excerpt);
    }

    ExternalHexalyProbe {
        ready: false,
        command: command.to_path_buf(),
        message: reason,
    }
}

fn wait_for_hexaly_output(
    mut child: std::process::Child,
    timeout_ms: u64,
) -> Result<(Output, bool), String> {
    let started = Instant::now();
    let timeout = Duration::from_millis(timeout_ms);
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if timeout_ms > 0 && started.elapsed() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    break;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(err) => return Err(format!("failed to poll Hexaly smoke process: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for Hexaly smoke process: {err}"))
}

fn short_output_excerpt(output: &str) -> String {
    let normalized = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("; ");
    const LIMIT: usize = 320;
    if normalized.chars().count() <= LIMIT {
        normalized
    } else {
        format!("{}...", normalized.chars().take(LIMIT).collect::<String>())
    }
}

fn unique_hexaly_probe_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    env::temp_dir().join(format!(
        "des-rs-hexaly-probe-{}-{nanos}",
        std::process::id()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn hexaly_probe_accepts_successful_smoke_trace() {
        let script = write_probe_script(
            "hexaly-success",
            "printf 'Optimal solution\\n  obj = 1\\n'\n",
        );
        let probe = probe_external_hexaly_command(&script, 5_000);
        assert!(probe.ready, "{}", probe.message);
        assert!(probe.message.contains("solved the local HXM smoke model"));
        let _ = fs::remove_file(script);
    }

    #[test]
    fn hexaly_probe_rejects_license_like_failure() {
        let script = write_probe_script(
            "hexaly-failure",
            "printf 'license file not found\\n' >&2\nexit 3\n",
        );
        let probe = probe_external_hexaly_command(&script, 5_000);
        assert!(!probe.ready);
        assert!(probe.message.contains("license file not found"));
        let _ = fs::remove_file(script);
    }

    fn write_probe_script(name: &str, body: &str) -> PathBuf {
        let path = unique_hexaly_probe_dir().with_file_name(format!(
            "des-rs-{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let script = format!("#!/bin/sh\n{body}");
        fs::write(&path, script).unwrap();
        make_executable(&path).unwrap();
        path
    }

    #[cfg(unix)]
    fn make_executable(path: &Path) -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)
    }

    #[cfg(not(unix))]
    fn make_executable(_path: &Path) -> io::Result<()> {
        Ok(())
    }
}
