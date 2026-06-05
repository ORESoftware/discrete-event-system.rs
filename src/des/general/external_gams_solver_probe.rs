//! Small GAMS-backed smoke probes for solver libraries exposed through a local
//! GAMS installation.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalGamsSolver {
    Knitro,
    Mosek,
}

impl ExternalGamsSolver {
    fn option_pair(self) -> (&'static str, &'static str) {
        match self {
            ExternalGamsSolver::Knitro => ("nlp", "knitro"),
            ExternalGamsSolver::Mosek => ("lp", "mosek"),
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            ExternalGamsSolver::Knitro => "Knitro",
            ExternalGamsSolver::Mosek => "MOSEK",
        }
    }

    fn model_family(self) -> &'static str {
        match self {
            ExternalGamsSolver::Knitro => "NLP",
            ExternalGamsSolver::Mosek => "LP",
        }
    }

    fn model_text(self) -> &'static str {
        match self {
            ExternalGamsSolver::Knitro => {
                r#"Variables x, y, z;
Positive Variables x, y;
Equations obj, c1;
obj.. z =e= sqr(x - 1) + sqr(y - 2);
c1.. x + y =g= 1;
Model m /all/;
x.l = 0.5; y.l = 1.5;
Solve m using NLP minimizing z;
"#
            }
            ExternalGamsSolver::Mosek => {
                r#"Variables x, y, z;
Positive Variables x, y;
Equations obj, c1, c2;
obj.. z =e= 3*x + 2*y;
c1.. x + y =l= 4;
c2.. x =l= 2;
Model m /all/;
Solve m using LP maximizing z;
"#
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalGamsSolverProbe {
    pub solver: ExternalGamsSolver,
    pub command: Option<PathBuf>,
    pub ready: bool,
    pub message: String,
}

pub fn probe_external_gams_solver(
    solver: ExternalGamsSolver,
    timeout_ms: u64,
) -> ExternalGamsSolverProbe {
    let Some(command) = find_external_gams_command() else {
        return ExternalGamsSolverProbe {
            solver,
            command: None,
            ready: false,
            message: "GAMS executable not found via ORES_GAMS_CMD, GAMS_CMD, GAMS dirs, or PATH"
                .to_string(),
        };
    };

    let stem = gams_probe_temp_stem(solver);
    let model_path = env::temp_dir().join(format!("{stem}.gms"));
    let listing_path = env::temp_dir().join(format!("{stem}.lst"));
    let cleanup_paths = [model_path.clone(), listing_path.clone()];

    if let Err(err) = fs::write(&model_path, solver.model_text()) {
        cleanup_gams_probe_files(&cleanup_paths);
        return ExternalGamsSolverProbe {
            solver,
            command: Some(command),
            ready: false,
            message: format!("failed to write GAMS smoke model: {err}"),
        };
    }

    let (option_name, option_value) = solver.option_pair();
    let mut process = Command::new(&command);
    process
        .arg(&model_path)
        .arg(format!("{option_name}={option_value}"))
        .arg("lo=0")
        .arg(format!("o={}", listing_path.display()))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(parent) = command.parent() {
        process.current_dir(parent);
    }

    let output = match process.spawn() {
        Ok(child) => wait_for_external_gams_output(child, timeout_ms),
        Err(err) => Err(format!("failed to start GAMS executable: {err}")),
    };
    let output = match output {
        Ok((output, timed_out)) => {
            if timed_out {
                cleanup_gams_probe_files(&cleanup_paths);
                return ExternalGamsSolverProbe {
                    solver,
                    command: Some(command),
                    ready: false,
                    message: format!("GAMS {} probe timed out", solver.display_name()),
                };
            }
            output
        }
        Err(err) => {
            cleanup_gams_probe_files(&cleanup_paths);
            return ExternalGamsSolverProbe {
                solver,
                command: Some(command),
                ready: false,
                message: err,
            };
        }
    };

    let listing = fs::read_to_string(&listing_path).unwrap_or_default();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let ready = output.status.success() && external_gams_listing_confirms_solver(solver, &listing);
    let message = if ready {
        format!(
            "{} solved the local GAMS {} smoke model",
            solver.display_name(),
            solver.model_family()
        )
    } else {
        let detail = first_non_empty_probe_detail(&listing, &stdout, &stderr);
        if detail.is_empty() {
            format!(
                "GAMS {} smoke model did not confirm solver readiness",
                solver.display_name()
            )
        } else {
            format!(
                "GAMS {} smoke model did not confirm solver readiness: {detail}",
                solver.display_name()
            )
        }
    };
    cleanup_gams_probe_files(&cleanup_paths);

    ExternalGamsSolverProbe {
        solver,
        command: Some(command),
        ready,
        message,
    }
}

pub fn find_external_gams_command() -> Option<PathBuf> {
    find_command_from_env(&["ORES_GAMS_CMD", "GAMS_CMD"])
        .or_else(|| find_gams_in_dirs(&["ORES_GAMS_DIR", "GAMS_HOME", "GAMS_DIR", "GAMSDIR"]))
        .or_else(|| find_command_in_path("gams"))
}

fn find_command_from_env(names: &[&str]) -> Option<PathBuf> {
    names.iter().find_map(|name| {
        env::var_os(name).and_then(|value| {
            if value.is_empty() {
                None
            } else {
                resolve_command_candidate(PathBuf::from(value))
            }
        })
    })
}

fn find_gams_in_dirs(names: &[&str]) -> Option<PathBuf> {
    names.iter().find_map(|name| {
        env::var_os(name).and_then(|value| {
            if value.is_empty() {
                None
            } else {
                find_gams_in_dir(Path::new(&value))
            }
        })
    })
}

fn resolve_command_candidate(candidate: PathBuf) -> Option<PathBuf> {
    if candidate.components().count() > 1 || candidate.is_absolute() {
        if command_file_exists(&candidate) {
            Some(candidate)
        } else {
            None
        }
    } else {
        candidate
            .file_name()
            .and_then(OsStr::to_str)
            .and_then(find_command_in_path)
    }
}

fn find_gams_in_dir(dir: &Path) -> Option<PathBuf> {
    [
        dir.join("gams"),
        dir.join("bin").join("gams"),
        dir.join("Resources").join("gams"),
    ]
    .into_iter()
    .find(|path| command_file_exists(path))
}

fn find_command_in_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|dir| dir.join(name))
            .find(|candidate| command_file_exists(candidate))
    })
}

fn command_file_exists(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|metadata| metadata.is_file())
}

pub fn wait_for_external_gams_output(
    mut child: Child,
    timeout_ms: u64,
) -> Result<(Output, bool), String> {
    let started = SystemTime::now();
    let timeout = Duration::from_millis(timeout_ms);
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if timeout_ms > 0 && started.elapsed().unwrap_or_default() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    break;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(err) => return Err(format!("failed to poll GAMS probe: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for GAMS probe: {err}"))
}

pub fn external_gams_listing_confirms_solver(solver: ExternalGamsSolver, listing: &str) -> bool {
    let upper = listing.to_ascii_uppercase();
    let (_, solver_option) = solver.option_pair();
    let solver_selected = upper
        .contains(&format!("SOLVER  {}", solver_option.to_ascii_uppercase()))
        || upper.contains(&format!("SOLVER {}", solver_option.to_ascii_uppercase()));
    solver_selected
        && upper.contains("NORMAL COMPLETION")
        && (upper.contains("MODEL STATUS      1 OPTIMAL")
            || upper.contains("MODEL STATUS      2 LOCALLY OPTIMAL")
            || upper.contains("OPTIMAL SOLUTION FOUND"))
}

fn first_non_empty_probe_detail(listing: &str, stdout: &str, stderr: &str) -> String {
    for text in [listing, stderr, stdout] {
        if let Some(line) = text.lines().map(str::trim).find(|line| !line.is_empty()) {
            return line.chars().take(240).collect();
        }
    }
    String::new()
}

fn gams_probe_temp_stem(solver: ExternalGamsSolver) -> String {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "des-rs-gams-{}-{}-{suffix}",
        solver.option_pair().1,
        std::process::id()
    )
}

fn cleanup_gams_probe_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}
