//! Rust-facing bridge for external/reference TSP solvers.
//!
//! The checked-in Python bridge (`scripts/tsp_reference.py`) computes an exact
//! Held-Karp reference for small dense TSPs and records OR-Tools Routing's
//! one-vehicle TSP result when OR-Tools is available locally.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalTspReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalTspReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalTspReferenceSolver::Auto => "auto",
            ExternalTspReferenceSolver::OrTools => "ortools",
            ExternalTspReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalTspReferenceOptions {
    pub solver: ExternalTspReferenceSolver,
}

impl Default for ExternalTspReferenceOptions {
    fn default() -> Self {
        ExternalTspReferenceOptions {
            solver: ExternalTspReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalTspReferenceStatus {
    Optimal,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalTspReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalTspReferenceStatus::Optimal => "optimal",
            ExternalTspReferenceStatus::Infeasible => "infeasible",
            ExternalTspReferenceStatus::Unsupported => "unsupported",
            ExternalTspReferenceStatus::NumericalError => "numerical-error",
            ExternalTspReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalTspPoint {
    pub id: Option<String>,
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalTspReferenceSolution {
    pub status: ExternalTspReferenceStatus,
    pub solver: String,
    pub tour: Vec<usize>,
    pub objective: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_tour: Vec<usize>,
    pub ortools_objective: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct TspReferencePayload {
    status: String,
    solver: Option<String>,
    tour: Option<Vec<usize>>,
    objective: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsTour")]
    ortools_tour: Option<Vec<usize>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalTspReferenceStatus {
    match status {
        "optimal" => ExternalTspReferenceStatus::Optimal,
        "infeasible" => ExternalTspReferenceStatus::Infeasible,
        "unsupported" => ExternalTspReferenceStatus::Unsupported,
        "unavailable" => ExternalTspReferenceStatus::Unavailable,
        _ => ExternalTspReferenceStatus::NumericalError,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalTspReferenceSolution {
    ExternalTspReferenceSolution {
        status: ExternalTspReferenceStatus::Unavailable,
        solver: "external-tsp-reference".to_string(),
        tour: Vec::new(),
        objective: None,
        ortools_status: None,
        ortools_tour: Vec::new(),
        ortools_objective: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(message: impl Into<String>, elapsed_ms: f64) -> ExternalTspReferenceSolution {
    ExternalTspReferenceSolution {
        status: ExternalTspReferenceStatus::NumericalError,
        solver: "external-tsp-reference".to_string(),
        tour: Vec::new(),
        objective: None,
        ortools_status: None,
        ortools_tour: Vec::new(),
        ortools_objective: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("tsp_reference.py")
}

fn run_tsp_reference_json(
    payload: Value,
    opts: &ExternalTspReferenceOptions,
) -> ExternalTspReferenceSolution {
    let started = Instant::now();
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command
        .arg(reference_script())
        .arg("--solver")
        .arg(opts.solver.as_arg());
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return unavailable(
                format!("failed to start tsp_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write tsp_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for tsp_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<TspReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalTspReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-tsp-reference".to_string()),
            tour: parsed.tour.unwrap_or_default(),
            objective: parsed.objective,
            ortools_status: parsed.ortools_status,
            ortools_tour: parsed.ortools_tour.unwrap_or_default(),
            ortools_objective: parsed.ortools_objective,
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            }),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse tsp_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_tsp_with_external_reference(
    distance_matrix: &[Vec<f64>],
    opts: &ExternalTspReferenceOptions,
) -> ExternalTspReferenceSolution {
    run_tsp_reference_json(
        json!({
            "distanceMatrix": distance_matrix,
        }),
        opts,
    )
}

pub fn solve_euclidean_tsp_with_external_reference(
    points: &[ExternalTspPoint],
    opts: &ExternalTspReferenceOptions,
) -> ExternalTspReferenceSolution {
    let points_json: Vec<Value> = points
        .iter()
        .enumerate()
        .map(|(idx, point)| {
            json!({
                "id": point.id.clone().unwrap_or_else(|| idx.to_string()),
                "x": point.x,
                "y": point.y,
            })
        })
        .collect();
    run_tsp_reference_json(
        json!({
            "points": points_json,
        }),
        opts,
    )
}
