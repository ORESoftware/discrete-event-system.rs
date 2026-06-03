//! Rust-facing bridge for external/reference graph-coloring solvers.
//!
//! The Python bridge (`scripts/graph_coloring_reference.py`) computes an exact
//! DSATUR reference and, when installed, solves the same graph-coloring model
//! with OR-Tools CP-SAT.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::graph_coloring::GraphColoringProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalGraphColoringReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalGraphColoringReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalGraphColoringReferenceSolver::Auto => "auto",
            ExternalGraphColoringReferenceSolver::OrTools => "ortools",
            ExternalGraphColoringReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalGraphColoringReferenceOptions {
    pub solver: ExternalGraphColoringReferenceSolver,
}

impl Default for ExternalGraphColoringReferenceOptions {
    fn default() -> Self {
        ExternalGraphColoringReferenceOptions {
            solver: ExternalGraphColoringReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalGraphColoringReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalGraphColoringReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalGraphColoringReferenceStatus::Optimal => "optimal",
            ExternalGraphColoringReferenceStatus::Feasible => "feasible",
            ExternalGraphColoringReferenceStatus::Infeasible => "infeasible",
            ExternalGraphColoringReferenceStatus::Unsupported => "unsupported",
            ExternalGraphColoringReferenceStatus::NumericalError => "numerical-error",
            ExternalGraphColoringReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalGraphColoringReferenceSolution {
    pub status: ExternalGraphColoringReferenceStatus,
    pub solver: String,
    pub color_indices: Vec<usize>,
    pub color_names: Vec<String>,
    pub used_color_count: Option<usize>,
    pub objective: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_color_indices: Vec<usize>,
    pub ortools_color_names: Vec<String>,
    pub ortools_used_color_count: Option<usize>,
    pub ortools_objective: Option<f64>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct GraphColoringReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "colorIndices")]
    color_indices: Option<Vec<usize>>,
    #[serde(rename = "colorNames")]
    color_names: Option<Vec<String>>,
    #[serde(rename = "usedColorCount")]
    used_color_count: Option<usize>,
    objective: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsColorIndices")]
    ortools_color_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsColorNames")]
    ortools_color_names: Option<Vec<String>>,
    #[serde(rename = "ortoolsUsedColorCount")]
    ortools_used_color_count: Option<usize>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalGraphColoringReferenceStatus {
    match status {
        "optimal" => ExternalGraphColoringReferenceStatus::Optimal,
        "feasible" => ExternalGraphColoringReferenceStatus::Feasible,
        "infeasible" => ExternalGraphColoringReferenceStatus::Infeasible,
        "unsupported" => ExternalGraphColoringReferenceStatus::Unsupported,
        "unavailable" => ExternalGraphColoringReferenceStatus::Unavailable,
        _ => ExternalGraphColoringReferenceStatus::NumericalError,
    }
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalGraphColoringReferenceSolution {
    ExternalGraphColoringReferenceSolution {
        status: ExternalGraphColoringReferenceStatus::Unavailable,
        solver: "external-graph-coloring-reference".to_string(),
        color_indices: Vec::new(),
        color_names: Vec::new(),
        used_color_count: None,
        objective: None,
        ortools_status: None,
        ortools_color_indices: Vec::new(),
        ortools_color_names: Vec::new(),
        ortools_used_color_count: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalGraphColoringReferenceSolution {
    ExternalGraphColoringReferenceSolution {
        status: ExternalGraphColoringReferenceStatus::NumericalError,
        solver: "external-graph-coloring-reference".to_string(),
        color_indices: Vec::new(),
        color_names: Vec::new(),
        used_color_count: None,
        objective: None,
        ortools_status: None,
        ortools_color_indices: Vec::new(),
        ortools_color_names: Vec::new(),
        ortools_used_color_count: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("graph_coloring_reference.py")
}

fn run_graph_coloring_reference_json(
    payload: Value,
    opts: &ExternalGraphColoringReferenceOptions,
) -> ExternalGraphColoringReferenceSolution {
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
                format!("failed to start graph_coloring_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write graph_coloring_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for graph_coloring_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<GraphColoringReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalGraphColoringReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-graph-coloring-reference".to_string()),
            color_indices: parsed.color_indices.unwrap_or_default(),
            color_names: parsed.color_names.unwrap_or_default(),
            used_color_count: parsed.used_color_count,
            objective: parsed.objective,
            ortools_status: parsed.ortools_status,
            ortools_color_indices: parsed.ortools_color_indices.unwrap_or_default(),
            ortools_color_names: parsed.ortools_color_names.unwrap_or_default(),
            ortools_used_color_count: parsed.ortools_used_color_count,
            ortools_objective: parsed.ortools_objective,
            ortools_objective_bound: parsed.ortools_objective_bound,
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
                "failed to parse graph_coloring_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_graph_coloring_with_external_reference(
    problem: &GraphColoringProblem,
    opts: &ExternalGraphColoringReferenceOptions,
) -> ExternalGraphColoringReferenceSolution {
    run_graph_coloring_reference_json(
        json!({
            "vertices": &problem.vertices,
            "edges": problem.edges.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
        }),
        opts,
    )
}
