//! Rust-facing bridge for external/reference weighted independent-set solvers.
//!
//! The Python bridge (`scripts/weighted_independent_set_reference.py`) computes
//! a deterministic exact branch-and-bound reference and, when installed, solves
//! the same conflict graph with OR-Tools CP-SAT.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::weighted_independent_set::WeightedIndependentSetProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalWeightedIndependentSetReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalWeightedIndependentSetReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalWeightedIndependentSetReferenceSolver::Auto => "auto",
            ExternalWeightedIndependentSetReferenceSolver::OrTools => "ortools",
            ExternalWeightedIndependentSetReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalWeightedIndependentSetReferenceOptions {
    pub solver: ExternalWeightedIndependentSetReferenceSolver,
}

impl Default for ExternalWeightedIndependentSetReferenceOptions {
    fn default() -> Self {
        ExternalWeightedIndependentSetReferenceOptions {
            solver: ExternalWeightedIndependentSetReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalWeightedIndependentSetReferenceStatus {
    Optimal,
    Feasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalWeightedIndependentSetReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalWeightedIndependentSetReferenceStatus::Optimal => "optimal",
            ExternalWeightedIndependentSetReferenceStatus::Feasible => "feasible",
            ExternalWeightedIndependentSetReferenceStatus::Unsupported => "unsupported",
            ExternalWeightedIndependentSetReferenceStatus::NumericalError => "numerical-error",
            ExternalWeightedIndependentSetReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalWeightedIndependentSetReferenceSolution {
    pub status: ExternalWeightedIndependentSetReferenceStatus,
    pub solver: String,
    pub selected_vertex_indices: Vec<usize>,
    pub selected_vertex_ids: Vec<String>,
    pub total_weight: Option<f64>,
    pub objective: Option<f64>,
    pub upper_bound: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_selected_vertex_indices: Vec<usize>,
    pub ortools_selected_vertex_ids: Vec<String>,
    pub ortools_total_weight: Option<f64>,
    pub ortools_objective: Option<f64>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct WeightedIndependentSetReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "selectedVertexIndices")]
    selected_vertex_indices: Option<Vec<usize>>,
    #[serde(rename = "selectedVertexIds")]
    selected_vertex_ids: Option<Vec<String>>,
    #[serde(rename = "totalWeight")]
    total_weight: Option<f64>,
    objective: Option<f64>,
    #[serde(rename = "upperBound")]
    upper_bound: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsSelectedVertexIndices")]
    ortools_selected_vertex_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsSelectedVertexIds")]
    ortools_selected_vertex_ids: Option<Vec<String>>,
    #[serde(rename = "ortoolsTotalWeight")]
    ortools_total_weight: Option<f64>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalWeightedIndependentSetReferenceStatus {
    match status {
        "optimal" => ExternalWeightedIndependentSetReferenceStatus::Optimal,
        "feasible" => ExternalWeightedIndependentSetReferenceStatus::Feasible,
        "unsupported" => ExternalWeightedIndependentSetReferenceStatus::Unsupported,
        "unavailable" => ExternalWeightedIndependentSetReferenceStatus::Unavailable,
        _ => ExternalWeightedIndependentSetReferenceStatus::NumericalError,
    }
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedIndependentSetReferenceSolution {
    ExternalWeightedIndependentSetReferenceSolution {
        status: ExternalWeightedIndependentSetReferenceStatus::Unavailable,
        solver: "external-weighted-independent-set-reference".to_string(),
        selected_vertex_indices: Vec::new(),
        selected_vertex_ids: Vec::new(),
        total_weight: None,
        objective: None,
        upper_bound: None,
        ortools_status: None,
        ortools_selected_vertex_indices: Vec::new(),
        ortools_selected_vertex_ids: Vec::new(),
        ortools_total_weight: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedIndependentSetReferenceSolution {
    ExternalWeightedIndependentSetReferenceSolution {
        status: ExternalWeightedIndependentSetReferenceStatus::NumericalError,
        solver: "external-weighted-independent-set-reference".to_string(),
        selected_vertex_indices: Vec::new(),
        selected_vertex_ids: Vec::new(),
        total_weight: None,
        objective: None,
        upper_bound: None,
        ortools_status: None,
        ortools_selected_vertex_indices: Vec::new(),
        ortools_selected_vertex_ids: Vec::new(),
        ortools_total_weight: None,
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
    root.join("scripts")
        .join("weighted_independent_set_reference.py")
}

fn run_weighted_independent_set_reference_json(
    payload: Value,
    opts: &ExternalWeightedIndependentSetReferenceOptions,
) -> ExternalWeightedIndependentSetReferenceSolution {
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
                format!(
                    "failed to start weighted_independent_set_reference.py with {python}: {err}"
                ),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write weighted_independent_set_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for weighted_independent_set_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<WeightedIndependentSetReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalWeightedIndependentSetReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-weighted-independent-set-reference".to_string()),
            selected_vertex_indices: parsed.selected_vertex_indices.unwrap_or_default(),
            selected_vertex_ids: parsed.selected_vertex_ids.unwrap_or_default(),
            total_weight: parsed.total_weight,
            objective: parsed.objective,
            upper_bound: parsed.upper_bound,
            ortools_status: parsed.ortools_status,
            ortools_selected_vertex_indices: parsed
                .ortools_selected_vertex_indices
                .unwrap_or_default(),
            ortools_selected_vertex_ids: parsed.ortools_selected_vertex_ids.unwrap_or_default(),
            ortools_total_weight: parsed.ortools_total_weight,
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
                "failed to parse weighted_independent_set_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_weighted_independent_set_with_external_reference(
    problem: &WeightedIndependentSetProblem,
    opts: &ExternalWeightedIndependentSetReferenceOptions,
) -> ExternalWeightedIndependentSetReferenceSolution {
    run_weighted_independent_set_reference_json(
        json!({
            "vertices": problem.vertices.iter().map(|vertex| json!({
                "id": &vertex.id,
                "weight": vertex.weight,
            })).collect::<Vec<_>>(),
            "edges": problem.edges.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
        }),
        opts,
    )
}
