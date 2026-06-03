//! Rust-facing bridge for external/reference minimum spanning tree solvers.
//!
//! The Python bridge (`scripts/minimum_spanning_tree_reference.py`) computes a
//! Kruskal reference and, when installed, solves the same graph with OR-Tools
//! CP-SAT using a root-flow connectivity formulation.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::minimum_spanning_tree::MinimumSpanningTreeProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMinimumSpanningTreeReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalMinimumSpanningTreeReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalMinimumSpanningTreeReferenceSolver::Auto => "auto",
            ExternalMinimumSpanningTreeReferenceSolver::OrTools => "ortools",
            ExternalMinimumSpanningTreeReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalMinimumSpanningTreeReferenceOptions {
    pub solver: ExternalMinimumSpanningTreeReferenceSolver,
}

impl Default for ExternalMinimumSpanningTreeReferenceOptions {
    fn default() -> Self {
        ExternalMinimumSpanningTreeReferenceOptions {
            solver: ExternalMinimumSpanningTreeReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMinimumSpanningTreeReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    NumericalError,
    Unavailable,
}

impl ExternalMinimumSpanningTreeReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalMinimumSpanningTreeReferenceStatus::Optimal => "optimal",
            ExternalMinimumSpanningTreeReferenceStatus::Feasible => "feasible",
            ExternalMinimumSpanningTreeReferenceStatus::Infeasible => "infeasible",
            ExternalMinimumSpanningTreeReferenceStatus::NumericalError => "numerical-error",
            ExternalMinimumSpanningTreeReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalMinimumSpanningTreeReferenceSolution {
    pub status: ExternalMinimumSpanningTreeReferenceStatus,
    pub solver: String,
    pub selected_edge_indices: Vec<usize>,
    pub selected_edge_ids: Vec<String>,
    pub objective: Option<f64>,
    pub total_weight: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_selected_edge_indices: Vec<usize>,
    pub ortools_selected_edge_ids: Vec<String>,
    pub ortools_objective: Option<f64>,
    pub ortools_total_weight: Option<f64>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct MinimumSpanningTreeReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "selectedEdgeIndices")]
    selected_edge_indices: Option<Vec<usize>>,
    #[serde(rename = "selectedEdgeIds")]
    selected_edge_ids: Option<Vec<String>>,
    objective: Option<f64>,
    #[serde(rename = "totalWeight")]
    total_weight: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsSelectedEdgeIndices")]
    ortools_selected_edge_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsSelectedEdgeIds")]
    ortools_selected_edge_ids: Option<Vec<String>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsTotalWeight")]
    ortools_total_weight: Option<f64>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalMinimumSpanningTreeReferenceStatus {
    match status {
        "optimal" => ExternalMinimumSpanningTreeReferenceStatus::Optimal,
        "feasible" => ExternalMinimumSpanningTreeReferenceStatus::Feasible,
        "infeasible" => ExternalMinimumSpanningTreeReferenceStatus::Infeasible,
        "unavailable" => ExternalMinimumSpanningTreeReferenceStatus::Unavailable,
        _ => ExternalMinimumSpanningTreeReferenceStatus::NumericalError,
    }
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMinimumSpanningTreeReferenceSolution {
    ExternalMinimumSpanningTreeReferenceSolution {
        status: ExternalMinimumSpanningTreeReferenceStatus::Unavailable,
        solver: "external-minimum-spanning-tree-reference".to_string(),
        selected_edge_indices: Vec::new(),
        selected_edge_ids: Vec::new(),
        objective: None,
        total_weight: None,
        ortools_status: None,
        ortools_selected_edge_indices: Vec::new(),
        ortools_selected_edge_ids: Vec::new(),
        ortools_objective: None,
        ortools_total_weight: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMinimumSpanningTreeReferenceSolution {
    ExternalMinimumSpanningTreeReferenceSolution {
        status: ExternalMinimumSpanningTreeReferenceStatus::NumericalError,
        solver: "external-minimum-spanning-tree-reference".to_string(),
        selected_edge_indices: Vec::new(),
        selected_edge_ids: Vec::new(),
        objective: None,
        total_weight: None,
        ortools_status: None,
        ortools_selected_edge_indices: Vec::new(),
        ortools_selected_edge_ids: Vec::new(),
        ortools_objective: None,
        ortools_total_weight: None,
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
        .join("minimum_spanning_tree_reference.py")
}

fn run_minimum_spanning_tree_reference_json(
    payload: Value,
    opts: &ExternalMinimumSpanningTreeReferenceOptions,
) -> ExternalMinimumSpanningTreeReferenceSolution {
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
                format!("failed to start minimum_spanning_tree_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write minimum_spanning_tree_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for minimum_spanning_tree_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<MinimumSpanningTreeReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalMinimumSpanningTreeReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-minimum-spanning-tree-reference".to_string()),
            selected_edge_indices: parsed.selected_edge_indices.unwrap_or_default(),
            selected_edge_ids: parsed.selected_edge_ids.unwrap_or_default(),
            objective: parsed.objective,
            total_weight: parsed.total_weight,
            ortools_status: parsed.ortools_status,
            ortools_selected_edge_indices: parsed.ortools_selected_edge_indices.unwrap_or_default(),
            ortools_selected_edge_ids: parsed.ortools_selected_edge_ids.unwrap_or_default(),
            ortools_objective: parsed.ortools_objective,
            ortools_total_weight: parsed.ortools_total_weight,
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
                "failed to parse minimum_spanning_tree_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_minimum_spanning_tree_with_external_reference(
    problem: &MinimumSpanningTreeProblem,
    opts: &ExternalMinimumSpanningTreeReferenceOptions,
) -> ExternalMinimumSpanningTreeReferenceSolution {
    run_minimum_spanning_tree_reference_json(
        json!({
            "vertices": &problem.vertices,
            "edges": problem.edges.iter().map(|edge| json!({
                "id": edge.id,
                "from": edge.from,
                "to": edge.to,
                "weight": edge.weight,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}
