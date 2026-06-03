//! Rust-facing bridge for external/reference set-cover solvers.
//!
//! The Python bridge (`scripts/set_cover_reference.py`) computes an exact
//! small-instance reference and, when installed, solves the same weighted set
//! cover with OR-Tools CP-SAT.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::set_cover::SetCoverProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalSetCoverReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalSetCoverReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalSetCoverReferenceSolver::Auto => "auto",
            ExternalSetCoverReferenceSolver::OrTools => "ortools",
            ExternalSetCoverReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalSetCoverReferenceOptions {
    pub solver: ExternalSetCoverReferenceSolver,
}

impl Default for ExternalSetCoverReferenceOptions {
    fn default() -> Self {
        ExternalSetCoverReferenceOptions {
            solver: ExternalSetCoverReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalSetCoverReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalSetCoverReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalSetCoverReferenceStatus::Optimal => "optimal",
            ExternalSetCoverReferenceStatus::Feasible => "feasible",
            ExternalSetCoverReferenceStatus::Infeasible => "infeasible",
            ExternalSetCoverReferenceStatus::Unsupported => "unsupported",
            ExternalSetCoverReferenceStatus::NumericalError => "numerical-error",
            ExternalSetCoverReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalSetCoverReferenceSolution {
    pub status: ExternalSetCoverReferenceStatus,
    pub solver: String,
    pub selected_set_indices: Vec<usize>,
    pub selected_set_ids: Vec<String>,
    pub objective: Option<f64>,
    pub covered_elements: Vec<String>,
    pub ortools_status: Option<String>,
    pub ortools_selected_set_indices: Vec<usize>,
    pub ortools_selected_set_ids: Vec<String>,
    pub ortools_objective: Option<f64>,
    pub ortools_covered_elements: Vec<String>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct SetCoverReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "selectedSetIndices")]
    selected_set_indices: Option<Vec<usize>>,
    #[serde(rename = "selectedSets")]
    selected_sets: Option<Vec<String>>,
    objective: Option<f64>,
    #[serde(rename = "coveredElements")]
    covered_elements: Option<Vec<String>>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsSelectedSetIndices")]
    ortools_selected_set_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsSelectedSets")]
    ortools_selected_sets: Option<Vec<String>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsCoveredElements")]
    ortools_covered_elements: Option<Vec<String>>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalSetCoverReferenceStatus {
    match status {
        "optimal" => ExternalSetCoverReferenceStatus::Optimal,
        "feasible" => ExternalSetCoverReferenceStatus::Feasible,
        "infeasible" => ExternalSetCoverReferenceStatus::Infeasible,
        "unsupported" => ExternalSetCoverReferenceStatus::Unsupported,
        "unavailable" => ExternalSetCoverReferenceStatus::Unavailable,
        _ => ExternalSetCoverReferenceStatus::NumericalError,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalSetCoverReferenceSolution {
    ExternalSetCoverReferenceSolution {
        status: ExternalSetCoverReferenceStatus::Unavailable,
        solver: "external-set-cover-reference".to_string(),
        selected_set_indices: Vec::new(),
        selected_set_ids: Vec::new(),
        objective: None,
        covered_elements: Vec::new(),
        ortools_status: None,
        ortools_selected_set_indices: Vec::new(),
        ortools_selected_set_ids: Vec::new(),
        ortools_objective: None,
        ortools_covered_elements: Vec::new(),
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalSetCoverReferenceSolution {
    ExternalSetCoverReferenceSolution {
        status: ExternalSetCoverReferenceStatus::NumericalError,
        solver: "external-set-cover-reference".to_string(),
        selected_set_indices: Vec::new(),
        selected_set_ids: Vec::new(),
        objective: None,
        covered_elements: Vec::new(),
        ortools_status: None,
        ortools_selected_set_indices: Vec::new(),
        ortools_selected_set_ids: Vec::new(),
        ortools_objective: None,
        ortools_covered_elements: Vec::new(),
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("set_cover_reference.py")
}

fn run_set_cover_reference_json(
    payload: Value,
    opts: &ExternalSetCoverReferenceOptions,
) -> ExternalSetCoverReferenceSolution {
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
                format!("failed to start set_cover_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write set_cover_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for set_cover_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<SetCoverReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalSetCoverReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-set-cover-reference".to_string()),
            selected_set_indices: parsed.selected_set_indices.unwrap_or_default(),
            selected_set_ids: parsed.selected_sets.unwrap_or_default(),
            objective: parsed.objective,
            covered_elements: parsed.covered_elements.unwrap_or_default(),
            ortools_status: parsed.ortools_status,
            ortools_selected_set_indices: parsed.ortools_selected_set_indices.unwrap_or_default(),
            ortools_selected_set_ids: parsed.ortools_selected_sets.unwrap_or_default(),
            ortools_objective: parsed.ortools_objective,
            ortools_covered_elements: parsed.ortools_covered_elements.unwrap_or_default(),
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
                "failed to parse set_cover_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_set_cover_with_external_reference(
    problem: &SetCoverProblem,
    opts: &ExternalSetCoverReferenceOptions,
) -> ExternalSetCoverReferenceSolution {
    run_set_cover_reference_json(
        json!({
            "universe": &problem.universe,
            "sets": problem.sets.iter().map(|set| json!({
                "id": &set.id,
                "cost": set.cost,
                "elements": &set.elements,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}
