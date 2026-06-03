//! Rust-facing bridge for external/reference 0/1 knapsack solvers.
//!
//! The Python bridge (`scripts/knapsack_reference.py`) computes a deterministic
//! exact branch-and-bound reference and, when installed, solves the same model
//! with OR-Tools CP-SAT.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::knapsack::KnapsackProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalKnapsackReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalKnapsackReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalKnapsackReferenceSolver::Auto => "auto",
            ExternalKnapsackReferenceSolver::OrTools => "ortools",
            ExternalKnapsackReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalKnapsackReferenceOptions {
    pub solver: ExternalKnapsackReferenceSolver,
}

impl Default for ExternalKnapsackReferenceOptions {
    fn default() -> Self {
        ExternalKnapsackReferenceOptions {
            solver: ExternalKnapsackReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalKnapsackReferenceStatus {
    Optimal,
    Feasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalKnapsackReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalKnapsackReferenceStatus::Optimal => "optimal",
            ExternalKnapsackReferenceStatus::Feasible => "feasible",
            ExternalKnapsackReferenceStatus::Unsupported => "unsupported",
            ExternalKnapsackReferenceStatus::NumericalError => "numerical-error",
            ExternalKnapsackReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalKnapsackReferenceSolution {
    pub status: ExternalKnapsackReferenceStatus,
    pub solver: String,
    pub selected_item_indices: Vec<usize>,
    pub selected_item_ids: Vec<String>,
    pub total_weight: Option<f64>,
    pub total_value: Option<f64>,
    pub objective: Option<f64>,
    pub upper_bound: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_selected_item_indices: Vec<usize>,
    pub ortools_selected_item_ids: Vec<String>,
    pub ortools_total_weight: Option<f64>,
    pub ortools_total_value: Option<f64>,
    pub ortools_objective: Option<f64>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct KnapsackReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "selectedItemIndices")]
    selected_item_indices: Option<Vec<usize>>,
    #[serde(rename = "selectedItemIds")]
    selected_item_ids: Option<Vec<String>>,
    #[serde(rename = "totalWeight")]
    total_weight: Option<f64>,
    #[serde(rename = "totalValue")]
    total_value: Option<f64>,
    objective: Option<f64>,
    #[serde(rename = "upperBound")]
    upper_bound: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsSelectedItemIndices")]
    ortools_selected_item_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsSelectedItemIds")]
    ortools_selected_item_ids: Option<Vec<String>>,
    #[serde(rename = "ortoolsTotalWeight")]
    ortools_total_weight: Option<f64>,
    #[serde(rename = "ortoolsTotalValue")]
    ortools_total_value: Option<f64>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalKnapsackReferenceStatus {
    match status {
        "optimal" => ExternalKnapsackReferenceStatus::Optimal,
        "feasible" => ExternalKnapsackReferenceStatus::Feasible,
        "unsupported" => ExternalKnapsackReferenceStatus::Unsupported,
        "unavailable" => ExternalKnapsackReferenceStatus::Unavailable,
        _ => ExternalKnapsackReferenceStatus::NumericalError,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalKnapsackReferenceSolution {
    ExternalKnapsackReferenceSolution {
        status: ExternalKnapsackReferenceStatus::Unavailable,
        solver: "external-knapsack-reference".to_string(),
        selected_item_indices: Vec::new(),
        selected_item_ids: Vec::new(),
        total_weight: None,
        total_value: None,
        objective: None,
        upper_bound: None,
        ortools_status: None,
        ortools_selected_item_indices: Vec::new(),
        ortools_selected_item_ids: Vec::new(),
        ortools_total_weight: None,
        ortools_total_value: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalKnapsackReferenceSolution {
    ExternalKnapsackReferenceSolution {
        status: ExternalKnapsackReferenceStatus::NumericalError,
        solver: "external-knapsack-reference".to_string(),
        selected_item_indices: Vec::new(),
        selected_item_ids: Vec::new(),
        total_weight: None,
        total_value: None,
        objective: None,
        upper_bound: None,
        ortools_status: None,
        ortools_selected_item_indices: Vec::new(),
        ortools_selected_item_ids: Vec::new(),
        ortools_total_weight: None,
        ortools_total_value: None,
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
    root.join("scripts").join("knapsack_reference.py")
}

fn run_knapsack_reference_json(
    payload: Value,
    opts: &ExternalKnapsackReferenceOptions,
) -> ExternalKnapsackReferenceSolution {
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
                format!("failed to start knapsack_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write knapsack_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for knapsack_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<KnapsackReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalKnapsackReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-knapsack-reference".to_string()),
            selected_item_indices: parsed.selected_item_indices.unwrap_or_default(),
            selected_item_ids: parsed.selected_item_ids.unwrap_or_default(),
            total_weight: parsed.total_weight,
            total_value: parsed.total_value,
            objective: parsed.objective,
            upper_bound: parsed.upper_bound,
            ortools_status: parsed.ortools_status,
            ortools_selected_item_indices: parsed.ortools_selected_item_indices.unwrap_or_default(),
            ortools_selected_item_ids: parsed.ortools_selected_item_ids.unwrap_or_default(),
            ortools_total_weight: parsed.ortools_total_weight,
            ortools_total_value: parsed.ortools_total_value,
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
                "failed to parse knapsack_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_knapsack_with_external_reference(
    problem: &KnapsackProblem,
    opts: &ExternalKnapsackReferenceOptions,
) -> ExternalKnapsackReferenceSolution {
    run_knapsack_reference_json(
        json!({
            "capacity": problem.capacity,
            "items": problem.items.iter().map(|item| json!({
                "id": &item.id,
                "weight": item.weight,
                "value": item.value,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}
