//! Rust-facing bridge for external/reference assignment solvers.
//!
//! The checked-in Python bridge (`scripts/assignment_reference.py`) computes an
//! exact small assignment reference, calls OR-Tools SimpleLinearSumAssignment
//! when costs can be integer-scaled, and also records SciPy's
//! `linear_sum_assignment` result when available.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalAssignmentReferenceSolver {
    Auto,
    OrTools,
    Scipy,
    Fallback,
}

impl ExternalAssignmentReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalAssignmentReferenceSolver::Auto => "auto",
            ExternalAssignmentReferenceSolver::OrTools => "ortools",
            ExternalAssignmentReferenceSolver::Scipy => "scipy",
            ExternalAssignmentReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalAssignmentReferenceOptions {
    pub solver: ExternalAssignmentReferenceSolver,
}

impl Default for ExternalAssignmentReferenceOptions {
    fn default() -> Self {
        ExternalAssignmentReferenceOptions {
            solver: ExternalAssignmentReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalAssignmentReferenceStatus {
    Optimal,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalAssignmentReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalAssignmentReferenceStatus::Optimal => "optimal",
            ExternalAssignmentReferenceStatus::Infeasible => "infeasible",
            ExternalAssignmentReferenceStatus::Unsupported => "unsupported",
            ExternalAssignmentReferenceStatus::NumericalError => "numerical-error",
            ExternalAssignmentReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalAssignmentReferenceSolution {
    pub status: ExternalAssignmentReferenceStatus,
    pub solver: String,
    pub assignment: Vec<i64>,
    pub objective: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_assignment: Vec<i64>,
    pub ortools_objective: Option<f64>,
    pub scipy_status: Option<String>,
    pub scipy_assignment: Vec<i64>,
    pub scipy_objective: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct AssignmentReferencePayload {
    status: String,
    solver: Option<String>,
    assignment: Option<Vec<i64>>,
    objective: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsAssignment")]
    ortools_assignment: Option<Vec<i64>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "scipyStatus")]
    scipy_status: Option<String>,
    #[serde(rename = "scipyAssignment")]
    scipy_assignment: Option<Vec<i64>>,
    #[serde(rename = "scipyObjective")]
    scipy_objective: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalAssignmentReferenceStatus {
    match status {
        "optimal" => ExternalAssignmentReferenceStatus::Optimal,
        "infeasible" => ExternalAssignmentReferenceStatus::Infeasible,
        "unsupported" => ExternalAssignmentReferenceStatus::Unsupported,
        "unavailable" => ExternalAssignmentReferenceStatus::Unavailable,
        _ => ExternalAssignmentReferenceStatus::NumericalError,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalAssignmentReferenceSolution {
    ExternalAssignmentReferenceSolution {
        status: ExternalAssignmentReferenceStatus::Unavailable,
        solver: "external-assignment-reference".to_string(),
        assignment: Vec::new(),
        objective: None,
        ortools_status: None,
        ortools_assignment: Vec::new(),
        ortools_objective: None,
        scipy_status: None,
        scipy_assignment: Vec::new(),
        scipy_objective: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalAssignmentReferenceSolution {
    ExternalAssignmentReferenceSolution {
        status: ExternalAssignmentReferenceStatus::NumericalError,
        solver: "external-assignment-reference".to_string(),
        assignment: Vec::new(),
        objective: None,
        ortools_status: None,
        ortools_assignment: Vec::new(),
        ortools_objective: None,
        scipy_status: None,
        scipy_assignment: Vec::new(),
        scipy_objective: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("assignment_reference.py")
}

fn run_assignment_reference_json(
    payload: Value,
    opts: &ExternalAssignmentReferenceOptions,
) -> ExternalAssignmentReferenceSolution {
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
                format!("failed to start assignment_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write assignment_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for assignment_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<AssignmentReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalAssignmentReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-assignment-reference".to_string()),
            assignment: parsed.assignment.unwrap_or_default(),
            objective: parsed.objective,
            ortools_status: parsed.ortools_status,
            ortools_assignment: parsed.ortools_assignment.unwrap_or_default(),
            ortools_objective: parsed.ortools_objective,
            scipy_status: parsed.scipy_status,
            scipy_assignment: parsed.scipy_assignment.unwrap_or_default(),
            scipy_objective: parsed.scipy_objective,
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
                "failed to parse assignment_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_assignment_with_external_reference(
    cost: &[Vec<f64>],
    opts: &ExternalAssignmentReferenceOptions,
) -> ExternalAssignmentReferenceSolution {
    run_assignment_reference_json(
        json!({
            "cost": cost,
        }),
        opts,
    )
}
