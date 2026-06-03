//! Rust-facing bridge for external/reference weighted Max-SAT solvers.
//!
//! The Python bridge (`scripts/weighted_max_sat_reference.py`) computes an
//! exact enumeration reference and, when installed, solves the same weighted
//! partial Max-SAT model with OR-Tools CP-SAT.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::weighted_max_sat::WeightedMaxSatProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalWeightedMaxSatReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalWeightedMaxSatReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalWeightedMaxSatReferenceSolver::Auto => "auto",
            ExternalWeightedMaxSatReferenceSolver::OrTools => "ortools",
            ExternalWeightedMaxSatReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalWeightedMaxSatReferenceOptions {
    pub solver: ExternalWeightedMaxSatReferenceSolver,
}

impl Default for ExternalWeightedMaxSatReferenceOptions {
    fn default() -> Self {
        ExternalWeightedMaxSatReferenceOptions {
            solver: ExternalWeightedMaxSatReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalWeightedMaxSatReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalWeightedMaxSatReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalWeightedMaxSatReferenceStatus::Optimal => "optimal",
            ExternalWeightedMaxSatReferenceStatus::Feasible => "feasible",
            ExternalWeightedMaxSatReferenceStatus::Infeasible => "infeasible",
            ExternalWeightedMaxSatReferenceStatus::Unsupported => "unsupported",
            ExternalWeightedMaxSatReferenceStatus::NumericalError => "numerical-error",
            ExternalWeightedMaxSatReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalWeightedMaxSatReferenceSolution {
    pub status: ExternalWeightedMaxSatReferenceStatus,
    pub solver: String,
    pub assignment: Vec<bool>,
    pub objective: Option<f64>,
    pub satisfied_soft_weight: Option<f64>,
    pub unsatisfied_soft_weight: Option<f64>,
    pub satisfied_clause_ids: Vec<String>,
    pub violated_hard_clause_ids: Vec<String>,
    pub ortools_status: Option<String>,
    pub ortools_assignment: Vec<bool>,
    pub ortools_objective: Option<f64>,
    pub ortools_satisfied_soft_weight: Option<f64>,
    pub ortools_unsatisfied_soft_weight: Option<f64>,
    pub ortools_satisfied_clause_ids: Vec<String>,
    pub ortools_violated_hard_clause_ids: Vec<String>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct WeightedMaxSatReferencePayload {
    status: String,
    solver: Option<String>,
    assignment: Option<Vec<bool>>,
    objective: Option<f64>,
    #[serde(rename = "satisfiedSoftWeight")]
    satisfied_soft_weight: Option<f64>,
    #[serde(rename = "unsatisfiedSoftWeight")]
    unsatisfied_soft_weight: Option<f64>,
    #[serde(rename = "satisfiedClauseIds")]
    satisfied_clause_ids: Option<Vec<String>>,
    #[serde(rename = "violatedHardClauseIds")]
    violated_hard_clause_ids: Option<Vec<String>>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsAssignment")]
    ortools_assignment: Option<Vec<bool>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsSatisfiedSoftWeight")]
    ortools_satisfied_soft_weight: Option<f64>,
    #[serde(rename = "ortoolsUnsatisfiedSoftWeight")]
    ortools_unsatisfied_soft_weight: Option<f64>,
    #[serde(rename = "ortoolsSatisfiedClauseIds")]
    ortools_satisfied_clause_ids: Option<Vec<String>>,
    #[serde(rename = "ortoolsViolatedHardClauseIds")]
    ortools_violated_hard_clause_ids: Option<Vec<String>>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalWeightedMaxSatReferenceStatus {
    match status {
        "optimal" => ExternalWeightedMaxSatReferenceStatus::Optimal,
        "feasible" => ExternalWeightedMaxSatReferenceStatus::Feasible,
        "infeasible" => ExternalWeightedMaxSatReferenceStatus::Infeasible,
        "unsupported" => ExternalWeightedMaxSatReferenceStatus::Unsupported,
        "unavailable" => ExternalWeightedMaxSatReferenceStatus::Unavailable,
        _ => ExternalWeightedMaxSatReferenceStatus::NumericalError,
    }
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedMaxSatReferenceSolution {
    ExternalWeightedMaxSatReferenceSolution {
        status: ExternalWeightedMaxSatReferenceStatus::Unavailable,
        solver: "external-weighted-max-sat-reference".to_string(),
        assignment: Vec::new(),
        objective: None,
        satisfied_soft_weight: None,
        unsatisfied_soft_weight: None,
        satisfied_clause_ids: Vec::new(),
        violated_hard_clause_ids: Vec::new(),
        ortools_status: None,
        ortools_assignment: Vec::new(),
        ortools_objective: None,
        ortools_satisfied_soft_weight: None,
        ortools_unsatisfied_soft_weight: None,
        ortools_satisfied_clause_ids: Vec::new(),
        ortools_violated_hard_clause_ids: Vec::new(),
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedMaxSatReferenceSolution {
    ExternalWeightedMaxSatReferenceSolution {
        status: ExternalWeightedMaxSatReferenceStatus::NumericalError,
        solver: "external-weighted-max-sat-reference".to_string(),
        assignment: Vec::new(),
        objective: None,
        satisfied_soft_weight: None,
        unsatisfied_soft_weight: None,
        satisfied_clause_ids: Vec::new(),
        violated_hard_clause_ids: Vec::new(),
        ortools_status: None,
        ortools_assignment: Vec::new(),
        ortools_objective: None,
        ortools_satisfied_soft_weight: None,
        ortools_unsatisfied_soft_weight: None,
        ortools_satisfied_clause_ids: Vec::new(),
        ortools_violated_hard_clause_ids: Vec::new(),
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("weighted_max_sat_reference.py")
}

fn run_weighted_max_sat_reference_json(
    payload: Value,
    opts: &ExternalWeightedMaxSatReferenceOptions,
) -> ExternalWeightedMaxSatReferenceSolution {
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
                format!("failed to start weighted_max_sat_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write weighted_max_sat_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for weighted_max_sat_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<WeightedMaxSatReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalWeightedMaxSatReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-weighted-max-sat-reference".to_string()),
            assignment: parsed.assignment.unwrap_or_default(),
            objective: parsed.objective,
            satisfied_soft_weight: parsed.satisfied_soft_weight,
            unsatisfied_soft_weight: parsed.unsatisfied_soft_weight,
            satisfied_clause_ids: parsed.satisfied_clause_ids.unwrap_or_default(),
            violated_hard_clause_ids: parsed.violated_hard_clause_ids.unwrap_or_default(),
            ortools_status: parsed.ortools_status,
            ortools_assignment: parsed.ortools_assignment.unwrap_or_default(),
            ortools_objective: parsed.ortools_objective,
            ortools_satisfied_soft_weight: parsed.ortools_satisfied_soft_weight,
            ortools_unsatisfied_soft_weight: parsed.ortools_unsatisfied_soft_weight,
            ortools_satisfied_clause_ids: parsed.ortools_satisfied_clause_ids.unwrap_or_default(),
            ortools_violated_hard_clause_ids: parsed
                .ortools_violated_hard_clause_ids
                .unwrap_or_default(),
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
                "failed to parse weighted_max_sat_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_weighted_max_sat_with_external_reference(
    problem: &WeightedMaxSatProblem,
    opts: &ExternalWeightedMaxSatReferenceOptions,
) -> ExternalWeightedMaxSatReferenceSolution {
    run_weighted_max_sat_reference_json(
        json!({
            "numVars": problem.num_vars,
            "clauses": problem.clauses.iter().map(|clause| json!({
                "id": clause.id,
                "literals": clause.literals,
                "weight": clause.weight,
                "hard": clause.hard,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}
