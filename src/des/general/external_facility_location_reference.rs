//! Rust-facing bridge for external/reference facility-location solvers.
//!
//! The Python bridge (`scripts/facility_location_reference.py`) computes an
//! exact small-instance reference and, when installed, solves the same
//! uncapacitated facility-location model with OR-Tools CP-SAT.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::facility_location::{FacilityLocationAssignment, FacilityLocationProblem};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalFacilityLocationReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalFacilityLocationReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalFacilityLocationReferenceSolver::Auto => "auto",
            ExternalFacilityLocationReferenceSolver::OrTools => "ortools",
            ExternalFacilityLocationReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalFacilityLocationReferenceOptions {
    pub solver: ExternalFacilityLocationReferenceSolver,
}

impl Default for ExternalFacilityLocationReferenceOptions {
    fn default() -> Self {
        ExternalFacilityLocationReferenceOptions {
            solver: ExternalFacilityLocationReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalFacilityLocationReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalFacilityLocationReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalFacilityLocationReferenceStatus::Optimal => "optimal",
            ExternalFacilityLocationReferenceStatus::Feasible => "feasible",
            ExternalFacilityLocationReferenceStatus::Infeasible => "infeasible",
            ExternalFacilityLocationReferenceStatus::Unsupported => "unsupported",
            ExternalFacilityLocationReferenceStatus::NumericalError => "numerical-error",
            ExternalFacilityLocationReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalFacilityLocationReferenceSolution {
    pub status: ExternalFacilityLocationReferenceStatus,
    pub solver: String,
    pub open_facility_indices: Vec<usize>,
    pub open_facility_ids: Vec<String>,
    pub assignments: Vec<FacilityLocationAssignment>,
    pub objective: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_open_facility_indices: Vec<usize>,
    pub ortools_open_facility_ids: Vec<String>,
    pub ortools_assignments: Vec<FacilityLocationAssignment>,
    pub ortools_objective: Option<f64>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct FacilityLocationAssignmentPayload {
    #[serde(rename = "customerIndex")]
    customer_index: usize,
    customer: String,
    #[serde(rename = "facilityIndex")]
    facility_index: usize,
    facility: String,
    cost: f64,
}

#[derive(Debug, Deserialize)]
struct FacilityLocationReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "openFacilityIndices")]
    open_facility_indices: Option<Vec<usize>>,
    #[serde(rename = "openFacilities")]
    open_facilities: Option<Vec<String>>,
    assignments: Option<Vec<FacilityLocationAssignmentPayload>>,
    objective: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsOpenFacilityIndices")]
    ortools_open_facility_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsOpenFacilities")]
    ortools_open_facilities: Option<Vec<String>>,
    #[serde(rename = "ortoolsAssignments")]
    ortools_assignments: Option<Vec<FacilityLocationAssignmentPayload>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalFacilityLocationReferenceStatus {
    match status {
        "optimal" => ExternalFacilityLocationReferenceStatus::Optimal,
        "feasible" => ExternalFacilityLocationReferenceStatus::Feasible,
        "infeasible" => ExternalFacilityLocationReferenceStatus::Infeasible,
        "unsupported" => ExternalFacilityLocationReferenceStatus::Unsupported,
        "unavailable" => ExternalFacilityLocationReferenceStatus::Unavailable,
        _ => ExternalFacilityLocationReferenceStatus::NumericalError,
    }
}

fn convert_assignments(
    assignments: Option<Vec<FacilityLocationAssignmentPayload>>,
) -> Vec<FacilityLocationAssignment> {
    assignments
        .unwrap_or_default()
        .into_iter()
        .map(|assignment| FacilityLocationAssignment {
            customer_index: assignment.customer_index,
            customer_id: assignment.customer,
            facility_index: assignment.facility_index,
            facility_id: assignment.facility,
            cost: assignment.cost,
        })
        .collect()
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalFacilityLocationReferenceSolution {
    ExternalFacilityLocationReferenceSolution {
        status: ExternalFacilityLocationReferenceStatus::Unavailable,
        solver: "external-facility-location-reference".to_string(),
        open_facility_indices: Vec::new(),
        open_facility_ids: Vec::new(),
        assignments: Vec::new(),
        objective: None,
        ortools_status: None,
        ortools_open_facility_indices: Vec::new(),
        ortools_open_facility_ids: Vec::new(),
        ortools_assignments: Vec::new(),
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalFacilityLocationReferenceSolution {
    ExternalFacilityLocationReferenceSolution {
        status: ExternalFacilityLocationReferenceStatus::NumericalError,
        solver: "external-facility-location-reference".to_string(),
        open_facility_indices: Vec::new(),
        open_facility_ids: Vec::new(),
        assignments: Vec::new(),
        objective: None,
        ortools_status: None,
        ortools_open_facility_indices: Vec::new(),
        ortools_open_facility_ids: Vec::new(),
        ortools_assignments: Vec::new(),
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
    root.join("scripts").join("facility_location_reference.py")
}

fn run_facility_location_reference_json(
    payload: Value,
    opts: &ExternalFacilityLocationReferenceOptions,
) -> ExternalFacilityLocationReferenceSolution {
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
                format!("failed to start facility_location_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write facility_location_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for facility_location_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<FacilityLocationReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalFacilityLocationReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-facility-location-reference".to_string()),
            open_facility_indices: parsed.open_facility_indices.unwrap_or_default(),
            open_facility_ids: parsed.open_facilities.unwrap_or_default(),
            assignments: convert_assignments(parsed.assignments),
            objective: parsed.objective,
            ortools_status: parsed.ortools_status,
            ortools_open_facility_indices: parsed.ortools_open_facility_indices.unwrap_or_default(),
            ortools_open_facility_ids: parsed.ortools_open_facilities.unwrap_or_default(),
            ortools_assignments: convert_assignments(parsed.ortools_assignments),
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
                "failed to parse facility_location_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_facility_location_with_external_reference(
    problem: &FacilityLocationProblem,
    opts: &ExternalFacilityLocationReferenceOptions,
) -> ExternalFacilityLocationReferenceSolution {
    run_facility_location_reference_json(
        json!({
            "facilities": &problem.facility_ids,
            "customers": &problem.customer_ids,
            "fixedCosts": &problem.fixed_costs,
            "serviceCosts": &problem.service_costs,
        }),
        opts,
    )
}
