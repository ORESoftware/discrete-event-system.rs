//! Rust-facing bridge for external/reference facility-location solvers.
//!
//! The native Rust reference computes an exact small-instance check without
//! Python startup. The Python bridge (`scripts/facility_location_reference.py`)
//! remains available for OR-Tools CP-SAT.

use std::collections::HashSet;
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
    RustExact,
    OrTools,
    Fallback,
}

impl ExternalFacilityLocationReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalFacilityLocationReferenceSolver::Auto => "auto",
            ExternalFacilityLocationReferenceSolver::RustExact => "rust-exact",
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

const RUST_FACILITY_LOCATION_MAX_EXACT_FACILITIES: usize = 24;
const RUST_FACILITY_LOCATION_EPS: f64 = 1e-9;

fn validate_rust_facility_location_problem(
    problem: &FacilityLocationProblem,
) -> Result<(), String> {
    if problem.facility_ids.is_empty() {
        return Err("facilities must be non-empty".to_string());
    }
    if problem.customer_ids.is_empty() {
        return Err("customers must be non-empty".to_string());
    }
    if problem.fixed_costs.len() != problem.facility_ids.len() {
        return Err("fixedCosts length must equal facilities length".to_string());
    }
    if problem.service_costs.len() != problem.facility_ids.len() {
        return Err("serviceCosts row count must equal facilities length".to_string());
    }

    let mut facilities = HashSet::new();
    for (index, facility) in problem.facility_ids.iter().enumerate() {
        if facility.trim().is_empty() {
            return Err(format!("facilities[{index}] must be non-empty"));
        }
        if !facilities.insert(facility.clone()) {
            return Err(format!("duplicate facility id {facility:?}"));
        }
        let fixed_cost = problem.fixed_costs[index];
        if !fixed_cost.is_finite() || fixed_cost < 0.0 {
            return Err(format!(
                "fixedCosts[{index}] must be finite and non-negative"
            ));
        }
    }

    let mut customers = HashSet::new();
    for (index, customer) in problem.customer_ids.iter().enumerate() {
        if customer.trim().is_empty() {
            return Err(format!("customers[{index}] must be non-empty"));
        }
        if !customers.insert(customer.clone()) {
            return Err(format!("duplicate customer id {customer:?}"));
        }
    }

    for (facility_index, row) in problem.service_costs.iter().enumerate() {
        if row.len() != problem.customer_ids.len() {
            return Err(format!(
                "serviceCosts[{facility_index}] length must equal customers length"
            ));
        }
        for (customer_index, &cost) in row.iter().enumerate() {
            if !cost.is_finite() || cost < 0.0 {
                return Err(format!(
                    "serviceCosts[{facility_index}][{customer_index}] must be finite and non-negative"
                ));
            }
        }
    }
    Ok(())
}

fn rust_facility_location_empty_solution(
    status: ExternalFacilityLocationReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalFacilityLocationReferenceSolution {
    ExternalFacilityLocationReferenceSolution {
        status,
        solver: solver.into(),
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

fn rust_facility_location_evaluate_open(
    problem: &FacilityLocationProblem,
    open_facility_indices: &[usize],
) -> Option<(f64, Vec<FacilityLocationAssignment>)> {
    if open_facility_indices.is_empty() {
        return None;
    }

    let mut objective = open_facility_indices
        .iter()
        .map(|&index| problem.fixed_costs[index])
        .sum::<f64>();
    let mut assignments = Vec::with_capacity(problem.customer_ids.len());
    for customer_index in 0..problem.customer_ids.len() {
        let mut best: Option<(usize, f64)> = None;
        for &facility_index in open_facility_indices {
            let cost = problem.service_costs[facility_index][customer_index];
            if best.is_none_or(|(best_index, best_cost)| {
                cost < best_cost - RUST_FACILITY_LOCATION_EPS
                    || ((cost - best_cost).abs() <= RUST_FACILITY_LOCATION_EPS
                        && facility_index < best_index)
            }) {
                best = Some((facility_index, cost));
            }
        }
        let (facility_index, cost) = best?;
        objective += cost;
        assignments.push(FacilityLocationAssignment {
            customer_index,
            customer_id: problem.customer_ids[customer_index].clone(),
            facility_index,
            facility_id: problem.facility_ids[facility_index].clone(),
            cost,
        });
    }

    Some((objective, assignments))
}

fn rust_facility_location_solution(
    problem: &FacilityLocationProblem,
    status: ExternalFacilityLocationReferenceStatus,
    mut open_facility_indices: Vec<usize>,
    assignments: Vec<FacilityLocationAssignment>,
    objective: f64,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalFacilityLocationReferenceSolution {
    open_facility_indices.sort_unstable();
    let open_facility_ids = open_facility_indices
        .iter()
        .map(|&index| problem.facility_ids[index].clone())
        .collect::<Vec<_>>();
    ExternalFacilityLocationReferenceSolution {
        status,
        solver: "rust:exact-facility-location".to_string(),
        open_facility_indices,
        open_facility_ids,
        assignments,
        objective: Some(objective),
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

fn solve_facility_location_with_rust_reference(
    problem: &FacilityLocationProblem,
) -> ExternalFacilityLocationReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_facility_location_problem(problem) {
        return rust_facility_location_empty_solution(
            ExternalFacilityLocationReferenceStatus::NumericalError,
            "rust:exact-facility-location",
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    if problem.facility_ids.len() > RUST_FACILITY_LOCATION_MAX_EXACT_FACILITIES {
        return rust_facility_location_empty_solution(
            ExternalFacilityLocationReferenceStatus::Unsupported,
            "rust:exact-facility-location",
            format!(
                "exact facility-location enumeration only practical for <= {RUST_FACILITY_LOCATION_MAX_EXACT_FACILITIES} facilities, got {}",
                problem.facility_ids.len()
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let mut best_open = Vec::new();
    let mut best_assignments = Vec::new();
    let mut best_objective = f64::INFINITY;
    let upper_mask = 1_u128 << problem.facility_ids.len();
    for mask in 1_u128..upper_mask {
        let open_facility_indices = (0..problem.facility_ids.len())
            .filter(|&index| mask & (1_u128 << index) != 0)
            .collect::<Vec<_>>();
        let Some((objective, assignments)) =
            rust_facility_location_evaluate_open(problem, &open_facility_indices)
        else {
            continue;
        };
        if objective < best_objective - RUST_FACILITY_LOCATION_EPS
            || ((objective - best_objective).abs() <= RUST_FACILITY_LOCATION_EPS
                && open_facility_indices < best_open)
        {
            best_open = open_facility_indices;
            best_assignments = assignments;
            best_objective = objective;
        }
    }

    if best_open.is_empty() {
        return rust_facility_location_empty_solution(
            ExternalFacilityLocationReferenceStatus::Infeasible,
            "rust:exact-facility-location",
            "no feasible facility subset",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    rust_facility_location_solution(
        problem,
        ExternalFacilityLocationReferenceStatus::Optimal,
        best_open,
        best_assignments,
        best_objective,
        "exact open-facility subset enumeration",
        started.elapsed().as_secs_f64() * 1000.0,
    )
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
    if matches!(
        opts.solver,
        ExternalFacilityLocationReferenceSolver::Auto
            | ExternalFacilityLocationReferenceSolver::RustExact
            | ExternalFacilityLocationReferenceSolver::Fallback
    ) {
        return solve_facility_location_with_rust_reference(problem);
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::facility_location::{
        build_sample_facility_location_problem, FacilityLocationProblem,
    };

    #[test]
    fn rust_reference_solves_sample_facility_location() {
        let problem = build_sample_facility_location_problem();
        let solution = solve_facility_location_with_external_reference(
            &problem,
            &ExternalFacilityLocationReferenceOptions {
                solver: ExternalFacilityLocationReferenceSolver::RustExact,
            },
        );

        assert_eq!(
            solution.status,
            ExternalFacilityLocationReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:exact-facility-location");
        assert_eq!(solution.open_facility_ids, vec!["North", "South"]);
        assert_eq!(solution.objective, Some(28.0));
        assert_eq!(solution.assignments.len(), problem.customer_ids.len());
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_with_tie_breaking() {
        let problem = FacilityLocationProblem {
            facility_ids: vec!["A".to_string(), "B".to_string()],
            customer_ids: vec!["C".to_string()],
            fixed_costs: vec![1.0, 1.0],
            service_costs: vec![vec![1.0], vec![1.0]],
        };

        let solution = solve_facility_location_with_external_reference(
            &problem,
            &ExternalFacilityLocationReferenceOptions {
                solver: ExternalFacilityLocationReferenceSolver::Fallback,
            },
        );

        assert_eq!(
            solution.status,
            ExternalFacilityLocationReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:exact-facility-location");
        assert_eq!(solution.open_facility_ids, vec!["A"]);
        assert_eq!(solution.objective, Some(2.0));
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_sample_facility_location_problem();

        let solution = solve_facility_location_with_external_reference(
            &problem,
            &ExternalFacilityLocationReferenceOptions::default(),
        );

        assert_eq!(
            solution.status,
            ExternalFacilityLocationReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:exact-facility-location");
        assert_eq!(solution.open_facility_ids, vec!["North", "South"]);
        assert_eq!(solution.objective, Some(28.0));
    }
}
