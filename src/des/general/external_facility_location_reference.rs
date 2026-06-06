//! Rust-facing bridge for external/reference facility-location solvers.
//!
//! The native Rust reference computes an exact small-instance check without
//! Python startup. Registered OR-Tools aliases default to that Rust reference;
//! explicit force-Python switches keep the inline OR-Tools adapter available
//! for compatibility validation.

use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

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

fn facility_location_reference_force_python_value(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    matches!(
        normalized.as_str(),
        "1" | "true"
            | "yes"
            | "on"
            | "python"
            | "py"
            | "legacy-python"
            | "python-reference"
            | "python-bridge"
    )
}

fn facility_location_python_reference_forced() -> bool {
    [
        "FACILITY_LOCATION_REFERENCE_FORCE_PYTHON",
        "FACILITY_LOCATION_REFERENCE_ORTOOLS_FORCE_PYTHON",
        "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
    ]
    .into_iter()
    .any(|key| {
        std::env::var(key)
            .map(|value| facility_location_reference_force_python_value(&value))
            .unwrap_or(false)
    })
}

fn should_use_rust_facility_location_reference(
    opts: &ExternalFacilityLocationReferenceOptions,
) -> bool {
    matches!(
        opts.solver,
        ExternalFacilityLocationReferenceSolver::Auto
            | ExternalFacilityLocationReferenceSolver::RustExact
            | ExternalFacilityLocationReferenceSolver::Fallback
    )
}

fn should_use_registered_facility_location_fallback(
    opts: &ExternalFacilityLocationReferenceOptions,
) -> bool {
    matches!(
        opts.solver,
        ExternalFacilityLocationReferenceSolver::OrTools
    ) && !facility_location_python_reference_forced()
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

fn relabel_registered_facility_location_fallback(
    mut solution: ExternalFacilityLocationReferenceSolution,
    opts: &ExternalFacilityLocationReferenceOptions,
) -> ExternalFacilityLocationReferenceSolution {
    if should_use_registered_facility_location_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-facility-location-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
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

fn facility_location_reference_timeout_ms() -> u64 {
    std::env::var("FACILITY_LOCATION_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_facility_location_adapter_output(
    mut child: std::process::Child,
    timeout_ms: u64,
) -> Result<(Output, bool), String> {
    let started = Instant::now();
    let timeout = Duration::from_millis(timeout_ms);
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if timeout_ms > 0 && started.elapsed() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    break;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(err) => {
                return Err(format!(
                    "failed to poll OR-Tools facility-location adapter: {err}"
                ))
            }
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for OR-Tools facility-location adapter: {err}"))
}

const ORTOOLS_FACILITY_LOCATION_SCALES: [i64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];

const ORTOOLS_FACILITY_LOCATION_ADAPTER: &str = r#"
import json
import sys

SOLVER = "ortools:cp-sat-facility-location"

def emit(status, message, data=None, open_indices=None, assignments=None, objective=None, objective_bound=None, ortools_status=None):
    data = {} if data is None else data
    facilities = data.get("facilities", [])
    open_indices = [] if open_indices is None else sorted(set(int(index) for index in open_indices))
    open_ids = [facilities[index] for index in open_indices] if facilities else []
    assignments = [] if assignments is None else assignments
    payload = {
        "status": status,
        "solver": SOLVER,
        "openFacilityIndices": open_indices,
        "openFacilities": open_ids,
        "assignments": assignments,
        "objective": objective,
        "message": message,
        "ortoolsStatus": ortools_status,
        "ortoolsOpenFacilityIndices": open_indices,
        "ortoolsOpenFacilities": open_ids,
        "ortoolsAssignments": assignments,
        "ortoolsObjective": objective,
        "ortoolsObjectiveBound": objective_bound,
    }
    print(json.dumps(payload))

try:
    from ortools.sat.python import cp_model
except Exception as exc:
    emit("unavailable", f"OR-Tools CP-SAT unavailable: {exc}", ortools_status="unavailable")
    raise SystemExit(0)

try:
    data = json.load(sys.stdin)
    facilities = data["facilities"]
    customers = data["customers"]
    fixed_costs = data["fixedCosts"]
    service_costs = data["serviceCosts"]
    scaled_fixed_costs = data["scaledFixedCosts"]
    scaled_service_costs = data["scaledServiceCosts"]
    scale = int(data["scale"])
    facility_count = len(facilities)
    customer_count = len(customers)

    model = cp_model.CpModel()
    y = [model.NewBoolVar(f"open_f{facility}") for facility in range(facility_count)]
    x = [
        [
            model.NewBoolVar(f"assign_f{facility}_c{customer}")
            for customer in range(customer_count)
        ]
        for facility in range(facility_count)
    ]
    for customer in range(customer_count):
        model.Add(sum(x[facility][customer] for facility in range(facility_count)) == 1)
    for facility in range(facility_count):
        for customer in range(customer_count):
            model.Add(x[facility][customer] <= y[facility])
    model.Minimize(
        sum(int(scaled_fixed_costs[facility]) * y[facility] for facility in range(facility_count))
        + sum(
            int(scaled_service_costs[facility][customer]) * x[facility][customer]
            for facility in range(facility_count)
            for customer in range(customer_count)
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        emit(
            "infeasible" if status_name == "infeasible" else status_name,
            f"OR-Tools CP-SAT status {status_name}",
            data=data,
            ortools_status=status_name,
        )
        raise SystemExit(0)

    open_indices = [facility for facility, var in enumerate(y) if solver.BooleanValue(var)]
    assignments = []
    objective = sum(fixed_costs[facility] for facility in open_indices)
    for customer in range(customer_count):
        assigned = [
            facility
            for facility in range(facility_count)
            if solver.BooleanValue(x[facility][customer])
        ]
        facility = int(assigned[0])
        cost = float(service_costs[facility][customer])
        objective += cost
        assignments.append(
            {
                "customerIndex": customer,
                "customer": customers[customer],
                "facilityIndex": facility,
                "facility": facilities[facility],
                "cost": cost,
            }
        )

    emit(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        f"OR-Tools CP-SAT status {status_name}",
        data=data,
        open_indices=open_indices,
        assignments=assignments,
        objective=float(objective),
        objective_bound=float(solver.BestObjectiveBound()) / scale,
        ortools_status=status_name,
    )
except Exception as exc:
    emit("numerical-error", str(exc), ortools_status="error")
    raise SystemExit(1)
"#;

fn scaled_ortools_facility_cost(value: f64, scale: i64) -> Option<i64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let scaled = value * scale as f64;
    let rounded = scaled.round();
    if (rounded - scaled).abs() > 1e-6
        || !rounded.is_finite()
        || rounded < 0.0
        || rounded > i64::MAX as f64
    {
        return None;
    }
    Some(rounded as i64)
}

fn choose_ortools_facility_location_scale(problem: &FacilityLocationProblem) -> Option<i64> {
    ORTOOLS_FACILITY_LOCATION_SCALES.into_iter().find(|scale| {
        problem
            .fixed_costs
            .iter()
            .chain(problem.service_costs.iter().flatten())
            .all(|&value| scaled_ortools_facility_cost(value, *scale).is_some())
    })
}

fn ortools_facility_location_payload(problem: &FacilityLocationProblem) -> Result<Value, String> {
    validate_rust_facility_location_problem(problem)?;
    let Some(scale) = choose_ortools_facility_location_scale(problem) else {
        return Err("OR-Tools CP-SAT bridge requires integer-scalable costs".to_string());
    };
    let scaled_fixed_costs = problem
        .fixed_costs
        .iter()
        .map(|&cost| {
            scaled_ortools_facility_cost(cost, scale)
                .expect("scale was selected only after checking fixed costs")
        })
        .collect::<Vec<_>>();
    let scaled_service_costs = problem
        .service_costs
        .iter()
        .map(|row| {
            row.iter()
                .map(|&cost| {
                    scaled_ortools_facility_cost(cost, scale)
                        .expect("scale was selected only after checking service costs")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "facilities": &problem.facility_ids,
        "customers": &problem.customer_ids,
        "fixedCosts": &problem.fixed_costs,
        "serviceCosts": &problem.service_costs,
        "scale": scale,
        "scaledFixedCosts": scaled_fixed_costs,
        "scaledServiceCosts": scaled_service_costs,
    }))
}

fn run_ortools_facility_location_reference(
    problem: &FacilityLocationProblem,
) -> ExternalFacilityLocationReferenceSolution {
    let started = Instant::now();
    let payload = match ortools_facility_location_payload(problem) {
        Ok(payload) => payload,
        Err(message) if message.contains("integer-scalable costs") => {
            return rust_facility_location_empty_solution(
                ExternalFacilityLocationReferenceStatus::Unsupported,
                "ortools:cp-sat-facility-location",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Err(message) => {
            return numerical_error(message, started.elapsed().as_secs_f64() * 1000.0);
        }
    };
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ORTOOLS_FACILITY_LOCATION_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return unavailable(
                format!("failed to start OR-Tools facility-location adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write OR-Tools facility-location adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = facility_location_reference_timeout_ms();
    let (output, timed_out) = match wait_for_facility_location_adapter_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0),
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("OR-Tools facility-location adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; OR-Tools facility-location adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
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
                    stderr.clone()
                }
            }),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse OR-Tools facility-location adapter output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_facility_location_with_external_reference(
    problem: &FacilityLocationProblem,
    opts: &ExternalFacilityLocationReferenceOptions,
) -> ExternalFacilityLocationReferenceSolution {
    if should_use_rust_facility_location_reference(opts)
        || should_use_registered_facility_location_fallback(opts)
    {
        return relabel_registered_facility_location_fallback(
            solve_facility_location_with_rust_reference(problem),
            opts,
        );
    }

    run_ortools_facility_location_reference(problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::facility_location::{
        build_sample_facility_location_problem, FacilityLocationProblem,
    };
    use std::sync::Mutex;

    static FACILITY_LOCATION_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => std::env::set_var(self.key, previous),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn facility_location_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "FACILITY_LOCATION_REFERENCE_FORCE_PYTHON",
            "FACILITY_LOCATION_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

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

    #[test]
    fn registered_ortools_alias_defaults_to_rust_reference_without_python() {
        let _lock = FACILITY_LOCATION_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guards = facility_location_force_python_off_guards();
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-facility-location-alias",
        );
        let problem = build_sample_facility_location_problem();

        let solution = solve_facility_location_with_external_reference(
            &problem,
            &ExternalFacilityLocationReferenceOptions {
                solver: ExternalFacilityLocationReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalFacilityLocationReferenceStatus::Optimal
        );
        assert_eq!(
            solution.solver,
            "rust:registered-facility-location-fallback-for-ortools"
        );
        assert_eq!(solution.open_facility_ids, vec!["North", "South"]);
        assert_eq!(solution.objective, Some(28.0));
        assert_eq!(solution.assignments.len(), problem.customer_ids.len());
        assert!(solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
    }

    #[test]
    fn facility_location_force_python_keeps_ortools_bridge_available() {
        let _lock = FACILITY_LOCATION_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("FACILITY_LOCATION_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-forced-facility-location",
        );
        let problem = build_sample_facility_location_problem();

        let solution = solve_facility_location_with_external_reference(
            &problem,
            &ExternalFacilityLocationReferenceOptions {
                solver: ExternalFacilityLocationReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalFacilityLocationReferenceStatus::Unavailable
        );
        assert!(solution
            .message
            .contains("OR-Tools facility-location adapter"));
    }

    #[test]
    fn ortools_adapter_reports_startup_without_repo_script() {
        let _lock = FACILITY_LOCATION_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("FACILITY_LOCATION_REFERENCE_FORCE_PYTHON", "1");
        let _guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-facility-location-ortools",
        );
        let problem = build_sample_facility_location_problem();

        let solution = solve_facility_location_with_external_reference(
            &problem,
            &ExternalFacilityLocationReferenceOptions {
                solver: ExternalFacilityLocationReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalFacilityLocationReferenceStatus::Unavailable
        );
        assert!(solution
            .message
            .contains("OR-Tools facility-location adapter"));
        assert!(!solution.message.contains("facility_location_reference.py"));
    }

    #[test]
    fn facility_location_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_facility_location_adapter_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
