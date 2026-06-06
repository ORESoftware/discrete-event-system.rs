//! Rust-facing bridge for external/reference weighted Max-SAT solvers.
//!
//! The native Rust reference computes a deterministic exact enumeration check
//! without Python startup. Registered OR-Tools aliases default to that Rust
//! reference; explicit force-Python switches keep the inline OR-Tools adapter
//! available for compatibility validation.

use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::weighted_max_sat::{WeightedMaxSatClause, WeightedMaxSatProblem};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalWeightedMaxSatReferenceSolver {
    Auto,
    RustEnumeration,
    OrTools,
    Fallback,
}

impl ExternalWeightedMaxSatReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalWeightedMaxSatReferenceSolver::Auto => "auto",
            ExternalWeightedMaxSatReferenceSolver::RustEnumeration => "rust-enumeration",
            ExternalWeightedMaxSatReferenceSolver::OrTools => "ortools",
            ExternalWeightedMaxSatReferenceSolver::Fallback => "fallback",
        }
    }
}

fn weighted_max_sat_reference_force_python_value(value: &str) -> bool {
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

fn weighted_max_sat_python_reference_forced() -> bool {
    [
        "WEIGHTED_MAX_SAT_REFERENCE_FORCE_PYTHON",
        "WEIGHTED_MAX_SAT_REFERENCE_ORTOOLS_FORCE_PYTHON",
        "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
    ]
    .into_iter()
    .any(|key| {
        std::env::var(key)
            .map(|value| weighted_max_sat_reference_force_python_value(&value))
            .unwrap_or(false)
    })
}

fn should_use_rust_weighted_max_sat_reference(
    opts: &ExternalWeightedMaxSatReferenceOptions,
) -> bool {
    matches!(
        opts.solver,
        ExternalWeightedMaxSatReferenceSolver::Auto
            | ExternalWeightedMaxSatReferenceSolver::RustEnumeration
            | ExternalWeightedMaxSatReferenceSolver::Fallback
    )
}

fn should_use_registered_weighted_max_sat_fallback(
    opts: &ExternalWeightedMaxSatReferenceOptions,
) -> bool {
    matches!(opts.solver, ExternalWeightedMaxSatReferenceSolver::OrTools)
        && !weighted_max_sat_python_reference_forced()
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
    #[serde(rename = "objectiveBound")]
    objective_bound: Option<f64>,
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

const RUST_WEIGHTED_MAX_SAT_MAX_EXACT_VARS: usize = 26;
const RUST_WEIGHTED_MAX_SAT_EPS: f64 = 1e-9;
const ORTOOLS_INTEGER_SCALES: [i64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
const ORTOOLS_WEIGHTED_MAX_SAT_SOLVER: &str = "ortools:cp-sat-weighted-max-sat";

const ORTOOLS_WEIGHTED_MAX_SAT_ADAPTER: &str = r#"
import json
import sys

SOLVER = "ortools:cp-sat-weighted-max-sat"


def literal_satisfied(literal, assignment):
    value = assignment[abs(int(literal)) - 1]
    return value if int(literal) > 0 else not value


def clause_satisfied(clause, assignment):
    return any(literal_satisfied(literal, assignment) for literal in clause["literals"])


def evaluate(problem, assignment):
    satisfied_soft_weight = 0.0
    unsatisfied_soft_weight = 0.0
    satisfied_clause_ids = []
    violated_hard_clause_ids = []
    for clause in problem["clauses"]:
        if clause_satisfied(clause, assignment):
            satisfied_clause_ids.append(clause["id"])
            if not clause["hard"]:
                satisfied_soft_weight += float(clause["weight"])
        elif clause["hard"]:
            violated_hard_clause_ids.append(clause["id"])
        else:
            unsatisfied_soft_weight += float(clause["weight"])
    return {
        "satisfiedSoftWeight": satisfied_soft_weight,
        "unsatisfiedSoftWeight": unsatisfied_soft_weight,
        "satisfiedClauseIds": satisfied_clause_ids,
        "violatedHardClauseIds": violated_hard_clause_ids,
    }


def output(status, assignment=None, evaluation=None, objective_bound=None, message=""):
    result = {
        "status": status,
        "solver": SOLVER,
        "assignment": [] if assignment is None else assignment,
        "objective": None if evaluation is None else evaluation["satisfiedSoftWeight"],
        "satisfiedSoftWeight": None if evaluation is None else evaluation["satisfiedSoftWeight"],
        "unsatisfiedSoftWeight": None if evaluation is None else evaluation["unsatisfiedSoftWeight"],
        "satisfiedClauseIds": [] if evaluation is None else evaluation["satisfiedClauseIds"],
        "violatedHardClauseIds": [] if evaluation is None else evaluation["violatedHardClauseIds"],
        "message": message,
    }
    if objective_bound is not None:
        result["objectiveBound"] = objective_bound
    return result


try:
    from ortools.sat.python import cp_model
except Exception as exc:
    print(json.dumps(output("unavailable", None, None, None, str(exc))))
    sys.exit(0)


try:
    problem = json.load(sys.stdin)
    model = cp_model.CpModel()
    variables = [
        model.NewBoolVar(f"x{index + 1}")
        for index in range(int(problem["numVars"]))
    ]
    objective_terms = []
    for clause_index, clause in enumerate(problem["clauses"]):
        literals = [
            variables[abs(int(literal)) - 1]
            if int(literal) > 0
            else variables[abs(int(literal)) - 1].Not()
            for literal in clause["literals"]
        ]
        if clause["hard"]:
            model.AddBoolOr(literals)
        else:
            sat = model.NewBoolVar(f"soft_{clause_index}_{clause['id']}")
            model.AddBoolOr(literals + [sat.Not()])
            scaled_weight = int(clause["scaledWeight"])
            if scaled_weight > 0:
                objective_terms.append(scaled_weight * sat)
    model.Maximize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(json.dumps(output(
            "infeasible" if status_name == "infeasible" else status_name,
            None,
            None,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )))
        sys.exit(0)
    assignment = [bool(solver.Value(var)) for var in variables]
    evaluation = evaluate(problem, assignment)
    print(json.dumps(output(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        assignment,
        evaluation,
        solver.BestObjectiveBound() / float(problem["weightScale"]),
        f"OR-Tools CP-SAT status {status_name}",
    )))
except Exception as exc:
    print(json.dumps({
        "status": "error",
        "solver": SOLVER,
        "assignment": [],
        "objective": None,
        "satisfiedSoftWeight": None,
        "unsatisfiedSoftWeight": None,
        "satisfiedClauseIds": [],
        "violatedHardClauseIds": [],
        "message": str(exc),
    }))
    sys.exit(1)
"#;

#[derive(Clone, Debug)]
struct RustWeightedMaxSatEvaluation {
    satisfied_soft_weight: f64,
    unsatisfied_soft_weight: f64,
    satisfied_clause_ids: Vec<String>,
    violated_hard_clause_ids: Vec<String>,
}

fn validate_rust_weighted_max_sat_problem(problem: &WeightedMaxSatProblem) -> Result<(), String> {
    if problem.num_vars == 0 {
        return Err("numVars must be positive".to_string());
    }
    if problem.clauses.is_empty() {
        return Err("clauses must be non-empty".to_string());
    }
    let mut ids = HashSet::new();
    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        if clause.id.trim().is_empty() {
            return Err(format!("clauses[{clause_index}].id must be non-empty"));
        }
        if !ids.insert(clause.id.clone()) {
            return Err(format!("duplicate clause id {:?}", clause.id));
        }
        if clause.literals.is_empty() {
            return Err(format!(
                "clauses[{clause_index}].literals must be non-empty"
            ));
        }
        if !clause.weight.is_finite() || clause.weight < 0.0 {
            return Err(format!(
                "clauses[{clause_index}].weight must be finite and non-negative"
            ));
        }
        for &literal in &clause.literals {
            let variable = literal.unsigned_abs() as usize;
            if literal == 0 || variable == 0 || variable > problem.num_vars {
                return Err(format!(
                    "clauses[{clause_index}] literal {literal} outside [1, numVars]"
                ));
            }
        }
    }
    Ok(())
}

fn rust_weighted_max_sat_empty_solution(
    status: ExternalWeightedMaxSatReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedMaxSatReferenceSolution {
    ExternalWeightedMaxSatReferenceSolution {
        status,
        solver: solver.into(),
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

fn relabel_registered_weighted_max_sat_fallback(
    mut solution: ExternalWeightedMaxSatReferenceSolution,
    opts: &ExternalWeightedMaxSatReferenceOptions,
) -> ExternalWeightedMaxSatReferenceSolution {
    if should_use_registered_weighted_max_sat_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-weighted-max-sat-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
}

fn rust_weighted_max_sat_literal_satisfied(literal: i64, assignment: &[bool]) -> bool {
    let value = assignment[(literal.unsigned_abs() as usize) - 1];
    if literal > 0 {
        value
    } else {
        !value
    }
}

fn rust_weighted_max_sat_clause_satisfied(
    clause: &WeightedMaxSatClause,
    assignment: &[bool],
) -> bool {
    clause
        .literals
        .iter()
        .any(|&literal| rust_weighted_max_sat_literal_satisfied(literal, assignment))
}

fn rust_weighted_max_sat_evaluate(
    problem: &WeightedMaxSatProblem,
    assignment: &[bool],
) -> RustWeightedMaxSatEvaluation {
    let mut satisfied_soft_weight = 0.0;
    let mut unsatisfied_soft_weight = 0.0;
    let mut satisfied_clause_ids = Vec::new();
    let mut violated_hard_clause_ids = Vec::new();
    for clause in &problem.clauses {
        if rust_weighted_max_sat_clause_satisfied(clause, assignment) {
            satisfied_clause_ids.push(clause.id.clone());
            if !clause.hard {
                satisfied_soft_weight += clause.weight;
            }
        } else if clause.hard {
            violated_hard_clause_ids.push(clause.id.clone());
        } else {
            unsatisfied_soft_weight += clause.weight;
        }
    }
    RustWeightedMaxSatEvaluation {
        satisfied_soft_weight,
        unsatisfied_soft_weight,
        satisfied_clause_ids,
        violated_hard_clause_ids,
    }
}

fn rust_weighted_max_sat_solution(
    status: ExternalWeightedMaxSatReferenceStatus,
    assignment: Vec<bool>,
    evaluation: RustWeightedMaxSatEvaluation,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedMaxSatReferenceSolution {
    ExternalWeightedMaxSatReferenceSolution {
        status,
        solver: "rust:exact-weighted-max-sat".to_string(),
        assignment,
        objective: Some(evaluation.satisfied_soft_weight),
        satisfied_soft_weight: Some(evaluation.satisfied_soft_weight),
        unsatisfied_soft_weight: Some(evaluation.unsatisfied_soft_weight),
        satisfied_clause_ids: evaluation.satisfied_clause_ids,
        violated_hard_clause_ids: evaluation.violated_hard_clause_ids,
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

fn solve_weighted_max_sat_with_rust_reference(
    problem: &WeightedMaxSatProblem,
) -> ExternalWeightedMaxSatReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_weighted_max_sat_problem(problem) {
        return rust_weighted_max_sat_empty_solution(
            ExternalWeightedMaxSatReferenceStatus::NumericalError,
            "rust:exact-weighted-max-sat",
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    if problem.num_vars > RUST_WEIGHTED_MAX_SAT_MAX_EXACT_VARS {
        return rust_weighted_max_sat_empty_solution(
            ExternalWeightedMaxSatReferenceStatus::Unsupported,
            "rust:exact-weighted-max-sat",
            format!(
                "exact weighted Max-SAT only practical for <= {RUST_WEIGHTED_MAX_SAT_MAX_EXACT_VARS} variables, got {}",
                problem.num_vars
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let mut best_assignment = None;
    let mut best_evaluation = None;
    let total = 1usize << problem.num_vars;
    for mask in 0..total {
        let assignment = (0..problem.num_vars)
            .map(|var| ((mask >> var) & 1) == 1)
            .collect::<Vec<_>>();
        let evaluation = rust_weighted_max_sat_evaluate(problem, &assignment);
        if !evaluation.violated_hard_clause_ids.is_empty() {
            continue;
        }
        let better =
            best_evaluation
                .as_ref()
                .is_none_or(|current: &RustWeightedMaxSatEvaluation| {
                    evaluation.satisfied_soft_weight
                        > current.satisfied_soft_weight + RUST_WEIGHTED_MAX_SAT_EPS
                });
        if better {
            best_assignment = Some(assignment);
            best_evaluation = Some(evaluation);
        }
    }

    match (best_assignment, best_evaluation) {
        (Some(assignment), Some(evaluation)) => rust_weighted_max_sat_solution(
            ExternalWeightedMaxSatReferenceStatus::Optimal,
            assignment,
            evaluation,
            "exact weighted Max-SAT enumeration",
            started.elapsed().as_secs_f64() * 1000.0,
        ),
        _ => rust_weighted_max_sat_empty_solution(
            ExternalWeightedMaxSatReferenceStatus::Infeasible,
            "rust:exact-weighted-max-sat",
            "no assignment satisfies all hard clauses",
            started.elapsed().as_secs_f64() * 1000.0,
        ),
    }
}

fn ortools_empty_solution(
    status: ExternalWeightedMaxSatReferenceStatus,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedMaxSatReferenceSolution {
    rust_weighted_max_sat_empty_solution(
        status,
        ORTOOLS_WEIGHTED_MAX_SAT_SOLVER,
        message,
        elapsed_ms,
    )
}

fn scaled_ortools_weight(value: f64, scale: i64) -> Option<i64> {
    let scaled = value * scale as f64;
    if !scaled.is_finite() || scaled < 0.0 || scaled > i64::MAX as f64 {
        return None;
    }
    let rounded = scaled.round();
    if (rounded - scaled).abs() <= 1e-6 {
        Some(rounded as i64)
    } else {
        None
    }
}

fn choose_ortools_weight_scale(problem: &WeightedMaxSatProblem) -> Option<i64> {
    ORTOOLS_INTEGER_SCALES.into_iter().find(|scale| {
        problem
            .clauses
            .iter()
            .all(|clause| clause.hard || scaled_ortools_weight(clause.weight, *scale).is_some())
    })
}

fn ortools_weighted_max_sat_payload(problem: &WeightedMaxSatProblem, weight_scale: i64) -> Value {
    json!({
        "numVars": problem.num_vars,
        "weightScale": weight_scale,
        "clauses": problem.clauses.iter().map(|clause| json!({
            "id": clause.id,
            "literals": clause.literals,
            "weight": clause.weight,
            "scaledWeight": if clause.hard {
                0
            } else {
                scaled_ortools_weight(clause.weight, weight_scale)
                    .expect("weight scale chosen for weighted Max-SAT soft clauses")
            },
            "hard": clause.hard,
        })).collect::<Vec<_>>(),
    })
}

fn weighted_max_sat_reference_timeout_ms() -> u64 {
    std::env::var("WEIGHTED_MAX_SAT_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_weighted_max_sat_reference_output(
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
                    "failed to poll OR-Tools weighted-max-sat adapter: {err}"
                ))
            }
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for OR-Tools weighted-max-sat adapter: {err}"))
}

fn run_ortools_weighted_max_sat_reference(
    problem: &WeightedMaxSatProblem,
) -> ExternalWeightedMaxSatReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_weighted_max_sat_problem(problem) {
        return ortools_empty_solution(
            ExternalWeightedMaxSatReferenceStatus::NumericalError,
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let Some(weight_scale) = choose_ortools_weight_scale(problem) else {
        return ortools_empty_solution(
            ExternalWeightedMaxSatReferenceStatus::Unsupported,
            "OR-Tools CP-SAT bridge requires integer-scalable soft clause weights",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    };
    let payload = ortools_weighted_max_sat_payload(problem, weight_scale);
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ORTOOLS_WEIGHTED_MAX_SAT_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return ortools_empty_solution(
                ExternalWeightedMaxSatReferenceStatus::Unavailable,
                format!("failed to start OR-Tools weighted-max-sat adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return ortools_empty_solution(
                ExternalWeightedMaxSatReferenceStatus::NumericalError,
                format!("failed to write OR-Tools weighted-max-sat adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = weighted_max_sat_reference_timeout_ms();
    let (output, timed_out) = match wait_for_weighted_max_sat_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => {
            return ortools_empty_solution(
                ExternalWeightedMaxSatReferenceStatus::NumericalError,
                err,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("OR-Tools weighted-max-sat adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; OR-Tools weighted-max-sat adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
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
            ortools_objective_bound: parsed.ortools_objective_bound.or(parsed.objective_bound),
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    stderr.clone()
                }
            }),
            elapsed_ms,
        },
        Err(err) => ortools_empty_solution(
            ExternalWeightedMaxSatReferenceStatus::NumericalError,
            format!(
                "failed to parse OR-Tools weighted-max-sat adapter output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_weighted_max_sat_with_external_reference(
    problem: &WeightedMaxSatProblem,
    opts: &ExternalWeightedMaxSatReferenceOptions,
) -> ExternalWeightedMaxSatReferenceSolution {
    if should_use_rust_weighted_max_sat_reference(opts)
        || should_use_registered_weighted_max_sat_fallback(opts)
    {
        return relabel_registered_weighted_max_sat_fallback(
            solve_weighted_max_sat_with_rust_reference(problem),
            opts,
        );
    }

    run_ortools_weighted_max_sat_reference(problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::weighted_max_sat::{
        build_sample_weighted_max_sat_problem, WeightedMaxSatClause,
    };
    use std::sync::Mutex;

    static WEIGHTED_MAX_SAT_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn weighted_max_sat_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "WEIGHTED_MAX_SAT_REFERENCE_FORCE_PYTHON",
            "WEIGHTED_MAX_SAT_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    #[test]
    fn rust_reference_solves_sample_weighted_max_sat() {
        let problem = build_sample_weighted_max_sat_problem();
        let solution = solve_weighted_max_sat_with_external_reference(
            &problem,
            &ExternalWeightedMaxSatReferenceOptions {
                solver: ExternalWeightedMaxSatReferenceSolver::RustEnumeration,
            },
        );

        assert_eq!(
            solution.status,
            ExternalWeightedMaxSatReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:exact-weighted-max-sat");
        assert_eq!(solution.objective, Some(16.0));
        assert_eq!(solution.satisfied_soft_weight, Some(16.0));
        assert_eq!(solution.assignment, vec![true, true, true]);
        assert!(solution.violated_hard_clause_ids.is_empty());
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_for_infeasible_hard_clauses() {
        let problem = WeightedMaxSatProblem {
            num_vars: 1,
            clauses: vec![
                WeightedMaxSatClause {
                    id: "must_be_true".to_string(),
                    literals: vec![1],
                    weight: 0.0,
                    hard: true,
                },
                WeightedMaxSatClause {
                    id: "must_be_false".to_string(),
                    literals: vec![-1],
                    weight: 0.0,
                    hard: true,
                },
            ],
        };

        let solution = solve_weighted_max_sat_with_external_reference(
            &problem,
            &ExternalWeightedMaxSatReferenceOptions {
                solver: ExternalWeightedMaxSatReferenceSolver::Fallback,
            },
        );

        assert_eq!(
            solution.status,
            ExternalWeightedMaxSatReferenceStatus::Infeasible
        );
        assert_eq!(solution.solver, "rust:exact-weighted-max-sat");
        assert!(solution.assignment.is_empty());
        assert!(solution.objective.is_none());
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_sample_weighted_max_sat_problem();

        let solution = solve_weighted_max_sat_with_external_reference(
            &problem,
            &ExternalWeightedMaxSatReferenceOptions::default(),
        );

        assert_eq!(
            solution.status,
            ExternalWeightedMaxSatReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:exact-weighted-max-sat");
        assert_eq!(solution.objective, Some(16.0));
        assert_eq!(solution.assignment, vec![true, true, true]);
    }

    #[test]
    fn registered_ortools_alias_defaults_to_rust_reference_without_python() {
        let _lock = WEIGHTED_MAX_SAT_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guards = weighted_max_sat_force_python_off_guards();
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-weighted-max-sat-alias",
        );
        let problem = build_sample_weighted_max_sat_problem();

        let solution = solve_weighted_max_sat_with_external_reference(
            &problem,
            &ExternalWeightedMaxSatReferenceOptions {
                solver: ExternalWeightedMaxSatReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalWeightedMaxSatReferenceStatus::Optimal
        );
        assert_eq!(
            solution.solver,
            "rust:registered-weighted-max-sat-fallback-for-ortools"
        );
        assert_eq!(solution.objective, Some(16.0));
        assert_eq!(solution.assignment, vec![true, true, true]);
        assert!(solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
    }

    #[test]
    fn weighted_max_sat_force_python_keeps_ortools_bridge_available() {
        let _lock = WEIGHTED_MAX_SAT_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("WEIGHTED_MAX_SAT_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-forced-weighted-max-sat",
        );
        let problem = build_sample_weighted_max_sat_problem();

        let solution = solve_weighted_max_sat_with_external_reference(
            &problem,
            &ExternalWeightedMaxSatReferenceOptions {
                solver: ExternalWeightedMaxSatReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalWeightedMaxSatReferenceStatus::Unavailable
        );
        assert_eq!(solution.solver, "ortools:cp-sat-weighted-max-sat");
        assert!(solution
            .message
            .contains("OR-Tools weighted-max-sat adapter"));
    }

    #[test]
    fn ortools_adapter_rejects_unscaled_weights_without_python() {
        let _lock = WEIGHTED_MAX_SAT_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("WEIGHTED_MAX_SAT_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let problem = WeightedMaxSatProblem {
            num_vars: 1,
            clauses: vec![WeightedMaxSatClause {
                id: "third".to_string(),
                literals: vec![1],
                weight: 1.0 / 3.0,
                hard: false,
            }],
        };

        let solution = solve_weighted_max_sat_with_external_reference(
            &problem,
            &ExternalWeightedMaxSatReferenceOptions {
                solver: ExternalWeightedMaxSatReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalWeightedMaxSatReferenceStatus::Unsupported
        );
        assert_eq!(solution.solver, "ortools:cp-sat-weighted-max-sat");
        assert!(solution
            .message
            .contains("requires integer-scalable soft clause weights"));
    }

    #[test]
    fn ortools_adapter_reports_startup_without_repo_script() {
        let _lock = WEIGHTED_MAX_SAT_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("WEIGHTED_MAX_SAT_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let problem = build_sample_weighted_max_sat_problem();

        let solution = solve_weighted_max_sat_with_external_reference(
            &problem,
            &ExternalWeightedMaxSatReferenceOptions {
                solver: ExternalWeightedMaxSatReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalWeightedMaxSatReferenceStatus::Unavailable
        );
        assert_eq!(solution.solver, "ortools:cp-sat-weighted-max-sat");
        assert!(solution
            .message
            .contains("OR-Tools weighted-max-sat adapter"));
        assert!(!solution.message.contains("weighted_max_sat_reference.py"));
    }

    #[test]
    fn weighted_max_sat_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_weighted_max_sat_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
