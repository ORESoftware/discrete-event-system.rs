//! Rust-facing bridge for external/reference weighted Max-SAT solvers.
//!
//! The native Rust reference computes a deterministic exact enumeration check
//! without Python startup. The Python bridge
//! (`scripts/weighted_max_sat_reference.py`) remains available for OR-Tools
//! CP-SAT.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
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

const RUST_WEIGHTED_MAX_SAT_MAX_EXACT_VARS: usize = 26;
const RUST_WEIGHTED_MAX_SAT_EPS: f64 = 1e-9;

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
                    "failed to poll weighted_max_sat_reference.py: {err}"
                ))
            }
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for weighted_max_sat_reference.py: {err}"))
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
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write weighted_max_sat_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = weighted_max_sat_reference_timeout_ms();
    let (output, timed_out) = match wait_for_weighted_max_sat_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0),
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("weighted_max_sat_reference.py timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; weighted_max_sat_reference.py timed out after {timeout_ms}ms")
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
                "failed to parse weighted_max_sat_reference.py output: {err}; stderr={}",
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
    if matches!(
        opts.solver,
        ExternalWeightedMaxSatReferenceSolver::Auto
            | ExternalWeightedMaxSatReferenceSolver::RustEnumeration
            | ExternalWeightedMaxSatReferenceSolver::Fallback
    ) {
        return solve_weighted_max_sat_with_rust_reference(problem);
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::weighted_max_sat::{
        build_sample_weighted_max_sat_problem, WeightedMaxSatClause,
    };

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
    fn weighted_max_sat_python_bridge_wait_enforces_timeout() {
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
