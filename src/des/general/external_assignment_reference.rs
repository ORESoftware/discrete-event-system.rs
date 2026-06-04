//! Rust-facing bridge for external/reference assignment solvers.
//!
//! The native Rust reference computes an exact small assignment check without
//! Python startup. The checked-in Python bridge (`scripts/assignment_reference.py`)
//! remains available for OR-Tools SimpleLinearSumAssignment when costs can be
//! integer-scaled, and also records SciPy's
//! `linear_sum_assignment` result when available.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalAssignmentReferenceSolver {
    Auto,
    RustDp,
    OrTools,
    Scipy,
    Fallback,
}

impl ExternalAssignmentReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalAssignmentReferenceSolver::Auto => "auto",
            ExternalAssignmentReferenceSolver::RustDp => "rust-dp",
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

const RUST_ASSIGNMENT_EPS: f64 = 1e-12;
const RUST_ASSIGNMENT_MAX_COLUMNS: usize = 128;

fn validate_rust_assignment_cost(cost: &[Vec<f64>]) -> Result<(usize, usize), String> {
    if cost.is_empty() {
        return Err("cost matrix must be non-empty".to_string());
    }
    let cols = cost[0].len();
    if cols == 0 {
        return Err("cost matrix rows must be non-empty".to_string());
    }
    if cost.len() > cols {
        return Err("assignment bridge requires rows <= columns".to_string());
    }
    for (row_index, row) in cost.iter().enumerate() {
        if row.len() != cols {
            return Err(format!(
                "cost row {row_index} length {} != {cols}",
                row.len()
            ));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(format!("cost row {row_index} contains a non-finite value"));
        }
    }
    Ok((cost.len(), cols))
}

fn assignment_empty_solution(
    status: ExternalAssignmentReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalAssignmentReferenceSolution {
    ExternalAssignmentReferenceSolution {
        status,
        solver: solver.into(),
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

fn rust_assignment_dp(
    cost: &[Vec<f64>],
    row: usize,
    used_mask: u128,
    memo: &mut HashMap<(usize, u128), (f64, Vec<i64>)>,
) -> (f64, Vec<i64>) {
    if row == cost.len() {
        return (0.0, Vec::new());
    }
    if let Some(cached) = memo.get(&(row, used_mask)) {
        return cached.clone();
    }

    let mut best_cost = f64::INFINITY;
    let mut best_assignment = Vec::<i64>::new();
    for col in 0..cost[row].len() {
        if used_mask & (1_u128 << col) != 0 {
            continue;
        }
        let (tail_cost, tail_assignment) =
            rust_assignment_dp(cost, row + 1, used_mask | (1_u128 << col), memo);
        let candidate_cost = cost[row][col] + tail_cost;
        let mut candidate_assignment = Vec::with_capacity(tail_assignment.len() + 1);
        candidate_assignment.push(col as i64);
        candidate_assignment.extend(tail_assignment);
        if candidate_cost < best_cost - RUST_ASSIGNMENT_EPS
            || ((candidate_cost - best_cost).abs() <= RUST_ASSIGNMENT_EPS
                && (best_assignment.is_empty() || candidate_assignment < best_assignment))
        {
            best_cost = candidate_cost;
            best_assignment = candidate_assignment;
        }
    }

    memo.insert((row, used_mask), (best_cost, best_assignment.clone()));
    (best_cost, best_assignment)
}

fn solve_assignment_with_rust_reference(cost: &[Vec<f64>]) -> ExternalAssignmentReferenceSolution {
    let started = Instant::now();
    let (_rows, cols) = match validate_rust_assignment_cost(cost) {
        Ok(shape) => shape,
        Err(message) => {
            return assignment_empty_solution(
                ExternalAssignmentReferenceStatus::NumericalError,
                "rust:assignment-dp",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if cols > RUST_ASSIGNMENT_MAX_COLUMNS {
        return assignment_empty_solution(
            ExternalAssignmentReferenceStatus::Unsupported,
            "rust:assignment-dp",
            format!(
                "Rust assignment DP supports <= {RUST_ASSIGNMENT_MAX_COLUMNS} columns, got {cols}"
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let mut memo = HashMap::new();
    let (objective, assignment) = rust_assignment_dp(cost, 0, 0, &mut memo);
    if !objective.is_finite() {
        return assignment_empty_solution(
            ExternalAssignmentReferenceStatus::Infeasible,
            "rust:assignment-dp",
            "no assignment",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    ExternalAssignmentReferenceSolution {
        status: ExternalAssignmentReferenceStatus::Optimal,
        solver: "rust:assignment-dp".to_string(),
        assignment,
        objective: Some(objective),
        ortools_status: None,
        ortools_assignment: Vec::new(),
        ortools_objective: None,
        scipy_status: None,
        scipy_assignment: Vec::new(),
        scipy_objective: None,
        message: "exact assignment dynamic program".to_string(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
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

fn assignment_reference_timeout_ms() -> u64 {
    std::env::var("ASSIGNMENT_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_assignment_reference_output(
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
            Err(err) => return Err(format!("failed to poll assignment_reference.py: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for assignment_reference.py: {err}"))
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
    let timeout_ms = assignment_reference_timeout_ms();
    let (mut output, timed_out) =
        match wait_for_assignment_reference_output(child, timeout_ms) {
            Ok(output) => output,
            Err(err) => {
                return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0);
            }
        };
    if timed_out {
        let timeout_message = format!(
            "assignment_reference.py timed out after {}ms",
            timeout_ms
        );
        if output.stderr.is_empty() {
            output.stderr = timeout_message.into_bytes();
        } else {
            let mut stderr = timeout_message.into_bytes();
            stderr.push(b'\n');
            stderr.extend(output.stderr);
            output.stderr = stderr;
        }
    }
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
    if matches!(
        opts.solver,
        ExternalAssignmentReferenceSolver::Auto
            | ExternalAssignmentReferenceSolver::RustDp
            | ExternalAssignmentReferenceSolver::Fallback
    ) {
        return solve_assignment_with_rust_reference(cost);
    }

    run_assignment_reference_json(
        json!({
            "cost": cost,
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_reference_solves_square_assignment() {
        let cost = vec![
            vec![8.0, 2.0, 5.0, 9.0],
            vec![6.0, 4.0, 7.0, 3.0],
            vec![5.0, 8.0, 1.0, 6.0],
            vec![7.0, 3.0, 4.0, 2.0],
        ];

        let solution = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions {
                solver: ExternalAssignmentReferenceSolver::RustDp,
            },
        );

        assert_eq!(solution.status, ExternalAssignmentReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:assignment-dp");
        assert_eq!(solution.assignment, vec![1, 0, 2, 3]);
        assert_eq!(solution.objective, Some(11.0));
        assert!(solution.ortools_status.is_none());
        assert!(solution.scipy_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_for_rectangular_assignment() {
        let cost = vec![vec![1.0, 1.0, 4.0], vec![1.0, 1.0, 2.0]];

        let solution = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions {
                solver: ExternalAssignmentReferenceSolver::Fallback,
            },
        );

        assert_eq!(solution.status, ExternalAssignmentReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:assignment-dp");
        assert_eq!(solution.assignment, vec![0, 1]);
        assert_eq!(solution.objective, Some(2.0));
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let cost = vec![vec![3.0, 1.0], vec![2.0, 4.0]];

        let solution = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalAssignmentReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:assignment-dp");
        assert_eq!(solution.assignment, vec![1, 0]);
        assert_eq!(solution.objective, Some(3.0));
    }

    #[test]
    fn assignment_python_bridge_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_assignment_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
