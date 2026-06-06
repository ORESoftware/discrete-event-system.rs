//! Rust-facing bridge for external/reference assignment solvers.
//!
//! The native Rust reference computes an exact small assignment check without
//! Python startup. Registered OR-Tools/SciPy aliases default to that Rust
//! reference; explicit force-Python switches keep the inline external adapters
//! available for compatibility validation.

use std::collections::HashMap;
use std::io::Write;
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

fn assignment_reference_force_python_value(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    matches!(
        normalized.as_str(),
        "1" | "true"
            | "yes"
            | "y"
            | "on"
            | "bridge"
            | "legacy-python"
            | "python-reference"
            | "python-bridge"
            | "legacy"
            | "compat"
            | "compatibility"
    )
}

fn assignment_python_reference_forced(solver: ExternalAssignmentReferenceSolver) -> bool {
    let solver_force_key = match solver {
        ExternalAssignmentReferenceSolver::OrTools => {
            Some("ASSIGNMENT_REFERENCE_ORTOOLS_FORCE_PYTHON")
        }
        ExternalAssignmentReferenceSolver::Scipy => Some("ASSIGNMENT_REFERENCE_SCIPY_FORCE_PYTHON"),
        _ => None,
    };
    [
        Some("ASSIGNMENT_REFERENCE_FORCE_PYTHON"),
        solver_force_key,
        Some("ORES_EXTERNAL_REFERENCE_FORCE_PYTHON"),
    ]
    .into_iter()
    .flatten()
    .any(|key| {
        std::env::var(key)
            .map(|value| assignment_reference_force_python_value(&value))
            .unwrap_or(false)
    })
}

fn should_use_rust_assignment_reference(opts: &ExternalAssignmentReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalAssignmentReferenceSolver::Auto
            | ExternalAssignmentReferenceSolver::RustDp
            | ExternalAssignmentReferenceSolver::Fallback
    )
}

fn should_use_registered_assignment_fallback(opts: &ExternalAssignmentReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalAssignmentReferenceSolver::OrTools | ExternalAssignmentReferenceSolver::Scipy
    ) && !assignment_python_reference_forced(opts.solver)
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
const ASSIGNMENT_INTEGER_SCALES: [i64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
const ORTOOLS_ASSIGNMENT_SOLVER: &str = "ortools:simple-linear-sum-assignment";
const SCIPY_ASSIGNMENT_SOLVER: &str = "scipy:linear_sum_assignment";

const ASSIGNMENT_EXTERNAL_ADAPTER: &str = r#"
import json
import sys

ORTOOLS_SOLVER = "ortools:simple-linear-sum-assignment"
SCIPY_SOLVER = "scipy:linear_sum_assignment"


def result(status, solver, assignment=None, objective=None, message=""):
    return {
        "status": status,
        "solver": solver,
        "assignment": [] if assignment is None else [int(value) for value in assignment],
        "objective": None if objective is None else float(objective),
        "message": message,
    }


def status_name(status):
    return str(status).split(".")[-1].lower()


def solve_ortools(problem):
    try:
        from ortools.graph.python import linear_sum_assignment
    except Exception as exc:
        return result("unavailable", ORTOOLS_SOLVER, message=str(exc))

    cost = problem["cost"]
    scaled_cost = problem["scaledCost"]
    scale = float(problem["costScale"])
    solver = linear_sum_assignment.SimpleLinearSumAssignment()
    for row, values in enumerate(scaled_cost):
        for col, value in enumerate(values):
            solver.add_arc_with_cost(row, col, int(value))
    status = solver.solve()
    if status != solver.OPTIMAL:
        mapped = status_name(status)
        return result(
            "infeasible" if mapped == "infeasible" else mapped,
            ORTOOLS_SOLVER,
            message=f"OR-Tools SimpleLinearSumAssignment status {mapped}",
        )
    assignment = [int(solver.right_mate(row)) for row in range(len(cost))]
    objective = sum(float(cost[row][col]) for row, col in enumerate(assignment))
    scaled_objective = solver.optimal_cost() / scale
    return result(
        "optimal",
        ORTOOLS_SOLVER,
        assignment=assignment,
        objective=objective if abs(objective - scaled_objective) <= 1e-7 else scaled_objective,
        message="OR-Tools SimpleLinearSumAssignment",
    )


def solve_scipy(problem):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:
        return result("unavailable", SCIPY_SOLVER, message=str(exc))

    cost = problem["cost"]
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = [-1 for _ in cost]
    objective = 0.0
    for row, col in zip(row_ind, col_ind):
        assignment[int(row)] = int(col)
        objective += float(cost[int(row)][int(col)])
    if any(col < 0 for col in assignment):
        return result("infeasible", SCIPY_SOLVER, message="not all rows assigned")
    return result(
        "optimal",
        SCIPY_SOLVER,
        assignment=assignment,
        objective=objective,
        message="SciPy linear_sum_assignment",
    )


try:
    problem = json.load(sys.stdin)
    solver = problem["solver"]
    if solver == "ortools":
        print(json.dumps(solve_ortools(problem)))
    elif solver == "scipy":
        print(json.dumps(solve_scipy(problem)))
    else:
        print(json.dumps(result("error", "assignment-reference", message=f"unknown solver {solver}")))
        sys.exit(1)
except Exception as exc:
    print(json.dumps(result("error", "assignment-reference", message=str(exc))))
    sys.exit(1)
"#;

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

fn relabel_registered_assignment_fallback(
    mut solution: ExternalAssignmentReferenceSolution,
    opts: &ExternalAssignmentReferenceOptions,
) -> ExternalAssignmentReferenceSolution {
    if should_use_registered_assignment_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-assignment-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
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

fn assignment_external_solver_name(opts: &ExternalAssignmentReferenceOptions) -> &'static str {
    match opts.solver {
        ExternalAssignmentReferenceSolver::OrTools => ORTOOLS_ASSIGNMENT_SOLVER,
        ExternalAssignmentReferenceSolver::Scipy => SCIPY_ASSIGNMENT_SOLVER,
        _ => "external-assignment-reference",
    }
}

fn assignment_adapter_empty_solution(
    status: ExternalAssignmentReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalAssignmentReferenceSolution {
    assignment_empty_solution(status, solver, message, elapsed_ms)
}

fn scaled_assignment_cost(value: f64, scale: i64) -> Option<i64> {
    let scaled = value * scale as f64;
    if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
        return None;
    }
    let rounded = scaled.round();
    if (rounded - scaled).abs() <= 1e-6 {
        Some(rounded as i64)
    } else {
        None
    }
}

fn choose_assignment_cost_scale(cost: &[Vec<f64>]) -> Option<i64> {
    ASSIGNMENT_INTEGER_SCALES.into_iter().find(|scale| {
        cost.iter().all(|row| {
            row.iter()
                .all(|&value| scaled_assignment_cost(value, *scale).is_some())
        })
    })
}

fn assignment_external_payload(
    cost: &[Vec<f64>],
    opts: &ExternalAssignmentReferenceOptions,
    cost_scale: Option<i64>,
) -> Value {
    let scaled_cost = cost_scale.map(|scale| {
        cost.iter()
            .map(|row| {
                row.iter()
                    .map(|&value| {
                        scaled_assignment_cost(value, scale)
                            .expect("cost scale chosen for assignment costs")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    json!({
        "solver": opts.solver.as_arg(),
        "cost": cost,
        "costScale": cost_scale.unwrap_or(1),
        "scaledCost": scaled_cost.unwrap_or_default(),
    })
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
            Err(err) => return Err(format!("failed to poll external assignment adapter: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for external assignment adapter: {err}"))
}

fn run_assignment_external_reference(
    cost: &[Vec<f64>],
    opts: &ExternalAssignmentReferenceOptions,
) -> ExternalAssignmentReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_assignment_cost(cost) {
        return assignment_adapter_empty_solution(
            ExternalAssignmentReferenceStatus::NumericalError,
            assignment_external_solver_name(opts),
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let cost_scale = if matches!(opts.solver, ExternalAssignmentReferenceSolver::OrTools) {
        let Some(cost_scale) = choose_assignment_cost_scale(cost) else {
            return assignment_adapter_empty_solution(
                ExternalAssignmentReferenceStatus::Unsupported,
                ORTOOLS_ASSIGNMENT_SOLVER,
                "OR-Tools SimpleLinearSumAssignment requires integer-scalable costs",
                started.elapsed().as_secs_f64() * 1000.0,
            );
        };
        Some(cost_scale)
    } else {
        None
    };
    let payload = assignment_external_payload(cost, opts, cost_scale);
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ASSIGNMENT_EXTERNAL_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return assignment_adapter_empty_solution(
                ExternalAssignmentReferenceStatus::Unavailable,
                assignment_external_solver_name(opts),
                format!("failed to start external assignment adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return assignment_adapter_empty_solution(
                ExternalAssignmentReferenceStatus::NumericalError,
                assignment_external_solver_name(opts),
                format!("failed to write external assignment adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    drop(child.stdin.take());
    let timeout_ms = assignment_reference_timeout_ms();
    let (mut output, timed_out) = match wait_for_assignment_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => {
            return assignment_adapter_empty_solution(
                ExternalAssignmentReferenceStatus::NumericalError,
                assignment_external_solver_name(opts),
                err,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    };
    if timed_out {
        let timeout_message = format!(
            "external assignment adapter timed out after {}ms",
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
        Err(err) => assignment_adapter_empty_solution(
            ExternalAssignmentReferenceStatus::NumericalError,
            assignment_external_solver_name(opts),
            format!(
                "failed to parse external assignment adapter output: {err}; stderr={}",
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
    if should_use_rust_assignment_reference(opts) || should_use_registered_assignment_fallback(opts)
    {
        return relabel_registered_assignment_fallback(
            solve_assignment_with_rust_reference(cost),
            opts,
        );
    }

    run_assignment_external_reference(cost, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ASSIGNMENT_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn assignment_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "ASSIGNMENT_REFERENCE_FORCE_PYTHON",
            "ASSIGNMENT_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ASSIGNMENT_REFERENCE_SCIPY_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    #[test]
    fn assignment_force_python_requires_explicit_compatibility_value() {
        for value in [
            "1",
            "true",
            " yes ",
            "ON",
            "bridge",
            "python_reference",
            "python-bridge",
            "legacy-python",
            "legacy",
            "compatibility",
        ] {
            assert!(
                assignment_reference_force_python_value(value),
                "{value:?} should enable the assignment compatibility bridge"
            );
        }

        for value in [
            "", "0", "false", "off", "python", "py", "auto", "rust", "native",
        ] {
            assert!(
                !assignment_reference_force_python_value(value),
                "{value:?} should keep Rust assignment fallback active"
            );
        }
    }

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
    fn registered_solver_aliases_default_to_rust_reference_without_python() {
        let _lock = ASSIGNMENT_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guards = assignment_force_python_off_guards();
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-assignment-aliases",
        );
        let cost = vec![vec![3.0, 1.0], vec![2.0, 4.0]];

        let ortools = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions {
                solver: ExternalAssignmentReferenceSolver::OrTools,
            },
        );
        assert_eq!(ortools.status, ExternalAssignmentReferenceStatus::Optimal);
        assert_eq!(
            ortools.solver,
            "rust:registered-assignment-fallback-for-ortools"
        );
        assert_eq!(ortools.assignment, vec![1, 0]);
        assert_eq!(ortools.objective, Some(3.0));
        assert!(ortools
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));

        let scipy = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions {
                solver: ExternalAssignmentReferenceSolver::Scipy,
            },
        );
        assert_eq!(scipy.status, ExternalAssignmentReferenceStatus::Optimal);
        assert_eq!(
            scipy.solver,
            "rust:registered-assignment-fallback-for-scipy"
        );
        assert_eq!(scipy.assignment, vec![1, 0]);
        assert_eq!(scipy.objective, Some(3.0));
    }

    #[test]
    fn assignment_force_python_keeps_external_adapters_available() {
        let _lock = ASSIGNMENT_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("ASSIGNMENT_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-forced-assignment");
        let cost = vec![vec![3.0, 1.0], vec![2.0, 4.0]];

        let ortools = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions {
                solver: ExternalAssignmentReferenceSolver::OrTools,
            },
        );
        let scipy = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions {
                solver: ExternalAssignmentReferenceSolver::Scipy,
            },
        );

        assert_eq!(
            ortools.status,
            ExternalAssignmentReferenceStatus::Unavailable
        );
        assert_eq!(ortools.solver, "ortools:simple-linear-sum-assignment");
        assert!(ortools.message.contains("external assignment adapter"));
        assert_eq!(scipy.status, ExternalAssignmentReferenceStatus::Unavailable);
        assert_eq!(scipy.solver, "scipy:linear_sum_assignment");
        assert!(scipy.message.contains("external assignment adapter"));
    }

    #[test]
    fn ortools_adapter_rejects_unscaled_costs_without_python() {
        let _lock = ASSIGNMENT_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard =
            EnvVarGuard::set("ASSIGNMENT_REFERENCE_ORTOOLS_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let cost = vec![vec![1.0 / 3.0]];

        let solution = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions {
                solver: ExternalAssignmentReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalAssignmentReferenceStatus::Unsupported
        );
        assert_eq!(solution.solver, "ortools:simple-linear-sum-assignment");
        assert!(solution.message.contains("integer-scalable costs"));
    }

    #[test]
    fn external_adapters_report_startup_without_repo_script() {
        let _lock = ASSIGNMENT_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("ASSIGNMENT_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let cost = vec![vec![3.0, 1.0], vec![2.0, 4.0]];

        let ortools = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions {
                solver: ExternalAssignmentReferenceSolver::OrTools,
            },
        );
        assert_eq!(
            ortools.status,
            ExternalAssignmentReferenceStatus::Unavailable
        );
        assert_eq!(ortools.solver, "ortools:simple-linear-sum-assignment");
        assert!(ortools.message.contains("external assignment adapter"));
        assert!(!ortools.message.contains("assignment_reference.py"));

        let scipy = solve_assignment_with_external_reference(
            &cost,
            &ExternalAssignmentReferenceOptions {
                solver: ExternalAssignmentReferenceSolver::Scipy,
            },
        );
        assert_eq!(scipy.status, ExternalAssignmentReferenceStatus::Unavailable);
        assert_eq!(scipy.solver, "scipy:linear_sum_assignment");
        assert!(scipy.message.contains("external assignment adapter"));
        assert!(!scipy.message.contains("assignment_reference.py"));
    }

    #[test]
    fn assignment_adapter_wait_enforces_timeout() {
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

    #[test]
    fn assignment_adapter_wait_observes_closed_stdin() {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg("cat >/dev/null; printf done")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn stdin reader");
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(b"{\"cost\":[[1]]}")
            .expect("write stdin");
        drop(child.stdin.take());

        let (output, timed_out) =
            wait_for_assignment_reference_output(child, 1_000).expect("closed stdin output");

        assert!(!timed_out);
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout), "done");
    }
}
