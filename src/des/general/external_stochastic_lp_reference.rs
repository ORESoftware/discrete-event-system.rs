//! Rust-facing bridge for external/reference stochastic LP solvers.
//!
//! The native Rust reference builds the extensive-form sample-average LP and
//! solves it through the Rust LP stack without Python startup. Registered
//! SciPy aliases default to that Rust reference; explicit force-Python switches
//! keep the inline SciPy/HiGHS adapter available for compatibility validation.

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::stochastic_lp::{solve_slp_monolithic, SLPProblem, SLPStatus, Scenario};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalStochasticLpReferenceSolver {
    Auto,
    RustMonolithic,
    Scipy,
    Fallback,
}

impl ExternalStochasticLpReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalStochasticLpReferenceSolver::Auto => "auto",
            ExternalStochasticLpReferenceSolver::RustMonolithic => "rust-monolithic",
            ExternalStochasticLpReferenceSolver::Scipy => "scipy",
            ExternalStochasticLpReferenceSolver::Fallback => "fallback",
        }
    }
}

fn stochastic_lp_reference_force_python_value(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    matches!(
        normalized.as_str(),
        "1" | "true"
            | "yes"
            | "y"
            | "on"
            | "bridge"
            | "scipy"
            | "external"
            | "legacy-python"
            | "python-reference"
            | "python-bridge"
            | "legacy"
            | "compat"
            | "compatibility"
    )
}

fn stochastic_lp_python_reference_forced() -> bool {
    [
        "STOCHASTIC_LP_REFERENCE_FORCE_PYTHON",
        "STOCHASTIC_LP_REFERENCE_SCIPY_FORCE_PYTHON",
        "STOCHASTIC_LP_EXTERNAL_BRIDGE",
        "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
    ]
    .into_iter()
    .any(|key| {
        std::env::var(key)
            .map(|value| stochastic_lp_reference_force_python_value(&value))
            .unwrap_or(false)
    })
}

fn should_use_rust_stochastic_lp_reference(opts: &ExternalStochasticLpReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalStochasticLpReferenceSolver::Auto
            | ExternalStochasticLpReferenceSolver::RustMonolithic
            | ExternalStochasticLpReferenceSolver::Fallback
    )
}

fn should_use_registered_stochastic_lp_fallback(
    opts: &ExternalStochasticLpReferenceOptions,
) -> bool {
    matches!(opts.solver, ExternalStochasticLpReferenceSolver::Scipy)
        && !stochastic_lp_python_reference_forced()
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalStochasticLpReferenceOptions {
    pub solver: ExternalStochasticLpReferenceSolver,
}

impl Default for ExternalStochasticLpReferenceOptions {
    fn default() -> Self {
        ExternalStochasticLpReferenceOptions {
            solver: ExternalStochasticLpReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalStochasticLpReferenceStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterLimit,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalStochasticLpReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalStochasticLpReferenceStatus::Optimal => "optimal",
            ExternalStochasticLpReferenceStatus::Infeasible => "infeasible",
            ExternalStochasticLpReferenceStatus::Unbounded => "unbounded",
            ExternalStochasticLpReferenceStatus::IterLimit => "iteration-limit",
            ExternalStochasticLpReferenceStatus::Unsupported => "unsupported",
            ExternalStochasticLpReferenceStatus::NumericalError => "numerical-error",
            ExternalStochasticLpReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalStochasticLpReferenceSolution {
    pub status: ExternalStochasticLpReferenceStatus,
    pub solver: String,
    pub x: Vec<f64>,
    pub objective: Option<f64>,
    pub c_first_x: Option<f64>,
    pub expected_q: Option<f64>,
    pub y_by_scenario: Vec<Vec<f64>>,
    pub scenario_values: Vec<f64>,
    pub iterations: Option<u64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct StochasticLpReferencePayload {
    status: String,
    solver: Option<String>,
    x: Option<Vec<f64>>,
    objective: Option<f64>,
    #[serde(rename = "cFirstX")]
    c_first_x: Option<f64>,
    #[serde(rename = "expectedQ")]
    expected_q: Option<f64>,
    #[serde(rename = "yByScenario")]
    y_by_scenario: Option<Vec<Vec<f64>>>,
    #[serde(rename = "scenarioValues")]
    scenario_values: Option<Vec<f64>>,
    iterations: Option<u64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalStochasticLpReferenceStatus {
    match status {
        "optimal" => ExternalStochasticLpReferenceStatus::Optimal,
        "infeasible" => ExternalStochasticLpReferenceStatus::Infeasible,
        "unbounded" => ExternalStochasticLpReferenceStatus::Unbounded,
        "iteration-limit" => ExternalStochasticLpReferenceStatus::IterLimit,
        "unsupported" => ExternalStochasticLpReferenceStatus::Unsupported,
        "unavailable" => ExternalStochasticLpReferenceStatus::Unavailable,
        _ => ExternalStochasticLpReferenceStatus::NumericalError,
    }
}

fn status_from_slp_status(status: SLPStatus) -> ExternalStochasticLpReferenceStatus {
    match status {
        SLPStatus::Optimal => ExternalStochasticLpReferenceStatus::Optimal,
        SLPStatus::Infeasible => ExternalStochasticLpReferenceStatus::Infeasible,
        SLPStatus::Unbounded => ExternalStochasticLpReferenceStatus::Unbounded,
        SLPStatus::IterLimit => ExternalStochasticLpReferenceStatus::IterLimit,
    }
}

fn validate_rust_stochastic_lp_problem(
    problem: &SLPProblem,
    scenarios: &[Scenario],
) -> Result<(), String> {
    let n_first = problem.c_first.len();
    let n_second = problem.q_second.len();
    if n_first == 0 || n_second == 0 {
        return Err("cFirst and qSecond must be non-empty".to_string());
    }
    if problem.c_first.iter().any(|value| !value.is_finite())
        || problem.q_second.iter().any(|value| !value.is_finite())
    {
        return Err("objective coefficients must be finite".to_string());
    }
    if problem.a_first.len() != problem.b_first.len() {
        return Err(format!(
            "aFirst rows {} != bFirst length {}",
            problem.a_first.len(),
            problem.b_first.len()
        ));
    }
    for (row_index, row) in problem.a_first.iter().enumerate() {
        if row.len() != n_first {
            return Err(format!(
                "aFirst[{row_index}] length {} != {n_first}",
                row.len()
            ));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(format!("aFirst[{row_index}] must contain finite numbers"));
        }
    }
    if problem.b_first.iter().any(|value| !value.is_finite()) {
        return Err("bFirst must contain finite numbers".to_string());
    }
    if problem.w_second.is_empty() {
        return Err("wSecond must be non-empty".to_string());
    }
    for (row_index, row) in problem.w_second.iter().enumerate() {
        if row.len() != n_second {
            return Err(format!(
                "wSecond[{row_index}] length {} != {n_second}",
                row.len()
            ));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(format!("wSecond[{row_index}] must contain finite numbers"));
        }
    }
    if scenarios.is_empty() {
        return Err("scenarios must be non-empty".to_string());
    }
    let default_prob = 1.0 / scenarios.len() as f64;
    let mut total_prob = 0.0;
    for (scenario_index, scenario) in scenarios.iter().enumerate() {
        if scenario.t.len() != problem.w_second.len() || scenario.h.len() != problem.w_second.len()
        {
            return Err(format!(
                "scenarios[{scenario_index}] must have {} recourse rows; got T={} h={}",
                problem.w_second.len(),
                scenario.t.len(),
                scenario.h.len()
            ));
        }
        for (row_index, row) in scenario.t.iter().enumerate() {
            if row.len() != n_first {
                return Err(format!(
                    "scenarios[{scenario_index}].t[{row_index}] length {} != {n_first}",
                    row.len()
                ));
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "scenarios[{scenario_index}].t[{row_index}] must contain finite numbers"
                ));
            }
        }
        if scenario.h.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "scenarios[{scenario_index}].h must contain finite numbers"
            ));
        }
        let probability = scenario.prob.unwrap_or(default_prob);
        if !probability.is_finite() || probability < 0.0 {
            return Err(format!(
                "scenarios[{scenario_index}].prob must be finite and non-negative"
            ));
        }
        total_prob += probability;
    }
    if (total_prob - 1.0).abs() > 1e-7 {
        return Err(format!(
            "scenario probabilities must sum to one, got {total_prob:.3e}"
        ));
    }
    Ok(())
}

fn rust_stochastic_lp_empty_solution(
    status: ExternalStochasticLpReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalStochasticLpReferenceSolution {
    ExternalStochasticLpReferenceSolution {
        status,
        solver: solver.into(),
        x: Vec::new(),
        objective: None,
        c_first_x: None,
        expected_q: None,
        y_by_scenario: Vec::new(),
        scenario_values: Vec::new(),
        iterations: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn relabel_registered_stochastic_lp_fallback(
    mut solution: ExternalStochasticLpReferenceSolution,
    opts: &ExternalStochasticLpReferenceOptions,
) -> ExternalStochasticLpReferenceSolution {
    if should_use_registered_stochastic_lp_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-stochastic-lp-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
}

fn solve_stochastic_lp_with_rust_reference(
    problem: &SLPProblem,
    scenarios: &[Scenario],
) -> ExternalStochasticLpReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_stochastic_lp_problem(problem, scenarios) {
        return rust_stochastic_lp_empty_solution(
            ExternalStochasticLpReferenceStatus::NumericalError,
            "rust:monolithic-slp",
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let solution = solve_slp_monolithic(problem.clone(), scenarios.to_vec());
    let status = status_from_slp_status(solution.status);
    if status != ExternalStochasticLpReferenceStatus::Optimal {
        return rust_stochastic_lp_empty_solution(
            status,
            "rust:monolithic-slp",
            format!("Rust monolithic SLP status {:?}", solution.status),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    ExternalStochasticLpReferenceSolution {
        status,
        solver: "rust:monolithic-slp".to_string(),
        x: solution.x,
        objective: Some(solution.objective),
        c_first_x: Some(solution.c_first_x),
        expected_q: Some(solution.expected_q),
        y_by_scenario: solution.y_by_scenario,
        scenario_values: solution.scenario_values,
        iterations: Some(solution.iterations as u64),
        message: "Rust extensive-form monolithic SAA".to_string(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalStochasticLpReferenceSolution {
    ExternalStochasticLpReferenceSolution {
        status: ExternalStochasticLpReferenceStatus::Unavailable,
        solver: "external-stochastic-lp-reference".to_string(),
        x: Vec::new(),
        objective: None,
        c_first_x: None,
        expected_q: None,
        y_by_scenario: Vec::new(),
        scenario_values: Vec::new(),
        iterations: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalStochasticLpReferenceSolution {
    ExternalStochasticLpReferenceSolution {
        status: ExternalStochasticLpReferenceStatus::NumericalError,
        solver: "external-stochastic-lp-reference".to_string(),
        x: Vec::new(),
        objective: None,
        c_first_x: None,
        expected_q: None,
        y_by_scenario: Vec::new(),
        scenario_values: Vec::new(),
        iterations: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn stochastic_lp_reference_timeout_ms() -> u64 {
    std::env::var("STOCHASTIC_LP_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_stochastic_lp_adapter_output(
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
            Err(err) => return Err(format!("failed to poll SciPy stochastic LP adapter: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for SciPy stochastic LP adapter: {err}"))
}

const SCIPY_STOCHASTIC_LP_ADAPTER: &str = r#"
import json
import sys

SOLVER = "scipy:highs-slp"

def emit(status, message, x=None, objective=None, c_first_x=None, expected_q=None, y_by_scenario=None, scenario_values=None, iterations=None):
    payload = {
        "status": status,
        "solver": SOLVER,
        "x": [] if x is None else x,
        "objective": objective,
        "cFirstX": c_first_x,
        "expectedQ": expected_q,
        "yByScenario": [] if y_by_scenario is None else y_by_scenario,
        "scenarioValues": [] if scenario_values is None else scenario_values,
        "iterations": iterations,
        "message": message,
    }
    print(json.dumps(payload, sort_keys=True))

try:
    from scipy import optimize
except Exception as exc:
    emit("unavailable", f"SciPy unavailable: {exc}")
    raise SystemExit(0)

def scipy_status(code):
    if code == 0:
        return "optimal"
    if code == 1:
        return "iteration-limit"
    if code == 2:
        return "infeasible"
    if code == 3:
        return "unbounded"
    return "numerical-error"

try:
    data = json.load(sys.stdin)
    c_first = [float(value) for value in data["cFirst"]]
    a_first = [[float(value) for value in row] for row in data.get("aFirst", [])]
    b_first = [float(value) for value in data.get("bFirst", [])]
    q_second = [float(value) for value in data["qSecond"]]
    w_second = [[float(value) for value in row] for row in data["wSecond"]]
    scenarios = data["scenarios"]
    n_first = len(c_first)
    n_second = len(q_second)
    total_vars = n_first + len(scenarios) * n_second

    c = [0.0 for _ in range(total_vars)]
    for j, value in enumerate(c_first):
        c[j] = -value
    default_prob = 1.0 / len(scenarios)
    for s, scenario in enumerate(scenarios):
        probability = float(scenario.get("prob", default_prob))
        for j, value in enumerate(q_second):
            c[n_first + s * n_second + j] = -probability * value

    a_ub = []
    b_ub = []
    for row, rhs in zip(a_first, b_first):
        out = [0.0 for _ in range(total_vars)]
        out[:n_first] = row
        a_ub.append(out)
        b_ub.append(rhs)

    for s, scenario in enumerate(scenarios):
        y_offset = n_first + s * n_second
        t_rows = [[float(value) for value in row] for row in scenario["t"]]
        h_values = [float(value) for value in scenario["h"]]
        for t_row, w_row, rhs in zip(t_rows, w_second, h_values):
            out = [0.0 for _ in range(total_vars)]
            out[:n_first] = t_row
            out[y_offset:y_offset + n_second] = w_row
            a_ub.append(out)
            b_ub.append(rhs)

    solution = optimize.linprog(
        c,
        A_ub=a_ub if a_ub else None,
        b_ub=b_ub if b_ub else None,
        bounds=[(0.0, None) for _ in range(total_vars)],
        method="highs",
    )
    status = scipy_status(int(solution.status))
    iterations = getattr(solution, "nit", None)
    if status != "optimal":
        emit(status, str(solution.message), iterations=iterations)
        raise SystemExit(0)

    values = [float(value) for value in solution.x]
    x = values[:n_first]
    y_by_scenario = []
    scenario_values = []
    for s in range(len(scenarios)):
        lo = n_first + s * n_second
        y = values[lo:lo + n_second]
        y_by_scenario.append(y)
        scenario_values.append(sum(q * yj for q, yj in zip(q_second, y)))

    c_first_x = sum(cj * xj for cj, xj in zip(c_first, x))
    expected_q = 0.0
    for scenario, value in zip(scenarios, scenario_values):
        expected_q += float(scenario.get("prob", default_prob)) * value
    emit(
        "optimal",
        str(solution.message),
        x=x,
        objective=c_first_x + expected_q,
        c_first_x=c_first_x,
        expected_q=expected_q,
        y_by_scenario=y_by_scenario,
        scenario_values=scenario_values,
        iterations=iterations,
    )
except Exception as exc:
    emit("numerical-error", str(exc))
    raise SystemExit(1)
"#;

fn stochastic_lp_payload(problem: &SLPProblem, scenarios: &[Scenario]) -> Result<Value, String> {
    validate_rust_stochastic_lp_problem(problem, scenarios)?;
    Ok(json!({
        "cFirst": &problem.c_first,
        "aFirst": &problem.a_first,
        "bFirst": &problem.b_first,
        "qSecond": &problem.q_second,
        "wSecond": &problem.w_second,
        "thetaLowerBound": problem.theta_lower_bound,
        "thetaUpperBound": problem.theta_upper_bound,
        "scenarios": scenarios.iter().map(|scenario| json!({
            "t": &scenario.t,
            "h": &scenario.h,
            "prob": scenario.prob,
        })).collect::<Vec<_>>(),
    }))
}

fn run_scipy_stochastic_lp_reference(
    problem: &SLPProblem,
    scenarios: &[Scenario],
) -> ExternalStochasticLpReferenceSolution {
    let started = Instant::now();
    let payload = match stochastic_lp_payload(problem, scenarios) {
        Ok(payload) => payload,
        Err(message) => return numerical_error(message, started.elapsed().as_secs_f64() * 1000.0),
    };
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(SCIPY_STOCHASTIC_LP_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return unavailable(
                format!("failed to start SciPy stochastic LP adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write SciPy stochastic LP adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = stochastic_lp_reference_timeout_ms();
    let (output, timed_out) = match wait_for_stochastic_lp_adapter_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0),
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("SciPy stochastic LP adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; SciPy stochastic LP adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
    match serde_json::from_slice::<StochasticLpReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalStochasticLpReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-stochastic-lp-reference".to_string()),
            x: parsed.x.unwrap_or_default(),
            objective: parsed.objective,
            c_first_x: parsed.c_first_x,
            expected_q: parsed.expected_q,
            y_by_scenario: parsed.y_by_scenario.unwrap_or_default(),
            scenario_values: parsed.scenario_values.unwrap_or_default(),
            iterations: parsed.iterations,
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
                "failed to parse SciPy stochastic LP adapter output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_stochastic_lp_with_external_reference(
    problem: &SLPProblem,
    scenarios: &[Scenario],
    opts: &ExternalStochasticLpReferenceOptions,
) -> ExternalStochasticLpReferenceSolution {
    if should_use_rust_stochastic_lp_reference(opts)
        || should_use_registered_stochastic_lp_fallback(opts)
    {
        return relabel_registered_stochastic_lp_fallback(
            solve_stochastic_lp_with_rust_reference(problem, scenarios),
            opts,
        );
    }

    run_scipy_stochastic_lp_reference(problem, scenarios)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::stochastic_lp::{
        build_production_scenarios, build_production_slp, UniformDemandSpec,
    };
    use std::sync::Mutex;

    static STOCHASTIC_LP_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn stochastic_lp_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "STOCHASTIC_LP_REFERENCE_FORCE_PYTHON",
            "STOCHASTIC_LP_REFERENCE_SCIPY_FORCE_PYTHON",
            "STOCHASTIC_LP_EXTERNAL_BRIDGE",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    #[test]
    fn stochastic_lp_force_python_requires_explicit_compatibility_value() {
        for value in [
            "1",
            "true",
            " yes ",
            "ON",
            "bridge",
            "scipy",
            "external",
            "python_reference",
            "python-bridge",
            "legacy-python",
            "legacy",
            "compatibility",
        ] {
            assert!(
                stochastic_lp_reference_force_python_value(value),
                "{value:?} should enable the stochastic LP compatibility bridge"
            );
        }

        for value in [
            "", "0", "false", "off", "python", "py", "auto", "rust", "native",
        ] {
            assert!(
                !stochastic_lp_reference_force_python_value(value),
                "{value:?} should keep Rust stochastic LP fallback active"
            );
        }
    }

    #[test]
    fn rust_reference_solves_sample_stochastic_lp() {
        let problem = build_production_slp(vec![1.0, 1.0], vec![3.0, 2.0], None);
        let scenarios = build_production_scenarios(
            UniformDemandSpec {
                ranges: vec![(5.0, 15.0), (10.0, 20.0)],
                seed: 42,
            },
            25,
        );

        let solution = solve_stochastic_lp_with_external_reference(
            &problem,
            &scenarios,
            &ExternalStochasticLpReferenceOptions {
                solver: ExternalStochasticLpReferenceSolver::RustMonolithic,
            },
        );

        assert_eq!(
            solution.status,
            ExternalStochasticLpReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:monolithic-slp");
        assert_eq!(solution.x.len(), problem.c_first.len());
        assert_eq!(solution.y_by_scenario.len(), scenarios.len());
        assert_eq!(solution.scenario_values.len(), scenarios.len());
        assert!(solution.objective.is_some());
        assert!(solution.expected_q.is_some());
    }

    #[test]
    fn fallback_alias_uses_rust_reference() {
        let problem = build_production_slp(vec![1.0], vec![3.0], None);
        let scenarios = build_production_scenarios(
            UniformDemandSpec {
                ranges: vec![(5.0, 15.0)],
                seed: 7,
            },
            5,
        );

        let solution = solve_stochastic_lp_with_external_reference(
            &problem,
            &scenarios,
            &ExternalStochasticLpReferenceOptions {
                solver: ExternalStochasticLpReferenceSolver::Fallback,
            },
        );

        assert_eq!(
            solution.status,
            ExternalStochasticLpReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:monolithic-slp");
        assert_eq!(solution.x.len(), 1);
        assert_eq!(solution.y_by_scenario.len(), scenarios.len());
    }

    #[test]
    fn scipy_alias_defaults_to_rust_reference_without_python() {
        let _lock = STOCHASTIC_LP_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guards = stochastic_lp_force_python_off_guards();
        let _python_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-stochastic-lp");
        let problem = build_production_slp(vec![1.0], vec![3.0], None);
        let scenarios = build_production_scenarios(
            UniformDemandSpec {
                ranges: vec![(5.0, 15.0)],
                seed: 11,
            },
            5,
        );

        let solution = solve_stochastic_lp_with_external_reference(
            &problem,
            &scenarios,
            &ExternalStochasticLpReferenceOptions {
                solver: ExternalStochasticLpReferenceSolver::Scipy,
            },
        );

        assert_eq!(
            solution.status,
            ExternalStochasticLpReferenceStatus::Optimal
        );
        assert_eq!(
            solution.solver,
            "rust:registered-stochastic-lp-fallback-for-scipy"
        );
        assert_eq!(solution.x.len(), 1);
        assert_eq!(solution.y_by_scenario.len(), scenarios.len());
        assert!(solution.objective.is_some());
        assert!(solution
            .message
            .contains("requested solver 'scipy' was validated with Rust fallback"));
    }

    #[test]
    fn stochastic_lp_force_python_keeps_scipy_bridge_available() {
        let _lock = STOCHASTIC_LP_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("STOCHASTIC_LP_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-forced-stochastic-lp",
        );
        let problem = build_production_slp(vec![1.0], vec![3.0], None);
        let scenarios = build_production_scenarios(
            UniformDemandSpec {
                ranges: vec![(5.0, 15.0)],
                seed: 13,
            },
            5,
        );

        let solution = solve_stochastic_lp_with_external_reference(
            &problem,
            &scenarios,
            &ExternalStochasticLpReferenceOptions {
                solver: ExternalStochasticLpReferenceSolver::Scipy,
            },
        );

        assert_eq!(
            solution.status,
            ExternalStochasticLpReferenceStatus::Unavailable
        );
        assert!(solution.message.contains("SciPy stochastic LP adapter"));
    }

    #[test]
    fn scipy_adapter_reports_startup_without_repo_script() {
        let _lock = STOCHASTIC_LP_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("STOCHASTIC_LP_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-stochastic-lp-scipy",
        );
        let problem = build_production_slp(vec![1.0], vec![3.0], None);
        let scenarios = build_production_scenarios(
            UniformDemandSpec {
                ranges: vec![(5.0, 15.0)],
                seed: 15,
            },
            5,
        );

        let solution = solve_stochastic_lp_with_external_reference(
            &problem,
            &scenarios,
            &ExternalStochasticLpReferenceOptions {
                solver: ExternalStochasticLpReferenceSolver::Scipy,
            },
        );

        assert_eq!(
            solution.status,
            ExternalStochasticLpReferenceStatus::Unavailable
        );
        assert!(solution.message.contains("SciPy stochastic LP adapter"));
        assert!(!solution.message.contains("stochastic_lp_reference.py"));
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_production_slp(vec![1.0], vec![3.0], None);
        let scenarios = build_production_scenarios(
            UniformDemandSpec {
                ranges: vec![(5.0, 15.0)],
                seed: 9,
            },
            5,
        );

        let solution = solve_stochastic_lp_with_external_reference(
            &problem,
            &scenarios,
            &ExternalStochasticLpReferenceOptions::default(),
        );

        assert_eq!(
            solution.status,
            ExternalStochasticLpReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:monolithic-slp");
        assert_eq!(solution.x.len(), 1);
        assert_eq!(solution.y_by_scenario.len(), scenarios.len());
    }

    #[test]
    fn stochastic_lp_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_stochastic_lp_adapter_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
