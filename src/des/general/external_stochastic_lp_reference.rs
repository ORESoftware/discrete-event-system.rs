//! Rust-facing bridge for external/reference stochastic LP solvers.
//!
//! The native Rust reference builds the extensive-form sample-average LP and
//! solves it through the Rust LP stack without Python startup. The checked-in
//! Python bridge (`scripts/stochastic_lp_reference.py`) remains available for
//! SciPy/HiGHS when a true external open-source comparison is requested.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

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

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("stochastic_lp_reference.py")
}

fn run_stochastic_lp_reference_json(
    payload: Value,
    opts: &ExternalStochasticLpReferenceOptions,
) -> ExternalStochasticLpReferenceSolution {
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
                format!("failed to start stochastic_lp_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write stochastic_lp_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for stochastic_lp_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
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
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            }),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse stochastic_lp_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
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
    if matches!(
        opts.solver,
        ExternalStochasticLpReferenceSolver::Auto
            | ExternalStochasticLpReferenceSolver::RustMonolithic
            | ExternalStochasticLpReferenceSolver::Fallback
    ) {
        return solve_stochastic_lp_with_rust_reference(problem, scenarios);
    }

    run_stochastic_lp_reference_json(
        json!({
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
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::stochastic_lp::{
        build_production_scenarios, build_production_slp, UniformDemandSpec,
    };

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
}
