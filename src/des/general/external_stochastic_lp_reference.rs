//! Rust-facing bridge for external/reference stochastic LP solvers.
//!
//! The checked-in Python bridge (`scripts/stochastic_lp_reference.py`) builds
//! the extensive-form sample-average LP and solves it with SciPy/HiGHS when
//! available. This gives native monolithic SAA and Benders/L-shaped solves a
//! same-input open-source reference without vendoring solver executables.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::stochastic_lp::{SLPProblem, Scenario};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalStochasticLpReferenceSolver {
    Auto,
    Scipy,
    Fallback,
}

impl ExternalStochasticLpReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalStochasticLpReferenceSolver::Auto => "auto",
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
