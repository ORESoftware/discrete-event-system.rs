//! Rust-facing bridge for external/reference nonlinear optimizers.
//!
//! The checked-in Python bridge (`scripts/nonlinear_reference.py`) prefers
//! installed open-source optimizers such as SciPy and can use NLopt when
//! available, then falls back to deterministic small-model references. This
//! module owns the typed library boundary for smooth unconstrained problems,
//! nonlinear least squares, and derivative-free benchmark minimization.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::advanced_optimization_models::{ParetoPortfolioPoint, PortfolioAsset};
use crate::des::general::nonlinear_optimization_models::CurveFitPoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearReferenceSolver {
    Auto,
    Scipy,
    Nlopt,
    Fallback,
}

impl ExternalNonlinearReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalNonlinearReferenceSolver::Auto => "auto",
            ExternalNonlinearReferenceSolver::Scipy => "scipy",
            ExternalNonlinearReferenceSolver::Nlopt => "nlopt",
            ExternalNonlinearReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearBenchmarkObjective {
    Sphere,
    Rastrigin,
    Rosenbrock,
}

impl ExternalNonlinearBenchmarkObjective {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalNonlinearBenchmarkObjective::Sphere => "sphere",
            ExternalNonlinearBenchmarkObjective::Rastrigin => "rastrigin",
            ExternalNonlinearBenchmarkObjective::Rosenbrock => "rosenbrock",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearReferenceOptions {
    pub solver: ExternalNonlinearReferenceSolver,
    pub max_iterations: Option<usize>,
}

impl Default for ExternalNonlinearReferenceOptions {
    fn default() -> Self {
        ExternalNonlinearReferenceOptions {
            solver: ExternalNonlinearReferenceSolver::Auto,
            max_iterations: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearReferenceStatus {
    Optimal,
    Feasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalNonlinearReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalNonlinearReferenceStatus::Optimal => "optimal",
            ExternalNonlinearReferenceStatus::Feasible => "feasible",
            ExternalNonlinearReferenceStatus::Unsupported => "unsupported",
            ExternalNonlinearReferenceStatus::NumericalError => "numerical-error",
            ExternalNonlinearReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearReferenceSolution {
    pub status: ExternalNonlinearReferenceStatus,
    pub solver: String,
    pub x: Vec<f64>,
    pub objective: Option<f64>,
    pub gradient_norm: Option<f64>,
    pub residual_norm: Option<f64>,
    pub iterations: Option<u64>,
    pub evaluations: Option<u64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct ExternalParetoPortfolioReferenceSolution {
    pub status: ExternalNonlinearReferenceStatus,
    pub solver: String,
    pub pareto_front: Vec<ParetoPortfolioPoint>,
    pub candidate_count: Option<u64>,
    pub hypervolume: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct NonlinearReferencePayload {
    status: String,
    solver: Option<String>,
    x: Option<Vec<f64>>,
    objective: Option<f64>,
    #[serde(rename = "gradientNorm")]
    gradient_norm: Option<f64>,
    #[serde(rename = "residualNorm")]
    residual_norm: Option<f64>,
    iterations: Option<u64>,
    evaluations: Option<u64>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ParetoPortfolioReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "paretoFront")]
    pareto_front: Option<Vec<ParetoPortfolioPointPayload>>,
    #[serde(rename = "candidateCount")]
    candidate_count: Option<u64>,
    hypervolume: Option<f64>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ParetoPortfolioPointPayload {
    weights: Vec<f64>,
    #[serde(rename = "expectedReturn")]
    expected_return: f64,
    risk: f64,
}

fn status_from_str(status: &str) -> ExternalNonlinearReferenceStatus {
    match status {
        "optimal" => ExternalNonlinearReferenceStatus::Optimal,
        "feasible" => ExternalNonlinearReferenceStatus::Feasible,
        "unsupported" => ExternalNonlinearReferenceStatus::Unsupported,
        "unavailable" => ExternalNonlinearReferenceStatus::Unavailable,
        _ => ExternalNonlinearReferenceStatus::NumericalError,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalNonlinearReferenceSolution {
    ExternalNonlinearReferenceSolution {
        status: ExternalNonlinearReferenceStatus::Unavailable,
        solver: "external-nonlinear-reference".to_string(),
        x: Vec::new(),
        objective: None,
        gradient_norm: None,
        residual_norm: None,
        iterations: None,
        evaluations: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalNonlinearReferenceSolution {
    ExternalNonlinearReferenceSolution {
        status: ExternalNonlinearReferenceStatus::NumericalError,
        solver: "external-nonlinear-reference".to_string(),
        x: Vec::new(),
        objective: None,
        gradient_norm: None,
        residual_norm: None,
        iterations: None,
        evaluations: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn pareto_unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalParetoPortfolioReferenceSolution {
    ExternalParetoPortfolioReferenceSolution {
        status: ExternalNonlinearReferenceStatus::Unavailable,
        solver: "external-nonlinear-reference".to_string(),
        pareto_front: Vec::new(),
        candidate_count: None,
        hypervolume: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn pareto_numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalParetoPortfolioReferenceSolution {
    ExternalParetoPortfolioReferenceSolution {
        status: ExternalNonlinearReferenceStatus::NumericalError,
        solver: "external-nonlinear-reference".to_string(),
        pareto_front: Vec::new(),
        candidate_count: None,
        hypervolume: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("nonlinear_reference.py")
}

fn run_reference_process(
    payload: Value,
    opts: &ExternalNonlinearReferenceOptions,
) -> (Instant, Result<std::process::Output, String>) {
    let started = Instant::now();
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command
        .arg(reference_script())
        .arg("--solver")
        .arg(opts.solver.as_arg());
    if let Some(max_iterations) = opts.max_iterations {
        command
            .arg("--max-iterations")
            .arg(max_iterations.to_string());
    }
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return (
                started,
                Err(format!(
                    "failed to start nonlinear_reference.py with {python}: {err}"
                )),
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return (
                started,
                Err(format!(
                    "failed to write nonlinear_reference.py stdin: {err}"
                )),
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return (
                started,
                Err(format!("failed to wait for nonlinear_reference.py: {err}")),
            )
        }
    };
    (started, Ok(output))
}

fn run_nonlinear_reference_json(
    payload: Value,
    opts: &ExternalNonlinearReferenceOptions,
) -> ExternalNonlinearReferenceSolution {
    let (started, output) = run_reference_process(payload, opts);
    let output = match output {
        Ok(output) => output,
        Err(message) => return unavailable(message, started.elapsed().as_secs_f64() * 1000.0),
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<NonlinearReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalNonlinearReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-nonlinear-reference".to_string()),
            x: parsed.x.unwrap_or_default(),
            objective: parsed.objective,
            gradient_norm: parsed.gradient_norm,
            residual_norm: parsed.residual_norm,
            iterations: parsed.iterations,
            evaluations: parsed.evaluations,
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
                "failed to parse nonlinear_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

fn run_pareto_portfolio_reference_json(
    payload: Value,
    opts: &ExternalNonlinearReferenceOptions,
) -> ExternalParetoPortfolioReferenceSolution {
    let (started, output) = run_reference_process(payload, opts);
    let output = match output {
        Ok(output) => output,
        Err(message) => {
            return pareto_unavailable(message, started.elapsed().as_secs_f64() * 1000.0)
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<ParetoPortfolioReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalParetoPortfolioReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-nonlinear-reference".to_string()),
            pareto_front: parsed
                .pareto_front
                .unwrap_or_default()
                .into_iter()
                .map(|point| ParetoPortfolioPoint {
                    weights: point.weights,
                    expected_return: point.expected_return,
                    risk: point.risk,
                })
                .collect(),
            candidate_count: parsed.candidate_count,
            hypervolume: parsed.hypervolume,
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            }),
            elapsed_ms,
        },
        Err(err) => pareto_numerical_error(
            format!(
                "failed to parse nonlinear_reference.py Pareto output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_rosenbrock_with_external_reference(
    x0: &[f64],
    opts: &ExternalNonlinearReferenceOptions,
) -> ExternalNonlinearReferenceSolution {
    run_nonlinear_reference_json(
        json!({
            "kind": "rosenbrock",
            "x0": x0,
        }),
        opts,
    )
}

pub fn solve_pareto_portfolio_with_external_reference(
    assets: &[PortfolioAsset],
    samples: usize,
    seed: u32,
    opts: &ExternalNonlinearReferenceOptions,
) -> ExternalParetoPortfolioReferenceSolution {
    run_pareto_portfolio_reference_json(
        json!({
            "kind": "pareto_portfolio",
            "assets": assets.iter().map(|asset| json!({
                "name": &asset.name,
                "expectedReturn": asset.expected_return,
                "risk": asset.risk,
            })).collect::<Vec<_>>(),
            "samples": samples,
            "seed": seed,
        }),
        opts,
    )
}

pub fn solve_exponential_fit_with_external_reference(
    points: &[CurveFitPoint],
    initial: &[f64],
    opts: &ExternalNonlinearReferenceOptions,
) -> ExternalNonlinearReferenceSolution {
    run_nonlinear_reference_json(
        json!({
            "kind": "least_squares",
            "points": points.iter().map(|point| json!({
                "x": point.x,
                "y": point.y,
            })).collect::<Vec<_>>(),
            "initial": initial,
        }),
        opts,
    )
}

pub fn solve_global_benchmark_with_external_reference(
    objective: ExternalNonlinearBenchmarkObjective,
    dimension: usize,
    lower: f64,
    upper: f64,
    opts: &ExternalNonlinearReferenceOptions,
) -> ExternalNonlinearReferenceSolution {
    run_nonlinear_reference_json(
        json!({
            "kind": "global_benchmark",
            "objective": objective.as_arg(),
            "dimension": dimension,
            "lower": lower,
            "upper": upper,
        }),
        opts,
    )
}
