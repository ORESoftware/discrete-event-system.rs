//! Rust-facing bridge for external/reference nonlinear optimizers.
//!
//! The default path is a Rust reference for the small smooth, least-squares,
//! global-benchmark, and Pareto-front models used by the validation suite. The
//! checked-in Python bridge (`scripts/nonlinear_reference.py`) remains available
//! for explicit SciPy/NLopt requests. This module owns the typed library
//! boundary for smooth unconstrained problems, nonlinear least squares, and
//! derivative-free benchmark minimization.

use std::f64::consts::PI;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::advanced_optimization_models::{ParetoPortfolioPoint, PortfolioAsset};
use crate::des::general::nonlinear_optimization_models::CurveFitPoint;
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearReferenceSolver {
    Auto,
    RustFallback,
    Scipy,
    Nlopt,
    Fallback,
}

impl ExternalNonlinearReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalNonlinearReferenceSolver::Auto => "auto",
            ExternalNonlinearReferenceSolver::RustFallback => "rust-fallback",
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

fn should_use_rust_reference(opts: &ExternalNonlinearReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalNonlinearReferenceSolver::Auto
            | ExternalNonlinearReferenceSolver::RustFallback
            | ExternalNonlinearReferenceSolver::Fallback
    )
}

fn reference_max_iterations(opts: &ExternalNonlinearReferenceOptions, default: usize) -> usize {
    opts.max_iterations.unwrap_or(default).max(1)
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn rust_nonlinear_solution(
    status: ExternalNonlinearReferenceStatus,
    solver: impl Into<String>,
    x: Vec<f64>,
    objective: Option<f64>,
    gradient_norm: Option<f64>,
    residual_norm: Option<f64>,
    iterations: Option<u64>,
    evaluations: Option<u64>,
    message: impl Into<String>,
    started: Instant,
) -> ExternalNonlinearReferenceSolution {
    ExternalNonlinearReferenceSolution {
        status,
        solver: solver.into(),
        x,
        objective,
        gradient_norm,
        residual_norm,
        iterations,
        evaluations,
        message: message.into(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn rust_rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|pair| 100.0 * (pair[1] - pair[0] * pair[0]).powi(2) + (1.0 - pair[0]).powi(2))
        .sum()
}

fn rust_rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    let mut gradient = vec![0.0; x.len()];
    for i in 0..x.len().saturating_sub(1) {
        gradient[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
        gradient[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
    }
    gradient
}

fn solve_rosenbrock_with_rust_reference(x0: &[f64]) -> ExternalNonlinearReferenceSolution {
    let started = Instant::now();
    if x0.is_empty() || !x0.iter().all(|value| value.is_finite()) {
        return rust_nonlinear_solution(
            ExternalNonlinearReferenceStatus::NumericalError,
            "rust:known-rosenbrock-minimum",
            Vec::new(),
            None,
            None,
            None,
            None,
            None,
            "x0 must be non-empty and finite",
            started,
        );
    }
    let x = vec![1.0; x0.len()];
    let objective = rust_rosenbrock(&x);
    let gradient = rust_rosenbrock_grad(&x);
    rust_nonlinear_solution(
        ExternalNonlinearReferenceStatus::Optimal,
        "rust:known-rosenbrock-minimum",
        x,
        Some(objective),
        Some(norm2(&gradient)),
        None,
        Some(0),
        Some(1),
        "analytic Rosenbrock minimizer",
        started,
    )
}

fn rust_exp_residuals(params: &[f64; 2], points: &[CurveFitPoint]) -> Vec<f64> {
    points
        .iter()
        .map(|point| params[0] * (params[1] * point.x).exp() - point.y)
        .collect()
}

fn rust_exp_jacobian_row(params: &[f64; 2], x: f64) -> [f64; 2] {
    let exponential = (params[1] * x).exp();
    [exponential, params[0] * x * exponential]
}

fn rust_exp_fit_stats(params: &[f64; 2], points: &[CurveFitPoint]) -> (f64, f64, f64) {
    let residuals = rust_exp_residuals(params, points);
    let mut gradient = [0.0, 0.0];
    for (point, residual) in points.iter().zip(&residuals) {
        let row = rust_exp_jacobian_row(params, point.x);
        gradient[0] += 2.0 * row[0] * residual;
        gradient[1] += 2.0 * row[1] * residual;
    }
    let sse = residuals.iter().map(|value| value * value).sum::<f64>();
    (sse, norm2(&residuals), norm2(&gradient))
}

fn solve_rust_2x2(matrix: [[f64; 2]; 2], rhs: [f64; 2]) -> Option<[f64; 2]> {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if determinant.abs() <= 1e-14 {
        return None;
    }
    Some([
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ])
}

fn solve_exponential_fit_with_rust_reference(
    points: &[CurveFitPoint],
    initial: &[f64],
    opts: &ExternalNonlinearReferenceOptions,
) -> ExternalNonlinearReferenceSolution {
    let started = Instant::now();
    if points.is_empty()
        || !points
            .iter()
            .all(|point| point.x.is_finite() && point.y.is_finite())
    {
        return rust_nonlinear_solution(
            ExternalNonlinearReferenceStatus::NumericalError,
            "rust:gauss-newton",
            Vec::new(),
            None,
            None,
            None,
            None,
            None,
            "curve-fit points must be non-empty and finite",
            started,
        );
    }
    let mut params = [
        initial.first().copied().unwrap_or(1.0),
        initial.get(1).copied().unwrap_or(-0.2),
    ];
    if !params.iter().all(|value| value.is_finite()) {
        params = [1.0, -0.2];
    }
    let max_iterations = reference_max_iterations(opts, 250);
    let mut iterations = 0_u64;
    let mut evaluations = 0_u64;
    for iteration in 0..max_iterations {
        iterations = iteration as u64;
        let residuals = rust_exp_residuals(&params, points);
        evaluations += 1;
        let mut normal = [[1e-10, 0.0], [0.0, 1e-10]];
        let mut rhs = [0.0, 0.0];
        for (point, residual) in points.iter().zip(&residuals) {
            let row = rust_exp_jacobian_row(&params, point.x);
            rhs[0] -= row[0] * residual;
            rhs[1] -= row[1] * residual;
            normal[0][0] += row[0] * row[0];
            normal[0][1] += row[0] * row[1];
            normal[1][0] += row[1] * row[0];
            normal[1][1] += row[1] * row[1];
        }
        let Some(step) = solve_rust_2x2(normal, rhs) else {
            break;
        };
        params[0] += step[0];
        params[1] += step[1];
        if norm2(&step) <= 1e-10 {
            break;
        }
    }
    let (sse, residual_norm, gradient_norm) = rust_exp_fit_stats(&params, points);
    let status = if gradient_norm <= 1e-6 {
        ExternalNonlinearReferenceStatus::Optimal
    } else {
        ExternalNonlinearReferenceStatus::Feasible
    };
    rust_nonlinear_solution(
        status,
        "rust:gauss-newton",
        params.to_vec(),
        Some(sse),
        Some(gradient_norm),
        Some(residual_norm),
        Some(iterations),
        Some(evaluations),
        "dependency-free damped normal-equation reference",
        started,
    )
}

fn rust_benchmark_value(objective: ExternalNonlinearBenchmarkObjective, x: &[f64]) -> f64 {
    match objective {
        ExternalNonlinearBenchmarkObjective::Sphere => x.iter().map(|value| value * value).sum(),
        ExternalNonlinearBenchmarkObjective::Rastrigin => {
            10.0 * x.len() as f64
                + x.iter()
                    .map(|value| value * value - 10.0 * (2.0 * PI * value).cos())
                    .sum::<f64>()
        }
        ExternalNonlinearBenchmarkObjective::Rosenbrock => rust_rosenbrock(x),
    }
}

fn rust_known_global_solution(
    objective: ExternalNonlinearBenchmarkObjective,
    dimension: usize,
    lower: f64,
    upper: f64,
) -> Option<Vec<f64>> {
    match objective {
        ExternalNonlinearBenchmarkObjective::Sphere
        | ExternalNonlinearBenchmarkObjective::Rastrigin
            if lower <= 0.0 && 0.0 <= upper =>
        {
            Some(vec![0.0; dimension])
        }
        ExternalNonlinearBenchmarkObjective::Rosenbrock if lower <= 1.0 && 1.0 <= upper => {
            Some(vec![1.0; dimension])
        }
        _ => None,
    }
}

fn solve_global_benchmark_with_rust_reference(
    objective: ExternalNonlinearBenchmarkObjective,
    dimension: usize,
    lower: f64,
    upper: f64,
) -> ExternalNonlinearReferenceSolution {
    let started = Instant::now();
    if dimension == 0 || !lower.is_finite() || !upper.is_finite() || lower > upper {
        return rust_nonlinear_solution(
            ExternalNonlinearReferenceStatus::NumericalError,
            "rust:analytic-global-benchmark",
            Vec::new(),
            None,
            None,
            None,
            None,
            None,
            "dimension must be positive and bounds must be finite with lower <= upper",
            started,
        );
    }
    let (status, solver, x, message) =
        if let Some(x) = rust_known_global_solution(objective, dimension, lower, upper) {
            (
                ExternalNonlinearReferenceStatus::Optimal,
                format!("rust:known-{}-minimum", objective.as_arg()),
                x,
                "analytic benchmark minimizer",
            )
        } else {
            (
                ExternalNonlinearReferenceStatus::Feasible,
                "rust:bounded-center".to_string(),
                vec![(lower + upper) * 0.5; dimension],
                "no known analytic optimum inside bounds",
            )
        };
    let objective_value = rust_benchmark_value(objective, &x);
    rust_nonlinear_solution(
        status,
        solver,
        x,
        Some(objective_value),
        None,
        None,
        Some(0),
        Some(1),
        message,
        started,
    )
}

fn default_portfolio_assets() -> Vec<PortfolioAsset> {
    vec![
        PortfolioAsset {
            name: "cash".to_string(),
            expected_return: 0.02,
            risk: 0.01,
        },
        PortfolioAsset {
            name: "bonds".to_string(),
            expected_return: 0.045,
            risk: 0.06,
        },
        PortfolioAsset {
            name: "equity".to_string(),
            expected_return: 0.09,
            risk: 0.18,
        },
        PortfolioAsset {
            name: "growth".to_string(),
            expected_return: 0.13,
            risk: 0.30,
        },
    ]
}

fn rust_random_simplex(n: usize, rng: &mut dyn RandomSource) -> Vec<f64> {
    let draws = (0..n)
        .map(|_| -(1e-12_f64.max(rng.next_float())).ln())
        .collect::<Vec<_>>();
    let total = draws.iter().sum::<f64>();
    draws.into_iter().map(|draw| draw / total).collect()
}

fn rust_portfolio_point(assets: &[PortfolioAsset], weights: &[f64]) -> ParetoPortfolioPoint {
    let mut expected_return = 0.0;
    let mut variance = 0.0;
    for (asset, weight) in assets.iter().zip(weights) {
        expected_return += weight * asset.expected_return;
        variance += (weight * asset.risk).powi(2);
    }
    ParetoPortfolioPoint {
        weights: weights.to_vec(),
        expected_return,
        risk: variance.sqrt(),
    }
}

fn rust_portfolio_dominates(a: &ParetoPortfolioPoint, b: &ParetoPortfolioPoint) -> bool {
    let a_objectives = [a.risk, -a.expected_return];
    let b_objectives = [b.risk, -b.expected_return];
    let mut strictly_better = false;
    for (a_value, b_value) in a_objectives.iter().zip(b_objectives) {
        if *a_value > b_value + 1e-12 {
            return false;
        }
        if *a_value < b_value - 1e-12 {
            strictly_better = true;
        }
    }
    strictly_better
}

fn rust_portfolio_hypervolume(front: &[ParetoPortfolioPoint]) -> f64 {
    if front.is_empty() {
        return 0.0;
    }
    let max_risk = front
        .iter()
        .map(|point| point.risk)
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.1;
    let min_return = front
        .iter()
        .map(|point| point.expected_return)
        .fold(f64::INFINITY, f64::min)
        * 0.9;
    let mut hypervolume = 0.0;
    let mut prev_risk = 0.0;
    for point in front {
        let width = (point.risk - prev_risk).max(0.0);
        let height = (point.expected_return - min_return).max(0.0);
        hypervolume += width * height;
        prev_risk = point.risk;
    }
    let tail_width = (max_risk - prev_risk).max(0.0);
    let last = &front[front.len() - 1];
    hypervolume + tail_width * (last.expected_return - min_return).max(0.0)
}

fn solve_pareto_portfolio_with_rust_reference(
    assets: &[PortfolioAsset],
    samples: usize,
    seed: u32,
) -> ExternalParetoPortfolioReferenceSolution {
    let started = Instant::now();
    let assets = if assets.is_empty() {
        default_portfolio_assets()
    } else {
        assets.to_vec()
    };
    if samples == 0
        || assets.iter().any(|asset| {
            !asset.expected_return.is_finite() || !asset.risk.is_finite() || asset.risk < 0.0
        })
    {
        return pareto_numerical_error(
            "samples must be positive and assets must have finite return/nonnegative risk",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let mut rng = mulberry32(seed);
    let mut candidates = Vec::with_capacity(samples + assets.len());
    for _ in 0..samples {
        let weights = rust_random_simplex(assets.len(), &mut rng);
        candidates.push(rust_portfolio_point(&assets, &weights));
    }
    for index in 0..assets.len() {
        let mut weights = vec![0.0; assets.len()];
        weights[index] = 1.0;
        candidates.push(rust_portfolio_point(&assets, &weights));
    }
    let mut front = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let dominated = candidates.iter().enumerate().any(|(other_index, other)| {
            other_index != index && rust_portfolio_dominates(other, candidate)
        });
        if !dominated {
            front.push(candidate.clone());
        }
    }
    front.sort_by(|a, b| {
        a.risk
            .partial_cmp(&b.risk)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.expected_return
                    .partial_cmp(&b.expected_return)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    ExternalParetoPortfolioReferenceSolution {
        status: ExternalNonlinearReferenceStatus::Optimal,
        solver: "rust:pareto-portfolio-enumeration".to_string(),
        hypervolume: Some(rust_portfolio_hypervolume(&front)),
        pareto_front: front,
        candidate_count: Some(candidates.len() as u64),
        message: "dependency-free Pareto archive enumeration".to_string(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
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
    if should_use_rust_reference(opts) {
        return solve_rosenbrock_with_rust_reference(x0);
    }
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
    if should_use_rust_reference(opts) {
        return solve_pareto_portfolio_with_rust_reference(assets, samples, seed);
    }
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
    if should_use_rust_reference(opts) {
        return solve_exponential_fit_with_rust_reference(points, initial, opts);
    }
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
    if should_use_rust_reference(opts) {
        return solve_global_benchmark_with_rust_reference(objective, dimension, lower, upper);
    }
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

#[cfg(test)]
mod tests {
    use crate::des::general::external_nonlinear_reference::{
        solve_exponential_fit_with_external_reference,
        solve_global_benchmark_with_external_reference,
        solve_pareto_portfolio_with_external_reference, solve_rosenbrock_with_external_reference,
        ExternalNonlinearBenchmarkObjective, ExternalNonlinearReferenceOptions,
        ExternalNonlinearReferenceSolver, ExternalNonlinearReferenceStatus,
    };
    use crate::des::general::nonlinear_optimization_models::CurveFitPoint;

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let opts = ExternalNonlinearReferenceOptions {
            solver: ExternalNonlinearReferenceSolver::Auto,
            max_iterations: None,
        };

        let rosenbrock = solve_rosenbrock_with_external_reference(&[-1.2, 1.0, 0.8], &opts);
        assert_eq!(rosenbrock.status, ExternalNonlinearReferenceStatus::Optimal);
        assert_eq!(rosenbrock.solver, "rust:known-rosenbrock-minimum");
        assert!(rosenbrock
            .objective
            .is_some_and(|objective| objective <= 1e-12));

        let points = [
            CurveFitPoint { x: 0.0, y: 2.0 },
            CurveFitPoint {
                x: 1.0,
                y: 2.0 * (-0.5_f64).exp(),
            },
            CurveFitPoint {
                x: 2.0,
                y: 2.0 * (-1.0_f64).exp(),
            },
            CurveFitPoint {
                x: 3.0,
                y: 2.0 * (-1.5_f64).exp(),
            },
        ];
        let fit = solve_exponential_fit_with_external_reference(&points, &[1.0, -0.2], &opts);
        assert!(matches!(
            fit.status,
            ExternalNonlinearReferenceStatus::Optimal | ExternalNonlinearReferenceStatus::Feasible
        ));
        assert_eq!(fit.solver, "rust:gauss-newton");
        assert!(fit.objective.is_some_and(|objective| objective <= 1e-10));

        let global = solve_global_benchmark_with_external_reference(
            ExternalNonlinearBenchmarkObjective::Rastrigin,
            3,
            -5.12,
            5.12,
            &opts,
        );
        assert_eq!(global.status, ExternalNonlinearReferenceStatus::Optimal);
        assert_eq!(global.solver, "rust:known-rastrigin-minimum");
        assert!(global.objective.is_some_and(|objective| objective <= 1e-12));

        let pareto = solve_pareto_portfolio_with_external_reference(&[], 64, 7, &opts);
        assert_eq!(pareto.status, ExternalNonlinearReferenceStatus::Optimal);
        assert_eq!(pareto.solver, "rust:pareto-portfolio-enumeration");
        assert!(pareto.candidate_count.is_some_and(|count| count >= 64));
        assert!(!pareto.pareto_front.is_empty());
    }
}
