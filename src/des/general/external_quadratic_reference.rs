//! Rust-facing bridge for external/reference quadratic solvers.
//!
//! The checked-in Python bridge (`scripts/qp_reference.py`) prefers installed
//! open-source solvers such as HiGHS/highspy or SciPy when available and falls
//! back to dependency-free exact/pattern-search routines for small models. This
//! module owns the library boundary: typed model serialization, subprocess
//! execution, status mapping, and elapsed-time accounting.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::qp::{
    MixedIntegerQuadraticProgram, QuadraticProgram, QuadraticallyConstrainedProgram,
    SecondOrderConeProgram,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalQuadraticReferenceSolver {
    Auto,
    Highs,
    Scipy,
    Fallback,
}

impl ExternalQuadraticReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalQuadraticReferenceSolver::Auto => "auto",
            ExternalQuadraticReferenceSolver::Highs => "highs",
            ExternalQuadraticReferenceSolver::Scipy => "scipy",
            ExternalQuadraticReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalQuadraticReferenceOptions {
    pub solver: ExternalQuadraticReferenceSolver,
    pub max_enumerations: Option<usize>,
}

impl Default for ExternalQuadraticReferenceOptions {
    fn default() -> Self {
        ExternalQuadraticReferenceOptions {
            solver: ExternalQuadraticReferenceSolver::Auto,
            max_enumerations: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalQuadraticReferenceStatus {
    Optimal,
    Infeasible,
    Unbounded,
    NumericalError,
    Unavailable,
}

impl ExternalQuadraticReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalQuadraticReferenceStatus::Optimal => "optimal",
            ExternalQuadraticReferenceStatus::Infeasible => "infeasible",
            ExternalQuadraticReferenceStatus::Unbounded => "unbounded",
            ExternalQuadraticReferenceStatus::NumericalError => "numerical-error",
            ExternalQuadraticReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalQuadraticReferenceSolution {
    pub status: ExternalQuadraticReferenceStatus,
    pub solver: String,
    pub x: Vec<f64>,
    pub objective: Option<f64>,
    pub dual_ub: Option<Vec<f64>>,
    pub dual_eq: Option<Vec<f64>>,
    pub dual_lower_bounds: Option<Vec<f64>>,
    pub dual_upper_bounds: Option<Vec<f64>>,
    pub reduced_gradient: Option<Vec<f64>>,
    pub iterations: Option<u64>,
    pub enumerated: Option<u64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct QuadraticReferencePayload {
    status: String,
    solver: Option<String>,
    x: Option<Vec<f64>>,
    objective: Option<f64>,
    #[serde(rename = "dualUB")]
    dual_ub: Option<Vec<f64>>,
    #[serde(rename = "dualEQ")]
    dual_eq: Option<Vec<f64>>,
    #[serde(rename = "dualLowerBounds")]
    dual_lower_bounds: Option<Vec<f64>>,
    #[serde(rename = "dualUpperBounds")]
    dual_upper_bounds: Option<Vec<f64>>,
    #[serde(rename = "reducedGradient")]
    reduced_gradient: Option<Vec<f64>>,
    iterations: Option<u64>,
    enumerated: Option<u64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalQuadraticReferenceStatus {
    match status {
        "optimal" => ExternalQuadraticReferenceStatus::Optimal,
        "infeasible" => ExternalQuadraticReferenceStatus::Infeasible,
        "unbounded" => ExternalQuadraticReferenceStatus::Unbounded,
        "unavailable" => ExternalQuadraticReferenceStatus::Unavailable,
        _ => ExternalQuadraticReferenceStatus::NumericalError,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalQuadraticReferenceSolution {
    ExternalQuadraticReferenceSolution {
        status: ExternalQuadraticReferenceStatus::Unavailable,
        solver: "external-quadratic-reference".to_string(),
        x: Vec::new(),
        objective: None,
        dual_ub: None,
        dual_eq: None,
        dual_lower_bounds: None,
        dual_upper_bounds: None,
        reduced_gradient: None,
        iterations: None,
        enumerated: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalQuadraticReferenceSolution {
    ExternalQuadraticReferenceSolution {
        status: ExternalQuadraticReferenceStatus::NumericalError,
        solver: "external-quadratic-reference".to_string(),
        x: Vec::new(),
        objective: None,
        dual_ub: None,
        dual_eq: None,
        dual_lower_bounds: None,
        dual_upper_bounds: None,
        reduced_gradient: None,
        iterations: None,
        enumerated: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("qp_reference.py")
}

fn run_quadratic_reference_json(
    payload: Value,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    let started = Instant::now();
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command
        .arg(reference_script())
        .arg("--solver")
        .arg(opts.solver.as_arg());
    if let Some(max_enumerations) = opts.max_enumerations {
        command
            .arg("--max-enumerations")
            .arg(max_enumerations.to_string());
    }
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return unavailable(
                format!("failed to start qp_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write qp_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for qp_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<QuadraticReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalQuadraticReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-quadratic-reference".to_string()),
            x: parsed.x.unwrap_or_default(),
            objective: parsed.objective,
            dual_ub: parsed.dual_ub,
            dual_eq: parsed.dual_eq,
            dual_lower_bounds: parsed.dual_lower_bounds,
            dual_upper_bounds: parsed.dual_upper_bounds,
            reduced_gradient: parsed.reduced_gradient,
            iterations: parsed.iterations,
            enumerated: parsed.enumerated,
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
                "failed to parse qp_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

fn qp_json(problem: &QuadraticProgram) -> Value {
    json!({
        "Q": &problem.q,
        "c": &problem.c,
        "A_ub": &problem.a_ub,
        "b_ub": &problem.b_ub,
        "A_eq": &problem.a_eq,
        "b_eq": &problem.b_eq,
        "lb": &problem.lb,
        "ub": &problem.ub,
        "var_names": &problem.var_names,
    })
}

pub fn solve_qp_with_external_reference(
    problem: &QuadraticProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    run_quadratic_reference_json(qp_json(problem), opts)
}

pub fn solve_miqp_with_external_reference(
    problem: &MixedIntegerQuadraticProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    let mut payload = qp_json(&problem.qp);
    if let Some(map) = payload.as_object_mut() {
        map.insert("integer_vars".to_string(), json!(&problem.integer_vars));
    }
    run_quadratic_reference_json(payload, opts)
}

pub fn solve_socp_with_external_reference(
    problem: &SecondOrderConeProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    run_quadratic_reference_json(
        json!({
            "c": &problem.c,
            "A_ub": &problem.a_ub,
            "b_ub": &problem.b_ub,
            "A_eq": &problem.a_eq,
            "b_eq": &problem.b_eq,
            "lb": &problem.lb,
            "ub": &problem.ub,
            "cones": problem.cones.iter().map(|cone| json!({
                "A": &cone.a,
                "b": &cone.b,
                "c": &cone.c,
                "d": cone.d,
                "name": &cone.name,
            })).collect::<Vec<_>>(),
            "var_names": &problem.var_names,
        }),
        opts,
    )
}

pub fn solve_qcp_with_external_reference(
    problem: &QuadraticallyConstrainedProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    run_quadratic_reference_json(
        json!({
            "Q": &problem.q,
            "c": &problem.c,
            "A_ub": &problem.a_ub,
            "b_ub": &problem.b_ub,
            "A_eq": &problem.a_eq,
            "b_eq": &problem.b_eq,
            "lb": &problem.lb,
            "ub": &problem.ub,
            "quadratic_constraints": problem.quadratic_constraints.iter().map(|constraint| json!({
                "Q": &constraint.q,
                "c": &constraint.c,
                "rhs": constraint.rhs,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "var_names": &problem.var_names,
        }),
        opts,
    )
}
