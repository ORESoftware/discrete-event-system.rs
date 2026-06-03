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
    MixedIntegerQuadraticProgram, MixedIntegerQuadraticallyConstrainedProgram,
    MixedIntegerSecondOrderConeProgram, QuadraticProgram, QuadraticallyConstrainedProgram,
    SecondOrderConeProgram,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalQuadraticReferenceSolver {
    Auto,
    Highs,
    Scipy,
    Osqp,
    Cvxpy,
    Scs,
    Clarabel,
    Ecos,
    Mosek,
    Copt,
    Qpoases,
    Proxqp,
    Cosmo,
    Sdpa,
    Csdp,
    Fallback,
}

impl ExternalQuadraticReferenceSolver {
    pub fn all() -> &'static [ExternalQuadraticReferenceSolver] {
        &[
            ExternalQuadraticReferenceSolver::Auto,
            ExternalQuadraticReferenceSolver::Highs,
            ExternalQuadraticReferenceSolver::Scipy,
            ExternalQuadraticReferenceSolver::Osqp,
            ExternalQuadraticReferenceSolver::Cvxpy,
            ExternalQuadraticReferenceSolver::Scs,
            ExternalQuadraticReferenceSolver::Clarabel,
            ExternalQuadraticReferenceSolver::Ecos,
            ExternalQuadraticReferenceSolver::Mosek,
            ExternalQuadraticReferenceSolver::Copt,
            ExternalQuadraticReferenceSolver::Qpoases,
            ExternalQuadraticReferenceSolver::Proxqp,
            ExternalQuadraticReferenceSolver::Cosmo,
            ExternalQuadraticReferenceSolver::Sdpa,
            ExternalQuadraticReferenceSolver::Csdp,
            ExternalQuadraticReferenceSolver::Fallback,
        ]
    }

    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalQuadraticReferenceSolver::Auto => "auto",
            ExternalQuadraticReferenceSolver::Highs => "highs",
            ExternalQuadraticReferenceSolver::Scipy => "scipy",
            ExternalQuadraticReferenceSolver::Osqp => "osqp",
            ExternalQuadraticReferenceSolver::Cvxpy => "cvxpy",
            ExternalQuadraticReferenceSolver::Scs => "scs",
            ExternalQuadraticReferenceSolver::Clarabel => "clarabel",
            ExternalQuadraticReferenceSolver::Ecos => "ecos",
            ExternalQuadraticReferenceSolver::Mosek => "mosek",
            ExternalQuadraticReferenceSolver::Copt => "copt",
            ExternalQuadraticReferenceSolver::Qpoases => "qpoases",
            ExternalQuadraticReferenceSolver::Proxqp => "proxqp",
            ExternalQuadraticReferenceSolver::Cosmo => "cosmo",
            ExternalQuadraticReferenceSolver::Sdpa => "sdpa",
            ExternalQuadraticReferenceSolver::Csdp => "csdp",
            ExternalQuadraticReferenceSolver::Fallback => "fallback",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ExternalQuadraticReferenceSolver::Auto => "Auto",
            ExternalQuadraticReferenceSolver::Highs => "HiGHS/highspy",
            ExternalQuadraticReferenceSolver::Scipy => "SciPy SLSQP",
            ExternalQuadraticReferenceSolver::Osqp => "OSQP",
            ExternalQuadraticReferenceSolver::Cvxpy => "CVXPY",
            ExternalQuadraticReferenceSolver::Scs => "SCS",
            ExternalQuadraticReferenceSolver::Clarabel => "Clarabel",
            ExternalQuadraticReferenceSolver::Ecos => "ECOS",
            ExternalQuadraticReferenceSolver::Mosek => "MOSEK",
            ExternalQuadraticReferenceSolver::Copt => "COPT",
            ExternalQuadraticReferenceSolver::Qpoases => "qpOASES",
            ExternalQuadraticReferenceSolver::Proxqp => "ProxQP",
            ExternalQuadraticReferenceSolver::Cosmo => "COSMO",
            ExternalQuadraticReferenceSolver::Sdpa => "SDPA",
            ExternalQuadraticReferenceSolver::Csdp => "CSDP",
            ExternalQuadraticReferenceSolver::Fallback => "Dependency-free fallback",
        }
    }

    pub fn family(self) -> ExternalQuadraticReferenceFamily {
        match self {
            ExternalQuadraticReferenceSolver::Auto => ExternalQuadraticReferenceFamily::Auto,
            ExternalQuadraticReferenceSolver::Highs
            | ExternalQuadraticReferenceSolver::Scipy
            | ExternalQuadraticReferenceSolver::Osqp => {
                ExternalQuadraticReferenceFamily::DirectPythonApi
            }
            ExternalQuadraticReferenceSolver::Cvxpy
            | ExternalQuadraticReferenceSolver::Scs
            | ExternalQuadraticReferenceSolver::Clarabel
            | ExternalQuadraticReferenceSolver::Ecos
            | ExternalQuadraticReferenceSolver::Mosek
            | ExternalQuadraticReferenceSolver::Copt => ExternalQuadraticReferenceFamily::Cvxpy,
            ExternalQuadraticReferenceSolver::Qpoases
            | ExternalQuadraticReferenceSolver::Proxqp
            | ExternalQuadraticReferenceSolver::Cosmo
            | ExternalQuadraticReferenceSolver::Sdpa
            | ExternalQuadraticReferenceSolver::Csdp => {
                ExternalQuadraticReferenceFamily::RegisteredConic
            }
            ExternalQuadraticReferenceSolver::Fallback => {
                ExternalQuadraticReferenceFamily::Fallback
            }
        }
    }

    pub fn supports_qp(self) -> bool {
        true
    }

    pub fn supports_miqp(self) -> bool {
        matches!(
            self,
            ExternalQuadraticReferenceSolver::Auto
                | ExternalQuadraticReferenceSolver::Highs
                | ExternalQuadraticReferenceSolver::Scipy
                | ExternalQuadraticReferenceSolver::Fallback
        )
    }

    pub fn supports_socp(self) -> bool {
        matches!(
            self,
            ExternalQuadraticReferenceSolver::Auto
                | ExternalQuadraticReferenceSolver::Scipy
                | ExternalQuadraticReferenceSolver::Cvxpy
                | ExternalQuadraticReferenceSolver::Scs
                | ExternalQuadraticReferenceSolver::Clarabel
                | ExternalQuadraticReferenceSolver::Ecos
                | ExternalQuadraticReferenceSolver::Mosek
                | ExternalQuadraticReferenceSolver::Copt
                | ExternalQuadraticReferenceSolver::Qpoases
                | ExternalQuadraticReferenceSolver::Proxqp
                | ExternalQuadraticReferenceSolver::Cosmo
                | ExternalQuadraticReferenceSolver::Sdpa
                | ExternalQuadraticReferenceSolver::Csdp
                | ExternalQuadraticReferenceSolver::Fallback
        )
    }

    pub fn supports_qcp(self) -> bool {
        self.supports_socp()
    }

    pub fn notes(self) -> &'static str {
        match self.family() {
            ExternalQuadraticReferenceFamily::Auto => {
                "Prefer installed Python-backed solvers, then use the checked-in fallback for small models."
            }
            ExternalQuadraticReferenceFamily::DirectPythonApi => {
                "Direct Python package bridge; reports unavailable when the package is not installed."
            }
            ExternalQuadraticReferenceFamily::Cvxpy => {
                "CVXPY-dispatched solver; reports unavailable when CVXPY or the requested backend is not installed."
            }
            ExternalQuadraticReferenceFamily::RegisteredConic => {
                "Registered conic backend name with a checked-in fallback for deterministic validation coverage."
            }
            ExternalQuadraticReferenceFamily::Fallback => {
                "Dependency-free active-set or pattern-search fallback for small validation models."
            }
        }
    }

    pub fn spec(self) -> ExternalQuadraticReferenceSolverSpec {
        ExternalQuadraticReferenceSolverSpec {
            solver: self,
            id: self.as_arg(),
            display_name: self.display_name(),
            family: self.family(),
            supports_qp: self.supports_qp(),
            supports_miqp: self.supports_miqp(),
            supports_socp: self.supports_socp(),
            supports_qcp: self.supports_qcp(),
            notes: self.notes(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalQuadraticReferenceFamily {
    Auto,
    DirectPythonApi,
    Cvxpy,
    RegisteredConic,
    Fallback,
}

impl ExternalQuadraticReferenceFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalQuadraticReferenceFamily::Auto => "auto",
            ExternalQuadraticReferenceFamily::DirectPythonApi => "direct-python-api",
            ExternalQuadraticReferenceFamily::Cvxpy => "cvxpy",
            ExternalQuadraticReferenceFamily::RegisteredConic => "registered-conic",
            ExternalQuadraticReferenceFamily::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalQuadraticReferenceSolverSpec {
    pub solver: ExternalQuadraticReferenceSolver,
    pub id: &'static str,
    pub display_name: &'static str,
    pub family: ExternalQuadraticReferenceFamily,
    pub supports_qp: bool,
    pub supports_miqp: bool,
    pub supports_socp: bool,
    pub supports_qcp: bool,
    pub notes: &'static str,
}

pub fn external_quadratic_reference_solver_specs() -> Vec<ExternalQuadraticReferenceSolverSpec> {
    ExternalQuadraticReferenceSolver::all()
        .iter()
        .copied()
        .map(ExternalQuadraticReferenceSolver::spec)
        .collect()
}

pub fn external_quadratic_reference_solver_manifest() -> Value {
    Value::Array(
        external_quadratic_reference_solver_specs()
            .into_iter()
            .map(|spec| {
                json!({
                    "id": spec.id,
                    "displayName": spec.display_name,
                    "family": spec.family.as_str(),
                    "supportsQp": spec.supports_qp,
                    "supportsMiqp": spec.supports_miqp,
                    "supportsSocp": spec.supports_socp,
                    "supportsQcp": spec.supports_qcp,
                    "notes": spec.notes,
                })
            })
            .collect(),
    )
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

fn socp_json(problem: &SecondOrderConeProgram) -> Value {
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
    })
}

pub fn solve_socp_with_external_reference(
    problem: &SecondOrderConeProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    run_quadratic_reference_json(socp_json(problem), opts)
}

pub fn solve_misocp_with_external_reference(
    problem: &MixedIntegerSecondOrderConeProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    let mut payload = socp_json(&problem.socp);
    if let Some(map) = payload.as_object_mut() {
        map.insert("integer_vars".to_string(), json!(&problem.integer_vars));
    }
    run_quadratic_reference_json(payload, opts)
}

fn qcp_json(problem: &QuadraticallyConstrainedProgram) -> Value {
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
    })
}

pub fn solve_qcp_with_external_reference(
    problem: &QuadraticallyConstrainedProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    run_quadratic_reference_json(qcp_json(problem), opts)
}

pub fn solve_miqcp_with_external_reference(
    problem: &MixedIntegerQuadraticallyConstrainedProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    let mut payload = qcp_json(&problem.qcp);
    if let Some(map) = payload.as_object_mut() {
        map.insert("integer_vars".to_string(), json!(&problem.integer_vars));
    }
    run_quadratic_reference_json(payload, opts)
}

#[cfg(test)]
mod tests {
    use crate::des::general::external_quadratic_reference::{
        external_quadratic_reference_solver_manifest, external_quadratic_reference_solver_specs,
        ExternalQuadraticReferenceFamily, ExternalQuadraticReferenceSolver,
    };

    #[test]
    fn solver_args_cover_python_bridge_names() {
        assert_eq!(ExternalQuadraticReferenceSolver::all().len(), 16);
        assert_eq!(ExternalQuadraticReferenceSolver::Mosek.as_arg(), "mosek");
        assert_eq!(ExternalQuadraticReferenceSolver::Copt.as_arg(), "copt");
        assert_eq!(
            ExternalQuadraticReferenceSolver::Qpoases.as_arg(),
            "qpoases"
        );
        assert_eq!(ExternalQuadraticReferenceSolver::Proxqp.as_arg(), "proxqp");

        assert_eq!(
            ExternalQuadraticReferenceSolver::Mosek.family(),
            ExternalQuadraticReferenceFamily::Cvxpy
        );
        assert_eq!(
            ExternalQuadraticReferenceSolver::Qpoases.family(),
            ExternalQuadraticReferenceFamily::RegisteredConic
        );
        assert!(ExternalQuadraticReferenceSolver::Osqp.supports_qp());
        assert!(!ExternalQuadraticReferenceSolver::Osqp.supports_socp());
        assert!(ExternalQuadraticReferenceSolver::Mosek.supports_qp());
        assert!(ExternalQuadraticReferenceSolver::Mosek.supports_socp());
        assert!(ExternalQuadraticReferenceSolver::Mosek.supports_qcp());
        assert!(!ExternalQuadraticReferenceSolver::Mosek.supports_miqp());
        assert!(ExternalQuadraticReferenceSolver::Fallback.supports_miqp());
    }

    #[test]
    fn solver_manifest_exposes_optional_quadratic_backends() {
        let specs = external_quadratic_reference_solver_specs();
        assert_eq!(specs.len(), 16);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.family == ExternalQuadraticReferenceFamily::Cvxpy)
                .count(),
            6
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.family == ExternalQuadraticReferenceFamily::RegisteredConic)
                .count(),
            5
        );
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalQuadraticReferenceSolver::Copt
                && spec.id == "copt"
                && spec.supports_qp
                && spec.supports_socp
                && spec.supports_qcp
                && !spec.supports_miqp
        }));

        let manifest = external_quadratic_reference_solver_manifest();
        let items = manifest.as_array().expect("manifest array");
        assert_eq!(items.len(), 16);
        assert!(items.iter().any(|item| {
            item.get("id").and_then(|value| value.as_str()) == Some("mosek")
                && item.get("family").and_then(|value| value.as_str()) == Some("cvxpy")
                && item.get("supportsQcp").and_then(|value| value.as_bool()) == Some(true)
        }));
    }
}
