//! Rust-facing bridge for external/reference quadratic solvers.
//!
//! The native Rust reference uses the crate's active-set, enumeration, and
//! pattern-search solvers without Python startup. The checked-in Python bridge
//! (`scripts/qp_reference.py`) remains available for installed open-source
//! solvers such as HiGHS/highspy, SciPy, OSQP, and CVXPY backends.

use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::qp::{
    solve_miqp_enumeration, solve_qcp_pattern_search, solve_qp_active_set,
    solve_socp_pattern_search, MIQPOptions, MixedIntegerQuadraticProgram,
    MixedIntegerQuadraticallyConstrainedProgram, MixedIntegerSecondOrderConeProgram, QPOptions,
    QPStatus, QcpOptions, QcpStatus, QuadraticProgram, QuadraticallyConstrainedProgram,
    SecondOrderConeProgram, SocpOptions, SocpStatus,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalQuadraticReferenceSolver {
    Auto,
    RustInternal,
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
            ExternalQuadraticReferenceSolver::RustInternal,
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
            ExternalQuadraticReferenceSolver::RustInternal => "rust",
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
            ExternalQuadraticReferenceSolver::RustInternal => "Rust internal",
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
            ExternalQuadraticReferenceSolver::RustInternal => {
                ExternalQuadraticReferenceFamily::Fallback
            }
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
                | ExternalQuadraticReferenceSolver::RustInternal
                | ExternalQuadraticReferenceSolver::Highs
                | ExternalQuadraticReferenceSolver::Scipy
                | ExternalQuadraticReferenceSolver::Fallback
        )
    }

    pub fn supports_socp(self) -> bool {
        matches!(
            self,
            ExternalQuadraticReferenceSolver::Auto
                | ExternalQuadraticReferenceSolver::RustInternal
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
                "Use the native Rust reference by default; explicit solver ids opt into Python-backed external bridges."
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
                "Dependency-free Rust active-set, enumeration, or pattern-search fallback for small validation models."
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

fn status_from_qp_status(status: QPStatus) -> ExternalQuadraticReferenceStatus {
    match status {
        QPStatus::Optimal => ExternalQuadraticReferenceStatus::Optimal,
        QPStatus::Infeasible => ExternalQuadraticReferenceStatus::Infeasible,
        QPStatus::NumericalError => ExternalQuadraticReferenceStatus::NumericalError,
    }
}

fn status_from_socp_status(status: SocpStatus) -> ExternalQuadraticReferenceStatus {
    match status {
        SocpStatus::Optimal => ExternalQuadraticReferenceStatus::Optimal,
        SocpStatus::Infeasible => ExternalQuadraticReferenceStatus::Infeasible,
        SocpStatus::NumericalError => ExternalQuadraticReferenceStatus::NumericalError,
    }
}

fn status_from_qcp_status(status: QcpStatus) -> ExternalQuadraticReferenceStatus {
    match status {
        QcpStatus::Optimal => ExternalQuadraticReferenceStatus::Optimal,
        QcpStatus::Infeasible => ExternalQuadraticReferenceStatus::Infeasible,
        QcpStatus::NumericalError => ExternalQuadraticReferenceStatus::NumericalError,
    }
}

fn rust_quadratic_empty_solution(
    status: ExternalQuadraticReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalQuadraticReferenceSolution {
    ExternalQuadraticReferenceSolution {
        status,
        solver: solver.into(),
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

fn panic_message(error: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = error.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = error.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "Rust quadratic reference panicked".to_string()
    }
}

fn is_rust_quadratic_solver(opts: &ExternalQuadraticReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalQuadraticReferenceSolver::Auto
            | ExternalQuadraticReferenceSolver::RustInternal
            | ExternalQuadraticReferenceSolver::Fallback
    )
}

fn solve_qp_with_rust_reference(
    problem: &QuadraticProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    let started = Instant::now();
    let max_active_sets = opts.max_enumerations.unwrap_or(1_000_000);
    let result = catch_unwind(AssertUnwindSafe(|| {
        solve_qp_active_set(
            problem,
            QPOptions {
                max_active_sets,
                ..QPOptions::default()
            },
        )
    }));
    let solution = match result {
        Ok(solution) => solution,
        Err(error) => {
            return rust_quadratic_empty_solution(
                ExternalQuadraticReferenceStatus::NumericalError,
                "rust:qp-active-set",
                panic_message(error),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let status = status_from_qp_status(solution.status);
    ExternalQuadraticReferenceSolution {
        status,
        solver: "rust:qp-active-set".to_string(),
        x: solution.x,
        objective: (status == ExternalQuadraticReferenceStatus::Optimal)
            .then_some(solution.objective),
        dual_ub: Some(solution.dual_ub),
        dual_eq: Some(solution.dual_eq),
        dual_lower_bounds: Some(solution.dual_lower_bounds),
        dual_upper_bounds: Some(solution.dual_upper_bounds),
        reduced_gradient: Some(solution.reduced_gradient),
        iterations: Some(solution.iterations as u64),
        enumerated: None,
        message: solution
            .message
            .unwrap_or_else(|| "Rust QP active-set enumeration".to_string()),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn solve_miqp_with_rust_reference(
    problem: &MixedIntegerQuadraticProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    let started = Instant::now();
    let max_enumerations = opts.max_enumerations.unwrap_or(1_000_000);
    let result = catch_unwind(AssertUnwindSafe(|| {
        solve_miqp_enumeration(
            problem,
            MIQPOptions {
                max_enumerations,
                qp_options: QPOptions::default(),
            },
        )
    }));
    let solution = match result {
        Ok(solution) => solution,
        Err(error) => {
            return rust_quadratic_empty_solution(
                ExternalQuadraticReferenceStatus::NumericalError,
                "rust:miqp-enumeration",
                panic_message(error),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let status = status_from_qp_status(solution.status);
    ExternalQuadraticReferenceSolution {
        status,
        solver: "rust:miqp-enumeration".to_string(),
        x: solution.x,
        objective: (status == ExternalQuadraticReferenceStatus::Optimal)
            .then_some(solution.objective),
        dual_ub: None,
        dual_eq: None,
        dual_lower_bounds: None,
        dual_upper_bounds: None,
        reduced_gradient: None,
        iterations: Some(solution.qp_subproblems as u64),
        enumerated: Some(solution.enumerated as u64),
        message: solution
            .message
            .unwrap_or_else(|| "Rust MIQP enumeration".to_string()),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn solve_socp_with_rust_reference(
    problem: &SecondOrderConeProgram,
) -> ExternalQuadraticReferenceSolution {
    let started = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(|| {
        solve_socp_pattern_search(problem, SocpOptions::default())
    }));
    let solution = match result {
        Ok(solution) => solution,
        Err(error) => {
            return rust_quadratic_empty_solution(
                ExternalQuadraticReferenceStatus::NumericalError,
                "rust:socp-pattern-search",
                panic_message(error),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let status = status_from_socp_status(solution.status);
    ExternalQuadraticReferenceSolution {
        status,
        solver: "rust:socp-pattern-search".to_string(),
        x: solution.x,
        objective: (status == ExternalQuadraticReferenceStatus::Optimal)
            .then_some(solution.objective),
        dual_ub: None,
        dual_eq: None,
        dual_lower_bounds: None,
        dual_upper_bounds: None,
        reduced_gradient: None,
        iterations: Some(solution.iterations as u64),
        enumerated: None,
        message: solution
            .message
            .unwrap_or_else(|| "Rust SOCP pattern search".to_string()),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn solve_qcp_with_rust_reference(
    problem: &QuadraticallyConstrainedProgram,
) -> ExternalQuadraticReferenceSolution {
    let started = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(|| {
        solve_qcp_pattern_search(problem, QcpOptions::default())
    }));
    let solution = match result {
        Ok(solution) => solution,
        Err(error) => {
            return rust_quadratic_empty_solution(
                ExternalQuadraticReferenceStatus::NumericalError,
                "rust:qcp-pattern-search",
                panic_message(error),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let status = status_from_qcp_status(solution.status);
    ExternalQuadraticReferenceSolution {
        status,
        solver: "rust:qcp-pattern-search".to_string(),
        x: solution.x,
        objective: (status == ExternalQuadraticReferenceStatus::Optimal)
            .then_some(solution.objective),
        dual_ub: None,
        dual_eq: None,
        dual_lower_bounds: None,
        dual_upper_bounds: None,
        reduced_gradient: None,
        iterations: Some(solution.iterations as u64),
        enumerated: None,
        message: solution
            .message
            .unwrap_or_else(|| "Rust QCP pattern search".to_string()),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IntegerDomain {
    var: usize,
    lower: i64,
    upper: i64,
}

fn integer_domains_for_rust_reference(
    kind: &str,
    n: usize,
    integer_vars: &[bool],
    lb: &Option<Vec<Option<f64>>>,
    ub: &Option<Vec<Option<f64>>>,
) -> Result<Vec<IntegerDomain>, String> {
    if integer_vars.len() != n {
        return Err(format!(
            "{kind}: integer_vars length {} != variable count {n}",
            integer_vars.len()
        ));
    }
    if let Some(bounds) = lb {
        if bounds.len() != n {
            return Err(format!(
                "{kind}: lb length {} != variable count {n}",
                bounds.len()
            ));
        }
    }
    if let Some(bounds) = ub {
        if bounds.len() != n {
            return Err(format!(
                "{kind}: ub length {} != variable count {n}",
                bounds.len()
            ));
        }
    }

    let mut domains = Vec::new();
    for (idx, &is_integer) in integer_vars.iter().enumerate() {
        if !is_integer {
            continue;
        }
        let lower = lb
            .as_ref()
            .and_then(|bounds| bounds[idx])
            .ok_or_else(|| format!("{kind}: integer variable {idx} needs a finite lower bound"))?;
        let upper = ub
            .as_ref()
            .and_then(|bounds| bounds[idx])
            .ok_or_else(|| format!("{kind}: integer variable {idx} needs a finite upper bound"))?;
        if !lower.is_finite() || !upper.is_finite() {
            return Err(format!(
                "{kind}: integer variable {idx} needs finite bounds"
            ));
        }
        let lower = lower.ceil();
        let upper = upper.floor();
        if lower > upper {
            return Err(format!(
                "{kind}: integer variable {idx} has no integer value in its bounds"
            ));
        }
        if lower < i64::MIN as f64 || upper > i64::MAX as f64 {
            return Err(format!(
                "{kind}: integer variable {idx} bounds exceed i64 enumeration range"
            ));
        }
        domains.push(IntegerDomain {
            var: idx,
            lower: lower as i64,
            upper: upper as i64,
        });
    }
    Ok(domains)
}

fn enumerate_integer_assignments<F>(
    depth: usize,
    domains: &[IntegerDomain],
    max_enumerations: usize,
    current: &mut Vec<(usize, f64)>,
    enumerated: &mut usize,
    hit_limit: &mut bool,
    visit: &mut F,
) where
    F: FnMut(&[(usize, f64)]),
{
    if *hit_limit {
        return;
    }
    if depth == domains.len() {
        *enumerated += 1;
        if *enumerated > max_enumerations {
            *hit_limit = true;
            return;
        }
        visit(current);
        return;
    }

    let domain = domains[depth];
    for value in domain.lower..=domain.upper {
        current.push((domain.var, value as f64));
        enumerate_integer_assignments(
            depth + 1,
            domains,
            max_enumerations,
            current,
            enumerated,
            hit_limit,
            visit,
        );
        current.pop();
        if *hit_limit {
            return;
        }
    }
}

fn fixed_socp_integer_subproblem(
    problem: &MixedIntegerSecondOrderConeProgram,
    assignment: &[(usize, f64)],
) -> SecondOrderConeProgram {
    let n = problem.socp.c.len();
    let mut sub = problem.socp.clone();
    let mut a_eq = sub.a_eq.clone().unwrap_or_default();
    let mut b_eq = sub.b_eq.clone().unwrap_or_default();
    let mut lb = sub.lb.clone().unwrap_or_else(|| vec![None; n]);
    let mut ub = sub.ub.clone().unwrap_or_else(|| vec![None; n]);
    for &(var, value) in assignment {
        let mut row = vec![0.0; n];
        row[var] = 1.0;
        a_eq.push(row);
        b_eq.push(value);
        lb[var] = Some(value);
        ub[var] = Some(value);
    }
    sub.a_eq = Some(a_eq);
    sub.b_eq = Some(b_eq);
    sub.lb = Some(lb);
    sub.ub = Some(ub);
    sub
}

fn fixed_qcp_integer_subproblem(
    problem: &MixedIntegerQuadraticallyConstrainedProgram,
    assignment: &[(usize, f64)],
) -> QuadraticallyConstrainedProgram {
    let n = problem.qcp.c.len();
    let mut sub = problem.qcp.clone();
    let mut a_eq = sub.a_eq.clone().unwrap_or_default();
    let mut b_eq = sub.b_eq.clone().unwrap_or_default();
    let mut lb = sub.lb.clone().unwrap_or_else(|| vec![None; n]);
    let mut ub = sub.ub.clone().unwrap_or_else(|| vec![None; n]);
    for &(var, value) in assignment {
        let mut row = vec![0.0; n];
        row[var] = 1.0;
        a_eq.push(row);
        b_eq.push(value);
        lb[var] = Some(value);
        ub[var] = Some(value);
    }
    sub.a_eq = Some(a_eq);
    sub.b_eq = Some(b_eq);
    sub.lb = Some(lb);
    sub.ub = Some(ub);
    sub
}

fn solve_misocp_with_rust_reference(
    problem: &MixedIntegerSecondOrderConeProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    let started = Instant::now();
    let domains = match integer_domains_for_rust_reference(
        "misocp",
        problem.socp.c.len(),
        &problem.integer_vars,
        &problem.socp.lb,
        &problem.socp.ub,
    ) {
        Ok(domains) => domains,
        Err(message) => {
            return rust_quadratic_empty_solution(
                ExternalQuadraticReferenceStatus::NumericalError,
                "rust:misocp-enumeration",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if domains.is_empty() {
        let mut solution = solve_socp_with_rust_reference(&problem.socp);
        solution.solver = "rust:misocp-enumeration".to_string();
        solution.enumerated = Some(1);
        solution.message = format!("no integer variables; {}", solution.message);
        return solution;
    }

    let max_enumerations = opts.max_enumerations.unwrap_or(1_000_000);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut current = Vec::with_capacity(domains.len());
        let mut best_x = Vec::new();
        let mut best_obj = f64::INFINITY;
        let mut iterations = 0u64;
        let mut enumerated = 0usize;
        let mut hit_limit = false;
        let mut saw_numerical = false;
        enumerate_integer_assignments(
            0,
            &domains,
            max_enumerations,
            &mut current,
            &mut enumerated,
            &mut hit_limit,
            &mut |assignment| {
                let sub = fixed_socp_integer_subproblem(problem, assignment);
                let solution = solve_socp_pattern_search(&sub, SocpOptions::default());
                iterations += solution.iterations as u64;
                match solution.status {
                    SocpStatus::Optimal if solution.objective < best_obj - 1e-7 => {
                        best_obj = solution.objective;
                        best_x = solution.x;
                    }
                    SocpStatus::NumericalError => saw_numerical = true,
                    SocpStatus::Optimal | SocpStatus::Infeasible => {}
                }
            },
        );
        (
            best_x,
            best_obj,
            iterations,
            enumerated,
            hit_limit,
            saw_numerical,
        )
    }));
    let (best_x, best_obj, iterations, enumerated, hit_limit, saw_numerical) = match result {
        Ok(result) => result,
        Err(error) => {
            return rust_quadratic_empty_solution(
                ExternalQuadraticReferenceStatus::NumericalError,
                "rust:misocp-enumeration",
                panic_message(error),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if hit_limit {
        return ExternalQuadraticReferenceSolution {
            status: ExternalQuadraticReferenceStatus::NumericalError,
            solver: "rust:misocp-enumeration".to_string(),
            x: best_x,
            objective: None,
            dual_ub: None,
            dual_eq: None,
            dual_lower_bounds: None,
            dual_upper_bounds: None,
            reduced_gradient: None,
            iterations: Some(iterations),
            enumerated: Some(enumerated as u64),
            message: "Rust MISOCP enumeration limit reached".to_string(),
            elapsed_ms,
        };
    }
    if best_x.is_empty() {
        return ExternalQuadraticReferenceSolution {
            status: if saw_numerical {
                ExternalQuadraticReferenceStatus::NumericalError
            } else {
                ExternalQuadraticReferenceStatus::Infeasible
            },
            solver: "rust:misocp-enumeration".to_string(),
            x: Vec::new(),
            objective: None,
            dual_ub: None,
            dual_eq: None,
            dual_lower_bounds: None,
            dual_upper_bounds: None,
            reduced_gradient: None,
            iterations: Some(iterations),
            enumerated: Some(enumerated as u64),
            message: if saw_numerical {
                "Rust MISOCP enumeration saw only numerical subproblem failures".to_string()
            } else {
                "no feasible integer SOCP assignment found".to_string()
            },
            elapsed_ms,
        };
    }
    ExternalQuadraticReferenceSolution {
        status: ExternalQuadraticReferenceStatus::Optimal,
        solver: "rust:misocp-enumeration".to_string(),
        x: best_x,
        objective: Some(best_obj),
        dual_ub: None,
        dual_eq: None,
        dual_lower_bounds: None,
        dual_upper_bounds: None,
        reduced_gradient: None,
        iterations: Some(iterations),
        enumerated: Some(enumerated as u64),
        message: "bounded Rust MISOCP enumeration over integer variables".to_string(),
        elapsed_ms,
    }
}

fn solve_miqcp_with_rust_reference(
    problem: &MixedIntegerQuadraticallyConstrainedProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    let started = Instant::now();
    let domains = match integer_domains_for_rust_reference(
        "miqcp",
        problem.qcp.c.len(),
        &problem.integer_vars,
        &problem.qcp.lb,
        &problem.qcp.ub,
    ) {
        Ok(domains) => domains,
        Err(message) => {
            return rust_quadratic_empty_solution(
                ExternalQuadraticReferenceStatus::NumericalError,
                "rust:miqcp-enumeration",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if domains.is_empty() {
        let mut solution = solve_qcp_with_rust_reference(&problem.qcp);
        solution.solver = "rust:miqcp-enumeration".to_string();
        solution.enumerated = Some(1);
        solution.message = format!("no integer variables; {}", solution.message);
        return solution;
    }

    let max_enumerations = opts.max_enumerations.unwrap_or(1_000_000);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut current = Vec::with_capacity(domains.len());
        let mut best_x = Vec::new();
        let mut best_obj = f64::INFINITY;
        let mut iterations = 0u64;
        let mut enumerated = 0usize;
        let mut hit_limit = false;
        let mut saw_numerical = false;
        enumerate_integer_assignments(
            0,
            &domains,
            max_enumerations,
            &mut current,
            &mut enumerated,
            &mut hit_limit,
            &mut |assignment| {
                let sub = fixed_qcp_integer_subproblem(problem, assignment);
                let solution = solve_qcp_pattern_search(&sub, QcpOptions::default());
                iterations += solution.iterations as u64;
                match solution.status {
                    QcpStatus::Optimal if solution.objective < best_obj - 1e-7 => {
                        best_obj = solution.objective;
                        best_x = solution.x;
                    }
                    QcpStatus::NumericalError => saw_numerical = true,
                    QcpStatus::Optimal | QcpStatus::Infeasible => {}
                }
            },
        );
        (
            best_x,
            best_obj,
            iterations,
            enumerated,
            hit_limit,
            saw_numerical,
        )
    }));
    let (best_x, best_obj, iterations, enumerated, hit_limit, saw_numerical) = match result {
        Ok(result) => result,
        Err(error) => {
            return rust_quadratic_empty_solution(
                ExternalQuadraticReferenceStatus::NumericalError,
                "rust:miqcp-enumeration",
                panic_message(error),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if hit_limit {
        return ExternalQuadraticReferenceSolution {
            status: ExternalQuadraticReferenceStatus::NumericalError,
            solver: "rust:miqcp-enumeration".to_string(),
            x: best_x,
            objective: None,
            dual_ub: None,
            dual_eq: None,
            dual_lower_bounds: None,
            dual_upper_bounds: None,
            reduced_gradient: None,
            iterations: Some(iterations),
            enumerated: Some(enumerated as u64),
            message: "Rust MIQCP enumeration limit reached".to_string(),
            elapsed_ms,
        };
    }
    if best_x.is_empty() {
        return ExternalQuadraticReferenceSolution {
            status: if saw_numerical {
                ExternalQuadraticReferenceStatus::NumericalError
            } else {
                ExternalQuadraticReferenceStatus::Infeasible
            },
            solver: "rust:miqcp-enumeration".to_string(),
            x: Vec::new(),
            objective: None,
            dual_ub: None,
            dual_eq: None,
            dual_lower_bounds: None,
            dual_upper_bounds: None,
            reduced_gradient: None,
            iterations: Some(iterations),
            enumerated: Some(enumerated as u64),
            message: if saw_numerical {
                "Rust MIQCP enumeration saw only numerical subproblem failures".to_string()
            } else {
                "no feasible integer QCP assignment found".to_string()
            },
            elapsed_ms,
        };
    }
    ExternalQuadraticReferenceSolution {
        status: ExternalQuadraticReferenceStatus::Optimal,
        solver: "rust:miqcp-enumeration".to_string(),
        x: best_x,
        objective: Some(best_obj),
        dual_ub: None,
        dual_eq: None,
        dual_lower_bounds: None,
        dual_upper_bounds: None,
        reduced_gradient: None,
        iterations: Some(iterations),
        enumerated: Some(enumerated as u64),
        message: "bounded Rust MIQCP enumeration over integer variables".to_string(),
        elapsed_ms,
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

fn quadratic_reference_timeout_ms() -> u64 {
    std::env::var("QUADRATIC_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_quadratic_reference_output(
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
            Err(err) => return Err(format!("failed to poll qp_reference.py: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for qp_reference.py: {err}"))
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
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write qp_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = quadratic_reference_timeout_ms();
    let (output, timed_out) = match wait_for_quadratic_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0),
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("qp_reference.py timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; qp_reference.py timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
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
                    stderr.clone()
                }
            }),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse qp_reference.py output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

pub fn quadratic_program_to_reference_json(problem: &QuadraticProgram) -> Value {
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
    if is_rust_quadratic_solver(opts) {
        return solve_qp_with_rust_reference(problem, opts);
    }

    run_quadratic_reference_json(quadratic_program_to_reference_json(problem), opts)
}

pub fn solve_miqp_with_external_reference(
    problem: &MixedIntegerQuadraticProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    if is_rust_quadratic_solver(opts) {
        return solve_miqp_with_rust_reference(problem, opts);
    }

    let mut payload = quadratic_program_to_reference_json(&problem.qp);
    if let Some(map) = payload.as_object_mut() {
        map.insert("integer_vars".to_string(), json!(&problem.integer_vars));
    }
    run_quadratic_reference_json(payload, opts)
}

pub fn second_order_cone_program_to_reference_json(problem: &SecondOrderConeProgram) -> Value {
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
    if is_rust_quadratic_solver(opts) {
        return solve_socp_with_rust_reference(problem);
    }

    run_quadratic_reference_json(second_order_cone_program_to_reference_json(problem), opts)
}

pub fn solve_misocp_with_external_reference(
    problem: &MixedIntegerSecondOrderConeProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    if is_rust_quadratic_solver(opts) {
        return solve_misocp_with_rust_reference(problem, opts);
    }

    let mut payload = second_order_cone_program_to_reference_json(&problem.socp);
    if let Some(map) = payload.as_object_mut() {
        map.insert("integer_vars".to_string(), json!(&problem.integer_vars));
    }
    run_quadratic_reference_json(payload, opts)
}

pub fn quadratically_constrained_program_to_reference_json(
    problem: &QuadraticallyConstrainedProgram,
) -> Value {
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
    if is_rust_quadratic_solver(opts) {
        return solve_qcp_with_rust_reference(problem);
    }

    run_quadratic_reference_json(
        quadratically_constrained_program_to_reference_json(problem),
        opts,
    )
}

pub fn solve_miqcp_with_external_reference(
    problem: &MixedIntegerQuadraticallyConstrainedProgram,
    opts: &ExternalQuadraticReferenceOptions,
) -> ExternalQuadraticReferenceSolution {
    if is_rust_quadratic_solver(opts) {
        return solve_miqcp_with_rust_reference(problem, opts);
    }

    let mut payload = quadratically_constrained_program_to_reference_json(&problem.qcp);
    if let Some(map) = payload.as_object_mut() {
        map.insert("integer_vars".to_string(), json!(&problem.integer_vars));
    }
    run_quadratic_reference_json(payload, opts)
}

#[cfg(test)]
mod tests {
    use super::wait_for_quadratic_reference_output;
    use crate::des::general::qp::{
        QuadraticConstraint, QuadraticProgram, SecondOrderCone, SecondOrderConeProgram,
    };

    use crate::des::general::external_quadratic_reference::{
        external_quadratic_reference_solver_manifest, external_quadratic_reference_solver_specs,
        solve_miqcp_with_external_reference, solve_misocp_with_external_reference,
        solve_qp_with_external_reference, ExternalQuadraticReferenceFamily,
        ExternalQuadraticReferenceOptions, ExternalQuadraticReferenceSolution,
        ExternalQuadraticReferenceSolver, ExternalQuadraticReferenceStatus,
    };
    use std::process::{Command, Stdio};

    #[test]
    fn solver_args_cover_python_bridge_names() {
        assert_eq!(ExternalQuadraticReferenceSolver::all().len(), 17);
        assert_eq!(
            ExternalQuadraticReferenceSolver::RustInternal.as_arg(),
            "rust"
        );
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
        assert!(ExternalQuadraticReferenceSolver::RustInternal.supports_miqp());
        assert!(ExternalQuadraticReferenceSolver::RustInternal.supports_socp());
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
        assert_eq!(specs.len(), 17);
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
            spec.solver == ExternalQuadraticReferenceSolver::RustInternal
                && spec.id == "rust"
                && spec.supports_qp
                && spec.supports_miqp
                && spec.supports_socp
                && spec.supports_qcp
        }));
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
        assert_eq!(items.len(), 17);
        assert!(items.iter().any(|item| {
            item.get("id").and_then(|value| value.as_str()) == Some("rust")
                && item.get("supportsMiqp").and_then(|value| value.as_bool()) == Some(true)
        }));
        assert!(items.iter().any(|item| {
            item.get("id").and_then(|value| value.as_str()) == Some("auto")
                && item
                    .get("notes")
                    .and_then(|value| value.as_str())
                    .is_some_and(|notes| notes.contains("native Rust reference"))
        }));
        assert!(items.iter().any(|item| {
            item.get("id").and_then(|value| value.as_str()) == Some("mosek")
                && item.get("family").and_then(|value| value.as_str()) == Some("cvxpy")
                && item.get("supportsQcp").and_then(|value| value.as_bool()) == Some(true)
        }));
    }

    fn rust_solver_options() -> ExternalQuadraticReferenceOptions {
        ExternalQuadraticReferenceOptions {
            solver: ExternalQuadraticReferenceSolver::RustInternal,
            ..Default::default()
        }
    }

    fn fallback_solver_options() -> ExternalQuadraticReferenceOptions {
        ExternalQuadraticReferenceOptions {
            solver: ExternalQuadraticReferenceSolver::Fallback,
            ..Default::default()
        }
    }

    fn assert_optimal(solution: &ExternalQuadraticReferenceSolution) {
        assert_eq!(
            solution.status,
            ExternalQuadraticReferenceStatus::Optimal,
            "{solution:?}"
        );
    }

    #[test]
    fn auto_prefers_rust_qp_reference_without_python() {
        let problem = QuadraticProgram {
            q: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            c: vec![-2.0, -4.0],
            lb: Some(vec![Some(0.0), Some(0.0)]),
            ub: Some(vec![Some(5.0), Some(5.0)]),
            ..Default::default()
        };

        let solution = solve_qp_with_external_reference(
            &problem,
            &ExternalQuadraticReferenceOptions::default(),
        );

        assert_optimal(&solution);
        assert_eq!(solution.solver, "rust:qp-active-set");
        assert!((solution.x[0] - 1.0).abs() <= 1e-7, "{solution:?}");
        assert!((solution.x[1] - 2.0).abs() <= 1e-7, "{solution:?}");
        assert!(solution.objective.is_some());
    }

    #[test]
    fn fallback_alias_uses_rust_quadratic_reference_without_python() {
        let qp = QuadraticProgram {
            q: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            c: vec![-2.0, -4.0],
            lb: Some(vec![Some(0.0), Some(0.0)]),
            ub: Some(vec![Some(5.0), Some(5.0)]),
            ..Default::default()
        };
        let qp_solution = solve_qp_with_external_reference(&qp, &fallback_solver_options());
        assert_optimal(&qp_solution);
        assert_eq!(qp_solution.solver, "rust:qp-active-set");
        assert!((qp_solution.x[0] - 1.0).abs() <= 1e-7, "{qp_solution:?}");
        assert!((qp_solution.x[1] - 2.0).abs() <= 1e-7, "{qp_solution:?}");

        let miqcp = super::MixedIntegerQuadraticallyConstrainedProgram {
            qcp: super::QuadraticallyConstrainedProgram {
                q: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                c: vec![0.0, 1.0],
                lb: Some(vec![Some(3.0), Some(0.0)]),
                ub: Some(vec![Some(3.0), Some(20.0)]),
                quadratic_constraints: vec![QuadraticConstraint {
                    q: vec![vec![1.0, 0.0], vec![0.0, 0.0]],
                    c: vec![0.0, -1.0],
                    rhs: 0.0,
                    name: Some("integer-square-epigraph".to_string()),
                }],
                ..Default::default()
            },
            integer_vars: vec![true, true],
        };
        let miqcp_solution =
            solve_miqcp_with_external_reference(&miqcp, &fallback_solver_options());
        assert_optimal(&miqcp_solution);
        assert_eq!(miqcp_solution.solver, "rust:miqcp-enumeration");
        assert!(miqcp_solution
            .objective
            .is_some_and(|objective| { (objective - 9.0).abs() <= 1e-7 }));
        assert_eq!(miqcp_solution.enumerated, Some(21));
    }

    #[test]
    fn rust_internal_solves_misocp_enumeration_reference() {
        let problem = super::MixedIntegerSecondOrderConeProgram {
            socp: SecondOrderConeProgram {
                c: vec![0.0, 0.0, 1.0],
                lb: Some(vec![Some(2.0), Some(2.0), Some(0.0)]),
                ub: Some(vec![Some(2.0), Some(2.0), Some(10.0)]),
                cones: vec![SecondOrderCone {
                    a: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
                    b: vec![0.0, 0.0],
                    c: vec![0.0, 0.0, 1.0],
                    d: 0.0,
                    name: Some("integer-norm".to_string()),
                }],
                ..Default::default()
            },
            integer_vars: vec![true, true, true],
        };
        let solution = solve_misocp_with_external_reference(&problem, &rust_solver_options());
        assert_optimal(&solution);
        assert_eq!(solution.solver, "rust:misocp-enumeration");
        assert!(solution
            .objective
            .is_some_and(|objective| { (objective - 3.0).abs() <= 1e-7 }));
        assert!((solution.x[0] - 2.0).abs() <= 1e-7, "{solution:?}");
        assert!((solution.x[1] - 2.0).abs() <= 1e-7, "{solution:?}");
        assert!((solution.x[2] - 3.0).abs() <= 1e-7, "{solution:?}");
        assert_eq!(solution.enumerated, Some(11));
    }

    #[test]
    fn rust_internal_solves_miqcp_enumeration_reference() {
        let problem = super::MixedIntegerQuadraticallyConstrainedProgram {
            qcp: super::QuadraticallyConstrainedProgram {
                q: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                c: vec![0.0, 1.0],
                lb: Some(vec![Some(3.0), Some(0.0)]),
                ub: Some(vec![Some(3.0), Some(20.0)]),
                quadratic_constraints: vec![QuadraticConstraint {
                    q: vec![vec![1.0, 0.0], vec![0.0, 0.0]],
                    c: vec![0.0, -1.0],
                    rhs: 0.0,
                    name: Some("integer-square-epigraph".to_string()),
                }],
                ..Default::default()
            },
            integer_vars: vec![true, true],
        };
        let solution = solve_miqcp_with_external_reference(&problem, &rust_solver_options());
        assert_optimal(&solution);
        assert_eq!(solution.solver, "rust:miqcp-enumeration");
        assert!(solution
            .objective
            .is_some_and(|objective| { (objective - 9.0).abs() <= 1e-7 }));
        assert!((solution.x[0] - 3.0).abs() <= 1e-7, "{solution:?}");
        assert!((solution.x[1] - 9.0).abs() <= 1e-7, "{solution:?}");
        assert_eq!(solution.enumerated, Some(21));
    }

    #[test]
    fn auto_prefers_rust_miqcp_reference_without_python() {
        let problem = super::MixedIntegerQuadraticallyConstrainedProgram {
            qcp: super::QuadraticallyConstrainedProgram {
                q: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                c: vec![0.0, 1.0],
                lb: Some(vec![Some(3.0), Some(0.0)]),
                ub: Some(vec![Some(3.0), Some(20.0)]),
                quadratic_constraints: vec![QuadraticConstraint {
                    q: vec![vec![1.0, 0.0], vec![0.0, 0.0]],
                    c: vec![0.0, -1.0],
                    rhs: 0.0,
                    name: Some("integer-square-epigraph".to_string()),
                }],
                ..Default::default()
            },
            integer_vars: vec![true, true],
        };

        let solution = solve_miqcp_with_external_reference(
            &problem,
            &ExternalQuadraticReferenceOptions::default(),
        );

        assert_optimal(&solution);
        assert_eq!(solution.solver, "rust:miqcp-enumeration");
        assert!(solution
            .objective
            .is_some_and(|objective| { (objective - 9.0).abs() <= 1e-7 }));
        assert_eq!(solution.enumerated, Some(21));
    }

    #[test]
    fn quadratic_python_bridge_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_quadratic_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
