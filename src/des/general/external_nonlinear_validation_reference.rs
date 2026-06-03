//! Rust-facing bridge for nonlinear validation payloads.
//!
//! The checked-in Python bridge (`scripts/nonlinear_validation_reference.py`)
//! accepts compact expression-based NLP smoke models and keeps heavyweight
//! solvers optional. This module exposes the same registered solver names through
//! typed Rust options, then delegates execution to the local Python reference.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearValidationReferenceSolver {
    Auto,
    Scipy,
    Ipopt,
    Bonmin,
    Minotaur,
    Couenne,
    Symphony,
    Knitro,
    Mosek,
    Baron,
    Copt,
    Casadi,
    Nlopt,
    NloptCli,
    Fallback,
}

impl ExternalNonlinearValidationReferenceSolver {
    pub fn all() -> &'static [ExternalNonlinearValidationReferenceSolver] {
        &[
            ExternalNonlinearValidationReferenceSolver::Auto,
            ExternalNonlinearValidationReferenceSolver::Scipy,
            ExternalNonlinearValidationReferenceSolver::Ipopt,
            ExternalNonlinearValidationReferenceSolver::Bonmin,
            ExternalNonlinearValidationReferenceSolver::Minotaur,
            ExternalNonlinearValidationReferenceSolver::Couenne,
            ExternalNonlinearValidationReferenceSolver::Symphony,
            ExternalNonlinearValidationReferenceSolver::Knitro,
            ExternalNonlinearValidationReferenceSolver::Mosek,
            ExternalNonlinearValidationReferenceSolver::Baron,
            ExternalNonlinearValidationReferenceSolver::Copt,
            ExternalNonlinearValidationReferenceSolver::Casadi,
            ExternalNonlinearValidationReferenceSolver::Nlopt,
            ExternalNonlinearValidationReferenceSolver::NloptCli,
            ExternalNonlinearValidationReferenceSolver::Fallback,
        ]
    }

    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalNonlinearValidationReferenceSolver::Auto => "auto",
            ExternalNonlinearValidationReferenceSolver::Scipy => "scipy",
            ExternalNonlinearValidationReferenceSolver::Ipopt => "ipopt",
            ExternalNonlinearValidationReferenceSolver::Bonmin => "bonmin",
            ExternalNonlinearValidationReferenceSolver::Minotaur => "minotaur",
            ExternalNonlinearValidationReferenceSolver::Couenne => "couenne",
            ExternalNonlinearValidationReferenceSolver::Symphony => "symphony",
            ExternalNonlinearValidationReferenceSolver::Knitro => "knitro",
            ExternalNonlinearValidationReferenceSolver::Mosek => "mosek",
            ExternalNonlinearValidationReferenceSolver::Baron => "baron",
            ExternalNonlinearValidationReferenceSolver::Copt => "copt",
            ExternalNonlinearValidationReferenceSolver::Casadi => "casadi",
            ExternalNonlinearValidationReferenceSolver::Nlopt => "nlopt",
            ExternalNonlinearValidationReferenceSolver::NloptCli => "nlopt-cli",
            ExternalNonlinearValidationReferenceSolver::Fallback => "fallback",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ExternalNonlinearValidationReferenceSolver::Auto => "Auto",
            ExternalNonlinearValidationReferenceSolver::Scipy => "SciPy SLSQP",
            ExternalNonlinearValidationReferenceSolver::Ipopt => "Ipopt",
            ExternalNonlinearValidationReferenceSolver::Bonmin => "Bonmin",
            ExternalNonlinearValidationReferenceSolver::Minotaur => "MINOTAUR",
            ExternalNonlinearValidationReferenceSolver::Couenne => "Couenne",
            ExternalNonlinearValidationReferenceSolver::Symphony => "COIN-OR SYMPHONY",
            ExternalNonlinearValidationReferenceSolver::Knitro => "Artelys Knitro",
            ExternalNonlinearValidationReferenceSolver::Mosek => "MOSEK",
            ExternalNonlinearValidationReferenceSolver::Baron => "BARON",
            ExternalNonlinearValidationReferenceSolver::Copt => "COPT",
            ExternalNonlinearValidationReferenceSolver::Casadi => "CasADi",
            ExternalNonlinearValidationReferenceSolver::Nlopt => "NLopt",
            ExternalNonlinearValidationReferenceSolver::NloptCli => "NLopt CLI",
            ExternalNonlinearValidationReferenceSolver::Fallback => "Pattern-search fallback",
        }
    }

    pub fn family(self) -> ExternalNonlinearValidationReferenceFamily {
        match self {
            ExternalNonlinearValidationReferenceSolver::Auto => {
                ExternalNonlinearValidationReferenceFamily::Auto
            }
            ExternalNonlinearValidationReferenceSolver::Scipy
            | ExternalNonlinearValidationReferenceSolver::Ipopt
            | ExternalNonlinearValidationReferenceSolver::Bonmin
            | ExternalNonlinearValidationReferenceSolver::Minotaur
            | ExternalNonlinearValidationReferenceSolver::Couenne
            | ExternalNonlinearValidationReferenceSolver::Symphony
            | ExternalNonlinearValidationReferenceSolver::Knitro
            | ExternalNonlinearValidationReferenceSolver::Mosek
            | ExternalNonlinearValidationReferenceSolver::Baron
            | ExternalNonlinearValidationReferenceSolver::Copt => {
                ExternalNonlinearValidationReferenceFamily::ScipyBridge
            }
            ExternalNonlinearValidationReferenceSolver::Casadi
            | ExternalNonlinearValidationReferenceSolver::Nlopt
            | ExternalNonlinearValidationReferenceSolver::NloptCli => {
                ExternalNonlinearValidationReferenceFamily::PackageBridge
            }
            ExternalNonlinearValidationReferenceSolver::Fallback => {
                ExternalNonlinearValidationReferenceFamily::Fallback
            }
        }
    }

    pub fn notes(self) -> &'static str {
        match self.family() {
            ExternalNonlinearValidationReferenceFamily::Auto => {
                "Prefer installed SciPy-backed validation, then use the bounded pattern-search fallback."
            }
            ExternalNonlinearValidationReferenceFamily::ScipyBridge => {
                "Registered NLP solver label routed through the local SciPy validation bridge when available, with deterministic fallback recovery."
            }
            ExternalNonlinearValidationReferenceFamily::PackageBridge => {
                "Package-specific bridge that checks the named Python package before falling back for smoke validation."
            }
            ExternalNonlinearValidationReferenceFamily::Fallback => {
                "Dependency-free bounded grid plus pattern-search reference for small NLP smoke models."
            }
        }
    }

    pub fn spec(self) -> ExternalNonlinearValidationReferenceSolverSpec {
        ExternalNonlinearValidationReferenceSolverSpec {
            solver: self,
            id: self.as_arg(),
            display_name: self.display_name(),
            family: self.family(),
            notes: self.notes(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearValidationReferenceFamily {
    Auto,
    ScipyBridge,
    PackageBridge,
    Fallback,
}

impl ExternalNonlinearValidationReferenceFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalNonlinearValidationReferenceFamily::Auto => "auto",
            ExternalNonlinearValidationReferenceFamily::ScipyBridge => "scipy-bridge",
            ExternalNonlinearValidationReferenceFamily::PackageBridge => "package-bridge",
            ExternalNonlinearValidationReferenceFamily::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalNonlinearValidationReferenceSolverSpec {
    pub solver: ExternalNonlinearValidationReferenceSolver,
    pub id: &'static str,
    pub display_name: &'static str,
    pub family: ExternalNonlinearValidationReferenceFamily,
    pub notes: &'static str,
}

pub fn external_nonlinear_validation_reference_solver_specs(
) -> Vec<ExternalNonlinearValidationReferenceSolverSpec> {
    ExternalNonlinearValidationReferenceSolver::all()
        .iter()
        .copied()
        .map(ExternalNonlinearValidationReferenceSolver::spec)
        .collect()
}

pub fn external_nonlinear_validation_reference_solver_manifest() -> Value {
    Value::Array(
        external_nonlinear_validation_reference_solver_specs()
            .into_iter()
            .map(|spec| {
                json!({
                    "id": spec.id,
                    "displayName": spec.display_name,
                    "family": spec.family.as_str(),
                    "notes": spec.notes,
                })
            })
            .collect(),
    )
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationReferenceOptions {
    pub solver: ExternalNonlinearValidationReferenceSolver,
}

impl Default for ExternalNonlinearValidationReferenceOptions {
    fn default() -> Self {
        ExternalNonlinearValidationReferenceOptions {
            solver: ExternalNonlinearValidationReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationVariable {
    pub name: String,
    pub lb: f64,
    pub ub: f64,
    pub start: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationConstraint {
    pub name: String,
    pub expr: String,
    pub sense: String,
    pub rhs: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationRequest {
    pub variables: Vec<ExternalNonlinearValidationVariable>,
    pub objective: String,
    pub constraints: Vec<ExternalNonlinearValidationConstraint>,
    pub sense: String,
}

impl ExternalNonlinearValidationRequest {
    pub fn to_json(&self) -> Value {
        json!({
            "kind": "nonlinear-validation",
            "variables": self.variables.iter().map(|variable| json!({
                "name": variable.name,
                "lb": variable.lb,
                "ub": variable.ub,
                "start": variable.start,
            })).collect::<Vec<_>>(),
            "objective": self.objective,
            "constraints": self.constraints.iter().map(|constraint| json!({
                "name": constraint.name,
                "expr": constraint.expr,
                "sense": constraint.sense,
                "rhs": constraint.rhs,
            })).collect::<Vec<_>>(),
            "sense": self.sense,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearValidationReferenceStatus {
    Optimal,
    Infeasible,
    Failed,
    NumericalError,
}

impl ExternalNonlinearValidationReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalNonlinearValidationReferenceStatus::Optimal => "optimal",
            ExternalNonlinearValidationReferenceStatus::Infeasible => "infeasible",
            ExternalNonlinearValidationReferenceStatus::Failed => "failed",
            ExternalNonlinearValidationReferenceStatus::NumericalError => "numerical-error",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationReferenceSolution {
    pub status: ExternalNonlinearValidationReferenceStatus,
    pub solver: String,
    pub x: Vec<f64>,
    pub objective: Option<f64>,
    pub message: String,
    pub iterations: Option<u64>,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct NonlinearValidationReferencePayload {
    status: String,
    solver: Option<String>,
    x: Option<Vec<f64>>,
    objective: Option<f64>,
    message: Option<String>,
    iterations: Option<u64>,
}

fn status_from_str(status: &str) -> ExternalNonlinearValidationReferenceStatus {
    match status {
        "optimal" => ExternalNonlinearValidationReferenceStatus::Optimal,
        "infeasible" => ExternalNonlinearValidationReferenceStatus::Infeasible,
        "failed" => ExternalNonlinearValidationReferenceStatus::Failed,
        _ => ExternalNonlinearValidationReferenceStatus::NumericalError,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalNonlinearValidationReferenceSolution {
    ExternalNonlinearValidationReferenceSolution {
        status: ExternalNonlinearValidationReferenceStatus::NumericalError,
        solver: "external-nonlinear-validation-reference".to_string(),
        x: Vec::new(),
        objective: None,
        message: message.into(),
        iterations: None,
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts")
        .join("nonlinear_validation_reference.py")
}

pub fn solve_nonlinear_validation_json_with_external_reference(
    payload: Value,
    opts: &ExternalNonlinearValidationReferenceOptions,
) -> ExternalNonlinearValidationReferenceSolution {
    let started = Instant::now();
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut child = match Command::new(&python)
        .arg(reference_script())
        .arg("--solver")
        .arg(opts.solver.as_arg())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return numerical_error(
                format!("failed to start nonlinear_validation_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write nonlinear_validation_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for nonlinear_validation_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<NonlinearValidationReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalNonlinearValidationReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-nonlinear-validation-reference".to_string()),
            x: parsed.x.unwrap_or_default(),
            objective: parsed.objective,
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            }),
            iterations: parsed.iterations,
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse nonlinear_validation_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_nonlinear_validation_with_external_reference(
    request: &ExternalNonlinearValidationRequest,
    opts: &ExternalNonlinearValidationReferenceOptions,
) -> ExternalNonlinearValidationReferenceSolution {
    solve_nonlinear_validation_json_with_external_reference(request.to_json(), opts)
}

#[cfg(test)]
mod tests {
    use crate::des::general::external_nonlinear_validation_reference::{
        external_nonlinear_validation_reference_solver_manifest,
        external_nonlinear_validation_reference_solver_specs,
        solve_nonlinear_validation_with_external_reference, ExternalNonlinearValidationConstraint,
        ExternalNonlinearValidationReferenceFamily, ExternalNonlinearValidationReferenceOptions,
        ExternalNonlinearValidationReferenceSolver, ExternalNonlinearValidationReferenceStatus,
        ExternalNonlinearValidationRequest, ExternalNonlinearValidationVariable,
    };

    #[test]
    fn solver_manifest_covers_registered_nonlinear_validation_tools() {
        let specs = external_nonlinear_validation_reference_solver_specs();
        assert_eq!(specs.len(), 15);
        assert_eq!(
            specs
                .iter()
                .filter(
                    |spec| spec.family == ExternalNonlinearValidationReferenceFamily::ScipyBridge
                )
                .count(),
            10
        );
        assert_eq!(
            specs
                .iter()
                .filter(
                    |spec| spec.family == ExternalNonlinearValidationReferenceFamily::PackageBridge
                )
                .count(),
            3
        );
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalNonlinearValidationReferenceSolver::Ipopt
                && spec.id == "ipopt"
                && spec.display_name == "Ipopt"
        }));
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalNonlinearValidationReferenceSolver::Casadi && spec.id == "casadi"
        }));

        let manifest = external_nonlinear_validation_reference_solver_manifest();
        let items = manifest.as_array().expect("manifest array");
        assert_eq!(items.len(), 15);
        assert!(items.iter().any(|item| {
            item.get("id").and_then(|value| value.as_str()) == Some("knitro")
                && item.get("family").and_then(|value| value.as_str()) == Some("scipy-bridge")
        }));
    }

    #[test]
    fn fallback_bridge_solves_small_expression_model() {
        let request = ExternalNonlinearValidationRequest {
            variables: vec![
                ExternalNonlinearValidationVariable {
                    name: "x".to_string(),
                    lb: 0.0,
                    ub: 3.0,
                    start: Some(0.2),
                },
                ExternalNonlinearValidationVariable {
                    name: "y".to_string(),
                    lb: 0.0,
                    ub: 3.0,
                    start: Some(0.2),
                },
            ],
            objective: "(x - 1)**2 + (y - 2)**2".to_string(),
            constraints: vec![ExternalNonlinearValidationConstraint {
                name: "demand".to_string(),
                expr: "x + y".to_string(),
                sense: ">=".to_string(),
                rhs: 1.0,
            }],
            sense: "min".to_string(),
        };
        let result = solve_nonlinear_validation_with_external_reference(
            &request,
            &ExternalNonlinearValidationReferenceOptions {
                solver: ExternalNonlinearValidationReferenceSolver::Fallback,
            },
        );
        assert_eq!(
            result.status,
            ExternalNonlinearValidationReferenceStatus::Optimal
        );
        assert_eq!(result.x.len(), 2);
        assert!(result.objective.is_some_and(|objective| objective <= 1e-6));
    }
}
