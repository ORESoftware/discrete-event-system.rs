//! Rust-facing bridge for CP-SAT and constraint-programming reference checks.
//!
//! `scripts/cp_sat_reference.py` accepts the crate's compact CP-SAT JSON model
//! and can directly use OR-Tools CP-SAT when installed, otherwise falling back to
//! exact enumeration for small validation models. Broader CP ecosystems such as
//! Choco, JaCoP, CPMpy, Conjure, clingo, SAT4J, and Open-WBO use the
//! `optimization_ecosystem_reference.py` smoke-model contract instead; this
//! module exposes both paths without pretending they share one model format.

use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::external_optimization_tools::{
    run_external_optimization_ecosystem_reference, ExternalOptimizationAdapterStatus,
    ExternalOptimizationTool,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalCpSatReferenceSolver {
    Auto,
    OrToolsCpSat,
    PythonEnumeration,
    ChocoSolver,
    JaCoP,
    IbmCpOptimizer,
    OrToolsJava,
    OrToolsPython,
    Cpmpy,
    PyCsp3,
    Conjure,
    SavileRow,
    Picat,
    Clingo,
    Clingcon,
    Sat4j,
    PySat,
    OpenWbo,
}

impl ExternalCpSatReferenceSolver {
    pub fn all() -> &'static [ExternalCpSatReferenceSolver] {
        &[
            ExternalCpSatReferenceSolver::Auto,
            ExternalCpSatReferenceSolver::OrToolsCpSat,
            ExternalCpSatReferenceSolver::PythonEnumeration,
            ExternalCpSatReferenceSolver::ChocoSolver,
            ExternalCpSatReferenceSolver::JaCoP,
            ExternalCpSatReferenceSolver::IbmCpOptimizer,
            ExternalCpSatReferenceSolver::OrToolsJava,
            ExternalCpSatReferenceSolver::OrToolsPython,
            ExternalCpSatReferenceSolver::Cpmpy,
            ExternalCpSatReferenceSolver::PyCsp3,
            ExternalCpSatReferenceSolver::Conjure,
            ExternalCpSatReferenceSolver::SavileRow,
            ExternalCpSatReferenceSolver::Picat,
            ExternalCpSatReferenceSolver::Clingo,
            ExternalCpSatReferenceSolver::Clingcon,
            ExternalCpSatReferenceSolver::Sat4j,
            ExternalCpSatReferenceSolver::PySat,
            ExternalCpSatReferenceSolver::OpenWbo,
        ]
    }

    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalCpSatReferenceSolver::Auto => "auto",
            ExternalCpSatReferenceSolver::OrToolsCpSat => "ortools-cp-sat",
            ExternalCpSatReferenceSolver::PythonEnumeration => "python-enumeration",
            ExternalCpSatReferenceSolver::ChocoSolver => "choco-solver",
            ExternalCpSatReferenceSolver::JaCoP => "jacop",
            ExternalCpSatReferenceSolver::IbmCpOptimizer => "ibm-cp-optimizer",
            ExternalCpSatReferenceSolver::OrToolsJava => "ortools-java",
            ExternalCpSatReferenceSolver::OrToolsPython => "ortools-python",
            ExternalCpSatReferenceSolver::Cpmpy => "cpmpy",
            ExternalCpSatReferenceSolver::PyCsp3 => "pycsp3",
            ExternalCpSatReferenceSolver::Conjure => "conjure",
            ExternalCpSatReferenceSolver::SavileRow => "savile-row",
            ExternalCpSatReferenceSolver::Picat => "picat",
            ExternalCpSatReferenceSolver::Clingo => "clingo",
            ExternalCpSatReferenceSolver::Clingcon => "clingcon",
            ExternalCpSatReferenceSolver::Sat4j => "sat4j",
            ExternalCpSatReferenceSolver::PySat => "pysat",
            ExternalCpSatReferenceSolver::OpenWbo => "open-wbo",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ExternalCpSatReferenceSolver::Auto => "Auto",
            ExternalCpSatReferenceSolver::OrToolsCpSat => "Google OR-Tools CP-SAT",
            ExternalCpSatReferenceSolver::PythonEnumeration => {
                "Dependency-free exact CP enumeration"
            }
            ExternalCpSatReferenceSolver::ChocoSolver => "Choco Solver",
            ExternalCpSatReferenceSolver::JaCoP => "JaCoP",
            ExternalCpSatReferenceSolver::IbmCpOptimizer => "IBM ILOG CP Optimizer",
            ExternalCpSatReferenceSolver::OrToolsJava => "OR-Tools Java",
            ExternalCpSatReferenceSolver::OrToolsPython => "OR-Tools Python",
            ExternalCpSatReferenceSolver::Cpmpy => "CPMpy",
            ExternalCpSatReferenceSolver::PyCsp3 => "PyCSP3",
            ExternalCpSatReferenceSolver::Conjure => "Conjure",
            ExternalCpSatReferenceSolver::SavileRow => "Savile Row",
            ExternalCpSatReferenceSolver::Picat => "Picat",
            ExternalCpSatReferenceSolver::Clingo => "clingo",
            ExternalCpSatReferenceSolver::Clingcon => "clingcon",
            ExternalCpSatReferenceSolver::Sat4j => "SAT4J",
            ExternalCpSatReferenceSolver::PySat => "PySAT",
            ExternalCpSatReferenceSolver::OpenWbo => "Open-WBO",
        }
    }

    pub fn family(self) -> ExternalCpSatReferenceFamily {
        match self {
            ExternalCpSatReferenceSolver::Auto => ExternalCpSatReferenceFamily::Auto,
            ExternalCpSatReferenceSolver::OrToolsCpSat => ExternalCpSatReferenceFamily::CpSatScript,
            ExternalCpSatReferenceSolver::PythonEnumeration => {
                ExternalCpSatReferenceFamily::Fallback
            }
            ExternalCpSatReferenceSolver::ChocoSolver
            | ExternalCpSatReferenceSolver::JaCoP
            | ExternalCpSatReferenceSolver::IbmCpOptimizer
            | ExternalCpSatReferenceSolver::OrToolsJava
            | ExternalCpSatReferenceSolver::OrToolsPython
            | ExternalCpSatReferenceSolver::Cpmpy
            | ExternalCpSatReferenceSolver::PyCsp3
            | ExternalCpSatReferenceSolver::Conjure
            | ExternalCpSatReferenceSolver::SavileRow
            | ExternalCpSatReferenceSolver::Picat
            | ExternalCpSatReferenceSolver::Clingo
            | ExternalCpSatReferenceSolver::Clingcon
            | ExternalCpSatReferenceSolver::Sat4j
            | ExternalCpSatReferenceSolver::PySat
            | ExternalCpSatReferenceSolver::OpenWbo => {
                ExternalCpSatReferenceFamily::EcosystemReference
            }
        }
    }

    pub fn direct_cp_sat_json_solver_arg(self) -> Option<&'static str> {
        match self {
            ExternalCpSatReferenceSolver::Auto => Some("auto"),
            ExternalCpSatReferenceSolver::OrToolsCpSat => Some("ortools-cp-sat"),
            ExternalCpSatReferenceSolver::PythonEnumeration => Some("fallback"),
            _ => None,
        }
    }

    pub fn ecosystem_tool(self) -> Option<ExternalOptimizationTool> {
        match self {
            ExternalCpSatReferenceSolver::ChocoSolver => {
                Some(ExternalOptimizationTool::ChocoSolver)
            }
            ExternalCpSatReferenceSolver::JaCoP => Some(ExternalOptimizationTool::Jacop),
            ExternalCpSatReferenceSolver::IbmCpOptimizer => {
                Some(ExternalOptimizationTool::IbmCpOptimizer)
            }
            ExternalCpSatReferenceSolver::OrToolsJava => {
                Some(ExternalOptimizationTool::OrToolsJava)
            }
            ExternalCpSatReferenceSolver::OrToolsPython => {
                Some(ExternalOptimizationTool::OrToolsPython)
            }
            ExternalCpSatReferenceSolver::Cpmpy => Some(ExternalOptimizationTool::Cpmpy),
            ExternalCpSatReferenceSolver::PyCsp3 => Some(ExternalOptimizationTool::PyCsp3),
            ExternalCpSatReferenceSolver::Conjure => Some(ExternalOptimizationTool::Conjure),
            ExternalCpSatReferenceSolver::SavileRow => Some(ExternalOptimizationTool::SavileRow),
            ExternalCpSatReferenceSolver::Picat => Some(ExternalOptimizationTool::Picat),
            ExternalCpSatReferenceSolver::Clingo => Some(ExternalOptimizationTool::Clingo),
            ExternalCpSatReferenceSolver::Clingcon => Some(ExternalOptimizationTool::Clingcon),
            ExternalCpSatReferenceSolver::Sat4j => Some(ExternalOptimizationTool::Sat4j),
            ExternalCpSatReferenceSolver::PySat => Some(ExternalOptimizationTool::PySat),
            ExternalCpSatReferenceSolver::OpenWbo => Some(ExternalOptimizationTool::OpenWbo),
            _ => None,
        }
    }

    pub fn supports_cp_sat_json(self) -> bool {
        self.direct_cp_sat_json_solver_arg().is_some()
    }

    pub fn supports_ecosystem_cp_assignment(self) -> bool {
        self.ecosystem_tool().is_some()
    }

    pub fn notes(self) -> &'static str {
        match self.family() {
            ExternalCpSatReferenceFamily::Auto => {
                "Use OR-Tools CP-SAT when installed; otherwise use exact Python enumeration for small CP-SAT JSON models."
            }
            ExternalCpSatReferenceFamily::CpSatScript => {
                "Direct same-input bridge through scripts/cp_sat_reference.py."
            }
            ExternalCpSatReferenceFamily::Fallback => {
                "Dependency-free exact enumeration bridge for small finite-domain CP-SAT JSON models."
            }
            ExternalCpSatReferenceFamily::EcosystemReference => {
                "Ecosystem smoke bridge through scripts/optimization_ecosystem_reference.py; uses the ecosystem CP-assignment contract rather than the CP-SAT JSON model."
            }
        }
    }

    pub fn spec(self) -> ExternalCpSatReferenceSolverSpec {
        ExternalCpSatReferenceSolverSpec {
            solver: self,
            id: self.as_arg(),
            display_name: self.display_name(),
            family: self.family(),
            supports_cp_sat_json: self.supports_cp_sat_json(),
            supports_ecosystem_cp_assignment: self.supports_ecosystem_cp_assignment(),
            notes: self.notes(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalCpSatReferenceFamily {
    Auto,
    CpSatScript,
    EcosystemReference,
    Fallback,
}

impl ExternalCpSatReferenceFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalCpSatReferenceFamily::Auto => "auto",
            ExternalCpSatReferenceFamily::CpSatScript => "cp-sat-script",
            ExternalCpSatReferenceFamily::EcosystemReference => "ecosystem-reference",
            ExternalCpSatReferenceFamily::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalCpSatReferenceSolverSpec {
    pub solver: ExternalCpSatReferenceSolver,
    pub id: &'static str,
    pub display_name: &'static str,
    pub family: ExternalCpSatReferenceFamily,
    pub supports_cp_sat_json: bool,
    pub supports_ecosystem_cp_assignment: bool,
    pub notes: &'static str,
}

pub fn external_cp_sat_reference_solver_specs() -> Vec<ExternalCpSatReferenceSolverSpec> {
    ExternalCpSatReferenceSolver::all()
        .iter()
        .copied()
        .map(ExternalCpSatReferenceSolver::spec)
        .collect()
}

pub fn external_cp_sat_reference_solver_manifest() -> Value {
    Value::Array(
        external_cp_sat_reference_solver_specs()
            .into_iter()
            .map(|spec| {
                json!({
                    "id": spec.id,
                    "displayName": spec.display_name,
                    "family": spec.family.as_str(),
                    "supportsCpSatJson": spec.supports_cp_sat_json,
                    "supportsEcosystemCpAssignment": spec.supports_ecosystem_cp_assignment,
                    "notes": spec.notes,
                })
            })
            .collect(),
    )
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalCpSatReferenceOptions {
    pub solver: ExternalCpSatReferenceSolver,
    pub enumerate_solutions: Option<usize>,
    pub assumption_core: bool,
}

impl Default for ExternalCpSatReferenceOptions {
    fn default() -> Self {
        ExternalCpSatReferenceOptions {
            solver: ExternalCpSatReferenceSolver::Auto,
            enumerate_solutions: None,
            assumption_core: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalCpSatReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Exhausted,
    Unavailable,
    Invalid,
    Unsupported,
    Failed,
    Unknown,
}

impl ExternalCpSatReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalCpSatReferenceStatus::Optimal => "optimal",
            ExternalCpSatReferenceStatus::Feasible => "feasible",
            ExternalCpSatReferenceStatus::Infeasible => "infeasible",
            ExternalCpSatReferenceStatus::Exhausted => "exhausted",
            ExternalCpSatReferenceStatus::Unavailable => "unavailable",
            ExternalCpSatReferenceStatus::Invalid => "invalid",
            ExternalCpSatReferenceStatus::Unsupported => "unsupported",
            ExternalCpSatReferenceStatus::Failed => "failed",
            ExternalCpSatReferenceStatus::Unknown => "unknown",
        }
    }

    pub fn from_label(label: &str) -> Self {
        match label {
            "optimal" => ExternalCpSatReferenceStatus::Optimal,
            "feasible" => ExternalCpSatReferenceStatus::Feasible,
            "infeasible" => ExternalCpSatReferenceStatus::Infeasible,
            "exhausted" => ExternalCpSatReferenceStatus::Exhausted,
            "unavailable" => ExternalCpSatReferenceStatus::Unavailable,
            "invalid" => ExternalCpSatReferenceStatus::Invalid,
            "unsupported" => ExternalCpSatReferenceStatus::Unsupported,
            "failed" => ExternalCpSatReferenceStatus::Failed,
            _ => ExternalCpSatReferenceStatus::Unknown,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalCpSatReferenceRun {
    pub solver: ExternalCpSatReferenceSolver,
    pub backend: String,
    pub status: ExternalCpSatReferenceStatus,
    pub assignment: Vec<i64>,
    pub objective: Option<f64>,
    pub nodes: Option<u64>,
    pub raw: Value,
    pub elapsed_ms: f64,
    pub message: String,
}

#[derive(Debug, Deserialize)]
struct CpSatScriptOutput {
    status: String,
    #[serde(default)]
    solver: String,
    #[serde(default)]
    assignment: Vec<i64>,
    #[serde(default)]
    objective: Option<f64>,
    #[serde(default)]
    nodes: Option<u64>,
    #[serde(default)]
    message: String,
}

pub fn external_cp_sat_reference_script() -> PathBuf {
    let root = env::var_os("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("cp_sat_reference.py")
}

fn python_command() -> PathBuf {
    env::var_os("PYTHON_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"))
}

fn script_working_dir(script: &Path) -> Option<PathBuf> {
    script
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
}

pub fn solve_cp_sat_json_with_external_reference(
    model: &Value,
    options: &ExternalCpSatReferenceOptions,
) -> ExternalCpSatReferenceRun {
    let Some(solver_arg) = options.solver.direct_cp_sat_json_solver_arg() else {
        return ExternalCpSatReferenceRun {
            solver: options.solver,
            backend: options.solver.as_arg().to_string(),
            status: ExternalCpSatReferenceStatus::Unsupported,
            assignment: Vec::new(),
            objective: None,
            nodes: None,
            raw: json!({
                "status": "unsupported",
                "solver": options.solver.as_arg(),
                "message": "solver uses the ecosystem CP-assignment contract, not CP-SAT JSON",
            }),
            elapsed_ms: 0.0,
            message: "solver uses the ecosystem CP-assignment contract, not CP-SAT JSON"
                .to_string(),
        };
    };

    let started = Instant::now();
    let script = external_cp_sat_reference_script();
    let mut command = Command::new(python_command());
    if let Some(working_dir) = script_working_dir(&script) {
        command.current_dir(working_dir);
    }
    command.arg(script).arg("--solver").arg(solver_arg);
    if let Some(limit) = options.enumerate_solutions {
        command
            .arg("--enumerate-solutions")
            .arg(limit.max(1).to_string());
    }
    if options.assumption_core {
        command.arg("--assumption-core");
    }

    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            return ExternalCpSatReferenceRun {
                solver: options.solver,
                backend: solver_arg.to_string(),
                status: ExternalCpSatReferenceStatus::Unavailable,
                assignment: Vec::new(),
                objective: None,
                nodes: None,
                raw: json!({"status": "unavailable", "message": e.to_string()}),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: e.to_string(),
            }
        }
    };

    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(e) = stdin.write_all(model.to_string().as_bytes()) {
            return ExternalCpSatReferenceRun {
                solver: options.solver,
                backend: solver_arg.to_string(),
                status: ExternalCpSatReferenceStatus::Failed,
                assignment: Vec::new(),
                objective: None,
                nodes: None,
                raw: json!({"status": "failed", "message": e.to_string()}),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: e.to_string(),
            };
        }
    }

    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(e) => {
            return ExternalCpSatReferenceRun {
                solver: options.solver,
                backend: solver_arg.to_string(),
                status: ExternalCpSatReferenceStatus::Failed,
                assignment: Vec::new(),
                objective: None,
                nodes: None,
                raw: json!({"status": "failed", "message": e.to_string()}),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: e.to_string(),
            }
        }
    };

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let raw = match serde_json::from_str::<Value>(stdout.trim()) {
        Ok(raw) => raw,
        Err(e) => {
            return ExternalCpSatReferenceRun {
                solver: options.solver,
                backend: solver_arg.to_string(),
                status: ExternalCpSatReferenceStatus::Failed,
                assignment: Vec::new(),
                objective: None,
                nodes: None,
                raw: json!({
                    "status": "failed",
                    "stdout": stdout.trim(),
                    "stderr": stderr,
                    "message": e.to_string(),
                }),
                elapsed_ms,
                message: e.to_string(),
            }
        }
    };

    let parsed = serde_json::from_value::<CpSatScriptOutput>(raw.clone()).ok();
    let status = parsed
        .as_ref()
        .map(|parsed| ExternalCpSatReferenceStatus::from_label(parsed.status.as_str()))
        .unwrap_or(ExternalCpSatReferenceStatus::Unknown);
    let message = parsed
        .as_ref()
        .map(|parsed| parsed.message.clone())
        .filter(|message| !message.is_empty())
        .unwrap_or(stderr);

    ExternalCpSatReferenceRun {
        solver: options.solver,
        backend: parsed
            .as_ref()
            .map(|parsed| parsed.solver.clone())
            .filter(|solver| !solver.is_empty())
            .unwrap_or_else(|| solver_arg.to_string()),
        status,
        assignment: parsed
            .as_ref()
            .map(|parsed| parsed.assignment.clone())
            .unwrap_or_default(),
        objective: parsed.as_ref().and_then(|parsed| parsed.objective),
        nodes: parsed.as_ref().and_then(|parsed| parsed.nodes),
        raw,
        elapsed_ms,
        message,
    }
}

pub fn solve_cp_assignment_with_external_reference(
    payload: &Value,
    solver: ExternalCpSatReferenceSolver,
) -> ExternalCpSatReferenceRun {
    let Some(tool) = solver.ecosystem_tool() else {
        return ExternalCpSatReferenceRun {
            solver,
            backend: solver.as_arg().to_string(),
            status: ExternalCpSatReferenceStatus::Unsupported,
            assignment: Vec::new(),
            objective: None,
            nodes: None,
            raw: json!({
                "status": "unsupported",
                "solver": solver.as_arg(),
                "message": "solver uses the direct CP-SAT JSON bridge, not ecosystem CP assignment",
            }),
            elapsed_ms: 0.0,
            message: "solver uses the direct CP-SAT JSON bridge, not ecosystem CP assignment"
                .to_string(),
        };
    };

    let run = run_external_optimization_ecosystem_reference(payload, tool);
    let raw = run.output.clone().unwrap_or_else(|| {
        json!({
            "status": run.status.as_str(),
            "tool": tool.as_str(),
            "message": run.message,
        })
    });
    let status_label = raw
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_else(|| run.status.as_str());
    let status = match run.status {
        ExternalOptimizationAdapterStatus::Ok => {
            ExternalCpSatReferenceStatus::from_label(status_label)
        }
        ExternalOptimizationAdapterStatus::Unavailable => ExternalCpSatReferenceStatus::Unavailable,
        ExternalOptimizationAdapterStatus::Failed => ExternalCpSatReferenceStatus::Failed,
        ExternalOptimizationAdapterStatus::InvalidOutput => ExternalCpSatReferenceStatus::Invalid,
    };
    let assignment = raw
        .get("x")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_f64)
                .map(|value| value.round() as i64)
                .collect()
        })
        .unwrap_or_default();
    let objective = raw.get("objective").and_then(Value::as_f64);

    ExternalCpSatReferenceRun {
        solver,
        backend: raw
            .get("backend")
            .and_then(Value::as_str)
            .unwrap_or(tool.as_str())
            .to_string(),
        status,
        assignment,
        objective,
        nodes: None,
        raw,
        elapsed_ms: run.elapsed_ms,
        message: run.message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cp_sat_model() -> Value {
        json!({
            "variables": [
                {"name": "x", "domain": [0, 1]},
                {"name": "y", "domain": [0, 1]}
            ],
            "constraints": [
                {
                    "kind": "linear",
                    "terms": [
                        {"var": 0, "coeff": 1},
                        {"var": 1, "coeff": 1}
                    ],
                    "sense": "eq",
                    "rhs": 1
                }
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 0, "coeff": 1},
                    {"var": 1, "coeff": 2}
                ]
            }
        })
    }

    #[test]
    fn cp_sat_reference_manifest_splits_direct_and_ecosystem_contracts() {
        let specs = external_cp_sat_reference_solver_specs();
        let direct = specs
            .iter()
            .filter(|spec| spec.supports_cp_sat_json)
            .count();
        let ecosystem = specs
            .iter()
            .filter(|spec| spec.supports_ecosystem_cp_assignment)
            .count();

        assert_eq!(specs.len(), 18);
        assert_eq!(direct, 3);
        assert_eq!(ecosystem, 15);
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalCpSatReferenceSolver::ChocoSolver
                && spec.family == ExternalCpSatReferenceFamily::EcosystemReference
        }));
        assert!(ExternalCpSatReferenceSolver::PythonEnumeration.supports_cp_sat_json());
        assert!(!ExternalCpSatReferenceSolver::ChocoSolver.supports_cp_sat_json());
    }

    #[test]
    fn cp_sat_python_enumeration_bridge_solves_same_input_json() {
        let run = solve_cp_sat_json_with_external_reference(
            &tiny_cp_sat_model(),
            &ExternalCpSatReferenceOptions {
                solver: ExternalCpSatReferenceSolver::PythonEnumeration,
                ..Default::default()
            },
        );

        assert_eq!(run.status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(run.assignment, vec![1, 0]);
        assert_eq!(run.objective, Some(1.0));
        assert_eq!(run.backend, "python:cp-enumeration");
    }

    #[test]
    fn cp_ecosystem_assignment_bridge_runs_choco_reference_contract() {
        let payload = json!({
            "kind": "ecosystem-cp-assignment",
            "costs": [[9, 2, 7], [6, 4, 3], [5, 8, 1]],
            "all_different": true
        });
        let run = solve_cp_assignment_with_external_reference(
            &payload,
            ExternalCpSatReferenceSolver::ChocoSolver,
        );

        assert_eq!(run.status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(run.objective, Some(9.0));
        assert_eq!(run.assignment, vec![1, 0, 2]);
        assert_eq!(run.backend, "builtin:constraint-programming");
    }
}
