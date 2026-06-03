//! Rust-facing bridge for CP-SAT and constraint-programming reference checks.
//!
//! The native Rust fallback accepts the crate's compact CP-SAT JSON model and
//! enumerates small finite-domain validation models without a Python dependency.
//! `scripts/cp_sat_reference.py` remains available for OR-Tools CP-SAT and
//! legacy Python fallback checks. Broader CP ecosystems such as Choco, JaCoP,
//! CPMpy, Conjure, clingo, SAT4J, and Open-WBO use the
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
    RustEnumeration,
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
            ExternalCpSatReferenceSolver::RustEnumeration,
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
            ExternalCpSatReferenceSolver::RustEnumeration => "rust-enumeration",
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
            ExternalCpSatReferenceSolver::RustEnumeration => "Rust exact CP enumeration",
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
            ExternalCpSatReferenceSolver::RustEnumeration
            | ExternalCpSatReferenceSolver::PythonEnumeration => {
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
            ExternalCpSatReferenceSolver::RustEnumeration => Some("rust-enumeration"),
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
        match self {
            ExternalCpSatReferenceSolver::Auto => {
                "Use the configured direct CP-SAT bridge; rust-enumeration is the dependency-free same-input fallback for small models."
            }
            ExternalCpSatReferenceSolver::OrToolsCpSat => {
                "Direct same-input OR-Tools CP-SAT bridge through scripts/cp_sat_reference.py."
            }
            ExternalCpSatReferenceSolver::RustEnumeration => {
                "Native Rust exact enumeration for small finite-domain CP-SAT JSON models."
            }
            ExternalCpSatReferenceSolver::PythonEnumeration => {
                "Legacy exact enumeration bridge through scripts/cp_sat_reference.py."
            }
            _ => {
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

fn cp_sat_error_run(
    solver: ExternalCpSatReferenceSolver,
    status: ExternalCpSatReferenceStatus,
    message: impl Into<String>,
    started: Instant,
) -> ExternalCpSatReferenceRun {
    let message = message.into();
    ExternalCpSatReferenceRun {
        solver,
        backend: "rust:cp-enumeration".to_string(),
        status,
        assignment: Vec::new(),
        objective: None,
        nodes: Some(0),
        raw: json!({
            "status": status.as_str(),
            "solver": "rust:cp-enumeration",
            "message": message,
        }),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        message,
    }
}

fn cp_sat_array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or non-array `{key}`"))
}

fn cp_sat_i64(value: &Value, key: &str) -> Result<i64, String> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing or non-integer `{key}`"))
}

fn cp_sat_usize(value: &Value, key: &str, len: usize) -> Result<usize, String> {
    let raw = cp_sat_i64(value, key)?;
    if raw < 0 || raw as usize >= len {
        return Err(format!("`{key}` index {raw} is outside 0..{len}"));
    }
    Ok(raw as usize)
}

fn cp_sat_domains(model: &Value) -> Result<Vec<Vec<i64>>, String> {
    cp_sat_array(model, "variables")?
        .iter()
        .enumerate()
        .map(|(idx, variable)| {
            let domain = cp_sat_array(variable, "domain")?
                .iter()
                .map(|value| {
                    value
                        .as_i64()
                        .ok_or_else(|| format!("variable {idx} has a non-integer domain value"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if domain.is_empty() {
                return Err(format!("variable {idx} has an empty domain"));
            }
            Ok(domain)
        })
        .collect()
}

fn cp_sat_literal_truth(assignment: &[i64], lit: &Value) -> Result<bool, String> {
    let var = cp_sat_usize(lit, "var", assignment.len())?;
    let positive = lit.get("positive").and_then(Value::as_bool).unwrap_or(true);
    Ok(if positive {
        assignment[var] == 1
    } else {
        assignment[var] == 0
    })
}

fn cp_sat_enforcement_active(assignment: &[i64], constraint: &Value) -> Result<bool, String> {
    let Some(enforcement) = constraint.get("enforcement") else {
        return Ok(true);
    };
    let literals = enforcement
        .as_array()
        .ok_or_else(|| "`enforcement` must be an array".to_string())?;
    for lit in literals {
        if !cp_sat_literal_truth(assignment, lit)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn cp_sat_linear_value(assignment: &[i64], terms: &[Value]) -> Result<i64, String> {
    let mut total = 0i64;
    for term in terms {
        let var = cp_sat_usize(term, "var", assignment.len())?;
        let coeff = cp_sat_i64(term, "coeff")?;
        total = total
            .checked_add(
                coeff
                    .checked_mul(assignment[var])
                    .ok_or_else(|| "linear term overflow".to_string())?,
            )
            .ok_or_else(|| "linear expression overflow".to_string())?;
    }
    Ok(total)
}

fn cp_sat_linear_constraint_ok(assignment: &[i64], constraint: &Value) -> Result<bool, String> {
    let value = cp_sat_linear_value(assignment, cp_sat_array(constraint, "terms")?)?;
    let rhs = cp_sat_i64(constraint, "rhs")?;
    match constraint
        .get("sense")
        .and_then(Value::as_str)
        .unwrap_or("eq")
    {
        "le" => Ok(value <= rhs),
        "ge" => Ok(value >= rhs),
        "eq" => Ok(value == rhs),
        sense => Err(format!("unsupported linear sense `{sense}`")),
    }
}

fn cp_sat_linear_domain_constraint_ok(
    assignment: &[i64],
    constraint: &Value,
) -> Result<bool, String> {
    let value = cp_sat_linear_value(assignment, cp_sat_array(constraint, "terms")?)?;
    for interval in cp_sat_array(constraint, "intervals")? {
        let lb = cp_sat_i64(interval, "lb")?;
        let ub = cp_sat_i64(interval, "ub")?;
        if lb <= value && value <= ub {
            return Ok(true);
        }
    }
    Ok(false)
}

fn cp_sat_bool_clause_ok(
    assignment: &[i64],
    constraint: &Value,
    mode: &str,
) -> Result<bool, String> {
    let literals = cp_sat_array(constraint, "literals")?;
    let true_count = literals.iter().try_fold(0usize, |count, lit| {
        Ok::<_, String>(count + usize::from(cp_sat_literal_truth(assignment, lit)?))
    })?;
    match mode {
        "or" | "at_least_one" => Ok(true_count >= 1),
        "and" => Ok(true_count == literals.len()),
        "xor" => Ok(true_count % 2 == 1),
        "at_most_one" => Ok(true_count <= 1),
        "exactly_one" => Ok(true_count == 1),
        _ => Err(format!("unsupported Boolean mode `{mode}`")),
    }
}

fn cp_sat_tuple_constraint_ok(
    assignment: &[i64],
    constraint: &Value,
    allowed: bool,
) -> Result<bool, String> {
    let vars = cp_sat_array(constraint, "vars")?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| "tuple variable index must be an integer".to_string())
                .and_then(|idx| {
                    if idx < 0 || idx as usize >= assignment.len() {
                        Err(format!("tuple variable index {idx} out of range"))
                    } else {
                        Ok(idx as usize)
                    }
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selected = vars.iter().map(|var| assignment[*var]).collect::<Vec<_>>();
    let mut listed = false;
    for tuple in cp_sat_array(constraint, "tuples")? {
        let tuple_values = tuple
            .as_array()
            .ok_or_else(|| "tuple entry must be an array".to_string())?
            .iter()
            .map(|value| {
                value
                    .as_i64()
                    .ok_or_else(|| "tuple value must be an integer".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if tuple_values == selected {
            listed = true;
            break;
        }
    }
    Ok(listed == allowed)
}

fn cp_sat_constraint_ok(assignment: &[i64], constraint: &Value) -> Result<bool, String> {
    let kind = constraint
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| "constraint missing string `kind`".to_string())?;
    let active = match kind {
        "enforced_linear"
        | "enforced_linear_domain"
        | "enforced_bool_or"
        | "enforced_at_least_one"
        | "enforced_bool_and"
        | "enforced_bool_xor"
        | "enforced_at_most_one"
        | "enforced_exactly_one"
        | "enforced_allowed_assignments"
        | "enforced_forbidden_assignments" => cp_sat_enforcement_active(assignment, constraint)?,
        _ => true,
    };
    if !active {
        return Ok(true);
    }
    match kind {
        "linear" | "enforced_linear" => cp_sat_linear_constraint_ok(assignment, constraint),
        "linear_domain" | "enforced_linear_domain" => {
            cp_sat_linear_domain_constraint_ok(assignment, constraint)
        }
        "all_different" => {
            let mut seen = std::collections::BTreeSet::new();
            for var in cp_sat_array(constraint, "vars")? {
                let idx = var
                    .as_i64()
                    .ok_or_else(|| "all_different variable must be an integer".to_string())?;
                if idx < 0 || idx as usize >= assignment.len() {
                    return Err(format!("all_different variable index {idx} out of range"));
                }
                if !seen.insert(assignment[idx as usize]) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        "bool_or" | "enforced_bool_or" => cp_sat_bool_clause_ok(assignment, constraint, "or"),
        "bool_and" | "enforced_bool_and" => cp_sat_bool_clause_ok(assignment, constraint, "and"),
        "bool_xor" | "enforced_bool_xor" => cp_sat_bool_clause_ok(assignment, constraint, "xor"),
        "at_most_one" | "enforced_at_most_one" => {
            cp_sat_bool_clause_ok(assignment, constraint, "at_most_one")
        }
        "at_least_one" | "enforced_at_least_one" => {
            cp_sat_bool_clause_ok(assignment, constraint, "at_least_one")
        }
        "exactly_one" | "enforced_exactly_one" => {
            cp_sat_bool_clause_ok(assignment, constraint, "exactly_one")
        }
        "implication" => {
            let antecedent = cp_sat_literal_truth(
                assignment,
                constraint
                    .get("antecedent")
                    .ok_or_else(|| "implication missing antecedent".to_string())?,
            )?;
            let consequent = cp_sat_literal_truth(
                assignment,
                constraint
                    .get("consequent")
                    .ok_or_else(|| "implication missing consequent".to_string())?,
            )?;
            Ok(!antecedent || consequent)
        }
        "allowed_assignments" | "enforced_allowed_assignments" => {
            cp_sat_tuple_constraint_ok(assignment, constraint, true)
        }
        "forbidden_assignments" | "enforced_forbidden_assignments" => {
            cp_sat_tuple_constraint_ok(assignment, constraint, false)
        }
        other => Err(format!(
            "rust-enumeration does not support constraint kind `{other}`"
        )),
    }
}

fn cp_sat_assignment_feasible(model: &Value, assignment: &[i64]) -> Result<bool, String> {
    for constraint in model
        .get("constraints")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if !cp_sat_constraint_ok(assignment, constraint)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn cp_sat_objective_value(model: &Value, assignment: &[i64]) -> Result<Option<i64>, String> {
    let Some(objective) = model.get("objective") else {
        return Ok(None);
    };
    Ok(Some(cp_sat_linear_value(
        assignment,
        cp_sat_array(objective, "terms")?,
    )?))
}

fn cp_sat_better_objective(model: &Value, candidate: i64, incumbent: i64) -> bool {
    let minimize = model
        .get("objective")
        .and_then(|objective| objective.get("sense"))
        .and_then(Value::as_str)
        .unwrap_or("min")
        != "max";
    if minimize {
        candidate < incumbent
    } else {
        candidate > incumbent
    }
}

fn solve_cp_sat_json_with_rust_enumeration(
    model: &Value,
    options: &ExternalCpSatReferenceOptions,
    started: Instant,
) -> ExternalCpSatReferenceRun {
    if options.assumption_core {
        return cp_sat_error_run(
            options.solver,
            ExternalCpSatReferenceStatus::Unsupported,
            "rust-enumeration does not compute assumption cores yet",
            started,
        );
    }

    let domains = match cp_sat_domains(model) {
        Ok(domains) => domains,
        Err(message) => {
            return cp_sat_error_run(
                options.solver,
                ExternalCpSatReferenceStatus::Invalid,
                message,
                started,
            )
        }
    };
    let mut assignment = vec![0; domains.len()];
    let mut nodes = 0u64;
    let mut best_assignment = None::<Vec<i64>>;
    let mut best_objective = None::<i64>;
    let mut solutions = Vec::<Value>::new();
    let solution_limit = options.enumerate_solutions.unwrap_or(usize::MAX).max(1);

    fn dfs(
        model: &Value,
        domains: &[Vec<i64>],
        assignment: &mut [i64],
        var_idx: usize,
        nodes: &mut u64,
        best_assignment: &mut Option<Vec<i64>>,
        best_objective: &mut Option<i64>,
        solutions: &mut Vec<Value>,
        solution_limit: usize,
    ) -> Result<(), String> {
        if var_idx == domains.len() {
            *nodes = nodes.saturating_add(1);
            if !cp_sat_assignment_feasible(model, assignment)? {
                return Ok(());
            }
            let objective = cp_sat_objective_value(model, assignment)?;
            let is_better = match (*best_objective, objective) {
                (Some(incumbent), Some(candidate)) => {
                    cp_sat_better_objective(model, candidate, incumbent)
                }
                (None, Some(_)) | (None, None) => best_assignment.is_none(),
                (Some(_), None) => false,
            };
            if is_better {
                *best_assignment = Some(assignment.to_vec());
                *best_objective = objective;
            }
            if solutions.len() < solution_limit {
                solutions.push(json!({
                    "assignment": assignment,
                    "objective": objective.map(|value| value as f64),
                }));
            }
            return Ok(());
        }
        for value in &domains[var_idx] {
            assignment[var_idx] = *value;
            dfs(
                model,
                domains,
                assignment,
                var_idx + 1,
                nodes,
                best_assignment,
                best_objective,
                solutions,
                solution_limit,
            )?;
        }
        Ok(())
    }

    if let Err(message) = dfs(
        model,
        &domains,
        &mut assignment,
        0,
        &mut nodes,
        &mut best_assignment,
        &mut best_objective,
        &mut solutions,
        solution_limit,
    ) {
        let status = if message.contains("does not support") {
            ExternalCpSatReferenceStatus::Unsupported
        } else {
            ExternalCpSatReferenceStatus::Invalid
        };
        return cp_sat_error_run(options.solver, status, message, started);
    }

    let Some(best) = best_assignment else {
        let raw = json!({
            "status": "infeasible",
            "solver": "rust:cp-enumeration",
            "assignment": [],
            "objective": null,
            "nodes": nodes,
        });
        return ExternalCpSatReferenceRun {
            solver: options.solver,
            backend: "rust:cp-enumeration".to_string(),
            status: ExternalCpSatReferenceStatus::Infeasible,
            assignment: Vec::new(),
            objective: None,
            nodes: Some(nodes),
            raw,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            message: String::new(),
        };
    };

    if model.get("objective").is_some() {
        let reverse = model
            .get("objective")
            .and_then(|objective| objective.get("sense"))
            .and_then(Value::as_str)
            .unwrap_or("min")
            == "max";
        solutions.sort_by(|a, b| {
            let lhs = a.get("objective").and_then(Value::as_f64).unwrap_or(0.0);
            let rhs = b.get("objective").and_then(Value::as_f64).unwrap_or(0.0);
            if reverse {
                rhs.partial_cmp(&lhs).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                lhs.partial_cmp(&rhs).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        solutions.truncate(solution_limit);
    }

    let objective = best_objective.map(|value| value as f64);
    let status = if model.get("objective").is_some() {
        ExternalCpSatReferenceStatus::Optimal
    } else {
        ExternalCpSatReferenceStatus::Feasible
    };
    let raw = json!({
        "status": status.as_str(),
        "solver": "rust:cp-enumeration",
        "assignment": best,
        "objective": objective,
        "nodes": nodes,
        "solutions": if options.enumerate_solutions.is_some() { Value::Array(solutions) } else { Value::Null },
        "message": "native Rust exact enumeration fallback",
    });
    ExternalCpSatReferenceRun {
        solver: options.solver,
        backend: "rust:cp-enumeration".to_string(),
        status,
        assignment: raw
            .get("assignment")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_default(),
        objective,
        nodes: Some(nodes),
        raw,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        message: "native Rust exact enumeration fallback".to_string(),
    }
}

pub fn solve_cp_sat_json_with_external_reference(
    model: &Value,
    options: &ExternalCpSatReferenceOptions,
) -> ExternalCpSatReferenceRun {
    let started = Instant::now();
    if options.solver == ExternalCpSatReferenceSolver::RustEnumeration {
        return solve_cp_sat_json_with_rust_enumeration(model, options, started);
    }

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

        assert_eq!(specs.len(), 19);
        assert_eq!(direct, 4);
        assert_eq!(ecosystem, 15);
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalCpSatReferenceSolver::RustEnumeration
                && spec.family == ExternalCpSatReferenceFamily::Fallback
                && spec.supports_cp_sat_json
        }));
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalCpSatReferenceSolver::ChocoSolver
                && spec.family == ExternalCpSatReferenceFamily::EcosystemReference
        }));
        assert!(ExternalCpSatReferenceSolver::RustEnumeration.supports_cp_sat_json());
        assert!(ExternalCpSatReferenceSolver::PythonEnumeration.supports_cp_sat_json());
        assert!(!ExternalCpSatReferenceSolver::ChocoSolver.supports_cp_sat_json());
    }

    #[test]
    fn cp_sat_rust_enumeration_solves_same_input_json() {
        let run = solve_cp_sat_json_with_external_reference(
            &tiny_cp_sat_model(),
            &ExternalCpSatReferenceOptions {
                solver: ExternalCpSatReferenceSolver::RustEnumeration,
                ..Default::default()
            },
        );

        assert_eq!(run.status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(run.assignment, vec![1, 0]);
        assert_eq!(run.objective, Some(1.0));
        assert_eq!(run.backend, "rust:cp-enumeration");
    }

    #[test]
    fn cp_sat_rust_enumeration_handles_bool_and_all_different() {
        let model = json!({
            "variables": [
                {"name": "a", "domain": [0, 1]},
                {"name": "b", "domain": [0, 1]},
                {"name": "c", "domain": [0, 1]}
            ],
            "constraints": [
                {
                    "kind": "exactly_one",
                    "literals": [
                        {"var": 0, "positive": true},
                        {"var": 1, "positive": true}
                    ]
                },
                {"kind": "implication", "antecedent": {"var": 0}, "consequent": {"var": 2}},
                {"kind": "linear_domain", "terms": [{"var": 2, "coeff": 1}], "intervals": [{"lb": 1, "ub": 1}]}
            ],
            "objective": {
                "sense": "max",
                "terms": [
                    {"var": 0, "coeff": 2},
                    {"var": 1, "coeff": 1}
                ]
            }
        });
        let run = solve_cp_sat_json_with_external_reference(
            &model,
            &ExternalCpSatReferenceOptions {
                solver: ExternalCpSatReferenceSolver::RustEnumeration,
                ..Default::default()
            },
        );

        assert_eq!(run.status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(run.assignment, vec![1, 0, 1]);
        assert_eq!(run.objective, Some(2.0));
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
