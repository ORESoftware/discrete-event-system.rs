//! Local command-line adapters for installed LP/MIP solvers.
//!
//! This module exposes a Rust-facing interface for solver executables that are
//! installed locally (for example through Homebrew) without vendoring any
//! external binaries into the repository. The solver-specific command lines and
//! solution parsers live in `scripts/linear_cli_reference.py`; this module owns
//! the library boundary: problem serialization, subprocess execution, typed
//! status mapping, and elapsed-time accounting.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Number, Value};

use crate::des::general::ip_mip_des::IPMIPProblem;
use crate::des::general::lp::LPProblem;

/// Linear model family to send to the external CLI bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliKind {
    Lp,
    Mip,
}

impl ExternalLinearCliKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliKind::Lp => "lp",
            ExternalLinearCliKind::Mip => "mip",
        }
    }
}

/// Solver executable family known to the local CLI bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliSolver {
    Highs,
    Glpk,
    Scip,
    Cbc,
    Clp,
    Gurobi,
    Cplex,
    Xpress,
    Lindo,
}

impl ExternalLinearCliSolver {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliSolver::Highs => "highs",
            ExternalLinearCliSolver::Glpk => "glpk",
            ExternalLinearCliSolver::Scip => "scip",
            ExternalLinearCliSolver::Cbc => "cbc",
            ExternalLinearCliSolver::Clp => "clp",
            ExternalLinearCliSolver::Gurobi => "gurobi",
            ExternalLinearCliSolver::Cplex => "cplex",
            ExternalLinearCliSolver::Xpress => "xpress",
            ExternalLinearCliSolver::Lindo => "lindo",
        }
    }

    /// Command aliases searched on `PATH` for this solver.
    pub fn command_aliases(self) -> &'static [&'static str] {
        match self {
            ExternalLinearCliSolver::Highs => &["highs"],
            ExternalLinearCliSolver::Glpk => &["glpsol"],
            ExternalLinearCliSolver::Scip => &["scip"],
            ExternalLinearCliSolver::Cbc => &["cbc"],
            ExternalLinearCliSolver::Clp => &["clp"],
            ExternalLinearCliSolver::Gurobi => &["gurobi_cl"],
            ExternalLinearCliSolver::Cplex => &["cplex"],
            ExternalLinearCliSolver::Xpress => &["optimizer", "xpress"],
            ExternalLinearCliSolver::Lindo => &["lindo", "lindoapi"],
        }
    }

    /// Environment variables that may point directly at the solver executable.
    ///
    /// These are intentionally local configuration only: users can install
    /// solvers anywhere on their machine and point the bridge at the command
    /// without committing binaries or absolute paths to version control.
    pub fn command_env_vars(self) -> &'static [&'static str] {
        match self {
            ExternalLinearCliSolver::Highs => &["HIGHS_CMD", "ORES_HIGHS_CMD"],
            ExternalLinearCliSolver::Glpk => &["GLPSOL_CMD", "GLPK_CMD", "ORES_GLPK_CMD"],
            ExternalLinearCliSolver::Scip => &["SCIP_CMD", "ORES_SCIP_CMD"],
            ExternalLinearCliSolver::Cbc => &["CBC_CMD", "ORES_CBC_CMD"],
            ExternalLinearCliSolver::Clp => &["CLP_CMD", "ORES_CLP_CMD"],
            ExternalLinearCliSolver::Gurobi => &["GUROBI_CL_CMD", "GUROBI_CMD", "ORES_GUROBI_CMD"],
            ExternalLinearCliSolver::Cplex => &["CPLEX_CMD", "ORES_CPLEX_CMD"],
            ExternalLinearCliSolver::Xpress => {
                &["XPRESS_CMD", "XPRESS_OPTIMIZER_CMD", "ORES_XPRESS_CMD"]
            }
            ExternalLinearCliSolver::Lindo => &["LINDO_CMD", "LINDOAPI_CMD", "ORES_LINDO_CMD"],
        }
    }

    /// Whether the checked-in bridge knows the non-interactive command and
    /// solution parser for this solver/model family.
    pub fn supports_kind(self, kind: ExternalLinearCliKind) -> bool {
        match kind {
            ExternalLinearCliKind::Lp => matches!(
                self,
                ExternalLinearCliSolver::Highs
                    | ExternalLinearCliSolver::Glpk
                    | ExternalLinearCliSolver::Scip
                    | ExternalLinearCliSolver::Cbc
                    | ExternalLinearCliSolver::Clp
                    | ExternalLinearCliSolver::Gurobi
                    | ExternalLinearCliSolver::Cplex
            ),
            ExternalLinearCliKind::Mip => matches!(
                self,
                ExternalLinearCliSolver::Highs
                    | ExternalLinearCliSolver::Glpk
                    | ExternalLinearCliSolver::Scip
                    | ExternalLinearCliSolver::Cbc
                    | ExternalLinearCliSolver::Gurobi
                    | ExternalLinearCliSolver::Cplex
            ),
        }
    }

    /// Installed open-source CLIs that can solve LPs through this bridge.
    pub fn open_source_lp() -> &'static [ExternalLinearCliSolver] {
        &[
            ExternalLinearCliSolver::Highs,
            ExternalLinearCliSolver::Glpk,
            ExternalLinearCliSolver::Scip,
            ExternalLinearCliSolver::Cbc,
            ExternalLinearCliSolver::Clp,
        ]
    }

    /// Installed open-source CLIs that can solve MIPs through this bridge.
    pub fn open_source_mip() -> &'static [ExternalLinearCliSolver] {
        &[
            ExternalLinearCliSolver::Highs,
            ExternalLinearCliSolver::Glpk,
            ExternalLinearCliSolver::Scip,
            ExternalLinearCliSolver::Cbc,
        ]
    }

    /// Optional commercial CLIs surfaced by the bridge when installed locally.
    pub fn optional_commercial_mip() -> &'static [ExternalLinearCliSolver] {
        &[
            ExternalLinearCliSolver::Gurobi,
            ExternalLinearCliSolver::Cplex,
            ExternalLinearCliSolver::Xpress,
            ExternalLinearCliSolver::Lindo,
        ]
    }
}

/// Availability/probe status for a local solver CLI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliProbeStatus {
    Ready,
    NotInstalled,
    BridgeUnsupported,
    SmokeFailed,
}

impl ExternalLinearCliProbeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliProbeStatus::Ready => "ready",
            ExternalLinearCliProbeStatus::NotInstalled => "not-installed",
            ExternalLinearCliProbeStatus::BridgeUnsupported => "bridge-unsupported",
            ExternalLinearCliProbeStatus::SmokeFailed => "smoke-failed",
        }
    }
}

/// Solve status reported by the local CLI bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliStatus {
    Optimal,
    Infeasible,
    Unbounded,
    Unavailable,
    NumericalError,
    Unknown,
}

impl ExternalLinearCliStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliStatus::Optimal => "optimal",
            ExternalLinearCliStatus::Infeasible => "infeasible",
            ExternalLinearCliStatus::Unbounded => "unbounded",
            ExternalLinearCliStatus::Unavailable => "unavailable",
            ExternalLinearCliStatus::NumericalError => "numerical-error",
            ExternalLinearCliStatus::Unknown => "unknown",
        }
    }

    pub fn from_str(status: &str) -> Self {
        match status {
            "optimal" => ExternalLinearCliStatus::Optimal,
            "infeasible" => ExternalLinearCliStatus::Infeasible,
            "unbounded" => ExternalLinearCliStatus::Unbounded,
            "unavailable" => ExternalLinearCliStatus::Unavailable,
            "numerical-error" => ExternalLinearCliStatus::NumericalError,
            _ => ExternalLinearCliStatus::Unknown,
        }
    }
}

/// Options for invoking a locally installed external solver CLI.
#[derive(Clone, Debug)]
pub struct ExternalLinearCliOptions {
    pub solver: ExternalLinearCliSolver,
    /// Solver time limit in seconds. Defaults to 10 seconds.
    pub time_limit_secs: Option<f64>,
    /// Python executable for the bridge. Defaults to `PYTHON_BIN`, then
    /// `PYTHON`, then `python3`.
    pub python: Option<String>,
    /// Override path to `linear_cli_reference.py`.
    pub script_path: Option<PathBuf>,
}

impl Default for ExternalLinearCliOptions {
    fn default() -> Self {
        Self {
            solver: ExternalLinearCliSolver::Highs,
            time_limit_secs: None,
            python: None,
            script_path: None,
        }
    }
}

/// Result returned by a local external solver CLI.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalLinearCliSolution {
    pub status: ExternalLinearCliStatus,
    pub solver: String,
    pub x: Vec<f64>,
    pub objective: Option<f64>,
    pub elapsed_ms: f64,
    pub message: String,
}

/// PATH/bridge/smoke-test probe for one local solver CLI.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalLinearCliProbe {
    pub kind: ExternalLinearCliKind,
    pub solver: ExternalLinearCliSolver,
    pub command: Option<PathBuf>,
    pub status: ExternalLinearCliProbeStatus,
    pub smoke_status: Option<ExternalLinearCliStatus>,
    pub elapsed_ms: f64,
    pub message: String,
}

#[derive(Debug, Deserialize)]
struct RawExternalLinearCliSolution {
    status: String,
    solver: String,
    x: Vec<f64>,
    objective: Option<f64>,
    message: String,
}

/// Serialize an [`LPProblem`] into the JSON contract accepted by
/// `scripts/linear_cli_reference.py`.
pub fn lp_problem_to_cli_json(problem: &LPProblem) -> Value {
    json!({
        "lp": {
            "sense": problem.sense.as_str(),
            "c": f64_vec(&problem.c),
            "a_ub": opt_matrix_f64(problem.a_ub.as_ref()),
            "b_ub": opt_vec_f64(problem.b_ub.as_ref()),
            "a_eq": opt_matrix_f64(problem.a_eq.as_ref()),
            "b_eq": opt_vec_f64(problem.b_eq.as_ref()),
            "lb": opt_vec_opt_f64(problem.lb.as_ref()),
            "ub": opt_vec_opt_f64(problem.ub.as_ref()),
            "var_names": option_strings(problem.var_names.as_ref()),
            "con_names": option_strings(problem.con_names.as_ref()),
        }
    })
}

/// Serialize an [`IPMIPProblem`] into the JSON contract accepted by
/// `scripts/linear_cli_reference.py`.
pub fn ipmip_problem_to_cli_json(problem: &IPMIPProblem) -> Value {
    json!({
        "sense": problem.sense.as_str(),
        "c": f64_vec(&problem.c),
        "a": matrix_f64(&problem.a),
        "b": f64_vec(&problem.b),
        "integer_vars": problem.integer_vars,
        "ub": opt_plain_vec_f64(problem.ub.as_ref()),
        "var_names": option_strings(problem.var_names.as_ref()),
        "con_names": option_strings(problem.con_names.as_ref()),
    })
}

/// Solve an LP through a locally installed command-line solver.
pub fn solve_lp_with_external_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Lp,
        lp_problem_to_cli_json(problem),
        opts,
    )
}

/// Solve an IP/MIP through a locally installed command-line solver.
pub fn solve_ipmip_with_external_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Return the first executable-like command path found for a solver's aliases.
pub fn external_linear_cli_command(solver: ExternalLinearCliSolver) -> Option<PathBuf> {
    find_first_command(solver.command_env_vars(), solver.command_aliases())
}

/// Probe one solver for installation, bridge support, and a tiny smoke solve.
pub fn probe_external_linear_cli_solver(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliProbe {
    let t0 = Instant::now();
    let solver = opts.solver;
    let command = external_linear_cli_command(solver);
    if command.is_none() {
        return ExternalLinearCliProbe {
            kind,
            solver,
            command,
            status: ExternalLinearCliProbeStatus::NotInstalled,
            smoke_status: None,
            elapsed_ms: elapsed_ms(t0),
            message: format!(
                "no executable found via env vars [{}] or PATH aliases [{}]",
                solver.command_env_vars().join(", "),
                solver.command_aliases().join(", ")
            ),
        };
    }

    if !solver.supports_kind(kind) {
        return ExternalLinearCliProbe {
            kind,
            solver,
            command,
            status: ExternalLinearCliProbeStatus::BridgeUnsupported,
            smoke_status: None,
            elapsed_ms: elapsed_ms(t0),
            message: format!(
                "{} is installed, but this bridge does not yet support {} solves for it",
                solver.as_str(),
                kind.as_str()
            ),
        };
    }

    let mut smoke_opts = opts.clone();
    if smoke_opts.time_limit_secs.is_none() {
        smoke_opts.time_limit_secs = Some(2.0);
    }
    let solution = match kind {
        ExternalLinearCliKind::Lp => {
            solve_lp_with_external_cli(&external_linear_cli_smoke_lp(), &smoke_opts)
        }
        ExternalLinearCliKind::Mip => {
            solve_ipmip_with_external_cli(&external_linear_cli_smoke_mip(), &smoke_opts)
        }
    };
    let smoke_ok = solution.status == ExternalLinearCliStatus::Optimal
        && solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1e-8)
        && solution.x.len() == 1
        && (solution.x[0] - 1.0).abs() <= 1e-8;

    ExternalLinearCliProbe {
        kind,
        solver,
        command,
        status: if smoke_ok {
            ExternalLinearCliProbeStatus::Ready
        } else {
            ExternalLinearCliProbeStatus::SmokeFailed
        },
        smoke_status: Some(solution.status),
        elapsed_ms: elapsed_ms(t0),
        message: if smoke_ok {
            format!(
                "{} solved the local {} smoke model",
                solver.as_str(),
                kind.as_str()
            )
        } else {
            format!(
                "{} smoke solve failed: status={} objective={:?} x={:?} message={}",
                solver.as_str(),
                solution.status.as_str(),
                solution.objective,
                solution.x,
                solution.message
            )
        },
    }
}

/// Probe a list of solver families using shared options.
pub fn probe_external_linear_cli_solvers(
    kind: ExternalLinearCliKind,
    solvers: &[ExternalLinearCliSolver],
    base_opts: &ExternalLinearCliOptions,
) -> Vec<ExternalLinearCliProbe> {
    solvers
        .iter()
        .copied()
        .map(|solver| {
            let mut opts = base_opts.clone();
            opts.solver = solver;
            probe_external_linear_cli_solver(kind, &opts)
        })
        .collect()
}

/// Solve a raw bridge-compatible JSON payload through a locally installed
/// command-line solver. This is useful for source-level features that compile
/// through the Python reference bridge before writing the solver LP file.
pub fn solve_linear_cli_json(
    kind: ExternalLinearCliKind,
    problem_json: Value,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let solver_name = opts.solver.as_str();
    let bridge_solver = format!("{solver_name}:cli");
    let stdin_json = match serde_json::to_string(&problem_json) {
        Ok(stdin_json) => stdin_json,
        Err(err) => {
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                format!("failed to serialize problem JSON: {err}"),
                elapsed_ms(t0),
            );
        }
    };
    let python = resolve_python(opts);
    let script_path = opts
        .script_path
        .clone()
        .unwrap_or_else(default_linear_cli_script_path);
    let time_limit = normalized_time_limit(opts.time_limit_secs);

    let mut child = match Command::new(&python)
        .arg(&script_path)
        .arg("--kind")
        .arg(kind.as_str())
        .arg("--solver")
        .arg(solver_name)
        .arg("--time-limit")
        .arg(time_limit.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start local CLI bridge with python '{}' and script '{}': {err}",
                    python,
                    script_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };

    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(stdin_json.as_bytes()) {
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                format!("failed to write local CLI bridge stdin: {err}"),
                elapsed_ms(t0),
            );
        }
    }

    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                format!("failed while waiting for local CLI bridge: {err}"),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);

    if !output.status.success() {
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
            elapsed,
        );
    }

    match serde_json::from_slice::<RawExternalLinearCliSolution>(&output.stdout) {
        Ok(raw) => ExternalLinearCliSolution {
            status: ExternalLinearCliStatus::from_str(&raw.status),
            solver: raw.solver,
            x: raw.x,
            objective: raw.objective,
            elapsed_ms: elapsed,
            message: raw.message,
        },
        Err(err) => external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to parse local CLI bridge output: {err}; stdout='{}'; stderr='{}'",
                String::from_utf8_lossy(&output.stdout).trim(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed,
        ),
    }
}

fn external_cli_failure(
    status: ExternalLinearCliStatus,
    solver: String,
    message: String,
    elapsed_ms: f64,
) -> ExternalLinearCliSolution {
    ExternalLinearCliSolution {
        status,
        solver,
        x: Vec::new(),
        objective: None,
        elapsed_ms,
        message,
    }
}

fn resolve_python(opts: &ExternalLinearCliOptions) -> String {
    opts.python
        .clone()
        .or_else(|| std::env::var("PYTHON_BIN").ok())
        .or_else(|| std::env::var("PYTHON").ok())
        .unwrap_or_else(|| "python3".to_string())
}

fn default_linear_cli_script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("linear_cli_reference.py")
}

fn find_first_command(env_vars: &[&str], aliases: &[&str]) -> Option<PathBuf> {
    let mut saw_configured_env = false;
    for env_var in env_vars {
        if let Some(value) = std::env::var_os(env_var) {
            if !value.to_string_lossy().trim().is_empty() {
                saw_configured_env = true;
                if let Some(path) = resolve_command_candidate(&PathBuf::from(value)) {
                    return Some(path);
                }
            }
        }
    }
    if saw_configured_env {
        return None;
    }

    let path_var = std::env::var_os("PATH")?;
    let path_dirs: Vec<PathBuf> = std::env::split_paths(&path_var).collect();
    for alias in aliases {
        let alias_path = PathBuf::from(alias);
        if let Some(path) = resolve_command_candidate(&alias_path) {
            return Some(path);
        }
        for dir in &path_dirs {
            let candidate = dir.join(alias);
            if let Some(path) = resolve_command_candidate(&candidate) {
                return Some(path);
            }
        }
    }
    None
}

fn resolve_command_candidate(candidate: &Path) -> Option<PathBuf> {
    if candidate.components().count() > 1 {
        return candidate.is_file().then(|| candidate.to_path_buf());
    }

    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let path = dir.join(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn external_linear_cli_smoke_lp() -> LPProblem {
    LPProblem {
        sense: crate::des::general::lp::Sense::Max,
        c: vec![1.0],
        a_ub: Some(vec![vec![1.0]]),
        b_ub: Some(vec![1.0]),
        ..Default::default()
    }
}

fn external_linear_cli_smoke_mip() -> IPMIPProblem {
    IPMIPProblem {
        sense: crate::des::general::lp::Sense::Max,
        c: vec![1.0],
        a: vec![vec![1.0]],
        b: vec![1.0],
        integer_vars: vec![true],
        ub: Some(vec![1.0]),
        var_names: None,
        con_names: None,
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    }
}

fn normalized_time_limit(time_limit_secs: Option<f64>) -> f64 {
    let value = time_limit_secs.unwrap_or(10.0);
    if value.is_finite() && value > 0.0 {
        value
    } else {
        10.0
    }
}

fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

fn f64_value(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn f64_vec(values: &[f64]) -> Value {
    Value::Array(values.iter().copied().map(f64_value).collect())
}

fn matrix_f64(rows: &[Vec<f64>]) -> Value {
    Value::Array(rows.iter().map(|row| f64_vec(row)).collect())
}

fn opt_vec_f64(values: Option<&Vec<f64>>) -> Value {
    values.map_or(Value::Null, |values| f64_vec(values))
}

fn opt_plain_vec_f64(values: Option<&Vec<f64>>) -> Value {
    values.map_or(Value::Null, |values| f64_vec(values))
}

fn opt_matrix_f64(rows: Option<&Vec<Vec<f64>>>) -> Value {
    rows.map_or(Value::Null, |rows| matrix_f64(rows))
}

fn opt_vec_opt_f64(values: Option<&Vec<Option<f64>>>) -> Value {
    values.map_or(Value::Null, |values| {
        Value::Array(
            values
                .iter()
                .map(|value| value.map_or(Value::Null, f64_value))
                .collect(),
        )
    })
}

fn option_strings(values: Option<&Vec<String>>) -> Value {
    values.map_or(Value::Null, |values| {
        Value::Array(values.iter().cloned().map(Value::String).collect())
    })
}

#[cfg(test)]
mod tests {
    use crate::des::general::external_linear_cli::{
        ipmip_problem_to_cli_json, lp_problem_to_cli_json, ExternalLinearCliKind,
        ExternalLinearCliProbeStatus, ExternalLinearCliSolver, ExternalLinearCliStatus,
    };
    use crate::des::general::ip_mip_des::IPMIPProblem;
    use crate::des::general::lp::{LPProblem, Sense};

    #[test]
    fn lp_payload_wraps_problem_for_bridge() {
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![1.0, 2.0],
            a_ub: Some(vec![vec![1.0, 1.0]]),
            b_ub: Some(vec![3.0]),
            ..Default::default()
        };
        let payload = lp_problem_to_cli_json(&p);
        assert_eq!(payload["lp"]["sense"], "min");
        assert_eq!(payload["lp"]["c"][1], 2.0);
        assert!(payload["lp"]["lb"].is_null());
    }

    #[test]
    fn ipmip_payload_uses_plain_mip_shape() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![4.0],
            a: vec![vec![1.0]],
            b: vec![1.0],
            integer_vars: vec![true],
            ub: Some(vec![1.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let payload = ipmip_problem_to_cli_json(&p);
        assert_eq!(payload["sense"], "max");
        assert_eq!(payload["integer_vars"][0], true);
        assert_eq!(payload["ub"][0], 1.0);
    }

    #[test]
    fn external_status_round_trips_bridge_spelling() {
        for status in [
            ExternalLinearCliStatus::Optimal,
            ExternalLinearCliStatus::Infeasible,
            ExternalLinearCliStatus::Unbounded,
            ExternalLinearCliStatus::Unavailable,
            ExternalLinearCliStatus::NumericalError,
            ExternalLinearCliStatus::Unknown,
        ] {
            assert_eq!(ExternalLinearCliStatus::from_str(status.as_str()), status);
        }
    }

    #[test]
    fn solver_aliases_and_kind_support_match_bridge_contract() {
        assert_eq!(ExternalLinearCliSolver::Glpk.command_aliases(), &["glpsol"]);
        assert_eq!(
            ExternalLinearCliSolver::Gurobi.command_env_vars(),
            &["GUROBI_CL_CMD", "GUROBI_CMD", "ORES_GUROBI_CMD"]
        );
        assert_eq!(
            ExternalLinearCliSolver::Lindo.command_env_vars(),
            &["LINDO_CMD", "LINDOAPI_CMD", "ORES_LINDO_CMD"]
        );
        assert_eq!(
            ExternalLinearCliSolver::Xpress.command_aliases(),
            &["optimizer", "xpress"]
        );
        assert!(ExternalLinearCliSolver::Highs.supports_kind(ExternalLinearCliKind::Lp));
        assert!(ExternalLinearCliSolver::Highs.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::Clp.supports_kind(ExternalLinearCliKind::Lp));
        assert!(!ExternalLinearCliSolver::Clp.supports_kind(ExternalLinearCliKind::Mip));
        assert!(!ExternalLinearCliSolver::Lindo.supports_kind(ExternalLinearCliKind::Lp));
        assert!(!ExternalLinearCliSolver::Xpress.supports_kind(ExternalLinearCliKind::Mip));
    }

    #[test]
    fn probe_status_strings_are_stable() {
        for (status, expected) in [
            (ExternalLinearCliProbeStatus::Ready, "ready"),
            (ExternalLinearCliProbeStatus::NotInstalled, "not-installed"),
            (
                ExternalLinearCliProbeStatus::BridgeUnsupported,
                "bridge-unsupported",
            ),
            (ExternalLinearCliProbeStatus::SmokeFailed, "smoke-failed"),
        ] {
            assert_eq!(status.as_str(), expected);
        }
    }
}
