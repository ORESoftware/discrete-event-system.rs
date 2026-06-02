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
use crate::des::general::lp::{LPProblem, Sense};

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

/// File/model format to hand to the external CLI bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliModelFormat {
    CplexLp,
    Mps,
}

impl ExternalLinearCliModelFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliModelFormat::CplexLp => "lp",
            ExternalLinearCliModelFormat::Mps => "mps",
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
            ExternalLinearCliSolver::Lindo => &["runlindo", "lindo", "lindoapi"],
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
                    | ExternalLinearCliSolver::Xpress
                    | ExternalLinearCliSolver::Lindo
            ),
            ExternalLinearCliKind::Mip => matches!(
                self,
                ExternalLinearCliSolver::Highs
                    | ExternalLinearCliSolver::Glpk
                    | ExternalLinearCliSolver::Scip
                    | ExternalLinearCliSolver::Cbc
                    | ExternalLinearCliSolver::Gurobi
                    | ExternalLinearCliSolver::Cplex
                    | ExternalLinearCliSolver::Xpress
                    | ExternalLinearCliSolver::Lindo
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
    /// Optional MIP node limit. Ignored for LP probes and solves.
    pub node_limit: Option<usize>,
    /// Optional relative MIP optimality gap. Ignored for LP probes and solves.
    pub relative_gap: Option<f64>,
    /// Optional solver thread count for CLI backends that support it.
    pub threads: Option<usize>,
    /// Optional deterministic random seed for CLI backends that support it.
    pub random_seed: Option<u32>,
    /// Model file format used by the bridge. Defaults to CPLEX LP syntax.
    pub model_format: ExternalLinearCliModelFormat,
    /// Python executable for the bridge. Defaults to `PYTHON_BIN`, then
    /// `PYTHON`, then `python3`.
    pub python: Option<String>,
    /// Optional explicit solver executable path/name. When set, this is passed
    /// to the bridge through a per-solver environment override instead of
    /// relying only on `PATH` discovery.
    pub command_path: Option<PathBuf>,
    /// Override path to `linear_cli_reference.py`.
    pub script_path: Option<PathBuf>,
}

impl Default for ExternalLinearCliOptions {
    fn default() -> Self {
        Self {
            solver: ExternalLinearCliSolver::Highs,
            time_limit_secs: None,
            node_limit: None,
            relative_gap: None,
            threads: None,
            random_seed: None,
            model_format: ExternalLinearCliModelFormat::CplexLp,
            python: None,
            command_path: None,
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

/// Export an LP as a CPLEX LP-format string accepted by the local CLI bridge.
///
/// The export uses stable `x0`, `x1`, ... column names so solver solution files
/// can be parsed back into vector positions without relying on display names.
pub fn lp_problem_to_cplex_lp_string(problem: &LPProblem) -> String {
    let n = problem.c.len();
    let lbs = problem.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]);
    let ubs = problem.ub.clone().unwrap_or_else(|| vec![None; n]);
    let integer_vars = vec![false; n];
    cplex_lp_string(
        problem.sense,
        &problem.c,
        problem.a_ub.as_deref().unwrap_or(&[]),
        problem.b_ub.as_deref().unwrap_or(&[]),
        problem.a_eq.as_deref().unwrap_or(&[]),
        problem.b_eq.as_deref().unwrap_or(&[]),
        &lbs,
        &ubs,
        &integer_vars,
    )
}

/// Export an IP/MIP as a CPLEX LP-format string accepted by many solver CLIs.
///
/// `IPMIPProblem` lower bounds are the branch-and-cut backend default of zero;
/// finite upper bounds and integer markers are emitted as LP `Bounds`,
/// `General`, and `Binary` sections.
pub fn ipmip_problem_to_cplex_lp_string(problem: &IPMIPProblem) -> String {
    let n = problem.c.len();
    let lbs = vec![Some(0.0); n];
    let ubs = problem
        .ub
        .as_ref()
        .map(|upper| upper.iter().copied().map(Some).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![None; n]);
    cplex_lp_string(
        problem.sense,
        &problem.c,
        &problem.a,
        &problem.b,
        &[],
        &[],
        &lbs,
        &ubs,
        &problem.integer_vars,
    )
}

/// Export an LP as a free-format MPS string.
///
/// MPS is the common file interchange format for commercial and open-source
/// LP/MIP solvers. This exporter keeps stable `x0`, `x1`, ... column names for
/// the same reason as the LP-format exporter.
pub fn lp_problem_to_mps_string(problem: &LPProblem) -> String {
    let n = problem.c.len();
    let lbs = problem.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]);
    let ubs = problem.ub.clone().unwrap_or_else(|| vec![None; n]);
    let integer_vars = vec![false; n];
    mps_string(
        problem.sense,
        &problem.c,
        problem.a_ub.as_deref().unwrap_or(&[]),
        problem.b_ub.as_deref().unwrap_or(&[]),
        problem.a_eq.as_deref().unwrap_or(&[]),
        problem.b_eq.as_deref().unwrap_or(&[]),
        &lbs,
        &ubs,
        &integer_vars,
    )
}

/// Export an IP/MIP as a free-format MPS string with integer markers.
pub fn ipmip_problem_to_mps_string(problem: &IPMIPProblem) -> String {
    let n = problem.c.len();
    let lbs = vec![Some(0.0); n];
    let ubs = problem
        .ub
        .as_ref()
        .map(|upper| upper.iter().copied().map(Some).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![None; n]);
    mps_string(
        problem.sense,
        &problem.c,
        &problem.a,
        &problem.b,
        &[],
        &[],
        &lbs,
        &ubs,
        &problem.integer_vars,
    )
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
    find_first_command(solver.command_aliases())
}

/// Return the configured command override, or the first command found on `PATH`.
pub fn external_linear_cli_command_with_options(
    solver: ExternalLinearCliSolver,
    opts: &ExternalLinearCliOptions,
) -> Option<PathBuf> {
    opts.command_path
        .as_ref()
        .cloned()
        .or_else(|| external_linear_cli_command(solver))
}

/// Probe one solver for installation, bridge support, and a tiny smoke solve.
pub fn probe_external_linear_cli_solver(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliProbe {
    let t0 = Instant::now();
    let solver = opts.solver;
    let command = external_linear_cli_command_with_options(solver, opts);
    if command.is_none() {
        return ExternalLinearCliProbe {
            kind,
            solver,
            command,
            status: ExternalLinearCliProbeStatus::NotInstalled,
            smoke_status: None,
            elapsed_ms: elapsed_ms(t0),
            message: format!(
                "no executable found on PATH for aliases: {}",
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
    let node_limit = normalized_node_limit(opts.node_limit);
    let relative_gap = normalized_relative_gap(opts.relative_gap);
    let threads = normalized_threads(opts.threads);
    let random_seed = normalized_random_seed(opts.random_seed);

    let mut command = Command::new(&python);
    command
        .arg(&script_path)
        .arg("--kind")
        .arg(kind.as_str())
        .arg("--solver")
        .arg(solver_name)
        .arg("--model-format")
        .arg(opts.model_format.as_str())
        .arg("--time-limit")
        .arg(time_limit.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if kind == ExternalLinearCliKind::Mip {
        if let Some(node_limit) = node_limit {
            command.arg("--node-limit").arg(node_limit.to_string());
        }
        if let Some(relative_gap) = relative_gap {
            command.arg("--relative-gap").arg(relative_gap.to_string());
        }
    }
    if let Some(threads) = threads {
        command.arg("--threads").arg(threads.to_string());
    }
    if let Some(random_seed) = random_seed {
        command.arg("--random-seed").arg(random_seed.to_string());
    }
    if let Some(command_path) = &opts.command_path {
        command.env(solver_command_env_var(opts.solver), command_path);
    }

    let mut child = match command.spawn() {
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

fn solver_command_env_var(solver: ExternalLinearCliSolver) -> String {
    format!("ORES_{}_BIN", solver.as_str().to_ascii_uppercase())
}

fn default_linear_cli_script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("linear_cli_reference.py")
}

fn cplex_lp_string(
    sense: Sense,
    c: &[f64],
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    lbs: &[Option<f64>],
    ubs: &[Option<f64>],
    integer_vars: &[bool],
) -> String {
    let n = c.len();
    let names = (0..n).map(|i| format!("x{i}")).collect::<Vec<_>>();
    let binary_vars = (0..n)
        .filter(|&i| {
            integer_vars.get(i).copied().unwrap_or(false)
                && lbs.get(i).copied().flatten().unwrap_or(0.0).abs() <= 1.0e-12
                && ubs
                    .get(i)
                    .copied()
                    .flatten()
                    .is_some_and(|ub| (ub - 1.0).abs() <= 1.0e-12)
        })
        .collect::<Vec<_>>();
    let binary_set = binary_vars
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let general_vars = (0..n)
        .filter(|&i| integer_vars.get(i).copied().unwrap_or(false) && !binary_set.contains(&i))
        .collect::<Vec<_>>();

    let mut out = String::new();
    out.push_str(match sense {
        Sense::Max => "Maximize\n",
        Sense::Min => "Minimize\n",
    });
    out.push_str(" obj: ");
    out.push_str(&lp_term_expr(c, &names));
    out.push('\n');
    out.push_str("Subject To\n");
    for (i, (row, rhs)) in le_rows.iter().zip(le_rhs).enumerate() {
        out.push_str(&format!(
            " c{i}: {} <= {}\n",
            lp_term_expr(row, &names),
            fmt_lp_number(*rhs)
        ));
    }
    for (i, (row, rhs)) in eq_rows.iter().zip(eq_rhs).enumerate() {
        out.push_str(&format!(
            " e{i}: {} = {}\n",
            lp_term_expr(row, &names),
            fmt_lp_number(*rhs)
        ));
    }
    if le_rows.is_empty() && eq_rows.is_empty() {
        out.push_str(" c0: 0 x0 <= 0\n");
    }
    out.push_str("Bounds\n");
    for i in 0..n {
        if binary_set.contains(&i) {
            continue;
        }
        let lb = lbs.get(i).copied().flatten();
        let ub = ubs.get(i).copied().flatten();
        match (lb, ub) {
            (None, None) => out.push_str(&format!(" {} free\n", names[i])),
            (None, Some(upper)) => {
                out.push_str(&format!(" {} <= {}\n", names[i], fmt_lp_number(upper)));
            }
            (Some(lower), None) => {
                out.push_str(&format!(" {} <= {}\n", fmt_lp_number(lower), names[i]));
            }
            (Some(lower), Some(upper)) => {
                out.push_str(&format!(
                    " {} <= {} <= {}\n",
                    fmt_lp_number(lower),
                    names[i],
                    fmt_lp_number(upper)
                ));
            }
        }
    }
    if !general_vars.is_empty() {
        out.push_str("General\n ");
        out.push_str(
            &general_vars
                .iter()
                .map(|&i| names[i].as_str())
                .collect::<Vec<_>>()
                .join(" "),
        );
        out.push('\n');
    }
    if !binary_vars.is_empty() {
        out.push_str("Binary\n ");
        out.push_str(
            &binary_vars
                .iter()
                .map(|&i| names[i].as_str())
                .collect::<Vec<_>>()
                .join(" "),
        );
        out.push('\n');
    }
    out.push_str("End\n");
    out
}

fn lp_term_expr(coefs: &[f64], names: &[String]) -> String {
    let mut parts = Vec::new();
    for (coef, name) in coefs.iter().zip(names) {
        if coef.abs() <= 1.0e-12 {
            continue;
        }
        let sign = if *coef < 0.0 { "-" } else { "+" };
        let mag = coef.abs();
        let body = if (mag - 1.0).abs() <= 1.0e-12 {
            name.clone()
        } else {
            format!("{} {name}", fmt_lp_number(mag))
        };
        if parts.is_empty() {
            parts.push(if sign == "-" {
                format!("- {body}")
            } else {
                body
            });
        } else {
            parts.push(format!("{sign} {body}"));
        }
    }
    if parts.is_empty() {
        format!("0 {}", names.first().map(String::as_str).unwrap_or("x0"))
    } else {
        parts.join(" ")
    }
}

fn fmt_lp_number(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let mut out = format!("{value:.12}");
    if out.contains('.') {
        while out.ends_with('0') {
            out.pop();
        }
        if out.ends_with('.') {
            out.pop();
        }
    }
    out
}

fn mps_string(
    sense: Sense,
    c: &[f64],
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    lbs: &[Option<f64>],
    ubs: &[Option<f64>],
    integer_vars: &[bool],
) -> String {
    let n = c.len();
    let names = (0..n).map(|i| format!("x{i}")).collect::<Vec<_>>();
    let le_names = (0..le_rows.len())
        .map(|i| format!("c{i}"))
        .collect::<Vec<_>>();
    let eq_names = (0..eq_rows.len())
        .map(|i| format!("e{i}"))
        .collect::<Vec<_>>();
    let integer_indices = (0..n)
        .filter(|&i| integer_vars.get(i).copied().unwrap_or(false))
        .collect::<Vec<_>>();
    let integer_set = integer_indices
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();

    let mut out = String::new();
    out.push_str("NAME          ORES\n");
    out.push_str("OBJSENSE\n");
    out.push_str(match sense {
        Sense::Max => " MAX\n",
        Sense::Min => " MIN\n",
    });
    out.push_str("ROWS\n");
    out.push_str(" N  OBJ\n");
    for row_name in &le_names {
        out.push_str(&format!(" L  {row_name}\n"));
    }
    for row_name in &eq_names {
        out.push_str(&format!(" E  {row_name}\n"));
    }
    out.push_str("COLUMNS\n");
    for i in 0..n {
        if !integer_set.contains(&i) {
            push_mps_column(
                &mut out, &names[i], c[i], le_rows, &le_names, eq_rows, &eq_names,
            );
        }
    }
    if !integer_indices.is_empty() {
        out.push_str("    MARK0000  'MARKER'                 'INTORG'\n");
        for &i in &integer_indices {
            push_mps_column(
                &mut out, &names[i], c[i], le_rows, &le_names, eq_rows, &eq_names,
            );
        }
        out.push_str("    MARK0001  'MARKER'                 'INTEND'\n");
    }
    if !le_rows.is_empty() || !eq_rows.is_empty() {
        out.push_str("RHS\n");
        for (row_name, rhs) in le_names.iter().zip(le_rhs) {
            out.push_str(&format!(
                "    RHS1      {row_name:<8}  {}\n",
                fmt_lp_number(*rhs)
            ));
        }
        for (row_name, rhs) in eq_names.iter().zip(eq_rhs) {
            out.push_str(&format!(
                "    RHS1      {row_name:<8}  {}\n",
                fmt_lp_number(*rhs)
            ));
        }
    }
    out.push_str("BOUNDS\n");
    for i in 0..n {
        let lb = lbs.get(i).copied().flatten();
        let ub = ubs.get(i).copied().flatten();
        if is_binary_bound(integer_vars, lbs, ubs, i) {
            out.push_str(&format!(" BV BND1      {}\n", names[i]));
            continue;
        }
        match (lb, ub) {
            (None, None) => out.push_str(&format!(" FR BND1      {}\n", names[i])),
            (None, Some(upper)) => {
                out.push_str(&format!(" MI BND1      {}\n", names[i]));
                out.push_str(&format!(
                    " UP BND1      {:<8}  {}\n",
                    names[i],
                    fmt_lp_number(upper)
                ));
            }
            (Some(lower), None) => {
                if lower.abs() > 1.0e-12 {
                    out.push_str(&format!(
                        " LO BND1      {:<8}  {}\n",
                        names[i],
                        fmt_lp_number(lower)
                    ));
                }
            }
            (Some(lower), Some(upper)) => {
                if (lower - upper).abs() <= 1.0e-12 {
                    out.push_str(&format!(
                        " FX BND1      {:<8}  {}\n",
                        names[i],
                        fmt_lp_number(lower)
                    ));
                } else {
                    if lower.abs() > 1.0e-12 {
                        out.push_str(&format!(
                            " LO BND1      {:<8}  {}\n",
                            names[i],
                            fmt_lp_number(lower)
                        ));
                    }
                    out.push_str(&format!(
                        " UP BND1      {:<8}  {}\n",
                        names[i],
                        fmt_lp_number(upper)
                    ));
                }
            }
        }
    }
    out.push_str("ENDATA\n");
    out
}

fn push_mps_column(
    out: &mut String,
    name: &str,
    obj_coeff: f64,
    le_rows: &[Vec<f64>],
    le_names: &[String],
    eq_rows: &[Vec<f64>],
    eq_names: &[String],
) {
    if obj_coeff.abs() > 1.0e-12 {
        out.push_str(&format!(
            "    {name:<8}  OBJ       {}\n",
            fmt_lp_number(obj_coeff)
        ));
    }
    for (row, row_name) in le_rows.iter().zip(le_names) {
        push_mps_row_coef(out, name, row_name, row);
    }
    for (row, row_name) in eq_rows.iter().zip(eq_names) {
        push_mps_row_coef(out, name, row_name, row);
    }
}

fn push_mps_row_coef(out: &mut String, col_name: &str, row_name: &str, row: &[f64]) {
    let Some(var_idx) = col_name
        .strip_prefix('x')
        .and_then(|idx| idx.parse::<usize>().ok())
    else {
        return;
    };
    let Some(&coef) = row.get(var_idx) else {
        return;
    };
    if coef.abs() > 1.0e-12 {
        out.push_str(&format!(
            "    {col_name:<8}  {row_name:<8}  {}\n",
            fmt_lp_number(coef)
        ));
    }
}

fn is_binary_bound(
    integer_vars: &[bool],
    lbs: &[Option<f64>],
    ubs: &[Option<f64>],
    index: usize,
) -> bool {
    integer_vars.get(index).copied().unwrap_or(false)
        && lbs.get(index).copied().flatten().unwrap_or(0.0).abs() <= 1.0e-12
        && ubs
            .get(index)
            .copied()
            .flatten()
            .is_some_and(|ub| (ub - 1.0).abs() <= 1.0e-12)
}

fn find_first_command(aliases: &[&str]) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    let path_dirs: Vec<PathBuf> = std::env::split_paths(&path_var).collect();
    for alias in aliases {
        let alias_path = Path::new(alias);
        if alias_path.components().count() > 1 && alias_path.is_file() {
            return Some(alias_path.to_path_buf());
        }
        for dir in &path_dirs {
            let candidate = dir.join(alias);
            if candidate.is_file() {
                return Some(candidate);
            }
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

fn normalized_node_limit(node_limit: Option<usize>) -> Option<usize> {
    node_limit.filter(|value| *value > 0)
}

fn normalized_relative_gap(relative_gap: Option<f64>) -> Option<f64> {
    relative_gap.filter(|value| value.is_finite() && *value >= 0.0)
}

fn normalized_threads(threads: Option<usize>) -> Option<usize> {
    threads.filter(|value| *value > 0)
}

fn normalized_random_seed(random_seed: Option<u32>) -> Option<u32> {
    random_seed.filter(|value| *value <= i32::MAX as u32)
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
        external_linear_cli_command_with_options, ipmip_problem_to_cli_json,
        ipmip_problem_to_cplex_lp_string, ipmip_problem_to_mps_string, lp_problem_to_cli_json,
        lp_problem_to_cplex_lp_string, lp_problem_to_mps_string, normalized_node_limit,
        normalized_random_seed, normalized_relative_gap, normalized_threads,
        solver_command_env_var, ExternalLinearCliKind, ExternalLinearCliModelFormat,
        ExternalLinearCliOptions, ExternalLinearCliProbeStatus, ExternalLinearCliSolver,
        ExternalLinearCliStatus,
    };
    use crate::des::general::ip_mip_des::IPMIPProblem;
    use crate::des::general::lp::{LPProblem, Sense};
    use std::path::PathBuf;

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
    fn lp_cplex_export_uses_bounds_and_equalities() {
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![1.0, -2.0],
            a_eq: Some(vec![vec![1.0, 1.0]]),
            b_eq: Some(vec![2.0]),
            lb: Some(vec![None, Some(1.0)]),
            ub: Some(vec![Some(4.0), None]),
            ..Default::default()
        };
        let text = lp_problem_to_cplex_lp_string(&p);
        assert!(text.starts_with("Minimize\n"));
        assert!(text.contains(" obj: x0 - 2 x1\n"));
        assert!(text.contains(" e0: x0 + x1 = 2\n"));
        assert!(text.contains(" x0 <= 4\n"));
        assert!(text.contains(" 1 <= x1\n"));
        assert!(text.ends_with("End\n"));
    }

    #[test]
    fn ipmip_cplex_export_marks_binary_and_general_integer_vars() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 2.0, 0.0],
            a: vec![vec![1.0, 1.0, 1.0]],
            b: vec![3.0],
            integer_vars: vec![true, true, false],
            ub: Some(vec![1.0, 5.0, 10.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let text = ipmip_problem_to_cplex_lp_string(&p);
        assert!(text.starts_with("Maximize\n"));
        assert!(text.contains(" c0: x0 + x1 + x2 <= 3\n"));
        assert!(text.contains(" 0 <= x1 <= 5\n"));
        assert!(text.contains(" 0 <= x2 <= 10\n"));
        assert!(text.contains("General\n x1\n"));
        assert!(text.contains("Binary\n x0\n"));
    }

    #[test]
    fn lp_mps_export_uses_rows_columns_rhs_and_bounds() {
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![1.0, -2.0],
            a_ub: Some(vec![vec![1.0, 2.0]]),
            b_ub: Some(vec![4.0]),
            a_eq: Some(vec![vec![1.0, 1.0]]),
            b_eq: Some(vec![3.0]),
            lb: Some(vec![None, Some(1.0)]),
            ub: Some(vec![Some(5.0), None]),
            ..Default::default()
        };
        let text = lp_problem_to_mps_string(&p);
        assert!(text.starts_with("NAME          ORES\n"));
        assert!(text.contains("OBJSENSE\n MIN\n"));
        assert!(text.contains(" L  c0\n"));
        assert!(text.contains(" E  e0\n"));
        assert!(text.contains("    x0        OBJ       1\n"));
        assert!(text.contains("    x1        c0        2\n"));
        assert!(text.contains("    RHS1      c0        4\n"));
        assert!(text.contains(" MI BND1      x0\n"));
        assert!(text.contains(" UP BND1      x0        5\n"));
        assert!(text.contains(" LO BND1      x1        1\n"));
        assert!(text.ends_with("ENDATA\n"));
    }

    #[test]
    fn ipmip_mps_export_marks_integers_and_binaries() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![1.0],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 5.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let text = ipmip_problem_to_mps_string(&p);
        assert!(text.contains("OBJSENSE\n MAX\n"));
        assert!(text.contains("'INTORG'"));
        assert!(text.contains("    x0        OBJ       3\n"));
        assert!(text.contains("    x1        c0        1\n"));
        assert!(text.contains("'INTEND'"));
        assert!(text.contains(" BV BND1      x0\n"));
        assert!(text.contains(" UP BND1      x1        5\n"));
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
    fn model_format_strings_match_bridge_contract() {
        assert_eq!(ExternalLinearCliModelFormat::CplexLp.as_str(), "lp");
        assert_eq!(ExternalLinearCliModelFormat::Mps.as_str(), "mps");
        assert_eq!(
            ExternalLinearCliOptions::default().model_format,
            ExternalLinearCliModelFormat::CplexLp
        );
        assert_eq!(ExternalLinearCliOptions::default().node_limit, None);
        assert_eq!(ExternalLinearCliOptions::default().relative_gap, None);
        assert_eq!(ExternalLinearCliOptions::default().threads, None);
        assert_eq!(ExternalLinearCliOptions::default().random_seed, None);
    }

    #[test]
    fn solve_controls_are_normalized_before_bridge_call() {
        assert_eq!(normalized_node_limit(Some(1)), Some(1));
        assert_eq!(normalized_node_limit(Some(0)), None);
        assert_eq!(normalized_node_limit(None), None);
        assert_eq!(normalized_relative_gap(Some(0.0)), Some(0.0));
        assert_eq!(normalized_relative_gap(Some(0.25)), Some(0.25));
        assert_eq!(normalized_relative_gap(Some(f64::INFINITY)), None);
        assert_eq!(normalized_relative_gap(Some(f64::NAN)), None);
        assert_eq!(normalized_relative_gap(Some(-0.1)), None);
        assert_eq!(normalized_relative_gap(None), None);
        assert_eq!(normalized_threads(Some(2)), Some(2));
        assert_eq!(normalized_threads(Some(0)), None);
        assert_eq!(normalized_threads(None), None);
        assert_eq!(normalized_random_seed(Some(7)), Some(7));
        assert_eq!(
            normalized_random_seed(Some(i32::MAX as u32)),
            Some(i32::MAX as u32)
        );
        assert_eq!(normalized_random_seed(Some(i32::MAX as u32 + 1)), None);
        assert_eq!(normalized_random_seed(None), None);
    }

    #[test]
    fn solver_aliases_and_kind_support_match_bridge_contract() {
        assert_eq!(ExternalLinearCliSolver::Glpk.command_aliases(), &["glpsol"]);
        assert_eq!(
            ExternalLinearCliSolver::Xpress.command_aliases(),
            &["optimizer", "xpress"]
        );
        assert_eq!(
            ExternalLinearCliSolver::Lindo.command_aliases(),
            &["runlindo", "lindo", "lindoapi"]
        );
        assert!(ExternalLinearCliSolver::Highs.supports_kind(ExternalLinearCliKind::Lp));
        assert!(ExternalLinearCliSolver::Highs.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::Clp.supports_kind(ExternalLinearCliKind::Lp));
        assert!(!ExternalLinearCliSolver::Clp.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::Lindo.supports_kind(ExternalLinearCliKind::Lp));
        assert!(ExternalLinearCliSolver::Lindo.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::Xpress.supports_kind(ExternalLinearCliKind::Lp));
        assert!(ExternalLinearCliSolver::Xpress.supports_kind(ExternalLinearCliKind::Mip));
    }

    #[test]
    fn command_override_is_preferred_over_path_lookup() {
        let configured = PathBuf::from("/opt/local/bin/highs");
        let opts = ExternalLinearCliOptions {
            solver: ExternalLinearCliSolver::Highs,
            command_path: Some(configured.clone()),
            ..Default::default()
        };
        assert_eq!(
            external_linear_cli_command_with_options(ExternalLinearCliSolver::Highs, &opts),
            Some(configured)
        );
    }

    #[test]
    fn command_override_env_names_are_stable() {
        assert_eq!(
            solver_command_env_var(ExternalLinearCliSolver::Highs),
            "ORES_HIGHS_BIN"
        );
        assert_eq!(
            solver_command_env_var(ExternalLinearCliSolver::Glpk),
            "ORES_GLPK_BIN"
        );
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
