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

use crate::des::general::ip_mip_des::{BranchOrCutConstraint, ConstraintKind, IPMIPProblem};
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

/// LP algorithm requested from CLI solvers that expose a comparable knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliLpAlgorithm {
    Simplex,
    Ipm,
}

impl ExternalLinearCliLpAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliLpAlgorithm::Simplex => "simplex",
            ExternalLinearCliLpAlgorithm::Ipm => "ipm",
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

/// MIP branching rule requested from CLI solvers that expose a comparable knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliBranchRule {
    FirstFractional,
    MostFractional,
}

impl ExternalLinearCliBranchRule {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliBranchRule::FirstFractional => "first-fractional",
            ExternalLinearCliBranchRule::MostFractional => "most-fractional",
        }
    }
}

/// MIP node-selection rule requested from CLI solvers that expose a comparable knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliNodeSelection {
    Dfs,
    BestBound,
}

impl ExternalLinearCliNodeSelection {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliNodeSelection::Dfs => "dfs",
            ExternalLinearCliNodeSelection::BestBound => "best-bound",
        }
    }
}

/// Presolve mode requested from CLI solvers that expose a comparable knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliPresolve {
    Auto,
    On,
    Off,
}

impl ExternalLinearCliPresolve {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliPresolve::Auto => "auto",
            ExternalLinearCliPresolve::On => "on",
            ExternalLinearCliPresolve::Off => "off",
        }
    }
}

/// MIP search feature mode requested from CLI solvers with comparable switches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliMipSwitch {
    Auto,
    On,
    Off,
}

impl ExternalLinearCliMipSwitch {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliMipSwitch::Auto => "auto",
            ExternalLinearCliMipSwitch::On => "on",
            ExternalLinearCliMipSwitch::Off => "off",
        }
    }
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
            ExternalLinearCliSolver::Lindo => &[
                "RUNLINDO_CMD",
                "LINDO_CMD",
                "LINDOAPI_CMD",
                "ORES_LINDO_CMD",
            ],
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
    Feasible,
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
            ExternalLinearCliStatus::Feasible => "feasible",
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
            "feasible" => ExternalLinearCliStatus::Feasible,
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
    /// LP algorithm family, when supported by the CLI.
    pub lp_algorithm: Option<ExternalLinearCliLpAlgorithm>,
    /// Branch-and-bound node limit for MIP solves, when supported by the CLI.
    pub max_nodes: Option<u64>,
    /// Feasible-solution limit for MIP solves, when supported by the CLI.
    pub solution_limit: Option<u64>,
    /// Requested number of external solution-pool members for MIP solves.
    pub solution_pool_size: Option<u64>,
    /// Relative MIP optimality gap tolerance, when supported by the CLI.
    pub relative_gap: Option<f64>,
    /// Absolute MIP optimality gap tolerance, when supported by the CLI.
    pub absolute_gap: Option<f64>,
    /// Objective target/limit for MIP solves, when supported by the CLI.
    pub objective_limit: Option<f64>,
    /// Primal/row feasibility tolerance, when supported by the CLI.
    pub primal_feasibility_tolerance: Option<f64>,
    /// Dual/reduced-cost feasibility tolerance, when supported by the CLI.
    pub dual_feasibility_tolerance: Option<f64>,
    /// Integer integrality tolerance for MIP solves, when supported by the CLI.
    pub integer_feasibility_tolerance: Option<f64>,
    /// Solver worker/thread cap, when supported by the CLI.
    pub threads: Option<u32>,
    /// Solver random seed, when supported by the CLI.
    pub random_seed: Option<u64>,
    /// Presolve mode, when supported by the CLI.
    pub presolve: Option<ExternalLinearCliPresolve>,
    /// MIP cut generation mode, when supported by the CLI.
    pub cuts: Option<ExternalLinearCliMipSwitch>,
    /// MIP primal heuristic mode, when supported by the CLI.
    pub heuristics: Option<ExternalLinearCliMipSwitch>,
    /// MIP branching rule, when supported by the CLI.
    pub branch_rule: Option<ExternalLinearCliBranchRule>,
    /// Per-variable MIP branch priorities in model variable order, when supported by the CLI.
    pub branch_priorities: Option<Vec<i32>>,
    /// MIP search node-selection rule, when supported by the CLI.
    pub node_selection: Option<ExternalLinearCliNodeSelection>,
    /// Optional MIP incumbent start in the bridge model's variable order.
    pub mip_start: Option<Vec<f64>>,
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
            lp_algorithm: None,
            max_nodes: None,
            solution_limit: None,
            solution_pool_size: None,
            relative_gap: None,
            absolute_gap: None,
            objective_limit: None,
            primal_feasibility_tolerance: None,
            dual_feasibility_tolerance: None,
            integer_feasibility_tolerance: None,
            threads: None,
            random_seed: None,
            presolve: None,
            cuts: None,
            heuristics: None,
            branch_rule: None,
            branch_priorities: None,
            node_selection: None,
            mip_start: None,
            python: None,
            script_path: None,
        }
    }
}

/// One member of an external MIP solution pool.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalLinearCliPoolMember {
    pub x: Vec<f64>,
    pub objective: f64,
}

/// Result returned by a local external solver CLI.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalLinearCliSolution {
    pub status: ExternalLinearCliStatus,
    pub solver: String,
    /// Version/build string reported by the external solver CLI, when available.
    pub solver_version: Option<String>,
    pub x: Vec<f64>,
    pub objective: Option<f64>,
    /// LP algorithm family accepted by the CLI, when reported.
    pub lp_algorithm: Option<String>,
    /// Best known dual bound reported by a MIP-capable CLI, when available.
    pub best_bound: Option<f64>,
    /// Feasible-solution limit accepted by the CLI, when reported.
    pub solution_limit: Option<u64>,
    /// Requested external solution-pool size, when reported.
    pub solution_pool_size: Option<u64>,
    /// External solution-pool members, when requested.
    pub solutions: Option<Vec<ExternalLinearCliPoolMember>>,
    /// Whether the external pool search proved there are no more feasible members.
    pub exhausted: Option<bool>,
    /// Relative MIP gap reported by a MIP-capable CLI, when available.
    pub mip_gap: Option<f64>,
    /// Absolute MIP gap reported by a MIP-capable CLI, when available.
    pub absolute_gap: Option<f64>,
    /// Objective target/limit accepted by the CLI, when reported.
    pub objective_limit: Option<f64>,
    /// Primal/row feasibility tolerance accepted by the CLI, when reported.
    pub primal_feasibility_tolerance: Option<f64>,
    /// Dual/reduced-cost feasibility tolerance accepted by the CLI, when reported.
    pub dual_feasibility_tolerance: Option<f64>,
    /// Integer integrality tolerance accepted by the CLI, when reported.
    pub integer_feasibility_tolerance: Option<f64>,
    /// Branch-and-bound nodes explored by a MIP-capable CLI, when available.
    pub nodes_explored: Option<u64>,
    /// Solver worker/thread cap accepted by the CLI, when reported.
    pub threads: Option<u32>,
    /// Solver random seed accepted by the CLI, when reported.
    pub random_seed: Option<u64>,
    /// Presolve mode accepted by the CLI, when reported.
    pub presolve: Option<String>,
    /// MIP cut generation mode accepted by the CLI, when reported.
    pub cuts: Option<String>,
    /// MIP primal heuristic mode accepted by the CLI, when reported.
    pub heuristics: Option<String>,
    /// MIP branching rule accepted by the CLI, when reported.
    pub branch_rule: Option<String>,
    /// Whether provided per-variable branch priorities were accepted by the CLI, when reported.
    pub branch_priorities_accepted: Option<bool>,
    /// Number of nonzero integer-variable branch priorities sent to the CLI, when reported.
    pub branch_priority_count: Option<u64>,
    /// MIP node-selection rule accepted by the CLI, when reported.
    pub node_selection: Option<String>,
    /// Whether a provided MIP start was accepted by the CLI, when reported.
    pub mip_start_accepted: Option<bool>,
    /// Objective value of the provided MIP start in the bridge model, when reported.
    pub mip_start_objective: Option<f64>,
    /// LP inequality row dual prices reported by a CLI, when available.
    pub dual_ub: Option<Vec<f64>>,
    /// LP equality row dual prices reported by a CLI, when available.
    pub dual_eq: Option<Vec<f64>>,
    /// LP reduced costs reported by a CLI, when available.
    pub reduced_costs: Option<Vec<f64>>,
    /// LP basis status for original variables, when reported by a CLI.
    pub var_basis: Option<Vec<String>>,
    /// LP basis status for rows (`A_ub` rows followed by `A_eq` rows), when reported by a CLI.
    pub row_basis: Option<Vec<String>>,
    /// LP simplex iterations reported by a CLI, when available.
    pub iterations: Option<u64>,
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
    pub solver_version: Option<String>,
    pub elapsed_ms: f64,
    pub message: String,
}

#[derive(Debug, Deserialize)]
struct RawExternalLinearCliSolution {
    status: String,
    solver: String,
    #[serde(rename = "solverVersion")]
    solver_version: Option<String>,
    x: Vec<f64>,
    objective: Option<f64>,
    #[serde(rename = "lpAlgorithm")]
    lp_algorithm: Option<String>,
    #[serde(rename = "bestBound")]
    best_bound: Option<f64>,
    #[serde(rename = "solutionLimit")]
    solution_limit: Option<u64>,
    #[serde(rename = "solutionPoolSize")]
    solution_pool_size: Option<u64>,
    solutions: Option<Vec<RawExternalLinearCliPoolMember>>,
    exhausted: Option<bool>,
    #[serde(rename = "mipGap")]
    mip_gap: Option<f64>,
    #[serde(rename = "absoluteGap")]
    absolute_gap: Option<f64>,
    #[serde(rename = "objectiveLimit")]
    objective_limit: Option<f64>,
    #[serde(rename = "primalFeasibilityTolerance")]
    primal_feasibility_tolerance: Option<f64>,
    #[serde(rename = "dualFeasibilityTolerance")]
    dual_feasibility_tolerance: Option<f64>,
    #[serde(rename = "integerFeasibilityTolerance")]
    integer_feasibility_tolerance: Option<f64>,
    #[serde(rename = "nodesExplored")]
    nodes_explored: Option<u64>,
    threads: Option<u32>,
    #[serde(rename = "randomSeed")]
    random_seed: Option<u64>,
    presolve: Option<String>,
    cuts: Option<String>,
    heuristics: Option<String>,
    #[serde(rename = "branchRule")]
    branch_rule: Option<String>,
    #[serde(rename = "branchPrioritiesAccepted")]
    branch_priorities_accepted: Option<bool>,
    #[serde(rename = "branchPriorityCount")]
    branch_priority_count: Option<u64>,
    #[serde(rename = "nodeSelection")]
    node_selection: Option<String>,
    #[serde(rename = "mipStartAccepted")]
    mip_start_accepted: Option<bool>,
    #[serde(rename = "mipStartObjective")]
    mip_start_objective: Option<f64>,
    #[serde(rename = "dualUB")]
    dual_ub: Option<Vec<f64>>,
    #[serde(rename = "dualEQ")]
    dual_eq: Option<Vec<f64>>,
    #[serde(rename = "reducedCosts")]
    reduced_costs: Option<Vec<f64>>,
    #[serde(rename = "varBasis")]
    var_basis: Option<Vec<String>>,
    #[serde(rename = "rowBasis")]
    row_basis: Option<Vec<String>>,
    iterations: Option<u64>,
    message: String,
}

#[derive(Debug, Deserialize)]
struct RawExternalLinearCliPoolMember {
    x: Vec<f64>,
    objective: f64,
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
        "lazy_constraints": lazy_constraints_json(problem.lazy_constraints.as_ref()),
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
            solver_version: None,
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
            solver_version: None,
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
        solver_version: solution.solver_version.clone(),
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
    let relative_gap = normalized_relative_gap(opts.relative_gap);
    let absolute_gap = normalized_absolute_gap(opts.absolute_gap);
    let objective_limit = normalized_objective_limit(opts.objective_limit);
    let primal_feasibility_tolerance = normalized_tolerance(opts.primal_feasibility_tolerance);
    let dual_feasibility_tolerance = normalized_tolerance(opts.dual_feasibility_tolerance);
    let integer_feasibility_tolerance = normalized_tolerance(opts.integer_feasibility_tolerance);
    let branch_priorities_json =
        match normalized_branch_priorities_json(opts.branch_priorities.as_deref()) {
            Ok(value) => value,
            Err(message) => {
                return external_cli_failure(
                    ExternalLinearCliStatus::NumericalError,
                    bridge_solver,
                    message,
                    elapsed_ms(t0),
                );
            }
        };
    let mip_start_json = match normalized_mip_start_json(opts.mip_start.as_deref()) {
        Ok(value) => value,
        Err(message) => {
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed_ms(t0),
            );
        }
    };

    let mut command = Command::new(&python);
    command
        .arg(&script_path)
        .arg("--kind")
        .arg(kind.as_str())
        .arg("--solver")
        .arg(solver_name)
        .arg("--time-limit")
        .arg(time_limit.to_string());
    if let Some(max_nodes) = opts.max_nodes {
        command.arg("--node-limit").arg(max_nodes.to_string());
    }
    if let Some(solution_limit) = opts.solution_limit {
        command
            .arg("--solution-limit")
            .arg(solution_limit.max(1).to_string());
    }
    if let Some(solution_pool_size) = opts.solution_pool_size {
        command
            .arg("--solution-pool-size")
            .arg(solution_pool_size.max(1).to_string());
    }
    if let Some(relative_gap) = relative_gap {
        command
            .arg("--relative-gap")
            .arg(format!("{relative_gap:.17}"));
    }
    if let Some(absolute_gap) = absolute_gap {
        command
            .arg("--absolute-gap")
            .arg(format!("{absolute_gap:.17}"));
    }
    if let Some(objective_limit) = objective_limit {
        command
            .arg("--objective-limit")
            .arg(format!("{objective_limit:.17}"));
    }
    if let Some(primal_feasibility_tolerance) = primal_feasibility_tolerance {
        command
            .arg("--primal-feasibility-tolerance")
            .arg(format!("{primal_feasibility_tolerance:.17}"));
    }
    if let Some(dual_feasibility_tolerance) = dual_feasibility_tolerance {
        command
            .arg("--dual-feasibility-tolerance")
            .arg(format!("{dual_feasibility_tolerance:.17}"));
    }
    if let Some(integer_feasibility_tolerance) = integer_feasibility_tolerance {
        command
            .arg("--integer-feasibility-tolerance")
            .arg(format!("{integer_feasibility_tolerance:.17}"));
    }
    if let Some(lp_algorithm) = opts.lp_algorithm {
        command.arg("--lp-algorithm").arg(lp_algorithm.as_str());
    }
    if let Some(threads) = opts.threads {
        command.arg("--threads").arg(threads.max(1).to_string());
    }
    if let Some(random_seed) = opts.random_seed {
        command.arg("--random-seed").arg(random_seed.to_string());
    }
    if let Some(presolve) = opts.presolve {
        command.arg("--presolve").arg(presolve.as_str());
    }
    if let Some(cuts) = opts.cuts {
        command.arg("--cuts").arg(cuts.as_str());
    }
    if let Some(heuristics) = opts.heuristics {
        command.arg("--heuristics").arg(heuristics.as_str());
    }
    if let Some(branch_rule) = opts.branch_rule {
        command.arg("--branch-rule").arg(branch_rule.as_str());
    }
    if let Some(branch_priorities_json) = branch_priorities_json {
        command
            .arg("--branch-priorities")
            .arg(branch_priorities_json);
    }
    if let Some(node_selection) = opts.node_selection {
        command.arg("--node-selection").arg(node_selection.as_str());
    }
    if let Some(mip_start_json) = mip_start_json {
        command.arg("--mip-start").arg(mip_start_json);
    }

    let mut child = match command
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
            solver_version: raw.solver_version,
            x: raw.x,
            objective: raw.objective,
            lp_algorithm: raw.lp_algorithm,
            best_bound: raw.best_bound,
            solution_limit: raw.solution_limit,
            solution_pool_size: raw.solution_pool_size,
            solutions: raw.solutions.map(|solutions| {
                solutions
                    .into_iter()
                    .map(|solution| ExternalLinearCliPoolMember {
                        x: solution.x,
                        objective: solution.objective,
                    })
                    .collect()
            }),
            exhausted: raw.exhausted,
            mip_gap: raw.mip_gap,
            absolute_gap: raw.absolute_gap,
            objective_limit: raw.objective_limit,
            primal_feasibility_tolerance: raw.primal_feasibility_tolerance,
            dual_feasibility_tolerance: raw.dual_feasibility_tolerance,
            integer_feasibility_tolerance: raw.integer_feasibility_tolerance,
            nodes_explored: raw.nodes_explored,
            threads: raw.threads,
            random_seed: raw.random_seed,
            presolve: raw.presolve,
            cuts: raw.cuts,
            heuristics: raw.heuristics,
            branch_rule: raw.branch_rule,
            branch_priorities_accepted: raw.branch_priorities_accepted,
            branch_priority_count: raw.branch_priority_count,
            node_selection: raw.node_selection,
            mip_start_accepted: raw.mip_start_accepted,
            mip_start_objective: raw.mip_start_objective,
            dual_ub: raw.dual_ub,
            dual_eq: raw.dual_eq,
            reduced_costs: raw.reduced_costs,
            var_basis: raw.var_basis,
            row_basis: raw.row_basis,
            iterations: raw.iterations,
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
        solver_version: None,
        x: Vec::new(),
        objective: None,
        lp_algorithm: None,
        best_bound: None,
        solution_limit: None,
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: None,
        absolute_gap: None,
        objective_limit: None,
        primal_feasibility_tolerance: None,
        dual_feasibility_tolerance: None,
        integer_feasibility_tolerance: None,
        nodes_explored: None,
        threads: None,
        random_seed: None,
        presolve: None,
        cuts: None,
        heuristics: None,
        branch_rule: None,
        branch_priorities_accepted: None,
        branch_priority_count: None,
        node_selection: None,
        mip_start_accepted: None,
        mip_start_objective: None,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        iterations: None,
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

fn normalized_relative_gap(relative_gap: Option<f64>) -> Option<f64> {
    relative_gap.filter(|value| value.is_finite() && *value >= 0.0)
}

fn normalized_absolute_gap(absolute_gap: Option<f64>) -> Option<f64> {
    absolute_gap.filter(|value| value.is_finite() && *value >= 0.0)
}

fn normalized_objective_limit(objective_limit: Option<f64>) -> Option<f64> {
    objective_limit.filter(|value| value.is_finite())
}

fn normalized_tolerance(tolerance: Option<f64>) -> Option<f64> {
    tolerance.filter(|value| value.is_finite() && *value > 0.0)
}

fn normalized_branch_priorities_json(
    branch_priorities: Option<&[i32]>,
) -> Result<Option<String>, String> {
    let Some(branch_priorities) = branch_priorities else {
        return Ok(None);
    };
    serde_json::to_string(branch_priorities)
        .map(Some)
        .map_err(|err| format!("failed to serialize branch_priorities: {err}"))
}

fn normalized_mip_start_json(mip_start: Option<&[f64]>) -> Result<Option<String>, String> {
    let Some(mip_start) = mip_start else {
        return Ok(None);
    };
    if mip_start.iter().any(|value| !value.is_finite()) {
        return Err("mip_start values must be finite".to_string());
    }
    serde_json::to_string(mip_start)
        .map(Some)
        .map_err(|err| format!("failed to serialize mip_start: {err}"))
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

fn lazy_constraints_json(rows: Option<&Vec<BranchOrCutConstraint>>) -> Value {
    rows.map_or(Value::Null, |rows| {
        Value::Array(
            rows.iter()
                .map(|row| {
                    json!({
                        "coefs": f64_vec(&row.coefs),
                        "rhs": f64_value(row.rhs),
                        "name": &row.name,
                        "kind": constraint_kind_name(row.kind),
                    })
                })
                .collect(),
        )
    })
}

fn constraint_kind_name(kind: ConstraintKind) -> &'static str {
    match kind {
        ConstraintKind::Branch => "branch",
        ConstraintKind::Cut => "cut",
        ConstraintKind::Lazy => "lazy",
    }
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
    use crate::des::general::ip_mip_des::{BranchOrCutConstraint, ConstraintKind, IPMIPProblem};
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
    fn ipmip_payload_includes_lazy_constraints_for_external_validation() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![2.0],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: Some(vec![BranchOrCutConstraint {
                coefs: vec![1.0, 1.0],
                rhs: 1.0,
                name: "lazy-at-most-one".to_string(),
                kind: ConstraintKind::Lazy,
            }]),
            variable_nodes: None,
            constraint_nodes: None,
        };
        let payload = ipmip_problem_to_cli_json(&p);
        assert_eq!(payload["lazy_constraints"][0]["coefs"][1], 1.0);
        assert_eq!(payload["lazy_constraints"][0]["rhs"], 1.0);
        assert_eq!(payload["lazy_constraints"][0]["name"], "lazy-at-most-one");
        assert_eq!(payload["lazy_constraints"][0]["kind"], "lazy");
    }

    #[test]
    fn external_status_round_trips_bridge_spelling() {
        for status in [
            ExternalLinearCliStatus::Optimal,
            ExternalLinearCliStatus::Feasible,
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
            &[
                "RUNLINDO_CMD",
                "LINDO_CMD",
                "LINDOAPI_CMD",
                "ORES_LINDO_CMD"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::Xpress.command_aliases(),
            &["optimizer", "xpress"]
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
