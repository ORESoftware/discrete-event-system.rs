//! Local command-line adapters for installed LP/MIP solvers.
//!
//! This module exposes a Rust-facing interface for solver executables that are
//! installed locally (for example through Homebrew) without vendoring any
//! external binaries into the repository. HiGHS and GLPK use direct Rust
//! subprocess paths for plain LP/MIP models; the remaining solver-specific
//! command lines, richer source-model expansion, and solution-pool iteration
//! still live in `scripts/linear_cli_reference.py`. This module owns the library
//! boundary: problem serialization, subprocess execution, typed status mapping,
//! and elapsed-time accounting.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Number, Value};

use crate::des::general::ip_mip_des::{
    BranchOrCutConstraint, ConstraintKind, GeneralLinearIPMIPProblem, IPMIPProblem,
    IndicatorConstraint, IndicatorIPMIPProblem, LinearRowConstraint, LowerBoundedIPMIPProblem,
    MultiObjectiveIPMIPProblem, PiecewiseLinearConstraint, PwlIPMIPProblem,
    QuadraticObjectiveIPMIPProblem, QuadraticObjectiveTerm, SemiIPMIPProblem, SemiVariable,
    SosIPMIPProblem, SourceIPMIPProblem, SpecialOrderedSet,
};
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
    Soplex,
    QsoptEx,
    LpSolve,
    Gurobi,
    Cplex,
    Xpress,
    Lindo,
}

/// Broad licensing/install class for a local CLI solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLinearCliLicenseClass {
    OpenSource,
    Commercial,
}

impl ExternalLinearCliLicenseClass {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliLicenseClass::OpenSource => "open-source",
            ExternalLinearCliLicenseClass::Commercial => "commercial",
        }
    }
}

/// Server/UI-facing manifest entry for one locally installed solver CLI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalLinearCliSolverSpec {
    pub solver: ExternalLinearCliSolver,
    pub id: &'static str,
    pub display_name: &'static str,
    pub license_class: ExternalLinearCliLicenseClass,
    pub command_aliases: &'static [&'static str],
    pub command_env_vars: &'static [&'static str],
    pub command_dir_env_vars: &'static [&'static str],
    pub supports_lp: bool,
    pub supports_mip: bool,
    pub notes: &'static str,
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
    pub fn all() -> &'static [ExternalLinearCliSolver] {
        &[
            ExternalLinearCliSolver::Highs,
            ExternalLinearCliSolver::Glpk,
            ExternalLinearCliSolver::Scip,
            ExternalLinearCliSolver::Cbc,
            ExternalLinearCliSolver::Clp,
            ExternalLinearCliSolver::Soplex,
            ExternalLinearCliSolver::QsoptEx,
            ExternalLinearCliSolver::LpSolve,
            ExternalLinearCliSolver::Gurobi,
            ExternalLinearCliSolver::Cplex,
            ExternalLinearCliSolver::Xpress,
            ExternalLinearCliSolver::Lindo,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ExternalLinearCliSolver::Highs => "highs",
            ExternalLinearCliSolver::Glpk => "glpk",
            ExternalLinearCliSolver::Scip => "scip",
            ExternalLinearCliSolver::Cbc => "cbc",
            ExternalLinearCliSolver::Clp => "clp",
            ExternalLinearCliSolver::Soplex => "soplex",
            ExternalLinearCliSolver::QsoptEx => "qsopt-ex",
            ExternalLinearCliSolver::LpSolve => "lp-solve",
            ExternalLinearCliSolver::Gurobi => "gurobi",
            ExternalLinearCliSolver::Cplex => "cplex",
            ExternalLinearCliSolver::Xpress => "xpress",
            ExternalLinearCliSolver::Lindo => "lindo",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ExternalLinearCliSolver::Highs => "HiGHS",
            ExternalLinearCliSolver::Glpk => "GLPK",
            ExternalLinearCliSolver::Scip => "SCIP",
            ExternalLinearCliSolver::Cbc => "COIN-OR CBC",
            ExternalLinearCliSolver::Clp => "COIN-OR CLP",
            ExternalLinearCliSolver::Soplex => "SoPlex",
            ExternalLinearCliSolver::QsoptEx => "QSopt_ex",
            ExternalLinearCliSolver::LpSolve => "lp_solve",
            ExternalLinearCliSolver::Gurobi => "Gurobi Optimizer",
            ExternalLinearCliSolver::Cplex => "IBM ILOG CPLEX",
            ExternalLinearCliSolver::Xpress => "FICO Xpress",
            ExternalLinearCliSolver::Lindo => "LINDO Systems",
        }
    }

    pub fn license_class(self) -> ExternalLinearCliLicenseClass {
        match self {
            ExternalLinearCliSolver::Highs
            | ExternalLinearCliSolver::Glpk
            | ExternalLinearCliSolver::Scip
            | ExternalLinearCliSolver::Cbc
            | ExternalLinearCliSolver::Clp
            | ExternalLinearCliSolver::Soplex
            | ExternalLinearCliSolver::QsoptEx
            | ExternalLinearCliSolver::LpSolve => ExternalLinearCliLicenseClass::OpenSource,
            ExternalLinearCliSolver::Gurobi
            | ExternalLinearCliSolver::Cplex
            | ExternalLinearCliSolver::Xpress
            | ExternalLinearCliSolver::Lindo => ExternalLinearCliLicenseClass::Commercial,
        }
    }

    pub fn notes(self) -> &'static str {
        match self {
            ExternalLinearCliSolver::Highs => {
                "Modern open-source LP/MIP solver; preferred open-source CLI for same-input LP/MIP cross-checks."
            }
            ExternalLinearCliSolver::Glpk => {
                "Open-source LP/MIP solver via glpsol; useful for independent GNU MathProg/LP sanity checks."
            }
            ExternalLinearCliSolver::Scip => {
                "Powerful open-source/source-available MIP/constraint solver; strong cross-checker for hard integer models."
            }
            ExternalLinearCliSolver::Cbc => {
                "COIN-OR open-source MIP solver; useful legacy and regression cross-check target."
            }
            ExternalLinearCliSolver::Clp => {
                "COIN-OR open-source LP solver; LP-only bridge target for simplex-style comparisons."
            }
            ExternalLinearCliSolver::Soplex => {
                "ZIB SoPlex LP solver; LP-only bridge target with floating-point and rational solve modes."
            }
            ExternalLinearCliSolver::QsoptEx => {
                "QSopt_ex exact rational LP solver; LP-only bridge target for certifying small LP optima."
            }
            ExternalLinearCliSolver::LpSolve => {
                "Open-source LP/MIP solver with a compact LP-file CLI; useful legacy and lightweight cross-check target."
            }
            ExternalLinearCliSolver::Gurobi => {
                "Commercial LP/MIP solver exposed only when installed locally and licensed."
            }
            ExternalLinearCliSolver::Cplex => {
                "Commercial IBM LP/MIP solver exposed only when installed locally and licensed."
            }
            ExternalLinearCliSolver::Xpress => {
                "Commercial FICO LP/MIP solver exposed only when installed locally and licensed."
            }
            ExternalLinearCliSolver::Lindo => {
                "Commercial LINDO Systems LP/MIP solver exposed only when installed locally and licensed."
            }
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
            ExternalLinearCliSolver::Soplex => &["soplex"],
            ExternalLinearCliSolver::QsoptEx => &["qsopt_ex", "qsopt-ex", "qsopt", "esolver"],
            ExternalLinearCliSolver::LpSolve => &["lp_solve", "lp-solve", "lpsolve"],
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
            ExternalLinearCliSolver::Highs => &[
                "HIGHS_CMD",
                "ORES_HIGHS_CMD",
                "ORES_HIGHS_BIN",
                "DES_HIGHS_BIN",
                "HIGHS_BIN",
            ],
            ExternalLinearCliSolver::Glpk => &[
                "GLPSOL_CMD",
                "GLPK_CMD",
                "ORES_GLPK_CMD",
                "ORES_GLPK_BIN",
                "DES_GLPK_BIN",
                "GLPK_BIN",
            ],
            ExternalLinearCliSolver::Scip => &[
                "SCIP_CMD",
                "ORES_SCIP_CMD",
                "ORES_SCIP_BIN",
                "DES_SCIP_BIN",
                "SCIP_BIN",
            ],
            ExternalLinearCliSolver::Cbc => &[
                "CBC_CMD",
                "ORES_CBC_CMD",
                "ORES_CBC_BIN",
                "DES_CBC_BIN",
                "CBC_BIN",
            ],
            ExternalLinearCliSolver::Clp => &[
                "CLP_CMD",
                "ORES_CLP_CMD",
                "ORES_CLP_BIN",
                "DES_CLP_BIN",
                "CLP_BIN",
            ],
            ExternalLinearCliSolver::Soplex => &[
                "SOPLEX_CMD",
                "ORES_SOPLEX_CMD",
                "ORES_SOPLEX_BIN",
                "DES_SOPLEX_BIN",
                "SOPLEX_BIN",
            ],
            ExternalLinearCliSolver::QsoptEx => &[
                "QSOPT_EX_CMD",
                "QSOPT_CMD",
                "ORES_QSOPT_EX_CMD",
                "ORES_QSOPT_EX_BIN",
                "DES_QSOPT_EX_BIN",
                "QSOPT_EX_BIN",
            ],
            ExternalLinearCliSolver::LpSolve => &[
                "LP_SOLVE_CMD",
                "LPSOLVE_CMD",
                "ORES_LP_SOLVE_CMD",
                "ORES_LPSOLVE_BIN",
                "DES_LPSOLVE_BIN",
                "LPSOLVE_BIN",
            ],
            ExternalLinearCliSolver::Gurobi => &[
                "GUROBI_CL_CMD",
                "GUROBI_CMD",
                "ORES_GUROBI_CMD",
                "ORES_GUROBI_BIN",
                "DES_GUROBI_BIN",
                "GUROBI_BIN",
            ],
            ExternalLinearCliSolver::Cplex => &[
                "CPLEX_CMD",
                "ORES_CPLEX_CMD",
                "ORES_CPLEX_BIN",
                "DES_CPLEX_BIN",
                "CPLEX_BIN",
            ],
            ExternalLinearCliSolver::Xpress => &[
                "XPRESS_CMD",
                "XPRESS_OPTIMIZER_CMD",
                "ORES_XPRESS_CMD",
                "ORES_XPRESS_BIN",
                "DES_XPRESS_BIN",
                "XPRESS_BIN",
            ],
            ExternalLinearCliSolver::Lindo => &[
                "RUNLINDO_CMD",
                "LINDO_CMD",
                "LINDOAPI_CMD",
                "ORES_LINDO_CMD",
                "ORES_LINDO_BIN",
                "DES_LINDO_BIN",
                "LINDO_BIN",
            ],
        }
    }

    /// Environment variables that may point at a solver installation directory.
    ///
    /// The bridge searches each directory for common CLI locations such as the
    /// directory itself, `bin/`, and one vendor/platform subdirectory under
    /// `bin/`.
    pub fn command_dir_env_vars(self) -> &'static [&'static str] {
        match self {
            ExternalLinearCliSolver::Highs => &["HIGHS_DIR", "HIGHS_HOME"],
            ExternalLinearCliSolver::Glpk => &["GLPK_DIR", "GLPK_HOME"],
            ExternalLinearCliSolver::Scip => &["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
            ExternalLinearCliSolver::Cbc => &["CBC_DIR", "CBC_HOME", "COINOR_DIR", "COINOR_HOME"],
            ExternalLinearCliSolver::Clp => &["CLP_DIR", "CLP_HOME", "COINOR_DIR", "COINOR_HOME"],
            ExternalLinearCliSolver::Soplex => &["SOPLEX_DIR", "SOPLEX_HOME"],
            ExternalLinearCliSolver::QsoptEx => {
                &["QSOPT_EX_DIR", "QSOPT_EX_HOME", "QSOPT_DIR", "QSOPT_HOME"]
            }
            ExternalLinearCliSolver::LpSolve => &[
                "LP_SOLVE_DIR",
                "LPSOLVE_DIR",
                "LP_SOLVE_HOME",
                "LPSOLVE_HOME",
            ],
            ExternalLinearCliSolver::Gurobi => &["GUROBI_HOME"],
            ExternalLinearCliSolver::Cplex => &["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
            ExternalLinearCliSolver::Xpress => &["XPRESSDIR", "XPRESS_DIR", "XPRESS_HOME"],
            ExternalLinearCliSolver::Lindo => {
                &["LINDO_HOME", "LINDO_DIR", "LINDOAPI_HOME", "LINDOAPI_DIR"]
            }
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
                    | ExternalLinearCliSolver::Soplex
                    | ExternalLinearCliSolver::QsoptEx
                    | ExternalLinearCliSolver::LpSolve
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
                    | ExternalLinearCliSolver::LpSolve
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
            ExternalLinearCliSolver::Soplex,
            ExternalLinearCliSolver::QsoptEx,
            ExternalLinearCliSolver::LpSolve,
        ]
    }

    /// Installed open-source CLIs that can solve MIPs through this bridge.
    pub fn open_source_mip() -> &'static [ExternalLinearCliSolver] {
        &[
            ExternalLinearCliSolver::Highs,
            ExternalLinearCliSolver::Glpk,
            ExternalLinearCliSolver::Scip,
            ExternalLinearCliSolver::Cbc,
            ExternalLinearCliSolver::LpSolve,
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

    pub fn spec(self) -> ExternalLinearCliSolverSpec {
        ExternalLinearCliSolverSpec {
            solver: self,
            id: self.as_str(),
            display_name: self.display_name(),
            license_class: self.license_class(),
            command_aliases: self.command_aliases(),
            command_env_vars: self.command_env_vars(),
            command_dir_env_vars: self.command_dir_env_vars(),
            supports_lp: self.supports_kind(ExternalLinearCliKind::Lp),
            supports_mip: self.supports_kind(ExternalLinearCliKind::Mip),
            notes: self.notes(),
        }
    }
}

pub fn external_linear_cli_solver_specs() -> Vec<ExternalLinearCliSolverSpec> {
    ExternalLinearCliSolver::all()
        .iter()
        .copied()
        .map(ExternalLinearCliSolver::spec)
        .collect()
}

pub fn external_linear_cli_solver_manifest() -> Value {
    Value::Array(
        external_linear_cli_solver_specs()
            .into_iter()
            .map(|spec| {
                json!({
                    "id": spec.id,
                    "displayName": spec.display_name,
                    "licenseClass": spec.license_class.as_str(),
                    "commandAliases": spec.command_aliases,
                    "commandEnvVars": spec.command_env_vars,
                    "commandDirEnvVars": spec.command_dir_env_vars,
                    "supportsLp": spec.supports_lp,
                    "supportsMip": spec.supports_mip,
                    "notes": spec.notes,
                })
            })
            .collect(),
    )
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
    /// Optional MIP node limit. Ignored for LP probes and solves.
    pub node_limit: Option<usize>,
    /// Model file format used by the bridge. Defaults to CPLEX LP syntax.
    pub model_format: ExternalLinearCliModelFormat,
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
            model_format: ExternalLinearCliModelFormat::CplexLp,
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
            command_path: None,
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
    /// Lexicographic objective values in priority order, when a multi-objective MIP is solved.
    pub objective_values: Option<Vec<f64>>,
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
    #[serde(rename = "objectiveValues")]
    objective_values: Option<Vec<f64>>,
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

#[derive(Clone, Debug, PartialEq)]
struct HighsCliModel {
    sense: Sense,
    c: Vec<f64>,
    le_rows: Vec<Vec<f64>>,
    le_rhs: Vec<f64>,
    eq_rows: Vec<Vec<f64>>,
    eq_rhs: Vec<f64>,
    lbs: Vec<Option<f64>>,
    ubs: Vec<Option<f64>>,
    integer_vars: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
struct HighsParsedSolution {
    status: String,
    x: Vec<f64>,
    dual_ub: Option<Vec<f64>>,
    dual_eq: Option<Vec<f64>>,
    reduced_costs: Option<Vec<f64>>,
    var_basis: Option<Vec<String>>,
    row_basis: Option<Vec<String>>,
}

struct ExternalLinearCliTempDir {
    path: PathBuf,
}

impl ExternalLinearCliTempDir {
    fn new(prefix: &str) -> io::Result<Self> {
        let base = std::env::temp_dir();
        for attempt in 0..100 {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0);
            let path = base.join(format!("{prefix}-{}-{nanos}-{attempt}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(err),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("could not create unique temporary directory for {prefix}"),
        ))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ExternalLinearCliTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
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

/// Serialize a lower-bounded IP/MIP wrapper into the CLI bridge contract.
pub fn lower_bounded_ipmip_problem_to_cli_json(problem: &LowerBoundedIPMIPProblem) -> Value {
    let mut payload = ipmip_problem_to_cli_json(&problem.base);
    insert_cli_field(&mut payload, "lb", f64_vec(&problem.lb));
    payload
}

/// Serialize a general-row IP/MIP wrapper into the CLI bridge contract.
pub fn general_linear_ipmip_problem_to_cli_json(problem: &GeneralLinearIPMIPProblem) -> Value {
    let mut payload = ipmip_problem_to_cli_json(&problem.base);
    insert_cli_field(
        &mut payload,
        "linear_constraints",
        linear_constraints_json(&problem.linear_constraints),
    );
    payload
}

/// Serialize an indicator-constraint IP/MIP wrapper into the CLI bridge contract.
pub fn indicator_ipmip_problem_to_cli_json(problem: &IndicatorIPMIPProblem) -> Value {
    let mut payload = ipmip_problem_to_cli_json(&problem.base);
    insert_cli_field(
        &mut payload,
        "indicators",
        indicator_constraints_json(&problem.indicators),
    );
    payload
}

/// Serialize an SOS IP/MIP wrapper into the CLI bridge contract.
pub fn sos_ipmip_problem_to_cli_json(problem: &SosIPMIPProblem) -> Value {
    let mut payload = ipmip_problem_to_cli_json(&problem.base);
    insert_cli_field(&mut payload, "sos", sos_sets_json(&problem.sos));
    payload
}

/// Serialize a semi-variable IP/MIP wrapper into the CLI bridge contract.
pub fn semi_ipmip_problem_to_cli_json(problem: &SemiIPMIPProblem) -> Value {
    let mut payload = ipmip_problem_to_cli_json(&problem.base);
    insert_cli_field(
        &mut payload,
        "semi_variables",
        semi_variables_json(&problem.semi_variables),
    );
    payload
}

/// Serialize a piecewise-linear IP/MIP wrapper into the CLI bridge contract.
pub fn pwl_ipmip_problem_to_cli_json(problem: &PwlIPMIPProblem) -> Value {
    let mut payload = ipmip_problem_to_cli_json(&problem.base);
    insert_cli_field(&mut payload, "pwl", pwl_constraints_json(&problem.pwl));
    payload
}

/// Serialize a quadratic-objective IP/MIP wrapper into the CLI bridge contract.
pub fn quadratic_objective_ipmip_problem_to_cli_json(
    problem: &QuadraticObjectiveIPMIPProblem,
) -> Value {
    let mut payload = ipmip_problem_to_cli_json(&problem.base);
    insert_cli_field(&mut payload, "lb", opt_plain_vec_f64(problem.lb.as_ref()));
    insert_cli_field(
        &mut payload,
        "quadratic_objective",
        quadratic_objective_json(&problem.quadratic_objective),
    );
    payload
}

/// Serialize a source-level IP/MIP model with general constraints into the CLI bridge contract.
pub fn source_ipmip_problem_to_cli_json(problem: &SourceIPMIPProblem) -> Value {
    let base = &problem.base;
    json!({
        "sense": base.sense.as_str(),
        "c": f64_vec(&base.c),
        "a": matrix_f64(&base.a),
        "b": f64_vec(&base.b),
        "integer_vars": base.integer_vars,
        "lb": opt_plain_vec_f64(problem.lb.as_ref()),
        "ub": opt_plain_vec_f64(base.ub.as_ref()),
        "var_names": option_strings(base.var_names.as_ref()),
        "con_names": option_strings(base.con_names.as_ref()),
        "lazy_constraints": lazy_constraints_json(base.lazy_constraints.as_ref()),
        "linear_constraints": linear_constraints_json(&problem.linear_constraints),
        "indicators": indicator_constraints_json(&problem.indicators),
        "sos": sos_sets_json(&problem.sos),
        "semi_variables": semi_variables_json(&problem.semi_variables),
        "pwl": pwl_constraints_json(&problem.pwl),
        "abs": Value::Array(
            problem
                .abs
                .iter()
                .map(|constraint| {
                    json!({
                        "arg_var": constraint.arg_var,
                        "target_var": constraint.target_var,
                        "name": &constraint.name,
                    })
                })
                .collect(),
        ),
        "maximums": Value::Array(
            problem
                .maximums
                .iter()
                .map(|constraint| {
                    json!({
                        "target_var": constraint.target_var,
                        "arg_vars": &constraint.arg_vars,
                        "constant": opt_f64_value(constraint.constant),
                        "name": &constraint.name,
                    })
                })
                .collect(),
        ),
        "minimums": Value::Array(
            problem
                .minimums
                .iter()
                .map(|constraint| {
                    json!({
                        "target_var": constraint.target_var,
                        "arg_vars": &constraint.arg_vars,
                        "constant": opt_f64_value(constraint.constant),
                        "name": &constraint.name,
                    })
                })
                .collect(),
        ),
        "logical": Value::Array(
            problem
                .logical
                .iter()
                .map(|constraint| {
                    json!({
                        "kind": constraint.kind.as_str(),
                        "target_var": constraint.target_var,
                        "arg_vars": &constraint.arg_vars,
                        "name": &constraint.name,
                    })
                })
                .collect(),
        ),
        "l1_norms": Value::Array(
            problem
                .l1_norms
                .iter()
                .map(|constraint| {
                    json!({
                        "target_var": constraint.target_var,
                        "arg_vars": &constraint.arg_vars,
                        "name": &constraint.name,
                    })
                })
                .collect(),
        ),
        "linf_norms": Value::Array(
            problem
                .linf_norms
                .iter()
                .map(|constraint| {
                    json!({
                        "target_var": constraint.target_var,
                        "arg_vars": &constraint.arg_vars,
                        "name": &constraint.name,
                    })
                })
                .collect(),
        ),
        "products": Value::Array(
            problem
                .products
                .iter()
                .map(|constraint| {
                    json!({
                        "target_var": constraint.target_var,
                        "x_var": constraint.x_var,
                        "y_var": constraint.y_var,
                        "name": &constraint.name,
                    })
                })
                .collect(),
        ),
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

/// Serialize a lexicographic multi-objective MIP into the CLI bridge contract.
pub fn multi_objective_ipmip_problem_to_cli_json(problem: &MultiObjectiveIPMIPProblem) -> Value {
    let mut payload = ipmip_problem_to_cli_json(&problem.base);
    if let Value::Object(ref mut object) = payload {
        object.insert(
            "multi_objectives".to_string(),
            Value::Array(
                problem
                    .objectives
                    .iter()
                    .map(|objective| {
                        json!({
                            "sense": objective.sense.as_str(),
                            "c": f64_vec(&objective.c),
                            "name": objective.name.as_deref(),
                        })
                    })
                    .collect(),
            ),
        );
    }
    payload
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

/// Solve a lower-bounded IP/MIP through a locally installed CLI solver.
pub fn solve_lower_bounded_ipmip_with_external_cli(
    problem: &LowerBoundedIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        lower_bounded_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Solve a general-row IP/MIP through a locally installed CLI solver.
pub fn solve_general_linear_ipmip_with_external_cli(
    problem: &GeneralLinearIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        general_linear_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Solve an indicator-constraint IP/MIP through a locally installed CLI solver.
pub fn solve_indicator_ipmip_with_external_cli(
    problem: &IndicatorIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        indicator_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Solve an SOS IP/MIP through a locally installed CLI solver.
pub fn solve_sos_ipmip_with_external_cli(
    problem: &SosIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        sos_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Solve a semi-variable IP/MIP through a locally installed CLI solver.
pub fn solve_semi_ipmip_with_external_cli(
    problem: &SemiIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        semi_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Solve a piecewise-linear IP/MIP through a locally installed CLI solver.
pub fn solve_pwl_ipmip_with_external_cli(
    problem: &PwlIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        pwl_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Solve a quadratic-objective IP/MIP through a locally installed CLI solver.
pub fn solve_quadratic_objective_ipmip_with_external_cli(
    problem: &QuadraticObjectiveIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        quadratic_objective_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Solve a lexicographic multi-objective MIP through a locally installed CLI solver.
pub fn solve_multi_objective_ipmip_with_external_cli(
    problem: &MultiObjectiveIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        multi_objective_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Solve a source-level IP/MIP through a locally installed command-line solver.
pub fn solve_source_ipmip_with_external_cli(
    problem: &SourceIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        source_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

/// Return the first executable-like command path found for a solver's aliases.
pub fn external_linear_cli_command(solver: ExternalLinearCliSolver) -> Option<PathBuf> {
    find_first_command(
        solver.command_env_vars(),
        solver.command_dir_env_vars(),
        solver.command_aliases(),
    )
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
            solver_version: None,
            elapsed_ms: elapsed_ms(t0),
            message: format!(
                "no executable found via command env vars [{}], install dir env vars [{}], or PATH aliases [{}]",
                solver.command_env_vars().join(", "),
                solver.command_dir_env_vars().join(", "),
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
    if let Some(solution) = solve_highs_cli_json_direct(kind, &problem_json, opts, t0) {
        return solution;
    }
    if let Some(solution) = solve_glpk_cli_json_direct(kind, &problem_json, opts, t0) {
        return solution;
    }
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
    let max_nodes = opts
        .max_nodes
        .or_else(|| opts.node_limit.map(|limit| limit as u64));
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
        .arg("--model-format")
        .arg(opts.model_format.as_str())
        .arg("--time-limit")
        .arg(time_limit.to_string());
    if let Some(max_nodes) = max_nodes {
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
    if let Some(command_path) = external_linear_cli_command_with_options(opts.solver, opts) {
        command.env(solver_command_env_var(opts.solver), command_path);
    }
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

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
            solver_version: raw.solver_version,
            x: raw.x,
            objective: raw.objective,
            objective_values: raw.objective_values,
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

fn solve_highs_cli_json_direct(
    kind: ExternalLinearCliKind,
    problem_json: &Value,
    opts: &ExternalLinearCliOptions,
    t0: Instant,
) -> Option<ExternalLinearCliSolution> {
    if opts.solver != ExternalLinearCliSolver::Highs {
        return None;
    }
    if opts.solution_pool_size.is_some() {
        return None;
    }

    let solver = "highs:cli".to_string();
    let model = match highs_model_from_cli_json(kind, problem_json) {
        Ok(Some(model)) => model,
        Ok(None) => return None,
        Err(message) => {
            return Some(external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                solver,
                message,
                elapsed_ms(t0),
            ));
        }
    };
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Highs, opts)
    else {
        return Some(external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            solver,
            "highs executable not found".to_string(),
            elapsed_ms(t0),
        ));
    };

    let temp_dir = match ExternalLinearCliTempDir::new("ores-highs-cli") {
        Ok(temp_dir) => temp_dir,
        Err(err) => {
            return Some(external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                solver,
                format!("failed to create temporary HiGHS workspace: {err}"),
                elapsed_ms(t0),
            ));
        }
    };
    let model_extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = temp_dir.path().join(format!("model.{model_extension}"));
    let solution_path = temp_dir.path().join("highs.sol");
    let model_text = highs_model_to_string(&model, opts.model_format);
    if let Err(err) = fs::write(&model_path, model_text) {
        return Some(external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            solver,
            format!("failed to write HiGHS model file: {err}"),
            elapsed_ms(t0),
        ));
    }

    let time_limit = normalized_time_limit(opts.time_limit_secs);
    let mut command = Command::new(&command_path);
    command
        .arg("--model_file")
        .arg(&model_path)
        .arg("--solution_file")
        .arg(&solution_path)
        .arg("--time_limit")
        .arg(time_limit.to_string());

    let mut mip_start_objective = None;
    if kind == ExternalLinearCliKind::Mip {
        if let Some(mip_start) = opts.mip_start.as_deref() {
            let mip_start = match normalized_highs_mip_start(mip_start, model.c.len()) {
                Ok(mip_start) => mip_start,
                Err(message) => {
                    return Some(external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        solver,
                        message,
                        elapsed_ms(t0),
                    ));
                }
            };
            let objective = dot_f64(&model.c, &mip_start);
            let start_path = temp_dir.path().join("highs-start.sol");
            if let Err(err) = fs::write(&start_path, highs_mip_start_string(&mip_start, objective))
            {
                return Some(external_cli_failure(
                    ExternalLinearCliStatus::NumericalError,
                    solver,
                    format!("failed to write HiGHS MIP start file: {err}"),
                    elapsed_ms(t0),
                ));
            }
            command.arg("--read_solution_file").arg(start_path);
            mip_start_objective = Some(objective);
        }
    }

    if let Some(options_text) = highs_options_file_text(kind, opts) {
        let options_path = temp_dir.path().join("highs.options");
        if let Err(err) = fs::write(&options_path, options_text) {
            return Some(external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                solver,
                format!("failed to write HiGHS options file: {err}"),
                elapsed_ms(t0),
            ));
        }
        command.arg("--options_file").arg(options_path);
    }
    if kind == ExternalLinearCliKind::Lp {
        if let Some(lp_algorithm) = opts.lp_algorithm {
            command.arg("--solver").arg(lp_algorithm.as_str());
        }
    }
    if let Some(random_seed) = opts.random_seed {
        command.arg("--random_seed").arg(random_seed.to_string());
    }
    if let Some(presolve) = opts.presolve {
        command
            .arg("--presolve")
            .arg(if presolve == ExternalLinearCliPresolve::Auto {
                "choose"
            } else {
                presolve.as_str()
            });
    }

    let output = match command
        .current_dir(temp_dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            return Some(external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                solver,
                format!(
                    "failed to start HiGHS command '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            ));
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let solver_version = highs_solver_version_from_output(&stdout, &stderr);

    if !solution_path.exists() {
        let status = classify_highs_status("", &stdout, &stderr);
        let message = nonempty_trimmed(&stderr).unwrap_or_else(|| stdout.trim().to_string());
        let status = if matches!(
            status,
            ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
        ) {
            status
        } else {
            ExternalLinearCliStatus::Unavailable
        };
        let mut solution = external_cli_failure(status, solver, message, elapsed_ms(t0));
        solution.solver_version = solver_version;
        return Some(solution);
    }

    let parsed = match parse_highs_solution_file(
        &solution_path,
        model.c.len(),
        model.le_rows.len(),
        model.eq_rows.len(),
    ) {
        Ok(parsed) => parsed,
        Err(message) => {
            return Some(external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                solver,
                message,
                elapsed_ms(t0),
            ));
        }
    };
    let status = classify_highs_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let status = if matches!(
            status,
            ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
        ) {
            status
        } else {
            ExternalLinearCliStatus::Unavailable
        };
        let mut solution = external_cli_failure(status, solver, parsed.status, elapsed_ms(t0));
        solution.solver_version = solver_version;
        return Some(solution);
    }

    let objective = dot_f64(&model.c, &parsed.x);
    let mut solution = ExternalLinearCliSolution {
        status,
        solver,
        solver_version,
        x: parsed.x,
        objective: Some(objective),
        objective_values: None,
        lp_algorithm: highs_lp_algorithm_feedback(kind, opts.lp_algorithm, &stdout, &stderr),
        best_bound: None,
        solution_limit: None,
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: None,
        absolute_gap: None,
        objective_limit: highs_objective_limit_feedback(
            kind,
            opts.objective_limit,
            &stdout,
            &stderr,
        ),
        primal_feasibility_tolerance: normalized_tolerance(opts.primal_feasibility_tolerance),
        dual_feasibility_tolerance: normalized_tolerance(opts.dual_feasibility_tolerance),
        integer_feasibility_tolerance: if kind == ExternalLinearCliKind::Mip {
            normalized_tolerance(opts.integer_feasibility_tolerance)
        } else {
            None
        },
        nodes_explored: None,
        threads: highs_threads_feedback(opts.threads, &stdout, &stderr),
        random_seed: highs_random_seed_feedback(opts.random_seed, &stdout, &stderr),
        presolve: highs_presolve_feedback(opts.presolve, &stdout, &stderr),
        cuts: None,
        heuristics: None,
        branch_rule: None,
        branch_priorities_accepted: None,
        branch_priority_count: None,
        node_selection: None,
        mip_start_accepted: None,
        mip_start_objective: None,
        dual_ub: if kind == ExternalLinearCliKind::Lp {
            parsed.dual_ub
        } else {
            None
        },
        dual_eq: if kind == ExternalLinearCliKind::Lp {
            parsed.dual_eq
        } else {
            None
        },
        reduced_costs: if kind == ExternalLinearCliKind::Lp {
            parsed.reduced_costs
        } else {
            None
        },
        var_basis: if kind == ExternalLinearCliKind::Lp {
            parsed.var_basis
        } else {
            None
        },
        row_basis: if kind == ExternalLinearCliKind::Lp {
            parsed.row_basis
        } else {
            None
        },
        iterations: highs_lp_iterations(kind, &stdout, &stderr),
        elapsed_ms: elapsed_ms(t0),
        message: parsed.status,
    };
    if kind == ExternalLinearCliKind::Mip {
        apply_highs_mip_quality(&mut solution, objective, &stdout, &stderr);
        if opts.mip_start.is_some() {
            solution.mip_start_accepted = Some(highs_mip_start_accepted(&stdout, &stderr));
            solution.mip_start_objective = mip_start_objective;
        }
    }
    Some(solution)
}

fn highs_model_from_cli_json(
    kind: ExternalLinearCliKind,
    problem_json: &Value,
) -> Result<Option<HighsCliModel>, String> {
    match kind {
        ExternalLinearCliKind::Lp => highs_lp_model_from_cli_json(problem_json).map(Some),
        ExternalLinearCliKind::Mip => highs_mip_model_from_cli_json(problem_json),
    }
}

fn highs_lp_model_from_cli_json(problem_json: &Value) -> Result<HighsCliModel, String> {
    let lp = problem_json.get("lp").unwrap_or(problem_json);
    let object = lp
        .as_object()
        .ok_or_else(|| "LP payload must be a JSON object".to_string())?;
    let c = required_f64_array(object, "c")?;
    let n = c.len();
    let sense = parse_cli_sense(object.get("sense"))?;
    let mut le_rows = optional_f64_matrix(object, &["A_ub", "a_ub"])?;
    let mut le_rhs = optional_f64_array(object, &["b_ub"])?;
    let mut eq_rows = optional_f64_matrix(object, &["A_eq", "a_eq"])?;
    let mut eq_rhs = optional_f64_array(object, &["b_eq"])?;
    append_linear_constraint_rows(
        object.get("linear_constraints"),
        n,
        &mut le_rows,
        &mut le_rhs,
        &mut eq_rows,
        &mut eq_rhs,
    )?;
    let lbs = optional_bound_array(object.get("lb"), n, Some(0.0), false, "lb")?;
    let ubs = optional_bound_array(object.get("ub"), n, None, false, "ub")?;
    validate_highs_model_dimensions(n, &le_rows, &le_rhs, &eq_rows, &eq_rhs, &lbs, &ubs)?;
    Ok(HighsCliModel {
        sense,
        c,
        le_rows,
        le_rhs,
        eq_rows,
        eq_rhs,
        lbs,
        ubs,
        integer_vars: vec![false; n],
    })
}

fn highs_mip_model_from_cli_json(problem_json: &Value) -> Result<Option<HighsCliModel>, String> {
    let object = problem_json
        .as_object()
        .ok_or_else(|| "MIP payload must be a JSON object".to_string())?;
    for key in [
        "indicators",
        "sos",
        "semi_variables",
        "pwl",
        "quadratic_objective",
        "abs",
        "maximums",
        "minimums",
        "logical",
        "l1_norms",
        "linf_norms",
        "products",
        "multi_objectives",
    ] {
        if json_field_has_content(object.get(key)) {
            return Ok(None);
        }
    }

    let c = required_f64_array(object, "c")?;
    let n = c.len();
    let sense = parse_cli_sense(object.get("sense"))?;
    let mut le_rows = optional_f64_matrix(object, &["a"])?;
    let mut le_rhs = optional_f64_array(object, &["b"])?;
    let mut eq_rows = Vec::new();
    let mut eq_rhs = Vec::new();
    append_linear_constraint_rows(
        object.get("linear_constraints"),
        n,
        &mut le_rows,
        &mut le_rhs,
        &mut eq_rows,
        &mut eq_rhs,
    )?;
    append_lazy_constraint_rows(object.get("lazy_constraints"), n, &mut le_rows, &mut le_rhs)?;
    let lbs = optional_bound_array(object.get("lb"), n, Some(0.0), true, "lb")?;
    let ubs = optional_bound_array(object.get("ub"), n, None, false, "ub")?;
    let integer_vars = optional_bool_array(object.get("integer_vars"), n, false, "integer_vars")?;
    validate_highs_model_dimensions(n, &le_rows, &le_rhs, &eq_rows, &eq_rhs, &lbs, &ubs)?;
    Ok(Some(HighsCliModel {
        sense,
        c,
        le_rows,
        le_rhs,
        eq_rows,
        eq_rhs,
        lbs,
        ubs,
        integer_vars,
    }))
}

fn highs_model_to_string(
    model: &HighsCliModel,
    model_format: ExternalLinearCliModelFormat,
) -> String {
    match model_format {
        ExternalLinearCliModelFormat::CplexLp => cplex_lp_string(
            model.sense,
            &model.c,
            &model.le_rows,
            &model.le_rhs,
            &model.eq_rows,
            &model.eq_rhs,
            &model.lbs,
            &model.ubs,
            &model.integer_vars,
        ),
        ExternalLinearCliModelFormat::Mps => mps_string(
            model.sense,
            &model.c,
            &model.le_rows,
            &model.le_rhs,
            &model.eq_rows,
            &model.eq_rhs,
            &model.lbs,
            &model.ubs,
            &model.integer_vars,
        ),
    }
}

fn highs_options_file_text(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> Option<String> {
    let mut text = String::new();
    if let Some(threads) = opts.threads.filter(|threads| *threads > 0) {
        text.push_str(&format!("threads = {threads}\n"));
    }
    if let Some(tolerance) = normalized_tolerance(opts.primal_feasibility_tolerance) {
        text.push_str(&format!("primal_feasibility_tolerance = {tolerance:.17}\n"));
    }
    if let Some(tolerance) = normalized_tolerance(opts.dual_feasibility_tolerance) {
        text.push_str(&format!("dual_feasibility_tolerance = {tolerance:.17}\n"));
    }
    if kind == ExternalLinearCliKind::Mip {
        let max_nodes = opts
            .max_nodes
            .or_else(|| opts.node_limit.map(|limit| limit as u64))
            .filter(|nodes| *nodes > 0);
        if let Some(max_nodes) = max_nodes {
            text.push_str(&format!("mip_max_nodes = {max_nodes}\n"));
        }
        if let Some(gap) = normalized_relative_gap(opts.relative_gap) {
            text.push_str(&format!("mip_rel_gap = {gap:.17}\n"));
        }
        if let Some(gap) = normalized_absolute_gap(opts.absolute_gap) {
            text.push_str(&format!("mip_abs_gap = {gap:.17}\n"));
        }
        if let Some(limit) = normalized_objective_limit(opts.objective_limit) {
            text.push_str(&format!("objective_target = {limit:.17}\n"));
        }
        if let Some(tolerance) = normalized_tolerance(opts.integer_feasibility_tolerance) {
            text.push_str(&format!("mip_feasibility_tolerance = {tolerance:.17}\n"));
        }
    }
    (!text.is_empty()).then_some(text)
}

fn parse_cli_sense(value: Option<&Value>) -> Result<Sense, String> {
    let Some(value) = value else {
        return Ok(Sense::Max);
    };
    let Some(text) = value.as_str() else {
        return Err("sense must be a string".to_string());
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "max" | "maximize" => Ok(Sense::Max),
        "min" | "minimize" => Ok(Sense::Min),
        other => Err(format!("unknown objective sense '{other}'")),
    }
}

fn required_f64_array(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<f64>, String> {
    let Some(value) = object.get(key) else {
        return Err(format!("missing required array '{key}'"));
    };
    f64_array_from_value(value, key)
}

fn optional_f64_array(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Result<Vec<f64>, String> {
    for key in keys {
        if let Some(value) = object.get(*key) {
            if value.is_null() {
                return Ok(Vec::new());
            }
            return f64_array_from_value(value, key);
        }
    }
    Ok(Vec::new())
}

fn optional_f64_matrix(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Result<Vec<Vec<f64>>, String> {
    for key in keys {
        if let Some(value) = object.get(*key) {
            if value.is_null() {
                return Ok(Vec::new());
            }
            let Some(rows) = value.as_array() else {
                return Err(format!("{key} must be an array of rows"));
            };
            return rows
                .iter()
                .enumerate()
                .map(|(idx, row)| f64_array_from_value(row, &format!("{key}[{idx}]")))
                .collect();
        }
    }
    Ok(Vec::new())
}

fn f64_array_from_value(value: &Value, key: &str) -> Result<Vec<f64>, String> {
    let Some(values) = value.as_array() else {
        return Err(format!("{key} must be an array"));
    };
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("{key}[{idx}] must be a finite number"))
        })
        .collect()
}

fn optional_bound_array(
    value: Option<&Value>,
    n: usize,
    default: Option<f64>,
    null_means_default: bool,
    name: &str,
) -> Result<Vec<Option<f64>>, String> {
    let Some(value) = value else {
        return Ok(vec![default; n]);
    };
    if value.is_null() {
        return Ok(vec![default; n]);
    }
    let Some(values) = value.as_array() else {
        return Err(format!("{name} must be an array"));
    };
    if values.len() != n {
        return Err(format!(
            "{name} length {} does not match variable count {n}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if value.is_null() {
                return Ok(if null_means_default { default } else { None });
            }
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .map(Some)
                .ok_or_else(|| format!("{name}[{idx}] must be finite or null"))
        })
        .collect()
}

fn optional_bool_array(
    value: Option<&Value>,
    n: usize,
    default: bool,
    name: &str,
) -> Result<Vec<bool>, String> {
    let Some(value) = value else {
        return Ok(vec![default; n]);
    };
    if value.is_null() {
        return Ok(vec![default; n]);
    }
    let Some(values) = value.as_array() else {
        return Err(format!("{name} must be an array"));
    };
    if values.len() != n {
        return Err(format!(
            "{name} length {} does not match variable count {n}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            value
                .as_bool()
                .ok_or_else(|| format!("{name}[{idx}] must be a boolean"))
        })
        .collect()
}

fn append_linear_constraint_rows(
    value: Option<&Value>,
    n: usize,
    le_rows: &mut Vec<Vec<f64>>,
    le_rhs: &mut Vec<f64>,
    eq_rows: &mut Vec<Vec<f64>>,
    eq_rhs: &mut Vec<f64>,
) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() {
        return Ok(());
    }
    let Some(rows) = value.as_array() else {
        return Err("linear_constraints must be an array".to_string());
    };
    for (idx, row_value) in rows.iter().enumerate() {
        let Some(row_object) = row_value.as_object() else {
            return Err(format!("linear_constraints[{idx}] must be an object"));
        };
        let row = required_f64_array(row_object, "coefs")?;
        if row.len() != n {
            return Err(format!(
                "linear_constraints[{idx}] coefficient length {} does not match variable count {n}",
                row.len()
            ));
        }
        let lower = optional_f64_field(
            row_object.get("lower"),
            &format!("linear_constraints[{idx}].lower"),
        )?;
        let upper = optional_f64_field(
            row_object.get("upper"),
            &format!("linear_constraints[{idx}].upper"),
        )?;
        if lower.is_none() && upper.is_none() {
            return Err(format!(
                "linear_constraints[{idx}] needs a lower or upper bound"
            ));
        }
        if let (Some(lower), Some(upper)) = (lower, upper) {
            if lower > upper + 1.0e-9 {
                return Err(format!("linear_constraints[{idx}] lower exceeds upper"));
            }
            if (lower - upper).abs() <= 1.0e-9 {
                eq_rows.push(row);
                eq_rhs.push(upper);
                continue;
            }
        }
        if let Some(upper) = upper {
            le_rows.push(row.clone());
            le_rhs.push(upper);
        }
        if let Some(lower) = lower {
            le_rows.push(row.iter().map(|value| -*value).collect());
            le_rhs.push(-lower);
        }
    }
    Ok(())
}

fn append_lazy_constraint_rows(
    value: Option<&Value>,
    n: usize,
    le_rows: &mut Vec<Vec<f64>>,
    le_rhs: &mut Vec<f64>,
) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() {
        return Ok(());
    }
    let Some(rows) = value.as_array() else {
        return Err("lazy_constraints must be an array".to_string());
    };
    for (idx, row_value) in rows.iter().enumerate() {
        let Some(row_object) = row_value.as_object() else {
            return Err(format!("lazy_constraints[{idx}] must be an object"));
        };
        let row = required_f64_array(row_object, "coefs")?;
        if row.len() != n {
            return Err(format!(
                "lazy_constraints[{idx}] coefficient length {} does not match variable count {n}",
                row.len()
            ));
        }
        let rhs = required_f64_field(
            row_object.get("rhs"),
            &format!("lazy_constraints[{idx}].rhs"),
        )?;
        le_rows.push(row);
        le_rhs.push(rhs);
    }
    Ok(())
}

fn optional_f64_field(value: Option<&Value>, name: &str) -> Result<Option<f64>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    required_f64_field(Some(value), name).map(Some)
}

fn required_f64_field(value: Option<&Value>, name: &str) -> Result<f64, String> {
    let Some(value) = value else {
        return Err(format!("missing required number '{name}'"));
    };
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{name} must be a finite number"))
}

fn validate_highs_model_dimensions(
    n: usize,
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    lbs: &[Option<f64>],
    ubs: &[Option<f64>],
) -> Result<(), String> {
    if le_rows.len() != le_rhs.len() {
        return Err("inequality matrix/RHS length mismatch".to_string());
    }
    if eq_rows.len() != eq_rhs.len() {
        return Err("equality matrix/RHS length mismatch".to_string());
    }
    if lbs.len() != n || ubs.len() != n {
        return Err("bound vector length mismatch".to_string());
    }
    for row in le_rows.iter().chain(eq_rows) {
        if row.len() != n {
            return Err("constraint row length mismatch".to_string());
        }
    }
    Ok(())
}

fn json_field_has_content(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Array(values)) => !values.is_empty(),
        Some(Value::Object(values)) => !values.is_empty(),
        Some(Value::String(value)) => !value.is_empty(),
        Some(_) => true,
    }
}

fn normalized_highs_mip_start(start: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if start.len() != n {
        return Err(format!(
            "mip_start length {} does not match variable count {n}",
            start.len()
        ));
    }
    if start.iter().any(|value| !value.is_finite()) {
        return Err("mip_start values must be finite".to_string());
    }
    Ok(start.to_vec())
}

fn highs_mip_start_string(start: &[f64], objective: f64) -> String {
    let mut out = String::new();
    out.push_str("Model status\nUnknown\n\n");
    out.push_str("# Primal solution values\nFeasible\n");
    out.push_str(&format!("Objective {objective:.17}\n"));
    out.push_str(&format!("# Columns {}\n", start.len()));
    for (idx, value) in start.iter().enumerate() {
        out.push_str(&format!("x{idx} {value:.17}\n"));
    }
    out.push_str(
        "# Rows 0\n\n# Dual solution values\nNone\n\n# Basis\nHiGHS_basis_file v2\nNone\n",
    );
    out
}

fn parse_highs_solution_file(
    path: &Path,
    n: usize,
    le_count: usize,
    eq_count: usize,
) -> Result<HighsParsedSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read HiGHS solution file '{}': {err}",
            path.display()
        )
    })?;
    Ok(parse_highs_solution_text(&text, n, le_count, eq_count))
}

fn parse_highs_solution_text(
    text: &str,
    n: usize,
    le_count: usize,
    eq_count: usize,
) -> HighsParsedSolution {
    let lines = text.lines().map(str::trim).collect::<Vec<_>>();
    let mut x = vec![0.0; n];
    let mut status = "unknown".to_string();
    let mut dual_columns = vec![None; n];
    let mut dual_rows: HashMap<String, f64> = HashMap::new();
    let mut var_basis: Vec<Option<String>> = vec![None; n];
    let mut row_basis: HashMap<String, String> = HashMap::new();
    let mut section: Option<&str> = None;
    let mut block: Option<&str> = None;
    let mut remaining = 0usize;

    for (idx, line) in lines.iter().enumerate() {
        if *line == "Model status" {
            if let Some(next) = lines.get(idx + 1) {
                status = next.to_ascii_lowercase();
            }
        }
        match *line {
            "# Primal solution values" => {
                section = Some("primal");
                block = None;
                continue;
            }
            "# Dual solution values" => {
                section = Some("dual");
                block = None;
                continue;
            }
            line if line.starts_with("# Basis") => {
                section = Some("basis");
                block = None;
                continue;
            }
            _ => {}
        }
        if section.is_none() {
            continue;
        }
        if line.starts_with("# Columns") {
            block = Some("columns");
            remaining = line
                .split_whitespace()
                .nth(2)
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            continue;
        }
        if line.starts_with("# Rows") {
            block = Some("rows");
            remaining = line
                .split_whitespace()
                .nth(2)
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            continue;
        }
        if line.is_empty() || line.starts_with('#') || block.is_none() {
            continue;
        }
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() >= 2 {
            match (section, block) {
                (Some("primal"), Some("columns")) => {
                    if let Some(var_idx) = parse_x_index(parts[0], n) {
                        if let Ok(value) = parts[1].parse::<f64>() {
                            x[var_idx] = value;
                        }
                    }
                }
                (Some("dual"), Some("columns")) => {
                    if let Some(var_idx) = parse_x_index(parts[0], n) {
                        if let Ok(value) = parts[1].parse::<f64>() {
                            dual_columns[var_idx] = Some(value);
                        }
                    }
                }
                (Some("basis"), Some("columns")) => {
                    if let Some(var_idx) = parse_x_index(parts[0], n) {
                        if let Some(status) = basis_status_from_token(parts[1]) {
                            var_basis[var_idx] = Some(status.to_string());
                        }
                    }
                }
                (Some("dual"), Some("rows")) => {
                    if let Ok(value) = parts[1].parse::<f64>() {
                        dual_rows.insert(parts[0].to_string(), value);
                    }
                }
                (Some("basis"), Some("rows")) => {
                    if let Some(status) = basis_status_from_token(parts[1]) {
                        row_basis.insert(parts[0].to_string(), status.to_string());
                    }
                }
                _ => {}
            }
        }
        remaining = remaining.saturating_sub(1);
        if remaining == 0 {
            block = None;
        }
    }

    let reduced_costs = dual_columns.into_iter().collect::<Option<Vec<_>>>();
    let dual_ub = collect_named_row_values(&dual_rows, "c", le_count);
    let dual_eq = collect_named_row_values(&dual_rows, "e", eq_count);
    let var_basis = var_basis.into_iter().collect::<Option<Vec<_>>>();
    let mut row_statuses = Vec::with_capacity(le_count + eq_count);
    for i in 0..le_count {
        row_statuses.push(row_basis.get(&format!("c{i}")).cloned());
    }
    for i in 0..eq_count {
        row_statuses.push(row_basis.get(&format!("e{i}")).cloned());
    }
    let row_basis = row_statuses.into_iter().collect::<Option<Vec<_>>>();

    HighsParsedSolution {
        status,
        x,
        dual_ub,
        dual_eq,
        reduced_costs,
        var_basis,
        row_basis,
    }
}

fn parse_x_index(name: &str, n: usize) -> Option<usize> {
    let idx = name.strip_prefix('x')?.parse::<usize>().ok()?;
    (idx < n).then_some(idx)
}

fn collect_named_row_values(
    values: &HashMap<String, f64>,
    prefix: &str,
    count: usize,
) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        out.push(*values.get(&format!("{prefix}{idx}"))?);
    }
    Some(out)
}

fn basis_status_from_token(token: &str) -> Option<&'static str> {
    match token.trim().to_ascii_lowercase().as_str() {
        "0" | "lower" | "at_lower" => Some("at_lower"),
        "1" | "basic" => Some("basic"),
        "2" | "upper" | "at_upper" => Some("at_upper"),
        "3" | "zero" => Some("zero"),
        "4" | "nonbasic" => Some("nonbasic"),
        "b" | "bs" => Some("basic"),
        "l" | "nl" => Some("at_lower"),
        "u" | "nu" => Some("at_upper"),
        "f" | "nf" | "free" => Some("free"),
        "s" | "ns" | "fixed" => Some("fixed"),
        "superbasic" => Some("superbasic"),
        _ => None,
    }
}

fn classify_highs_status(status: &str, stdout: &str, stderr: &str) -> ExternalLinearCliStatus {
    let parsed = status.to_ascii_lowercase();
    if parsed.contains("primal infeasible")
        || (parsed.contains("infeasible") && !parsed.contains("dual"))
    {
        return ExternalLinearCliStatus::Infeasible;
    }
    if parsed.contains("dual infeasible") || parsed.contains("unbounded") {
        return ExternalLinearCliStatus::Unbounded;
    }
    if parsed.contains("optimal") {
        return ExternalLinearCliStatus::Optimal;
    }
    if parsed.contains("feasible") || parsed.contains("solution limit") {
        return ExternalLinearCliStatus::Feasible;
    }

    let text = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    if [
        "no primal feasible",
        "primal infeasible",
        "linear relaxation infeasible",
        "no feasible solution",
        "no solution exists",
        "integer infeasible",
        "problem has no feasible",
    ]
    .iter()
    .any(|marker| text.contains(marker))
    {
        return ExternalLinearCliStatus::Infeasible;
    }
    if [
        "has unbounded solution",
        "linear relaxation unbounded",
        "dual infeasible",
        "unbounded",
    ]
    .iter()
    .any(|marker| text.contains(marker))
    {
        return ExternalLinearCliStatus::Unbounded;
    }
    if [
        "stopped on solution limit",
        "solution limit reached",
        "exiting on maximum solutions",
        "partial search - best objective",
        "integer solution of",
        "feasibility pump exiting with objective",
    ]
    .iter()
    .any(|marker| text.contains(marker))
    {
        return ExternalLinearCliStatus::Feasible;
    }
    ExternalLinearCliStatus::Unknown
}

fn highs_solver_version_from_output(stdout: &str, stderr: &str) -> Option<String> {
    for line in stdout.lines().chain(stderr.lines()) {
        let line = line.trim();
        for prefix in ["Running HiGHS ", "HiGHS version "] {
            if let Some(rest) = line.strip_prefix(prefix) {
                if let Some(version) = rest.split_whitespace().next() {
                    return Some(format!("HiGHS {version}"));
                }
            }
        }
    }
    None
}

fn highs_lp_iterations(kind: ExternalLinearCliKind, stdout: &str, stderr: &str) -> Option<u64> {
    if kind != ExternalLinearCliKind::Lp {
        return None;
    }
    for line in stdout.lines().chain(stderr.lines()) {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("simplex") && lowered.contains("iterations") {
            if let Some(value) = first_float_after_colon(stripped) {
                return (value >= 0.0).then_some(value.round() as u64);
            }
        }
    }
    None
}

fn apply_highs_mip_quality(
    solution: &mut ExternalLinearCliSolution,
    objective: f64,
    stdout: &str,
    stderr: &str,
) {
    let mut best_bound = None;
    let mut mip_gap = None;
    let mut nodes_explored = None;

    for line in stdout.lines().chain(stderr.lines()) {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("dual bound") {
            best_bound = first_float_after_colon(stripped);
        } else if lowered.starts_with("gap") {
            if let Some(value) = first_float(stripped) {
                mip_gap = Some(if stripped.contains('%') {
                    value / 100.0
                } else {
                    value
                });
            }
        } else if lowered.starts_with("nodes") {
            if let Some(value) = first_float_after_colon(stripped) {
                nodes_explored = (value >= 0.0).then_some(value.round() as u64);
            }
        }
    }

    if let Some(best_bound) = best_bound.filter(|value| value.is_finite()) {
        solution.best_bound = Some(best_bound);
        solution.absolute_gap = Some((best_bound - objective).abs().max(0.0));
        if mip_gap.is_none() {
            mip_gap = Some((best_bound - objective).abs() / objective.abs().max(1.0));
        }
    }
    solution.mip_gap = mip_gap
        .filter(|value| value.is_finite())
        .map(|value| value.max(0.0));
    solution.nodes_explored = nodes_explored;
}

fn highs_lp_algorithm_feedback(
    kind: ExternalLinearCliKind,
    lp_algorithm: Option<ExternalLinearCliLpAlgorithm>,
    stdout: &str,
    stderr: &str,
) -> Option<String> {
    if kind != ExternalLinearCliKind::Lp {
        return None;
    }
    let lp_algorithm = lp_algorithm?;
    let text = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    text.contains(&format!(
        "set option solver to \"{}\"",
        lp_algorithm.as_str()
    ))
    .then(|| lp_algorithm.as_str().to_string())
}

fn highs_objective_limit_feedback(
    kind: ExternalLinearCliKind,
    objective_limit: Option<f64>,
    stdout: &str,
    stderr: &str,
) -> Option<f64> {
    if kind != ExternalLinearCliKind::Mip {
        return None;
    }
    let objective_limit = normalized_objective_limit(objective_limit)?;
    let text = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    text.contains("set option objective_target to")
        .then_some(objective_limit)
}

fn highs_threads_feedback(threads: Option<u32>, stdout: &str, stderr: &str) -> Option<u32> {
    let threads = threads.filter(|threads| *threads > 0)?;
    let text = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    text.contains(&format!("set option threads to {threads}"))
        .then_some(threads)
}

fn highs_random_seed_feedback(random_seed: Option<u64>, stdout: &str, stderr: &str) -> Option<u64> {
    let random_seed = random_seed?;
    let text = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    text.contains(&format!("set option random_seed to {random_seed}"))
        .then_some(random_seed)
}

fn highs_presolve_feedback(
    presolve: Option<ExternalLinearCliPresolve>,
    stdout: &str,
    stderr: &str,
) -> Option<String> {
    let presolve = presolve?;
    let highs_presolve = if presolve == ExternalLinearCliPresolve::Auto {
        "choose"
    } else {
        presolve.as_str()
    };
    let text = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    text.contains(&format!("set option presolve to \"{highs_presolve}\""))
        .then(|| presolve.as_str().to_string())
}

fn highs_mip_start_accepted(stdout: &str, stderr: &str) -> bool {
    let text = format!("{stdout}\n{stderr}");
    let lowered = text.to_ascii_lowercase();
    if lowered.contains("mip start solution is feasible") {
        return true;
    }
    if !lowered.contains("assessing feasibility of mip") {
        return false;
    }
    let infeasibilities = text
        .lines()
        .filter(|line| line.to_ascii_lowercase().contains("infeasibilities"))
        .filter_map(first_float)
        .collect::<Vec<_>>();
    infeasibilities.len() >= 3
        && infeasibilities
            .iter()
            .take(3)
            .all(|value| value.abs() <= 1.0e-9)
}

fn first_float_after_colon(text: &str) -> Option<f64> {
    let text = text.split_once(':').map(|(_, rest)| rest).unwrap_or(text);
    first_float(text)
}

fn first_float(text: &str) -> Option<f64> {
    text.split(|ch: char| ch.is_whitespace() || ch == ',' || ch == '(' || ch == ')')
        .filter_map(|token| {
            let token = token.trim().trim_end_matches('%');
            (!token.is_empty())
                .then(|| token.parse::<f64>().ok())
                .flatten()
        })
        .next()
}

fn dot_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

fn nonempty_trimmed(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

fn solve_glpk_cli_json_direct(
    kind: ExternalLinearCliKind,
    problem_json: &Value,
    opts: &ExternalLinearCliOptions,
    t0: Instant,
) -> Option<ExternalLinearCliSolution> {
    if opts.solver != ExternalLinearCliSolver::Glpk {
        return None;
    }
    if opts.solution_pool_size.is_some() {
        return None;
    }

    let solver = "glpk:cli".to_string();
    let model = match highs_model_from_cli_json(kind, problem_json) {
        Ok(Some(model)) => model,
        Ok(None) => return None,
        Err(message) => {
            return Some(external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                solver,
                message,
                elapsed_ms(t0),
            ));
        }
    };
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Glpk, opts)
    else {
        return Some(external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            solver,
            "glpk executable not found".to_string(),
            elapsed_ms(t0),
        ));
    };

    let temp_dir = match ExternalLinearCliTempDir::new("ores-glpk-cli") {
        Ok(temp_dir) => temp_dir,
        Err(err) => {
            return Some(external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                solver,
                format!("failed to create temporary GLPK workspace: {err}"),
                elapsed_ms(t0),
            ));
        }
    };
    let model_format = opts.model_format;
    let model_extension = match model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = temp_dir.path().join(format!("model.{model_extension}"));
    let solution_path = temp_dir.path().join("glpk.sol");
    if let Err(err) = fs::write(&model_path, glpk_model_to_string(&model, model_format)) {
        return Some(external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            solver,
            format!("failed to write GLPK model file: {err}"),
            elapsed_ms(t0),
        ));
    }

    let time_limit = normalized_time_limit(opts.time_limit_secs);
    let time_limit_secs = time_limit.ceil().max(1.0) as u64;
    let mut command = Command::new(&command_path);
    match model_format {
        ExternalLinearCliModelFormat::CplexLp => {
            command.arg("--lp");
        }
        ExternalLinearCliModelFormat::Mps => {
            command.arg("--freemps");
        }
    }
    command.arg(&model_path);
    command.arg(match model.sense {
        Sense::Max => "--max",
        Sense::Min => "--min",
    });
    match kind {
        ExternalLinearCliKind::Lp => {
            command
                .arg("--output")
                .arg(solution_path.with_extension("report"))
                .arg("--write")
                .arg(&solution_path)
                .arg("--tmlim")
                .arg(time_limit_secs.to_string());
            if opts.presolve == Some(ExternalLinearCliPresolve::Off) {
                command.arg("--nopresol");
            } else if opts.presolve == Some(ExternalLinearCliPresolve::On) {
                command.arg("--presol");
            }
            if opts.lp_algorithm == Some(ExternalLinearCliLpAlgorithm::Simplex) {
                command.arg("--simplex");
            } else if opts.lp_algorithm == Some(ExternalLinearCliLpAlgorithm::Ipm) {
                command.arg("--interior");
            }
        }
        ExternalLinearCliKind::Mip => {
            command
                .arg("-o")
                .arg(&solution_path)
                .arg("--tmlim")
                .arg(time_limit_secs.to_string());
            if opts.presolve == Some(ExternalLinearCliPresolve::Off) {
                command.arg("--nointopt");
            } else if opts.presolve == Some(ExternalLinearCliPresolve::On) {
                command.arg("--intopt");
            }
            if opts.branch_rule == Some(ExternalLinearCliBranchRule::FirstFractional) {
                command.arg("--first");
            } else if opts.branch_rule == Some(ExternalLinearCliBranchRule::MostFractional) {
                command.arg("--mostf");
            }
            if opts.node_selection == Some(ExternalLinearCliNodeSelection::Dfs) {
                command.arg("--dfs");
            } else if opts.node_selection == Some(ExternalLinearCliNodeSelection::BestBound) {
                command.arg("--bestb");
            }
            if let Some(gap) = normalized_relative_gap(opts.relative_gap) {
                command.arg("--mipgap").arg(format!("{gap:.17}"));
            }
            if opts.cuts == Some(ExternalLinearCliMipSwitch::On) {
                command.arg("--cuts");
            }
        }
    }
    if let Some(random_seed) = opts.random_seed {
        command.arg("--seed").arg(random_seed.to_string());
    }

    let output = match command
        .current_dir(temp_dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            return Some(external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                solver,
                format!(
                    "failed to start GLPK command '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            ));
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let solver_version = glpk_solver_version_from_output(&stdout, &stderr);

    if !solution_path.exists() {
        let status = classify_highs_status("", &stdout, &stderr);
        let message = nonempty_trimmed(&stderr).unwrap_or_else(|| stdout.trim().to_string());
        let status = if matches!(
            status,
            ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
        ) {
            status
        } else {
            ExternalLinearCliStatus::Unavailable
        };
        let mut solution = external_cli_failure(status, solver, message, elapsed_ms(t0));
        solution.solver_version = solver_version;
        return Some(solution);
    }

    let parsed = match parse_glpk_solution_file(
        &solution_path,
        model.c.len(),
        model.le_rows.len(),
        model.eq_rows.len(),
    ) {
        Ok(parsed) => parsed,
        Err(message) => {
            return Some(external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                solver,
                message,
                elapsed_ms(t0),
            ));
        }
    };
    let status = classify_highs_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let status = if matches!(
            status,
            ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
        ) {
            status
        } else {
            ExternalLinearCliStatus::Unavailable
        };
        let mut solution = external_cli_failure(status, solver, parsed.status, elapsed_ms(t0));
        solution.solver_version = solver_version;
        return Some(solution);
    }

    let objective = dot_f64(&model.c, &parsed.x);
    let mut solution = ExternalLinearCliSolution {
        status,
        solver,
        solver_version,
        x: parsed.x,
        objective: Some(objective),
        objective_values: None,
        lp_algorithm: glpk_lp_algorithm_feedback(kind, opts.lp_algorithm, &stdout, &stderr),
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
        random_seed: glpk_random_seed_feedback(opts.random_seed, &stdout, &stderr),
        presolve: glpk_presolve_feedback(kind, opts.presolve, &stdout, &stderr),
        cuts: glpk_cuts_feedback(kind, opts.cuts, &stdout, &stderr),
        heuristics: None,
        branch_rule: glpk_branch_rule_feedback(kind, opts.branch_rule, &stdout, &stderr),
        branch_priorities_accepted: None,
        branch_priority_count: None,
        node_selection: glpk_node_selection_feedback(kind, opts.node_selection, &stdout, &stderr),
        mip_start_accepted: None,
        mip_start_objective: None,
        dual_ub: if kind == ExternalLinearCliKind::Lp {
            parsed.dual_ub
        } else {
            None
        },
        dual_eq: if kind == ExternalLinearCliKind::Lp {
            parsed.dual_eq
        } else {
            None
        },
        reduced_costs: if kind == ExternalLinearCliKind::Lp {
            parsed.reduced_costs
        } else {
            None
        },
        var_basis: if kind == ExternalLinearCliKind::Lp {
            parsed.var_basis
        } else {
            None
        },
        row_basis: if kind == ExternalLinearCliKind::Lp {
            parsed.row_basis
        } else {
            None
        },
        iterations: glpk_lp_iterations(kind, &stdout, &stderr),
        elapsed_ms: elapsed_ms(t0),
        message: parsed.status,
    };
    if kind == ExternalLinearCliKind::Mip {
        apply_glpk_mip_quality(&mut solution, &stdout, &stderr);
    }
    Some(solution)
}

fn glpk_model_to_string(
    model: &HighsCliModel,
    model_format: ExternalLinearCliModelFormat,
) -> String {
    match model_format {
        ExternalLinearCliModelFormat::CplexLp => cplex_lp_string(
            model.sense,
            &model.c,
            &model.le_rows,
            &model.le_rhs,
            &model.eq_rows,
            &model.eq_rhs,
            &model.lbs,
            &model.ubs,
            &model.integer_vars,
        ),
        ExternalLinearCliModelFormat::Mps => mps_string_with_objsense(
            model.sense,
            &model.c,
            &model.le_rows,
            &model.le_rhs,
            &model.eq_rows,
            &model.eq_rhs,
            &model.lbs,
            &model.ubs,
            &model.integer_vars,
            false,
        ),
    }
}

fn parse_glpk_solution_file(
    path: &Path,
    n: usize,
    le_count: usize,
    eq_count: usize,
) -> Result<HighsParsedSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read GLPK solution file '{}': {err}",
            path.display()
        )
    })?;
    Ok(parse_glpk_solution_text(&text, n, le_count, eq_count))
}

fn parse_glpk_solution_text(
    text: &str,
    n: usize,
    le_count: usize,
    eq_count: usize,
) -> HighsParsedSolution {
    let mut x = vec![0.0; n];
    let mut status = "unknown".to_string();
    let mut row_duals = vec![None; le_count + eq_count];
    let mut reduced_costs = vec![None; n];
    let mut var_basis = vec![None; n];
    let mut row_basis = vec![None; le_count + eq_count];
    let mut in_named_columns = false;

    for line in text.lines() {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() >= 3 && parts[0] == "c" && parts[1] == "Status:" {
            status = parts[2..].join(" ").to_ascii_lowercase();
        } else if parts.len() >= 2 && parts[0] == "Status:" {
            status = parts[1..].join(" ").to_ascii_lowercase();
        } else if line.contains("Column name") {
            in_named_columns = true;
        } else if in_named_columns
            && stripped_starts(line, &["Integer feasibility", "KKT.", "End of output"])
        {
            in_named_columns = false;
        } else if in_named_columns
            && parts.len() >= 3
            && parts[0].chars().all(|ch| ch.is_ascii_digit())
            && parts[1].starts_with('x')
        {
            if let Some(idx) = parse_x_index(parts[1], n) {
                if let Some(value) = parts
                    .iter()
                    .skip(2)
                    .find_map(|token| (*token != "*").then(|| token.parse::<f64>().ok()).flatten())
                {
                    x[idx] = value;
                }
            }
        } else if parts.len() >= 3 && parts[0] == "j" {
            let idx = parts[1]
                .parse::<usize>()
                .ok()
                .and_then(|idx| idx.checked_sub(1));
            if let Some(idx) = idx.filter(|idx| *idx < n) {
                if parts.len() >= 4 && parts[2].parse::<f64>().is_err() {
                    if let Ok(value) = parts[3].parse::<f64>() {
                        x[idx] = value;
                    }
                    if let Some(status) = basis_status_from_token(parts[2]) {
                        var_basis[idx] = Some(status.to_string());
                    }
                    if parts.len() >= 5 {
                        if let Ok(value) = parts[4].parse::<f64>() {
                            reduced_costs[idx] = Some(value);
                        }
                    }
                } else if let Ok(value) = parts[2].parse::<f64>() {
                    x[idx] = value;
                }
            }
        } else if parts.len() >= 5 && parts[0] == "i" {
            let idx = parts[1]
                .parse::<usize>()
                .ok()
                .and_then(|idx| idx.checked_sub(1));
            if let Some(idx) = idx.filter(|idx| *idx < row_duals.len()) {
                if let Some(status) = basis_status_from_token(parts[2]) {
                    row_basis[idx] = Some(status.to_string());
                }
                if let Ok(value) = parts[4].parse::<f64>() {
                    row_duals[idx] = Some(value);
                }
            }
        }
    }

    let dual_ub = row_duals[..le_count]
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>();
    let dual_eq = row_duals[le_count..]
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>();
    HighsParsedSolution {
        status,
        x,
        dual_ub,
        dual_eq,
        reduced_costs: reduced_costs.into_iter().collect::<Option<Vec<_>>>(),
        var_basis: var_basis.into_iter().collect::<Option<Vec<_>>>(),
        row_basis: row_basis.into_iter().collect::<Option<Vec<_>>>(),
    }
}

fn stripped_starts(text: &str, prefixes: &[&str]) -> bool {
    let stripped = text.trim();
    prefixes.iter().any(|prefix| stripped.starts_with(prefix))
}

fn glpk_solver_version_from_output(stdout: &str, stderr: &str) -> Option<String> {
    for line in stdout.lines().chain(stderr.lines()) {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("GLPSOL--GLPK LP/MIP Solver ") {
            if let Some(version) = rest.split_whitespace().next() {
                return Some(format!("GLPK {version}"));
            }
        }
    }
    None
}

fn glpk_lp_iterations(kind: ExternalLinearCliKind, stdout: &str, stderr: &str) -> Option<u64> {
    if kind != ExternalLinearCliKind::Lp {
        return None;
    }
    for line in stdout.lines().chain(stderr.lines()) {
        let stripped = line.trim_start_matches('*').trim();
        let Some((iteration, rest)) = stripped.split_once(':') else {
            continue;
        };
        if rest.trim_start().starts_with("obj") {
            if let Ok(iteration) = iteration.trim().parse::<u64>() {
                return Some(iteration);
            }
        }
    }
    None
}

fn apply_glpk_mip_quality(solution: &mut ExternalLinearCliSolution, stdout: &str, stderr: &str) {
    for line in stdout.lines().chain(stderr.lines()) {
        let lowered = line.to_ascii_lowercase();
        if lowered.contains("mip gap") || lowered.contains("relative gap") {
            if let Some(gap) = first_float(line) {
                solution.mip_gap = Some(if line.contains('%') { gap / 100.0 } else { gap });
            }
        }
        if lowered.contains("tree is empty") || lowered.contains("integer optimization begins") {
            solution.nodes_explored.get_or_insert(0);
        }
    }
}

fn glpk_lp_algorithm_feedback(
    kind: ExternalLinearCliKind,
    lp_algorithm: Option<ExternalLinearCliLpAlgorithm>,
    stdout: &str,
    stderr: &str,
) -> Option<String> {
    if kind != ExternalLinearCliKind::Lp {
        return None;
    }
    let lp_algorithm = lp_algorithm?;
    let flag = match lp_algorithm {
        ExternalLinearCliLpAlgorithm::Simplex => "--simplex",
        ExternalLinearCliLpAlgorithm::Ipm => "--interior",
    };
    format!("{stdout}\n{stderr}")
        .contains(flag)
        .then(|| lp_algorithm.as_str().to_string())
}

fn glpk_random_seed_feedback(random_seed: Option<u64>, stdout: &str, stderr: &str) -> Option<u64> {
    let random_seed = random_seed?;
    format!("{stdout}\n{stderr}")
        .contains(&format!("--seed {random_seed}"))
        .then_some(random_seed)
}

fn glpk_presolve_feedback(
    kind: ExternalLinearCliKind,
    presolve: Option<ExternalLinearCliPresolve>,
    stdout: &str,
    stderr: &str,
) -> Option<String> {
    let presolve = presolve?;
    let text = format!("{stdout}\n{stderr}");
    let accepted = match (kind, presolve) {
        (ExternalLinearCliKind::Lp, ExternalLinearCliPresolve::Off) => text.contains("--nopresol"),
        (ExternalLinearCliKind::Lp, ExternalLinearCliPresolve::On) => text.contains("--presol"),
        (ExternalLinearCliKind::Mip, ExternalLinearCliPresolve::Off) => text.contains("--nointopt"),
        (ExternalLinearCliKind::Mip, ExternalLinearCliPresolve::On) => text.contains("--intopt"),
        (_, ExternalLinearCliPresolve::Auto) => false,
    };
    accepted.then(|| presolve.as_str().to_string())
}

fn glpk_cuts_feedback(
    kind: ExternalLinearCliKind,
    cuts: Option<ExternalLinearCliMipSwitch>,
    stdout: &str,
    stderr: &str,
) -> Option<String> {
    (kind == ExternalLinearCliKind::Mip
        && cuts == Some(ExternalLinearCliMipSwitch::On)
        && format!("{stdout}\n{stderr}").contains("--cuts"))
    .then(|| "on".to_string())
}

fn glpk_branch_rule_feedback(
    kind: ExternalLinearCliKind,
    branch_rule: Option<ExternalLinearCliBranchRule>,
    stdout: &str,
    stderr: &str,
) -> Option<String> {
    if kind != ExternalLinearCliKind::Mip {
        return None;
    }
    let branch_rule = branch_rule?;
    let flag = match branch_rule {
        ExternalLinearCliBranchRule::FirstFractional => "--first",
        ExternalLinearCliBranchRule::MostFractional => "--mostf",
    };
    format!("{stdout}\n{stderr}")
        .contains(flag)
        .then(|| branch_rule.as_str().to_string())
}

fn glpk_node_selection_feedback(
    kind: ExternalLinearCliKind,
    node_selection: Option<ExternalLinearCliNodeSelection>,
    stdout: &str,
    stderr: &str,
) -> Option<String> {
    if kind != ExternalLinearCliKind::Mip {
        return None;
    }
    let node_selection = node_selection?;
    let flag = match node_selection {
        ExternalLinearCliNodeSelection::Dfs => "--dfs",
        ExternalLinearCliNodeSelection::BestBound => "--bestb",
    };
    format!("{stdout}\n{stderr}")
        .contains(flag)
        .then(|| node_selection.as_str().to_string())
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
        objective_values: None,
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
    mps_string_with_objsense(
        sense,
        c,
        le_rows,
        le_rhs,
        eq_rows,
        eq_rhs,
        lbs,
        ubs,
        integer_vars,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn mps_string_with_objsense(
    sense: Sense,
    c: &[f64],
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    lbs: &[Option<f64>],
    ubs: &[Option<f64>],
    integer_vars: &[bool],
    include_objsense: bool,
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
    if include_objsense {
        out.push_str("OBJSENSE\n");
        out.push_str(match sense {
            Sense::Max => " MAX\n",
            Sense::Min => " MIN\n",
        });
    }
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

fn find_first_command(
    command_env_vars: &[&str],
    command_dir_env_vars: &[&str],
    aliases: &[&str],
) -> Option<PathBuf> {
    let mut saw_configured_env = false;
    for env_var in command_env_vars {
        if let Some(value) = std::env::var_os(env_var) {
            if !value.to_string_lossy().trim().is_empty() {
                saw_configured_env = true;
                if let Some(path) = resolve_command_candidate(&PathBuf::from(value)) {
                    return Some(path);
                }
            }
        }
    }

    for env_var in command_dir_env_vars {
        if let Some(value) = std::env::var_os(env_var) {
            if !value.to_string_lossy().trim().is_empty() {
                saw_configured_env = true;
                if let Some(path) = find_command_in_install_dir(&PathBuf::from(value), aliases) {
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

fn find_command_in_install_dir(root: &Path, aliases: &[&str]) -> Option<PathBuf> {
    let mut candidate_dirs = vec![root.to_path_buf(), root.join("bin")];
    if let Ok(children) = std::fs::read_dir(root) {
        for child in children.flatten() {
            let child_path = child.path();
            if !child_path.is_dir() {
                continue;
            }
            let child_bin = child_path.join("bin");
            candidate_dirs.push(child_bin.clone());
            if let Ok(platform_dirs) = std::fs::read_dir(&child_bin) {
                for platform_dir in platform_dirs.flatten() {
                    let platform_path = platform_dir.path();
                    if platform_path.is_dir() {
                        candidate_dirs.push(platform_path);
                    }
                }
            }
        }
    }

    for dir in candidate_dirs {
        for alias in aliases {
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

#[cfg(test)]
fn normalized_node_limit(node_limit: Option<usize>) -> Option<usize> {
    node_limit.filter(|value| *value > 0)
}

#[cfg(test)]
fn normalized_threads(threads: Option<u32>) -> Option<u32> {
    threads.filter(|value| *value > 0)
}

#[cfg(test)]
fn normalized_random_seed(random_seed: Option<u32>) -> Option<u32> {
    random_seed.filter(|value| *value <= i32::MAX as u32)
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

fn opt_f64_value(value: Option<f64>) -> Value {
    value.map_or(Value::Null, f64_value)
}

fn insert_cli_field(payload: &mut Value, key: &str, value: Value) {
    if let Value::Object(object) = payload {
        object.insert(key.to_string(), value);
    }
}

fn linear_constraints_json(constraints: &[LinearRowConstraint]) -> Value {
    Value::Array(
        constraints
            .iter()
            .map(|constraint| {
                json!({
                    "coefs": f64_vec(&constraint.coefs),
                    "lower": opt_f64_value(constraint.lower),
                    "upper": opt_f64_value(constraint.upper),
                    "name": &constraint.name,
                })
            })
            .collect(),
    )
}

fn indicator_constraints_json(constraints: &[IndicatorConstraint]) -> Value {
    Value::Array(
        constraints
            .iter()
            .map(|indicator| {
                json!({
                    "binary_var": indicator.binary_var,
                    "active_value": indicator.active_value,
                    "coefs": f64_vec(&indicator.coefs),
                    "sense": indicator.sense.as_str(),
                    "rhs": f64_value(indicator.rhs),
                    "name": &indicator.name,
                })
            })
            .collect(),
    )
}

fn sos_sets_json(sets: &[SpecialOrderedSet]) -> Value {
    Value::Array(
        sets.iter()
            .map(|set| {
                json!({
                    "kind": set.kind.as_str(),
                    "vars": &set.vars,
                    "weights": opt_plain_vec_f64(set.weights.as_ref()),
                    "name": &set.name,
                })
            })
            .collect(),
    )
}

fn semi_variables_json(variables: &[SemiVariable]) -> Value {
    Value::Array(
        variables
            .iter()
            .map(|semi| {
                json!({
                    "kind": semi.kind.as_str(),
                    "var": semi.var,
                    "lower": f64_value(semi.lower),
                    "name": &semi.name,
                })
            })
            .collect(),
    )
}

fn pwl_constraints_json(constraints: &[PiecewiseLinearConstraint]) -> Value {
    Value::Array(
        constraints
            .iter()
            .map(|pwl| {
                json!({
                    "x_var": pwl.x_var,
                    "y_var": pwl.y_var,
                    "points": Value::Array(
                        pwl.points
                            .iter()
                            .map(|point| {
                                json!({
                                    "x": f64_value(point.x),
                                    "y": f64_value(point.y),
                                })
                            })
                            .collect(),
                    ),
                    "name": &pwl.name,
                })
            })
            .collect(),
    )
}

fn quadratic_objective_json(terms: &[QuadraticObjectiveTerm]) -> Value {
    Value::Array(
        terms
            .iter()
            .map(|term| {
                json!({
                    "x_var": term.x_var,
                    "y_var": term.y_var,
                    "coeff": f64_value(term.coeff),
                    "name": &term.name,
                })
            })
            .collect(),
    )
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
        external_linear_cli_command, external_linear_cli_command_with_options,
        external_linear_cli_solver_manifest, external_linear_cli_solver_specs,
        find_command_in_install_dir, general_linear_ipmip_problem_to_cli_json,
        indicator_ipmip_problem_to_cli_json, ipmip_problem_to_cli_json,
        ipmip_problem_to_cplex_lp_string, ipmip_problem_to_mps_string,
        lower_bounded_ipmip_problem_to_cli_json, lp_problem_to_cli_json,
        lp_problem_to_cplex_lp_string, lp_problem_to_mps_string,
        multi_objective_ipmip_problem_to_cli_json, normalized_node_limit, normalized_random_seed,
        normalized_relative_gap, normalized_threads, pwl_ipmip_problem_to_cli_json,
        quadratic_objective_ipmip_problem_to_cli_json, semi_ipmip_problem_to_cli_json,
        solve_ipmip_with_external_cli, solve_lp_with_external_cli, solver_command_env_var,
        sos_ipmip_problem_to_cli_json, source_ipmip_problem_to_cli_json, ExternalLinearCliKind,
        ExternalLinearCliLicenseClass, ExternalLinearCliModelFormat, ExternalLinearCliOptions,
        ExternalLinearCliProbeStatus, ExternalLinearCliSolver, ExternalLinearCliStatus,
    };
    use crate::des::general::ip_mip_des::{
        build_fixed_charge_indicator_ip, build_general_linear_rows_ip,
        build_lower_bounded_production_ip, build_piecewise_linear_reward_ip,
        build_quadratic_objective_mix_ip, build_semi_continuous_gate_ip, build_sos1_choice_ip,
        build_source_feature_mix_ip, BranchOrCutConstraint, ConstraintKind, IPMIPProblem,
        LexicographicObjective, MultiObjectiveIPMIPProblem,
    };
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
    fn source_ipmip_payload_includes_general_constraint_sections() {
        let p = build_source_feature_mix_ip();
        let payload = source_ipmip_problem_to_cli_json(&p);
        assert_eq!(payload["sense"], "max");
        assert_eq!(payload["lb"][0], 1.0);
        assert_eq!(
            payload["linear_constraints"][0]["name"],
            "activity_reward_budget"
        );
        assert_eq!(payload["indicators"][0]["sense"], "le");
        assert_eq!(payload["pwl"][0]["points"][1]["y"], 5.0);
        assert!(payload["products"].as_array().is_some_and(Vec::is_empty));
    }

    #[test]
    fn source_wrapper_payloads_include_owned_sections() {
        let lower = lower_bounded_ipmip_problem_to_cli_json(&build_lower_bounded_production_ip());
        assert_eq!(lower["lb"][0], 3.0);

        let general = general_linear_ipmip_problem_to_cli_json(&build_general_linear_rows_ip());
        assert_eq!(general["linear_constraints"][2]["name"], "throughput_range");
        assert_eq!(general["linear_constraints"][2]["upper"], 7.0);

        let indicator = indicator_ipmip_problem_to_cli_json(&build_fixed_charge_indicator_ip());
        assert_eq!(indicator["indicators"][0]["active_value"], false);
        assert_eq!(indicator["indicators"][0]["sense"], "le");

        let sos = sos_ipmip_problem_to_cli_json(&build_sos1_choice_ip());
        assert_eq!(sos["sos"][0]["kind"], "sos1");
        assert_eq!(sos["sos"][0]["weights"][2], 3.0);

        let semi = semi_ipmip_problem_to_cli_json(&build_semi_continuous_gate_ip());
        assert_eq!(semi["semi_variables"][0]["kind"], "semi_continuous");
        assert_eq!(semi["semi_variables"][0]["lower"], 3.0);

        let pwl = pwl_ipmip_problem_to_cli_json(&build_piecewise_linear_reward_ip());
        assert_eq!(pwl["pwl"][0]["points"][1]["y"], 4.0);

        let quadratic =
            quadratic_objective_ipmip_problem_to_cli_json(&build_quadratic_objective_mix_ip());
        assert_eq!(
            quadratic["quadratic_objective"][0]["name"],
            "machine_premium_bonus"
        );
        assert_eq!(quadratic["lb"][2], 1.0);
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
    fn multi_objective_payload_includes_lexicographic_stages() {
        let base = IPMIPProblem {
            sense: Sense::Max,
            c: vec![0.0, 0.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![1.0],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let p = MultiObjectiveIPMIPProblem {
            base,
            objectives: vec![
                LexicographicObjective {
                    sense: Sense::Max,
                    c: vec![1.0, 1.0],
                    name: Some("cardinality".to_string()),
                },
                LexicographicObjective {
                    sense: Sense::Max,
                    c: vec![3.0, 1.0],
                    name: Some("preference".to_string()),
                },
            ],
        };
        let payload = multi_objective_ipmip_problem_to_cli_json(&p);
        assert_eq!(payload["multi_objectives"][0]["sense"], "max");
        assert_eq!(payload["multi_objectives"][0]["name"], "cardinality");
        assert_eq!(payload["multi_objectives"][1]["c"][0], 3.0);
        assert_eq!(payload["integer_vars"][1], true);
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
    fn highs_solution_parser_extracts_primal_duals_and_basis() {
        let text = r#"
Model status
Optimal

# Primal solution values
Feasible
Objective 34
# Columns 2
x0 6
x1 4
# Rows 2
c0 10
e0 4

# Dual solution values
Feasible
# Columns 2
x0 0
x1 -1.5
# Rows 2
c0 3
e0 -2

# Basis
HiGHS_basis_file v2
# Columns 2
x0 1
x1 0
# Rows 2
c0 2
e0 1
"#;
        let parsed = super::parse_highs_solution_text(text, 2, 1, 1);
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![6.0, 4.0]);
        assert_eq!(parsed.dual_ub, Some(vec![3.0]));
        assert_eq!(parsed.dual_eq, Some(vec![-2.0]));
        assert_eq!(parsed.reduced_costs, Some(vec![0.0, -1.5]));
        assert_eq!(
            parsed.var_basis,
            Some(vec!["basic".to_string(), "at_lower".to_string()])
        );
        assert_eq!(
            parsed.row_basis,
            Some(vec!["at_upper".to_string(), "basic".to_string()])
        );
    }

    #[test]
    fn glpk_solution_parser_extracts_machine_solution_fields() {
        let text = r#"
c Status: OPTIMAL
i 1 b 1 3
i 2 u 2 -2
j 1 b 6 0
j 2 l 4 -1.5
"#;
        let parsed = super::parse_glpk_solution_text(text, 2, 1, 1);
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![6.0, 4.0]);
        assert_eq!(parsed.dual_ub, Some(vec![3.0]));
        assert_eq!(parsed.dual_eq, Some(vec![-2.0]));
        assert_eq!(parsed.reduced_costs, Some(vec![0.0, -1.5]));
        assert_eq!(
            parsed.var_basis,
            Some(vec!["basic".to_string(), "at_lower".to_string()])
        );
        assert_eq!(
            parsed.row_basis,
            Some(vec!["basic".to_string(), "at_upper".to_string()])
        );
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
    fn highs_direct_plain_lp_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Highs) else {
            eprintln!("SKIP direct HiGHS LP solve: highs command not installed");
            return;
        };
        let solution = solve_lp_with_external_cli(
            &super::external_linear_cli_smoke_lp(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-highs-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "highs:cli");
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
        assert!(solution
            .solver_version
            .as_deref()
            .is_some_and(|version| version.starts_with("HiGHS ")));
    }

    #[test]
    fn highs_direct_plain_mip_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Highs) else {
            eprintln!("SKIP direct HiGHS MIP solve: highs command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-highs-direct".to_string()),
                time_limit_secs: Some(2.0),
                random_seed: Some(7),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "highs:cli");
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
        assert!(solution
            .best_bound
            .is_some_and(|bound| (bound - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn glpk_direct_plain_lp_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Glpk) else {
            eprintln!("SKIP direct GLPK LP solve: glpsol command not installed");
            return;
        };
        let solution = solve_lp_with_external_cli(
            &super::external_linear_cli_smoke_lp(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Glpk,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-glpk-direct".to_string()),
                model_format: ExternalLinearCliModelFormat::Mps,
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "glpk:cli");
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
        assert!(solution
            .solver_version
            .as_deref()
            .is_some_and(|version| version.starts_with("GLPK ")));
    }

    #[test]
    fn glpk_direct_plain_mip_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Glpk) else {
            eprintln!("SKIP direct GLPK MIP solve: glpsol command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Glpk,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-glpk-direct".to_string()),
                model_format: ExternalLinearCliModelFormat::Mps,
                time_limit_secs: Some(2.0),
                random_seed: Some(7),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "glpk:cli");
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn solver_aliases_and_kind_support_match_bridge_contract() {
        assert_eq!(ExternalLinearCliSolver::all().len(), 12);
        assert_eq!(ExternalLinearCliSolver::Glpk.command_aliases(), &["glpsol"]);
        assert_eq!(
            ExternalLinearCliSolver::Highs.command_env_vars(),
            &[
                "HIGHS_CMD",
                "ORES_HIGHS_CMD",
                "ORES_HIGHS_BIN",
                "DES_HIGHS_BIN",
                "HIGHS_BIN"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::Glpk.command_env_vars(),
            &[
                "GLPSOL_CMD",
                "GLPK_CMD",
                "ORES_GLPK_CMD",
                "ORES_GLPK_BIN",
                "DES_GLPK_BIN",
                "GLPK_BIN"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::Scip.command_env_vars(),
            &[
                "SCIP_CMD",
                "ORES_SCIP_CMD",
                "ORES_SCIP_BIN",
                "DES_SCIP_BIN",
                "SCIP_BIN"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::Cbc.command_env_vars(),
            &[
                "CBC_CMD",
                "ORES_CBC_CMD",
                "ORES_CBC_BIN",
                "DES_CBC_BIN",
                "CBC_BIN"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::Clp.command_env_vars(),
            &[
                "CLP_CMD",
                "ORES_CLP_CMD",
                "ORES_CLP_BIN",
                "DES_CLP_BIN",
                "CLP_BIN"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::Soplex.command_env_vars(),
            &[
                "SOPLEX_CMD",
                "ORES_SOPLEX_CMD",
                "ORES_SOPLEX_BIN",
                "DES_SOPLEX_BIN",
                "SOPLEX_BIN"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::QsoptEx.command_env_vars(),
            &[
                "QSOPT_EX_CMD",
                "QSOPT_CMD",
                "ORES_QSOPT_EX_CMD",
                "ORES_QSOPT_EX_BIN",
                "DES_QSOPT_EX_BIN",
                "QSOPT_EX_BIN"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::LpSolve.command_env_vars(),
            &[
                "LP_SOLVE_CMD",
                "LPSOLVE_CMD",
                "ORES_LP_SOLVE_CMD",
                "ORES_LPSOLVE_BIN",
                "DES_LPSOLVE_BIN",
                "LPSOLVE_BIN"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::Gurobi.command_env_vars(),
            &[
                "GUROBI_CL_CMD",
                "GUROBI_CMD",
                "ORES_GUROBI_CMD",
                "ORES_GUROBI_BIN",
                "DES_GUROBI_BIN",
                "GUROBI_BIN"
            ]
        );
        assert_eq!(
            ExternalLinearCliSolver::Lindo.command_env_vars(),
            &[
                "RUNLINDO_CMD",
                "LINDO_CMD",
                "LINDOAPI_CMD",
                "ORES_LINDO_CMD",
                "ORES_LINDO_BIN",
                "DES_LINDO_BIN",
                "LINDO_BIN"
            ]
        );
        assert!(ExternalLinearCliSolver::Highs
            .command_env_vars()
            .contains(&solver_command_env_var(ExternalLinearCliSolver::Highs).as_str()));
        assert!(ExternalLinearCliSolver::Cplex
            .command_env_vars()
            .contains(&"CPLEX_BIN"));
        assert!(ExternalLinearCliSolver::Xpress
            .command_env_vars()
            .contains(&"XPRESS_BIN"));
        assert_eq!(
            ExternalLinearCliSolver::Gurobi.command_dir_env_vars(),
            &["GUROBI_HOME"]
        );
        assert!(ExternalLinearCliSolver::Highs
            .command_dir_env_vars()
            .contains(&"HIGHS_HOME"));
        assert!(ExternalLinearCliSolver::Cbc
            .command_dir_env_vars()
            .contains(&"COINOR_DIR"));
        assert!(ExternalLinearCliSolver::Cplex
            .command_dir_env_vars()
            .contains(&"CPLEX_STUDIO_DIR"));
        assert!(ExternalLinearCliSolver::Lindo
            .command_dir_env_vars()
            .contains(&"LINDOAPI_HOME"));
        assert!(ExternalLinearCliSolver::LpSolve
            .command_dir_env_vars()
            .contains(&"LP_SOLVE_HOME"));
        assert!(ExternalLinearCliSolver::Soplex
            .command_dir_env_vars()
            .contains(&"SOPLEX_HOME"));
        assert!(ExternalLinearCliSolver::QsoptEx
            .command_dir_env_vars()
            .contains(&"QSOPT_EX_HOME"));
        assert_eq!(
            ExternalLinearCliSolver::Xpress.command_aliases(),
            &["optimizer", "xpress"]
        );
        assert_eq!(
            ExternalLinearCliSolver::Lindo.command_aliases(),
            &["runlindo", "lindo", "lindoapi"]
        );
        assert_eq!(
            ExternalLinearCliSolver::LpSolve.command_aliases(),
            &["lp_solve", "lp-solve", "lpsolve"]
        );
        assert_eq!(
            ExternalLinearCliSolver::Soplex.command_aliases(),
            &["soplex"]
        );
        assert_eq!(
            ExternalLinearCliSolver::QsoptEx.command_aliases(),
            &["qsopt_ex", "qsopt-ex", "qsopt", "esolver"]
        );
        assert!(ExternalLinearCliSolver::Highs.supports_kind(ExternalLinearCliKind::Lp));
        assert!(ExternalLinearCliSolver::Highs.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::Clp.supports_kind(ExternalLinearCliKind::Lp));
        assert!(!ExternalLinearCliSolver::Clp.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::Soplex.supports_kind(ExternalLinearCliKind::Lp));
        assert!(!ExternalLinearCliSolver::Soplex.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::QsoptEx.supports_kind(ExternalLinearCliKind::Lp));
        assert!(!ExternalLinearCliSolver::QsoptEx.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::LpSolve.supports_kind(ExternalLinearCliKind::Lp));
        assert!(ExternalLinearCliSolver::LpSolve.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::Lindo.supports_kind(ExternalLinearCliKind::Lp));
        assert!(ExternalLinearCliSolver::Lindo.supports_kind(ExternalLinearCliKind::Mip));
        assert!(ExternalLinearCliSolver::Xpress.supports_kind(ExternalLinearCliKind::Lp));
        assert!(ExternalLinearCliSolver::Xpress.supports_kind(ExternalLinearCliKind::Mip));
    }

    #[test]
    fn solver_manifest_matches_bridge_contract() {
        let specs = external_linear_cli_solver_specs();
        assert_eq!(specs.len(), 12);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.license_class == ExternalLinearCliLicenseClass::OpenSource)
                .count(),
            8
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.license_class == ExternalLinearCliLicenseClass::Commercial)
                .count(),
            4
        );

        let highs = ExternalLinearCliSolver::Highs.spec();
        assert_eq!(highs.id, "highs");
        assert_eq!(highs.display_name, "HiGHS");
        assert_eq!(highs.license_class.as_str(), "open-source");
        assert!(highs.command_aliases.contains(&"highs"));
        assert!(highs.command_env_vars.contains(&"ORES_HIGHS_BIN"));
        assert!(highs.command_dir_env_vars.contains(&"HIGHS_HOME"));
        assert!(highs.supports_lp);
        assert!(highs.supports_mip);

        let clp = ExternalLinearCliSolver::Clp.spec();
        assert_eq!(clp.display_name, "COIN-OR CLP");
        assert!(clp.supports_lp);
        assert!(!clp.supports_mip);

        let soplex = ExternalLinearCliSolver::Soplex.spec();
        assert_eq!(soplex.id, "soplex");
        assert!(soplex.command_env_vars.contains(&"SOPLEX_CMD"));
        assert!(soplex.supports_lp);
        assert!(!soplex.supports_mip);

        let qsopt_ex = ExternalLinearCliSolver::QsoptEx.spec();
        assert_eq!(qsopt_ex.id, "qsopt-ex");
        assert_eq!(qsopt_ex.display_name, "QSopt_ex");
        assert!(qsopt_ex.command_aliases.contains(&"qsopt_ex"));
        assert!(qsopt_ex.command_aliases.contains(&"esolver"));
        assert!(qsopt_ex.command_env_vars.contains(&"QSOPT_EX_CMD"));
        assert!(qsopt_ex.command_dir_env_vars.contains(&"QSOPT_EX_HOME"));
        assert!(qsopt_ex.supports_lp);
        assert!(!qsopt_ex.supports_mip);

        let lindo = ExternalLinearCliSolver::Lindo.spec();
        assert_eq!(
            lindo.license_class,
            ExternalLinearCliLicenseClass::Commercial
        );
        assert!(lindo.command_env_vars.contains(&"LINDOAPI_CMD"));
        assert!(lindo.command_dir_env_vars.contains(&"LINDOAPI_HOME"));
        assert!(lindo.supports_lp);
        assert!(lindo.supports_mip);

        let manifest = external_linear_cli_solver_manifest();
        let items = manifest.as_array().expect("manifest array");
        assert_eq!(items.len(), 12);
        assert!(items.iter().any(|item| {
            item.get("id").and_then(|value| value.as_str()) == Some("cbc")
                && item
                    .get("commandAliases")
                    .and_then(|value| value.as_array())
                    .is_some_and(|aliases| aliases.iter().any(|alias| alias == "cbc"))
                && item.get("supportsMip").and_then(|value| value.as_bool()) == Some(true)
        }));
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
    fn installed_open_source_cli_solvers_cross_check_same_input_smoke_models() {
        let lp = super::external_linear_cli_smoke_lp();
        let mut lp_checked = 0;
        for solver in ExternalLinearCliSolver::open_source_lp().iter().copied() {
            let Some(command) = external_linear_cli_command(solver) else {
                eprintln!("SKIP LP {}: command not installed", solver.as_str());
                continue;
            };
            let solution = solve_lp_with_external_cli(
                &lp,
                &ExternalLinearCliOptions {
                    solver,
                    command_path: Some(command.clone()),
                    time_limit_secs: Some(2.0),
                    ..Default::default()
                },
            );
            assert_eq!(
                solution.status,
                ExternalLinearCliStatus::Optimal,
                "LP {} via {:?}: {}",
                solver.as_str(),
                command,
                solution.message
            );
            assert_eq!(solution.solver, format!("{}:cli", solver.as_str()));
            assert_eq!(solution.x.len(), 1, "LP {} x length", solver.as_str());
            assert!(
                (solution.x[0] - 1.0).abs() <= 1.0e-8,
                "LP {} x={:?}",
                solver.as_str(),
                solution.x
            );
            assert!(
                solution
                    .objective
                    .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8),
                "LP {} objective={:?}",
                solver.as_str(),
                solution.objective
            );
            lp_checked += 1;
        }

        let mip = super::external_linear_cli_smoke_mip();
        let mut mip_checked = 0;
        for solver in ExternalLinearCliSolver::open_source_mip().iter().copied() {
            let Some(command) = external_linear_cli_command(solver) else {
                eprintln!("SKIP MIP {}: command not installed", solver.as_str());
                continue;
            };
            let solution = solve_ipmip_with_external_cli(
                &mip,
                &ExternalLinearCliOptions {
                    solver,
                    command_path: Some(command.clone()),
                    time_limit_secs: Some(2.0),
                    random_seed: Some(7),
                    ..Default::default()
                },
            );
            assert_eq!(
                solution.status,
                ExternalLinearCliStatus::Optimal,
                "MIP {} via {:?}: {}",
                solver.as_str(),
                command,
                solution.message
            );
            assert_eq!(solution.solver, format!("{}:cli", solver.as_str()));
            assert_eq!(solution.x.len(), 1, "MIP {} x length", solver.as_str());
            assert!(
                (solution.x[0] - 1.0).abs() <= 1.0e-8,
                "MIP {} x={:?}",
                solver.as_str(),
                solution.x
            );
            assert!(
                solution
                    .objective
                    .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8),
                "MIP {} objective={:?}",
                solver.as_str(),
                solution.objective
            );
            mip_checked += 1;
        }

        if lp_checked == 0 && mip_checked == 0 {
            eprintln!("SKIP open-source CLI same-input smoke: no solver commands installed");
        }
    }

    #[test]
    fn install_dir_lookup_handles_vendor_bin_layout() {
        let root = std::env::temp_dir().join(format!(
            "des-external-linear-cli-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let command = root
            .join("cplex")
            .join("bin")
            .join("x86-64_osx")
            .join("cplex");
        std::fs::create_dir_all(command.parent().unwrap()).unwrap();
        std::fs::write(&command, b"").unwrap();

        assert_eq!(
            find_command_in_install_dir(&root, &["cplex"]),
            Some(command)
        );

        std::fs::remove_dir_all(root).unwrap();
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
