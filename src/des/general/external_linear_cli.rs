//! Local command-line adapters for installed LP/MIP solvers.
//!
//! This module exposes a Rust-facing interface for solver executables that are
//! installed locally (for example through Homebrew) without vendoring any
//! external binaries into the repository. Plain LP/MIP solves for the common
//! open-source and commercial CLIs run through native Rust model writers and
//! parser paths where practical. `scripts/linear_cli_reference.py` remains an
//! explicit compatibility bridge for solver options or source features that do
//! not yet have a Rust direct path.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Number, Value};

use crate::des::general::ip_mip_des::{
    linearize_general_linear_problem, linearize_indicator_problem, linearize_lower_bounds_problem,
    linearize_pwl_problem, linearize_quadratic_objective_problem, linearize_semi_problem,
    linearize_sos_problem, linearize_source_ipmip_problem, AbsoluteValueConstraint,
    BranchOrCutConstraint, ConstraintKind, GeneralLinearIPMIPProblem, IPMIPProblem,
    IndicatorConstraint, IndicatorIPMIPProblem, IndicatorSense, L1NormConstraint,
    LInfNormConstraint, LexicographicObjective, LinearRowConstraint, LogicalConstraint,
    LogicalConstraintKind, LowerBoundedIPMIPProblem, MaximumConstraint, MinimumConstraint,
    MultiObjectiveIPMIPProblem, PiecewiseLinearConstraint, PiecewiseLinearPoint, ProductConstraint,
    PwlIPMIPProblem, QuadraticObjectiveIPMIPProblem, QuadraticObjectiveTerm, SemiIPMIPProblem,
    SemiVariable, SemiVariableKind, SosIPMIPProblem, SourceIPMIPProblem, SpecialOrderedSet,
    SpecialOrderedSetKind,
};
use crate::des::general::lp::{LPProblem, Sense};
use crate::des::shared::linalg::{LinearSystem, Matrix};

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
    /// Python executable for the explicit compatibility bridge. Native Rust
    /// direct paths ignore this. Defaults to `PYTHON_BIN`, then `PYTHON`, then
    /// `python3` only when the compatibility bridge is reached.
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
struct PlainLinearCliModel {
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

/// Serialize an [`LPProblem`] into the shared linear-CLI JSON contract.
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

/// Serialize an [`IPMIPProblem`] into the shared linear-CLI JSON contract.
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
/// lazy/cut rows are emitted as ordinary `<=` rows for same-input external
/// validation; finite upper bounds and integer markers are emitted as LP
/// `Bounds`, `General`, and `Binary` sections.
pub fn ipmip_problem_to_cplex_lp_string(problem: &IPMIPProblem) -> String {
    let n = problem.c.len();
    let lbs = vec![Some(0.0); n];
    let ubs = problem
        .ub
        .as_ref()
        .map(|upper| upper.iter().copied().map(Some).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![None; n]);
    let (le_rows, le_rhs) = ipmip_le_rows_with_lazy(problem);
    cplex_lp_string(
        problem.sense,
        &problem.c,
        &le_rows,
        &le_rhs,
        &[],
        &[],
        &lbs,
        &ubs,
        &problem.integer_vars,
    )
}

fn lp_problem_to_lpsolve_lp_string(problem: &LPProblem) -> String {
    let n = problem.c.len();
    let lbs = problem.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]);
    let ubs = problem.ub.clone().unwrap_or_else(|| vec![None; n]);
    let integer_vars = vec![false; n];
    lpsolve_lp_string(
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

fn ipmip_problem_to_lpsolve_lp_string(problem: &IPMIPProblem) -> String {
    let n = problem.c.len();
    let lbs = vec![Some(0.0); n];
    let ubs = problem
        .ub
        .as_ref()
        .map(|upper| upper.iter().copied().map(Some).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![None; n]);
    let (le_rows, le_rhs) = ipmip_le_rows_with_lazy(problem);
    lpsolve_lp_string(
        problem.sense,
        &problem.c,
        &le_rows,
        &le_rhs,
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
    lp_problem_to_mps_string_with_objsense(problem, true)
}

fn lp_problem_to_mps_string_with_objsense(problem: &LPProblem, include_objsense: bool) -> String {
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
        include_objsense,
    )
}

/// Export an IP/MIP as a free-format MPS string with integer markers.
pub fn ipmip_problem_to_mps_string(problem: &IPMIPProblem) -> String {
    ipmip_problem_to_mps_string_with_objsense(problem, true)
}

fn ipmip_problem_to_mps_string_with_objsense(
    problem: &IPMIPProblem,
    include_objsense: bool,
) -> String {
    let n = problem.c.len();
    let lbs = vec![Some(0.0); n];
    let ubs = problem
        .ub
        .as_ref()
        .map(|upper| upper.iter().copied().map(Some).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![None; n]);
    let (le_rows, le_rhs) = ipmip_le_rows_with_lazy(problem);
    mps_string(
        problem.sense,
        &problem.c,
        &le_rows,
        &le_rhs,
        &[],
        &[],
        &lbs,
        &ubs,
        &problem.integer_vars,
        include_objsense,
    )
}

fn ipmip_le_rows_with_lazy(problem: &IPMIPProblem) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rows = problem.a.clone();
    let mut rhs = problem.b.clone();
    if let Some(lazy_constraints) = &problem.lazy_constraints {
        for constraint in lazy_constraints {
            debug_assert_eq!(
                constraint.coefs.len(),
                problem.c.len(),
                "lazy constraint row length must match variable count"
            );
            if constraint.coefs.len() == problem.c.len() {
                rows.push(constraint.coefs.clone());
                rhs.push(constraint.rhs);
            }
        }
    }
    (rows, rhs)
}

fn ipmip_total_le_row_count(problem: &IPMIPProblem) -> usize {
    problem.a.len()
        + problem.lazy_constraints.as_ref().map_or(0, |constraints| {
            constraints
                .iter()
                .filter(|constraint| constraint.coefs.len() == problem.c.len())
                .count()
        })
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
    if should_use_native_highs_cli(ExternalLinearCliKind::Lp, opts) {
        return solve_lp_with_native_highs_cli(problem, opts);
    }
    if should_use_native_gurobi_cli(ExternalLinearCliKind::Lp, opts) {
        return solve_lp_with_native_gurobi_cli(problem, opts);
    }
    if should_use_native_cplex_cli(ExternalLinearCliKind::Lp, opts) {
        return solve_lp_with_native_cplex_cli(problem, opts);
    }
    if should_use_native_xpress_cli(ExternalLinearCliKind::Lp, opts) {
        return solve_lp_with_native_xpress_cli(problem, opts);
    }
    if should_use_native_lindo_cli(ExternalLinearCliKind::Lp, opts) {
        return solve_lp_with_native_lindo_cli(problem, opts);
    }
    if should_use_native_glpk_cli(ExternalLinearCliKind::Lp, opts) {
        return solve_lp_with_native_glpk_cli(problem, opts);
    }
    if should_use_native_scip_cli(ExternalLinearCliKind::Lp, opts) {
        return solve_lp_with_native_scip_cli(problem, opts);
    }
    if should_use_native_cbc_cli(ExternalLinearCliKind::Lp, opts) {
        return solve_lp_with_native_cbc_cli(problem, opts);
    }
    if should_use_native_clp_cli(opts) {
        return solve_lp_with_native_clp_cli(problem, opts);
    }
    if should_use_native_soplex_cli(opts) {
        return solve_lp_with_native_soplex_cli(problem, opts);
    }
    if should_use_native_qsopt_ex_cli(opts) {
        return solve_lp_with_native_qsopt_ex_cli(problem, opts);
    }
    if should_use_native_lp_solve_cli(ExternalLinearCliKind::Lp, opts) {
        return solve_lp_with_native_lp_solve_cli(problem, opts);
    }
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
    if should_use_native_highs_cli(ExternalLinearCliKind::Mip, opts) {
        return solve_ipmip_with_native_highs_cli(problem, opts);
    }
    if should_use_native_gurobi_cli(ExternalLinearCliKind::Mip, opts) {
        return solve_ipmip_with_native_gurobi_cli(problem, opts);
    }
    if should_use_native_cplex_cli(ExternalLinearCliKind::Mip, opts) {
        return solve_ipmip_with_native_cplex_cli(problem, opts);
    }
    if should_use_native_xpress_cli(ExternalLinearCliKind::Mip, opts) {
        return solve_ipmip_with_native_xpress_cli(problem, opts);
    }
    if should_use_native_lindo_cli(ExternalLinearCliKind::Mip, opts) {
        return solve_ipmip_with_native_lindo_cli(problem, opts);
    }
    if should_use_native_glpk_cli(ExternalLinearCliKind::Mip, opts) {
        return solve_ipmip_with_native_glpk_cli(problem, opts);
    }
    if should_use_native_scip_cli(ExternalLinearCliKind::Mip, opts) {
        return solve_ipmip_with_native_scip_cli(problem, opts);
    }
    if should_use_native_cbc_cli(ExternalLinearCliKind::Mip, opts) {
        return solve_ipmip_with_native_cbc_cli(problem, opts);
    }
    if should_use_native_lp_solve_cli(ExternalLinearCliKind::Mip, opts) {
        return solve_ipmip_with_native_lp_solve_cli(problem, opts);
    }
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
    if should_use_rust_lower_bounded_source_cli(opts) {
        let source_lb = problem.lb.clone();
        let (linearized, objective_offset) = linearize_lower_bounds_problem(problem);
        return solve_linearized_ipmip_with_external_cli(
            &linearized,
            opts,
            &source_lb,
            objective_offset,
            source_lb.len(),
        );
    }
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
    if should_use_rust_linearized_source_cli(opts) {
        let linearized = linearize_general_linear_problem(problem);
        return solve_linearized_ipmip_with_external_cli(
            &linearized,
            opts,
            &[],
            0.0,
            problem.base.c.len(),
        );
    }
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
    if should_use_rust_linearized_source_cli(opts) {
        let linearized = linearize_indicator_problem(problem);
        return solve_linearized_ipmip_with_external_cli(
            &linearized,
            opts,
            &[],
            0.0,
            problem.base.c.len(),
        );
    }
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
    if should_use_rust_linearized_source_cli(opts) {
        let linearized = linearize_sos_problem(problem);
        return solve_linearized_ipmip_with_external_cli(
            &linearized,
            opts,
            &[],
            0.0,
            problem.base.c.len(),
        );
    }
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
    if should_use_rust_linearized_source_cli(opts) {
        let linearized = linearize_semi_problem(problem);
        return solve_linearized_ipmip_with_external_cli(
            &linearized,
            opts,
            &[],
            0.0,
            problem.base.c.len(),
        );
    }
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
    if should_use_rust_linearized_source_cli(opts) {
        let linearized = linearize_pwl_problem(problem);
        return solve_linearized_ipmip_with_external_cli(
            &linearized,
            opts,
            &[],
            0.0,
            problem.base.c.len(),
        );
    }
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
    if should_use_rust_linearized_source_cli(opts) {
        let source_lb = problem
            .lb
            .clone()
            .unwrap_or_else(|| vec![0.0; problem.base.c.len()]);
        let (linearized, objective_offset, original_var_count) =
            linearize_quadratic_objective_problem(problem);
        return solve_linearized_ipmip_with_external_cli(
            &linearized,
            opts,
            &source_lb,
            objective_offset,
            original_var_count,
        );
    }
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
    if should_use_rust_multi_objective_cli(opts) {
        return solve_multi_objective_ipmip_with_rust_external_cli(problem, opts);
    }
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
    if should_use_rust_linearized_source_cli(opts) {
        let source_lb = problem
            .lb
            .clone()
            .unwrap_or_else(|| vec![0.0; problem.base.c.len()]);
        let (linearized, objective_offset, original_var_count) =
            linearize_source_ipmip_problem(problem);
        return solve_linearized_ipmip_with_external_cli(
            &linearized,
            opts,
            &source_lb,
            objective_offset,
            original_var_count,
        );
    }
    solve_linear_cli_json(
        ExternalLinearCliKind::Mip,
        source_ipmip_problem_to_cli_json(problem),
        opts,
    )
}

fn should_use_rust_linearized_source_cli(opts: &ExternalLinearCliOptions) -> bool {
    opts.script_path.is_none()
        && linearized_source_mip_start_can_stay_native(opts)
        && (opts.branch_priorities.is_none()
            || matches!(
                opts.solver,
                ExternalLinearCliSolver::Scip | ExternalLinearCliSolver::Cbc
            ))
}

fn linearized_source_mip_start_can_stay_native(opts: &ExternalLinearCliOptions) -> bool {
    opts.mip_start.is_none()
        || (opts.solution_pool_size.is_none()
            && matches!(
                opts.solver,
                ExternalLinearCliSolver::Highs
                    | ExternalLinearCliSolver::Scip
                    | ExternalLinearCliSolver::Cbc
            ))
}

fn should_use_rust_lower_bounded_source_cli(opts: &ExternalLinearCliOptions) -> bool {
    opts.script_path.is_none()
}

fn should_use_rust_multi_objective_cli(opts: &ExternalLinearCliOptions) -> bool {
    opts.script_path.is_none()
}

fn solve_multi_objective_ipmip_with_rust_external_cli(
    problem: &MultiObjectiveIPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let solver_name = opts.solver.as_str();
    let bridge_solver = format!("{solver_name}:cli");
    if problem.objectives.is_empty() {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "multi_objectives must be non-empty".to_string(),
            elapsed_ms(t0),
        );
    }
    let variable_count = problem.base.c.len();
    for (idx, objective) in problem.objectives.iter().enumerate() {
        if objective.c.len() != variable_count {
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                format!(
                    "multi_objective {idx} coefficient length {} does not match variable count {variable_count}",
                    objective.c.len()
                ),
                elapsed_ms(t0),
            );
        }
        if objective
            .c
            .iter()
            .enumerate()
            .any(|(_, coef)| !coef.is_finite())
        {
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                format!("multi_objective {idx} coefficients must be finite"),
                elapsed_ms(t0),
            );
        }
    }

    let mut working = problem.base.clone();
    let last_stage = problem.objectives.len() - 1;
    let mut final_solution = None;
    for (idx, objective) in problem.objectives.iter().enumerate() {
        working.sense = objective.sense;
        working.c = objective.c.clone();
        let mut stage_opts = opts.clone();
        if idx > 0 {
            stage_opts.mip_start = None;
        }
        let mut stage_solution = solve_ipmip_with_external_cli(&working, &stage_opts);
        if stage_solution.status != ExternalLinearCliStatus::Optimal {
            stage_solution.objective_values = Some(Vec::new());
            stage_solution.elapsed_ms = elapsed_ms(t0);
            return stage_solution;
        }

        let optimum = dot_f64(&objective.c, &stage_solution.x);
        if idx < last_stage {
            append_external_lexicographic_lock_rows(
                &mut working,
                objective.c.clone(),
                optimum,
                objective
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("multi_objective_{idx}")),
            );
        } else {
            final_solution = Some(stage_solution);
        }
    }

    let mut solution = final_solution.expect("non-empty objectives produce a final stage");
    debug_assert_eq!(solution.status, ExternalLinearCliStatus::Optimal);
    let final_x = solution.x.clone();
    let objective_values = problem
        .objectives
        .iter()
        .map(|objective| dot_f64(&objective.c, &final_x))
        .collect::<Vec<_>>();
    solution.objective = objective_values.last().copied();
    solution.objective_values = Some(objective_values);
    solution.elapsed_ms = elapsed_ms(t0);
    solution.message = "sequential lexicographic optimization".to_string();
    solution
}

fn append_external_lexicographic_lock_rows(
    problem: &mut IPMIPProblem,
    row: Vec<f64>,
    rhs: f64,
    name: String,
) {
    problem.a.push(row.clone());
    problem.b.push(rhs);
    problem.a.push(row.into_iter().map(|coef| -coef).collect());
    problem.b.push(-rhs);
    if let Some(con_names) = problem.con_names.as_mut() {
        con_names.push(format!("{name}_le"));
        con_names.push(format!("{name}_ge"));
    }
}

fn solve_linearized_ipmip_with_external_cli(
    linearized: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
    source_lb: &[f64],
    objective_offset: f64,
    original_var_count: usize,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let mut solve_opts = opts.clone();
    if let Some(objective_limit) = solve_opts.objective_limit.as_mut() {
        if objective_offset.is_finite() {
            *objective_limit -= objective_offset;
        }
    }
    if let Err(message) = shift_linearized_external_mip_start(
        &mut solve_opts,
        source_lb,
        original_var_count,
        linearized.c.len(),
    ) {
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            format!("{}:cli", opts.solver.as_str()),
            message,
            elapsed_ms(t0),
        );
    }
    if let Err(message) =
        pad_linearized_external_branch_priorities(&mut solve_opts, linearized.c.len())
    {
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            format!("{}:cli", opts.solver.as_str()),
            message,
            elapsed_ms(t0),
        );
    }
    let solution = solve_ipmip_with_external_cli(linearized, &solve_opts);
    postprocess_linearized_external_solution(
        solution,
        opts,
        source_lb,
        objective_offset,
        original_var_count,
    )
}

fn postprocess_linearized_external_solution(
    mut solution: ExternalLinearCliSolution,
    opts: &ExternalLinearCliOptions,
    source_lb: &[f64],
    objective_offset: f64,
    original_var_count: usize,
) -> ExternalLinearCliSolution {
    shift_external_solution_x(&mut solution.x, source_lb, original_var_count);
    solution.objective = solution
        .objective
        .map(|objective| add_finite_objective_offset(objective, objective_offset));
    solution.best_bound = solution
        .best_bound
        .map(|best_bound| add_finite_objective_offset(best_bound, objective_offset));
    solution.mip_start_objective = solution
        .mip_start_objective
        .map(|objective| add_finite_objective_offset(objective, objective_offset));
    if let Some(solutions) = solution.solutions.as_mut() {
        for member in solutions {
            shift_external_solution_x(&mut member.x, source_lb, original_var_count);
            member.objective = add_finite_objective_offset(member.objective, objective_offset);
        }
    }
    if let Some(objective_limit) = opts.objective_limit {
        solution.objective_limit = Some(objective_limit);
    }
    if let (Some(best_bound), Some(objective)) = (solution.best_bound, solution.objective) {
        if best_bound.is_finite() && objective.is_finite() {
            let absolute_gap = (best_bound - objective).abs().max(0.0);
            solution.absolute_gap = Some(absolute_gap);
            solution.mip_gap = Some(absolute_gap / objective.abs().max(1.0));
        }
    }
    solution
}

fn shift_linearized_external_mip_start(
    opts: &mut ExternalLinearCliOptions,
    source_lb: &[f64],
    original_var_count: usize,
    linearized_var_count: usize,
) -> Result<(), String> {
    let Some(start) = opts.mip_start.as_mut() else {
        return Ok(());
    };
    if original_var_count > linearized_var_count {
        return Err(format!(
            "original variable count {original_var_count} exceeds linearized variable count {linearized_var_count}"
        ));
    }
    if start.len() != original_var_count && start.len() != linearized_var_count {
        return Err(format!(
            "mip_start length {} must match original variable count {} or linearized variable count {}",
            start.len(),
            original_var_count,
            linearized_var_count
        ));
    }
    if !source_lb.is_empty() {
        if source_lb.len() != original_var_count {
            return Err(format!(
                "lower-bound vector length {} does not match original variable count {}",
                source_lb.len(),
                original_var_count
            ));
        }
        for (value, lower) in start
            .iter_mut()
            .take(original_var_count)
            .zip(source_lb.iter())
        {
            *value -= *lower;
        }
    }
    if start.len() < linearized_var_count {
        start.resize(linearized_var_count, 0.0);
    }
    Ok(())
}

fn pad_linearized_external_branch_priorities(
    opts: &mut ExternalLinearCliOptions,
    linearized_var_count: usize,
) -> Result<(), String> {
    let Some(priorities) = opts.branch_priorities.as_mut() else {
        return Ok(());
    };
    if priorities.len() > linearized_var_count {
        return Err(format!(
            "branch_priorities length {} exceeds linearized variable count {}",
            priorities.len(),
            linearized_var_count
        ));
    }
    priorities.resize(linearized_var_count, 0);
    Ok(())
}

fn shift_external_solution_x(x: &mut [f64], source_lb: &[f64], original_var_count: usize) {
    if x.is_empty() {
        return;
    }
    for (value, lower) in x.iter_mut().take(original_var_count).zip(source_lb.iter()) {
        *value += *lower;
    }
}

fn add_finite_objective_offset(value: f64, offset: f64) -> f64 {
    if value.is_finite() && offset.is_finite() {
        value + offset
    } else {
        value
    }
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

fn external_linear_cli_env_configured(names: &[&str]) -> bool {
    names.iter().any(|name| {
        std::env::var_os(name).is_some_and(|value| !value.to_string_lossy().trim().is_empty())
    })
}

fn gams_lindo_command_with_options(opts: &ExternalLinearCliOptions) -> Option<PathBuf> {
    if opts.solver != ExternalLinearCliSolver::Lindo
        || opts.command_path.is_some()
        || external_linear_cli_env_configured(ExternalLinearCliSolver::Lindo.command_env_vars())
        || external_linear_cli_env_configured(ExternalLinearCliSolver::Lindo.command_dir_env_vars())
        || (external_linear_cli_command(ExternalLinearCliSolver::Lindo).is_some()
            && !lindo_gams_control_options_requested(opts))
    {
        return None;
    }
    find_first_command(
        &["ORES_LINDO_GAMS_CMD", "ORES_GAMS_CMD", "GAMS_CMD"],
        &[
            "ORES_LINDO_GAMS_DIR",
            "ORES_GAMS_DIR",
            "GAMS_HOME",
            "GAMS_DIR",
            "GAMSDIR",
        ],
        &["gams"],
    )
}

fn lindo_gams_control_options_requested(opts: &ExternalLinearCliOptions) -> bool {
    opts.mip_start.is_some()
        || opts.max_nodes.is_some()
        || opts.node_limit.is_some()
        || opts.relative_gap.is_some()
}

/// Probe one solver for installation, bridge support, and a tiny smoke solve.
pub fn probe_external_linear_cli_solver(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliProbe {
    let t0 = Instant::now();
    let solver = opts.solver;
    let command = external_linear_cli_command_with_options(solver, opts);
    let command = command.or_else(|| gams_lindo_command_with_options(opts));
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

/// Solve a raw linear-CLI JSON payload through a locally installed command-line
/// solver. Rust direct paths are attempted first; the Python compatibility
/// bridge is used only when no native route supports the requested shape.
pub fn solve_linear_cli_json(
    kind: ExternalLinearCliKind,
    problem_json: Value,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let solver_name = opts.solver.as_str();
    let bridge_solver = format!("{solver_name}:cli");
    if let Some(solution) = solve_native_plain_cli_json_direct(kind, &problem_json, opts, t0) {
        return solution;
    }
    if let Some(solution) =
        solve_rust_quadratic_objective_cli_json_direct(kind, &problem_json, opts, t0)
    {
        return solution;
    }
    if let Some(solution) =
        solve_rust_multi_objective_cli_json_direct(kind, &problem_json, opts, t0)
    {
        return solution;
    }
    if let Some(solution) = solve_rust_source_cli_json_direct(kind, &problem_json, opts, t0) {
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
    command.env("LINEAR_CLI_REFERENCE_FROM_RUST", "1");
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

    match child.stdin.take() {
        Some(mut stdin) => {
            if let Err(err) = stdin.write_all(stdin_json.as_bytes()) {
                let _ = child.kill();
                let _ = child.wait();
                return external_cli_failure(
                    ExternalLinearCliStatus::NumericalError,
                    bridge_solver,
                    format!("failed to write local CLI bridge stdin: {err}"),
                    elapsed_ms(t0),
                );
            }
        }
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                "local CLI bridge stdin pipe was unavailable".to_string(),
                elapsed_ms(t0),
            );
        }
    }

    let timeout_ms = linear_cli_reference_timeout_ms();
    let (output, timed_out) = match wait_for_linear_cli_reference_output(child, timeout_ms) {
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
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("linear_cli_reference.py timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; linear_cli_reference.py timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };

    if !output.status.success() {
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            stderr,
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
                stderr
            ),
            elapsed,
        ),
    }
}

fn should_use_native_highs_cli(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> bool {
    if opts.solver != ExternalLinearCliSolver::Highs
        || opts.script_path.is_some()
        || opts.solution_pool_size.is_some()
        || !mip_switch_is_none_or_auto(opts.cuts)
        || !mip_switch_is_none_or_auto(opts.heuristics)
        || opts.branch_rule.is_some()
        || opts.branch_priorities.is_some()
        || opts.node_selection.is_some()
        || (kind == ExternalLinearCliKind::Lp
            && (opts.solution_limit.is_some() || opts.mip_start.is_some()))
    {
        return false;
    }
    if kind == ExternalLinearCliKind::Lp {
        return true;
    }
    true
}

fn mip_switch_is_none_or_auto(value: Option<ExternalLinearCliMipSwitch>) -> bool {
    matches!(value, None | Some(ExternalLinearCliMipSwitch::Auto))
}

fn solve_lp_with_native_highs_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => lp_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => lp_problem_to_mps_string(problem),
    };
    let le_count = problem.a_ub.as_ref().map_or(0, Vec::len);
    let eq_count = problem.a_eq.as_ref().map_or(0, Vec::len);
    solve_native_highs_cli_model(
        ExternalLinearCliKind::Lp,
        &model_text,
        problem.c.len(),
        le_count,
        eq_count,
        &problem.c,
        opts,
    )
}

fn solve_ipmip_with_native_highs_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => ipmip_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => ipmip_problem_to_mps_string(problem),
    };
    solve_native_highs_cli_model(
        ExternalLinearCliKind::Mip,
        &model_text,
        problem.c.len(),
        ipmip_total_le_row_count(problem),
        0,
        &problem.c,
        opts,
    )
}

fn should_use_native_gurobi_cli(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> bool {
    if opts.solver != ExternalLinearCliSolver::Gurobi
        || opts.script_path.is_some()
        || opts.lp_algorithm.is_some()
        || opts.solution_pool_size.is_some()
        || opts.cuts.is_some()
        || opts.heuristics.is_some()
        || opts.branch_rule.is_some()
        || opts.branch_priorities.is_some()
        || opts.node_selection.is_some()
        || opts.mip_start.is_some()
    {
        return false;
    }
    if kind == ExternalLinearCliKind::Lp {
        return opts.max_nodes.is_none()
            && opts.node_limit.is_none()
            && opts.solution_limit.is_none()
            && opts.relative_gap.is_none()
            && opts.absolute_gap.is_none()
            && opts.objective_limit.is_none()
            && opts.integer_feasibility_tolerance.is_none();
    }
    true
}

fn solve_lp_with_native_gurobi_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => lp_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => lp_problem_to_mps_string(problem),
    };
    solve_native_gurobi_cli_model(
        ExternalLinearCliKind::Lp,
        &model_text,
        problem.c.len(),
        &problem.c,
        opts,
    )
}

fn solve_ipmip_with_native_gurobi_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => ipmip_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => ipmip_problem_to_mps_string(problem),
    };
    solve_native_gurobi_cli_model(
        ExternalLinearCliKind::Mip,
        &model_text,
        problem.c.len(),
        &problem.c,
        opts,
    )
}

fn should_use_native_cplex_cli(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> bool {
    if opts.solver != ExternalLinearCliSolver::Cplex
        || opts.script_path.is_some()
        || opts.lp_algorithm.is_some()
        || opts.solution_pool_size.is_some()
        || opts.cuts.is_some()
        || opts.heuristics.is_some()
        || opts.branch_rule.is_some()
        || opts.branch_priorities.is_some()
        || opts.node_selection.is_some()
        || opts.mip_start.is_some()
        || opts.objective_limit.is_some()
    {
        return false;
    }
    if kind == ExternalLinearCliKind::Lp {
        return opts.max_nodes.is_none()
            && opts.node_limit.is_none()
            && opts.solution_limit.is_none()
            && opts.relative_gap.is_none()
            && opts.absolute_gap.is_none()
            && opts.integer_feasibility_tolerance.is_none();
    }
    true
}

fn solve_lp_with_native_cplex_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => lp_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => lp_problem_to_mps_string(problem),
    };
    solve_native_cplex_cli_model(
        ExternalLinearCliKind::Lp,
        &model_text,
        problem.c.len(),
        &problem.c,
        opts,
    )
}

fn solve_ipmip_with_native_cplex_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => ipmip_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => ipmip_problem_to_mps_string(problem),
    };
    solve_native_cplex_cli_model(
        ExternalLinearCliKind::Mip,
        &model_text,
        problem.c.len(),
        &problem.c,
        opts,
    )
}

fn should_use_native_xpress_cli(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> bool {
    if opts.solver != ExternalLinearCliSolver::Xpress
        || opts.script_path.is_some()
        || opts.lp_algorithm.is_some()
        || opts.solution_pool_size.is_some()
        || opts.cuts.is_some()
        || opts.heuristics.is_some()
        || opts.branch_rule.is_some()
        || opts.branch_priorities.is_some()
        || opts.node_selection.is_some()
        || opts.mip_start.is_some()
        || opts.max_nodes.is_some()
        || opts.node_limit.is_some()
        || opts.relative_gap.is_some()
        || opts.absolute_gap.is_some()
        || opts.objective_limit.is_some()
    {
        return false;
    }
    if kind == ExternalLinearCliKind::Lp {
        return opts.solution_limit.is_none() && opts.integer_feasibility_tolerance.is_none();
    }
    true
}

fn solve_lp_with_native_xpress_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => lp_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => lp_problem_to_mps_string(problem),
    };
    solve_native_xpress_cli_model(
        ExternalLinearCliKind::Lp,
        &model_text,
        problem.c.len(),
        &problem.c,
        opts,
    )
}

fn solve_ipmip_with_native_xpress_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => ipmip_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => ipmip_problem_to_mps_string(problem),
    };
    solve_native_xpress_cli_model(
        ExternalLinearCliKind::Mip,
        &model_text,
        problem.c.len(),
        &problem.c,
        opts,
    )
}

fn should_use_native_lindo_cli(
    _kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> bool {
    if opts.solver != ExternalLinearCliSolver::Lindo
        || opts.script_path.is_some()
        || opts.lp_algorithm.is_some()
        || opts.solution_pool_size.is_some()
        || opts.cuts.is_some()
        || opts.heuristics.is_some()
        || opts.branch_rule.is_some()
        || opts.branch_priorities.is_some()
        || opts.node_selection.is_some()
        || opts.solution_limit.is_some()
        || opts.absolute_gap.is_some()
        || opts.objective_limit.is_some()
        || opts.primal_feasibility_tolerance.is_some()
        || opts.dual_feasibility_tolerance.is_some()
        || opts.integer_feasibility_tolerance.is_some()
        || opts.threads.is_some()
        || opts.random_seed.is_some()
        || opts.presolve.is_some()
    {
        return false;
    }
    if lindo_gams_control_options_requested(opts) && gams_lindo_command_with_options(opts).is_none()
    {
        return false;
    }
    true
}

fn solve_lp_with_native_lindo_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    if let Some(gams_command) = gams_lindo_command_with_options(opts) {
        return solve_native_lindo_gams_model(
            ExternalLinearCliKind::Lp,
            &lp_problem_to_plain_linear_model(problem),
            &problem.c,
            opts,
            &gams_command,
        );
    }
    let model_text = lp_problem_to_mps_string(problem);
    solve_native_lindo_cli_model(
        ExternalLinearCliKind::Lp,
        problem.sense,
        &model_text,
        problem.c.len(),
        &problem.c,
        opts,
    )
}

fn solve_ipmip_with_native_lindo_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    if let Some(gams_command) = gams_lindo_command_with_options(opts) {
        return solve_native_lindo_gams_model(
            ExternalLinearCliKind::Mip,
            &ipmip_problem_to_plain_linear_model(problem),
            &problem.c,
            opts,
            &gams_command,
        );
    }
    let model_text = ipmip_problem_to_mps_string(problem);
    solve_native_lindo_cli_model(
        ExternalLinearCliKind::Mip,
        problem.sense,
        &model_text,
        problem.c.len(),
        &problem.c,
        opts,
    )
}

fn should_use_native_glpk_cli(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> bool {
    if opts.solver != ExternalLinearCliSolver::Glpk
        || opts.script_path.is_some()
        || opts.solution_limit.is_some()
        || opts.solution_pool_size.is_some()
        || opts.absolute_gap.is_some()
        || opts.objective_limit.is_some()
        || opts.primal_feasibility_tolerance.is_some()
        || opts.dual_feasibility_tolerance.is_some()
        || opts.integer_feasibility_tolerance.is_some()
        || opts.branch_priorities.is_some()
        || opts.mip_start.is_some()
        || opts.random_seed.is_some_and(|seed| seed > i32::MAX as u64)
    {
        return false;
    }

    match kind {
        ExternalLinearCliKind::Lp => {
            opts.max_nodes.is_none()
                && opts.node_limit.is_none()
                && opts.relative_gap.is_none()
                && opts.threads.is_none()
                && mip_switch_is_none_or_auto(opts.cuts)
                && mip_switch_is_none_or_auto(opts.heuristics)
                && opts.branch_rule.is_none()
                && opts.node_selection.is_none()
        }
        ExternalLinearCliKind::Mip => {
            opts.lp_algorithm.is_none()
                && opts.max_nodes.is_none()
                && opts.node_limit.is_none()
                && matches!(
                    opts.heuristics,
                    None | Some(ExternalLinearCliMipSwitch::Auto)
                )
        }
    }
}

fn solve_lp_with_native_glpk_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => lp_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => lp_problem_to_mps_string_with_objsense(problem, false),
    };
    let le_count = problem.a_ub.as_ref().map_or(0, Vec::len);
    let eq_count = problem.a_eq.as_ref().map_or(0, Vec::len);
    solve_native_glpk_cli_model(
        ExternalLinearCliKind::Lp,
        problem.sense,
        &model_text,
        problem.c.len(),
        le_count,
        eq_count,
        &problem.c,
        opts,
    )
}

fn solve_ipmip_with_native_glpk_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => ipmip_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => {
            ipmip_problem_to_mps_string_with_objsense(problem, false)
        }
    };
    solve_native_glpk_cli_model(
        ExternalLinearCliKind::Mip,
        problem.sense,
        &model_text,
        problem.c.len(),
        ipmip_total_le_row_count(problem),
        0,
        &problem.c,
        opts,
    )
}

fn should_use_native_scip_cli(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> bool {
    if opts.solver != ExternalLinearCliSolver::Scip
        || opts.script_path.is_some()
        || opts.lp_algorithm.is_some()
        || opts.integer_feasibility_tolerance.is_some()
        || opts.branch_rule.is_some()
        || (kind == ExternalLinearCliKind::Lp && opts.branch_priorities.is_some())
        || opts.node_selection.is_some()
        || (kind == ExternalLinearCliKind::Lp && opts.mip_start.is_some())
        || !matches!(
            opts.heuristics,
            None | Some(ExternalLinearCliMipSwitch::Auto | ExternalLinearCliMipSwitch::Off)
        )
    {
        return false;
    }

    if kind == ExternalLinearCliKind::Lp {
        return opts.max_nodes.is_none()
            && opts.node_limit.is_none()
            && opts.solution_limit.is_none()
            && opts.solution_pool_size.is_none()
            && opts.relative_gap.is_none()
            && opts.absolute_gap.is_none()
            && opts.objective_limit.is_none()
            && mip_switch_is_none_or_auto(opts.cuts)
            && mip_switch_is_none_or_auto(opts.heuristics);
    }

    true
}

fn solve_lp_with_native_scip_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => lp_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => lp_problem_to_mps_string(problem),
    };
    let le_rows = problem.a_ub.as_deref().unwrap_or(&[]);
    let le_rhs = problem.b_ub.as_deref().unwrap_or(&[]);
    let eq_rows = problem.a_eq.as_deref().unwrap_or(&[]);
    let eq_rhs = problem.b_eq.as_deref().unwrap_or(&[]);
    solve_native_scip_cli_model(
        ExternalLinearCliKind::Lp,
        &model_text,
        problem.c.len(),
        problem.sense,
        le_rows,
        le_rhs,
        eq_rows,
        eq_rhs,
        problem.lb.as_deref(),
        problem.ub.as_deref(),
        None,
        &problem.c,
        opts,
    )
}

fn solve_ipmip_with_native_scip_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    if opts.solution_pool_size.is_some() {
        return solve_native_mip_solution_pool_cli_model(
            ipmip_problem_to_plain_linear_model(problem),
            opts,
        );
    }
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => ipmip_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => ipmip_problem_to_mps_string(problem),
    };
    solve_native_scip_cli_model(
        ExternalLinearCliKind::Mip,
        &model_text,
        problem.c.len(),
        problem.sense,
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        Some(&problem.integer_vars),
        &problem.c,
        opts,
    )
}

fn solve_native_scip_cli_model(
    kind: ExternalLinearCliKind,
    model_text: &str,
    variable_count: usize,
    sense: Sense,
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    lower_bounds: Option<&[Option<f64>]>,
    upper_bounds: Option<&[Option<f64>]>,
    integer_vars: Option<&[bool]>,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "scip:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Scip, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "scip executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = native_scip_temp_path("model", extension);
    let solution_path = native_scip_temp_path("solution", "sol");
    let start_path = (kind == ExternalLinearCliKind::Mip && opts.mip_start.is_some())
        .then(|| native_scip_temp_path("start", "sol"));
    let mut cleanup_paths = vec![model_path.clone(), solution_path.clone()];
    if let Some(start_path) = &start_path {
        cleanup_paths.push(start_path.clone());
    }
    let active_branch_priorities = if kind == ExternalLinearCliKind::Mip {
        match active_branch_priorities(
            opts.branch_priorities.as_deref(),
            integer_vars,
            variable_count,
        ) {
            Ok(priorities) => priorities,
            Err(message) => {
                cleanup_native_scip_temp_files(&cleanup_paths);
                return external_cli_failure(
                    ExternalLinearCliStatus::NumericalError,
                    bridge_solver,
                    message,
                    elapsed_ms(t0),
                );
            }
        }
    } else {
        Vec::new()
    };

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_scip_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write SCIP model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let mip_start_objective = if kind == ExternalLinearCliKind::Mip {
        match opts.mip_start.as_deref() {
            Some(mip_start) => {
                if mip_start.len() != variable_count {
                    cleanup_native_scip_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        format!(
                            "mip_start length {} does not match variable count {}",
                            mip_start.len(),
                            variable_count
                        ),
                        elapsed_ms(t0),
                    );
                }
                if mip_start.iter().any(|value| !value.is_finite()) {
                    cleanup_native_scip_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        "mip_start values must be finite".to_string(),
                        elapsed_ms(t0),
                    );
                }
                let objective = dot_f64(objective_coefficients, mip_start);
                if !objective.is_finite() {
                    cleanup_native_scip_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        "mip_start objective must be finite".to_string(),
                        elapsed_ms(t0),
                    );
                }
                let Some(start_path) = &start_path else {
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        "internal SCIP MIP-start path was unavailable".to_string(),
                        elapsed_ms(t0),
                    );
                };
                if let Err(err) =
                    fs::write(start_path, native_scip_mip_start_text(mip_start, objective))
                {
                    cleanup_native_scip_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        format!(
                            "failed to write SCIP MIP-start file '{}': {err}",
                            start_path.display()
                        ),
                        elapsed_ms(t0),
                    );
                }
                Some(objective)
            }
            None => None,
        }
    } else {
        None
    };

    let mut command = Command::new(&command_path);
    add_native_scip_option_commands(&mut command, kind, opts);
    if kind == ExternalLinearCliKind::Lp {
        command
            .arg("-c")
            .arg("set presolving maxrounds 0")
            .arg("-c")
            .arg("set separating maxrounds 0")
            .arg("-c")
            .arg("set heuristics trivial freq -1");
    }
    command
        .arg("-c")
        .arg(format!("read {}", model_path.display()));
    for (idx, priority) in &active_branch_priorities {
        command
            .arg("-c")
            .arg("set branching priority")
            .arg("-c")
            .arg(format!("x{idx}"))
            .arg("-c")
            .arg(priority.to_string());
    }
    if let Some(start_path) = &start_path {
        command
            .arg("-c")
            .arg(format!("read {}", start_path.display()));
    }
    command
        .arg("-c")
        .arg(format!(
            "set limits time {:.17}",
            normalized_time_limit(opts.time_limit_secs)
        ))
        .arg("-c")
        .arg("optimize")
        .arg("-c")
        .arg(format!("write solution {}", solution_path.display()));
    if kind == ExternalLinearCliKind::Lp {
        command.arg("-c").arg("display dualsolution");
    }
    if kind == ExternalLinearCliKind::Mip {
        command.arg("-c").arg("display statistics");
    }
    command
        .arg("-c")
        .arg("quit")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_scip_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start SCIP executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_scip_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_scip_solver_version(&command_path));

    let parsed =
        match parse_native_scip_solution_file(&solution_path, variable_count, &stdout, &stderr) {
            Ok(parsed) => parsed,
            Err(message) => {
                let status = classify_native_linear_status("", &stdout, &stderr);
                cleanup_native_scip_temp_files(&cleanup_paths);
                let mut failure = external_cli_failure(
                    if matches!(
                        status,
                        ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
                    ) {
                        status
                    } else {
                        ExternalLinearCliStatus::Unavailable
                    },
                    bridge_solver,
                    message,
                    elapsed,
                );
                failure.solver_version = solver_version;
                return failure;
            }
        };
    cleanup_native_scip_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let objective = dot_f64(objective_coefficients, &parsed.x);
    let quality = parse_scip_mip_quality(kind, objective, &stdout, &stderr);
    let (mip_start_accepted, mip_start_objective) = parse_scip_mip_start_feedback(
        kind,
        opts.mip_start.as_deref(),
        mip_start_objective,
        &stdout,
        &stderr,
    );
    let (branch_priorities_accepted, branch_priority_count) =
        parse_scip_branch_priority_feedback(kind, active_branch_priorities.len(), &stdout, &stderr);
    let certificate = (kind == ExternalLinearCliKind::Lp).then(|| {
        parse_scip_lp_certificate_fields(
            &stdout,
            sense,
            objective_coefficients,
            le_rows,
            le_rhs,
            eq_rows,
            eq_rhs,
            lower_bounds,
            upper_bounds,
            &parsed.x,
        )
    });
    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x,
        objective: Some(objective),
        objective_values: None,
        lp_algorithm: None,
        best_bound: quality.best_bound,
        solution_limit: (kind == ExternalLinearCliKind::Mip)
            .then(|| opts.solution_limit.map(|limit| limit.max(1)))
            .flatten(),
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: quality.mip_gap,
        absolute_gap: quality.absolute_gap,
        objective_limit: (kind == ExternalLinearCliKind::Mip)
            .then(|| normalized_objective_limit(opts.objective_limit))
            .flatten(),
        primal_feasibility_tolerance: normalized_tolerance(opts.primal_feasibility_tolerance),
        dual_feasibility_tolerance: normalized_tolerance(opts.dual_feasibility_tolerance),
        integer_feasibility_tolerance: None,
        nodes_explored: quality.nodes_explored,
        threads: opts.threads.filter(|threads| *threads > 0),
        random_seed: opts.random_seed,
        presolve: opts.presolve.map(|presolve| presolve.as_str().to_string()),
        cuts: (kind == ExternalLinearCliKind::Mip)
            .then(|| opts.cuts.map(|cuts| cuts.as_str().to_string()))
            .flatten(),
        heuristics: (kind == ExternalLinearCliKind::Mip)
            .then(|| {
                opts.heuristics
                    .map(|heuristics| heuristics.as_str().to_string())
            })
            .flatten(),
        branch_rule: None,
        branch_priorities_accepted,
        branch_priority_count,
        node_selection: None,
        mip_start_accepted,
        mip_start_objective,
        dual_ub: certificate
            .as_ref()
            .and_then(|fields| fields.dual_ub.clone()),
        dual_eq: certificate
            .as_ref()
            .and_then(|fields| fields.dual_eq.clone()),
        reduced_costs: certificate
            .as_ref()
            .and_then(|fields| fields.reduced_costs.clone()),
        var_basis: certificate
            .as_ref()
            .and_then(|fields| fields.var_basis.clone()),
        row_basis: certificate.and_then(|fields| fields.row_basis),
        iterations: None,
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn add_native_scip_option_commands(
    command: &mut Command,
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) {
    if let Some(presolve) = opts.presolve {
        match presolve {
            ExternalLinearCliPresolve::On => {
                command.arg("-c").arg("set presolving maxrounds -1");
            }
            ExternalLinearCliPresolve::Off => {
                command.arg("-c").arg("set presolving maxrounds 0");
            }
            ExternalLinearCliPresolve::Auto => {}
        }
    }
    if let Some(random_seed) = opts.random_seed {
        command
            .arg("-c")
            .arg(format!("set randomization randomseedshift {random_seed}"));
    }
    if let Some(threads) = opts.threads.filter(|threads| *threads > 0) {
        command
            .arg("-c")
            .arg(format!("set parallel maxnthreads {threads}"));
    }
    if let Some(tolerance) = normalized_tolerance(opts.primal_feasibility_tolerance) {
        command
            .arg("-c")
            .arg(format!("set numerics feastol {tolerance:.17}"));
    }
    if let Some(tolerance) = normalized_tolerance(opts.dual_feasibility_tolerance) {
        command
            .arg("-c")
            .arg(format!("set numerics dualfeastol {tolerance:.17}"));
    }
    if kind != ExternalLinearCliKind::Mip {
        return;
    }
    if let Some(cuts) = opts.cuts {
        match cuts {
            ExternalLinearCliMipSwitch::On => {
                command
                    .arg("-c")
                    .arg("set separating maxrounds -1")
                    .arg("-c")
                    .arg("set separating maxroundsroot -1");
            }
            ExternalLinearCliMipSwitch::Off => {
                command
                    .arg("-c")
                    .arg("set separating maxrounds 0")
                    .arg("-c")
                    .arg("set separating maxroundsroot 0");
            }
            ExternalLinearCliMipSwitch::Auto => {}
        }
    }
    if opts.heuristics == Some(ExternalLinearCliMipSwitch::Off) {
        command.arg("-c").arg("set heuristics emphasis off");
    }
    if let Some(max_nodes) = opts
        .max_nodes
        .or_else(|| opts.node_limit.map(|limit| limit as u64))
        .filter(|limit| *limit > 0)
    {
        command
            .arg("-c")
            .arg(format!("set limits nodes {max_nodes}"));
    }
    if let Some(solution_limit) = opts.solution_limit.filter(|limit| *limit > 0) {
        command
            .arg("-c")
            .arg(format!("set limits solutions {solution_limit}"));
    }
    if let Some(relative_gap) = normalized_relative_gap(opts.relative_gap) {
        command
            .arg("-c")
            .arg(format!("set limits gap {relative_gap:.17}"));
    }
    if let Some(absolute_gap) = normalized_absolute_gap(opts.absolute_gap) {
        command
            .arg("-c")
            .arg(format!("set limits absgap {absolute_gap:.17}"));
    }
    if let Some(objective_limit) = normalized_objective_limit(opts.objective_limit) {
        command
            .arg("-c")
            .arg(format!("set limits primal {objective_limit:.17}"));
    }
}

fn should_use_native_cbc_cli(kind: ExternalLinearCliKind, opts: &ExternalLinearCliOptions) -> bool {
    if opts.solver != ExternalLinearCliSolver::Cbc
        || opts.script_path.is_some()
        || opts.lp_algorithm.is_some()
        || opts.objective_limit.is_some()
        || opts.branch_rule.is_some()
        || opts.random_seed.is_some_and(|seed| seed > i32::MAX as u64)
    {
        return false;
    }

    if kind == ExternalLinearCliKind::Lp {
        return opts.max_nodes.is_none()
            && opts.node_limit.is_none()
            && opts.solution_limit.is_none()
            && opts.solution_pool_size.is_none()
            && opts.relative_gap.is_none()
            && opts.absolute_gap.is_none()
            && opts.integer_feasibility_tolerance.is_none()
            && opts.threads.is_none()
            && opts.random_seed.is_none()
            && matches!(opts.presolve, None | Some(ExternalLinearCliPresolve::Auto))
            && mip_switch_is_none_or_auto(opts.cuts)
            && mip_switch_is_none_or_auto(opts.heuristics)
            && opts.branch_priorities.is_none()
            && opts.node_selection.is_none()
            && opts.mip_start.is_none();
    }

    true
}

fn solve_native_plain_cli_json_direct(
    kind: ExternalLinearCliKind,
    problem_json: &Value,
    opts: &ExternalLinearCliOptions,
    t0: Instant,
) -> Option<ExternalLinearCliSolution> {
    let use_highs = should_use_native_highs_cli(kind, opts);
    let use_gurobi = should_use_native_gurobi_cli(kind, opts);
    let use_cplex = should_use_native_cplex_cli(kind, opts);
    let use_xpress = should_use_native_xpress_cli(kind, opts);
    let use_lindo = should_use_native_lindo_cli(kind, opts);
    let use_glpk = should_use_native_glpk_cli(kind, opts);
    let use_scip = should_use_native_scip_cli(kind, opts);
    let use_cbc = should_use_native_cbc_cli(kind, opts);
    let use_clp = kind == ExternalLinearCliKind::Lp && should_use_native_clp_cli(opts);
    let use_soplex = kind == ExternalLinearCliKind::Lp && should_use_native_soplex_cli(opts);
    let use_qsopt_ex = kind == ExternalLinearCliKind::Lp && should_use_native_qsopt_ex_cli(opts);
    let use_lp_solve = should_use_native_lp_solve_cli(kind, opts);
    if !use_highs
        && !use_gurobi
        && !use_cplex
        && !use_xpress
        && !use_lindo
        && !use_glpk
        && !use_scip
        && !use_cbc
        && !use_clp
        && !use_soplex
        && !use_qsopt_ex
        && !use_lp_solve
    {
        return None;
    }

    let solver = format!("{}:cli", opts.solver.as_str());
    let model = match plain_linear_model_from_cli_json(kind, problem_json) {
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

    if kind == ExternalLinearCliKind::Mip
        && opts.solution_pool_size.is_some()
        && matches!(
            opts.solver,
            ExternalLinearCliSolver::Scip | ExternalLinearCliSolver::Cbc
        )
    {
        return Some(solve_native_mip_solution_pool_cli_model(model, opts));
    }

    let include_objsense = opts.solver != ExternalLinearCliSolver::Glpk;
    let model_text = plain_linear_model_to_string(&model, opts.model_format, include_objsense);
    let solution = match opts.solver {
        ExternalLinearCliSolver::Highs => solve_native_highs_cli_model(
            kind,
            &model_text,
            model.c.len(),
            model.le_rows.len(),
            model.eq_rows.len(),
            &model.c,
            opts,
        ),
        ExternalLinearCliSolver::Gurobi => {
            solve_native_gurobi_cli_model(kind, &model_text, model.c.len(), &model.c, opts)
        }
        ExternalLinearCliSolver::Cplex => {
            solve_native_cplex_cli_model(kind, &model_text, model.c.len(), &model.c, opts)
        }
        ExternalLinearCliSolver::Xpress => {
            solve_native_xpress_cli_model(kind, &model_text, model.c.len(), &model.c, opts)
        }
        ExternalLinearCliSolver::Lindo => {
            if let Some(gams_command) = gams_lindo_command_with_options(opts) {
                solve_native_lindo_gams_model(kind, &model, &model.c, opts, &gams_command)
            } else {
                let model_text =
                    plain_linear_model_to_string(&model, ExternalLinearCliModelFormat::Mps, true);
                solve_native_lindo_cli_model(
                    kind,
                    model.sense,
                    &model_text,
                    model.c.len(),
                    &model.c,
                    opts,
                )
            }
        }
        ExternalLinearCliSolver::Glpk => solve_native_glpk_cli_model(
            kind,
            model.sense,
            &model_text,
            model.c.len(),
            model.le_rows.len(),
            model.eq_rows.len(),
            &model.c,
            opts,
        ),
        ExternalLinearCliSolver::Scip => solve_native_scip_cli_model(
            kind,
            &model_text,
            model.c.len(),
            model.sense,
            &model.le_rows,
            &model.le_rhs,
            &model.eq_rows,
            &model.eq_rhs,
            Some(&model.lbs),
            Some(&model.ubs),
            Some(&model.integer_vars),
            &model.c,
            opts,
        ),
        ExternalLinearCliSolver::Cbc => solve_native_cbc_cli_model(
            kind,
            model.sense,
            &model_text,
            model.c.len(),
            model.le_rows.len(),
            model.eq_rows.len(),
            Some(&model.integer_vars),
            &model.c,
            opts,
        ),
        ExternalLinearCliSolver::Clp if kind == ExternalLinearCliKind::Lp => {
            solve_native_clp_cli_model(
                model.sense,
                &model_text,
                model.c.len(),
                model.le_rows.len(),
                model.eq_rows.len(),
                &model.c,
                opts,
            )
        }
        ExternalLinearCliSolver::Soplex if kind == ExternalLinearCliKind::Lp => {
            solve_native_soplex_cli_model(
                &model_text,
                model.c.len(),
                model.le_rows.len(),
                model.eq_rows.len(),
                &model.c,
                opts,
            )
        }
        ExternalLinearCliSolver::QsoptEx if kind == ExternalLinearCliKind::Lp => {
            let model_text =
                plain_linear_model_to_string(&model, ExternalLinearCliModelFormat::CplexLp, true);
            solve_native_qsopt_ex_cli_model(
                &model_text,
                model.c.len(),
                &model.le_rows,
                &model.le_rhs,
                &model.eq_rows,
                &model.eq_rhs,
                Some(&model.lbs),
                Some(&model.ubs),
                &model.c,
                opts,
            )
        }
        ExternalLinearCliSolver::LpSolve => solve_native_lp_solve_cli_model(
            kind,
            &plain_linear_model_to_lpsolve_lp_string(&model),
            model.c.len(),
            model.le_rows.len(),
            model.eq_rows.len(),
            &model.c,
            opts,
        ),
        _ => return None,
    };
    Some(solution)
}

fn solve_rust_quadratic_objective_cli_json_direct(
    kind: ExternalLinearCliKind,
    problem_json: &Value,
    opts: &ExternalLinearCliOptions,
    t0: Instant,
) -> Option<ExternalLinearCliSolution> {
    if kind != ExternalLinearCliKind::Mip || !should_use_rust_linearized_source_cli(opts) {
        return None;
    }
    let solver = format!("{}:cli", opts.solver.as_str());
    match quadratic_objective_ipmip_problem_from_cli_json(problem_json) {
        Ok(Some(problem)) => Some(solve_quadratic_objective_ipmip_with_external_cli(
            &problem, opts,
        )),
        Ok(None) => None,
        Err(message) => Some(external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            solver,
            message,
            elapsed_ms(t0),
        )),
    }
}

fn solve_rust_multi_objective_cli_json_direct(
    kind: ExternalLinearCliKind,
    problem_json: &Value,
    opts: &ExternalLinearCliOptions,
    t0: Instant,
) -> Option<ExternalLinearCliSolution> {
    if kind != ExternalLinearCliKind::Mip || !should_use_rust_multi_objective_cli(opts) {
        return None;
    }
    let solver = format!("{}:cli", opts.solver.as_str());
    match multi_objective_ipmip_problem_from_cli_json(problem_json) {
        Ok(Some(problem)) => Some(solve_multi_objective_ipmip_with_rust_external_cli(
            &problem, opts,
        )),
        Ok(None) => None,
        Err(message) => Some(external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            solver,
            message,
            elapsed_ms(t0),
        )),
    }
}

fn solve_rust_source_cli_json_direct(
    kind: ExternalLinearCliKind,
    problem_json: &Value,
    opts: &ExternalLinearCliOptions,
    t0: Instant,
) -> Option<ExternalLinearCliSolution> {
    if kind != ExternalLinearCliKind::Mip || !should_use_rust_linearized_source_cli(opts) {
        return None;
    }
    let solver = format!("{}:cli", opts.solver.as_str());
    match source_ipmip_problem_from_cli_json(problem_json) {
        Ok(Some(problem)) => Some(solve_source_ipmip_with_external_cli(&problem, opts)),
        Ok(None) => None,
        Err(message) => Some(external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            solver,
            message,
            elapsed_ms(t0),
        )),
    }
}

fn plain_linear_model_from_cli_json(
    kind: ExternalLinearCliKind,
    problem_json: &Value,
) -> Result<Option<PlainLinearCliModel>, String> {
    match kind {
        ExternalLinearCliKind::Lp => plain_linear_lp_model_from_cli_json(problem_json).map(Some),
        ExternalLinearCliKind::Mip => plain_linear_mip_model_from_cli_json(problem_json),
    }
}

fn plain_linear_lp_model_from_cli_json(
    problem_json: &Value,
) -> Result<PlainLinearCliModel, String> {
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
    validate_plain_linear_model_dimensions(n, &le_rows, &le_rhs, &eq_rows, &eq_rhs, &lbs, &ubs)?;
    Ok(PlainLinearCliModel {
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

fn plain_linear_mip_model_from_cli_json(
    problem_json: &Value,
) -> Result<Option<PlainLinearCliModel>, String> {
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
    validate_plain_linear_model_dimensions(n, &le_rows, &le_rhs, &eq_rows, &eq_rhs, &lbs, &ubs)?;
    Ok(Some(PlainLinearCliModel {
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

const RUST_SOURCE_CLI_FEATURE_KEYS: &[&str] = &[
    "linear_constraints",
    "indicators",
    "sos",
    "semi_variables",
    "pwl",
    "abs",
    "maximums",
    "minimums",
    "logical",
    "l1_norms",
    "linf_norms",
    "products",
];

fn quadratic_objective_ipmip_problem_from_cli_json(
    problem_json: &Value,
) -> Result<Option<QuadraticObjectiveIPMIPProblem>, String> {
    let Some(object) = problem_json.as_object() else {
        return Ok(None);
    };
    let Some(terms_json) =
        optional_json_array(object.get("quadratic_objective"), "quadratic_objective")?
    else {
        return Ok(None);
    };
    if terms_json.is_empty() {
        return Ok(None);
    }
    if json_field_has_content(object.get("multi_objectives"))
        || RUST_SOURCE_CLI_FEATURE_KEYS
            .iter()
            .any(|key| json_field_has_content(object.get(*key)))
    {
        return Ok(None);
    }

    let c = required_f64_array(object, "c")?;
    let n = c.len();
    let sense = parse_cli_sense(object.get("sense"))?;
    let a = optional_f64_matrix(object, &["a"])?;
    let b = optional_f64_array(object, &["b"])?;
    validate_source_base_rows(n, &a, &b)?;
    let integer_vars = optional_bool_array(object.get("integer_vars"), n, false, "integer_vars")?;
    let lb = optional_plain_f64_array_exact(object.get("lb"), n, "lb")?;
    let ub = optional_plain_f64_array_exact(object.get("ub"), n, "ub")?;
    let var_names = optional_string_array_exact(object.get("var_names"), n, "var_names")?;
    let con_names = optional_string_array_exact(object.get("con_names"), a.len(), "con_names")?;
    let lazy_constraints = parse_branch_or_cut_constraints(object.get("lazy_constraints"), n)?;
    let mut quadratic_objective = Vec::with_capacity(terms_json.len());
    for (idx, term_json) in terms_json.iter().enumerate() {
        let path = format!("quadratic_objective[{idx}]");
        let term_object = required_object(term_json, &path)?;
        let x_var = required_usize_field(term_object, "x_var", &format!("{path}.x_var"))?;
        validate_source_var_index(x_var, n, &format!("{path}.x_var"))?;
        let y_var = required_usize_field(term_object, "y_var", &format!("{path}.y_var"))?;
        validate_source_var_index(y_var, n, &format!("{path}.y_var"))?;
        quadratic_objective.push(QuadraticObjectiveTerm {
            x_var,
            y_var,
            coeff: required_f64_field(term_object.get("coeff"), &format!("{path}.coeff"))?,
            name: optional_string_field(term_object.get("name"), &format!("{path}.name"))?,
        });
    }

    Ok(Some(QuadraticObjectiveIPMIPProblem {
        base: IPMIPProblem {
            sense,
            c,
            a,
            b,
            integer_vars,
            ub,
            var_names,
            con_names,
            lazy_constraints,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb,
        quadratic_objective,
    }))
}

fn multi_objective_ipmip_problem_from_cli_json(
    problem_json: &Value,
) -> Result<Option<MultiObjectiveIPMIPProblem>, String> {
    let Some(object) = problem_json.as_object() else {
        return Ok(None);
    };
    let Some(objectives_json) =
        optional_json_array(object.get("multi_objectives"), "multi_objectives")?
    else {
        return Ok(None);
    };
    if objectives_json.is_empty() {
        return Err("multi_objectives must be non-empty".to_string());
    }
    if json_field_has_content(object.get("quadratic_objective"))
        || RUST_SOURCE_CLI_FEATURE_KEYS
            .iter()
            .any(|key| json_field_has_content(object.get(*key)))
    {
        return Ok(None);
    }

    let c = required_f64_array(object, "c")?;
    let n = c.len();
    if let Some(lb) = optional_plain_f64_array_exact(object.get("lb"), n, "lb")? {
        if lb.iter().any(|value| value.abs() > 1e-12) {
            return Ok(None);
        }
    }
    let sense = parse_cli_sense(object.get("sense"))?;
    let a = optional_f64_matrix(object, &["a"])?;
    let b = optional_f64_array(object, &["b"])?;
    validate_source_base_rows(n, &a, &b)?;
    let integer_vars = optional_bool_array(object.get("integer_vars"), n, false, "integer_vars")?;
    let ub = optional_plain_f64_array_exact(object.get("ub"), n, "ub")?;
    let var_names = optional_string_array_exact(object.get("var_names"), n, "var_names")?;
    let con_names = optional_string_array_exact(object.get("con_names"), a.len(), "con_names")?;
    let lazy_constraints = parse_branch_or_cut_constraints(object.get("lazy_constraints"), n)?;
    let mut objectives = Vec::with_capacity(objectives_json.len());
    for (idx, objective_json) in objectives_json.iter().enumerate() {
        let path = format!("multi_objectives[{idx}]");
        let objective_object = required_object(objective_json, &path)?;
        let objective_c = f64_array_from_value(
            objective_object
                .get("c")
                .ok_or_else(|| format!("missing required array '{path}.c'"))?,
            &format!("{path}.c"),
        )?;
        if objective_c.len() != n {
            return Err(format!(
                "{path}.c length {} does not match variable count {n}",
                objective_c.len()
            ));
        }
        objectives.push(LexicographicObjective {
            sense: parse_cli_sense(objective_object.get("sense"))?,
            c: objective_c,
            name: optional_string_field(objective_object.get("name"), &format!("{path}.name"))?,
        });
    }

    Ok(Some(MultiObjectiveIPMIPProblem {
        base: IPMIPProblem {
            sense,
            c,
            a,
            b,
            integer_vars,
            ub,
            var_names,
            con_names,
            lazy_constraints,
            variable_nodes: None,
            constraint_nodes: None,
        },
        objectives,
    }))
}

fn source_ipmip_problem_from_cli_json(
    problem_json: &Value,
) -> Result<Option<SourceIPMIPProblem>, String> {
    let Some(object) = problem_json.as_object() else {
        return Ok(None);
    };
    if json_field_has_content(object.get("quadratic_objective"))
        || json_field_has_content(object.get("multi_objectives"))
        || !RUST_SOURCE_CLI_FEATURE_KEYS
            .iter()
            .any(|key| json_field_has_content(object.get(*key)))
    {
        return Ok(None);
    }

    let c = required_f64_array(object, "c")?;
    let n = c.len();
    let sense = parse_cli_sense(object.get("sense"))?;
    let a = optional_f64_matrix(object, &["a"])?;
    let b = optional_f64_array(object, &["b"])?;
    validate_source_base_rows(n, &a, &b)?;
    let integer_vars = optional_bool_array(object.get("integer_vars"), n, false, "integer_vars")?;
    let lb = optional_plain_f64_array_exact(object.get("lb"), n, "lb")?;
    let ub = optional_plain_f64_array_exact(object.get("ub"), n, "ub")?;
    let var_names = optional_string_array_exact(object.get("var_names"), n, "var_names")?;
    let con_names = optional_string_array_exact(object.get("con_names"), a.len(), "con_names")?;
    let lazy_constraints = parse_branch_or_cut_constraints(object.get("lazy_constraints"), n)?;

    Ok(Some(SourceIPMIPProblem {
        base: IPMIPProblem {
            sense,
            c,
            a,
            b,
            integer_vars,
            ub,
            var_names,
            con_names,
            lazy_constraints,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb,
        linear_constraints: parse_source_linear_constraints(object.get("linear_constraints"), n)?,
        indicators: parse_source_indicators(object.get("indicators"), n)?,
        sos: parse_source_sos_sets(object.get("sos"), n)?,
        semi_variables: parse_source_semi_variables(object.get("semi_variables"), n)?,
        pwl: parse_source_pwl_constraints(object.get("pwl"), n)?,
        abs: parse_source_abs_constraints(object.get("abs"), n)?,
        maximums: parse_source_maximum_constraints(object.get("maximums"), n)?,
        minimums: parse_source_minimum_constraints(object.get("minimums"), n)?,
        logical: parse_source_logical_constraints(object.get("logical"), n)?,
        l1_norms: parse_source_l1_norm_constraints(object.get("l1_norms"), n)?,
        linf_norms: parse_source_linf_norm_constraints(object.get("linf_norms"), n)?,
        products: parse_source_product_constraints(object.get("products"), n)?,
    }))
}

fn validate_source_base_rows(n: usize, a: &[Vec<f64>], b: &[f64]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!(
            "a row count {} does not match b length {}",
            a.len(),
            b.len()
        ));
    }
    for (idx, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(format!(
                "a[{idx}] length {} does not match variable count {n}",
                row.len()
            ));
        }
    }
    Ok(())
}

fn optional_plain_f64_array_exact(
    value: Option<&Value>,
    n: usize,
    name: &str,
) -> Result<Option<Vec<f64>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let values = f64_array_from_value(value, name)?;
    if values.len() != n {
        return Err(format!(
            "{name} length {} does not match variable count {n}",
            values.len()
        ));
    }
    Ok(Some(values))
}

fn optional_string_array_exact(
    value: Option<&Value>,
    n: usize,
    name: &str,
) -> Result<Option<Vec<String>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(values) = value.as_array() else {
        return Err(format!("{name} must be an array"));
    };
    if values.len() != n {
        return Err(format!(
            "{name} length {} does not match expected length {n}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{name}[{idx}] must be a string"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn optional_json_array<'a>(
    value: Option<&'a Value>,
    name: &str,
) -> Result<Option<&'a Vec<Value>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_array()
        .map(Some)
        .ok_or_else(|| format!("{name} must be an array"))
}

fn required_object<'a>(
    value: &'a Value,
    name: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{name} must be an object"))
}

fn optional_string_field(value: Option<&Value>, name: &str) -> Result<Option<String>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_str()
        .map(|value| Some(value.to_string()))
        .ok_or_else(|| format!("{name} must be a string or null"))
}

fn required_usize_field(
    object: &serde_json::Map<String, Value>,
    key: &str,
    name: &str,
) -> Result<usize, String> {
    let Some(value) = object.get(key) else {
        return Err(format!("missing required integer '{name}'"));
    };
    usize_from_value(value, name)
}

fn usize_from_value(value: &Value, name: &str) -> Result<usize, String> {
    let Some(raw) = value.as_u64() else {
        return Err(format!("{name} must be a non-negative integer"));
    };
    usize::try_from(raw).map_err(|_| format!("{name} is too large for this platform"))
}

fn required_bool_field(
    object: &serde_json::Map<String, Value>,
    key: &str,
    name: &str,
) -> Result<bool, String> {
    let Some(value) = object.get(key) else {
        return Err(format!("missing required boolean '{name}'"));
    };
    value
        .as_bool()
        .ok_or_else(|| format!("{name} must be a boolean"))
}

fn validate_source_var_index(index: usize, n: usize, name: &str) -> Result<(), String> {
    if index >= n {
        return Err(format!(
            "{name} index {index} is outside variable count {n}"
        ));
    }
    Ok(())
}

fn required_index_array(
    object: &serde_json::Map<String, Value>,
    key: &str,
    n: usize,
    name: &str,
) -> Result<Vec<usize>, String> {
    let Some(value) = object.get(key) else {
        return Err(format!("missing required array '{name}'"));
    };
    let Some(values) = value.as_array() else {
        return Err(format!("{name} must be an array"));
    };
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let index = usize_from_value(value, &format!("{name}[{idx}]"))?;
            validate_source_var_index(index, n, &format!("{name}[{idx}]"))?;
            Ok(index)
        })
        .collect()
}

fn parse_constraint_kind(value: Option<&Value>, name: &str) -> Result<ConstraintKind, String> {
    let Some(value) = value else {
        return Ok(ConstraintKind::Lazy);
    };
    if value.is_null() {
        return Ok(ConstraintKind::Lazy);
    }
    let Some(text) = value.as_str() else {
        return Err(format!("{name} must be a string"));
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "branch" => Ok(ConstraintKind::Branch),
        "cut" => Ok(ConstraintKind::Cut),
        "lazy" => Ok(ConstraintKind::Lazy),
        other => Err(format!("{name} has unknown constraint kind '{other}'")),
    }
}

fn parse_branch_or_cut_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Option<Vec<BranchOrCutConstraint>>, String> {
    let Some(rows) = optional_json_array(value, "lazy_constraints")? else {
        return Ok(None);
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("lazy_constraints[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let coefs = required_f64_array(row_object, "coefs")?;
            if coefs.len() != n {
                return Err(format!(
                    "{path}.coefs length {} does not match variable count {n}",
                    coefs.len()
                ));
            }
            Ok(BranchOrCutConstraint {
                coefs,
                rhs: required_f64_field(row_object.get("rhs"), &format!("{path}.rhs"))?,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?
                    .unwrap_or_else(|| format!("lazy_constraint_{idx}")),
                kind: parse_constraint_kind(row_object.get("kind"), &format!("{path}.kind"))?,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn parse_source_linear_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<LinearRowConstraint>, String> {
    let Some(rows) = optional_json_array(value, "linear_constraints")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("linear_constraints[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let coefs = required_f64_array(row_object, "coefs")?;
            if coefs.len() != n {
                return Err(format!(
                    "{path}.coefs length {} does not match variable count {n}",
                    coefs.len()
                ));
            }
            Ok(LinearRowConstraint {
                coefs,
                lower: optional_f64_field(row_object.get("lower"), &format!("{path}.lower"))?,
                upper: optional_f64_field(row_object.get("upper"), &format!("{path}.upper"))?,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_indicator_sense(value: Option<&Value>, name: &str) -> Result<IndicatorSense, String> {
    let Some(value) = value else {
        return Err(format!("missing required string '{name}'"));
    };
    let Some(text) = value.as_str() else {
        return Err(format!("{name} must be a string"));
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "le" | "<=" => Ok(IndicatorSense::Le),
        "ge" | ">=" => Ok(IndicatorSense::Ge),
        "eq" | "=" | "==" => Ok(IndicatorSense::Eq),
        other => Err(format!("{name} has unknown indicator sense '{other}'")),
    }
}

fn parse_source_indicators(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<IndicatorConstraint>, String> {
    let Some(rows) = optional_json_array(value, "indicators")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("indicators[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let binary_var =
                required_usize_field(row_object, "binary_var", &format!("{path}.binary_var"))?;
            validate_source_var_index(binary_var, n, &format!("{path}.binary_var"))?;
            let coefs = required_f64_array(row_object, "coefs")?;
            if coefs.len() != n {
                return Err(format!(
                    "{path}.coefs length {} does not match variable count {n}",
                    coefs.len()
                ));
            }
            Ok(IndicatorConstraint {
                binary_var,
                active_value: required_bool_field(
                    row_object,
                    "active_value",
                    &format!("{path}.active_value"),
                )?,
                coefs,
                sense: parse_indicator_sense(row_object.get("sense"), &format!("{path}.sense"))?,
                rhs: required_f64_field(row_object.get("rhs"), &format!("{path}.rhs"))?,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_sos_kind(value: Option<&Value>, name: &str) -> Result<SpecialOrderedSetKind, String> {
    let Some(value) = value else {
        return Err(format!("missing required string '{name}'"));
    };
    let Some(text) = value.as_str() else {
        return Err(format!("{name} must be a string"));
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "sos1" | "1" => Ok(SpecialOrderedSetKind::Sos1),
        "sos2" | "2" => Ok(SpecialOrderedSetKind::Sos2),
        other => Err(format!("{name} has unknown SOS kind '{other}'")),
    }
}

fn optional_plain_f64_array_len(
    value: Option<&Value>,
    len: usize,
    name: &str,
) -> Result<Option<Vec<f64>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let values = f64_array_from_value(value, name)?;
    if values.len() != len {
        return Err(format!(
            "{name} length {} does not match expected length {len}",
            values.len()
        ));
    }
    Ok(Some(values))
}

fn parse_source_sos_sets(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<SpecialOrderedSet>, String> {
    let Some(rows) = optional_json_array(value, "sos")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("sos[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let vars = required_index_array(row_object, "vars", n, &format!("{path}.vars"))?;
            Ok(SpecialOrderedSet {
                kind: parse_sos_kind(row_object.get("kind"), &format!("{path}.kind"))?,
                weights: optional_plain_f64_array_len(
                    row_object.get("weights"),
                    vars.len(),
                    &format!("{path}.weights"),
                )?,
                vars,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_semi_variable_kind(value: Option<&Value>, name: &str) -> Result<SemiVariableKind, String> {
    let Some(value) = value else {
        return Err(format!("missing required string '{name}'"));
    };
    let Some(text) = value.as_str() else {
        return Err(format!("{name} must be a string"));
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "semi_continuous" | "semi-continuous" | "semicontinuous" => {
            Ok(SemiVariableKind::SemiContinuous)
        }
        "semi_integer" | "semi-integer" | "semiinteger" => Ok(SemiVariableKind::SemiInteger),
        other => Err(format!("{name} has unknown semi-variable kind '{other}'")),
    }
}

fn parse_source_semi_variables(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<SemiVariable>, String> {
    let Some(rows) = optional_json_array(value, "semi_variables")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("semi_variables[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let var = required_usize_field(row_object, "var", &format!("{path}.var"))?;
            validate_source_var_index(var, n, &format!("{path}.var"))?;
            Ok(SemiVariable {
                kind: parse_semi_variable_kind(row_object.get("kind"), &format!("{path}.kind"))?,
                var,
                lower: required_f64_field(row_object.get("lower"), &format!("{path}.lower"))?,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_source_pwl_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<PiecewiseLinearConstraint>, String> {
    let Some(rows) = optional_json_array(value, "pwl")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("pwl[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let x_var = required_usize_field(row_object, "x_var", &format!("{path}.x_var"))?;
            let y_var = required_usize_field(row_object, "y_var", &format!("{path}.y_var"))?;
            validate_source_var_index(x_var, n, &format!("{path}.x_var"))?;
            validate_source_var_index(y_var, n, &format!("{path}.y_var"))?;
            let Some(points) =
                optional_json_array(row_object.get("points"), &format!("{path}.points"))?
            else {
                return Err(format!("missing required array '{}.points'", path));
            };
            let points = points
                .iter()
                .enumerate()
                .map(|(point_idx, point_value)| {
                    let point_path = format!("{path}.points[{point_idx}]");
                    let point_object = required_object(point_value, &point_path)?;
                    Ok::<PiecewiseLinearPoint, String>(PiecewiseLinearPoint {
                        x: required_f64_field(point_object.get("x"), &format!("{point_path}.x"))?,
                        y: required_f64_field(point_object.get("y"), &format!("{point_path}.y"))?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(PiecewiseLinearConstraint {
                x_var,
                y_var,
                points,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_source_abs_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<AbsoluteValueConstraint>, String> {
    let Some(rows) = optional_json_array(value, "abs")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("abs[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let arg_var = required_usize_field(row_object, "arg_var", &format!("{path}.arg_var"))?;
            let target_var =
                required_usize_field(row_object, "target_var", &format!("{path}.target_var"))?;
            validate_source_var_index(arg_var, n, &format!("{path}.arg_var"))?;
            validate_source_var_index(target_var, n, &format!("{path}.target_var"))?;
            Ok(AbsoluteValueConstraint {
                arg_var,
                target_var,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_source_maximum_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<MaximumConstraint>, String> {
    let Some(rows) = optional_json_array(value, "maximums")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("maximums[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let target_var =
                required_usize_field(row_object, "target_var", &format!("{path}.target_var"))?;
            validate_source_var_index(target_var, n, &format!("{path}.target_var"))?;
            Ok(MaximumConstraint {
                target_var,
                arg_vars: required_index_array(
                    row_object,
                    "arg_vars",
                    n,
                    &format!("{path}.arg_vars"),
                )?,
                constant: optional_f64_field(
                    row_object.get("constant"),
                    &format!("{path}.constant"),
                )?,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_source_minimum_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<MinimumConstraint>, String> {
    let Some(rows) = optional_json_array(value, "minimums")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("minimums[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let target_var =
                required_usize_field(row_object, "target_var", &format!("{path}.target_var"))?;
            validate_source_var_index(target_var, n, &format!("{path}.target_var"))?;
            Ok(MinimumConstraint {
                target_var,
                arg_vars: required_index_array(
                    row_object,
                    "arg_vars",
                    n,
                    &format!("{path}.arg_vars"),
                )?,
                constant: optional_f64_field(
                    row_object.get("constant"),
                    &format!("{path}.constant"),
                )?,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_logical_constraint_kind(
    value: Option<&Value>,
    name: &str,
) -> Result<LogicalConstraintKind, String> {
    let Some(value) = value else {
        return Err(format!("missing required string '{name}'"));
    };
    let Some(text) = value.as_str() else {
        return Err(format!("{name} must be a string"));
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "and" => Ok(LogicalConstraintKind::And),
        "or" => Ok(LogicalConstraintKind::Or),
        other => Err(format!("{name} has unknown logical kind '{other}'")),
    }
}

fn parse_source_logical_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<LogicalConstraint>, String> {
    let Some(rows) = optional_json_array(value, "logical")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("logical[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let target_var =
                required_usize_field(row_object, "target_var", &format!("{path}.target_var"))?;
            validate_source_var_index(target_var, n, &format!("{path}.target_var"))?;
            Ok(LogicalConstraint {
                kind: parse_logical_constraint_kind(
                    row_object.get("kind"),
                    &format!("{path}.kind"),
                )?,
                target_var,
                arg_vars: required_index_array(
                    row_object,
                    "arg_vars",
                    n,
                    &format!("{path}.arg_vars"),
                )?,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_source_l1_norm_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<L1NormConstraint>, String> {
    let Some(rows) = optional_json_array(value, "l1_norms")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("l1_norms[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let target_var =
                required_usize_field(row_object, "target_var", &format!("{path}.target_var"))?;
            validate_source_var_index(target_var, n, &format!("{path}.target_var"))?;
            Ok(L1NormConstraint {
                target_var,
                arg_vars: required_index_array(
                    row_object,
                    "arg_vars",
                    n,
                    &format!("{path}.arg_vars"),
                )?,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_source_linf_norm_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<LInfNormConstraint>, String> {
    let Some(rows) = optional_json_array(value, "linf_norms")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("linf_norms[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let target_var =
                required_usize_field(row_object, "target_var", &format!("{path}.target_var"))?;
            validate_source_var_index(target_var, n, &format!("{path}.target_var"))?;
            Ok(LInfNormConstraint {
                target_var,
                arg_vars: required_index_array(
                    row_object,
                    "arg_vars",
                    n,
                    &format!("{path}.arg_vars"),
                )?,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn parse_source_product_constraints(
    value: Option<&Value>,
    n: usize,
) -> Result<Vec<ProductConstraint>, String> {
    let Some(rows) = optional_json_array(value, "products")? else {
        return Ok(Vec::new());
    };
    rows.iter()
        .enumerate()
        .map(|(idx, row_value)| {
            let path = format!("products[{idx}]");
            let row_object = required_object(row_value, &path)?;
            let target_var =
                required_usize_field(row_object, "target_var", &format!("{path}.target_var"))?;
            let x_var = required_usize_field(row_object, "x_var", &format!("{path}.x_var"))?;
            let y_var = required_usize_field(row_object, "y_var", &format!("{path}.y_var"))?;
            validate_source_var_index(target_var, n, &format!("{path}.target_var"))?;
            validate_source_var_index(x_var, n, &format!("{path}.x_var"))?;
            validate_source_var_index(y_var, n, &format!("{path}.y_var"))?;
            Ok(ProductConstraint {
                target_var,
                x_var,
                y_var,
                name: optional_string_field(row_object.get("name"), &format!("{path}.name"))?,
            })
        })
        .collect()
}

fn plain_linear_model_to_string(
    model: &PlainLinearCliModel,
    model_format: ExternalLinearCliModelFormat,
    include_objsense: bool,
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
            include_objsense,
        ),
    }
}

fn plain_linear_model_to_lpsolve_lp_string(model: &PlainLinearCliModel) -> String {
    lpsolve_lp_string(
        model.sense,
        &model.c,
        &model.le_rows,
        &model.le_rhs,
        &model.eq_rows,
        &model.eq_rhs,
        &model.lbs,
        &model.ubs,
        &model.integer_vars,
    )
}

fn lp_problem_to_plain_linear_model(problem: &LPProblem) -> PlainLinearCliModel {
    let n = problem.c.len();
    PlainLinearCliModel {
        sense: problem.sense,
        c: problem.c.clone(),
        le_rows: problem.a_ub.clone().unwrap_or_default(),
        le_rhs: problem.b_ub.clone().unwrap_or_default(),
        eq_rows: problem.a_eq.clone().unwrap_or_default(),
        eq_rhs: problem.b_eq.clone().unwrap_or_default(),
        lbs: problem.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]),
        ubs: problem.ub.clone().unwrap_or_else(|| vec![None; n]),
        integer_vars: vec![false; n],
    }
}

fn ipmip_problem_to_plain_linear_model(problem: &IPMIPProblem) -> PlainLinearCliModel {
    let n = problem.c.len();
    let (le_rows, le_rhs) = ipmip_le_rows_with_lazy(problem);
    PlainLinearCliModel {
        sense: problem.sense,
        c: problem.c.clone(),
        le_rows,
        le_rhs,
        eq_rows: Vec::new(),
        eq_rhs: Vec::new(),
        lbs: vec![Some(0.0); n],
        ubs: problem
            .ub
            .as_ref()
            .map(|upper| upper.iter().copied().map(Some).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![None; n]),
        integer_vars: problem.integer_vars.clone(),
    }
}

fn solve_native_mip_solution_pool_cli_model(
    mut working: PlainLinearCliModel,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = format!("{}:cli", opts.solver.as_str());
    let solution_pool_size = opts.solution_pool_size.unwrap_or(1).max(1);
    let original_n = working.c.len();
    let original_c = working.c.clone();
    let integer_indices = solution_pool_integer_indices(&working.integer_vars);
    if let Some(message) =
        validate_solution_pool_bounds(&working.lbs, &working.ubs, &integer_indices)
    {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            message,
            elapsed_ms(t0),
        );
    }

    let mut solutions = Vec::new();
    let mut seen = HashSet::new();
    let mut exhausted = false;
    let mut message = String::new();
    let mut overall_status = ExternalLinearCliStatus::Optimal;
    let mut last_result = None;

    for pool_idx in 0..solution_pool_size {
        let mut stage_opts = opts.clone();
        stage_opts.solution_pool_size = None;
        if pool_idx > 0 {
            stage_opts.mip_start = None;
        }
        if let Some(branch_priorities) = opts.branch_priorities.as_deref() {
            let mut working_priorities = branch_priorities
                .iter()
                .copied()
                .take(working.c.len())
                .collect::<Vec<_>>();
            working_priorities.resize(working.c.len(), 0);
            stage_opts.branch_priorities = Some(working_priorities);
        }

        let result = solve_native_mip_solution_pool_stage(&working, &stage_opts);
        if matches!(
            result.status,
            ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
        ) {
            exhausted = result.status == ExternalLinearCliStatus::Infeasible;
            message = if exhausted {
                "pool exhausted by no-good cuts".to_string()
            } else {
                result.message.clone()
            };
            last_result = Some(result);
            break;
        }
        if !matches!(
            result.status,
            ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
        ) {
            overall_status = if solutions.is_empty() {
                result.status
            } else {
                ExternalLinearCliStatus::Feasible
            };
            message = result.message.clone();
            last_result = Some(result);
            break;
        }
        if result.status == ExternalLinearCliStatus::Feasible {
            overall_status = ExternalLinearCliStatus::Feasible;
        }
        if result.x.len() < original_n {
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                format!(
                    "solution pool stage returned {} values for {original_n} original variables",
                    result.x.len()
                ),
                elapsed_ms(t0),
            );
        }

        let x = result.x[..original_n].to_vec();
        let key = solution_pool_assignment_key(&x, &integer_indices);
        if !seen.insert(key) {
            overall_status = ExternalLinearCliStatus::Feasible;
            message = "pool search stopped after duplicate integer assignment".to_string();
            last_result = Some(result);
            break;
        }

        let objective = dot_f64(&original_c, &x);
        solutions.push(ExternalLinearCliPoolMember {
            x: x.clone(),
            objective,
        });
        if let Some(cut_error) = add_solution_pool_no_good_cut(&mut working, &integer_indices, &x) {
            exhausted = true;
            message = cut_error;
            last_result = Some(result);
            break;
        }
        last_result = Some(result);
    }

    if solutions.len() as u64 == solution_pool_size && !exhausted {
        message = "pool reached solution_pool_size".to_string();
    }

    let last_result = last_result.unwrap_or_else(|| {
        external_cli_failure(
            overall_status,
            bridge_solver.clone(),
            message.clone(),
            elapsed_ms(t0),
        )
    });
    if solutions.is_empty() {
        let mut failure = external_cli_failure(
            if exhausted {
                ExternalLinearCliStatus::Infeasible
            } else {
                overall_status
            },
            bridge_solver,
            message,
            elapsed_ms(t0),
        );
        failure.solver_version = last_result.solver_version;
        failure.solutions = Some(Vec::new());
        failure.solution_pool_size = Some(solution_pool_size);
        failure.exhausted = Some(exhausted);
        return failure;
    }

    let first = solutions[0].clone();
    ExternalLinearCliSolution {
        status: overall_status,
        solver: bridge_solver,
        solver_version: last_result.solver_version,
        x: first.x,
        objective: Some(first.objective),
        objective_values: None,
        lp_algorithm: None,
        best_bound: None,
        solution_limit: None,
        solution_pool_size: Some(solution_pool_size),
        solutions: Some(solutions),
        exhausted: Some(exhausted),
        mip_gap: None,
        absolute_gap: None,
        objective_limit: None,
        primal_feasibility_tolerance: last_result.primal_feasibility_tolerance,
        dual_feasibility_tolerance: last_result.dual_feasibility_tolerance,
        integer_feasibility_tolerance: last_result.integer_feasibility_tolerance,
        nodes_explored: None,
        threads: None,
        random_seed: None,
        presolve: None,
        cuts: None,
        heuristics: None,
        branch_rule: None,
        branch_priorities_accepted: last_result.branch_priorities_accepted,
        branch_priority_count: last_result.branch_priority_count,
        node_selection: None,
        mip_start_accepted: None,
        mip_start_objective: None,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        iterations: None,
        elapsed_ms: elapsed_ms(t0),
        message,
    }
}

fn solve_native_mip_solution_pool_stage(
    model: &PlainLinearCliModel,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = plain_linear_model_to_string(model, opts.model_format, true);
    match opts.solver {
        ExternalLinearCliSolver::Scip => solve_native_scip_cli_model(
            ExternalLinearCliKind::Mip,
            &model_text,
            model.c.len(),
            model.sense,
            &model.le_rows,
            &model.le_rhs,
            &model.eq_rows,
            &model.eq_rhs,
            Some(&model.lbs),
            Some(&model.ubs),
            Some(&model.integer_vars),
            &model.c,
            opts,
        ),
        ExternalLinearCliSolver::Cbc => solve_native_cbc_cli_model(
            ExternalLinearCliKind::Mip,
            model.sense,
            &model_text,
            model.c.len(),
            model.le_rows.len(),
            model.eq_rows.len(),
            Some(&model.integer_vars),
            &model.c,
            opts,
        ),
        solver => external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            format!("{}:cli", solver.as_str()),
            "native solution pool is only implemented for scip and cbc".to_string(),
            0.0,
        ),
    }
}

fn solution_pool_integer_indices(integer_vars: &[bool]) -> Vec<usize> {
    integer_vars
        .iter()
        .enumerate()
        .filter_map(|(idx, is_integer)| is_integer.then_some(idx))
        .collect()
}

fn validate_solution_pool_bounds(
    lbs: &[Option<f64>],
    ubs: &[Option<f64>],
    integer_indices: &[usize],
) -> Option<String> {
    if integer_indices.is_empty() {
        return Some("solution pool requires at least one integer variable".to_string());
    }
    for &idx in integer_indices {
        let lb = lbs.get(idx).copied().flatten().unwrap_or(0.0);
        let Some(ub) = ubs.get(idx).copied().flatten() else {
            return Some(format!(
                "solution pool requires finite upper bound for integer variable x{idx}"
            ));
        };
        if !lb.is_finite() {
            return Some(format!(
                "solution pool requires finite lower bound for integer variable x{idx}"
            ));
        }
        if !ub.is_finite() {
            return Some(format!(
                "solution pool requires finite upper bound for integer variable x{idx}"
            ));
        }
        if (lb - lb.round()).abs() > 1.0e-9 || (ub - ub.round()).abs() > 1.0e-9 {
            return Some(format!(
                "solution pool requires integral bounds for integer variable x{idx}"
            ));
        }
    }
    None
}

fn solution_pool_assignment_key(x: &[f64], integer_indices: &[usize]) -> Vec<i64> {
    integer_indices
        .iter()
        .map(|&idx| x[idx].round() as i64)
        .collect()
}

fn add_solution_pool_no_good_cut(
    working: &mut PlainLinearCliModel,
    integer_indices: &[usize],
    assignment: &[f64],
) -> Option<String> {
    let n = working.c.len();
    if working.lbs.len() != n || working.ubs.len() != n || working.integer_vars.len() != n {
        return Some("solution pool no-good cut saw inconsistent working model dimensions".into());
    }
    if working
        .le_rows
        .iter()
        .chain(&working.eq_rows)
        .any(|row| row.len() != n)
    {
        return Some("solution pool no-good cut saw inconsistent working row dimensions".into());
    }

    let mut deviation_vars = Vec::new();
    for &idx in integer_indices {
        if idx >= assignment.len() {
            return Some(format!(
                "solution pool assignment is missing integer variable x{idx}"
            ));
        }
        let value = assignment[idx].round();
        let lb = working.lbs[idx].unwrap_or(0.0);
        let Some(ub) = working.ubs[idx] else {
            return Some(format!(
                "solution pool requires finite upper bound for integer variable x{idx}"
            ));
        };
        if value < lb - 1.0e-9 || value > ub + 1.0e-9 {
            return Some(format!(
                "solution pool assignment for x{idx} is outside its bounds"
            ));
        }

        if value > lb + 1.0e-9 {
            let deviation = append_solution_pool_deviation_var(working);
            let mut row = vec![0.0; working.c.len()];
            row[idx] = 1.0;
            row[deviation] = ub - value + 1.0;
            working.le_rows.push(row);
            working.le_rhs.push(ub);
            deviation_vars.push(deviation);
        }

        if value < ub - 1.0e-9 {
            let deviation = append_solution_pool_deviation_var(working);
            let mut row = vec![0.0; working.c.len()];
            row[idx] = -1.0;
            row[deviation] = value + 1.0 - lb;
            working.le_rows.push(row);
            working.le_rhs.push(-lb);
            deviation_vars.push(deviation);
        }
    }

    if deviation_vars.is_empty() {
        return Some(
            "solution pool could not create a no-good cut for a singleton integer domain"
                .to_string(),
        );
    }

    let mut row = vec![0.0; working.c.len()];
    for deviation in deviation_vars {
        row[deviation] = -1.0;
    }
    working.le_rows.push(row);
    working.le_rhs.push(-1.0);
    None
}

fn append_solution_pool_deviation_var(working: &mut PlainLinearCliModel) -> usize {
    let deviation = working.c.len();
    working.c.push(0.0);
    working.lbs.push(Some(0.0));
    working.ubs.push(Some(1.0));
    working.integer_vars.push(true);
    for row in working.le_rows.iter_mut().chain(&mut working.eq_rows) {
        row.push(0.0);
    }
    deviation
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

fn validate_plain_linear_model_dimensions(
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

fn solve_lp_with_native_cbc_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => lp_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => lp_problem_to_mps_string(problem),
    };
    let le_count = problem.a_ub.as_ref().map_or(0, Vec::len);
    let eq_count = problem.a_eq.as_ref().map_or(0, Vec::len);
    solve_native_cbc_cli_model(
        ExternalLinearCliKind::Lp,
        problem.sense,
        &model_text,
        problem.c.len(),
        le_count,
        eq_count,
        None,
        &problem.c,
        opts,
    )
}

fn solve_ipmip_with_native_cbc_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    if opts.solution_pool_size.is_some() {
        return solve_native_mip_solution_pool_cli_model(
            ipmip_problem_to_plain_linear_model(problem),
            opts,
        );
    }
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => ipmip_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => ipmip_problem_to_mps_string(problem),
    };
    solve_native_cbc_cli_model(
        ExternalLinearCliKind::Mip,
        problem.sense,
        &model_text,
        problem.c.len(),
        ipmip_total_le_row_count(problem),
        0,
        Some(&problem.integer_vars),
        &problem.c,
        opts,
    )
}

fn should_use_native_clp_cli(opts: &ExternalLinearCliOptions) -> bool {
    opts.solver == ExternalLinearCliSolver::Clp
        && opts.script_path.is_none()
        && opts.lp_algorithm.is_none()
        && opts.max_nodes.is_none()
        && opts.node_limit.is_none()
        && opts.solution_limit.is_none()
        && opts.solution_pool_size.is_none()
        && opts.relative_gap.is_none()
        && opts.absolute_gap.is_none()
        && opts.objective_limit.is_none()
        && opts.primal_feasibility_tolerance.is_none()
        && opts.dual_feasibility_tolerance.is_none()
        && opts.integer_feasibility_tolerance.is_none()
        && opts.threads.is_none()
        && opts.random_seed.is_none()
        && opts.presolve.is_none()
        && opts.cuts.is_none()
        && opts.heuristics.is_none()
        && opts.branch_rule.is_none()
        && opts.branch_priorities.is_none()
        && opts.node_selection.is_none()
        && opts.mip_start.is_none()
}

fn solve_lp_with_native_clp_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => lp_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => lp_problem_to_mps_string(problem),
    };
    let le_count = problem.a_ub.as_ref().map_or(0, Vec::len);
    let eq_count = problem.a_eq.as_ref().map_or(0, Vec::len);
    solve_native_clp_cli_model(
        problem.sense,
        &model_text,
        problem.c.len(),
        le_count,
        eq_count,
        &problem.c,
        opts,
    )
}

fn should_use_native_soplex_cli(opts: &ExternalLinearCliOptions) -> bool {
    opts.solver == ExternalLinearCliSolver::Soplex
        && opts.script_path.is_none()
        && opts.lp_algorithm.is_none()
        && opts.max_nodes.is_none()
        && opts.node_limit.is_none()
        && opts.solution_limit.is_none()
        && opts.solution_pool_size.is_none()
        && opts.relative_gap.is_none()
        && opts.absolute_gap.is_none()
        && opts.objective_limit.is_none()
        && opts.primal_feasibility_tolerance.is_none()
        && opts.dual_feasibility_tolerance.is_none()
        && opts.integer_feasibility_tolerance.is_none()
        && opts.threads.is_none()
        && opts.random_seed.is_none()
        && opts.presolve.is_none()
        && opts.cuts.is_none()
        && opts.heuristics.is_none()
        && opts.branch_rule.is_none()
        && opts.branch_priorities.is_none()
        && opts.node_selection.is_none()
        && opts.mip_start.is_none()
}

fn solve_lp_with_native_soplex_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => lp_problem_to_cplex_lp_string(problem),
        ExternalLinearCliModelFormat::Mps => lp_problem_to_mps_string(problem),
    };
    let le_count = problem.a_ub.as_ref().map_or(0, Vec::len);
    let eq_count = problem.a_eq.as_ref().map_or(0, Vec::len);
    solve_native_soplex_cli_model(
        &model_text,
        problem.c.len(),
        le_count,
        eq_count,
        &problem.c,
        opts,
    )
}

fn should_use_native_qsopt_ex_cli(opts: &ExternalLinearCliOptions) -> bool {
    opts.solver == ExternalLinearCliSolver::QsoptEx
        && opts.script_path.is_none()
        && opts.lp_algorithm.is_none()
        && opts.max_nodes.is_none()
        && opts.node_limit.is_none()
        && opts.solution_limit.is_none()
        && opts.solution_pool_size.is_none()
        && opts.relative_gap.is_none()
        && opts.absolute_gap.is_none()
        && opts.objective_limit.is_none()
        && opts.primal_feasibility_tolerance.is_none()
        && opts.dual_feasibility_tolerance.is_none()
        && opts.integer_feasibility_tolerance.is_none()
        && opts.threads.is_none()
        && opts.random_seed.is_none()
        && opts.presolve.is_none()
        && opts.cuts.is_none()
        && opts.heuristics.is_none()
        && opts.branch_rule.is_none()
        && opts.branch_priorities.is_none()
        && opts.node_selection.is_none()
        && opts.mip_start.is_none()
}

fn solve_lp_with_native_qsopt_ex_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = lp_problem_to_cplex_lp_string(problem);
    let le_rows = problem.a_ub.as_deref().unwrap_or(&[]);
    let le_rhs = problem.b_ub.as_deref().unwrap_or(&[]);
    let eq_rows = problem.a_eq.as_deref().unwrap_or(&[]);
    let eq_rhs = problem.b_eq.as_deref().unwrap_or(&[]);
    solve_native_qsopt_ex_cli_model(
        &model_text,
        problem.c.len(),
        le_rows,
        le_rhs,
        eq_rows,
        eq_rhs,
        problem.lb.as_deref(),
        problem.ub.as_deref(),
        &problem.c,
        opts,
    )
}

fn should_use_native_lp_solve_cli(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
) -> bool {
    if opts.solver != ExternalLinearCliSolver::LpSolve
        || opts.script_path.is_some()
        || opts.lp_algorithm.is_some()
        || opts.solution_limit.is_some()
        || opts.solution_pool_size.is_some()
        || opts.objective_limit.is_some()
        || opts.primal_feasibility_tolerance.is_some()
        || opts.dual_feasibility_tolerance.is_some()
        || opts.integer_feasibility_tolerance.is_some()
        || opts.presolve.is_some()
        || opts.cuts.is_some()
        || opts.heuristics.is_some()
        || opts.branch_rule.is_some()
        || opts.branch_priorities.is_some()
        || opts.node_selection.is_some()
        || opts.mip_start.is_some()
    {
        return false;
    }

    if kind == ExternalLinearCliKind::Lp {
        return opts.max_nodes.is_none()
            && opts.node_limit.is_none()
            && opts.relative_gap.is_none()
            && opts.absolute_gap.is_none();
    }

    true
}

fn solve_lp_with_native_lp_solve_cli(
    problem: &LPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = lp_problem_to_lpsolve_lp_string(problem);
    let le_count = problem.a_ub.as_ref().map_or(0, Vec::len);
    let eq_count = problem.a_eq.as_ref().map_or(0, Vec::len);
    solve_native_lp_solve_cli_model(
        ExternalLinearCliKind::Lp,
        &model_text,
        problem.c.len(),
        le_count,
        eq_count,
        &problem.c,
        opts,
    )
}

fn solve_ipmip_with_native_lp_solve_cli(
    problem: &IPMIPProblem,
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let model_text = ipmip_problem_to_lpsolve_lp_string(problem);
    solve_native_lp_solve_cli_model(
        ExternalLinearCliKind::Mip,
        &model_text,
        problem.c.len(),
        ipmip_total_le_row_count(problem),
        0,
        &problem.c,
        opts,
    )
}

fn solve_native_qsopt_ex_cli_model(
    model_text: &str,
    variable_count: usize,
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    lower_bounds: Option<&[Option<f64>]>,
    upper_bounds: Option<&[Option<f64>]>,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "qsopt-ex:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::QsoptEx, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "qsopt_ex executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let model_path = native_qsopt_ex_temp_path("model", "lp");
    let solution_path = native_qsopt_ex_temp_path("solution", "sol");
    let basis_path = native_qsopt_ex_temp_path("basis", "bas");
    let cleanup_paths = vec![
        model_path.clone(),
        solution_path.clone(),
        basis_path.clone(),
    ];
    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_qsopt_ex_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write QSopt_ex model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let output = match Command::new(&command_path)
        .arg("-L")
        .arg("-O")
        .arg(&solution_path)
        .arg("-b")
        .arg(&basis_path)
        .arg(&model_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_qsopt_ex_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start QSopt_ex executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_qsopt_ex_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_qsopt_ex_solver_version(&command_path));

    if !solution_path.exists() {
        let status = classify_native_linear_status("", &stdout, &stderr);
        cleanup_native_qsopt_ex_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message("", &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let parsed = match parse_native_qsopt_ex_solution_file(
        &solution_path,
        Some(&basis_path),
        variable_count,
        le_rows.len(),
        eq_rows.len(),
        &stdout,
        &stderr,
    ) {
        Ok(parsed) => parsed,
        Err(message) => {
            cleanup_native_qsopt_ex_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    cleanup_native_qsopt_ex_temp_files(&cleanup_paths);

    let mut parsed = parsed;
    if qsopt_ex_lp_certificate_needs_reconstruction(
        &parsed,
        objective_coefficients,
        le_rows,
        eq_rows,
    ) {
        if let Some((dual_ub, dual_eq, reduced_costs)) =
            qsopt_ex_lp_certificate_from_basis(&parsed, objective_coefficients, le_rows, eq_rows)
        {
            parsed.dual_ub = Some(dual_ub);
            parsed.dual_eq = Some(dual_eq);
            parsed.reduced_costs = Some(reduced_costs);
        }
    }
    if parsed.var_basis.is_none() || parsed.row_basis.is_none() {
        let (inferred_var_basis, inferred_row_basis) = qsopt_ex_lp_basis_from_solution(
            &parsed,
            le_rows,
            le_rhs,
            eq_rows,
            eq_rhs,
            lower_bounds,
            upper_bounds,
        );
        if parsed.var_basis.is_none() {
            parsed.var_basis = inferred_var_basis;
        }
        if parsed.row_basis.is_none() {
            parsed.row_basis = inferred_row_basis;
        }
    }

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(dot_f64(objective_coefficients, &parsed.x)),
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
        dual_ub: parsed.dual_ub,
        dual_eq: parsed.dual_eq,
        reduced_costs: parsed.reduced_costs,
        var_basis: parsed.var_basis,
        row_basis: parsed.row_basis,
        iterations: None,
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_lp_solve_cli_model(
    kind: ExternalLinearCliKind,
    model_text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "lp-solve:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::LpSolve, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "lp_solve executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let model_path = native_lp_solve_temp_path("model", "lp");
    let basis_path = native_lp_solve_temp_path("basis", "bas");
    let cleanup_paths = vec![model_path.clone(), basis_path.clone()];
    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_lp_solve_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write lp_solve model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let mut command = Command::new(&command_path);
    command
        .arg("-timeout")
        .arg(glpk_time_limit_arg(opts.time_limit_secs));
    if kind == ExternalLinearCliKind::Lp {
        command.arg("-S4").arg("-wbas").arg(&basis_path);
    } else if kind == ExternalLinearCliKind::Mip {
        if !matches!(opts.presolve, Some(ExternalLinearCliPresolve::Off)) {
            command.arg("-presolve");
        }
        command.arg("-v5").arg("-S2");
        if let Some(relative_gap) = normalized_relative_gap(opts.relative_gap) {
            command.arg("-gr").arg(format!("{relative_gap:.17}"));
        }
        if let Some(absolute_gap) = normalized_absolute_gap(opts.absolute_gap) {
            command.arg("-ga").arg(format!("{absolute_gap:.17}"));
        }
    }
    command
        .arg(&model_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_lp_solve_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start lp_solve executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_lp_solve_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_lp_solve_solver_version(&command_path));
    let parsed =
        parse_native_lp_solve_solution_text(&format!("{stdout}\n{stderr}"), variable_count);
    let certificate = (kind == ExternalLinearCliKind::Lp).then(|| {
        parse_native_lp_solve_lp_certificate_fields(
            &format!("{stdout}\n{stderr}"),
            basis_path.as_path(),
            variable_count,
            le_count,
            eq_count,
        )
    });
    cleanup_native_lp_solve_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let objective = parsed
        .objective
        .unwrap_or_else(|| dot_f64(objective_coefficients, &parsed.x));
    let quality = parse_lp_solve_mip_quality(
        kind,
        status,
        objective,
        &stdout,
        &stderr,
        opts.max_nodes.is_some() || opts.node_limit.is_some(),
    );

    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(objective),
        objective_values: None,
        lp_algorithm: None,
        best_bound: quality.best_bound,
        solution_limit: None,
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: quality.mip_gap,
        absolute_gap: quality.absolute_gap,
        objective_limit: None,
        primal_feasibility_tolerance: None,
        dual_feasibility_tolerance: None,
        integer_feasibility_tolerance: None,
        nodes_explored: quality.nodes_explored,
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
        dual_ub: certificate
            .as_ref()
            .and_then(|fields| fields.dual_ub.clone()),
        dual_eq: certificate
            .as_ref()
            .and_then(|fields| fields.dual_eq.clone()),
        reduced_costs: certificate
            .as_ref()
            .and_then(|fields| fields.reduced_costs.clone()),
        var_basis: certificate
            .as_ref()
            .and_then(|fields| fields.var_basis.clone()),
        row_basis: certificate
            .as_ref()
            .and_then(|fields| fields.row_basis.clone()),
        iterations: None,
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_cbc_cli_model(
    kind: ExternalLinearCliKind,
    sense: Sense,
    model_text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    integer_vars: Option<&[bool]>,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "cbc:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Cbc, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "cbc executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = native_cbc_temp_path("model", extension);
    let solution_path = native_cbc_temp_path("solution", "sol");
    let basis_path = native_cbc_temp_path("basis", "bas");
    let start_path = (kind == ExternalLinearCliKind::Mip && opts.mip_start.is_some())
        .then(|| native_cbc_temp_path("start", "sol"));
    let active_branch_priorities = if kind == ExternalLinearCliKind::Mip {
        match active_branch_priorities(
            opts.branch_priorities.as_deref(),
            integer_vars,
            variable_count,
        ) {
            Ok(priorities) => priorities,
            Err(message) => {
                return external_cli_failure(
                    ExternalLinearCliStatus::NumericalError,
                    bridge_solver,
                    message,
                    elapsed_ms(t0),
                );
            }
        }
    } else {
        Vec::new()
    };
    let priority_path =
        (!active_branch_priorities.is_empty()).then(|| native_cbc_temp_path("priority", "csv"));
    let mut cleanup_paths = vec![
        model_path.clone(),
        solution_path.clone(),
        basis_path.clone(),
    ];
    if let Some(start_path) = &start_path {
        cleanup_paths.push(start_path.clone());
    }
    if let Some(priority_path) = &priority_path {
        cleanup_paths.push(priority_path.clone());
    }

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_cbc_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write CBC model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }
    if let Some(priority_path) = &priority_path {
        if let Err(err) = fs::write(
            priority_path,
            native_cbc_branch_priorities_text(&active_branch_priorities),
        ) {
            cleanup_native_cbc_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                format!(
                    "failed to write CBC branch-priority file '{}': {err}",
                    priority_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    }

    let mip_start_objective = if kind == ExternalLinearCliKind::Mip {
        match opts.mip_start.as_deref() {
            Some(mip_start) => {
                if mip_start.len() != variable_count {
                    cleanup_native_cbc_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        format!(
                            "mip_start length {} does not match variable count {}",
                            mip_start.len(),
                            variable_count
                        ),
                        elapsed_ms(t0),
                    );
                }
                if mip_start.iter().any(|value| !value.is_finite()) {
                    cleanup_native_cbc_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        "mip_start values must be finite".to_string(),
                        elapsed_ms(t0),
                    );
                }
                let objective = dot_f64(objective_coefficients, mip_start);
                if !objective.is_finite() {
                    cleanup_native_cbc_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        "mip_start objective must be finite".to_string(),
                        elapsed_ms(t0),
                    );
                }
                let Some(start_path) = &start_path else {
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        "internal CBC MIP-start path was unavailable".to_string(),
                        elapsed_ms(t0),
                    );
                };
                if let Err(err) = fs::write(start_path, native_cbc_mip_start_text(mip_start)) {
                    cleanup_native_cbc_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        format!(
                            "failed to write CBC MIP-start file '{}': {err}",
                            start_path.display()
                        ),
                        elapsed_ms(t0),
                    );
                }
                Some(objective)
            }
            None => None,
        }
    } else {
        None
    };

    let mut command = Command::new(&command_path);
    command
        .arg(&model_path)
        .arg("-seconds")
        .arg(cbc_time_limit_arg(opts.time_limit_secs));
    if opts.model_format == ExternalLinearCliModelFormat::Mps {
        command.arg(match sense {
            Sense::Max => "-max",
            Sense::Min => "-min",
        });
    }
    if let Some(seed) = normalized_cbc_random_seed(opts.random_seed) {
        command
            .arg("-randomS")
            .arg(seed.to_string())
            .arg("-randomC")
            .arg(seed.to_string());
    }
    if let Some(threads) = opts.threads.filter(|threads| *threads > 0) {
        command.arg("-threads").arg(threads.to_string());
    }
    if let Some(tolerance) = normalized_tolerance(opts.primal_feasibility_tolerance) {
        command.arg("-primalT").arg(format!("{tolerance:.17}"));
    }
    if let Some(tolerance) = normalized_tolerance(opts.dual_feasibility_tolerance) {
        command.arg("-dualT").arg(format!("{tolerance:.17}"));
    }
    match opts.presolve {
        Some(ExternalLinearCliPresolve::Off) => {
            command.arg("-presolve").arg("off");
            if kind == ExternalLinearCliKind::Mip {
                command.arg("-preprocess").arg("off");
            }
        }
        Some(ExternalLinearCliPresolve::On) => {
            command.arg("-presolve").arg("on");
        }
        Some(ExternalLinearCliPresolve::Auto) | None => {}
    }
    if kind == ExternalLinearCliKind::Lp {
        command.arg("-printingOptions").arg("all");
    }
    if kind == ExternalLinearCliKind::Mip {
        if let Some(cuts) = opts.cuts {
            match cuts {
                ExternalLinearCliMipSwitch::On | ExternalLinearCliMipSwitch::Off => {
                    command.arg("-cuts").arg(cuts.as_str());
                }
                ExternalLinearCliMipSwitch::Auto => {}
            }
        }
        if let Some(heuristics) = opts.heuristics {
            match heuristics {
                ExternalLinearCliMipSwitch::On | ExternalLinearCliMipSwitch::Off => {
                    command.arg("-heuristicsOnOff").arg(heuristics.as_str());
                }
                ExternalLinearCliMipSwitch::Auto => {}
            }
        }
        if let Some(priority_path) = &priority_path {
            command.arg("-priorityIn").arg(priority_path);
        }
        if let Some(start_path) = &start_path {
            command.arg("-mipstart").arg(start_path);
        }
        if let Some(node_limit) = opts
            .max_nodes
            .or_else(|| opts.node_limit.map(|limit| limit as u64))
        {
            command.arg("-maxNodes").arg(node_limit.to_string());
        }
        if let Some(solution_limit) = opts.solution_limit.map(|limit| limit.max(1)) {
            command.arg("-maxSolutions").arg(solution_limit.to_string());
        }
        match opts.node_selection {
            Some(ExternalLinearCliNodeSelection::Dfs) => {
                command.arg("-nodeStrategy").arg("depth");
            }
            Some(ExternalLinearCliNodeSelection::BestBound) => {
                command.arg("-nodeStrategy").arg("fewest");
            }
            None => {}
        }
        if let Some(tolerance) = normalized_tolerance(opts.integer_feasibility_tolerance) {
            command.arg("-integerT").arg(format!("{tolerance:.17}"));
        }
        if let Some(relative_gap) = normalized_relative_gap(opts.relative_gap) {
            command.arg("-ratioGap").arg(format!("{relative_gap:.17}"));
        }
        if let Some(absolute_gap) = normalized_absolute_gap(opts.absolute_gap) {
            command
                .arg("-allowableGap")
                .arg(format!("{absolute_gap:.17}"));
        }
    }
    command.arg("-solve").arg("-solution").arg(&solution_path);
    if kind == ExternalLinearCliKind::Lp {
        command.arg("-basisOut").arg(&basis_path);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_cbc_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start CBC executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_cbc_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_cbc_solver_version(&command_path));

    if !solution_path.exists() {
        let status = classify_native_linear_status("", &stdout, &stderr);
        cleanup_native_cbc_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message("", &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let parsed = match parse_native_cbc_solution_file(
        &solution_path,
        variable_count,
        le_count,
        eq_count,
        (kind == ExternalLinearCliKind::Lp).then_some(basis_path.as_path()),
    ) {
        Ok(parsed) => parsed,
        Err(message) => {
            cleanup_native_cbc_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    cleanup_native_cbc_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let objective = dot_f64(objective_coefficients, &parsed.x);
    let quality = parse_cbc_mip_quality(kind, status, objective, &stdout, &stderr);
    let (mip_start_accepted, mip_start_objective) = parse_cbc_mip_start_feedback(
        kind,
        opts.mip_start.as_deref(),
        mip_start_objective,
        &stdout,
        &stderr,
    );
    let (branch_priorities_accepted, branch_priority_count) =
        parse_cbc_branch_priority_feedback(kind, active_branch_priorities.len(), &stdout, &stderr);
    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(objective),
        objective_values: None,
        lp_algorithm: None,
        best_bound: quality.best_bound,
        solution_limit: (kind == ExternalLinearCliKind::Mip)
            .then(|| opts.solution_limit.map(|limit| limit.max(1)))
            .flatten(),
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: quality.mip_gap,
        absolute_gap: quality.absolute_gap,
        objective_limit: None,
        primal_feasibility_tolerance: normalized_tolerance(opts.primal_feasibility_tolerance),
        dual_feasibility_tolerance: normalized_tolerance(opts.dual_feasibility_tolerance),
        integer_feasibility_tolerance: (kind == ExternalLinearCliKind::Mip)
            .then(|| normalized_tolerance(opts.integer_feasibility_tolerance))
            .flatten(),
        nodes_explored: quality.nodes_explored,
        threads: opts.threads.filter(|threads| *threads > 0),
        random_seed: normalized_cbc_random_seed(opts.random_seed),
        presolve: opts.presolve.map(|presolve| presolve.as_str().to_string()),
        cuts: (kind == ExternalLinearCliKind::Mip)
            .then(|| opts.cuts.map(|cuts| cuts.as_str().to_string()))
            .flatten(),
        heuristics: (kind == ExternalLinearCliKind::Mip)
            .then(|| {
                opts.heuristics
                    .map(|heuristics| heuristics.as_str().to_string())
            })
            .flatten(),
        branch_rule: None,
        branch_priorities_accepted,
        branch_priority_count,
        node_selection: (kind == ExternalLinearCliKind::Mip)
            .then(|| {
                opts.node_selection
                    .map(|node_selection| node_selection.as_str().to_string())
            })
            .flatten(),
        mip_start_accepted,
        mip_start_objective,
        dual_ub: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.dual_ub)
            .flatten(),
        dual_eq: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.dual_eq)
            .flatten(),
        reduced_costs: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.reduced_costs)
            .flatten(),
        var_basis: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.var_basis)
            .flatten(),
        row_basis: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.row_basis)
            .flatten(),
        iterations: (kind == ExternalLinearCliKind::Lp)
            .then(|| parse_cbc_lp_iterations(&stdout, &stderr))
            .flatten(),
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_clp_cli_model(
    sense: Sense,
    model_text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "clp:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Clp, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "clp executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = native_clp_temp_path("model", extension);
    let solution_path = native_clp_temp_path("solution", "sol");
    let basis_path = native_clp_temp_path("basis", "bas");
    let cleanup_paths = vec![
        model_path.clone(),
        solution_path.clone(),
        basis_path.clone(),
    ];

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_clp_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write CLP model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let mut command = Command::new(&command_path);
    command
        .arg(&model_path)
        .arg("-seconds")
        .arg(cbc_time_limit_arg(opts.time_limit_secs));
    if opts.model_format == ExternalLinearCliModelFormat::Mps {
        command.arg(match sense {
            Sense::Max => "-max",
            Sense::Min => "-min",
        });
    }
    command
        .arg("-printingOptions")
        .arg("all")
        .arg("-solve")
        .arg("-solution")
        .arg(&solution_path)
        .arg("-basisOut")
        .arg(&basis_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_clp_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start CLP executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_clp_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_clp_solver_version(&command_path));

    if !solution_path.exists() {
        let status = classify_native_linear_status("", &stdout, &stderr);
        cleanup_native_clp_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message("", &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let parsed = match parse_native_cbc_solution_file(
        &solution_path,
        variable_count,
        le_count,
        eq_count,
        Some(basis_path.as_path()),
    ) {
        Ok(parsed) => parsed,
        Err(message) => {
            cleanup_native_clp_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    cleanup_native_clp_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(dot_f64(objective_coefficients, &parsed.x)),
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
        dual_ub: parsed.dual_ub,
        dual_eq: parsed.dual_eq,
        reduced_costs: parsed.reduced_costs,
        var_basis: parsed.var_basis,
        row_basis: parsed.row_basis,
        iterations: parse_cbc_lp_iterations(&stdout, &stderr),
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_soplex_cli_model(
    model_text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "soplex:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Soplex, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "soplex executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = native_soplex_temp_path("model", extension);
    let solution_path = native_soplex_temp_path("solution", "sol");
    let dual_path = native_soplex_temp_path("dual", "sol");
    let basis_path = native_soplex_temp_path("basis", "bas");
    let cleanup_paths = vec![
        model_path.clone(),
        solution_path.clone(),
        dual_path.clone(),
        basis_path.clone(),
    ];

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_soplex_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write SoPlex model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let output = match Command::new(&command_path)
        .arg("-v3")
        .arg(format!(
            "-t{:.17}",
            normalized_time_limit(opts.time_limit_secs)
        ))
        .arg(format!("-x={}", solution_path.display()))
        .arg(format!("-y={}", dual_path.display()))
        .arg(format!("--writebas={}", basis_path.display()))
        .arg(&model_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_soplex_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start SoPlex executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_soplex_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_soplex_solver_version(&command_path));

    let parsed = match parse_native_soplex_solution_file(
        &solution_path,
        Some(dual_path.as_path()),
        Some(basis_path.as_path()),
        variable_count,
        le_count,
        eq_count,
        &stdout,
        &stderr,
    ) {
        Ok(parsed) => parsed,
        Err(message) => {
            let status = classify_native_linear_status("", &stdout, &stderr);
            cleanup_native_soplex_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                if matches!(
                    status,
                    ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
                ) {
                    status
                } else {
                    ExternalLinearCliStatus::Unavailable
                },
                bridge_solver,
                message,
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    cleanup_native_soplex_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(dot_f64(objective_coefficients, &parsed.x)),
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
        dual_ub: parsed.dual_ub,
        dual_eq: parsed.dual_eq,
        reduced_costs: parsed.reduced_costs,
        var_basis: parsed.var_basis,
        row_basis: parsed.row_basis,
        iterations: parse_soplex_lp_iterations(&stdout, &stderr),
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_glpk_cli_model(
    kind: ExternalLinearCliKind,
    sense: Sense,
    model_text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "glpk:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Glpk, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "glpsol executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = native_glpk_temp_path("model", extension);
    let solution_path = native_glpk_temp_path("solution", "sol");
    let report_path = native_glpk_temp_path("report", "txt");
    let log_path = native_glpk_temp_path("log", "log");
    let cleanup_paths = vec![
        model_path.clone(),
        solution_path.clone(),
        report_path.clone(),
        log_path.clone(),
    ];

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_glpk_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write GLPK model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let mut command = Command::new(&command_path);
    command
        .arg(match opts.model_format {
            ExternalLinearCliModelFormat::CplexLp => "--lp",
            ExternalLinearCliModelFormat::Mps => "--freemps",
        })
        .arg(&model_path)
        .arg(match sense {
            Sense::Max => "--max",
            Sense::Min => "--min",
        })
        .arg("--tmlim")
        .arg(glpk_time_limit_arg(opts.time_limit_secs))
        .arg("--log")
        .arg(&log_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    match kind {
        ExternalLinearCliKind::Lp => {
            command
                .arg("--output")
                .arg(&report_path)
                .arg("--write")
                .arg(&solution_path);
            match opts.presolve {
                Some(ExternalLinearCliPresolve::Off) => {
                    command.arg("--nopresol");
                }
                Some(ExternalLinearCliPresolve::On) => {
                    command.arg("--presol");
                }
                Some(ExternalLinearCliPresolve::Auto) | None => {}
            }
            match opts.lp_algorithm {
                Some(ExternalLinearCliLpAlgorithm::Simplex) => {
                    command.arg("--simplex");
                }
                Some(ExternalLinearCliLpAlgorithm::Ipm) => {
                    command.arg("--interior");
                }
                None => {}
            }
        }
        ExternalLinearCliKind::Mip => {
            command.arg("-o").arg(&solution_path);
            match opts.presolve {
                Some(ExternalLinearCliPresolve::Off) => {
                    command.arg("--nointopt");
                }
                Some(ExternalLinearCliPresolve::On) => {
                    command.arg("--intopt");
                }
                Some(ExternalLinearCliPresolve::Auto) | None => {}
            }
            match opts.branch_rule {
                Some(ExternalLinearCliBranchRule::FirstFractional) => {
                    command.arg("--first");
                }
                Some(ExternalLinearCliBranchRule::MostFractional) => {
                    command.arg("--mostf");
                }
                None => {}
            }
            match opts.node_selection {
                Some(ExternalLinearCliNodeSelection::Dfs) => {
                    command.arg("--dfs");
                }
                Some(ExternalLinearCliNodeSelection::BestBound) => {
                    command.arg("--bestb");
                }
                None => {}
            }
            if let Some(relative_gap) = normalized_relative_gap(opts.relative_gap) {
                command.arg("--mipgap").arg(format!("{relative_gap:.17}"));
            }
            if opts.cuts == Some(ExternalLinearCliMipSwitch::On) {
                command.arg("--cuts");
            }
        }
    }
    if let Some(seed) = normalized_glpk_random_seed(opts.random_seed) {
        command.arg("--seed").arg(seed.to_string());
    }

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_glpk_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start GLPK executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_glpk_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_glpk_solver_version(&command_path));

    if !solution_path.exists() {
        let status = classify_native_linear_status("", &stdout, &stderr);
        cleanup_native_glpk_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message("", &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let parsed =
        match parse_native_glpk_solution_file(&solution_path, variable_count, le_count, eq_count) {
            Ok(parsed) => parsed,
            Err(message) => {
                cleanup_native_glpk_temp_files(&cleanup_paths);
                let mut failure = external_cli_failure(
                    ExternalLinearCliStatus::NumericalError,
                    bridge_solver,
                    message,
                    elapsed,
                );
                failure.solver_version = solver_version;
                return failure;
            }
        };
    cleanup_native_glpk_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let objective = dot_f64(objective_coefficients, &parsed.x);
    let quality = parse_glpk_mip_quality(kind, status, objective, &stdout, &stderr);
    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(objective),
        objective_values: None,
        lp_algorithm: (kind == ExternalLinearCliKind::Lp)
            .then(|| {
                opts.lp_algorithm
                    .map(|algorithm| algorithm.as_str().to_string())
            })
            .flatten(),
        best_bound: quality.best_bound,
        solution_limit: None,
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: quality.mip_gap,
        absolute_gap: quality.absolute_gap,
        objective_limit: None,
        primal_feasibility_tolerance: None,
        dual_feasibility_tolerance: None,
        integer_feasibility_tolerance: None,
        nodes_explored: quality.nodes_explored,
        threads: None,
        random_seed: normalized_glpk_random_seed(opts.random_seed),
        presolve: opts.presolve.map(|presolve| presolve.as_str().to_string()),
        cuts: (kind == ExternalLinearCliKind::Mip
            && opts.cuts == Some(ExternalLinearCliMipSwitch::On))
        .then_some("on".to_string()),
        heuristics: None,
        branch_rule: (kind == ExternalLinearCliKind::Mip)
            .then(|| {
                opts.branch_rule
                    .map(|branch_rule| branch_rule.as_str().to_string())
            })
            .flatten(),
        branch_priorities_accepted: None,
        branch_priority_count: None,
        node_selection: (kind == ExternalLinearCliKind::Mip)
            .then(|| {
                opts.node_selection
                    .map(|node_selection| node_selection.as_str().to_string())
            })
            .flatten(),
        mip_start_accepted: None,
        mip_start_objective: None,
        dual_ub: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.dual_ub)
            .flatten(),
        dual_eq: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.dual_eq)
            .flatten(),
        reduced_costs: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.reduced_costs)
            .flatten(),
        var_basis: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.var_basis)
            .flatten(),
        row_basis: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.row_basis)
            .flatten(),
        iterations: (kind == ExternalLinearCliKind::Lp)
            .then(|| parse_glpk_lp_iterations(&stdout, &stderr))
            .flatten(),
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_gurobi_cli_model(
    kind: ExternalLinearCliKind,
    model_text: &str,
    variable_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "gurobi:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Gurobi, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "gurobi executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = native_gurobi_temp_path("model", extension);
    let solution_path = native_gurobi_temp_path("solution", "sol");
    let cleanup_paths = vec![model_path.clone(), solution_path.clone()];

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_gurobi_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write Gurobi model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let mut command = Command::new(&command_path);
    command
        .arg(format!("ResultFile={}", solution_path.display()))
        .arg(format!(
            "TimeLimit={:.17}",
            normalized_time_limit(opts.time_limit_secs)
        ))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if kind == ExternalLinearCliKind::Mip {
        if let Some(max_nodes) = opts
            .max_nodes
            .or_else(|| opts.node_limit.map(|limit| limit as u64))
        {
            command.arg(format!("NodeLimit={max_nodes}"));
        }
        if let Some(solution_limit) = opts.solution_limit {
            command.arg(format!("SolutionLimit={}", solution_limit.max(1)));
        }
        if let Some(relative_gap) = normalized_relative_gap(opts.relative_gap) {
            command.arg(format!("MIPGap={relative_gap:.17}"));
        }
        if let Some(absolute_gap) = normalized_absolute_gap(opts.absolute_gap) {
            command.arg(format!("MIPGapAbs={absolute_gap:.17}"));
        }
        if let Some(objective_limit) = normalized_objective_limit(opts.objective_limit) {
            command.arg(format!("BestObjStop={objective_limit:.17}"));
        }
        if let Some(tolerance) = normalized_tolerance(opts.integer_feasibility_tolerance) {
            command.arg(format!("IntFeasTol={tolerance:.17}"));
        }
    }
    if let Some(tolerance) = normalized_tolerance(opts.primal_feasibility_tolerance) {
        command.arg(format!("FeasibilityTol={tolerance:.17}"));
    }
    if let Some(tolerance) = normalized_tolerance(opts.dual_feasibility_tolerance) {
        command.arg(format!("OptimalityTol={tolerance:.17}"));
    }
    if let Some(threads) = opts.threads.filter(|threads| *threads > 0) {
        command.arg(format!("Threads={threads}"));
    }
    if let Some(seed) = opts.random_seed.filter(|seed| *seed <= i32::MAX as u64) {
        command.arg(format!("Seed={seed}"));
    }
    if let Some(presolve) = opts.presolve {
        let value = match presolve {
            ExternalLinearCliPresolve::Auto => -1,
            ExternalLinearCliPresolve::On => 1,
            ExternalLinearCliPresolve::Off => 0,
        };
        command.arg(format!("Presolve={value}"));
    }
    command.arg(&model_path);

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_gurobi_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start Gurobi executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_gurobi_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_gurobi_solver_version(&command_path));

    if !solution_path.exists() {
        let status = classify_native_linear_status("", &stdout, &stderr);
        cleanup_native_gurobi_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message("", &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let parsed = match parse_native_named_solution_file(&solution_path, variable_count, "optimal") {
        Ok(parsed) => parsed,
        Err(message) => {
            cleanup_native_gurobi_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    cleanup_native_gurobi_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(dot_f64(objective_coefficients, &parsed.x)),
        objective_values: None,
        lp_algorithm: None,
        best_bound: None,
        solution_limit: (kind == ExternalLinearCliKind::Mip)
            .then_some(opts.solution_limit)
            .flatten(),
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: None,
        absolute_gap: None,
        objective_limit: (kind == ExternalLinearCliKind::Mip)
            .then(|| normalized_objective_limit(opts.objective_limit))
            .flatten(),
        primal_feasibility_tolerance: normalized_tolerance(opts.primal_feasibility_tolerance),
        dual_feasibility_tolerance: normalized_tolerance(opts.dual_feasibility_tolerance),
        integer_feasibility_tolerance: (kind == ExternalLinearCliKind::Mip)
            .then(|| normalized_tolerance(opts.integer_feasibility_tolerance))
            .flatten(),
        nodes_explored: None,
        threads: opts.threads.filter(|threads| *threads > 0),
        random_seed: opts.random_seed.filter(|seed| *seed <= i32::MAX as u64),
        presolve: opts.presolve.map(|presolve| presolve.as_str().to_string()),
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
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_cplex_cli_model(
    kind: ExternalLinearCliKind,
    model_text: &str,
    variable_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "cplex:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Cplex, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "cplex executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = native_cplex_temp_path("model", extension);
    let solution_path = native_cplex_temp_path("solution", "sol");
    let cleanup_paths = vec![model_path.clone(), solution_path.clone()];

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_cplex_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write CPLEX model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let mut command = Command::new(&command_path);
    command
        .arg("-c")
        .arg(format!("read {}", model_path.display()))
        .arg(format!(
            "set timelimit {:.17}",
            normalized_time_limit(opts.time_limit_secs)
        ))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if kind == ExternalLinearCliKind::Mip {
        if let Some(max_nodes) = opts
            .max_nodes
            .or_else(|| opts.node_limit.map(|limit| limit as u64))
        {
            command.arg(format!("set mip limits nodes {max_nodes}"));
        }
        if let Some(solution_limit) = opts.solution_limit {
            command.arg(format!(
                "set mip limits solutions {}",
                solution_limit.max(1)
            ));
        }
        if let Some(relative_gap) = normalized_relative_gap(opts.relative_gap) {
            command.arg(format!("set mip tolerances mipgap {relative_gap:.17}"));
        }
        if let Some(absolute_gap) = normalized_absolute_gap(opts.absolute_gap) {
            command.arg(format!("set mip tolerances absmipgap {absolute_gap:.17}"));
        }
        if let Some(tolerance) = normalized_tolerance(opts.integer_feasibility_tolerance) {
            command.arg(format!("set mip tolerances integrality {tolerance:.17}"));
        }
    }
    if let Some(tolerance) = normalized_tolerance(opts.primal_feasibility_tolerance) {
        command.arg(format!(
            "set simplex tolerances feasibility {tolerance:.17}"
        ));
    }
    if let Some(tolerance) = normalized_tolerance(opts.dual_feasibility_tolerance) {
        command.arg(format!("set simplex tolerances optimality {tolerance:.17}"));
    }
    if let Some(threads) = opts.threads.filter(|threads| *threads > 0) {
        command.arg(format!("set threads {threads}"));
    }
    if let Some(seed) = opts.random_seed.filter(|seed| *seed <= i32::MAX as u64) {
        command.arg(format!("set randomseed {seed}"));
    }
    if let Some(presolve) = opts.presolve {
        let value = match presolve {
            ExternalLinearCliPresolve::Auto => -1,
            ExternalLinearCliPresolve::On => 1,
            ExternalLinearCliPresolve::Off => 0,
        };
        command.arg(format!("set preprocessing presolve {value}"));
    }
    command
        .arg("optimize")
        .arg(format!("write {}", solution_path.display()))
        .arg("quit");

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_cplex_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start CPLEX executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_cplex_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_cplex_solver_version(&command_path));

    if !solution_path.exists() {
        let status = classify_native_linear_status("", &stdout, &stderr);
        cleanup_native_cplex_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message("", &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let parsed = match parse_native_cplex_solution_file(&solution_path, variable_count) {
        Ok(parsed) => parsed,
        Err(message) => {
            cleanup_native_cplex_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    cleanup_native_cplex_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(dot_f64(objective_coefficients, &parsed.x)),
        objective_values: None,
        lp_algorithm: None,
        best_bound: None,
        solution_limit: (kind == ExternalLinearCliKind::Mip)
            .then_some(opts.solution_limit)
            .flatten(),
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: None,
        absolute_gap: None,
        objective_limit: None,
        primal_feasibility_tolerance: normalized_tolerance(opts.primal_feasibility_tolerance),
        dual_feasibility_tolerance: normalized_tolerance(opts.dual_feasibility_tolerance),
        integer_feasibility_tolerance: (kind == ExternalLinearCliKind::Mip)
            .then(|| normalized_tolerance(opts.integer_feasibility_tolerance))
            .flatten(),
        nodes_explored: None,
        threads: opts.threads.filter(|threads| *threads > 0),
        random_seed: opts.random_seed.filter(|seed| *seed <= i32::MAX as u64),
        presolve: opts.presolve.map(|presolve| presolve.as_str().to_string()),
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
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_xpress_cli_model(
    kind: ExternalLinearCliKind,
    model_text: &str,
    variable_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "xpress:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Xpress, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "xpress executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = native_xpress_temp_path("model", extension);
    let solution_path = native_xpress_temp_path("solution", "sol");
    let solution_data_path = native_xpress_solution_data_path(&solution_path);
    let script_path = native_xpress_temp_path("commands", "txt");
    let header_path = native_xpress_header_path(&solution_path);
    let cleanup_paths = vec![
        model_path.clone(),
        solution_path.clone(),
        solution_data_path.clone(),
        script_path.clone(),
        header_path,
    ];

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_xpress_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write Xpress model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }
    let script_text = native_xpress_command_text(kind, &model_path, &solution_path, opts);
    if let Err(err) = fs::write(&script_path, script_text) {
        cleanup_native_xpress_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write Xpress command file '{}': {err}",
                script_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let mut command = Command::new(&command_path);
    command
        .arg(format!("@{}", script_path.display()))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(parent) = model_path.parent() {
        command.current_dir(parent);
    }

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_xpress_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start Xpress executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_xpress_solver_version(&format!("{stdout}\n{stderr}"));

    if !solution_path.exists() && !solution_data_path.exists() {
        let status = classify_native_linear_status("", &stdout, &stderr);
        cleanup_native_xpress_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message("", &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let parsed = match parse_native_xpress_solution_file(&solution_path, variable_count) {
        Ok(parsed) => parsed,
        Err(message) => {
            cleanup_native_xpress_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    cleanup_native_xpress_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(dot_f64(objective_coefficients, &parsed.x)),
        objective_values: None,
        lp_algorithm: None,
        best_bound: None,
        solution_limit: (kind == ExternalLinearCliKind::Mip)
            .then_some(opts.solution_limit)
            .flatten(),
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: None,
        absolute_gap: None,
        objective_limit: None,
        primal_feasibility_tolerance: normalized_tolerance(opts.primal_feasibility_tolerance),
        dual_feasibility_tolerance: normalized_tolerance(opts.dual_feasibility_tolerance),
        integer_feasibility_tolerance: (kind == ExternalLinearCliKind::Mip)
            .then(|| normalized_tolerance(opts.integer_feasibility_tolerance))
            .flatten(),
        nodes_explored: None,
        threads: opts.threads.filter(|threads| *threads > 0),
        random_seed: opts.random_seed.filter(|seed| *seed <= i32::MAX as u64),
        presolve: opts.presolve.map(|presolve| presolve.as_str().to_string()),
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
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_lindo_cli_model(
    kind: ExternalLinearCliKind,
    sense: Sense,
    model_text: &str,
    variable_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "lindo:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Lindo, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "lindo executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let model_path = native_lindo_temp_path("model", "mps");
    let solution_path = native_lindo_solution_path(&model_path);
    let cleanup_paths = vec![model_path.clone(), solution_path.clone()];

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_lindo_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write LINDO model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let mut command = Command::new(&command_path);
    command
        .arg(&model_path)
        .arg("-sol")
        .arg(match sense {
            Sense::Max => "-max",
            Sense::Min => "-min",
        })
        .arg(if kind == ExternalLinearCliKind::Mip {
            "-mip"
        } else {
            "-lp"
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(parent) = model_path.parent() {
        command.current_dir(parent);
    }

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_lindo_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start LINDO executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_lindo_solver_version(&format!("{stdout}\n{stderr}"));

    if !solution_path.exists() {
        let status = classify_native_linear_status("", &stdout, &stderr);
        cleanup_native_lindo_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message("", &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let parsed = match parse_native_lindo_solution_file(&solution_path, variable_count) {
        Ok(parsed) => parsed,
        Err(message) => {
            cleanup_native_lindo_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    cleanup_native_lindo_temp_files(&cleanup_paths);

    let status = classify_native_linear_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(dot_f64(objective_coefficients, &parsed.x)),
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
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

fn solve_native_lindo_gams_model(
    kind: ExternalLinearCliKind,
    model: &PlainLinearCliModel,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
    gams_command: &Path,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "lindo:gams".to_string();
    let model_path = native_lindo_gams_temp_path("model", "gms");
    let solution_path = native_lindo_gams_temp_path("solution", "txt");
    let listing_path = native_lindo_gams_temp_path("listing", "lst");
    let cleanup_paths = vec![
        model_path.clone(),
        solution_path.clone(),
        listing_path.clone(),
    ];

    let mip_start_objective = match native_lindo_gams_mip_start_objective(
        kind,
        model.c.len(),
        objective_coefficients,
        opts,
    ) {
        Ok(objective) => objective,
        Err(message) => {
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed_ms(t0),
            );
        }
    };
    let model_text = gams_lindo_model_text(kind, model, &solution_path, opts);
    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_lindo_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write GAMS LINDO model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let mut command = Command::new(gams_command);
    command
        .arg(&model_path)
        .arg("lo=0")
        .arg(format!("o={}", listing_path.display()))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(parent) = gams_command.parent() {
        command.current_dir(parent);
    }

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_lindo_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start GAMS LINDO executable '{}': {err}",
                    gams_command.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let listing = fs::read_to_string(&listing_path).unwrap_or_default();
    let diagnostics = format!("{stdout}\n{stderr}\n{listing}");
    let solver_version = parse_lindo_solver_version(&diagnostics).or_else(|| {
        parse_gams_solver_version(&diagnostics).map(|version| format!("GAMS {version}"))
    });

    if !solution_path.exists() {
        let status = classify_native_linear_status("", &diagnostics, "");
        cleanup_native_lindo_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_solver_message("", &diagnostics, ""),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let solution_text = match fs::read_to_string(&solution_path) {
        Ok(text) => text,
        Err(err) => {
            cleanup_native_lindo_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                format!(
                    "failed to read GAMS LINDO solution file '{}': {err}",
                    solution_path.display()
                ),
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    let parsed = parse_native_lindo_gams_solution_text(
        &solution_text,
        model.c.len(),
        model.le_rows.len(),
        model.eq_rows.len(),
        &listing,
    );
    cleanup_native_lindo_temp_files(&cleanup_paths);

    let status = ExternalLinearCliStatus::from_str(&parsed.status);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            parsed.status,
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let objective = dot_f64(objective_coefficients, &parsed.x);
    let (var_basis, row_basis) = if kind == ExternalLinearCliKind::Lp {
        lindo_gams_lp_basis_from_solution(model, &parsed)
    } else {
        (None, None)
    };
    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x.clone(),
        objective: Some(objective),
        objective_values: None,
        lp_algorithm: None,
        best_bound: (kind == ExternalLinearCliKind::Mip
            && status == ExternalLinearCliStatus::Optimal)
            .then_some(objective),
        solution_limit: None,
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: (kind == ExternalLinearCliKind::Mip && status == ExternalLinearCliStatus::Optimal)
            .then_some(0.0),
        absolute_gap: (kind == ExternalLinearCliKind::Mip
            && status == ExternalLinearCliStatus::Optimal)
            .then_some(0.0),
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
        mip_start_objective,
        dual_ub: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.dual_ub)
            .flatten(),
        dual_eq: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.dual_eq)
            .flatten(),
        reduced_costs: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.reduced_costs)
            .flatten(),
        var_basis,
        row_basis,
        iterations: parse_gams_iteration_count(&listing),
        elapsed_ms: elapsed,
        message: "GAMS LINDO solve".to_string(),
    }
}

fn lindo_gams_lp_basis_from_solution(
    model: &PlainLinearCliModel,
    parsed: &ParsedNativeNamedSolution,
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    let (Some(dual_ub), Some(_dual_eq), Some(reduced_costs)) = (
        parsed.dual_ub.as_deref(),
        parsed.dual_eq.as_deref(),
        parsed.reduced_costs.as_deref(),
    ) else {
        return (None, None);
    };
    infer_lp_basis_from_complementarity(
        &parsed.x,
        Some(&model.lbs),
        Some(&model.ubs),
        &model.le_rows,
        &model.le_rhs,
        dual_ub,
        &model.eq_rows,
        &model.eq_rhs,
        reduced_costs,
    )
}

fn gams_lindo_model_text(
    kind: ExternalLinearCliKind,
    model: &PlainLinearCliModel,
    solution_path: &Path,
    opts: &ExternalLinearCliOptions,
) -> String {
    let mut continuous = Vec::new();
    let mut binary = Vec::new();
    let mut general = Vec::new();
    for i in 0..model.c.len() {
        let name = gams_var_name(i);
        if model.integer_vars.get(i).copied().unwrap_or(false) {
            if is_binary_bound(&model.integer_vars, &model.lbs, &model.ubs, i) {
                binary.push(name);
            } else {
                general.push(name);
            }
        } else {
            continuous.push(name);
        }
    }

    let mut out = String::new();
    out.push_str("Variable z;\n");
    push_gams_declaration(&mut out, "Variables", &continuous);
    push_gams_declaration(&mut out, "Binary Variables", &binary);
    push_gams_declaration(&mut out, "Integer Variables", &general);
    out.push_str("Equations obj");
    for i in 0..model.le_rows.len() {
        out.push_str(&format!(", le{i}"));
    }
    for i in 0..model.eq_rows.len() {
        out.push_str(&format!(", eq{i}"));
    }
    out.push_str(";\n");
    out.push_str(&format!("obj.. z =e= {};\n", gams_linear_expr(&model.c)));
    for (i, (row, rhs)) in model.le_rows.iter().zip(&model.le_rhs).enumerate() {
        out.push_str(&format!(
            "le{i}.. {} =l= {};\n",
            gams_linear_expr(row),
            fmt_lp_number(*rhs)
        ));
    }
    for (i, (row, rhs)) in model.eq_rows.iter().zip(&model.eq_rhs).enumerate() {
        out.push_str(&format!(
            "eq{i}.. {} =e= {};\n",
            gams_linear_expr(row),
            fmt_lp_number(*rhs)
        ));
    }
    for i in 0..model.c.len() {
        let name = gams_var_name(i);
        if let Some(lower) = model.lbs.get(i).copied().flatten() {
            if lower.is_finite() {
                out.push_str(&format!("{name}.lo = {};\n", fmt_lp_number(lower)));
            }
        }
        if let Some(upper) = model.ubs.get(i).copied().flatten() {
            if upper.is_finite() {
                out.push_str(&format!("{name}.up = {};\n", fmt_lp_number(upper)));
            }
        }
    }
    if kind == ExternalLinearCliKind::Mip {
        if let Some(mip_start) = opts.mip_start.as_deref() {
            for (i, value) in mip_start.iter().copied().enumerate() {
                out.push_str(&format!("x{i}.l = {};\n", fmt_lp_number(value)));
            }
        }
    }
    if let Some(seconds) = opts
        .time_limit_secs
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
    {
        out.push_str(&format!("option reslim = {};\n", fmt_lp_number(seconds)));
    }
    if kind == ExternalLinearCliKind::Mip {
        if let Some(relative_gap) = normalized_relative_gap(opts.relative_gap) {
            out.push_str(&format!(
                "option optcr = {};\n",
                fmt_lp_number(relative_gap)
            ));
        }
    }
    out.push_str("option lp = lindo;\n");
    out.push_str("option mip = lindo;\n");
    out.push_str("Model m /all/;\n");
    let model_kind = if kind == ExternalLinearCliKind::Mip {
        "mip"
    } else {
        "lp"
    };
    let direction = match model.sense {
        Sense::Max => "maximizing",
        Sense::Min => "minimizing",
    };
    out.push_str(&format!("Solve m using {model_kind} {direction} z;\n"));
    out.push_str(&format!(
        "File result / {} /;\n",
        gams_single_quoted_path(solution_path)
    ));
    out.push_str("put result;\n");
    out.push_str("put 'modelstat ' m.modelstat:0:0 /;\n");
    out.push_str("put 'solvestat ' m.solvestat:0:0 /;\n");
    out.push_str("put 'objective ' z.l:0:17 /;\n");
    for i in 0..model.c.len() {
        out.push_str(&format!("put 'x{i} ' x{i}.l:0:17 /;\n"));
    }
    if kind == ExternalLinearCliKind::Lp {
        for i in 0..model.le_rows.len() {
            out.push_str(&format!("put 'le{i}.m ' le{i}.m:0:17 /;\n"));
        }
        for i in 0..model.eq_rows.len() {
            out.push_str(&format!("put 'eq{i}.m ' eq{i}.m:0:17 /;\n"));
        }
        for i in 0..model.c.len() {
            out.push_str(&format!("put 'x{i}.m ' x{i}.m:0:17 /;\n"));
        }
    }
    out.push_str("putclose result;\n");
    out
}

fn native_lindo_gams_mip_start_objective(
    kind: ExternalLinearCliKind,
    variable_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> Result<Option<f64>, String> {
    if kind != ExternalLinearCliKind::Mip {
        return Ok(None);
    }
    let Some(mip_start) = opts.mip_start.as_deref() else {
        return Ok(None);
    };
    if mip_start.len() != variable_count {
        return Err(format!(
            "mip_start length {} does not match variable count {}",
            mip_start.len(),
            variable_count
        ));
    }
    if mip_start.iter().any(|value| !value.is_finite()) {
        return Err("mip_start values must be finite".to_string());
    }
    let objective = dot_f64(objective_coefficients, mip_start);
    if !objective.is_finite() {
        return Err("mip_start objective must be finite".to_string());
    }
    Ok(Some(objective))
}

fn push_gams_declaration(out: &mut String, keyword: &str, names: &[String]) {
    if !names.is_empty() {
        out.push_str(keyword);
        out.push(' ');
        out.push_str(&names.join(", "));
        out.push_str(";\n");
    }
}

fn gams_var_name(index: usize) -> String {
    format!("x{index}")
}

fn gams_linear_expr(coefs: &[f64]) -> String {
    let mut out = String::new();
    for (idx, coef) in coefs.iter().copied().enumerate() {
        if !coef.is_finite() || coef.abs() <= 1.0e-12 {
            continue;
        }
        let magnitude = coef.abs();
        let term = if (magnitude - 1.0).abs() <= 1.0e-12 {
            gams_var_name(idx)
        } else {
            format!("{}*{}", fmt_lp_number(magnitude), gams_var_name(idx))
        };
        if out.is_empty() {
            if coef < 0.0 {
                out.push_str("- ");
            }
            out.push_str(&term);
        } else if coef < 0.0 {
            out.push_str(" - ");
            out.push_str(&term);
        } else {
            out.push_str(" + ");
            out.push_str(&term);
        }
    }
    if out.is_empty() {
        "0".to_string()
    } else {
        out
    }
}

fn gams_single_quoted_path(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "''"))
}

fn parse_native_lindo_gams_solution_text(
    text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    listing: &str,
) -> ParsedNativeNamedSolution {
    let mut x = vec![0.0; variable_count];
    let mut reduced_costs = vec![None::<f64>; variable_count];
    let mut dual_ub = vec![None::<f64>; le_count];
    let mut dual_eq = vec![None::<f64>; eq_count];
    let mut modelstat = None;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next().and_then(parse_f64_token) else {
            continue;
        };
        if key.eq_ignore_ascii_case("modelstat") {
            modelstat = Some(value.round() as i32);
            continue;
        }

        let key = key.to_ascii_lowercase();
        if let Some(name) = key.strip_suffix(".m") {
            if let Some(index_text) = name.strip_prefix('x') {
                if let Ok(index) = index_text.parse::<usize>() {
                    if index < reduced_costs.len() {
                        reduced_costs[index] = Some(clean_certificate_value(value));
                    }
                }
            } else if let Some(index_text) = name.strip_prefix("le") {
                if let Ok(index) = index_text.parse::<usize>() {
                    if index < dual_ub.len() {
                        dual_ub[index] = Some(clean_certificate_value(value));
                    }
                }
            } else if let Some(index_text) = name.strip_prefix("eq") {
                if let Ok(index) = index_text.parse::<usize>() {
                    if index < dual_eq.len() {
                        dual_eq[index] = Some(clean_certificate_value(value));
                    }
                }
            }
        } else if let Some(index_text) = key.strip_prefix('x') {
            if let Ok(index) = index_text.parse::<usize>() {
                if index < x.len() {
                    x[index] = value;
                }
            }
        }
    }
    ParsedNativeNamedSolution {
        status: gams_model_status(modelstat, listing).to_string(),
        x,
        reduced_costs: all_some_f64(&reduced_costs),
        dual_ub: all_some_f64(&dual_ub),
        dual_eq: all_some_f64(&dual_eq),
    }
}

fn gams_model_status(modelstat: Option<i32>, listing: &str) -> &'static str {
    match modelstat {
        Some(1 | 2) => "optimal",
        Some(8) => "feasible",
        Some(3 | 18) => "unbounded",
        Some(4 | 5 | 6 | 10 | 19) => "infeasible",
        _ => match classify_native_linear_status("", listing, "") {
            ExternalLinearCliStatus::Optimal => "optimal",
            ExternalLinearCliStatus::Feasible => "feasible",
            ExternalLinearCliStatus::Infeasible => "infeasible",
            ExternalLinearCliStatus::Unbounded => "unbounded",
            ExternalLinearCliStatus::NumericalError => "numerical-error",
            ExternalLinearCliStatus::Unavailable => "unavailable",
            ExternalLinearCliStatus::Unknown => "unknown",
        },
    }
}

fn parse_gams_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some((_, rest)) = line.split_once("GAMS ") {
            let version = rest
                .split_whitespace()
                .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                .unwrap_or("")
                .trim();
            if !version.is_empty() {
                return Some(version.to_string());
            }
        }
    }
    None
}

fn parse_gams_iteration_count(text: &str) -> Option<u64> {
    for line in text.lines() {
        let stripped = line.trim();
        if stripped.starts_with("ITERATION COUNT") {
            return first_float_after_colon(stripped)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .map(|value| value.round() as u64);
        }
    }
    None
}

fn solve_native_highs_cli_model(
    kind: ExternalLinearCliKind,
    model_text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    objective_coefficients: &[f64],
    opts: &ExternalLinearCliOptions,
) -> ExternalLinearCliSolution {
    let t0 = Instant::now();
    let bridge_solver = "highs:cli".to_string();
    let Some(command_path) =
        external_linear_cli_command_with_options(ExternalLinearCliSolver::Highs, opts)
    else {
        return external_cli_failure(
            ExternalLinearCliStatus::Unavailable,
            bridge_solver,
            "highs executable not found".to_string(),
            elapsed_ms(t0),
        );
    };

    let extension = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "lp",
        ExternalLinearCliModelFormat::Mps => "mps",
    };
    let model_path = native_highs_temp_path("model", extension);
    let solution_path = native_highs_temp_path("solution", "sol");
    let options_path = native_highs_temp_path("options", "options");
    let log_path = native_highs_temp_path("log", "log");
    let start_path = (kind == ExternalLinearCliKind::Mip && opts.mip_start.is_some())
        .then(|| native_highs_temp_path("start", "sol"));
    let mut cleanup_paths = vec![model_path.clone(), solution_path.clone(), log_path.clone()];
    if let Some(start_path) = &start_path {
        cleanup_paths.push(start_path.clone());
    }

    if let Err(err) = fs::write(&model_path, model_text) {
        cleanup_native_highs_temp_files(&cleanup_paths);
        return external_cli_failure(
            ExternalLinearCliStatus::NumericalError,
            bridge_solver,
            format!(
                "failed to write HiGHS model file '{}': {err}",
                model_path.display()
            ),
            elapsed_ms(t0),
        );
    }

    let options_text = native_highs_options_text(kind, opts, &log_path);
    if let Some(options_text) = options_text {
        if let Err(err) = fs::write(&options_path, options_text) {
            cleanup_native_highs_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                format!(
                    "failed to write HiGHS options file '{}': {err}",
                    options_path.display()
                ),
                elapsed_ms(t0),
            );
        }
        cleanup_paths.push(options_path.clone());
    }

    let mip_start_objective = if kind == ExternalLinearCliKind::Mip {
        match opts.mip_start.as_deref() {
            Some(mip_start) => {
                if mip_start.len() != variable_count {
                    cleanup_native_highs_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        format!(
                            "mip_start length {} does not match variable count {}",
                            mip_start.len(),
                            variable_count
                        ),
                        elapsed_ms(t0),
                    );
                }
                if mip_start.iter().any(|value| !value.is_finite()) {
                    cleanup_native_highs_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        "mip_start values must be finite".to_string(),
                        elapsed_ms(t0),
                    );
                }
                let objective = dot_f64(objective_coefficients, mip_start);
                if !objective.is_finite() {
                    cleanup_native_highs_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        "mip_start objective must be finite".to_string(),
                        elapsed_ms(t0),
                    );
                }
                let Some(start_path) = &start_path else {
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        "internal HiGHS MIP-start path was unavailable".to_string(),
                        elapsed_ms(t0),
                    );
                };
                if let Err(err) = fs::write(
                    start_path,
                    native_highs_mip_start_text(mip_start, objective),
                ) {
                    cleanup_native_highs_temp_files(&cleanup_paths);
                    return external_cli_failure(
                        ExternalLinearCliStatus::NumericalError,
                        bridge_solver,
                        format!(
                            "failed to write HiGHS MIP-start file '{}': {err}",
                            start_path.display()
                        ),
                        elapsed_ms(t0),
                    );
                }
                Some(objective)
            }
            None => None,
        }
    } else {
        None
    };

    let mut command = Command::new(&command_path);
    command
        .arg("--model_file")
        .arg(&model_path)
        .arg("--solution_file")
        .arg(&solution_path)
        .arg("--time_limit")
        .arg(normalized_time_limit(opts.time_limit_secs).to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if options_path.exists() {
        command.arg("--options_file").arg(&options_path);
    }
    if let Some(start_path) = &start_path {
        command.arg("--read_solution_file").arg(start_path);
    }
    if kind == ExternalLinearCliKind::Lp {
        if let Some(lp_algorithm) = opts.lp_algorithm {
            command.arg("--solver").arg(lp_algorithm.as_str());
        }
    }
    if let Some(random_seed) = normalized_highs_random_seed(opts.random_seed) {
        command.arg("--random_seed").arg(random_seed.to_string());
    }
    if let Some(presolve) = opts.presolve {
        command.arg("--presolve").arg(match presolve {
            ExternalLinearCliPresolve::Auto => "choose",
            ExternalLinearCliPresolve::On => "on",
            ExternalLinearCliPresolve::Off => "off",
        });
    }

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            cleanup_native_highs_temp_files(&cleanup_paths);
            return external_cli_failure(
                ExternalLinearCliStatus::Unavailable,
                bridge_solver,
                format!(
                    "failed to start HiGHS executable '{}': {err}",
                    command_path.display()
                ),
                elapsed_ms(t0),
            );
        }
    };
    let elapsed = elapsed_ms(t0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let solver_version = parse_highs_solver_version(&format!("{stdout}\n{stderr}"))
        .or_else(|| probe_highs_solver_version(&command_path));

    if !solution_path.exists() {
        let status = classify_highs_status("", &stdout, &stderr);
        cleanup_native_highs_temp_files(&cleanup_paths);
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_highs_message("", &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let parsed = match parse_native_highs_solution_file(
        &solution_path,
        variable_count,
        le_count,
        eq_count,
    ) {
        Ok(parsed) => parsed,
        Err(message) => {
            cleanup_native_highs_temp_files(&cleanup_paths);
            let mut failure = external_cli_failure(
                ExternalLinearCliStatus::NumericalError,
                bridge_solver,
                message,
                elapsed,
            );
            failure.solver_version = solver_version;
            return failure;
        }
    };
    cleanup_native_highs_temp_files(&cleanup_paths);

    let status = classify_highs_status(&parsed.status, &stdout, &stderr);
    if !matches!(
        status,
        ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
    ) {
        let mut failure = external_cli_failure(
            if matches!(
                status,
                ExternalLinearCliStatus::Infeasible | ExternalLinearCliStatus::Unbounded
            ) {
                status
            } else {
                ExternalLinearCliStatus::Unavailable
            },
            bridge_solver,
            native_highs_message(&parsed.status, &stdout, &stderr),
            elapsed,
        );
        failure.solver_version = solver_version;
        return failure;
    }

    let objective = dot_f64(objective_coefficients, &parsed.x);
    let quality = parse_highs_mip_quality(kind, objective, &stdout, &stderr);
    let (mip_start_accepted, mip_start_objective) = parse_highs_mip_start_feedback(
        kind,
        opts.mip_start.as_deref(),
        mip_start_objective,
        &stdout,
        &stderr,
    );
    ExternalLinearCliSolution {
        status,
        solver: bridge_solver,
        solver_version,
        x: parsed.x,
        objective: Some(objective),
        objective_values: None,
        lp_algorithm: (kind == ExternalLinearCliKind::Lp)
            .then(|| {
                opts.lp_algorithm
                    .map(|algorithm| algorithm.as_str().to_string())
            })
            .flatten(),
        best_bound: quality.best_bound,
        solution_limit: (kind == ExternalLinearCliKind::Mip)
            .then(|| opts.solution_limit.map(|limit| limit.max(1)))
            .flatten(),
        solution_pool_size: None,
        solutions: None,
        exhausted: None,
        mip_gap: quality.mip_gap,
        absolute_gap: quality.absolute_gap,
        objective_limit: (kind == ExternalLinearCliKind::Mip)
            .then(|| normalized_objective_limit(opts.objective_limit))
            .flatten(),
        primal_feasibility_tolerance: normalized_tolerance(opts.primal_feasibility_tolerance),
        dual_feasibility_tolerance: normalized_tolerance(opts.dual_feasibility_tolerance),
        integer_feasibility_tolerance: (kind == ExternalLinearCliKind::Mip)
            .then(|| normalized_tolerance(opts.integer_feasibility_tolerance))
            .flatten(),
        nodes_explored: quality.nodes_explored,
        threads: opts.threads.filter(|threads| *threads > 0),
        random_seed: normalized_highs_random_seed(opts.random_seed),
        presolve: opts.presolve.map(|presolve| presolve.as_str().to_string()),
        cuts: None,
        heuristics: None,
        branch_rule: None,
        branch_priorities_accepted: None,
        branch_priority_count: None,
        node_selection: None,
        mip_start_accepted,
        mip_start_objective,
        dual_ub: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.dual_ub)
            .flatten(),
        dual_eq: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.dual_eq)
            .flatten(),
        reduced_costs: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.reduced_costs)
            .flatten(),
        var_basis: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.var_basis)
            .flatten(),
        row_basis: (kind == ExternalLinearCliKind::Lp)
            .then_some(parsed.row_basis)
            .flatten(),
        iterations: (kind == ExternalLinearCliKind::Lp)
            .then(|| parse_highs_lp_iterations(&stdout, &stderr))
            .flatten(),
        elapsed_ms: elapsed,
        message: parsed.status,
    }
}

#[derive(Default)]
struct ParsedNativeHighsSolution {
    status: String,
    x: Vec<f64>,
    reduced_costs: Option<Vec<f64>>,
    dual_ub: Option<Vec<f64>>,
    dual_eq: Option<Vec<f64>>,
    var_basis: Option<Vec<String>>,
    row_basis: Option<Vec<String>>,
}

#[derive(Default)]
struct ParsedNativeNamedSolution {
    status: String,
    x: Vec<f64>,
    reduced_costs: Option<Vec<f64>>,
    dual_ub: Option<Vec<f64>>,
    dual_eq: Option<Vec<f64>>,
}

#[derive(Default)]
struct ParsedNativeGlpkSolution {
    status: String,
    x: Vec<f64>,
    reduced_costs: Option<Vec<f64>>,
    dual_ub: Option<Vec<f64>>,
    dual_eq: Option<Vec<f64>>,
    var_basis: Option<Vec<String>>,
    row_basis: Option<Vec<String>>,
}

#[derive(Default)]
struct ParsedNativeScipSolution {
    status: String,
    x: Vec<f64>,
}

#[derive(Default)]
struct ParsedNativeCbcSolution {
    status: String,
    x: Vec<f64>,
    reduced_costs: Option<Vec<f64>>,
    dual_ub: Option<Vec<f64>>,
    dual_eq: Option<Vec<f64>>,
    var_basis: Option<Vec<String>>,
    row_basis: Option<Vec<String>>,
}

#[derive(Default)]
struct ParsedNativeSoplexSolution {
    status: String,
    x: Vec<f64>,
    reduced_costs: Option<Vec<f64>>,
    dual_ub: Option<Vec<f64>>,
    dual_eq: Option<Vec<f64>>,
    var_basis: Option<Vec<String>>,
    row_basis: Option<Vec<String>>,
}

#[derive(Default)]
struct ParsedNativeLpSolveSolution {
    status: String,
    x: Vec<f64>,
    objective: Option<f64>,
}

#[derive(Default)]
struct NativeLpSolveCertificateFields {
    reduced_costs: Option<Vec<f64>>,
    dual_ub: Option<Vec<f64>>,
    dual_eq: Option<Vec<f64>>,
    var_basis: Option<Vec<String>>,
    row_basis: Option<Vec<String>>,
}

#[derive(Default)]
struct HighsMipQuality {
    best_bound: Option<f64>,
    mip_gap: Option<f64>,
    absolute_gap: Option<f64>,
    nodes_explored: Option<u64>,
}

fn native_highs_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("highs", stem, extension)
}

fn native_gurobi_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("gurobi", stem, extension)
}

fn native_cplex_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("cplex", stem, extension)
}

fn native_xpress_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("xpress", stem, extension)
}

fn native_lindo_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("lindo", stem, extension)
}

fn native_lindo_gams_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("lindo-gams", stem, extension)
}

fn native_xpress_header_path(solution_path: &Path) -> PathBuf {
    native_xpress_companion_path(solution_path, ".hdr")
}

fn native_xpress_solution_data_path(solution_path: &Path) -> PathBuf {
    native_xpress_companion_path(solution_path, ".asc")
}

fn native_xpress_companion_path(solution_path: &Path, suffix: &str) -> PathBuf {
    let mut path = solution_path.as_os_str().to_os_string();
    path.push(suffix);
    PathBuf::from(path)
}

fn native_lindo_solution_path(model_path: &Path) -> PathBuf {
    let mut path = model_path.to_path_buf();
    path.set_extension("sol");
    path
}

fn native_solver_temp_path(solver: &str, stem: &str, extension: &str) -> PathBuf {
    static TEMP_PATH_COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let sequence = TEMP_PATH_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ores-native-{solver}-{stem}-{}-{nanos}-{sequence}.{extension}",
        std::process::id()
    ))
}

fn cleanup_native_highs_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn cleanup_native_gurobi_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn cleanup_native_cplex_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn cleanup_native_xpress_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn cleanup_native_lindo_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn native_xpress_command_text(
    kind: ExternalLinearCliKind,
    model_path: &Path,
    solution_path: &Path,
    opts: &ExternalLinearCliOptions,
) -> String {
    let mut text = String::new();
    let time_limit = normalized_time_limit(opts.time_limit_secs).ceil().max(1.0) as u64;
    text.push_str(&format!("MAXTIME = -{time_limit}\n"));
    if kind == ExternalLinearCliKind::Mip {
        if let Some(solution_limit) = opts.solution_limit {
            text.push_str(&format!("MAXSOLS = {}\n", solution_limit.max(1)));
        }
    }
    if let Some(tolerance) = normalized_tolerance(opts.primal_feasibility_tolerance) {
        text.push_str(&format!("FEASTOL = {tolerance:.17}\n"));
    }
    if let Some(tolerance) = normalized_tolerance(opts.dual_feasibility_tolerance) {
        text.push_str(&format!("OPTTOL = {tolerance:.17}\n"));
    }
    if kind == ExternalLinearCliKind::Mip {
        if let Some(tolerance) = normalized_tolerance(opts.integer_feasibility_tolerance) {
            text.push_str(&format!("MIPTOL = {tolerance:.17}\n"));
        }
    }
    if let Some(threads) = opts.threads.filter(|threads| *threads > 0) {
        text.push_str(&format!("THREADS = {threads}\n"));
    }
    if let Some(seed) = opts.random_seed.filter(|seed| *seed <= i32::MAX as u64) {
        text.push_str(&format!("RANDOMSEED = {seed}\n"));
    }
    if let Some(presolve) = opts.presolve {
        let value = match presolve {
            ExternalLinearCliPresolve::Auto => -1,
            ExternalLinearCliPresolve::On => 1,
            ExternalLinearCliPresolve::Off => 0,
        };
        text.push_str(&format!("PRESOLVE = {value}\n"));
    }
    let read_flag = match opts.model_format {
        ExternalLinearCliModelFormat::CplexLp => "-l",
        ExternalLinearCliModelFormat::Mps => "-m",
    };
    text.push_str(&format!("readprob {read_flag} {}\n", model_path.display()));
    text.push_str(if kind == ExternalLinearCliKind::Mip {
        "mipoptimize\n"
    } else {
        "lpoptimize\n"
    });
    text.push_str(&format!("writesol {} -npa\n", solution_path.display()));
    text.push_str("quit\n");
    text
}

fn native_highs_mip_start_text(start: &[f64], objective: f64) -> String {
    let mut text = String::new();
    text.push_str("Model status\n");
    text.push_str("Unknown\n\n");
    text.push_str("# Primal solution values\n");
    text.push_str("Feasible\n");
    text.push_str(&format!("Objective {objective:.17}\n"));
    text.push_str(&format!("# Columns {}\n", start.len()));
    for (idx, value) in start.iter().enumerate() {
        text.push_str(&format!("x{idx} {value:.17}\n"));
    }
    text.push_str("# Rows 0\n\n");
    text.push_str("# Dual solution values\n");
    text.push_str("None\n\n");
    text.push_str("# Basis\n");
    text.push_str("HiGHS_basis_file v2\n");
    text.push_str("None\n");
    text
}

fn native_glpk_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("glpk", stem, extension)
}

fn cleanup_native_glpk_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn native_scip_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("scip", stem, extension)
}

fn native_scip_mip_start_text(start: &[f64], objective: f64) -> String {
    let mut text = String::new();
    text.push_str("solution status: feasible\n");
    text.push_str(&format!("objective value: {objective:.17}\n"));
    for (idx, value) in start.iter().enumerate() {
        text.push_str(&format!("x{idx} {value:.17}\n"));
    }
    text
}

fn cleanup_native_scip_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn native_cbc_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("cbc", stem, extension)
}

fn native_cbc_mip_start_text(start: &[f64]) -> String {
    let mut text = String::new();
    for (idx, value) in start.iter().enumerate() {
        text.push_str(&format!("{idx} x{idx} {value:.17}\n"));
    }
    text
}

fn native_cbc_branch_priorities_text(active_priorities: &[(usize, i32)]) -> String {
    let highest = active_priorities
        .iter()
        .map(|(_, priority)| *priority)
        .max()
        .unwrap_or(0);
    let mut text = "name,priority\n".to_string();
    for (idx, priority) in active_priorities {
        let cbc_priority = highest - *priority + 1;
        text.push_str(&format!("x{idx},{cbc_priority}\n"));
    }
    text
}

fn cleanup_native_cbc_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn native_clp_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("clp", stem, extension)
}

fn cleanup_native_clp_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn native_soplex_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("soplex", stem, extension)
}

fn cleanup_native_soplex_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn native_qsopt_ex_temp_path(stem: &str, extension: &str) -> PathBuf {
    native_solver_temp_path("qsopt-ex", stem, extension)
}

fn cleanup_native_qsopt_ex_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn native_lp_solve_temp_path(stem: &str, extension: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(
        "ores-native-lp-solve-{stem}-{}-{nanos}.{extension}",
        std::process::id()
    ))
}

fn cleanup_native_lp_solve_temp_files(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn glpk_time_limit_arg(time_limit_secs: Option<f64>) -> String {
    normalized_time_limit(time_limit_secs)
        .ceil()
        .max(1.0)
        .to_string()
}

fn normalized_glpk_random_seed(random_seed: Option<u64>) -> Option<u64> {
    random_seed.filter(|seed| *seed <= i32::MAX as u64)
}

fn cbc_time_limit_arg(time_limit_secs: Option<f64>) -> String {
    format!("{:.17}", normalized_time_limit(time_limit_secs))
}

fn normalized_cbc_random_seed(random_seed: Option<u64>) -> Option<u64> {
    random_seed.filter(|seed| *seed <= i32::MAX as u64)
}

fn native_highs_options_text(
    kind: ExternalLinearCliKind,
    opts: &ExternalLinearCliOptions,
    log_path: &Path,
) -> Option<String> {
    let mut lines = vec![format!("log_file = {}", log_path.display())];
    if let Some(threads) = opts.threads.filter(|threads| *threads > 0) {
        lines.push(format!("threads = {threads}"));
    }
    if let Some(tolerance) = normalized_tolerance(opts.primal_feasibility_tolerance) {
        lines.push(format!("primal_feasibility_tolerance = {tolerance:.17}"));
    }
    if let Some(tolerance) = normalized_tolerance(opts.dual_feasibility_tolerance) {
        lines.push(format!("dual_feasibility_tolerance = {tolerance:.17}"));
    }
    if kind == ExternalLinearCliKind::Mip {
        if let Some(max_nodes) = opts
            .max_nodes
            .or_else(|| opts.node_limit.map(|limit| limit as u64))
            .filter(|limit| *limit > 0)
        {
            lines.push(format!("mip_max_nodes = {max_nodes}"));
        }
        if let Some(solution_limit) = opts.solution_limit {
            lines.push(format!(
                "mip_max_improving_sols = {}",
                solution_limit.max(1)
            ));
        }
        if let Some(relative_gap) = normalized_relative_gap(opts.relative_gap) {
            lines.push(format!("mip_rel_gap = {relative_gap:.17}"));
        }
        if let Some(absolute_gap) = normalized_absolute_gap(opts.absolute_gap) {
            lines.push(format!("mip_abs_gap = {absolute_gap:.17}"));
        }
        if let Some(objective_limit) = normalized_objective_limit(opts.objective_limit) {
            lines.push(format!("objective_target = {objective_limit:.17}"));
        }
        if let Some(tolerance) = normalized_tolerance(opts.integer_feasibility_tolerance) {
            lines.push(format!("mip_feasibility_tolerance = {tolerance:.17}"));
        }
    }
    if lines.is_empty() {
        None
    } else {
        let mut text = lines.join("\n");
        text.push('\n');
        Some(text)
    }
}

fn parse_native_highs_solution_file(
    path: &Path,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
) -> Result<ParsedNativeHighsSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read HiGHS solution file '{}': {err}",
            path.display()
        )
    })?;
    parse_native_highs_solution_text(&text, variable_count, le_count, eq_count)
}

fn parse_native_highs_solution_text(
    text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
) -> Result<ParsedNativeHighsSolution, String> {
    let mut x = vec![0.0; variable_count];
    let mut status = "unknown".to_string();
    let mut dual_columns = vec![None::<f64>; variable_count];
    let mut dual_rows = std::collections::BTreeMap::<String, f64>::new();
    let mut var_basis = vec![None::<String>; variable_count];
    let mut row_basis = std::collections::BTreeMap::<String, String>::new();
    let lines = text.lines().map(str::trim).collect::<Vec<_>>();
    let mut section: Option<&str> = None;
    let mut block: Option<&str> = None;
    let mut remaining = 0_usize;

    for (idx, line) in lines.iter().enumerate() {
        if *line == "Model status" {
            if let Some(next) = lines.get(idx + 1) {
                status = next.to_ascii_lowercase();
            }
            continue;
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
            let value = parts[1].parse::<f64>().ok();
            match (section, block) {
                (Some("primal"), Some("columns")) => {
                    if let (Some(index), Some(value)) = (highs_variable_index(parts[0]), value) {
                        if index < variable_count {
                            x[index] = value;
                        }
                    }
                }
                (Some("dual"), Some("columns")) => {
                    if let (Some(index), Some(value)) = (highs_variable_index(parts[0]), value) {
                        if index < variable_count {
                            dual_columns[index] = Some(value);
                        }
                    }
                }
                (Some("dual"), Some("rows")) => {
                    if let Some(value) = value {
                        dual_rows.insert(parts[0].to_string(), value);
                    }
                }
                (Some("basis"), Some("columns")) => {
                    if let Some(index) = highs_variable_index(parts[0]) {
                        if index < variable_count {
                            if let Some(status) = highs_basis_status(parts[1]) {
                                var_basis[index] = Some(status.to_string());
                            }
                        }
                    }
                }
                (Some("basis"), Some("rows")) => {
                    if let Some(status) = highs_basis_status(parts[1]) {
                        row_basis.insert(parts[0].to_string(), status.to_string());
                    }
                }
                _ => {}
            }
        }

        if remaining > 0 {
            remaining -= 1;
            if remaining == 0 {
                block = None;
            }
        }
    }

    let reduced_costs = all_some_f64(&dual_columns);
    let dual_ub = row_values(&dual_rows, "c", le_count);
    let dual_eq = row_values(&dual_rows, "e", eq_count);
    let var_basis = all_some_string(&var_basis);
    let mut rows = Vec::with_capacity(le_count + eq_count);
    for idx in 0..le_count {
        rows.push(row_basis.get(&format!("c{idx}")).cloned());
    }
    for idx in 0..eq_count {
        rows.push(row_basis.get(&format!("e{idx}")).cloned());
    }
    let row_basis = all_some_string(&rows);

    Ok(ParsedNativeHighsSolution {
        status,
        x,
        reduced_costs,
        dual_ub,
        dual_eq,
        var_basis,
        row_basis,
    })
}

fn parse_native_glpk_solution_file(
    path: &Path,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
) -> Result<ParsedNativeGlpkSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read GLPK solution file '{}': {err}",
            path.display()
        )
    })?;
    parse_native_glpk_solution_text(&text, variable_count, le_count, eq_count)
}

fn parse_native_glpk_solution_text(
    text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
) -> Result<ParsedNativeGlpkSolution, String> {
    let mut x = vec![0.0; variable_count];
    let mut status = "unknown".to_string();
    let row_count = le_count + eq_count;
    let mut row_duals = vec![None::<f64>; row_count];
    let mut reduced_costs = vec![None::<f64>; variable_count];
    let mut var_basis = vec![None::<String>; variable_count];
    let mut row_basis = vec![None::<String>; row_count];
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
            && (line.trim().starts_with("Integer feasibility")
                || line.trim().starts_with("KKT.")
                || line.trim().starts_with("End of output"))
        {
            in_named_columns = false;
        } else if in_named_columns
            && parts.len() >= 3
            && parts[0].chars().all(|ch| ch.is_ascii_digit())
            && parts[1].starts_with('x')
        {
            if let Some(index) = highs_variable_index(parts[1]) {
                if index < variable_count {
                    if let Some(value) = parts[2..]
                        .iter()
                        .copied()
                        .filter(|token| *token != "*")
                        .find_map(parse_f64_token)
                    {
                        x[index] = value;
                    }
                }
            }
        } else if parts.len() >= 3 && parts[0] == "j" {
            let Some(index) = parts[1]
                .parse::<usize>()
                .ok()
                .and_then(|value| value.checked_sub(1))
            else {
                continue;
            };
            if index < variable_count {
                if parts.len() >= 4 && parse_f64_token(parts[2]).is_none() {
                    if let Some(value) = parse_f64_token(parts[3]) {
                        x[index] = value;
                    }
                    if let Some(status) = highs_basis_status(parts[2]) {
                        var_basis[index] = Some(status.to_string());
                    }
                    if parts.len() >= 5 {
                        reduced_costs[index] = parse_f64_token(parts[4]);
                    }
                } else if let Some(value) = parse_f64_token(parts[2]) {
                    x[index] = value;
                }
            }
        } else if parts.len() >= 5 && parts[0] == "i" {
            let Some(index) = parts[1]
                .parse::<usize>()
                .ok()
                .and_then(|value| value.checked_sub(1))
            else {
                continue;
            };
            if index < row_count {
                if let Some(status) = highs_basis_status(parts[2]) {
                    row_basis[index] = Some(status.to_string());
                }
                row_duals[index] = parse_f64_token(parts[4]);
            }
        }
    }

    let dual_ub = all_some_f64(&row_duals[..le_count]);
    let dual_eq = all_some_f64(&row_duals[le_count..]);
    Ok(ParsedNativeGlpkSolution {
        status,
        x,
        reduced_costs: all_some_f64(&reduced_costs),
        dual_ub,
        dual_eq,
        var_basis: all_some_string(&var_basis),
        row_basis: all_some_string(&row_basis),
    })
}

fn parse_native_scip_solution_file(
    path: &Path,
    variable_count: usize,
    stdout: &str,
    stderr: &str,
) -> Result<ParsedNativeScipSolution, String> {
    if !path.exists() {
        return Err(native_solver_message("", stdout, stderr));
    }
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read SCIP solution file '{}': {err}",
            path.display()
        )
    })?;
    Ok(parse_native_scip_solution_text(&text, variable_count))
}

fn parse_native_scip_solution_text(text: &str, variable_count: usize) -> ParsedNativeScipSolution {
    let mut parsed = ParsedNativeScipSolution {
        status: "unknown".to_string(),
        x: vec![0.0; variable_count],
    };
    for line in text.lines() {
        let stripped = line.trim();
        if let Some((_, status)) = stripped.split_once("solution status:") {
            parsed.status = status.trim().to_ascii_lowercase();
            continue;
        }
        if let Some((index, value)) = parse_named_variable_value_line(stripped, variable_count) {
            parsed.x[index] = value;
        }
    }
    parsed
}

struct ScipLpCertificateFields {
    dual_ub: Option<Vec<f64>>,
    dual_eq: Option<Vec<f64>>,
    reduced_costs: Option<Vec<f64>>,
    var_basis: Option<Vec<String>>,
    row_basis: Option<Vec<String>>,
}

fn parse_scip_lp_certificate_fields(
    stdout: &str,
    _sense: Sense,
    objective_coefficients: &[f64],
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    lower_bounds: Option<&[Option<f64>]>,
    upper_bounds: Option<&[Option<f64>]>,
    x: &[f64],
) -> ScipLpCertificateFields {
    let mut dual_ub = vec![0.0; le_rows.len()];
    let mut dual_eq = vec![0.0; eq_rows.len()];
    let mut saw_dual = false;
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next().and_then(parse_f64_token) else {
            continue;
        };
        if let Some(index) = numbered_name_suffix(name, "c") {
            if index < dual_ub.len() {
                dual_ub[index] = clean_certificate_value(value);
                saw_dual = true;
            }
        } else if let Some(index) = numbered_name_suffix(name, "e") {
            if index < dual_eq.len() {
                dual_eq[index] = clean_certificate_value(value);
                saw_dual = true;
            }
        }
    }

    let reduced_costs = saw_dual.then(|| {
        reduced_costs_from_row_duals(objective_coefficients, le_rows, &dual_ub, eq_rows, &dual_eq)
    });
    let (var_basis, row_basis) = reduced_costs
        .as_deref()
        .map(|reduced_costs| {
            infer_lp_basis_from_complementarity(
                x,
                lower_bounds,
                upper_bounds,
                le_rows,
                le_rhs,
                &dual_ub,
                eq_rows,
                eq_rhs,
                reduced_costs,
            )
        })
        .unwrap_or((None, None));
    ScipLpCertificateFields {
        dual_ub: saw_dual.then_some(dual_ub),
        dual_eq: saw_dual.then_some(dual_eq),
        reduced_costs,
        var_basis,
        row_basis,
    }
}

fn infer_lp_basis_from_complementarity(
    x: &[f64],
    lower_bounds: Option<&[Option<f64>]>,
    upper_bounds: Option<&[Option<f64>]>,
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    dual_ub: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    reduced_costs: &[f64],
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    const TOL: f64 = 1.0e-7;
    if x.len() != reduced_costs.len()
        || le_rows.len() != le_rhs.len()
        || le_rows.len() != dual_ub.len()
        || eq_rows.len() != eq_rhs.len()
    {
        return (None, None);
    }

    let mut var_basis = Vec::with_capacity(x.len());
    for (idx, (&value, &reduced_cost)) in x.iter().zip(reduced_costs).enumerate() {
        let lower = lp_lower_bound_at(lower_bounds, idx);
        let upper = lp_upper_bound_at(upper_bounds, idx);
        let at_lower = lower.is_some_and(|bound| (value - bound).abs() <= TOL);
        let at_upper = upper.is_some_and(|bound| (value - bound).abs() <= TOL);
        let fixed = lower
            .zip(upper)
            .is_some_and(|(lower, upper)| (lower - upper).abs() <= TOL);
        let status = if fixed && (at_lower || at_upper) {
            "fixed"
        } else if at_lower && reduced_cost.abs() > TOL {
            "at_lower"
        } else if at_upper && reduced_cost.abs() > TOL {
            "at_upper"
        } else if !at_lower && !at_upper && reduced_cost.abs() <= TOL {
            "basic"
        } else {
            return (None, None);
        };
        var_basis.push(status.to_string());
    }

    let mut row_basis = Vec::with_capacity(le_rows.len() + eq_rows.len());
    for ((row, &rhs), &dual) in le_rows.iter().zip(le_rhs).zip(dual_ub) {
        let activity = dot_f64(row, x);
        let slack = rhs - activity;
        if slack < -TOL {
            return (Some(var_basis), None);
        }
        if slack.abs() <= TOL {
            if dual.abs() > TOL {
                row_basis.push("at_upper".to_string());
            } else {
                return (Some(var_basis), None);
            }
        } else {
            row_basis.push("basic".to_string());
        }
    }
    for (row, &rhs) in eq_rows.iter().zip(eq_rhs) {
        if (dot_f64(row, x) - rhs).abs() > TOL {
            return (Some(var_basis), None);
        }
        row_basis.push("fixed".to_string());
    }

    (Some(var_basis), Some(row_basis))
}

fn lp_lower_bound_at(bounds: Option<&[Option<f64>]>, index: usize) -> Option<f64> {
    bounds
        .and_then(|bounds| bounds.get(index).copied())
        .unwrap_or(Some(0.0))
}

fn lp_upper_bound_at(bounds: Option<&[Option<f64>]>, index: usize) -> Option<f64> {
    bounds
        .and_then(|bounds| bounds.get(index).copied())
        .unwrap_or(None)
}

fn reduced_costs_from_row_duals(
    objective_coefficients: &[f64],
    le_rows: &[Vec<f64>],
    dual_ub: &[f64],
    eq_rows: &[Vec<f64>],
    dual_eq: &[f64],
) -> Vec<f64> {
    objective_coefficients
        .iter()
        .enumerate()
        .map(|(col, coeff)| {
            let le_part = le_rows
                .iter()
                .zip(dual_ub)
                .filter_map(|(row, dual)| row.get(col).map(|value| value * dual))
                .sum::<f64>();
            let eq_part = eq_rows
                .iter()
                .zip(dual_eq)
                .filter_map(|(row, dual)| row.get(col).map(|value| value * dual))
                .sum::<f64>();
            clean_certificate_value(coeff - le_part - eq_part)
        })
        .collect()
}

fn clean_certificate_value(value: f64) -> f64 {
    if value.abs() <= 1e-8 {
        0.0
    } else {
        value
    }
}

fn numbered_name_suffix(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)
        .filter(|suffix| !suffix.is_empty())
        .and_then(|suffix| suffix.parse::<usize>().ok())
}

fn parse_native_cbc_solution_file(
    path: &Path,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    basis_path: Option<&Path>,
) -> Result<ParsedNativeCbcSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read CBC solution file '{}': {err}",
            path.display()
        )
    })?;
    let basis_text =
        match basis_path.filter(|path| path.exists()) {
            Some(path) => Some(fs::read_to_string(path).map_err(|err| {
                format!("failed to read CBC basis file '{}': {err}", path.display())
            })?),
            None => None,
        };
    parse_native_cbc_solution_text(
        &text,
        variable_count,
        le_count,
        eq_count,
        basis_text.as_deref(),
    )
}

fn parse_native_cbc_solution_text(
    text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    basis_text: Option<&str>,
) -> Result<ParsedNativeCbcSolution, String> {
    let mut x = vec![0.0; variable_count];
    let mut status = "unknown".to_string();
    let row_count = le_count + eq_count;
    let mut row_duals = vec![None::<f64>; row_count];
    let mut reduced_costs = vec![None::<f64>; variable_count];

    for (line_no, line) in text.lines().enumerate() {
        let mut parts = line.split_whitespace().collect::<Vec<_>>();
        if line_no == 0 && !line.trim().is_empty() {
            status = line.trim().to_ascii_lowercase();
            continue;
        }
        if parts.first() == Some(&"**") {
            parts.remove(0);
        }
        if parts.len() >= 3 && signed_usize_token(parts[0]).is_some() && parts[1].starts_with('x') {
            if let Some(index) = highs_variable_index(parts[1]) {
                if index < variable_count {
                    if let Some(value) = parse_f64_token(parts[2]) {
                        x[index] = value;
                    }
                    if parts.len() >= 4 {
                        reduced_costs[index] = parse_f64_token(parts[3]);
                    }
                }
            }
        } else if parts.len() >= 4 && signed_usize_token(parts[0]).is_some() {
            let row_index = if let Some(suffix) = parts[1].strip_prefix('c') {
                suffix.parse::<usize>().ok()
            } else if let Some(suffix) = parts[1].strip_prefix('e') {
                suffix
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| le_count.checked_add(index))
            } else {
                None
            };
            if let Some(index) = row_index.filter(|index| *index < row_count) {
                row_duals[index] = parse_f64_token(parts[3]);
            }
        }
    }

    let (var_basis, row_basis) = basis_text
        .map(|text| parse_native_cbc_basis_text(text, variable_count, le_count, eq_count))
        .unwrap_or((None, None));
    Ok(ParsedNativeCbcSolution {
        status,
        x,
        reduced_costs: all_some_f64(&reduced_costs),
        dual_ub: all_some_f64(&row_duals[..le_count]),
        dual_eq: all_some_f64(&row_duals[le_count..]),
        var_basis,
        row_basis,
    })
}

fn parse_native_cbc_basis_text(
    text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    let mut var_basis = vec![None::<String>; variable_count];
    let mut row_basis = vec![Some("basic".to_string()); le_count];
    row_basis.extend((0..eq_count).map(|_| Some("fixed".to_string())));

    for line in text.lines() {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.is_empty() || matches!(parts[0], "NAME" | "ENDATA") {
            continue;
        }
        let code = parts[0].to_ascii_uppercase();
        if parts.len() >= 2 && parts[1].starts_with('x') {
            if let Some(index) = highs_variable_index(parts[1]) {
                if index < variable_count {
                    let status = match code.as_str() {
                        "BS" | "XL" | "XU" => Some("basic"),
                        "LL" => Some("at_lower"),
                        "UL" => Some("at_upper"),
                        "FX" => Some("fixed"),
                        "FR" => Some("free"),
                        _ => None,
                    };
                    if let Some(status) = status {
                        var_basis[index] = Some(status.to_string());
                    }
                }
            }
        }
        if matches!(code.as_str(), "XL" | "XU") && parts.len() >= 3 {
            let status = if code == "XL" { "at_lower" } else { "at_upper" };
            if let Some(index) = basis_row_index(parts[2], le_count, eq_count) {
                row_basis[index] = Some(status.to_string());
            }
        }
    }

    (all_some_string(&var_basis), all_some_string(&row_basis))
}

fn parse_native_soplex_solution_file(
    path: &Path,
    dual_path: Option<&Path>,
    basis_path: Option<&Path>,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    stdout: &str,
    stderr: &str,
) -> Result<ParsedNativeSoplexSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read SoPlex solution file '{}': {err}",
            path.display()
        )
    })?;
    let mut parsed = parse_native_soplex_solution_text(&text, variable_count, stdout, stderr);
    if let Some(path) = dual_path.filter(|path| path.exists()) {
        let text = fs::read_to_string(path).map_err(|err| {
            format!(
                "failed to read SoPlex dual solution file '{}': {err}",
                path.display()
            )
        })?;
        let fields = parse_native_soplex_dual_text(&text, variable_count, le_count, eq_count);
        parsed.reduced_costs = fields.reduced_costs;
        parsed.dual_ub = fields.dual_ub;
        parsed.dual_eq = fields.dual_eq;
    }
    if let Some(path) = basis_path.filter(|path| path.exists()) {
        let text = fs::read_to_string(path).map_err(|err| {
            format!(
                "failed to read SoPlex basis file '{}': {err}",
                path.display()
            )
        })?;
        let (var_basis, row_basis) =
            parse_native_cbc_basis_text(&text, variable_count, le_count, eq_count);
        parsed.var_basis = var_basis;
        parsed.row_basis = row_basis;
    }
    Ok(parsed)
}

fn parse_native_soplex_solution_text(
    text: &str,
    variable_count: usize,
    stdout: &str,
    stderr: &str,
) -> ParsedNativeSoplexSolution {
    let mut x = vec![0.0; variable_count];
    let lower = format!("{stdout}\n{stderr}\n{text}").to_ascii_lowercase();
    let status = if lower.contains("infeasible") {
        "infeasible"
    } else if lower.contains("unbounded") {
        "unbounded"
    } else if lower.contains("problem is solved [optimal]") || lower.contains("primal solution") {
        "optimal"
    } else {
        "unknown"
    }
    .to_string();

    for line in text.lines() {
        if let Some((idx, value)) = parse_named_variable_value_line(line, variable_count) {
            x[idx] = value;
        }
    }
    ParsedNativeSoplexSolution {
        status,
        x,
        ..Default::default()
    }
}

fn basis_row_index(name: &str, le_count: usize, eq_count: usize) -> Option<usize> {
    if let Some(index) = name
        .strip_prefix('c')
        .and_then(|suffix| suffix.parse::<usize>().ok())
        .filter(|index| *index < le_count)
    {
        return Some(index);
    }
    name.strip_prefix('e')
        .and_then(|suffix| suffix.parse::<usize>().ok())
        .filter(|index| *index < eq_count)
        .map(|index| le_count + index)
}

#[derive(Default)]
struct NativeSoplexDualFields {
    reduced_costs: Option<Vec<f64>>,
    dual_ub: Option<Vec<f64>>,
    dual_eq: Option<Vec<f64>>,
}

fn parse_native_soplex_dual_text(
    text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
) -> NativeSoplexDualFields {
    let mut row_duals = vec![0.0; le_count + eq_count];
    let mut reduced_costs = vec![0.0; variable_count];
    let mut section = None::<&str>;
    let mut saw_dual = false;
    let mut saw_reduced = false;

    for line in text.lines() {
        let stripped = line.trim();
        if stripped.is_empty() {
            continue;
        }
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("dual solution") {
            section = Some("dual");
            saw_dual = true;
            continue;
        }
        if lowered.starts_with("reduced costs") {
            section = Some("reduced");
            saw_reduced = true;
            continue;
        }
        if lowered.starts_with("all other") {
            continue;
        }

        let parts = stripped.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let Some(value) = parts.iter().rev().find_map(|token| parse_f64_token(token)) else {
            continue;
        };
        match section {
            Some("dual") => {
                if let Some(index) = basis_row_index(parts[0], le_count, eq_count) {
                    row_duals[index] = value;
                }
            }
            Some("reduced") => {
                if let Some(index) =
                    highs_variable_index(parts[0]).filter(|index| *index < variable_count)
                {
                    reduced_costs[index] = value;
                }
            }
            _ => {}
        }
    }

    NativeSoplexDualFields {
        reduced_costs: saw_reduced.then_some(reduced_costs),
        dual_ub: saw_dual.then(|| row_duals[..le_count].to_vec()),
        dual_eq: saw_dual.then(|| row_duals[le_count..].to_vec()),
    }
}

fn parse_native_qsopt_ex_solution_file(
    path: &Path,
    basis_path: Option<&Path>,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    stdout: &str,
    stderr: &str,
) -> Result<ParsedNativeSoplexSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read QSopt_ex solution file '{}': {err}",
            path.display()
        )
    })?;
    let mut parsed = parse_native_qsopt_ex_solution_text(
        &text,
        variable_count,
        le_count,
        eq_count,
        stdout,
        stderr,
    );
    if let Some(path) = basis_path.filter(|path| path.exists()) {
        let text = fs::read_to_string(path).map_err(|err| {
            format!(
                "failed to read QSopt_ex basis file '{}': {err}",
                path.display()
            )
        })?;
        let (var_basis, row_basis) =
            parse_native_qsopt_ex_basis_text(&text, variable_count, le_count, eq_count);
        parsed.var_basis = var_basis;
        parsed.row_basis = row_basis;
    }
    Ok(parsed)
}

fn parse_native_qsopt_ex_solution_text(
    text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
    stdout: &str,
    stderr: &str,
) -> ParsedNativeSoplexSolution {
    let mut x = vec![0.0; variable_count];
    let mut reduced_costs = vec![0.0; variable_count];
    let mut row_duals = vec![0.0; le_count + eq_count];
    let solution_lower = text.to_ascii_lowercase();
    let combined_lower = format!("{stdout}\n{stderr}\n{text}").to_ascii_lowercase();
    let status = if solution_lower.contains("status optimal")
        || combined_lower.contains("problem solved exactly")
    {
        "optimal"
    } else if solution_lower.contains("status infeasible") || combined_lower.contains("infeasible")
    {
        "infeasible"
    } else if solution_lower.contains("status unbounded") || combined_lower.contains("unbounded") {
        "unbounded"
    } else if solution_lower.contains("optimal")
        || solution_lower.contains("objective")
        || solution_lower.contains("primal solution")
    {
        "optimal"
    } else {
        "unknown"
    }
    .to_string();

    let mut section = None::<&str>;
    let mut saw_reduced_costs = false;
    let mut saw_pi = false;
    for line in text.lines() {
        let stripped = line.trim();
        let upper = stripped.to_ascii_uppercase();
        if upper == "VARS:" {
            section = Some("vars");
            continue;
        }
        if upper.starts_with("REDUCED COST") {
            section = Some("reduced");
            saw_reduced_costs = true;
            continue;
        }
        if upper.starts_with("PI") {
            section = Some("pi");
            saw_pi = true;
            continue;
        }
        if upper.starts_with("SLACK") {
            section = None;
            continue;
        }

        match section {
            Some("vars") => {
                if let Some((index, value)) =
                    parse_named_variable_value_line(stripped, variable_count)
                {
                    x[index] = value;
                }
            }
            Some("reduced") => {
                if let Some((index, value)) =
                    parse_prefixed_value_line(stripped, 'x', variable_count)
                {
                    reduced_costs[index] = value;
                }
            }
            Some("pi") => {
                if let Some((index, value)) = parse_prefixed_value_line(stripped, 'c', le_count) {
                    row_duals[index] = value;
                    continue;
                }
                if let Some((index, value)) = parse_prefixed_value_line(stripped, 'e', eq_count) {
                    row_duals[le_count + index] = value;
                }
            }
            _ => {}
        }
    }

    ParsedNativeSoplexSolution {
        status,
        x,
        reduced_costs: saw_reduced_costs.then_some(reduced_costs),
        dual_ub: (saw_pi && le_count > 0).then(|| row_duals[..le_count].to_vec()),
        dual_eq: (saw_pi && eq_count > 0).then(|| row_duals[le_count..].to_vec()),
        var_basis: None,
        row_basis: None,
    }
}

fn parse_native_qsopt_ex_basis_text(
    text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    let mut var_basis = vec![None::<String>; variable_count];
    let mut row_basis = vec![Some("basic".to_string()); le_count + eq_count];
    for line in text.lines() {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 3 || !matches!(parts[0].to_ascii_uppercase().as_str(), "XL" | "XU") {
            continue;
        }
        if let Some(index) = highs_variable_index(parts[1]).filter(|index| *index < variable_count)
        {
            var_basis[index] = Some("basic".to_string());
        }
        if let Some(index) = basis_row_index(parts[2], le_count, eq_count) {
            row_basis[index] = Some("at_upper".to_string());
        }
    }
    (all_some_string(&var_basis), all_some_string(&row_basis))
}

fn qsopt_ex_lp_basis_from_solution(
    parsed: &ParsedNativeSoplexSolution,
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    lower_bounds: Option<&[Option<f64>]>,
    upper_bounds: Option<&[Option<f64>]>,
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    let (Some(dual_ub), Some(reduced_costs)) =
        (parsed.dual_ub.as_deref(), parsed.reduced_costs.as_deref())
    else {
        return (None, None);
    };
    infer_lp_basis_from_complementarity(
        &parsed.x,
        lower_bounds,
        upper_bounds,
        le_rows,
        le_rhs,
        dual_ub,
        eq_rows,
        eq_rhs,
        reduced_costs,
    )
}

fn qsopt_ex_lp_certificate_needs_reconstruction(
    parsed: &ParsedNativeSoplexSolution,
    objective_coefficients: &[f64],
    le_rows: &[Vec<f64>],
    eq_rows: &[Vec<f64>],
) -> bool {
    const TOL: f64 = 1.0e-6;
    let Some(reduced_costs) = parsed
        .reduced_costs
        .as_deref()
        .filter(|values| values.len() == objective_coefficients.len())
    else {
        return true;
    };
    let dual_ub = if le_rows.is_empty() {
        parsed.dual_ub.as_deref().unwrap_or(&[])
    } else {
        let Some(values) = parsed
            .dual_ub
            .as_deref()
            .filter(|values| values.len() == le_rows.len())
        else {
            return true;
        };
        values
    };
    let dual_eq = if eq_rows.is_empty() {
        parsed.dual_eq.as_deref().unwrap_or(&[])
    } else {
        let Some(values) = parsed
            .dual_eq
            .as_deref()
            .filter(|values| values.len() == eq_rows.len())
        else {
            return true;
        };
        values
    };
    let expected =
        reduced_costs_from_row_duals(objective_coefficients, le_rows, dual_ub, eq_rows, dual_eq);
    max_abs_diff(reduced_costs, &expected).is_none_or(|diff| diff > TOL)
}

fn qsopt_ex_lp_certificate_from_basis(
    parsed: &ParsedNativeSoplexSolution,
    objective_coefficients: &[f64],
    le_rows: &[Vec<f64>],
    eq_rows: &[Vec<f64>],
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    const TOL: f64 = 1.0e-7;
    let var_basis = parsed.var_basis.as_deref()?;
    let row_basis = parsed.row_basis.as_deref()?;
    if parsed.x.len() != objective_coefficients.len()
        || var_basis.len() != objective_coefficients.len()
        || row_basis.len() != le_rows.len() + eq_rows.len()
    {
        return None;
    }

    let active_le = row_basis
        .iter()
        .take(le_rows.len())
        .enumerate()
        .filter_map(|(index, status)| qsopt_ex_row_basis_is_active(status).then_some(index))
        .collect::<Vec<_>>();
    let basic_vars = var_basis
        .iter()
        .enumerate()
        .filter_map(|(index, status)| (status == "basic").then_some(index))
        .collect::<Vec<_>>();
    let unknown_count = active_le.len() + eq_rows.len();
    if basic_vars.len() != unknown_count {
        return None;
    }
    if unknown_count == 0 {
        if objective_coefficients.iter().any(|value| value.abs() > TOL) {
            return None;
        }
        let dual_ub = vec![0.0; le_rows.len()];
        let dual_eq = vec![0.0; eq_rows.len()];
        let reduced_costs = reduced_costs_from_row_duals(
            objective_coefficients,
            le_rows,
            &dual_ub,
            eq_rows,
            &dual_eq,
        );
        return Some((dual_ub, dual_eq, reduced_costs));
    }

    let mut system: Matrix = Vec::with_capacity(unknown_count);
    let mut rhs = Vec::with_capacity(unknown_count);
    for &column in &basic_vars {
        let mut row = Vec::with_capacity(unknown_count);
        for &row_index in &active_le {
            row.push(le_rows[row_index].get(column).copied().unwrap_or(0.0));
        }
        for eq_row in eq_rows {
            row.push(eq_row.get(column).copied().unwrap_or(0.0));
        }
        system.push(row);
        rhs.push(objective_coefficients[column]);
    }
    let solution = LinearSystem::new(&system, &rhs, 1.0e-10).try_solve()?;
    if solution.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let mut dual_ub = vec![0.0; le_rows.len()];
    for (offset, &row_index) in active_le.iter().enumerate() {
        let value = clean_certificate_value(solution[offset]);
        if value < -TOL {
            return None;
        }
        dual_ub[row_index] = value.max(0.0);
    }
    let dual_eq = solution[active_le.len()..]
        .iter()
        .map(|value| clean_certificate_value(*value))
        .collect::<Vec<_>>();
    let reduced_costs =
        reduced_costs_from_row_duals(objective_coefficients, le_rows, &dual_ub, eq_rows, &dual_eq);
    if basic_vars
        .iter()
        .any(|&index| reduced_costs[index].abs() > TOL)
    {
        return None;
    }
    Some((dual_ub, dual_eq, reduced_costs))
}

fn qsopt_ex_row_basis_is_active(status: &str) -> bool {
    matches!(status, "at_upper" | "at_lower" | "fixed")
}

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> Option<f64> {
    if lhs.len() != rhs.len() {
        return None;
    }
    Some(
        lhs.iter()
            .zip(rhs)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max),
    )
}

fn parse_native_lp_solve_solution_text(
    text: &str,
    variable_count: usize,
) -> ParsedNativeLpSolveSolution {
    let mut parsed = ParsedNativeLpSolveSolution {
        status: "unknown".to_string(),
        x: vec![0.0; variable_count],
        objective: None,
    };
    let lowered = text.to_ascii_lowercase();
    if lowered.contains("infeasible") || lowered.contains("no feasible") {
        parsed.status = "infeasible".to_string();
    } else if lowered.contains("unbounded") {
        parsed.status = "unbounded".to_string();
    } else if lowered.contains("value of objective function")
        || lowered.contains("actual values of the variables")
    {
        parsed.status = "optimal".to_string();
    }

    let mut in_variable_values = false;
    for line in text.lines() {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("value of objective function") {
            parsed.objective = stripped
                .split_once(':')
                .and_then(|(_, value)| parse_f64_token(value.trim()));
        }
        if lowered.starts_with("actual values of the variables") {
            in_variable_values = true;
            continue;
        }
        if lowered.starts_with("actual values of the constraints")
            || lowered.starts_with("objective function limits")
            || lowered.starts_with("dual values")
            || lowered.starts_with("cpu time")
        {
            in_variable_values = false;
        }
        if !in_variable_values {
            continue;
        }
        if let Some((index, value)) = parse_named_variable_value_line(stripped, variable_count) {
            parsed.x[index] = value;
        }
    }
    parsed
}

fn parse_native_lp_solve_lp_certificate_fields(
    text: &str,
    basis_path: &Path,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
) -> NativeLpSolveCertificateFields {
    let mut row_duals = vec![None::<f64>; le_count + eq_count];
    let mut reduced_costs = vec![None::<f64>; variable_count];
    let mut in_dual_table = false;

    for line in text.lines() {
        let stripped = line.trim();
        if stripped.is_empty() {
            continue;
        }
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("dual values with") {
            in_dual_table = true;
            continue;
        }
        if in_dual_table && lowered.starts_with("dual value") {
            continue;
        }

        let parts = stripped.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        if let Some(index) = basis_row_index(parts[0], le_count, eq_count) {
            if let Some(value) = parse_f64_token(parts[1]) {
                row_duals[index] = Some(clean_certificate_value(value));
            }
        } else if let Some(index) =
            highs_variable_index(parts[0]).filter(|index| *index < variable_count)
        {
            let numeric_tokens = parts
                .iter()
                .skip(1)
                .filter_map(|token| parse_f64_token(token))
                .collect::<Vec<_>>();
            let value = if in_dual_table {
                numeric_tokens.first().copied()
            } else {
                numeric_tokens.get(1).copied()
            };
            if let Some(value) = value {
                reduced_costs[index] = Some(clean_certificate_value(value));
            }
        }
    }

    let (var_basis, row_basis) = if basis_path.exists() {
        match fs::read_to_string(basis_path) {
            Ok(text) => parse_native_lp_solve_basis_text(&text, variable_count, le_count, eq_count),
            Err(_) => (None, None),
        }
    } else {
        (None, None)
    };

    NativeLpSolveCertificateFields {
        reduced_costs: all_some_f64(&reduced_costs),
        dual_ub: all_some_f64(&row_duals[..le_count]),
        dual_eq: all_some_f64(&row_duals[le_count..]),
        var_basis,
        row_basis,
    }
}

fn parse_native_lp_solve_basis_text(
    text: &str,
    variable_count: usize,
    le_count: usize,
    eq_count: usize,
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    let mut var_basis = vec![None::<String>; variable_count];
    let mut row_basis = vec![Some("basic".to_string()); le_count];
    row_basis.extend((0..eq_count).map(|_| Some("fixed".to_string())));

    for line in text.lines() {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.is_empty() || matches!(parts[0], "NAME" | "ENDATA") {
            continue;
        }
        let code = parts[0].to_ascii_uppercase();
        if parts.len() >= 2 && parts[1].starts_with('x') {
            if let Some(index) =
                highs_variable_index(parts[1]).filter(|index| *index < variable_count)
            {
                let status = match code.as_str() {
                    "BS" | "XL" | "XU" => Some("basic"),
                    "LL" => Some("at_lower"),
                    "UL" => Some("at_upper"),
                    "FX" => Some("fixed"),
                    "FR" => Some("free"),
                    _ => None,
                };
                if let Some(status) = status {
                    var_basis[index] = Some(status.to_string());
                }
            }
        }
        if matches!(code.as_str(), "XL" | "XU") && parts.len() >= 3 {
            if let Some(index) = basis_row_index(parts[2], le_count, eq_count) {
                if parts[2].starts_with('c') {
                    let status = if code == "XL" { "at_upper" } else { "at_lower" };
                    row_basis[index] = Some(status.to_string());
                }
            }
        }
    }

    (all_some_string(&var_basis), all_some_string(&row_basis))
}

fn parse_native_cplex_solution_file(
    path: &Path,
    variable_count: usize,
) -> Result<ParsedNativeNamedSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read CPLEX solution file '{}': {err}",
            path.display()
        )
    })?;
    Ok(parse_native_cplex_solution_text(&text, variable_count))
}

fn parse_native_cplex_solution_text(
    text: &str,
    variable_count: usize,
) -> ParsedNativeNamedSolution {
    let mut x = vec![0.0; variable_count];
    let mut status = "optimal".to_string();
    let mut saw_cplex_xml = false;

    for tag in xml_start_tags(text, "header") {
        saw_cplex_xml = true;
        if let Some(value) = xml_attribute_value(&tag, "solutionStatusString")
            .or_else(|| xml_attribute_value(&tag, "solutionStatus"))
            .or_else(|| xml_attribute_value(&tag, "solutionStatusValue"))
        {
            status = cplex_solution_status(&value);
        }
    }

    for tag in xml_start_tags(text, "variable") {
        saw_cplex_xml = true;
        let Some(name) = xml_attribute_value(&tag, "name") else {
            continue;
        };
        let Some(index) = highs_variable_index(&name).filter(|index| *index < variable_count)
        else {
            continue;
        };
        let Some(value) =
            xml_attribute_value(&tag, "value").and_then(|value| parse_f64_token(&value))
        else {
            continue;
        };
        x[index] = value;
    }

    if saw_cplex_xml {
        ParsedNativeNamedSolution {
            status,
            x,
            ..Default::default()
        }
    } else {
        parse_native_named_solution_text(text, variable_count, "optimal")
    }
}

fn cplex_solution_status(value: &str) -> String {
    let stripped = value.trim();
    let lower = stripped.to_ascii_lowercase();
    if lower.chars().any(|ch| ch.is_ascii_alphabetic()) {
        return lower;
    }
    match stripped {
        "1" | "101" | "102" => "optimal".to_string(),
        "2" | "118" | "119" => "unbounded".to_string(),
        "3" | "103" => "infeasible".to_string(),
        "104" | "105" | "106" | "107" | "108" | "109" => "feasible".to_string(),
        _ => lower,
    }
}

fn xml_start_tags<'a>(text: &'a str, tag_name: &str) -> Vec<&'a str> {
    let needle = format!("<{tag_name}");
    let mut tags = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let Some(offset) = text[cursor..].find(&needle) else {
            break;
        };
        let start = cursor + offset;
        let name_end = start + needle.len();
        if text
            .as_bytes()
            .get(name_end)
            .is_some_and(|byte| !matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | b'/' | b'>'))
        {
            cursor = name_end;
            continue;
        }
        let Some(end_offset) = text[start..].find('>') else {
            break;
        };
        let end = start + end_offset + 1;
        tags.push(&text[start..end]);
        cursor = end;
    }
    tags
}

fn xml_attribute_value(tag: &str, name: &str) -> Option<String> {
    let bytes = tag.as_bytes();
    let mut cursor = 0;
    while cursor < tag.len() {
        let offset = tag[cursor..].find(name)?;
        let start = cursor + offset;
        let end = start + name.len();
        if start > 0 && is_ascii_word_byte(bytes[start - 1]) {
            cursor = end;
            continue;
        }
        if end < bytes.len() && is_ascii_word_byte(bytes[end]) {
            cursor = end;
            continue;
        }
        let mut pos = end;
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if bytes.get(pos) != Some(&b'=') {
            cursor = end;
            continue;
        }
        pos += 1;
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let quote = *bytes.get(pos)?;
        if !matches!(quote, b'\'' | b'"') {
            cursor = pos;
            continue;
        }
        pos += 1;
        let value_start = pos;
        while pos < bytes.len() && bytes[pos] != quote {
            pos += 1;
        }
        if pos >= bytes.len() {
            return None;
        }
        return Some(xml_decode_attribute(&tag[value_start..pos]));
    }
    None
}

fn xml_decode_attribute(value: &str) -> String {
    if !value.contains('&') {
        return value.to_string();
    }
    value
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn parse_native_xpress_solution_file(
    path: &Path,
    variable_count: usize,
) -> Result<ParsedNativeNamedSolution, String> {
    let data_path = if path.exists() {
        path.to_path_buf()
    } else {
        let companion = native_xpress_solution_data_path(path);
        if companion.exists() {
            companion
        } else {
            path.to_path_buf()
        }
    };
    let text = fs::read_to_string(&data_path).map_err(|err| {
        format!(
            "failed to read Xpress solution file '{}': {err}",
            data_path.display()
        )
    })?;
    let header_path = native_xpress_header_path(path);
    let header = fs::read_to_string(&header_path).ok();
    Ok(parse_native_xpress_solution_text(
        &text,
        header.as_deref(),
        variable_count,
    ))
}

fn parse_native_xpress_solution_text(
    text: &str,
    header: Option<&str>,
    variable_count: usize,
) -> ParsedNativeNamedSolution {
    let mut x = vec![0.0; variable_count];
    let mut status = "optimal".to_string();
    if let Some(header) = header {
        let lower = header.to_ascii_lowercase();
        if lower.contains("infeas") {
            status = "infeasible".to_string();
        } else if lower.contains("unbounded") {
            status = "unbounded".to_string();
        } else if lower.contains("optimal") {
            status = "optimal".to_string();
        }
    }

    for line in text.lines() {
        let fields = split_xpress_solution_line(line);
        if fields.len() < 2 {
            continue;
        }
        for (name_pos, name) in fields.iter().enumerate() {
            let clean_name = name.trim().trim_matches('"').trim_matches('\'');
            let Some(index) =
                highs_variable_index(clean_name).filter(|index| *index < variable_count)
            else {
                continue;
            };
            if let Some(value) = fields[name_pos + 1..]
                .iter()
                .map(|field| field.trim().trim_matches('"').trim_matches('\''))
                .find_map(parse_f64_token)
            {
                x[index] = value;
            }
            break;
        }
    }
    ParsedNativeNamedSolution {
        status,
        x,
        ..Default::default()
    }
}

fn split_xpress_solution_line(line: &str) -> Vec<String> {
    let stripped = line.trim();
    if stripped.is_empty() || stripped.starts_with('#') {
        return Vec::new();
    }
    if stripped.contains(';') {
        return split_delimited_solution_fields(stripped, ';');
    }
    if stripped.contains(',') {
        return split_delimited_solution_fields(stripped, ',');
    }
    stripped.split_whitespace().map(|s| s.to_string()).collect()
}

fn split_delimited_solution_fields(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quote = None;
    for ch in line.chars() {
        if let Some(active_quote) = quote {
            if ch == active_quote {
                quote = None;
            } else {
                field.push(ch);
            }
            continue;
        }
        if matches!(ch, '"' | '\'') {
            quote = Some(ch);
            continue;
        }
        if ch == delimiter {
            let trimmed = field.trim();
            if !trimmed.is_empty() {
                fields.push(trimmed.to_string());
            }
            field.clear();
            continue;
        }
        field.push(ch);
    }
    let trimmed = field.trim();
    if !trimmed.is_empty() {
        fields.push(trimmed.to_string());
    }
    fields
}

fn parse_native_lindo_solution_file(
    path: &Path,
    variable_count: usize,
) -> Result<ParsedNativeNamedSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read LINDO solution file '{}': {err}",
            path.display()
        )
    })?;
    Ok(parse_native_lindo_solution_text(&text, variable_count))
}

fn parse_native_lindo_solution_text(
    text: &str,
    variable_count: usize,
) -> ParsedNativeNamedSolution {
    let mut x = vec![0.0; variable_count];
    let lower = text.to_ascii_lowercase();
    let status = if lower.contains("infeasible") || lower.contains("no feasible") {
        "infeasible"
    } else if lower.contains("unbounded") {
        "unbounded"
    } else if lower.contains("optimal") || lower.contains("objective") {
        "optimal"
    } else {
        "unknown"
    }
    .to_string();

    for line in text.lines() {
        if let Some((index, value)) = parse_named_variable_value_line(line.trim(), variable_count) {
            x[index] = value;
        }
    }
    ParsedNativeNamedSolution {
        status,
        x,
        ..Default::default()
    }
}

fn parse_native_named_solution_file(
    path: &Path,
    variable_count: usize,
    default_status: &str,
) -> Result<ParsedNativeNamedSolution, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read named solution file '{}': {err}",
            path.display()
        )
    })?;
    Ok(parse_native_named_solution_text(
        &text,
        variable_count,
        default_status,
    ))
}

fn parse_native_named_solution_text(
    text: &str,
    variable_count: usize,
    default_status: &str,
) -> ParsedNativeNamedSolution {
    let mut x = vec![0.0; variable_count];
    let mut status = default_status.to_string();
    for line in text.lines() {
        let stripped = line.trim();
        let lower = stripped.to_ascii_lowercase();
        if lower.contains("objective") && lower.contains("value") {
            status = default_status.to_string();
            continue;
        }
        if let Some((index, value)) = parse_named_variable_value_line(stripped, variable_count) {
            x[index] = value;
        }
    }
    ParsedNativeNamedSolution {
        status,
        x,
        ..Default::default()
    }
}

fn parse_named_variable_value_line(line: &str, variable_count: usize) -> Option<(usize, f64)> {
    parse_prefixed_value_line(line, 'x', variable_count)
}

fn parse_prefixed_value_line(line: &str, prefix: char, count: usize) -> Option<(usize, f64)> {
    let bytes = line.as_bytes();
    let prefix = prefix.to_ascii_lowercase() as u8;
    for start in 0..bytes.len() {
        if bytes[start].to_ascii_lowercase() != prefix {
            continue;
        }
        if start > 0 && is_ascii_word_byte(bytes[start - 1]) {
            continue;
        }
        let mut end = start + 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end == start + 1 {
            continue;
        }
        if end < bytes.len() && is_ascii_word_byte(bytes[end]) {
            continue;
        }
        let index = line[start + 1..end].parse::<usize>().ok()?;
        if index >= count {
            continue;
        }
        for token in
            line[end..].split(|ch: char| ch.is_whitespace() || matches!(ch, ',' | ';' | ':' | '='))
        {
            if let Some(value) = parse_f64_token(token) {
                return Some((index, value));
            }
        }
        let mut before_value = None;
        for token in line[..start]
            .split(|ch: char| ch.is_whitespace() || matches!(ch, ',' | ';' | ':' | '='))
        {
            if let Some(value) = parse_f64_token(token) {
                before_value = Some(value);
            }
        }
        if let Some(value) = before_value {
            return Some((index, value));
        }
    }
    None
}

fn is_ascii_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn highs_variable_index(name: &str) -> Option<usize> {
    name.strip_prefix('x')?.parse::<usize>().ok()
}

fn signed_usize_token(token: &str) -> Option<usize> {
    token
        .trim_start_matches('-')
        .chars()
        .all(|ch| ch.is_ascii_digit())
        .then(|| token.trim_start_matches('-').parse::<usize>().ok())
        .flatten()
}

fn highs_basis_status(token: &str) -> Option<&'static str> {
    match token.trim().to_ascii_lowercase().as_str() {
        "0" => Some("at_lower"),
        "1" => Some("basic"),
        "2" => Some("at_upper"),
        "3" => Some("zero"),
        "4" => Some("nonbasic"),
        "b" | "bs" => Some("basic"),
        "l" | "nl" => Some("at_lower"),
        "u" | "nu" => Some("at_upper"),
        "f" | "nf" => Some("free"),
        "s" | "ns" => Some("superbasic"),
        _ => None,
    }
}

fn all_some_f64(values: &[Option<f64>]) -> Option<Vec<f64>> {
    values.iter().copied().collect()
}

fn all_some_string(values: &[Option<String>]) -> Option<Vec<String>> {
    values.iter().cloned().collect()
}

fn row_values(
    rows: &std::collections::BTreeMap<String, f64>,
    prefix: &str,
    count: usize,
) -> Option<Vec<f64>> {
    (0..count)
        .map(|idx| rows.get(&format!("{prefix}{idx}")).copied())
        .collect()
}

fn classify_highs_status(status: &str, stdout: &str, stderr: &str) -> ExternalLinearCliStatus {
    classify_native_linear_status(status, stdout, stderr)
}

fn classify_native_linear_status(
    status: &str,
    stdout: &str,
    stderr: &str,
) -> ExternalLinearCliStatus {
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

fn parse_highs_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        for marker in ["Running HiGHS ", "HiGHS version "] {
            if let Some(rest) = line.split_once(marker).map(|(_, rest)| rest) {
                let version = rest
                    .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')')
                    .next()
                    .unwrap_or("")
                    .trim();
                if version.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
                    return Some(format!("HiGHS {version}"));
                }
            }
        }
    }
    None
}

fn probe_highs_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_highs_solver_version(&text)
}

fn parse_glpk_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some(rest) = line.split_once("GLPSOL--GLPK LP/MIP Solver ") {
            let version = rest
                .1
                .split(|ch: char| ch.is_whitespace() || ch == ',')
                .next()
                .unwrap_or("")
                .trim();
            if version.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
                return Some(format!("GLPK {version}"));
            }
        }
    }
    None
}

fn probe_glpk_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_glpk_solver_version(&text)
}

fn parse_scip_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some((_, rest)) = line.split_once("SCIP version ") {
            let version = rest
                .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')' || ch == '[')
                .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                .unwrap_or("")
                .trim();
            if !version.is_empty() {
                return Some(format!("SCIP {version}"));
            }
        }
    }
    None
}

fn probe_scip_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_scip_solver_version(&text)
}

fn parse_cbc_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some((_, rest)) = line.split_once("Version:") {
            let version = rest
                .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')')
                .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                .unwrap_or("")
                .trim();
            if !version.is_empty() {
                return Some(format!("CBC {version}"));
            }
        }
    }
    None
}

fn probe_cbc_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("-version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_cbc_solver_version(&text)
}

fn parse_clp_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        let rest = if let Some((_, rest)) = line.split_once("Version:") {
            rest
        } else if let Some((_, rest)) = line.split_once("Coin LP version ") {
            rest
        } else {
            continue;
        };
        let version = rest
            .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')')
            .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
            .unwrap_or("")
            .trim();
        if !version.is_empty() {
            return Some(format!("CLP {version}"));
        }
    }
    None
}

fn probe_clp_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("-version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_clp_solver_version(&text)
}

fn parse_soplex_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some((_, rest)) = line.split_once("SoPlex version ") {
            let version = rest
                .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')' || ch == '[')
                .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                .unwrap_or("")
                .trim();
            if !version.is_empty() {
                return Some(format!("SoPlex {version}"));
            }
        }
    }
    None
}

fn probe_soplex_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("-v0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_soplex_solver_version(&text)
}

fn parse_qsopt_ex_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        for marker in ["Using QSopt_ex ", "QSopt_ex "] {
            if let Some((_, rest)) = line.split_once(marker) {
                let version = rest
                    .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')')
                    .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                    .unwrap_or("")
                    .trim();
                if !version.is_empty() {
                    return Some(format!("QSopt_ex {version}"));
                }
            }
        }
    }
    None
}

fn probe_qsopt_ex_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_qsopt_ex_solver_version(&text)
}

fn parse_lp_solve_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some((_, rest)) = line.split_once("lp_solve version ") {
            let version = rest
                .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ':')
                .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                .unwrap_or("")
                .trim();
            if !version.is_empty() {
                return Some(format!("lp_solve {version}"));
            }
        }
    }
    None
}

fn probe_lp_solve_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("-h")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_lp_solve_solver_version(&text)
}

fn parse_gurobi_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        for marker in ["Gurobi Optimizer version ", "Gurobi Optimizer "] {
            if let Some((_, rest)) = line.split_once(marker) {
                let version = rest
                    .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')')
                    .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                    .unwrap_or("")
                    .trim();
                if !version.is_empty() {
                    return Some(format!("Gurobi {version}"));
                }
            }
        }
    }
    None
}

fn probe_gurobi_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_gurobi_solver_version(&text)
}

fn parse_cplex_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        for marker in [
            "IBM ILOG CPLEX Interactive Optimizer ",
            "IBM ILOG CPLEX Optimizer ",
            "CPLEX Interactive Optimizer ",
            "CPLEX Optimizer ",
            "CPLEX ",
        ] {
            if let Some((_, rest)) = line.split_once(marker) {
                let version = rest
                    .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')')
                    .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                    .unwrap_or("")
                    .trim();
                if !version.is_empty() {
                    return Some(format!("CPLEX {version}"));
                }
            }
        }
    }
    None
}

fn probe_cplex_solver_version(command_path: &Path) -> Option<String> {
    let output = Command::new(command_path)
        .arg("-c")
        .arg("quit")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_cplex_solver_version(&text)
}

fn parse_xpress_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        for marker in ["Xpress Optimizer ", "Xpress "] {
            if let Some((_, rest)) = line.split_once(marker) {
                let version = rest
                    .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')')
                    .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                    .unwrap_or("")
                    .trim();
                if !version.is_empty() {
                    return Some(format!("Xpress {version}"));
                }
            }
        }
    }
    None
}

fn parse_lindo_solver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        let lower_line = line.to_ascii_lowercase();
        for marker in [
            "lindo api version ",
            "lindo api ",
            "lindo optimizer ",
            "lindo ",
        ] {
            if let Some(start) = lower_line.find(marker) {
                let rest = &line[start + marker.len()..];
                let version = rest
                    .split(|ch: char| ch.is_whitespace() || ch == ',' || ch == ')')
                    .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
                    .unwrap_or("")
                    .trim();
                if !version.is_empty() {
                    return Some(format!("LINDO {version}"));
                }
            }
        }
    }
    None
}

fn parse_highs_lp_iterations(stdout: &str, stderr: &str) -> Option<u64> {
    for line in format!("{stdout}\n{stderr}").lines() {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("simplex") && lowered.contains("iterations") {
            if let Some(value) = first_float_after_colon(stripped) {
                if value >= 0.0 && value.is_finite() {
                    return Some(value.round() as u64);
                }
            }
        }
    }
    None
}

fn parse_glpk_lp_iterations(stdout: &str, stderr: &str) -> Option<u64> {
    for line in format!("{stdout}\n{stderr}").lines() {
        let stripped = line.trim();
        let stripped = stripped.strip_prefix('*').unwrap_or(stripped).trim();
        let Some((prefix, rest)) = stripped.split_once(':') else {
            continue;
        };
        if prefix.trim().chars().all(|ch| ch.is_ascii_digit()) && rest.trim().starts_with("obj") {
            if let Ok(iterations) = prefix.trim().parse::<u64>() {
                return Some(iterations);
            }
        }
    }
    None
}

fn parse_cbc_lp_iterations(stdout: &str, stderr: &str) -> Option<u64> {
    for line in format!("{stdout}\n{stderr}").lines() {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if !lowered.contains("iterations") {
            continue;
        }
        let before_iterations = lowered.split("iterations").next().unwrap_or("");
        for token in before_iterations
            .split(|ch: char| !ch.is_ascii_digit())
            .rev()
        {
            if token.is_empty() {
                continue;
            }
            if let Ok(iterations) = token.parse::<u64>() {
                return Some(iterations);
            }
        }
    }
    None
}

fn parse_soplex_lp_iterations(stdout: &str, stderr: &str) -> Option<u64> {
    for line in format!("{stdout}\n{stderr}").lines() {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("iterations") {
            if let Some(value) = first_float_after_colon(stripped) {
                if value >= 0.0 && value.is_finite() {
                    return Some(value.round() as u64);
                }
            }
        }
    }
    None
}

fn parse_highs_mip_quality(
    kind: ExternalLinearCliKind,
    objective: f64,
    stdout: &str,
    stderr: &str,
) -> HighsMipQuality {
    if kind != ExternalLinearCliKind::Mip {
        return HighsMipQuality::default();
    }
    let mut quality = HighsMipQuality::default();
    for line in format!("{stdout}\n{stderr}").lines() {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("dual bound") {
            quality.best_bound = first_float_after_colon(stripped);
        } else if lowered.starts_with("gap") {
            quality.mip_gap = first_float(stripped).map(|gap| {
                if stripped.contains('%') {
                    gap / 100.0
                } else {
                    gap
                }
            });
        } else if lowered.starts_with("nodes") {
            quality.nodes_explored = first_float_after_colon(stripped)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .map(|value| value.round() as u64);
        }
    }
    if let Some(best_bound) = quality.best_bound.filter(|value| value.is_finite()) {
        quality.absolute_gap = Some((best_bound - objective).abs().max(0.0));
        if quality.mip_gap.is_none() {
            quality.mip_gap = Some((best_bound - objective).abs() / objective.abs().max(1.0));
        }
    }
    quality.mip_gap = quality
        .mip_gap
        .filter(|value| value.is_finite())
        .map(|value| value.max(0.0));
    quality
}

fn parse_highs_mip_start_feedback(
    kind: ExternalLinearCliKind,
    mip_start: Option<&[f64]>,
    start_objective: Option<f64>,
    stdout: &str,
    stderr: &str,
) -> (Option<bool>, Option<f64>) {
    if kind != ExternalLinearCliKind::Mip || mip_start.is_none() {
        return (None, None);
    }
    let text = format!("{stdout}\n{stderr}");
    let lowered = text.to_ascii_lowercase();
    let infeasibilities = mip_start_infeasibility_values(&text);
    let accepted = lowered.contains("assessing feasibility of mip")
        && infeasibilities.len() >= 3
        && infeasibilities
            .iter()
            .take(3)
            .all(|value| value.abs() <= 1.0e-9);
    (
        Some(accepted),
        start_objective.filter(|value| value.is_finite()),
    )
}

fn mip_start_infeasibility_values(text: &str) -> Vec<f64> {
    text.lines()
        .filter(|line| line.to_ascii_lowercase().contains("infeasibilities"))
        .filter_map(first_float)
        .collect()
}

fn parse_glpk_mip_quality(
    kind: ExternalLinearCliKind,
    status: ExternalLinearCliStatus,
    objective: f64,
    stdout: &str,
    stderr: &str,
) -> HighsMipQuality {
    if kind != ExternalLinearCliKind::Mip {
        return HighsMipQuality::default();
    }
    let mut quality = HighsMipQuality::default();
    for line in format!("{stdout}\n{stderr}").lines() {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if lowered.contains("integer optimal solution found by mip preprocessor") {
            quality.nodes_explored = Some(0);
        }
        if lowered.contains("mip =") {
            if let Some(gap) = percent_gap_from_line(stripped) {
                quality.mip_gap = Some(gap);
            }
        }
    }
    fill_optimal_mip_quality(status, objective, &mut quality);
    quality
}

fn parse_lp_solve_mip_quality(
    kind: ExternalLinearCliKind,
    status: ExternalLinearCliStatus,
    objective: f64,
    stdout: &str,
    stderr: &str,
    suppress_nodes: bool,
) -> HighsMipQuality {
    if kind != ExternalLinearCliKind::Mip {
        return HighsMipQuality::default();
    }
    let mut quality = HighsMipQuality::default();
    for line in format!("{stdout}\n{stderr}").lines() {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if !lowered.contains("solution") || !lowered.contains("nodes") {
            continue;
        }

        let parts = stripped.split_whitespace().collect::<Vec<_>>();
        for (idx, token) in parts.iter().enumerate() {
            if token.trim_matches(|ch: char| !ch.is_ascii_alphabetic()) == "nodes" && idx > 0 {
                if !suppress_nodes {
                    quality.nodes_explored = parse_f64_token(parts[idx - 1])
                        .filter(|value| value.is_finite() && *value >= 0.0)
                        .map(|value| value.round() as u64);
                }
                break;
            }
        }
        if let Some(gap_start) = lowered.find("gap") {
            quality.mip_gap = first_float(&stripped[gap_start..]).map(|gap| {
                if stripped.contains('%') {
                    gap / 100.0
                } else {
                    gap
                }
            });
        }
    }

    let exact_optimal = quality.mip_gap.map_or(true, |gap| gap.abs() <= 1.0e-12);
    if exact_optimal {
        fill_optimal_mip_quality(status, objective, &mut quality);
    }
    quality.mip_gap = quality
        .mip_gap
        .filter(|value| value.is_finite())
        .map(|value| value.max(0.0));
    quality
}

fn parse_cbc_mip_quality(
    kind: ExternalLinearCliKind,
    status: ExternalLinearCliStatus,
    objective: f64,
    stdout: &str,
    stderr: &str,
) -> HighsMipQuality {
    if kind != ExternalLinearCliKind::Mip {
        return HighsMipQuality::default();
    }
    let mut quality = HighsMipQuality::default();
    for line in format!("{stdout}\n{stderr}").lines() {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("enumerated nodes") {
            quality.nodes_explored = first_float_after_colon(stripped)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .map(|value| value.round() as u64);
        } else if lowered.starts_with("gap") {
            quality.mip_gap = first_float_after_colon(stripped).map(|gap| {
                if stripped.contains('%') {
                    gap / 100.0
                } else {
                    gap
                }
            });
        } else if lowered.starts_with("lower bound")
            || lowered.starts_with("best possible")
            || lowered.starts_with("best bound")
        {
            quality.best_bound =
                first_float_after_colon(stripped).or_else(|| first_float(stripped));
        }
    }
    fill_optimal_mip_quality(status, objective, &mut quality);
    quality
}

fn parse_cbc_mip_start_feedback(
    kind: ExternalLinearCliKind,
    mip_start: Option<&[f64]>,
    start_objective: Option<f64>,
    stdout: &str,
    stderr: &str,
) -> (Option<bool>, Option<f64>) {
    if kind != ExternalLinearCliKind::Mip || mip_start.is_none() {
        return (None, None);
    }
    let lowered = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    let rejected = lowered.contains("mipstart solution is not valid")
        || lowered.contains("mipstart values could not be used")
        || lowered.contains("mipstart file not valid");
    let accepted = !rejected
        && (lowered.contains("mipstart values read")
            || lowered.contains("mipstart provided solution")
            || lowered.contains("integer solution"));
    (
        Some(accepted),
        start_objective.filter(|value| value.is_finite()),
    )
}

fn parse_cbc_branch_priority_feedback(
    kind: ExternalLinearCliKind,
    active_count: usize,
    stdout: &str,
    stderr: &str,
) -> (Option<bool>, Option<u64>) {
    if kind != ExternalLinearCliKind::Mip || active_count == 0 {
        return (None, None);
    }
    let accepted = format!("{stdout}\n{stderr}")
        .to_ascii_lowercase()
        .contains("priorityin");
    (Some(accepted), accepted.then_some(active_count as u64))
}

fn parse_scip_mip_quality(
    kind: ExternalLinearCliKind,
    objective: f64,
    stdout: &str,
    stderr: &str,
) -> HighsMipQuality {
    if kind != ExternalLinearCliKind::Mip {
        return HighsMipQuality::default();
    }
    let mut quality = HighsMipQuality::default();
    for line in format!("{stdout}\n{stderr}").lines() {
        let stripped = line.trim();
        let lowered = stripped.to_ascii_lowercase();
        if lowered.starts_with("solving nodes") {
            quality.nodes_explored = first_float_after_colon(stripped)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .map(|value| value.round() as u64);
        } else if lowered.starts_with("dual bound") {
            quality.best_bound = first_float_after_colon(stripped);
        } else if lowered.starts_with("gap") {
            quality.mip_gap = first_float_after_colon(stripped).map(|gap| {
                if stripped.contains('%') {
                    gap / 100.0
                } else {
                    gap
                }
            });
        }
    }
    if let Some(best_bound) = quality.best_bound.filter(|value| value.is_finite()) {
        quality.absolute_gap = Some((best_bound - objective).abs().max(0.0));
        if quality.mip_gap.is_none() {
            quality.mip_gap = Some((best_bound - objective).abs() / objective.abs().max(1.0));
        }
    }
    quality.mip_gap = quality
        .mip_gap
        .filter(|value| value.is_finite())
        .map(|value| value.max(0.0));
    quality
}

fn parse_scip_mip_start_feedback(
    kind: ExternalLinearCliKind,
    mip_start: Option<&[f64]>,
    start_objective: Option<f64>,
    stdout: &str,
    stderr: &str,
) -> (Option<bool>, Option<f64>) {
    if kind != ExternalLinearCliKind::Mip || mip_start.is_none() {
        return (None, None);
    }
    let lowered = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    let accepted =
        lowered.contains("accepted as candidate") || lowered.contains("solution candidate storage");
    (
        Some(accepted),
        start_objective.filter(|value| value.is_finite()),
    )
}

fn parse_scip_branch_priority_feedback(
    kind: ExternalLinearCliKind,
    active_count: usize,
    stdout: &str,
    stderr: &str,
) -> (Option<bool>, Option<u64>) {
    if kind != ExternalLinearCliKind::Mip || active_count == 0 {
        return (None, None);
    }
    let accepted = format!("{stdout}\n{stderr}")
        .to_ascii_lowercase()
        .contains("branching priority of variable");
    (Some(accepted), accepted.then_some(active_count as u64))
}

fn fill_optimal_mip_quality(
    status: ExternalLinearCliStatus,
    objective: f64,
    quality: &mut HighsMipQuality,
) {
    if status != ExternalLinearCliStatus::Optimal || !objective.is_finite() {
        return;
    }
    if quality.best_bound.is_none() {
        quality.best_bound = Some(objective);
    }
    if quality.mip_gap.is_none() {
        quality.mip_gap = Some(0.0);
    }
    if quality.absolute_gap.is_none() {
        quality.absolute_gap = Some(0.0);
    }
}

fn percent_gap_from_line(line: &str) -> Option<f64> {
    let mut saw_percent = false;
    for token in line.split_whitespace().rev() {
        let token = token.trim_matches(|ch: char| matches!(ch, ',' | '(' | ')' | '[' | ']'));
        if token.contains('%') {
            saw_percent = true;
            if let Some(value) = parse_f64_token(token) {
                return Some((value / 100.0).max(0.0));
            }
        } else if saw_percent {
            if let Some(value) = parse_f64_token(token) {
                return Some((value / 100.0).max(0.0));
            }
        }
    }
    None
}

fn first_float_after_colon(line: &str) -> Option<f64> {
    let text = line.split_once(':').map_or(line, |(_, rest)| rest);
    first_float(text)
}

fn first_float(text: &str) -> Option<f64> {
    text.split(|ch: char| ch.is_whitespace() || ch == ',' || ch == '(' || ch == ')')
        .find_map(parse_f64_token)
}

fn parse_f64_token(token: &str) -> Option<f64> {
    let token = token
        .trim()
        .trim_matches(|ch: char| matches!(ch, '%' | ',' | '(' | ')' | '[' | ']' | '*'));
    if let Ok(value) = token.parse::<f64>() {
        return Some(value);
    }

    let mut rational_parts = token.split('/');
    let numerator = rational_parts.next()?;
    let denominator = rational_parts.next()?;
    if rational_parts.next().is_some() {
        return None;
    }
    let numerator = numerator.parse::<f64>().ok()?;
    let denominator = denominator.parse::<f64>().ok()?;
    if denominator == 0.0 {
        return None;
    }
    let value = numerator / denominator;
    value.is_finite().then_some(value)
}

fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn normalized_highs_random_seed(random_seed: Option<u64>) -> Option<u64> {
    random_seed.filter(|seed| *seed <= i32::MAX as u64)
}

fn native_highs_message(status: &str, stdout: &str, stderr: &str) -> String {
    native_solver_message(status, stdout, stderr)
}

fn native_solver_message(status: &str, stdout: &str, stderr: &str) -> String {
    if !status.trim().is_empty() {
        return status.to_string();
    }
    let stderr = stderr.trim();
    if !stderr.is_empty() {
        return stderr.to_string();
    }
    let stdout = stdout.trim();
    if !stdout.is_empty() {
        return stdout.to_string();
    }
    "solver produced no diagnostic output".to_string()
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
        if binary_set.contains(&i) && lp_variable_is_referenced(i, c, le_rows, eq_rows) {
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

fn lp_variable_is_referenced(
    index: usize,
    c: &[f64],
    le_rows: &[Vec<f64>],
    eq_rows: &[Vec<f64>],
) -> bool {
    c.get(index).is_some_and(|coef| coef.abs() > 1.0e-12)
        || le_rows
            .iter()
            .chain(eq_rows)
            .any(|row| row.get(index).is_some_and(|coef| coef.abs() > 1.0e-12))
}

fn lpsolve_lp_string(
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
    let mut out = String::new();
    out.push_str(match sense {
        Sense::Max => "max: ",
        Sense::Min => "min: ",
    });
    out.push_str(&lp_term_expr(c, &names));
    out.push_str(";\n");

    for (i, (row, rhs)) in le_rows.iter().zip(le_rhs).enumerate() {
        out.push_str(&format!(
            "c{i}: {} <= {};\n",
            lp_term_expr(row, &names),
            fmt_lp_number(*rhs)
        ));
    }
    for (i, (row, rhs)) in eq_rows.iter().zip(eq_rhs).enumerate() {
        out.push_str(&format!(
            "e{i}: {} = {};\n",
            lp_term_expr(row, &names),
            fmt_lp_number(*rhs)
        ));
    }
    if le_rows.is_empty() && eq_rows.is_empty() {
        out.push_str(&format!(
            "c0: 0 {} <= 0;\n",
            names.first().map(String::as_str).unwrap_or("x0")
        ));
    }

    for (i, name) in names.iter().enumerate() {
        if let Some(lower) = lbs
            .get(i)
            .copied()
            .flatten()
            .filter(|value| value.is_finite())
        {
            out.push_str(&format!("{name} >= {};\n", fmt_lp_number(lower)));
        }
        if let Some(upper) = ubs
            .get(i)
            .copied()
            .flatten()
            .filter(|value| value.is_finite())
        {
            out.push_str(&format!("{name} <= {};\n", fmt_lp_number(upper)));
        }
    }
    let integer_names = names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            integer_vars
                .get(i)
                .copied()
                .unwrap_or(false)
                .then_some(name.as_str())
        })
        .collect::<Vec<_>>();
    if !integer_names.is_empty() {
        out.push_str("int ");
        out.push_str(&integer_names.join(", "));
        out.push_str(";\n");
    }
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
    let mut emitted = false;
    if obj_coeff.abs() > 1.0e-12 {
        out.push_str(&format!(
            "    {name:<8}  OBJ       {}\n",
            fmt_lp_number(obj_coeff)
        ));
        emitted = true;
    }
    for (row, row_name) in le_rows.iter().zip(le_names) {
        emitted |= push_mps_row_coef(out, name, row_name, row);
    }
    for (row, row_name) in eq_rows.iter().zip(eq_names) {
        emitted |= push_mps_row_coef(out, name, row_name, row);
    }
    if !emitted {
        out.push_str(&format!("    {name:<8}  OBJ       0\n"));
    }
}

fn push_mps_row_coef(out: &mut String, col_name: &str, row_name: &str, row: &[f64]) -> bool {
    let Some(var_idx) = col_name
        .strip_prefix('x')
        .and_then(|idx| idx.parse::<usize>().ok())
    else {
        return false;
    };
    let Some(&coef) = row.get(var_idx) else {
        return false;
    };
    if coef.abs() > 1.0e-12 {
        out.push_str(&format!(
            "    {col_name:<8}  {row_name:<8}  {}\n",
            fmt_lp_number(coef)
        ));
        return true;
    }
    false
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

fn active_branch_priorities(
    branch_priorities: Option<&[i32]>,
    integer_vars: Option<&[bool]>,
    variable_count: usize,
) -> Result<Vec<(usize, i32)>, String> {
    let Some(branch_priorities) = branch_priorities else {
        return Ok(Vec::new());
    };
    if branch_priorities.len() != variable_count {
        return Err(format!(
            "branch_priorities length {} does not match variable count {}",
            branch_priorities.len(),
            variable_count
        ));
    }
    Ok(branch_priorities
        .iter()
        .enumerate()
        .filter_map(|(idx, priority)| {
            if *priority == 0 {
                return None;
            }
            if integer_vars
                .and_then(|vars| vars.get(idx))
                .copied()
                .is_some_and(|is_integer| !is_integer)
            {
                return None;
            }
            Some((idx, *priority))
        })
        .collect())
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

fn linear_cli_reference_timeout_ms() -> u64 {
    std::env::var("LINEAR_CLI_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_linear_cli_reference_output(
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
            Err(err) => return Err(format!("failed to poll local CLI bridge: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed while waiting for local CLI bridge: {err}"))
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
    use super::wait_for_linear_cli_reference_output;
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
        solve_ipmip_with_external_cli, solve_lower_bounded_ipmip_with_external_cli,
        solve_lp_with_external_cli, solve_multi_objective_ipmip_with_external_cli,
        solve_source_ipmip_with_external_cli, solver_command_env_var,
        sos_ipmip_problem_to_cli_json, source_ipmip_problem_to_cli_json,
        ExternalLinearCliBranchRule, ExternalLinearCliKind, ExternalLinearCliLicenseClass,
        ExternalLinearCliLpAlgorithm, ExternalLinearCliMipSwitch, ExternalLinearCliModelFormat,
        ExternalLinearCliNodeSelection, ExternalLinearCliOptions, ExternalLinearCliPresolve,
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
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

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
    fn linear_cli_python_bridge_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_linear_cli_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }

    #[test]
    fn linear_cli_python_bridge_wait_observes_closed_stdin() {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg("cat >/dev/null; printf done")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn stdin reader");
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(b"{\"kind\":\"lp\",\"solver\":\"highs\"}")
            .expect("write stdin");
        drop(child.stdin.take());

        let (output, timed_out) =
            wait_for_linear_cli_reference_output(child, 1_000).expect("closed stdin output");

        assert!(!timed_out);
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout), "done");
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
    fn quadratic_objective_json_uses_rust_linearization_without_python_bridge() {
        let payload =
            quadratic_objective_ipmip_problem_to_cli_json(&build_quadratic_objective_mix_ip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(PathBuf::from(
                    "/definitely/not-a-highs-quadratic-objective-json-binary",
                )),
                python: Some("/definitely/not-a-python-for-quadratic-json".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );

        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "highs:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "quadratic objective JSON unexpectedly used Python bridge: {}",
            solution.message
        );
    }

    #[test]
    fn quadratic_objective_json_rejects_bad_term_index_without_python_bridge() {
        let mut payload =
            quadratic_objective_ipmip_problem_to_cli_json(&build_quadratic_objective_mix_ip());
        payload["quadratic_objective"][0]["x_var"] = serde_json::json!(999);

        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(PathBuf::from(
                    "/definitely/not-a-highs-quadratic-objective-json-binary",
                )),
                python: Some("/definitely/not-a-python-for-bad-quadratic-json".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );

        assert_eq!(solution.status, ExternalLinearCliStatus::NumericalError);
        assert_eq!(solution.solver, "highs:cli");
        assert!(
            solution
                .message
                .contains("quadratic_objective[0].x_var index 999 is outside variable count"),
            "{}",
            solution.message
        );
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "bad quadratic objective JSON unexpectedly used Python bridge: {}",
            solution.message
        );
    }

    #[test]
    fn linearized_mip_start_shifts_lower_bounded_source_values() {
        let mut opts = ExternalLinearCliOptions {
            mip_start: Some(vec![3.0, 5.0]),
            ..Default::default()
        };

        super::shift_linearized_external_mip_start(&mut opts, &[3.0, 0.0], 2, 2)
            .expect("shift mip start");

        assert_eq!(opts.mip_start, Some(vec![0.0, 5.0]));
    }

    #[test]
    fn linearized_mip_start_pads_auxiliary_variables() {
        let mut opts = ExternalLinearCliOptions {
            mip_start: Some(vec![3.0, 5.0]),
            ..Default::default()
        };

        super::shift_linearized_external_mip_start(&mut opts, &[1.0, 0.0], 2, 5)
            .expect("pad mip start");

        assert_eq!(opts.mip_start, Some(vec![2.0, 5.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn rust_linearized_source_cli_allows_native_branch_priority_solvers() {
        assert!(super::should_use_rust_linearized_source_cli(
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Scip,
                branch_priorities: Some(vec![5]),
                ..Default::default()
            }
        ));
        assert!(super::should_use_rust_linearized_source_cli(
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Cbc,
                branch_priorities: Some(vec![5]),
                ..Default::default()
            }
        ));
        assert!(!super::should_use_rust_linearized_source_cli(
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                branch_priorities: Some(vec![5]),
                ..Default::default()
            }
        ));
        assert!(super::should_use_rust_linearized_source_cli(
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                ..Default::default()
            }
        ));
    }

    #[test]
    fn rust_linearized_source_cli_allows_native_mip_start_solvers() {
        for solver in [
            ExternalLinearCliSolver::Highs,
            ExternalLinearCliSolver::Scip,
            ExternalLinearCliSolver::Cbc,
        ] {
            assert!(
                super::should_use_rust_linearized_source_cli(&ExternalLinearCliOptions {
                    solver,
                    mip_start: Some(vec![0.0]),
                    ..Default::default()
                }),
                "{} should keep source MIP starts on the Rust/native path",
                solver.as_str()
            );
        }
        assert!(!super::should_use_rust_linearized_source_cli(
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Glpk,
                mip_start: Some(vec![0.0]),
                ..Default::default()
            }
        ));
        assert!(!super::should_use_rust_linearized_source_cli(
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                mip_start: Some(vec![0.0]),
                solution_pool_size: Some(2),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn linearized_branch_priorities_pad_auxiliary_variables() {
        let mut opts = ExternalLinearCliOptions {
            branch_priorities: Some(vec![7, 0]),
            ..Default::default()
        };

        super::pad_linearized_external_branch_priorities(&mut opts, 5)
            .expect("pad branch priorities");

        assert_eq!(opts.branch_priorities, Some(vec![7, 0, 0, 0, 0]));
    }

    #[test]
    fn linearized_branch_priorities_reject_too_many_values() {
        let mut opts = ExternalLinearCliOptions {
            branch_priorities: Some(vec![7, 0, 3]),
            ..Default::default()
        };

        let error = super::pad_linearized_external_branch_priorities(&mut opts, 2)
            .expect_err("too many branch priorities should fail");

        assert!(error.contains("exceeds linearized variable count"));
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
    fn ipmip_cplex_export_declares_unused_binary_columns() {
        let p = IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.0, 0.0, 0.0],
            a: vec![vec![1.0, 1.0, 0.0]],
            b: vec![1.0],
            integer_vars: vec![true, true, true],
            ub: Some(vec![1.0, 1.0, 1.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let text = ipmip_problem_to_cplex_lp_string(&p);
        assert!(text.contains("Bounds\n 0 <= x2 <= 1\n"));
        assert!(text.contains("Binary\n x0 x1 x2\n"));
    }

    #[test]
    fn ipmip_lp_solve_export_uses_solver_lp_dialect() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 2.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![1.0],
            integer_vars: vec![true, false],
            ub: Some(vec![1.0, 2.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let text = super::ipmip_problem_to_lpsolve_lp_string(&p);
        assert!(text.starts_with("max: x0 + 2 x1;\n"));
        assert!(text.contains("c0: x0 + x1 <= 1;\n"));
        assert!(text.contains("x0 <= 1;\n"));
        assert!(text.contains("int x0;\n"));
    }

    #[test]
    fn ipmip_exports_lazy_constraints_as_cli_rows() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0],
            a: vec![vec![1.0, 0.0]],
            b: vec![1.0],
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

        let lp_text = ipmip_problem_to_cplex_lp_string(&p);
        assert!(lp_text.contains(" c0: x0 <= 1\n"));
        assert!(lp_text.contains(" c1: x0 + x1 <= 1\n"));

        let mps_text = ipmip_problem_to_mps_string(&p);
        assert!(mps_text.contains(" L  c1\n"));
        assert!(mps_text.contains("    x0        c1        1\n"));
        assert!(mps_text.contains("    x1        c1        1\n"));
        assert!(mps_text.contains("    RHS1      c1        1\n"));
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
    fn glpk_mps_export_omits_objsense_header() {
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![1.0, 2.0],
            a_ub: Some(vec![vec![1.0, 1.0]]),
            b_ub: Some(vec![3.0]),
            ..Default::default()
        };
        let text = super::lp_problem_to_mps_string_with_objsense(&p, false);
        assert!(text.starts_with("NAME          ORES\nROWS\n"));
        assert!(!text.contains("OBJSENSE"));
        assert!(text.contains(" N  OBJ\n"));
        assert!(text.ends_with("ENDATA\n"));
    }

    #[test]
    fn mps_export_keeps_bound_only_columns_defined() {
        let p = IPMIPProblem {
            sense: Sense::Min,
            c: vec![1.0, 0.0, 0.0],
            a: vec![vec![1.0, 1.0, 0.0]],
            b: vec![1.0],
            integer_vars: vec![true, true, true],
            ub: Some(vec![1.0, 1.0, 1.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let text = ipmip_problem_to_mps_string(&p);
        assert!(text.contains("    x2        OBJ       0\n"));
        assert!(text.contains(" BV BND1      x2\n"));
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
    fn multi_objective_uses_native_highs_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Highs) else {
            eprintln!("SKIP multi-objective HiGHS solve: highs command not installed");
            return;
        };
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

        let solution = solve_multi_objective_ipmip_with_external_cli(
            &p,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-multi-objective".to_string()),
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
        assert_eq!(solution.x, vec![1.0, 0.0]);
        assert_eq!(solution.objective_values, Some(vec![1.0, 3.0]));
        assert_eq!(solution.objective, Some(3.0));
        assert_eq!(solution.message, "sequential lexicographic optimization");
    }

    #[test]
    fn multi_objective_json_uses_native_highs_without_python_bridge() {
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
        let payload = multi_objective_ipmip_problem_to_cli_json(&MultiObjectiveIPMIPProblem {
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
        });

        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(PathBuf::from(
                    "/definitely/not-a-highs-multi-objective-json-binary",
                )),
                python: Some("/definitely/not-a-python-for-multi-objective-json".to_string()),
                time_limit_secs: Some(2.0),
                random_seed: Some(7),
                ..Default::default()
            },
        );

        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "highs:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "multi-objective JSON unexpectedly used Python bridge: {}",
            solution.message
        );
    }

    #[test]
    fn multi_objective_json_rejects_bad_stage_width_without_python_bridge() {
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
        let mut payload = multi_objective_ipmip_problem_to_cli_json(&MultiObjectiveIPMIPProblem {
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
        });
        payload["multi_objectives"][1]["c"] = serde_json::json!([3.0]);

        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(PathBuf::from(
                    "/definitely/not-a-highs-multi-objective-json-binary",
                )),
                python: Some("/definitely/not-a-python-for-bad-multi-objective-json".to_string()),
                time_limit_secs: Some(2.0),
                random_seed: Some(7),
                ..Default::default()
            },
        );

        assert_eq!(solution.status, ExternalLinearCliStatus::NumericalError);
        assert_eq!(solution.solver, "highs:cli");
        assert!(
            solution
                .message
                .contains("multi_objectives[1].c length 1 does not match variable count 2"),
            "{}",
            solution.message
        );
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "bad multi-objective JSON unexpectedly used Python bridge: {}",
            solution.message
        );
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
    fn native_highs_solution_parser_reads_primal_dual_and_basis_sections() {
        let text = "\
Model status
Optimal

# Primal solution values
Feasible
Objective 5
# Columns 2
x0 1
x1 2
# Rows 2
c0 0
e0 0

# Dual solution values
Feasible
# Columns 2
x0 0.5
x1 -0.25
# Rows 2
c0 3
e0 -4

# Basis
HiGHS_basis_file v2
# Columns 2
x0 1
x1 2
# Rows 2
c0 0
e0 1
";
        let parsed = super::parse_native_highs_solution_text(text, 2, 1, 1).unwrap();
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![1.0, 2.0]);
        assert_eq!(parsed.reduced_costs, Some(vec![0.5, -0.25]));
        assert_eq!(parsed.dual_ub, Some(vec![3.0]));
        assert_eq!(parsed.dual_eq, Some(vec![-4.0]));
        assert_eq!(
            parsed.var_basis,
            Some(vec!["basic".to_string(), "at_upper".to_string()])
        );
        assert_eq!(
            parsed.row_basis,
            Some(vec!["at_lower".to_string(), "basic".to_string()])
        );
    }

    #[test]
    fn native_highs_mip_quality_parser_reads_report_bounds_gap_and_nodes() {
        let stdout = "\
Solving report
  Status            Optimal
  Primal bound      41
  Dual bound        40
  Gap               2.5% (tolerance: 0.01%)
  Nodes             12
";
        let quality = super::parse_highs_mip_quality(ExternalLinearCliKind::Mip, 41.0, stdout, "");
        assert_eq!(quality.best_bound, Some(40.0));
        assert_eq!(quality.nodes_explored, Some(12));
        assert!((quality.mip_gap.unwrap() - 0.025).abs() <= 1.0e-12);
        assert_eq!(quality.absolute_gap, Some(1.0));
    }

    #[test]
    fn native_highs_mip_start_feedback_parser_reads_infeasibility_report() {
        let stdout = "\
Assessing feasibility of MIP using primal feasibility and integrality tolerance of       1e-06
Solution has               num          max          sum
Col     infeasibilities      0            0            0
Integer infeasibilities      0            0            0
Row     infeasibilities      0            0            0
";
        let (accepted, objective) = super::parse_highs_mip_start_feedback(
            ExternalLinearCliKind::Mip,
            Some(&[1.0, 0.0]),
            Some(3.0),
            stdout,
            "",
        );
        assert_eq!(accepted, Some(true));
        assert_eq!(objective, Some(3.0));
    }

    #[test]
    fn native_highs_solution_limit_uses_native_mip_options() {
        let opts = ExternalLinearCliOptions {
            solver: ExternalLinearCliSolver::Highs,
            solution_limit: Some(3),
            python: Some("/definitely/not-a-python-for-highs-solution-limit".to_string()),
            ..Default::default()
        };

        assert!(super::should_use_native_highs_cli(
            ExternalLinearCliKind::Mip,
            &opts
        ));
        assert!(!super::should_use_native_highs_cli(
            ExternalLinearCliKind::Lp,
            &opts
        ));
        let options_text = super::native_highs_options_text(
            ExternalLinearCliKind::Mip,
            &opts,
            &PathBuf::from("highs.log"),
        )
        .expect("HiGHS options text");
        assert!(options_text.contains("mip_max_improving_sols = 3"));
    }

    #[test]
    fn native_glpk_solution_parser_reads_plain_lp_sections() {
        let text = "\
c Status:     OPTIMAL
s bas 2 2 f f 12
i 1 u 4 3
i 2 b 6 -4
j 1 b 1 0.5
j 2 u 2 -0.25
e o f
";
        let parsed = super::parse_native_glpk_solution_text(text, 2, 1, 1).unwrap();
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![1.0, 2.0]);
        assert_eq!(parsed.reduced_costs, Some(vec![0.5, -0.25]));
        assert_eq!(parsed.dual_ub, Some(vec![3.0]));
        assert_eq!(parsed.dual_eq, Some(vec![-4.0]));
        assert_eq!(
            parsed.var_basis,
            Some(vec!["basic".to_string(), "at_upper".to_string()])
        );
        assert_eq!(
            parsed.row_basis,
            Some(vec!["at_upper".to_string(), "basic".to_string()])
        );
    }

    #[test]
    fn native_glpk_solution_parser_reads_printable_mip_columns() {
        let text = "\
Status:     INTEGER OPTIMAL

   No. Column name       Activity     Lower bound   Upper bound
------ ------------ ------------- ------------- -------------
     1 x0           *             1             0             1
     2 x1           *             0             0             1

Integer feasibility conditions:
";
        let parsed = super::parse_native_glpk_solution_text(text, 2, 0, 0).unwrap();
        assert_eq!(parsed.status, "integer optimal");
        assert_eq!(parsed.x, vec![1.0, 0.0]);
    }

    #[test]
    fn native_cbc_solution_parser_reads_lp_solution_and_basis() {
        let solution_text = "\
Optimal - objective value 12.00000000
      0 c0                     4                       3
      1 e0                     6                      -4
      2 x0                     1                     0.5
      3 x1                     2                   -0.25
";
        let basis_text = "\
NAME          ORES
 XL x0 c0 0
 LL x1 0
ENDATA
";
        let parsed =
            super::parse_native_cbc_solution_text(solution_text, 2, 1, 1, Some(basis_text))
                .unwrap();
        assert_eq!(parsed.status, "optimal - objective value 12.00000000");
        assert_eq!(parsed.x, vec![1.0, 2.0]);
        assert_eq!(parsed.reduced_costs, Some(vec![0.5, -0.25]));
        assert_eq!(parsed.dual_ub, Some(vec![3.0]));
        assert_eq!(parsed.dual_eq, Some(vec![-4.0]));
        assert_eq!(
            parsed.var_basis,
            Some(vec!["basic".to_string(), "at_lower".to_string()])
        );
        assert_eq!(
            parsed.row_basis,
            Some(vec!["at_lower".to_string(), "fixed".to_string()])
        );
    }

    #[test]
    fn native_cbc_mip_quality_parser_reads_optimal_nodes() {
        let stdout = "\
Result - Optimal solution found

Objective value:                1.00000000
Enumerated nodes:               7
Total iterations:               11
";
        let quality = super::parse_cbc_mip_quality(
            ExternalLinearCliKind::Mip,
            ExternalLinearCliStatus::Optimal,
            1.0,
            stdout,
            "",
        );
        assert_eq!(quality.nodes_explored, Some(7));
        assert_eq!(quality.best_bound, Some(1.0));
        assert_eq!(quality.mip_gap, Some(0.0));
        assert_eq!(quality.absolute_gap, Some(0.0));
    }

    #[test]
    fn native_cbc_mip_start_feedback_parser_reads_values_read_message() {
        let stdout = "\
opening mipstart file start.sol.
MIPStart values read for 2 variables.
";
        let (accepted, objective) = super::parse_cbc_mip_start_feedback(
            ExternalLinearCliKind::Mip,
            Some(&[1.0, 0.0]),
            Some(3.0),
            stdout,
            "",
        );
        assert_eq!(accepted, Some(true));
        assert_eq!(objective, Some(3.0));
    }

    #[test]
    fn native_cbc_branch_priority_feedback_parser_reads_priorityin_command() {
        let stdout = "command line - cbc model.lp -priorityIn priorities.csv -solve\n";
        let (accepted, count) =
            super::parse_cbc_branch_priority_feedback(ExternalLinearCliKind::Mip, 2, stdout, "");
        assert_eq!(accepted, Some(true));
        assert_eq!(count, Some(2));
    }

    #[test]
    fn native_glpk_mip_quality_parser_reads_preprocessor_optimal() {
        let stdout = "\
GLPK Integer Optimizer 5.0
Objective value =   1.000000000e+00
INTEGER OPTIMAL SOLUTION FOUND BY MIP PREPROCESSOR
";
        let quality = super::parse_glpk_mip_quality(
            ExternalLinearCliKind::Mip,
            ExternalLinearCliStatus::Optimal,
            1.0,
            stdout,
            "",
        );
        assert_eq!(quality.nodes_explored, Some(0));
        assert_eq!(quality.best_bound, Some(1.0));
        assert_eq!(quality.mip_gap, Some(0.0));
        assert_eq!(quality.absolute_gap, Some(0.0));
    }

    #[test]
    fn native_clp_version_parser_uses_clp_label() {
        let text = "Coin LP version 1.17.11, build Mar 11 2026\n";
        assert_eq!(
            super::parse_clp_solver_version(text),
            Some("CLP 1.17.11".to_string())
        );
    }

    #[test]
    fn native_soplex_solution_parser_reads_named_values_and_status() {
        let solution_text = "\
Primal solution
x0 = 1.5
2.25 x1
";
        let stdout = "SoPlex version 8.0.2\nProblem is solved [optimal]\nIterations : 7\n";
        let parsed = super::parse_native_soplex_solution_text(solution_text, 2, stdout, "");
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![1.5, 2.25]);
        assert_eq!(
            super::parse_soplex_solver_version(stdout),
            Some("SoPlex 8.0.2".to_string())
        );
        assert_eq!(super::parse_soplex_lp_iterations(stdout, ""), Some(7));
    }

    #[test]
    fn native_soplex_dual_and_basis_parsers_read_certificates() {
        let dual_text = "\
Dual solution (name, value):
c0                         1.5
e0                        -0.5
All other dual values are zero (within 1.0e-16).

Reduced costs (name, value):
x1                        -2.0
All other reduced costs are zero (within 1.0e-16).
";
        let fields = super::parse_native_soplex_dual_text(dual_text, 2, 1, 1);
        assert_eq!(fields.dual_ub, Some(vec![1.5]));
        assert_eq!(fields.dual_eq, Some(vec![-0.5]));
        assert_eq!(fields.reduced_costs, Some(vec![0.0, -2.0]));

        let basis_text = "\
NAME  soplex.bas
 XU x0             c0
 XL x1             e0
ENDATA
";
        let (var_basis, row_basis) = super::parse_native_cbc_basis_text(basis_text, 2, 1, 1);
        assert_eq!(
            var_basis,
            Some(vec!["basic".to_string(), "basic".to_string()])
        );
        assert_eq!(
            row_basis,
            Some(vec!["at_upper".to_string(), "at_lower".to_string()])
        );
    }

    #[test]
    fn native_qsopt_ex_solution_parser_reads_certificates_basis_and_version() {
        let solution_text = "\
status optimal
VARS:
x0 1
x2 -3.5
REDUCED COSTS:
x0 0.25
x2 -0.75
PI:
c0 2
e0 -1
";
        let stdout = "Using QSopt_ex 2.5.10\nProblem solved exactly\n";
        let mut parsed =
            super::parse_native_qsopt_ex_solution_text(solution_text, 3, 1, 1, stdout, "");
        let basis_text = "\
 XL x0 c0
 XU x1 e0
";
        let (var_basis, row_basis) = super::parse_native_qsopt_ex_basis_text(basis_text, 3, 1, 1);
        parsed.var_basis = var_basis;
        parsed.row_basis = row_basis;

        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![1.0, 0.0, -3.5]);
        assert_eq!(parsed.reduced_costs, Some(vec![0.25, 0.0, -0.75]));
        assert_eq!(parsed.dual_ub, Some(vec![2.0]));
        assert_eq!(parsed.dual_eq, Some(vec![-1.0]));
        assert_eq!(parsed.var_basis, None);
        assert_eq!(
            parsed.row_basis,
            Some(vec!["at_upper".to_string(), "at_upper".to_string()])
        );
        assert_eq!(
            super::parse_qsopt_ex_solver_version(stdout),
            Some("QSopt_ex 2.5.10".to_string())
        );
    }

    #[test]
    fn native_qsopt_ex_solution_parser_reads_exact_rational_values() {
        let solution_text = "\
status OPTIMAL
\tValue = 0
VARS:
x1 = 499999987/125000000
REDUCED COST:
x0 = -1
PI:
SLACK:
c0 = 13/125000000
";
        let parsed = super::parse_native_qsopt_ex_solution_text(solution_text, 2, 2, 0, "", "");

        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x[0], 0.0);
        assert!((parsed.x[1] - 3.999999896).abs() <= 1.0e-12);
        assert_eq!(parsed.reduced_costs, Some(vec![-1.0, 0.0]));
        assert_eq!(parsed.dual_ub, Some(vec![0.0, 0.0]));
    }

    #[test]
    fn native_qsopt_ex_solution_basis_infers_from_certificates() {
        let solution_text = "\
status optimal
VARS:
x0 6
x1 4
REDUCED COSTS:
x0 0
x1 0
PI:
c0 2.3333333333333335
c1 0
c2 0.6666666666666666
";
        let parsed = super::parse_native_qsopt_ex_solution_text(solution_text, 2, 3, 0, "", "");
        let le_rows = vec![vec![1.0, 2.0], vec![-3.0, 1.0], vec![1.0, -1.0]];
        let le_rhs = vec![14.0, 0.0, 2.0];
        let lower_bounds = vec![Some(0.0), Some(0.0)];
        let upper_bounds = vec![None, None];
        let (var_basis, row_basis) = super::qsopt_ex_lp_basis_from_solution(
            &parsed,
            &le_rows,
            &le_rhs,
            &[],
            &[],
            Some(&lower_bounds),
            Some(&upper_bounds),
        );

        assert_eq!(
            var_basis,
            Some(vec!["basic".to_string(), "basic".to_string()])
        );
        assert_eq!(
            row_basis,
            Some(vec![
                "at_upper".to_string(),
                "basic".to_string(),
                "at_upper".to_string()
            ])
        );
    }

    #[test]
    fn native_qsopt_ex_solution_reconstructs_zero_pi_certificates_from_basis() {
        let solution_text = "\
status optimal
VARS:
x0 6
x1 4
REDUCED COSTS:
x0 0
x1 0
PI:
c0 0
c1 0
c2 0
";
        let mut parsed = super::parse_native_qsopt_ex_solution_text(solution_text, 2, 3, 0, "", "");
        parsed.var_basis = Some(vec!["basic".to_string(), "basic".to_string()]);
        parsed.row_basis = Some(vec![
            "at_upper".to_string(),
            "basic".to_string(),
            "at_upper".to_string(),
        ]);
        let objective = vec![3.0, 4.0];
        let le_rows = vec![vec![1.0, 2.0], vec![-3.0, 1.0], vec![1.0, -1.0]];
        let eq_rows = Vec::<Vec<f64>>::new();

        assert!(super::qsopt_ex_lp_certificate_needs_reconstruction(
            &parsed, &objective, &le_rows, &eq_rows
        ));
        let (dual_ub, dual_eq, reduced_costs) =
            super::qsopt_ex_lp_certificate_from_basis(&parsed, &objective, &le_rows, &eq_rows)
                .expect("certificate from basis");

        assert!((dual_ub[0] - 7.0 / 3.0).abs() <= 1.0e-8);
        assert_eq!(dual_ub[1], 0.0);
        assert!((dual_ub[2] - 2.0 / 3.0).abs() <= 1.0e-8);
        assert_eq!(dual_eq, Vec::<f64>::new());
        assert_eq!(reduced_costs, vec![0.0, 0.0]);

        parsed.dual_ub = Some(dual_ub);
        parsed.dual_eq = Some(dual_eq);
        parsed.reduced_costs = Some(reduced_costs);
        assert!(!super::qsopt_ex_lp_certificate_needs_reconstruction(
            &parsed, &objective, &le_rows, &eq_rows
        ));
    }

    #[test]
    fn native_scip_solution_parser_reads_named_values_and_quality() {
        let solution_text = "\
solution status: optimal solution found
objective value: 4
x0 1.5
x1 2.25
";
        let stdout = "\
SCIP version 10.0.2 [precision: 8 byte]
Solving Nodes      : 3
Dual Bound         : 4
Gap                : 0.00 %
";
        let parsed = super::parse_native_scip_solution_text(solution_text, 2);
        let quality = super::parse_scip_mip_quality(ExternalLinearCliKind::Mip, 4.0, stdout, "");
        assert_eq!(parsed.status, "optimal solution found");
        assert_eq!(parsed.x, vec![1.5, 2.25]);
        assert_eq!(
            super::parse_scip_solver_version(stdout),
            Some("SCIP 10.0.2".to_string())
        );
        assert_eq!(quality.nodes_explored, Some(3));
        assert_eq!(quality.best_bound, Some(4.0));
        assert_eq!(quality.mip_gap, Some(0.0));
        assert_eq!(quality.absolute_gap, Some(0.0));
    }

    #[test]
    fn native_scip_mip_start_feedback_parser_reads_candidate_storage() {
        let stdout = "\
primal solution from solution file <start.sol> was accepted as candidate
1/1 feasible solution given by solution candidate storage, new primal bound 0.000000e+00
";
        let (accepted, objective) = super::parse_scip_mip_start_feedback(
            ExternalLinearCliKind::Mip,
            Some(&[0.0, 1.0]),
            Some(5.0),
            stdout,
            "",
        );
        assert_eq!(accepted, Some(true));
        assert_eq!(objective, Some(5.0));
    }

    #[test]
    fn native_scip_branch_priority_feedback_parser_reads_priority_message() {
        let stdout = "branching priority of variable <x1> set to 10\n";
        let (accepted, count) =
            super::parse_scip_branch_priority_feedback(ExternalLinearCliKind::Mip, 1, stdout, "");
        assert_eq!(accepted, Some(true));
        assert_eq!(count, Some(1));
    }

    #[test]
    fn active_branch_priorities_filters_zeros_and_continuous_vars() {
        let priorities =
            super::active_branch_priorities(Some(&[5, 0, 3]), Some(&[true, true, false]), 3)
                .unwrap();
        assert_eq!(priorities, vec![(0, 5)]);
        assert!(super::active_branch_priorities(Some(&[1, 2]), Some(&[true]), 1).is_err());
    }

    #[test]
    fn native_scip_dual_parser_builds_lp_certificates() {
        let stdout = "\
objective value:                                   34
x0                                                  6 \t(obj:3)
x1                                                  4 \t(obj:4)


c0                                   2.33333333333333
c2                                  0.666666666666667
";
        let fields = super::parse_scip_lp_certificate_fields(
            stdout,
            Sense::Max,
            &[3.0, 4.0],
            &[vec![1.0, 2.0], vec![-3.0, 1.0], vec![1.0, -1.0]],
            &[14.0, 0.0, 2.0],
            &[],
            &[],
            None,
            None,
            &[6.0, 4.0],
        );
        let dual_ub = fields.dual_ub.unwrap();
        assert!((dual_ub[0] - 7.0 / 3.0).abs() <= 1e-8);
        assert_eq!(dual_ub[1], 0.0);
        assert!((dual_ub[2] - 2.0 / 3.0).abs() <= 1e-8);
        assert_eq!(fields.dual_eq, Some(Vec::new()));
        assert_eq!(fields.reduced_costs, Some(vec![0.0, 0.0]));
        assert_eq!(
            fields.var_basis,
            Some(vec!["basic".to_string(), "basic".to_string()])
        );
        assert_eq!(
            fields.row_basis,
            Some(vec![
                "at_upper".to_string(),
                "basic".to_string(),
                "at_upper".to_string()
            ])
        );
    }

    #[test]
    fn native_scip_dual_parser_reads_equality_rows_and_starred_values() {
        let stdout = "\
objective value:                                    2
x0                                                  2 \t(obj:1)


e0                                                  1*
";
        let fields = super::parse_scip_lp_certificate_fields(
            stdout,
            Sense::Max,
            &[1.0],
            &[],
            &[],
            &[vec![1.0]],
            &[2.0],
            None,
            None,
            &[2.0],
        );
        assert_eq!(fields.dual_ub, Some(Vec::new()));
        assert_eq!(fields.dual_eq, Some(vec![1.0]));
        assert_eq!(fields.reduced_costs, Some(vec![0.0]));
    }

    #[test]
    fn inferred_lp_basis_leaves_degenerate_rows_unreported() {
        let (var_basis, row_basis) = super::infer_lp_basis_from_complementarity(
            &[1.0],
            None,
            None,
            &[vec![1.0]],
            &[1.0],
            &[0.0],
            &[],
            &[],
            &[0.0],
        );
        assert_eq!(var_basis, Some(vec!["basic".to_string()]));
        assert_eq!(row_basis, None);
    }

    #[test]
    fn native_numeric_parser_tolerates_solver_punctuation() {
        assert_eq!(super::parse_f64_token("0.00%"), Some(0.0));
        assert_eq!(super::parse_f64_token("(7.5%),"), Some(7.5));
        assert_eq!(super::parse_f64_token("[3.25]"), Some(3.25));
        assert_eq!(super::parse_f64_token("2*"), Some(2.0));
        assert_eq!(
            super::parse_f64_token("499999987/125000000"),
            Some(3.999999896)
        );
        assert_eq!(
            super::first_float("objective value 499999987/125000000"),
            Some(3.999999896)
        );
        assert_eq!(super::percent_gap_from_line("Gap: (7.5%),"), Some(0.075));
    }

    #[test]
    fn native_solver_message_falls_back_when_diagnostics_are_empty() {
        assert_eq!(
            super::native_solver_message("", "  \n", "\t"),
            "solver produced no diagnostic output"
        );
        assert_eq!(
            super::native_solver_message("", "stdout detail", " stderr detail "),
            "stderr detail"
        );
        assert_eq!(
            super::native_solver_message("optimal", "stdout detail", "stderr detail"),
            "optimal"
        );
    }

    #[test]
    fn native_lp_solve_solution_parser_reads_stdout_values_and_version() {
        let stdout = "\
Usage of lp_solve version 5.5.2.14:

Value of objective function: 12.00000000

Actual values of the variables:
x0                              4
x1                              0

Dual values with from - till limits:
                           Dual value            From            Till
x0                                  0          -1e+30           1e+30
";
        let parsed = super::parse_native_lp_solve_solution_text(stdout, 2);
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.objective, Some(12.0));
        assert_eq!(parsed.x, vec![4.0, 0.0]);
        assert_eq!(
            super::parse_lp_solve_solver_version(stdout),
            Some("lp_solve 5.5.2.14".to_string())
        );
    }

    #[test]
    fn native_named_solution_parser_reads_gurobi_style_values_and_version() {
        let text = "\
# Objective value = 7
x0 1
x2 -3.5
x9 99
";
        let parsed = super::parse_native_named_solution_text(text, 3, "optimal");
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![1.0, 0.0, -3.5]);
        assert_eq!(
            super::parse_gurobi_solver_version("Gurobi Optimizer version 12.0.1 build v12.0.1"),
            Some("Gurobi 12.0.1".to_string())
        );
    }

    #[test]
    fn native_cplex_solution_parser_reads_xml_values_status_and_version() {
        let text = r#"
<?xml version="1.0" encoding="UTF-8"?>
<CPLEXSolution version="1.2">
  <header solutionStatusString="integer optimal solution" objectiveValue="7"/>
  <variables>
    <variable name="x0" index="0" value="1"/>
    <variable name="x2" index="2" value="-3.5"/>
    <variable name="x9" index="9" value="99"/>
  </variables>
</CPLEXSolution>
"#;
        let parsed = super::parse_native_cplex_solution_text(text, 3);
        assert_eq!(parsed.status, "integer optimal solution");
        assert_eq!(parsed.x, vec![1.0, 0.0, -3.5]);

        let numeric_status = super::parse_native_cplex_solution_text(
            r#"<CPLEXSolution><header solutionStatusValue="103"/></CPLEXSolution>"#,
            1,
        );
        assert_eq!(numeric_status.status, "infeasible");
        assert_eq!(
            super::parse_cplex_solver_version("IBM ILOG CPLEX Interactive Optimizer 22.1.1.0"),
            Some("CPLEX 22.1.1.0".to_string())
        );
    }

    #[test]
    fn native_xpress_solution_parser_reads_delimited_values_status_and_version() {
        let text = "\
# Columns
\"x0\", 1
x1; 0
x2; -3.5
x9; 99
";
        let parsed = super::parse_native_xpress_solution_text(
            text,
            Some("Global search complete: optimal solution found"),
            3,
        );
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![1.0, 0.0, -3.5]);

        let infeasible =
            super::parse_native_xpress_solution_text("x0 1", Some("Problem is infeasible"), 1);
        assert_eq!(infeasible.status, "infeasible");
        assert_eq!(
            super::parse_xpress_solver_version("FICO Xpress Optimizer 9.4.0"),
            Some("Xpress 9.4.0".to_string())
        );
    }

    #[test]
    fn native_xpress_solution_file_parser_reads_asc_companion() {
        let base_path = std::env::temp_dir().join(format!(
            "des-rs-xpress-solution-parser-{}.sol",
            std::process::id()
        ));
        let data_path = super::native_xpress_solution_data_path(&base_path);
        let header_path = super::native_xpress_header_path(&base_path);
        std::fs::write(&data_path, "x0 1\nx1 0\n").unwrap();
        std::fs::write(
            &header_path,
            "Global search complete: optimal solution found",
        )
        .unwrap();

        let parsed = super::parse_native_xpress_solution_file(&base_path, 2).unwrap();

        assert_eq!(data_path, base_path.with_extension("sol.asc"));
        assert_eq!(header_path, base_path.with_extension("sol.hdr"));
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![1.0, 0.0]);

        let _ = std::fs::remove_file(&data_path);
        let _ = std::fs::remove_file(&header_path);
    }

    #[test]
    fn native_lindo_solution_parser_reads_named_values_status_and_version() {
        let text = "\
Objective value: 7
x0 1
x1 0
x2 -3.5
x9 99
";
        let parsed = super::parse_native_lindo_solution_text(text, 3);
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![1.0, 0.0, -3.5]);

        let infeasible = super::parse_native_lindo_solution_text("No feasible solution", 1);
        assert_eq!(infeasible.status, "infeasible");
        assert_eq!(
            super::parse_lindo_solver_version("LINDO Optimizer 14.0"),
            Some("LINDO 14.0".to_string())
        );
    }

    #[test]
    fn gams_lindo_mip_model_emits_start_and_basic_controls() {
        let model = super::PlainLinearCliModel {
            sense: Sense::Max,
            c: vec![60.0, 100.0, 120.0],
            le_rows: vec![vec![10.0, 20.0, 30.0]],
            le_rhs: vec![50.0],
            eq_rows: Vec::new(),
            eq_rhs: Vec::new(),
            lbs: vec![Some(0.0), Some(0.0), Some(0.0)],
            ubs: vec![Some(1.0), Some(1.0), Some(1.0)],
            integer_vars: vec![true, true, true],
        };
        let opts = ExternalLinearCliOptions {
            solver: ExternalLinearCliSolver::Lindo,
            time_limit_secs: Some(2.5),
            node_limit: Some(3),
            relative_gap: Some(0.25),
            mip_start: Some(vec![0.0, 1.0, 1.0]),
            ..Default::default()
        };
        let text = super::gams_lindo_model_text(
            ExternalLinearCliKind::Mip,
            &model,
            std::path::Path::new("/tmp/des-rs-lindo-solution.txt"),
            &opts,
        );

        assert!(text.contains("Binary Variables x0, x1, x2;"));
        assert!(text.contains("x0.l = 0;"));
        assert!(text.contains("x1.l = 1;"));
        assert!(text.contains("x2.l = 1;"));
        assert!(text.contains("option reslim = 2.5;"));
        assert!(text.contains("option optcr = 0.25;"));
        assert!(!text.contains("nodlim"));
        assert!(text.contains("Solve m using mip maximizing z;"));
        assert_eq!(
            super::native_lindo_gams_mip_start_objective(
                ExternalLinearCliKind::Mip,
                model.c.len(),
                &model.c,
                &opts
            ),
            Ok(Some(220.0))
        );
        assert_eq!(
            super::native_lindo_gams_mip_start_objective(
                ExternalLinearCliKind::Mip,
                model.c.len(),
                &model.c,
                &ExternalLinearCliOptions {
                    solver: ExternalLinearCliSolver::Lindo,
                    mip_start: Some(vec![1.0]),
                    ..Default::default()
                }
            ),
            Err("mip_start length 1 does not match variable count 3".to_string())
        );
    }

    #[test]
    fn gams_lindo_lp_model_emits_certificate_marginal_fields() {
        let model = super::PlainLinearCliModel {
            sense: Sense::Max,
            c: vec![3.0, 4.0],
            le_rows: vec![vec![1.0, 2.0], vec![-3.0, 1.0], vec![1.0, -1.0]],
            le_rhs: vec![14.0, 0.0, 2.0],
            eq_rows: vec![vec![1.0, 0.0]],
            eq_rhs: vec![6.0],
            lbs: vec![Some(0.0), Some(0.0)],
            ubs: vec![None, None],
            integer_vars: vec![false, false],
        };
        let text = super::gams_lindo_model_text(
            ExternalLinearCliKind::Lp,
            &model,
            std::path::Path::new("/tmp/des-rs-lindo-solution.txt"),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Lindo,
                ..Default::default()
            },
        );

        assert!(text.contains("Solve m using lp maximizing z;"));
        assert!(text.contains("put 'le0.m ' le0.m:0:17 /;"));
        assert!(text.contains("put 'le1.m ' le1.m:0:17 /;"));
        assert!(text.contains("put 'le2.m ' le2.m:0:17 /;"));
        assert!(text.contains("put 'eq0.m ' eq0.m:0:17 /;"));
        assert!(text.contains("put 'x0.m ' x0.m:0:17 /;"));
        assert!(text.contains("put 'x1.m ' x1.m:0:17 /;"));
    }

    #[test]
    fn native_lindo_gams_solution_parser_reads_lp_marginals() {
        let row_senses = "\
modelstat 1
objective 34
x0 6
x1 4
le0.m 2.333333333333333
le1.m 0
le2.m 0.666666666666667
x0.m 0
x1.m 0
";
        let parsed = super::parse_native_lindo_gams_solution_text(row_senses, 2, 3, 0, "");
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![6.0, 4.0]);
        let dual_ub = parsed.dual_ub.as_ref().unwrap();
        assert!((dual_ub[0] - 7.0 / 3.0).abs() <= 1.0e-12);
        assert_eq!(dual_ub[1], 0.0);
        assert!((dual_ub[2] - 2.0 / 3.0).abs() <= 1.0e-12);
        assert_eq!(parsed.dual_eq, Some(Vec::new()));
        assert_eq!(parsed.reduced_costs, Some(vec![0.0, 0.0]));
        let model = super::PlainLinearCliModel {
            sense: Sense::Max,
            c: vec![3.0, 4.0],
            le_rows: vec![vec![1.0, 2.0], vec![-3.0, 1.0], vec![1.0, -1.0]],
            le_rhs: vec![14.0, 0.0, 2.0],
            eq_rows: Vec::new(),
            eq_rhs: Vec::new(),
            lbs: vec![Some(0.0), Some(0.0)],
            ubs: vec![None, None],
            integer_vars: vec![false, false],
        };
        let (var_basis, row_basis) = super::lindo_gams_lp_basis_from_solution(&model, &parsed);
        assert_eq!(
            var_basis,
            Some(vec!["basic".to_string(), "basic".to_string()])
        );
        assert_eq!(
            row_basis,
            Some(vec![
                "at_upper".to_string(),
                "basic".to_string(),
                "at_upper".to_string()
            ])
        );

        let equality = "\
modelstat 1
objective 2
x0 2
eq0.m 1
x0.m 0
";
        let parsed = super::parse_native_lindo_gams_solution_text(equality, 1, 0, 1, "");
        assert_eq!(parsed.status, "optimal");
        assert_eq!(parsed.x, vec![2.0]);
        assert_eq!(parsed.dual_ub, Some(Vec::new()));
        assert_eq!(parsed.dual_eq, Some(vec![1.0]));
        assert_eq!(parsed.reduced_costs, Some(vec![0.0]));
        let equality_model = super::PlainLinearCliModel {
            sense: Sense::Max,
            c: vec![1.0],
            le_rows: Vec::new(),
            le_rhs: Vec::new(),
            eq_rows: vec![vec![1.0]],
            eq_rhs: vec![2.0],
            lbs: vec![Some(0.0)],
            ubs: vec![None],
            integer_vars: vec![false],
        };
        let (var_basis, row_basis) =
            super::lindo_gams_lp_basis_from_solution(&equality_model, &parsed);
        assert_eq!(var_basis, Some(vec!["basic".to_string()]));
        assert_eq!(row_basis, Some(vec!["fixed".to_string()]));
    }

    #[test]
    fn native_lp_solve_lp_certificate_parser_reads_duals_reduced_costs_and_basis() {
        let stdout = "\
Value of objective function: 34.00000000

Actual values of the variables:
x0                              6
x1                              4

Dual values with from - till limits:
                           Dual value            From            Till
c0                           2.333333               2           1e+30
c1                                  0          -1e+30           1e+30
c2                          0.6666667              -4              14
x0                                  0          -1e+30           1e+30
x1                                  0          -1e+30           1e+30
";
        let basis_text = "\
NAME           Rows 3 Cols 2 Iters 3
 XL x0        c0
 XL x1        c2
ENDATA
";
        let basis_path = super::native_lp_solve_temp_path("test-basis", "bas");
        std::fs::write(&basis_path, basis_text).unwrap();
        let fields =
            super::parse_native_lp_solve_lp_certificate_fields(stdout, &basis_path, 2, 3, 0);
        let _ = std::fs::remove_file(&basis_path);

        assert_eq!(fields.dual_ub, Some(vec![2.333333, 0.0, 0.6666667]));
        assert_eq!(fields.dual_eq, Some(Vec::new()));
        assert_eq!(fields.reduced_costs, Some(vec![0.0, 0.0]));
        assert_eq!(
            fields.var_basis,
            Some(vec!["basic".to_string(), "basic".to_string()])
        );
        assert_eq!(
            fields.row_basis,
            Some(vec![
                "at_upper".to_string(),
                "basic".to_string(),
                "at_upper".to_string()
            ])
        );
    }

    #[test]
    fn native_lp_solve_certificate_parser_reads_duals_and_reduced_costs() {
        let stdout = "\
c0                              2.5
e0                             -1.0
x0                              4 0.25
x1                              0 -0.5
";
        let fields = super::parse_native_lp_solve_lp_certificate_fields(
            stdout,
            std::path::Path::new("/tmp/des-rs-missing-lp-solve-test.bas"),
            2,
            1,
            1,
        );
        assert_eq!(fields.dual_ub, Some(vec![2.5]));
        assert_eq!(fields.dual_eq, Some(vec![-1.0]));
        assert_eq!(fields.reduced_costs, Some(vec![0.25, -0.5]));
        assert_eq!(fields.var_basis, None);
        assert_eq!(fields.row_basis, None);
    }

    #[test]
    fn native_lp_solve_lp_certificate_parser_reads_equality_sensitivity_output() {
        let stdout = "\
Value of objective function: 2.00000000

Actual values of the variables:
x0                              2

Actual values of the constraints:
e0                              2

Objective function limits:
                                 From            Till       FromValue
x0                             -1e+30           1e+30          -1e+30

Dual values with from - till limits:
                           Dual value            From            Till
e0                                  1               0           1e+30
x0                                  0          -1e+30           1e+30
";
        let basis_text = "\
NAME           Rows 1 Cols 1 Iters 1
 XL x0        e0
ENDATA
";
        let basis_path = super::native_lp_solve_temp_path("test-eq-basis", "bas");
        std::fs::write(&basis_path, basis_text).unwrap();
        let fields =
            super::parse_native_lp_solve_lp_certificate_fields(stdout, &basis_path, 1, 0, 1);
        let _ = std::fs::remove_file(&basis_path);

        assert_eq!(fields.dual_ub, Some(Vec::new()));
        assert_eq!(fields.dual_eq, Some(vec![1.0]));
        assert_eq!(fields.reduced_costs, Some(vec![0.0]));
        assert_eq!(fields.var_basis, Some(vec!["basic".to_string()]));
        assert_eq!(fields.row_basis, Some(vec!["fixed".to_string()]));
    }

    #[test]
    fn native_lp_solve_mip_quality_parser_reads_gap_nodes_and_exact_bound() {
        let stdout = "\
Feasible solution                  1 after          2 iter,         1 nodes (gap 0.0%)
Optimal solution                   1 after          2 iter,         1 nodes (gap 0.0%).
";
        let quality = super::parse_lp_solve_mip_quality(
            ExternalLinearCliKind::Mip,
            ExternalLinearCliStatus::Optimal,
            1.0,
            stdout,
            "",
            false,
        );
        assert_eq!(quality.nodes_explored, Some(1));
        assert_eq!(quality.best_bound, Some(1.0));
        assert_eq!(quality.mip_gap, Some(0.0));
        assert_eq!(quality.absolute_gap, Some(0.0));
    }

    #[test]
    fn native_lp_solve_mip_quality_parser_respects_positive_gap_and_node_suppression() {
        let stdout = "\
Optimal solution                 220 after          5 iter,         4 nodes (gap 8.3%).
";
        let quality = super::parse_lp_solve_mip_quality(
            ExternalLinearCliKind::Mip,
            ExternalLinearCliStatus::Optimal,
            220.0,
            stdout,
            "",
            true,
        );
        assert_eq!(quality.nodes_explored, None);
        assert_eq!(quality.best_bound, None);
        assert!((quality.mip_gap.unwrap() - 0.083).abs() <= 1.0e-12);
        assert_eq!(quality.absolute_gap, None);
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
    fn native_highs_plain_lp_succeeds_without_python_bridge() {
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
    }

    #[test]
    fn native_highs_plain_mip_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Highs) else {
            eprintln!("SKIP direct HiGHS MIP solve: highs command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-highs-mip-direct".to_string()),
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
            .is_some_and(|best_bound| (best_bound - 1.0).abs() <= 1.0e-8));
        assert!(solution
            .mip_gap
            .is_some_and(|mip_gap| mip_gap.abs() <= 1.0e-8));
        assert!(solution
            .absolute_gap
            .is_some_and(|absolute_gap| absolute_gap.abs() <= 1.0e-8));
        assert!(solution.nodes_explored.is_some());
    }

    #[test]
    fn native_highs_mip_start_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Highs) else {
            eprintln!("SKIP direct HiGHS MIP-start solve: highs command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-highs-mip-start".to_string()),
                time_limit_secs: Some(2.0),
                random_seed: Some(7),
                mip_start: Some(vec![0.0]),
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
        assert_eq!(solution.mip_start_accepted, Some(true));
        assert_eq!(solution.mip_start_objective, Some(0.0));
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn lower_bounded_mip_start_uses_native_highs_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Highs) else {
            eprintln!("SKIP lower-bounded HiGHS MIP-start solve: highs command not installed");
            return;
        };
        let mut problem = build_lower_bounded_production_ip();
        problem.base.integer_vars = vec![true, true];
        let solution = solve_lower_bounded_ipmip_with_external_cli(
            &problem,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-lower-bounded-mip-start".to_string()),
                time_limit_secs: Some(2.0),
                random_seed: Some(7),
                mip_start: Some(vec![3.0, 5.0]),
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
        assert_eq!(solution.x, vec![3.0, 5.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 13.0).abs() <= 1.0e-8));
        assert!(solution
            .mip_start_objective
            .is_some_and(|objective| (objective - 13.0).abs() <= 1.0e-8));
    }

    #[test]
    fn native_highs_solution_limit_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Highs) else {
            eprintln!("SKIP direct HiGHS solution-limit solve: highs command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-highs-solution-limit".to_string()),
                time_limit_secs: Some(2.0),
                random_seed: Some(7),
                solution_limit: Some(1),
                ..Default::default()
            },
        );
        assert!(
            matches!(
                solution.status,
                ExternalLinearCliStatus::Optimal | ExternalLinearCliStatus::Feasible
            ),
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "highs:cli");
        assert_eq!(solution.solution_limit, Some(1));
        assert_eq!(solution.x.len(), 1);
        assert!(solution.objective.is_some());
    }

    #[test]
    fn native_glpk_plain_lp_succeeds_without_python_bridge() {
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
    }

    #[test]
    fn native_glpk_lp_algorithm_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Glpk) else {
            eprintln!("SKIP direct GLPK LP algorithm solve: glpsol command not installed");
            return;
        };
        let solution = solve_lp_with_external_cli(
            &super::external_linear_cli_smoke_lp(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Glpk,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-glpk-lp-algorithm".to_string()),
                time_limit_secs: Some(2.0),
                lp_algorithm: Some(ExternalLinearCliLpAlgorithm::Ipm),
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
        assert_eq!(solution.lp_algorithm.as_deref(), Some("ipm"));
        assert!(solution
            .x
            .first()
            .is_some_and(|value| (value - 1.0).abs() <= 1.0e-6));
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-6));
    }

    #[test]
    fn native_highs_json_plain_lp_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Highs) else {
            eprintln!("SKIP direct HiGHS JSON LP solve: highs command not installed");
            return;
        };
        let payload = lp_problem_to_cli_json(&super::external_linear_cli_smoke_lp());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Lp,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-highs-json-direct".to_string()),
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
    }

    #[test]
    fn native_highs_json_plain_mip_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Highs) else {
            eprintln!("SKIP direct HiGHS JSON MIP solve: highs command not installed");
            return;
        };
        let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-highs-json-mip-direct".to_string()),
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
            .is_some_and(|best_bound| (best_bound - 1.0).abs() <= 1.0e-8));
        assert!(solution
            .mip_gap
            .is_some_and(|mip_gap| mip_gap.abs() <= 1.0e-8));
    }

    #[test]
    fn native_gurobi_json_plain_lp_stays_off_python_bridge() {
        let payload = lp_problem_to_cli_json(&super::external_linear_cli_smoke_lp());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Lp,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Gurobi,
                command_path: Some(PathBuf::from("/definitely/not-a-gurobi-binary")),
                python: Some("/definitely/not-a-python-for-gurobi-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "gurobi:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_gurobi_json_plain_mip_stays_off_python_bridge() {
        let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Gurobi,
                command_path: Some(PathBuf::from("/definitely/not-a-gurobi-binary")),
                python: Some("/definitely/not-a-python-for-gurobi-mip-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "gurobi:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_cplex_json_plain_lp_stays_off_python_bridge() {
        let payload = lp_problem_to_cli_json(&super::external_linear_cli_smoke_lp());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Lp,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Cplex,
                command_path: Some(PathBuf::from("/definitely/not-a-cplex-binary")),
                python: Some("/definitely/not-a-python-for-cplex-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "cplex:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_cplex_json_plain_mip_stays_off_python_bridge() {
        let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Cplex,
                command_path: Some(PathBuf::from("/definitely/not-a-cplex-binary")),
                python: Some("/definitely/not-a-python-for-cplex-mip-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "cplex:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_xpress_json_plain_lp_stays_off_python_bridge() {
        let payload = lp_problem_to_cli_json(&super::external_linear_cli_smoke_lp());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Lp,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Xpress,
                command_path: Some(PathBuf::from("/definitely/not-an-xpress-binary")),
                python: Some("/definitely/not-a-python-for-xpress-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "xpress:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_xpress_json_plain_mip_stays_off_python_bridge() {
        let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Xpress,
                command_path: Some(PathBuf::from("/definitely/not-an-xpress-binary")),
                python: Some("/definitely/not-a-python-for-xpress-mip-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "xpress:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_lindo_json_plain_lp_stays_off_python_bridge() {
        let payload = lp_problem_to_cli_json(&super::external_linear_cli_smoke_lp());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Lp,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Lindo,
                command_path: Some(PathBuf::from("/definitely/not-a-lindo-binary")),
                python: Some("/definitely/not-a-python-for-lindo-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "lindo:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_lindo_json_plain_mip_stays_off_python_bridge() {
        let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Lindo,
                command_path: Some(PathBuf::from("/definitely/not-a-lindo-binary")),
                python: Some("/definitely/not-a-python-for-lindo-mip-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "lindo:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_glpk_json_plain_mip_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Glpk) else {
            eprintln!("SKIP direct GLPK JSON MIP solve: glpsol command not installed");
            return;
        };
        let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Glpk,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-glpk-json-direct".to_string()),
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
    }

    #[test]
    fn native_glpk_mip_controls_succeed_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Glpk) else {
            eprintln!("SKIP direct GLPK MIP-control solve: glpsol command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Glpk,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-glpk-mip-controls".to_string()),
                time_limit_secs: Some(2.0),
                relative_gap: Some(0.25),
                threads: Some(1),
                random_seed: Some(7),
                presolve: Some(ExternalLinearCliPresolve::Off),
                branch_rule: Some(ExternalLinearCliBranchRule::FirstFractional),
                node_selection: Some(ExternalLinearCliNodeSelection::Dfs),
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
        assert_eq!(solution.random_seed, Some(7));
        assert_eq!(solution.presolve.as_deref(), Some("off"));
        assert_eq!(solution.branch_rule.as_deref(), Some("first-fractional"));
        assert_eq!(solution.node_selection.as_deref(), Some("dfs"));
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn native_glpk_mip_cuts_succeed_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Glpk) else {
            eprintln!("SKIP direct GLPK MIP-cut solve: glpsol command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Glpk,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-glpk-mip-cuts".to_string()),
                time_limit_secs: Some(2.0),
                cuts: Some(ExternalLinearCliMipSwitch::On),
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
        assert_eq!(solution.cuts.as_deref(), Some("on"));
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn solution_pool_no_good_cut_handles_general_integer_domains() {
        let mut model = super::PlainLinearCliModel {
            sense: Sense::Max,
            c: vec![1.0],
            le_rows: Vec::new(),
            le_rhs: Vec::new(),
            eq_rows: vec![vec![1.0]],
            eq_rhs: vec![1.0],
            lbs: vec![Some(0.0)],
            ubs: vec![Some(2.0)],
            integer_vars: vec![true],
        };
        let integer_indices = super::solution_pool_integer_indices(&model.integer_vars);
        assert_eq!(
            super::validate_solution_pool_bounds(&model.lbs, &model.ubs, &integer_indices),
            None
        );
        assert_eq!(
            super::add_solution_pool_no_good_cut(&mut model, &integer_indices, &[1.0]),
            None
        );
        assert_eq!(model.c, vec![1.0, 0.0, 0.0]);
        assert_eq!(model.lbs, vec![Some(0.0), Some(0.0), Some(0.0)]);
        assert_eq!(model.ubs, vec![Some(2.0), Some(1.0), Some(1.0)]);
        assert_eq!(model.integer_vars, vec![true, true, true]);
        assert_eq!(model.eq_rows, vec![vec![1.0, 0.0, 0.0]]);
        assert_eq!(
            model.le_rows,
            vec![
                vec![1.0, 2.0, 0.0],
                vec![-1.0, 0.0, 2.0],
                vec![0.0, -1.0, -1.0]
            ]
        );
        assert_eq!(model.le_rhs, vec![2.0, -0.0, -1.0]);
    }

    fn native_solution_pool_smoke_problem() -> IPMIPProblem {
        IPMIPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![1.0],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        }
    }

    fn assert_native_solution_pool(
        solution: &crate::des::general::external_linear_cli::ExternalLinearCliSolution,
        solver: &str,
    ) {
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, format!("{solver}:cli"));
        assert_eq!(solution.solution_pool_size, Some(3));
        assert_eq!(solution.exhausted, Some(false));
        assert_eq!(solution.message, "pool reached solution_pool_size");
        assert_eq!(solution.x, vec![1.0, 0.0]);
        assert_eq!(solution.objective, Some(3.0));
        let solutions = solution.solutions.as_deref().unwrap_or(&[]);
        assert_eq!(solutions.len(), 3);
        let expected = [
            (vec![1.0, 0.0], 3.0),
            (vec![0.0, 1.0], 2.0),
            (vec![0.0, 0.0], 0.0),
        ];
        for (member, (expected_x, expected_objective)) in solutions.iter().zip(expected) {
            assert_eq!(member.x, expected_x);
            assert!((member.objective - expected_objective).abs() <= 1.0e-8);
        }
    }

    #[test]
    fn native_scip_solution_pool_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Scip) else {
            eprintln!("SKIP direct SCIP solution-pool solve: scip command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &native_solution_pool_smoke_problem(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Scip,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-scip-solution-pool".to_string()),
                time_limit_secs: Some(5.0),
                solution_pool_size: Some(3),
                branch_priorities: Some(vec![3, 2]),
                ..Default::default()
            },
        );
        assert_native_solution_pool(&solution, "scip");
        assert_eq!(solution.branch_priorities_accepted, Some(true));
        assert_eq!(solution.branch_priority_count, Some(2));
    }

    #[test]
    fn native_scip_json_plain_mip_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Scip) else {
            eprintln!("SKIP direct SCIP JSON MIP solve: scip command not installed");
            return;
        };
        let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Scip,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-scip-json-direct".to_string()),
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
        assert_eq!(solution.solver, "scip:cli");
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn native_scip_mip_start_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Scip) else {
            eprintln!("SKIP direct SCIP MIP-start solve: scip command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Scip,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-scip-mip-start".to_string()),
                time_limit_secs: Some(2.0),
                random_seed: Some(7),
                mip_start: Some(vec![0.0]),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "scip:cli");
        assert_eq!(solution.mip_start_accepted, Some(true));
        assert_eq!(solution.mip_start_objective, Some(0.0));
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn native_scip_branch_priorities_succeed_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Scip) else {
            eprintln!("SKIP direct SCIP branch-priority solve: scip command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Scip,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-scip-branch-priority".to_string()),
                time_limit_secs: Some(2.0),
                random_seed: Some(7),
                branch_priorities: Some(vec![5]),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "scip:cli");
        assert_eq!(solution.branch_priorities_accepted, Some(true));
        assert_eq!(solution.branch_priority_count, Some(1));
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn native_scip_json_unused_binary_mip_reports_infeasible() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Scip) else {
            eprintln!("SKIP direct SCIP unused-binary MIP solve: scip command not installed");
            return;
        };
        let problem = IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.0, 0.0, 0.0],
            a: vec![
                vec![-1.0, 0.0, 0.0],
                vec![0.0, -1.0, 0.0],
                vec![1.0, 1.0, 0.0],
            ],
            b: vec![-1.0, -1.0, 1.0],
            integer_vars: vec![true, true, true],
            ub: Some(vec![1.0, 1.0, 1.0]),
            var_names: None,
            con_names: None,
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            ipmip_problem_to_cli_json(&problem),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Scip,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-scip-unused-binary".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Infeasible,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "scip:cli");
    }

    #[test]
    fn native_cbc_json_plain_mip_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Cbc) else {
            eprintln!("SKIP direct CBC JSON MIP solve: cbc command not installed");
            return;
        };
        let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Cbc,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-cbc-json-direct".to_string()),
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
        assert_eq!(solution.solver, "cbc:cli");
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn native_cbc_solution_pool_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Cbc) else {
            eprintln!("SKIP direct CBC solution-pool solve: cbc command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &native_solution_pool_smoke_problem(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Cbc,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-cbc-solution-pool".to_string()),
                time_limit_secs: Some(5.0),
                solution_pool_size: Some(3),
                branch_priorities: Some(vec![3, 2]),
                ..Default::default()
            },
        );
        assert_native_solution_pool(&solution, "cbc");
        assert_eq!(solution.branch_priorities_accepted, Some(true));
        assert_eq!(solution.branch_priority_count, Some(2));
    }

    #[test]
    fn native_cbc_mip_start_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Cbc) else {
            eprintln!("SKIP direct CBC MIP-start solve: cbc command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Cbc,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-cbc-mip-start".to_string()),
                time_limit_secs: Some(2.0),
                mip_start: Some(vec![0.0]),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "cbc:cli");
        assert_eq!(solution.mip_start_accepted, Some(true));
        assert_eq!(solution.mip_start_objective, Some(0.0));
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn native_cbc_branch_priorities_succeed_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Cbc) else {
            eprintln!("SKIP direct CBC branch-priority solve: cbc command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Cbc,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-cbc-branch-priority".to_string()),
                time_limit_secs: Some(2.0),
                branch_priorities: Some(vec![5]),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "cbc:cli");
        assert_eq!(solution.branch_priorities_accepted, Some(true));
        assert_eq!(solution.branch_priority_count, Some(1));
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn native_cbc_mip_controls_succeed_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Cbc) else {
            eprintln!("SKIP direct CBC MIP-control solve: cbc command not installed");
            return;
        };
        let solution = solve_ipmip_with_external_cli(
            &super::external_linear_cli_smoke_mip(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Cbc,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-cbc-mip-controls".to_string()),
                time_limit_secs: Some(2.0),
                relative_gap: Some(0.25),
                threads: Some(1),
                random_seed: Some(7),
                presolve: Some(ExternalLinearCliPresolve::Off),
                primal_feasibility_tolerance: Some(1e-7),
                dual_feasibility_tolerance: Some(2e-7),
                integer_feasibility_tolerance: Some(1e-6),
                cuts: Some(ExternalLinearCliMipSwitch::Off),
                heuristics: Some(ExternalLinearCliMipSwitch::Off),
                node_selection: Some(ExternalLinearCliNodeSelection::Dfs),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "cbc:cli");
        assert_eq!(solution.threads, Some(1));
        assert_eq!(solution.random_seed, Some(7));
        assert_eq!(solution.presolve.as_deref(), Some("off"));
        assert_eq!(solution.cuts.as_deref(), Some("off"));
        assert_eq!(solution.heuristics.as_deref(), Some("off"));
        assert_eq!(solution.node_selection.as_deref(), Some("dfs"));
        assert!(solution
            .primal_feasibility_tolerance
            .is_some_and(|tol| (tol - 1e-7).abs() <= 1e-12));
        assert!(solution
            .dual_feasibility_tolerance
            .is_some_and(|tol| (tol - 2e-7).abs() <= 1e-12));
        assert!(solution
            .integer_feasibility_tolerance
            .is_some_and(|tol| (tol - 1e-6).abs() <= 1e-12));
        assert_eq!(solution.x, vec![1.0]);
    }

    #[test]
    fn native_cbc_lp_tolerances_succeed_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Cbc) else {
            eprintln!("SKIP direct CBC LP-tolerance solve: cbc command not installed");
            return;
        };
        let solution = solve_lp_with_external_cli(
            &super::external_linear_cli_smoke_lp(),
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Cbc,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-cbc-lp-tolerances".to_string()),
                time_limit_secs: Some(2.0),
                primal_feasibility_tolerance: Some(1e-7),
                dual_feasibility_tolerance: Some(2e-7),
                ..Default::default()
            },
        );
        assert_eq!(
            solution.status,
            ExternalLinearCliStatus::Optimal,
            "{}",
            solution.message
        );
        assert_eq!(solution.solver, "cbc:cli");
        assert!(solution
            .primal_feasibility_tolerance
            .is_some_and(|tol| (tol - 1e-7).abs() <= 1e-12));
        assert!(solution
            .dual_feasibility_tolerance
            .is_some_and(|tol| (tol - 2e-7).abs() <= 1e-12));
        assert_eq!(solution.x, vec![1.0]);
    }

    #[test]
    fn native_clp_json_plain_lp_succeeds_without_python_bridge() {
        let Some(command) = external_linear_cli_command(ExternalLinearCliSolver::Clp) else {
            eprintln!("SKIP direct CLP JSON LP solve: clp command not installed");
            return;
        };
        let payload = lp_problem_to_cli_json(&super::external_linear_cli_smoke_lp());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Lp,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Clp,
                command_path: Some(command),
                python: Some("/definitely/not-a-python-for-clp-json-direct".to_string()),
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
        assert_eq!(solution.solver, "clp:cli");
        assert_eq!(solution.x, vec![1.0]);
        assert!(solution
            .objective
            .is_some_and(|objective| (objective - 1.0).abs() <= 1.0e-8));
    }

    #[test]
    fn native_soplex_json_plain_lp_stays_off_python_bridge() {
        let payload = lp_problem_to_cli_json(&super::external_linear_cli_smoke_lp());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Lp,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Soplex,
                command_path: Some(PathBuf::from("/definitely/not-a-soplex-binary")),
                python: Some("/definitely/not-a-python-for-soplex-json-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "soplex:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_qsoptex_json_plain_lp_stays_off_python_bridge() {
        let payload = lp_problem_to_cli_json(&super::external_linear_cli_smoke_lp());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Lp,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::QsoptEx,
                command_path: Some(PathBuf::from("/definitely/not-a-qsopt-ex-binary")),
                python: Some("/definitely/not-a-python-for-qsoptex-json-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "qsopt-ex:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_lpsolve_json_plain_mip_stays_off_python_bridge() {
        let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::LpSolve,
                command_path: Some(PathBuf::from("/definitely/not-an-lp-solve-binary")),
                python: Some("/definitely/not-a-python-for-lpsolve-json-direct".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "lp-solve:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "{}",
            solution.message
        );
    }

    #[test]
    fn native_open_source_json_auto_mip_controls_stay_off_python_bridge() {
        for (solver, command_path) in [
            (
                ExternalLinearCliSolver::Highs,
                "/definitely/not-a-highs-binary",
            ),
            (
                ExternalLinearCliSolver::Glpk,
                "/definitely/not-a-glpsol-binary",
            ),
            (
                ExternalLinearCliSolver::Scip,
                "/definitely/not-a-scip-binary",
            ),
            (ExternalLinearCliSolver::Cbc, "/definitely/not-a-cbc-binary"),
        ] {
            let payload = ipmip_problem_to_cli_json(&super::external_linear_cli_smoke_mip());
            let solution = super::solve_linear_cli_json(
                ExternalLinearCliKind::Mip,
                payload,
                &ExternalLinearCliOptions {
                    solver,
                    command_path: Some(PathBuf::from(command_path)),
                    python: Some(format!(
                        "/definitely/not-a-python-for-{}-auto-controls",
                        solver.as_str()
                    )),
                    time_limit_secs: Some(2.0),
                    presolve: Some(ExternalLinearCliPresolve::Auto),
                    cuts: Some(ExternalLinearCliMipSwitch::Auto),
                    heuristics: Some(ExternalLinearCliMipSwitch::Auto),
                    ..Default::default()
                },
            );
            assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
            assert_eq!(solution.solver, format!("{}:cli", solver.as_str()));
            assert!(
                !solution.message.to_ascii_lowercase().contains("python"),
                "{} auto controls unexpectedly used Python bridge: {}",
                solver.as_str(),
                solution.message
            );
        }
    }

    #[test]
    fn native_open_source_json_auto_lp_controls_stay_off_python_bridge() {
        for (solver, command_path) in [
            (
                ExternalLinearCliSolver::Highs,
                "/definitely/not-a-highs-binary",
            ),
            (
                ExternalLinearCliSolver::Glpk,
                "/definitely/not-a-glpsol-binary",
            ),
            (
                ExternalLinearCliSolver::Scip,
                "/definitely/not-a-scip-binary",
            ),
            (ExternalLinearCliSolver::Cbc, "/definitely/not-a-cbc-binary"),
        ] {
            let payload = lp_problem_to_cli_json(&super::external_linear_cli_smoke_lp());
            let solution = super::solve_linear_cli_json(
                ExternalLinearCliKind::Lp,
                payload,
                &ExternalLinearCliOptions {
                    solver,
                    command_path: Some(PathBuf::from(command_path)),
                    python: Some(format!(
                        "/definitely/not-a-python-for-{}-auto-lp-controls",
                        solver.as_str()
                    )),
                    time_limit_secs: Some(2.0),
                    presolve: Some(ExternalLinearCliPresolve::Auto),
                    cuts: Some(ExternalLinearCliMipSwitch::Auto),
                    heuristics: Some(ExternalLinearCliMipSwitch::Auto),
                    ..Default::default()
                },
            );
            assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
            assert_eq!(solution.solver, format!("{}:cli", solver.as_str()));
            assert!(
                !solution.message.to_ascii_lowercase().contains("python"),
                "{} LP auto controls unexpectedly used Python bridge: {}",
                solver.as_str(),
                solution.message
            );
        }
    }

    #[test]
    fn source_feature_model_uses_rust_linearization_with_python_override() {
        let problem = build_source_feature_mix_ip();
        let solution = solve_source_ipmip_with_external_cli(
            &problem,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(PathBuf::from("/definitely/not-a-highs-source-binary")),
                python: Some("/definitely/not-a-python-for-source-linearization".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );

        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "highs:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "source linearization unexpectedly used Python bridge: {}",
            solution.message
        );
    }

    #[test]
    fn source_feature_json_uses_rust_linearization_without_python_bridge() {
        let payload = source_ipmip_problem_to_cli_json(&build_source_feature_mix_ip());
        let solution = super::solve_linear_cli_json(
            ExternalLinearCliKind::Mip,
            payload,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                command_path: Some(PathBuf::from("/definitely/not-a-highs-source-json-binary")),
                python: Some("/definitely/not-a-python-for-source-json-linearization".to_string()),
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );

        assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
        assert_eq!(solution.solver, "highs:cli");
        assert!(
            !solution.message.to_ascii_lowercase().contains("python"),
            "source JSON linearization unexpectedly used Python bridge: {}",
            solution.message
        );
    }

    #[test]
    fn source_feature_json_branch_priorities_use_rust_linearization_without_python_bridge() {
        for (solver, command_path) in [
            (
                ExternalLinearCliSolver::Scip,
                "/definitely/not-a-scip-source-json-priority-binary",
            ),
            (
                ExternalLinearCliSolver::Cbc,
                "/definitely/not-a-cbc-source-json-priority-binary",
            ),
        ] {
            let problem = build_source_feature_mix_ip();
            let priority_count = problem.base.c.len();
            let payload = source_ipmip_problem_to_cli_json(&problem);
            let solution = super::solve_linear_cli_json(
                ExternalLinearCliKind::Mip,
                payload,
                &ExternalLinearCliOptions {
                    solver,
                    command_path: Some(PathBuf::from(command_path)),
                    python: Some(format!(
                        "/definitely/not-a-python-for-{}-source-json-priorities",
                        solver.as_str()
                    )),
                    time_limit_secs: Some(2.0),
                    branch_priorities: Some(vec![5; priority_count]),
                    ..Default::default()
                },
            );

            assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
            assert_eq!(solution.solver, format!("{}:cli", solver.as_str()));
            assert!(
                !solution.message.to_ascii_lowercase().contains("python"),
                "{} source JSON priorities unexpectedly used Python bridge: {}",
                solver.as_str(),
                solution.message
            );
        }
    }

    #[test]
    fn source_feature_json_mip_start_uses_rust_linearization_without_python_bridge() {
        for (solver, command_path) in [
            (
                ExternalLinearCliSolver::Highs,
                "/definitely/not-a-highs-source-json-mip-start-binary",
            ),
            (
                ExternalLinearCliSolver::Scip,
                "/definitely/not-a-scip-source-json-mip-start-binary",
            ),
            (
                ExternalLinearCliSolver::Cbc,
                "/definitely/not-a-cbc-source-json-mip-start-binary",
            ),
        ] {
            let problem = build_source_feature_mix_ip();
            let payload = source_ipmip_problem_to_cli_json(&problem);
            let solution = super::solve_linear_cli_json(
                ExternalLinearCliKind::Mip,
                payload,
                &ExternalLinearCliOptions {
                    solver,
                    command_path: Some(PathBuf::from(command_path)),
                    python: Some(format!(
                        "/definitely/not-a-python-for-{}-source-json-mip-start",
                        solver.as_str()
                    )),
                    time_limit_secs: Some(2.0),
                    mip_start: Some(vec![0.0; problem.base.c.len()]),
                    ..Default::default()
                },
            );

            assert_eq!(solution.status, ExternalLinearCliStatus::Unavailable);
            assert_eq!(solution.solver, format!("{}:cli", solver.as_str()));
            assert!(
                !solution.message.to_ascii_lowercase().contains("python"),
                "{} source JSON MIP start unexpectedly used Python bridge: {}",
                solver.as_str(),
                solution.message
            );
        }
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
