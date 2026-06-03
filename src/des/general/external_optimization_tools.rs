//! Local adapter/probe surface for optimization ecosystems that are not plain
//! LP/MIP command-line solvers.
//!
//! Java CP/planning systems and Rust modeling/binding crates are usually wired
//! into an application through a small local wrapper. This module gives those
//! wrappers a stable JSON-in/JSON-out contract while keeping jars, native
//! libraries, and generated executables out of version control.

use super::external_linear_cli::{
    probe_external_linear_cli_solver, ExternalLinearCliKind, ExternalLinearCliOptions,
    ExternalLinearCliProbeStatus, ExternalLinearCliSolver,
};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalOptimizationTool {
    ChocoSolver,
    Jacop,
    IbmCpOptimizer,
    OptaPlanner,
    Timefold,
    JMetal,
    MoeaFramework,
    Ecj,
    OjAlgo,
    OrToolsJava,
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
    Pyomo,
    Pulp,
    Cvxpy,
    Cvxopt,
    PyScipOpt,
    PythonMip,
    GurobiPy,
    Docplex,
    OrToolsPython,
    ScipyOptimize,
    Jump,
    Ampl,
    Gams,
    Hexaly,
    Minotaur,
    Symphony,
    Ipopt,
    Bonmin,
    Couenne,
    Knitro,
    Mosek,
    Baron,
    Copt,
    Casadi,
    Osqp,
    Scs,
    Clarabel,
    Ecos,
    Qpoases,
    Proxqp,
    Cosmo,
    Sdpa,
    Csdp,
    HighsCli,
    GlpkCli,
    ScipCli,
    CbcCli,
    ClpCli,
    GurobiCli,
    CplexCli,
    XpressCli,
    LindoCli,
    GoodLp,
    LpModeler,
    RustLinprog,
    Argmin,
    Nlopt,
    HighsRust,
    ScipRust,
    CbcRust,
}

impl ExternalOptimizationTool {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationTool::ChocoSolver => "choco-solver",
            ExternalOptimizationTool::Jacop => "jacop",
            ExternalOptimizationTool::IbmCpOptimizer => "ibm-cp-optimizer",
            ExternalOptimizationTool::OptaPlanner => "optaplanner",
            ExternalOptimizationTool::Timefold => "timefold",
            ExternalOptimizationTool::JMetal => "jmetal",
            ExternalOptimizationTool::MoeaFramework => "moea-framework",
            ExternalOptimizationTool::Ecj => "ecj",
            ExternalOptimizationTool::OjAlgo => "ojalgo",
            ExternalOptimizationTool::OrToolsJava => "ortools-java",
            ExternalOptimizationTool::Cpmpy => "cpmpy",
            ExternalOptimizationTool::PyCsp3 => "pycsp3",
            ExternalOptimizationTool::Conjure => "conjure",
            ExternalOptimizationTool::SavileRow => "savile-row",
            ExternalOptimizationTool::Picat => "picat",
            ExternalOptimizationTool::Clingo => "clingo",
            ExternalOptimizationTool::Clingcon => "clingcon",
            ExternalOptimizationTool::Sat4j => "sat4j",
            ExternalOptimizationTool::PySat => "pysat",
            ExternalOptimizationTool::OpenWbo => "open-wbo",
            ExternalOptimizationTool::Pyomo => "pyomo",
            ExternalOptimizationTool::Pulp => "pulp",
            ExternalOptimizationTool::Cvxpy => "cvxpy",
            ExternalOptimizationTool::Cvxopt => "cvxopt",
            ExternalOptimizationTool::PyScipOpt => "pyscipopt",
            ExternalOptimizationTool::PythonMip => "python-mip",
            ExternalOptimizationTool::GurobiPy => "gurobipy",
            ExternalOptimizationTool::Docplex => "docplex",
            ExternalOptimizationTool::OrToolsPython => "ortools-python",
            ExternalOptimizationTool::ScipyOptimize => "scipy-optimize",
            ExternalOptimizationTool::Jump => "jump",
            ExternalOptimizationTool::Ampl => "ampl",
            ExternalOptimizationTool::Gams => "gams",
            ExternalOptimizationTool::Hexaly => "hexaly",
            ExternalOptimizationTool::Minotaur => "minotaur",
            ExternalOptimizationTool::Symphony => "symphony",
            ExternalOptimizationTool::Ipopt => "ipopt",
            ExternalOptimizationTool::Bonmin => "bonmin",
            ExternalOptimizationTool::Couenne => "couenne",
            ExternalOptimizationTool::Knitro => "knitro",
            ExternalOptimizationTool::Mosek => "mosek",
            ExternalOptimizationTool::Baron => "baron",
            ExternalOptimizationTool::Copt => "copt",
            ExternalOptimizationTool::Casadi => "casadi",
            ExternalOptimizationTool::Osqp => "osqp",
            ExternalOptimizationTool::Scs => "scs",
            ExternalOptimizationTool::Clarabel => "clarabel",
            ExternalOptimizationTool::Ecos => "ecos",
            ExternalOptimizationTool::Qpoases => "qpoases",
            ExternalOptimizationTool::Proxqp => "proxqp",
            ExternalOptimizationTool::Cosmo => "cosmo",
            ExternalOptimizationTool::Sdpa => "sdpa",
            ExternalOptimizationTool::Csdp => "csdp",
            ExternalOptimizationTool::HighsCli => "highs-cli",
            ExternalOptimizationTool::GlpkCli => "glpk-cli",
            ExternalOptimizationTool::ScipCli => "scip-cli",
            ExternalOptimizationTool::CbcCli => "cbc-cli",
            ExternalOptimizationTool::ClpCli => "clp-cli",
            ExternalOptimizationTool::GurobiCli => "gurobi-cli",
            ExternalOptimizationTool::CplexCli => "cplex-cli",
            ExternalOptimizationTool::XpressCli => "xpress-cli",
            ExternalOptimizationTool::LindoCli => "lindo-cli",
            ExternalOptimizationTool::GoodLp => "good-lp",
            ExternalOptimizationTool::LpModeler => "lp-modeler",
            ExternalOptimizationTool::RustLinprog => "rust-linprog",
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::Nlopt => "nlopt",
            ExternalOptimizationTool::HighsRust => "highs-rust",
            ExternalOptimizationTool::ScipRust => "scip-rust",
            ExternalOptimizationTool::CbcRust => "cbc-rust",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ExternalOptimizationTool::ChocoSolver => "Choco Solver",
            ExternalOptimizationTool::Jacop => "JaCoP",
            ExternalOptimizationTool::IbmCpOptimizer => "IBM ILOG CP Optimizer",
            ExternalOptimizationTool::OptaPlanner => "OptaPlanner",
            ExternalOptimizationTool::Timefold => "Timefold Solver",
            ExternalOptimizationTool::JMetal => "jMetal",
            ExternalOptimizationTool::MoeaFramework => "MOEA Framework",
            ExternalOptimizationTool::Ecj => "ECJ",
            ExternalOptimizationTool::OjAlgo => "ojAlgo",
            ExternalOptimizationTool::OrToolsJava => "Google OR-Tools Java",
            ExternalOptimizationTool::Cpmpy => "CPMpy",
            ExternalOptimizationTool::PyCsp3 => "PyCSP3",
            ExternalOptimizationTool::Conjure => "Conjure",
            ExternalOptimizationTool::SavileRow => "Savile Row",
            ExternalOptimizationTool::Picat => "Picat",
            ExternalOptimizationTool::Clingo => "clingo",
            ExternalOptimizationTool::Clingcon => "clingcon",
            ExternalOptimizationTool::Sat4j => "SAT4J",
            ExternalOptimizationTool::PySat => "PySAT",
            ExternalOptimizationTool::OpenWbo => "Open-WBO",
            ExternalOptimizationTool::Pyomo => "Pyomo",
            ExternalOptimizationTool::Pulp => "PuLP",
            ExternalOptimizationTool::Cvxpy => "CVXPY",
            ExternalOptimizationTool::Cvxopt => "CVXOPT",
            ExternalOptimizationTool::PyScipOpt => "PySCIPOpt",
            ExternalOptimizationTool::PythonMip => "Python-MIP",
            ExternalOptimizationTool::GurobiPy => "gurobipy",
            ExternalOptimizationTool::Docplex => "DOcplex",
            ExternalOptimizationTool::OrToolsPython => "Google OR-Tools Python",
            ExternalOptimizationTool::ScipyOptimize => "SciPy optimize",
            ExternalOptimizationTool::Jump => "JuMP",
            ExternalOptimizationTool::Ampl => "AMPL",
            ExternalOptimizationTool::Gams => "GAMS",
            ExternalOptimizationTool::Hexaly => "Hexaly Optimizer",
            ExternalOptimizationTool::Minotaur => "MINOTAUR",
            ExternalOptimizationTool::Symphony => "COIN-OR SYMPHONY",
            ExternalOptimizationTool::Ipopt => "Ipopt",
            ExternalOptimizationTool::Bonmin => "Bonmin",
            ExternalOptimizationTool::Couenne => "Couenne",
            ExternalOptimizationTool::Knitro => "Artelys Knitro",
            ExternalOptimizationTool::Mosek => "MOSEK",
            ExternalOptimizationTool::Baron => "BARON",
            ExternalOptimizationTool::Copt => "COPT",
            ExternalOptimizationTool::Casadi => "CasADi",
            ExternalOptimizationTool::Osqp => "OSQP",
            ExternalOptimizationTool::Scs => "SCS",
            ExternalOptimizationTool::Clarabel => "Clarabel",
            ExternalOptimizationTool::Ecos => "ECOS",
            ExternalOptimizationTool::Qpoases => "qpOASES",
            ExternalOptimizationTool::Proxqp => "ProxQP",
            ExternalOptimizationTool::Cosmo => "COSMO",
            ExternalOptimizationTool::Sdpa => "SDPA",
            ExternalOptimizationTool::Csdp => "CSDP",
            ExternalOptimizationTool::HighsCli => "HiGHS CLI",
            ExternalOptimizationTool::GlpkCli => "GLPK glpsol CLI",
            ExternalOptimizationTool::ScipCli => "SCIP CLI",
            ExternalOptimizationTool::CbcCli => "COIN-OR CBC CLI",
            ExternalOptimizationTool::ClpCli => "COIN-OR CLP CLI",
            ExternalOptimizationTool::GurobiCli => "Gurobi Optimizer CLI",
            ExternalOptimizationTool::CplexCli => "IBM ILOG CPLEX Optimizer CLI",
            ExternalOptimizationTool::XpressCli => "FICO Xpress Optimizer CLI",
            ExternalOptimizationTool::LindoCli => "LINDO Systems CLI",
            ExternalOptimizationTool::GoodLp => "good_lp",
            ExternalOptimizationTool::LpModeler => "lp-modeler",
            ExternalOptimizationTool::RustLinprog => "rust-linprog",
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::Nlopt => "NLopt Rust bindings",
            ExternalOptimizationTool::HighsRust => "HiGHS Rust bindings",
            ExternalOptimizationTool::ScipRust => "SCIP Rust bindings",
            ExternalOptimizationTool::CbcRust => "CBC Rust bindings",
        }
    }

    pub fn language(self) -> ExternalOptimizationLanguage {
        match self {
            ExternalOptimizationTool::ChocoSolver
            | ExternalOptimizationTool::Jacop
            | ExternalOptimizationTool::IbmCpOptimizer
            | ExternalOptimizationTool::OptaPlanner
            | ExternalOptimizationTool::Timefold
            | ExternalOptimizationTool::JMetal
            | ExternalOptimizationTool::MoeaFramework
            | ExternalOptimizationTool::Ecj
            | ExternalOptimizationTool::OjAlgo
            | ExternalOptimizationTool::OrToolsJava
            | ExternalOptimizationTool::SavileRow
            | ExternalOptimizationTool::Sat4j => ExternalOptimizationLanguage::Java,
            ExternalOptimizationTool::Pyomo
            | ExternalOptimizationTool::Cpmpy
            | ExternalOptimizationTool::PyCsp3
            | ExternalOptimizationTool::PySat
            | ExternalOptimizationTool::Pulp
            | ExternalOptimizationTool::Cvxpy
            | ExternalOptimizationTool::Cvxopt
            | ExternalOptimizationTool::PyScipOpt
            | ExternalOptimizationTool::PythonMip
            | ExternalOptimizationTool::GurobiPy
            | ExternalOptimizationTool::Docplex
            | ExternalOptimizationTool::OrToolsPython
            | ExternalOptimizationTool::ScipyOptimize
            | ExternalOptimizationTool::Casadi
            | ExternalOptimizationTool::Osqp
            | ExternalOptimizationTool::Scs
            | ExternalOptimizationTool::Clarabel
            | ExternalOptimizationTool::Ecos => ExternalOptimizationLanguage::Python,
            ExternalOptimizationTool::Jump => ExternalOptimizationLanguage::Julia,
            ExternalOptimizationTool::Ampl
            | ExternalOptimizationTool::Gams
            | ExternalOptimizationTool::Hexaly
            | ExternalOptimizationTool::Minotaur
            | ExternalOptimizationTool::Symphony
            | ExternalOptimizationTool::Ipopt
            | ExternalOptimizationTool::Bonmin
            | ExternalOptimizationTool::Couenne
            | ExternalOptimizationTool::Knitro
            | ExternalOptimizationTool::Mosek
            | ExternalOptimizationTool::Baron
            | ExternalOptimizationTool::Copt
            | ExternalOptimizationTool::Qpoases
            | ExternalOptimizationTool::Proxqp
            | ExternalOptimizationTool::Cosmo
            | ExternalOptimizationTool::Sdpa
            | ExternalOptimizationTool::Csdp
            | ExternalOptimizationTool::HighsCli
            | ExternalOptimizationTool::GlpkCli
            | ExternalOptimizationTool::ScipCli
            | ExternalOptimizationTool::CbcCli
            | ExternalOptimizationTool::ClpCli
            | ExternalOptimizationTool::GurobiCli
            | ExternalOptimizationTool::CplexCli
            | ExternalOptimizationTool::XpressCli
            | ExternalOptimizationTool::LindoCli
            | ExternalOptimizationTool::Conjure
            | ExternalOptimizationTool::Picat
            | ExternalOptimizationTool::Clingo
            | ExternalOptimizationTool::Clingcon
            | ExternalOptimizationTool::OpenWbo => ExternalOptimizationLanguage::Native,
            ExternalOptimizationTool::GoodLp
            | ExternalOptimizationTool::LpModeler
            | ExternalOptimizationTool::RustLinprog
            | ExternalOptimizationTool::Argmin
            | ExternalOptimizationTool::Nlopt
            | ExternalOptimizationTool::HighsRust
            | ExternalOptimizationTool::ScipRust
            | ExternalOptimizationTool::CbcRust => ExternalOptimizationLanguage::Rust,
        }
    }

    pub fn family(self) -> ExternalOptimizationFamily {
        match self {
            ExternalOptimizationTool::ChocoSolver
            | ExternalOptimizationTool::Jacop
            | ExternalOptimizationTool::IbmCpOptimizer
            | ExternalOptimizationTool::Cpmpy
            | ExternalOptimizationTool::PyCsp3
            | ExternalOptimizationTool::Conjure
            | ExternalOptimizationTool::SavileRow
            | ExternalOptimizationTool::Picat
            | ExternalOptimizationTool::Clingo
            | ExternalOptimizationTool::Clingcon
            | ExternalOptimizationTool::Sat4j
            | ExternalOptimizationTool::PySat
            | ExternalOptimizationTool::OpenWbo => {
                ExternalOptimizationFamily::ConstraintProgramming
            }
            ExternalOptimizationTool::OptaPlanner => {
                ExternalOptimizationFamily::PlanningMetaheuristic
            }
            ExternalOptimizationTool::Timefold => ExternalOptimizationFamily::PlanningMetaheuristic,
            ExternalOptimizationTool::JMetal
            | ExternalOptimizationTool::MoeaFramework
            | ExternalOptimizationTool::Ecj => {
                ExternalOptimizationFamily::EvolutionaryMultiObjective
            }
            ExternalOptimizationTool::OjAlgo
            | ExternalOptimizationTool::Pyomo
            | ExternalOptimizationTool::Pulp
            | ExternalOptimizationTool::PythonMip
            | ExternalOptimizationTool::Docplex
            | ExternalOptimizationTool::Jump
            | ExternalOptimizationTool::Ampl
            | ExternalOptimizationTool::Gams
            | ExternalOptimizationTool::Symphony
            | ExternalOptimizationTool::HighsCli
            | ExternalOptimizationTool::GlpkCli
            | ExternalOptimizationTool::ScipCli
            | ExternalOptimizationTool::CbcCli
            | ExternalOptimizationTool::ClpCli
            | ExternalOptimizationTool::GurobiCli
            | ExternalOptimizationTool::CplexCli
            | ExternalOptimizationTool::XpressCli
            | ExternalOptimizationTool::LindoCli
            | ExternalOptimizationTool::GoodLp
            | ExternalOptimizationTool::LpModeler
            | ExternalOptimizationTool::RustLinprog => ExternalOptimizationFamily::LinearMip,
            ExternalOptimizationTool::Cvxpy
            | ExternalOptimizationTool::Cvxopt
            | ExternalOptimizationTool::Mosek
            | ExternalOptimizationTool::Copt
            | ExternalOptimizationTool::Osqp
            | ExternalOptimizationTool::Scs
            | ExternalOptimizationTool::Clarabel
            | ExternalOptimizationTool::Ecos
            | ExternalOptimizationTool::Qpoases
            | ExternalOptimizationTool::Proxqp
            | ExternalOptimizationTool::Cosmo
            | ExternalOptimizationTool::Sdpa
            | ExternalOptimizationTool::Csdp => ExternalOptimizationFamily::ConvexOptimization,
            ExternalOptimizationTool::OrToolsJava | ExternalOptimizationTool::OrToolsPython => {
                ExternalOptimizationFamily::CpSatRouting
            }
            ExternalOptimizationTool::Hexaly => ExternalOptimizationFamily::HybridOptimization,
            ExternalOptimizationTool::Argmin
            | ExternalOptimizationTool::Nlopt
            | ExternalOptimizationTool::ScipyOptimize
            | ExternalOptimizationTool::Minotaur
            | ExternalOptimizationTool::Ipopt
            | ExternalOptimizationTool::Bonmin
            | ExternalOptimizationTool::Couenne
            | ExternalOptimizationTool::Knitro
            | ExternalOptimizationTool::Baron
            | ExternalOptimizationTool::Casadi => ExternalOptimizationFamily::NonlinearOptimization,
            ExternalOptimizationTool::HighsRust
            | ExternalOptimizationTool::ScipRust
            | ExternalOptimizationTool::CbcRust
            | ExternalOptimizationTool::PyScipOpt
            | ExternalOptimizationTool::GurobiPy => ExternalOptimizationFamily::NativeSolverBinding,
        }
    }

    pub fn exactness(self) -> ExternalOptimizationExactness {
        match self {
            ExternalOptimizationTool::ChocoSolver
            | ExternalOptimizationTool::Jacop
            | ExternalOptimizationTool::IbmCpOptimizer
            | ExternalOptimizationTool::OjAlgo
            | ExternalOptimizationTool::OrToolsJava
            | ExternalOptimizationTool::OrToolsPython
            | ExternalOptimizationTool::RustLinprog
            | ExternalOptimizationTool::HighsRust
            | ExternalOptimizationTool::ScipRust
            | ExternalOptimizationTool::CbcRust
            | ExternalOptimizationTool::Picat
            | ExternalOptimizationTool::Clingo
            | ExternalOptimizationTool::Clingcon
            | ExternalOptimizationTool::Sat4j
            | ExternalOptimizationTool::PySat
            | ExternalOptimizationTool::OpenWbo
            | ExternalOptimizationTool::HighsCli
            | ExternalOptimizationTool::GlpkCli
            | ExternalOptimizationTool::ScipCli
            | ExternalOptimizationTool::CbcCli
            | ExternalOptimizationTool::ClpCli
            | ExternalOptimizationTool::Symphony
            | ExternalOptimizationTool::GurobiCli
            | ExternalOptimizationTool::CplexCli
            | ExternalOptimizationTool::XpressCli
            | ExternalOptimizationTool::LindoCli => ExternalOptimizationExactness::Exact,
            ExternalOptimizationTool::OptaPlanner
            | ExternalOptimizationTool::Timefold
            | ExternalOptimizationTool::Hexaly
            | ExternalOptimizationTool::JMetal
            | ExternalOptimizationTool::MoeaFramework
            | ExternalOptimizationTool::Ecj => ExternalOptimizationExactness::Heuristic,
            ExternalOptimizationTool::GoodLp
            | ExternalOptimizationTool::Cpmpy
            | ExternalOptimizationTool::PyCsp3
            | ExternalOptimizationTool::Conjure
            | ExternalOptimizationTool::SavileRow
            | ExternalOptimizationTool::LpModeler
            | ExternalOptimizationTool::Pyomo
            | ExternalOptimizationTool::Pulp
            | ExternalOptimizationTool::Cvxpy
            | ExternalOptimizationTool::PythonMip
            | ExternalOptimizationTool::Docplex
            | ExternalOptimizationTool::Jump
            | ExternalOptimizationTool::Ampl
            | ExternalOptimizationTool::Gams => ExternalOptimizationExactness::ModelingLayer,
            ExternalOptimizationTool::Casadi => ExternalOptimizationExactness::ModelingLayer,
            ExternalOptimizationTool::Argmin
            | ExternalOptimizationTool::Nlopt
            | ExternalOptimizationTool::Cvxopt
            | ExternalOptimizationTool::Minotaur
            | ExternalOptimizationTool::Ipopt
            | ExternalOptimizationTool::Bonmin
            | ExternalOptimizationTool::Couenne
            | ExternalOptimizationTool::Knitro
            | ExternalOptimizationTool::Mosek
            | ExternalOptimizationTool::Baron
            | ExternalOptimizationTool::Copt
            | ExternalOptimizationTool::Osqp
            | ExternalOptimizationTool::Scs
            | ExternalOptimizationTool::Clarabel
            | ExternalOptimizationTool::Ecos
            | ExternalOptimizationTool::Qpoases
            | ExternalOptimizationTool::Proxqp
            | ExternalOptimizationTool::Cosmo
            | ExternalOptimizationTool::Sdpa
            | ExternalOptimizationTool::Csdp
            | ExternalOptimizationTool::PyScipOpt
            | ExternalOptimizationTool::GurobiPy
            | ExternalOptimizationTool::ScipyOptimize => ExternalOptimizationExactness::Numerical,
        }
    }

    pub fn linear_cli_solver(self) -> Option<ExternalLinearCliSolver> {
        match self {
            ExternalOptimizationTool::HighsCli => Some(ExternalLinearCliSolver::Highs),
            ExternalOptimizationTool::GlpkCli => Some(ExternalLinearCliSolver::Glpk),
            ExternalOptimizationTool::ScipCli => Some(ExternalLinearCliSolver::Scip),
            ExternalOptimizationTool::CbcCli => Some(ExternalLinearCliSolver::Cbc),
            ExternalOptimizationTool::ClpCli => Some(ExternalLinearCliSolver::Clp),
            ExternalOptimizationTool::GurobiCli => Some(ExternalLinearCliSolver::Gurobi),
            ExternalOptimizationTool::CplexCli => Some(ExternalLinearCliSolver::Cplex),
            ExternalOptimizationTool::XpressCli => Some(ExternalLinearCliSolver::Xpress),
            ExternalOptimizationTool::LindoCli => Some(ExternalLinearCliSolver::Lindo),
            _ => None,
        }
    }

    pub fn adapter_command_aliases(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::ChocoSolver => {
                &["ores-choco-solver-adapter", "choco-solver-adapter"]
            }
            ExternalOptimizationTool::Jacop => &["ores-jacop-adapter", "jacop-adapter"],
            ExternalOptimizationTool::IbmCpOptimizer => {
                &["ores-ibm-cp-optimizer-adapter", "cpoptimizer-adapter"]
            }
            ExternalOptimizationTool::OptaPlanner => {
                &["ores-optaplanner-adapter", "optaplanner-adapter"]
            }
            ExternalOptimizationTool::Timefold => &["ores-timefold-adapter", "timefold-adapter"],
            ExternalOptimizationTool::JMetal => &["ores-jmetal-adapter", "jmetal-adapter"],
            ExternalOptimizationTool::MoeaFramework => {
                &["ores-moea-framework-adapter", "moea-framework-adapter"]
            }
            ExternalOptimizationTool::Ecj => &["ores-ecj-adapter", "ecj-adapter"],
            ExternalOptimizationTool::OjAlgo => &["ores-ojalgo-adapter", "ojalgo-adapter"],
            ExternalOptimizationTool::OrToolsJava => {
                &["ores-ortools-java-adapter", "ortools-java-adapter"]
            }
            ExternalOptimizationTool::Cpmpy => &["ores-cpmpy-adapter", "cpmpy-adapter"],
            ExternalOptimizationTool::PyCsp3 => &["ores-pycsp3-adapter", "pycsp3-adapter"],
            ExternalOptimizationTool::Conjure => &["ores-conjure-adapter", "conjure-adapter"],
            ExternalOptimizationTool::SavileRow => {
                &["ores-savile-row-adapter", "savile-row-adapter"]
            }
            ExternalOptimizationTool::Picat => &["ores-picat-adapter", "picat-adapter"],
            ExternalOptimizationTool::Clingo => &["ores-clingo-adapter", "clingo-adapter"],
            ExternalOptimizationTool::Clingcon => &["ores-clingcon-adapter", "clingcon-adapter"],
            ExternalOptimizationTool::Sat4j => &["ores-sat4j-adapter", "sat4j-adapter"],
            ExternalOptimizationTool::PySat => &["ores-pysat-adapter", "pysat-adapter"],
            ExternalOptimizationTool::OpenWbo => &["ores-open-wbo-adapter", "open-wbo-adapter"],
            ExternalOptimizationTool::Pyomo => &["ores-pyomo-adapter", "pyomo-adapter"],
            ExternalOptimizationTool::Pulp => &["ores-pulp-adapter", "pulp-adapter"],
            ExternalOptimizationTool::Cvxpy => &["ores-cvxpy-adapter", "cvxpy-adapter"],
            ExternalOptimizationTool::Cvxopt => &["ores-cvxopt-adapter", "cvxopt-adapter"],
            ExternalOptimizationTool::PyScipOpt => &["ores-pyscipopt-adapter", "pyscipopt-adapter"],
            ExternalOptimizationTool::PythonMip => {
                &["ores-python-mip-adapter", "python-mip-adapter"]
            }
            ExternalOptimizationTool::GurobiPy => &["ores-gurobipy-adapter", "gurobipy-adapter"],
            ExternalOptimizationTool::Docplex => &["ores-docplex-adapter", "docplex-adapter"],
            ExternalOptimizationTool::OrToolsPython => {
                &["ores-ortools-python-adapter", "ortools-python-adapter"]
            }
            ExternalOptimizationTool::ScipyOptimize => {
                &["ores-scipy-optimize-adapter", "scipy-optimize-adapter"]
            }
            ExternalOptimizationTool::Jump => &["ores-jump-adapter", "jump-adapter"],
            ExternalOptimizationTool::Ampl => &["ores-ampl-adapter", "ampl-adapter"],
            ExternalOptimizationTool::Gams => &["ores-gams-adapter", "gams-adapter"],
            ExternalOptimizationTool::Hexaly => &["ores-hexaly-adapter", "hexaly-adapter"],
            ExternalOptimizationTool::Minotaur => &["ores-minotaur-adapter", "minotaur-adapter"],
            ExternalOptimizationTool::Symphony => &["ores-symphony-adapter", "symphony-adapter"],
            ExternalOptimizationTool::Ipopt => &["ores-ipopt-adapter", "ipopt-adapter"],
            ExternalOptimizationTool::Bonmin => &["ores-bonmin-adapter", "bonmin-adapter"],
            ExternalOptimizationTool::Couenne => &["ores-couenne-adapter", "couenne-adapter"],
            ExternalOptimizationTool::Knitro => &["ores-knitro-adapter", "knitro-adapter"],
            ExternalOptimizationTool::Mosek => &["ores-mosek-adapter", "mosek-adapter"],
            ExternalOptimizationTool::Baron => &["ores-baron-adapter", "baron-adapter"],
            ExternalOptimizationTool::Copt => &["ores-copt-adapter", "copt-adapter"],
            ExternalOptimizationTool::Casadi => &["ores-casadi-adapter", "casadi-adapter"],
            ExternalOptimizationTool::Osqp => &["ores-osqp-adapter", "osqp-adapter"],
            ExternalOptimizationTool::Scs => &["ores-scs-adapter", "scs-adapter"],
            ExternalOptimizationTool::Clarabel => &["ores-clarabel-adapter", "clarabel-adapter"],
            ExternalOptimizationTool::Ecos => &["ores-ecos-adapter", "ecos-adapter"],
            ExternalOptimizationTool::Qpoases => &["ores-qpoases-adapter", "qpoases-adapter"],
            ExternalOptimizationTool::Proxqp => &["ores-proxqp-adapter", "proxqp-adapter"],
            ExternalOptimizationTool::Cosmo => &["ores-cosmo-adapter", "cosmo-adapter"],
            ExternalOptimizationTool::Sdpa => &["ores-sdpa-adapter", "sdpa-adapter"],
            ExternalOptimizationTool::Csdp => &["ores-csdp-adapter", "csdp-adapter"],
            ExternalOptimizationTool::HighsCli
            | ExternalOptimizationTool::GlpkCli
            | ExternalOptimizationTool::ScipCli
            | ExternalOptimizationTool::CbcCli
            | ExternalOptimizationTool::ClpCli
            | ExternalOptimizationTool::GurobiCli
            | ExternalOptimizationTool::CplexCli
            | ExternalOptimizationTool::XpressCli
            | ExternalOptimizationTool::LindoCli => &[],
            ExternalOptimizationTool::GoodLp => &["ores-good-lp-adapter", "good-lp-adapter"],
            ExternalOptimizationTool::LpModeler => {
                &["ores-lp-modeler-adapter", "lp-modeler-adapter"]
            }
            ExternalOptimizationTool::RustLinprog => {
                &["ores-rust-linprog-adapter", "rust-linprog-adapter"]
            }
            ExternalOptimizationTool::Argmin => &["ores-argmin-adapter", "argmin-adapter"],
            ExternalOptimizationTool::Nlopt => &["ores-nlopt-adapter", "nlopt-adapter"],
            ExternalOptimizationTool::HighsRust => {
                &["ores-highs-rust-adapter", "highs-rust-adapter"]
            }
            ExternalOptimizationTool::ScipRust => &["ores-scip-rust-adapter", "scip-rust-adapter"],
            ExternalOptimizationTool::CbcRust => &["ores-cbc-rust-adapter", "cbc-rust-adapter"],
        }
    }

    pub fn cargo_crates(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::GoodLp => &["good_lp"],
            ExternalOptimizationTool::LpModeler => &["lp-modeler"],
            ExternalOptimizationTool::RustLinprog => &["rust-linprog", "linprog"],
            ExternalOptimizationTool::Argmin => &["argmin"],
            ExternalOptimizationTool::Nlopt => &["nlopt", "nlopt-rs", "nlopt-sys"],
            ExternalOptimizationTool::HighsRust => &["highs", "highs-sys", "highs-rs"],
            ExternalOptimizationTool::ScipRust => &["russcip", "scip-sys"],
            ExternalOptimizationTool::CbcRust => &["coin_cbc", "cbc-sys"],
            _ => &[],
        }
    }

    pub fn python_modules(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::Pyomo => &["pyomo"],
            ExternalOptimizationTool::Cpmpy => &["cpmpy"],
            ExternalOptimizationTool::PyCsp3 => &["pycsp3"],
            ExternalOptimizationTool::PySat => &["pysat"],
            ExternalOptimizationTool::Pulp => &["pulp"],
            ExternalOptimizationTool::Cvxpy => &["cvxpy"],
            ExternalOptimizationTool::Cvxopt => &["cvxopt"],
            ExternalOptimizationTool::PyScipOpt => &["pyscipopt"],
            ExternalOptimizationTool::PythonMip => &["mip"],
            ExternalOptimizationTool::GurobiPy => &["gurobipy"],
            ExternalOptimizationTool::Docplex => &["docplex"],
            ExternalOptimizationTool::OrToolsPython => &["ortools"],
            ExternalOptimizationTool::ScipyOptimize => &["scipy"],
            ExternalOptimizationTool::Casadi => &["casadi"],
            ExternalOptimizationTool::Osqp => &["osqp"],
            ExternalOptimizationTool::Scs => &["scs"],
            ExternalOptimizationTool::Clarabel => &["clarabel"],
            ExternalOptimizationTool::Ecos => &["ecos"],
            _ => &[],
        }
    }

    pub fn julia_packages(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::Jump => &["JuMP"],
            _ => &[],
        }
    }

    pub fn notes(self) -> &'static str {
        match self {
            ExternalOptimizationTool::ChocoSolver => {
                "Java-native CP solver for finite-domain scheduling and combinatorial models"
            }
            ExternalOptimizationTool::Jacop => {
                "Java CP solver used in research/teaching finite-domain models"
            }
            ExternalOptimizationTool::IbmCpOptimizer => {
                "Commercial CP engine with strong interval scheduling support"
            }
            ExternalOptimizationTool::OptaPlanner => {
                "Java planning/metaheuristic system for timetabling, routing, and rostering"
            }
            ExternalOptimizationTool::Timefold => {
                "Open-source Java/Kotlin planning solver forked from OptaPlanner"
            }
            ExternalOptimizationTool::JMetal => {
                "Java evolutionary multi-objective optimization framework"
            }
            ExternalOptimizationTool::MoeaFramework => {
                "Java evolutionary and Pareto-front optimization framework"
            }
            ExternalOptimizationTool::Ecj => "Java evolutionary computation framework",
            ExternalOptimizationTool::OjAlgo => {
                "Java numerical optimization and self-contained LP/MIP-style solver"
            }
            ExternalOptimizationTool::OrToolsJava => {
                "Java API surface for OR-Tools CP-SAT, routing, and linear solvers"
            }
            ExternalOptimizationTool::Cpmpy => {
                "Python CP modeling layer for solver-agnostic constraint model cross-checks"
            }
            ExternalOptimizationTool::PyCsp3 => {
                "Python/XCSP3 modeling layer for constraint-problem validation"
            }
            ExternalOptimizationTool::Conjure => {
                "Essence constraint-modeling frontend for generating independent solver models"
            }
            ExternalOptimizationTool::SavileRow => {
                "Constraint-model reformulation tool for Essence Prime and SAT/SMT/MIP backends"
            }
            ExternalOptimizationTool::Picat => {
                "Logic-based CP/MIP/SAT programming language for independent model checks"
            }
            ExternalOptimizationTool::Clingo => {
                "Answer-set programming solver for combinatorial logic-model cross-checks"
            }
            ExternalOptimizationTool::Clingcon => {
                "Answer-set and finite-domain constraint solver in the Potassco ecosystem"
            }
            ExternalOptimizationTool::Sat4j => {
                "Java SAT and pseudo-Boolean solver library for JVM-side validation"
            }
            ExternalOptimizationTool::PySat => {
                "Python SAT toolkit wrapping multiple SAT solvers and cardinality encodings"
            }
            ExternalOptimizationTool::OpenWbo => {
                "MaxSAT solver for weighted-CNF objective and feasibility cross-checks"
            }
            ExternalOptimizationTool::Pyomo => {
                "Python algebraic modeling system for LP, MIP, QP, NLP, MINLP, stochastic, and bilevel models"
            }
            ExternalOptimizationTool::Pulp => {
                "COIN-OR Python LP/MIP modeling layer that writes LP/MPS and calls external solvers"
            }
            ExternalOptimizationTool::Cvxpy => {
                "Python embedded modeling language for convex optimization and conic solver cross-checks"
            }
            ExternalOptimizationTool::Cvxopt => {
                "Python package for convex optimization, dense/sparse matrices, and conic/QP routines"
            }
            ExternalOptimizationTool::PyScipOpt => {
                "Python interface to SCIP for MIP, MINLP, custom plugins, and solution-pool checks"
            }
            ExternalOptimizationTool::PythonMip => {
                "Python modeling layer for mixed-integer linear programming with CBC and Gurobi backends"
            }
            ExternalOptimizationTool::GurobiPy => {
                "Official Python API for Gurobi Optimizer models, parameters, callbacks, and attributes"
            }
            ExternalOptimizationTool::Docplex => {
                "IBM DOcplex object-oriented Python modeling API for CPLEX and CP Optimizer"
            }
            ExternalOptimizationTool::OrToolsPython => {
                "Python API surface for OR-Tools CP-SAT, routing, and linear solver validation"
            }
            ExternalOptimizationTool::ScipyOptimize => {
                "SciPy numerical optimization routines for nonlinear and least-squares reference checks"
            }
            ExternalOptimizationTool::Jump => {
                "Julia algebraic modeling layer using MathOptInterface-compatible solvers"
            }
            ExternalOptimizationTool::Ampl => {
                "AMPL algebraic modeling system with unified commercial and open-source solver interfaces"
            }
            ExternalOptimizationTool::Gams => {
                "GAMS algebraic modeling system and solver infrastructure for large-scale optimization"
            }
            ExternalOptimizationTool::Hexaly => {
                "Hybrid optimization solver/modeler for routing, scheduling, nonlinear, and CP-style models"
            }
            ExternalOptimizationTool::Minotaur => {
                "Open-source mixed-integer nonlinear optimization toolkit for MINLP cross-checks"
            }
            ExternalOptimizationTool::Symphony => {
                "COIN-OR branch-and-cut MILP solver framework for independent MIP checks"
            }
            ExternalOptimizationTool::Ipopt => {
                "Interior-point nonlinear programming solver for smooth NLP validation"
            }
            ExternalOptimizationTool::Bonmin => {
                "COIN-OR mixed-integer nonlinear programming solver"
            }
            ExternalOptimizationTool::Couenne => {
                "COIN-OR global optimization solver for nonconvex MINLP models"
            }
            ExternalOptimizationTool::Knitro => {
                "Commercial nonlinear and mixed-integer nonlinear optimization solver"
            }
            ExternalOptimizationTool::Mosek => {
                "Commercial conic, quadratic, semidefinite, and mixed-integer optimization solver"
            }
            ExternalOptimizationTool::Baron => {
                "Commercial global optimization solver for nonlinear and mixed-integer nonlinear models"
            }
            ExternalOptimizationTool::Copt => {
                "Commercial LP/QP/QCP/MIP and conic optimization solver"
            }
            ExternalOptimizationTool::Casadi => {
                "Python symbolic/numeric optimization and automatic-differentiation modeling layer"
            }
            ExternalOptimizationTool::Osqp => {
                "Operator-splitting quadratic-program solver for convex QP cross-checks"
            }
            ExternalOptimizationTool::Scs => {
                "Splitting conic solver for convex cone-program cross-checks"
            }
            ExternalOptimizationTool::Clarabel => {
                "Interior-point conic solver with quadratic objective support"
            }
            ExternalOptimizationTool::Ecos => {
                "Embedded conic solver for SOCP-style reference checks"
            }
            ExternalOptimizationTool::Qpoases => {
                "Active-set quadratic-program solver for dense and online QP checks"
            }
            ExternalOptimizationTool::Proxqp => {
                "Proximal quadratic-program solver for convex QP validation"
            }
            ExternalOptimizationTool::Cosmo => {
                "Conic splitting solver for convex cone-program validation"
            }
            ExternalOptimizationTool::Sdpa => {
                "Semidefinite programming solver for SDP cross-checks"
            }
            ExternalOptimizationTool::Csdp => {
                "COIN-OR semidefinite programming solver for SDP cross-checks"
            }
            ExternalOptimizationTool::HighsCli => {
                "Native HiGHS command-line bridge for local LP/MIP/QP smoke checks and cross-validation"
            }
            ExternalOptimizationTool::GlpkCli => {
                "Native GLPK glpsol command-line bridge for local LP/MIP cross-validation"
            }
            ExternalOptimizationTool::ScipCli => {
                "Native SCIP command-line bridge for local LP/MIP and constraint-integer cross-validation"
            }
            ExternalOptimizationTool::CbcCli => {
                "Native COIN-OR CBC command-line bridge for local MIP cross-validation"
            }
            ExternalOptimizationTool::ClpCli => {
                "Native COIN-OR CLP command-line bridge for local LP cross-validation"
            }
            ExternalOptimizationTool::GurobiCli => {
                "Commercial Gurobi Optimizer command-line bridge using local, non-vendored executables"
            }
            ExternalOptimizationTool::CplexCli => {
                "Commercial IBM ILOG CPLEX Optimizer command-line bridge using local installations"
            }
            ExternalOptimizationTool::XpressCli => {
                "Commercial FICO Xpress Optimizer command-line bridge using local installations"
            }
            ExternalOptimizationTool::LindoCli => {
                "Commercial LINDO Systems command-line bridge using local installations"
            }
            ExternalOptimizationTool::GoodLp => {
                "Rust LP/MIP modeling layer that delegates to solver backends"
            }
            ExternalOptimizationTool::LpModeler => "Rust LP modeling DSL",
            ExternalOptimizationTool::RustLinprog => "Rust-first lightweight linear programming",
            ExternalOptimizationTool::Argmin => {
                "Rust nonlinear optimization algorithms for gradient and derivative-free runs"
            }
            ExternalOptimizationTool::Nlopt => {
                "Rust bindings to NLopt nonlinear optimization algorithms"
            }
            ExternalOptimizationTool::HighsRust => "Rust bindings to HiGHS LP/MIP/QP",
            ExternalOptimizationTool::ScipRust => "Rust bindings to SCIP MIP/CP stack",
            ExternalOptimizationTool::CbcRust => "Rust bindings to COIN-OR CBC MIP",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalOptimizationLanguage {
    Java,
    Python,
    Julia,
    Native,
    Rust,
}

impl ExternalOptimizationLanguage {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationLanguage::Java => "java",
            ExternalOptimizationLanguage::Python => "python",
            ExternalOptimizationLanguage::Julia => "julia",
            ExternalOptimizationLanguage::Native => "native",
            ExternalOptimizationLanguage::Rust => "rust",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalOptimizationFamily {
    ConstraintProgramming,
    PlanningMetaheuristic,
    EvolutionaryMultiObjective,
    LinearMip,
    CpSatRouting,
    ConvexOptimization,
    NonlinearOptimization,
    HybridOptimization,
    NativeSolverBinding,
}

impl ExternalOptimizationFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationFamily::ConstraintProgramming => "constraint-programming",
            ExternalOptimizationFamily::PlanningMetaheuristic => "planning-metaheuristic",
            ExternalOptimizationFamily::EvolutionaryMultiObjective => {
                "evolutionary-multi-objective"
            }
            ExternalOptimizationFamily::LinearMip => "linear-mip",
            ExternalOptimizationFamily::CpSatRouting => "cp-sat-routing",
            ExternalOptimizationFamily::ConvexOptimization => "convex-optimization",
            ExternalOptimizationFamily::NonlinearOptimization => "nonlinear-optimization",
            ExternalOptimizationFamily::HybridOptimization => "hybrid-optimization",
            ExternalOptimizationFamily::NativeSolverBinding => "native-solver-binding",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalOptimizationExactness {
    Exact,
    Heuristic,
    ModelingLayer,
    Numerical,
}

impl ExternalOptimizationExactness {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationExactness::Exact => "exact",
            ExternalOptimizationExactness::Heuristic => "heuristic",
            ExternalOptimizationExactness::ModelingLayer => "modeling-layer",
            ExternalOptimizationExactness::Numerical => "numerical",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalOptimizationToolSpec {
    pub tool: ExternalOptimizationTool,
    pub display_name: &'static str,
    pub language: ExternalOptimizationLanguage,
    pub family: ExternalOptimizationFamily,
    pub exactness: ExternalOptimizationExactness,
    pub adapter_env_names: Vec<String>,
    pub artifact_env_names: Vec<String>,
    pub adapter_command_aliases: &'static [&'static str],
    pub cargo_crates: &'static [&'static str],
    pub notes: &'static str,
}

pub fn external_optimization_tools() -> &'static [ExternalOptimizationTool] {
    &[
        ExternalOptimizationTool::ChocoSolver,
        ExternalOptimizationTool::Jacop,
        ExternalOptimizationTool::IbmCpOptimizer,
        ExternalOptimizationTool::OptaPlanner,
        ExternalOptimizationTool::Timefold,
        ExternalOptimizationTool::JMetal,
        ExternalOptimizationTool::MoeaFramework,
        ExternalOptimizationTool::Ecj,
        ExternalOptimizationTool::OjAlgo,
        ExternalOptimizationTool::OrToolsJava,
        ExternalOptimizationTool::Cpmpy,
        ExternalOptimizationTool::PyCsp3,
        ExternalOptimizationTool::Conjure,
        ExternalOptimizationTool::SavileRow,
        ExternalOptimizationTool::Picat,
        ExternalOptimizationTool::Clingo,
        ExternalOptimizationTool::Clingcon,
        ExternalOptimizationTool::Sat4j,
        ExternalOptimizationTool::PySat,
        ExternalOptimizationTool::OpenWbo,
        ExternalOptimizationTool::Pyomo,
        ExternalOptimizationTool::Pulp,
        ExternalOptimizationTool::Cvxpy,
        ExternalOptimizationTool::Cvxopt,
        ExternalOptimizationTool::PyScipOpt,
        ExternalOptimizationTool::PythonMip,
        ExternalOptimizationTool::GurobiPy,
        ExternalOptimizationTool::Docplex,
        ExternalOptimizationTool::OrToolsPython,
        ExternalOptimizationTool::ScipyOptimize,
        ExternalOptimizationTool::Jump,
        ExternalOptimizationTool::Ampl,
        ExternalOptimizationTool::Gams,
        ExternalOptimizationTool::Hexaly,
        ExternalOptimizationTool::Minotaur,
        ExternalOptimizationTool::Symphony,
        ExternalOptimizationTool::Ipopt,
        ExternalOptimizationTool::Bonmin,
        ExternalOptimizationTool::Couenne,
        ExternalOptimizationTool::Knitro,
        ExternalOptimizationTool::Mosek,
        ExternalOptimizationTool::Baron,
        ExternalOptimizationTool::Copt,
        ExternalOptimizationTool::Casadi,
        ExternalOptimizationTool::Osqp,
        ExternalOptimizationTool::Scs,
        ExternalOptimizationTool::Clarabel,
        ExternalOptimizationTool::Ecos,
        ExternalOptimizationTool::Qpoases,
        ExternalOptimizationTool::Proxqp,
        ExternalOptimizationTool::Cosmo,
        ExternalOptimizationTool::Sdpa,
        ExternalOptimizationTool::Csdp,
        ExternalOptimizationTool::HighsCli,
        ExternalOptimizationTool::GlpkCli,
        ExternalOptimizationTool::ScipCli,
        ExternalOptimizationTool::CbcCli,
        ExternalOptimizationTool::ClpCli,
        ExternalOptimizationTool::GurobiCli,
        ExternalOptimizationTool::CplexCli,
        ExternalOptimizationTool::XpressCli,
        ExternalOptimizationTool::LindoCli,
        ExternalOptimizationTool::GoodLp,
        ExternalOptimizationTool::LpModeler,
        ExternalOptimizationTool::RustLinprog,
        ExternalOptimizationTool::Argmin,
        ExternalOptimizationTool::Nlopt,
        ExternalOptimizationTool::HighsRust,
        ExternalOptimizationTool::ScipRust,
        ExternalOptimizationTool::CbcRust,
    ]
}

pub fn external_optimization_tool_specs() -> Vec<ExternalOptimizationToolSpec> {
    external_optimization_tools()
        .iter()
        .copied()
        .map(external_optimization_tool_spec)
        .collect()
}

pub fn external_optimization_tool_spec(
    tool: ExternalOptimizationTool,
) -> ExternalOptimizationToolSpec {
    ExternalOptimizationToolSpec {
        tool,
        display_name: tool.display_name(),
        language: tool.language(),
        family: tool.family(),
        exactness: tool.exactness(),
        adapter_env_names: adapter_env_names(tool),
        artifact_env_names: artifact_env_names(tool),
        adapter_command_aliases: tool.adapter_command_aliases(),
        cargo_crates: tool.cargo_crates(),
        notes: tool.notes(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalOptimizationProbeStatus {
    Ready,
    NotConfigured,
    RuntimeMissing,
    AdapterMissing,
}

impl ExternalOptimizationProbeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationProbeStatus::Ready => "ready",
            ExternalOptimizationProbeStatus::NotConfigured => "not-configured",
            ExternalOptimizationProbeStatus::RuntimeMissing => "runtime-missing",
            ExternalOptimizationProbeStatus::AdapterMissing => "adapter-missing",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternalOptimizationAdapterOptions {
    pub tool: ExternalOptimizationTool,
    pub command_path: Option<PathBuf>,
    pub java_command: Option<PathBuf>,
    pub cargo_manifest_dir: Option<PathBuf>,
    pub working_dir: Option<PathBuf>,
    pub time_limit_secs: Option<f64>,
    pub extra_args: Vec<String>,
}

impl Default for ExternalOptimizationAdapterOptions {
    fn default() -> Self {
        Self {
            tool: ExternalOptimizationTool::ChocoSolver,
            command_path: None,
            java_command: None,
            cargo_manifest_dir: None,
            working_dir: None,
            time_limit_secs: None,
            extra_args: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalOptimizationProbe {
    pub tool: ExternalOptimizationTool,
    pub status: ExternalOptimizationProbeStatus,
    pub command: Option<PathBuf>,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalOptimizationAdapterStatus {
    Ok,
    Unavailable,
    Failed,
    InvalidOutput,
}

impl ExternalOptimizationAdapterStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationAdapterStatus::Ok => "ok",
            ExternalOptimizationAdapterStatus::Unavailable => "unavailable",
            ExternalOptimizationAdapterStatus::Failed => "failed",
            ExternalOptimizationAdapterStatus::InvalidOutput => "invalid-output",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalOptimizationAdapterRun {
    pub tool: ExternalOptimizationTool,
    pub status: ExternalOptimizationAdapterStatus,
    pub output: Option<Value>,
    pub elapsed_ms: f64,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct ExternalOptimizationAdapterInvocation {
    pub label: String,
    pub options: ExternalOptimizationAdapterOptions,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExternalOptimizationNormalizedResult {
    pub status: Option<String>,
    pub objective: Option<f64>,
    pub solution: Option<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalOptimizationComparisonRun {
    pub label: String,
    pub run: ExternalOptimizationAdapterRun,
    pub normalized: ExternalOptimizationNormalizedResult,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalOptimizationComparisonReport {
    pub objective_tolerance: f64,
    pub solution_tolerance: f64,
    pub all_successful: bool,
    pub statuses_agree: bool,
    pub objectives_agree: bool,
    pub solutions_agree: bool,
    pub agreement: bool,
    pub reference_status: Option<String>,
    pub reference_objective: Option<f64>,
    pub reference_solution: Option<Vec<f64>>,
    pub runs: Vec<ExternalOptimizationComparisonRun>,
}

pub fn adapter_env_names(tool: ExternalOptimizationTool) -> Vec<String> {
    let key = env_key(tool);
    vec![
        format!("ORES_{key}_ADAPTER"),
        format!("DES_{key}_ADAPTER"),
        format!("{key}_ADAPTER"),
    ]
}

pub fn artifact_env_names(tool: ExternalOptimizationTool) -> Vec<String> {
    let key = env_key(tool);
    let suffix = match tool.language() {
        ExternalOptimizationLanguage::Java => "CLASSPATH",
        ExternalOptimizationLanguage::Python => "PYTHON",
        ExternalOptimizationLanguage::Julia => "JULIA",
        ExternalOptimizationLanguage::Native => "DIR",
        ExternalOptimizationLanguage::Rust => "CRATE",
    };
    let mut names = vec![
        format!("ORES_{key}_{suffix}"),
        format!("DES_{key}_{suffix}"),
        format!("{key}_{suffix}"),
    ];
    match tool {
        ExternalOptimizationTool::IbmCpOptimizer => {
            names.push("CPLEX_STUDIO_DIR".to_string());
        }
        ExternalOptimizationTool::OrToolsJava => {
            names.push("ORTOOLS_JAVA_HOME".to_string());
        }
        ExternalOptimizationTool::Cpmpy => {
            names.push("CPMPY_PYTHON".to_string());
        }
        ExternalOptimizationTool::PyCsp3 => {
            names.push("PYCSP3_PYTHON".to_string());
        }
        ExternalOptimizationTool::Conjure => {
            names.push("CONJURE_HOME".to_string());
        }
        ExternalOptimizationTool::SavileRow => {
            names.push("SAVILEROW_HOME".to_string());
        }
        ExternalOptimizationTool::Picat => {
            names.push("PICAT_HOME".to_string());
        }
        ExternalOptimizationTool::Clingo => {
            names.push("CLINGO_HOME".to_string());
        }
        ExternalOptimizationTool::Clingcon => {
            names.push("CLINGCON_HOME".to_string());
        }
        ExternalOptimizationTool::Sat4j => {
            names.push("SAT4J_HOME".to_string());
        }
        ExternalOptimizationTool::PySat => {
            names.push("PYSAT_PYTHON".to_string());
        }
        ExternalOptimizationTool::OpenWbo => {
            names.push("OPEN_WBO_HOME".to_string());
        }
        ExternalOptimizationTool::Pyomo => {
            names.push("PYOMO_PYTHON".to_string());
        }
        ExternalOptimizationTool::Pulp => {
            names.push("PULP_PYTHON".to_string());
        }
        ExternalOptimizationTool::Cvxpy => {
            names.push("CVXPY_PYTHON".to_string());
        }
        ExternalOptimizationTool::Cvxopt => {
            names.push("CVXOPT_PYTHON".to_string());
        }
        ExternalOptimizationTool::PyScipOpt => {
            names.push("PYSCIPOPT_PYTHON".to_string());
            names.push("SCIPOPTDIR".to_string());
        }
        ExternalOptimizationTool::PythonMip => {
            names.push("PYTHON_MIP_PYTHON".to_string());
        }
        ExternalOptimizationTool::GurobiPy => {
            names.push("GUROBI_HOME".to_string());
            names.push("GRB_LICENSE_FILE".to_string());
        }
        ExternalOptimizationTool::Docplex => {
            names.push("DOCPLEX_PYTHON".to_string());
            names.push("CPLEX_STUDIO_DIR".to_string());
        }
        ExternalOptimizationTool::OrToolsPython => {
            names.push("ORTOOLS_PYTHON".to_string());
        }
        ExternalOptimizationTool::ScipyOptimize => {
            names.push("SCIPY_PYTHON".to_string());
        }
        ExternalOptimizationTool::Jump => {
            names.push("JULIA_PROJECT".to_string());
        }
        ExternalOptimizationTool::Ampl => {
            names.push("AMPL_HOME".to_string());
            names.push("AMPL_DIR".to_string());
        }
        ExternalOptimizationTool::Gams => {
            names.push("GAMS_DIR".to_string());
            names.push("GAMSDIR".to_string());
        }
        ExternalOptimizationTool::Hexaly => {
            names.push("HEXALY_HOME".to_string());
            names.push("LOCALSOLVER_HOME".to_string());
        }
        ExternalOptimizationTool::Minotaur => {
            names.push("MINOTAUR_DIR".to_string());
        }
        ExternalOptimizationTool::Symphony => {
            names.push("SYMPHONY_DIR".to_string());
            names.push("COINOR_DIR".to_string());
        }
        ExternalOptimizationTool::Ipopt => {
            names.push("IPOPT_DIR".to_string());
        }
        ExternalOptimizationTool::Bonmin => {
            names.push("BONMIN_DIR".to_string());
        }
        ExternalOptimizationTool::Couenne => {
            names.push("COUENNE_DIR".to_string());
        }
        ExternalOptimizationTool::Knitro => {
            names.push("ARTELYS_LICENSE".to_string());
        }
        ExternalOptimizationTool::Mosek => {
            names.push("MOSEKLM_LICENSE_FILE".to_string());
        }
        ExternalOptimizationTool::Baron => {
            names.push("BARON_LICENSE".to_string());
        }
        ExternalOptimizationTool::Copt => {
            names.push("COPT_HOME".to_string());
        }
        ExternalOptimizationTool::Casadi => {
            names.push("CASADI_PYTHON".to_string());
        }
        ExternalOptimizationTool::Osqp => {
            names.push("OSQP_PYTHON".to_string());
        }
        ExternalOptimizationTool::Scs => {
            names.push("SCS_PYTHON".to_string());
        }
        ExternalOptimizationTool::Clarabel => {
            names.push("CLARABEL_PYTHON".to_string());
        }
        ExternalOptimizationTool::Ecos => {
            names.push("ECOS_PYTHON".to_string());
        }
        ExternalOptimizationTool::Qpoases => {
            names.push("QPOASES_DIR".to_string());
        }
        ExternalOptimizationTool::Proxqp => {
            names.push("PROXQP_DIR".to_string());
        }
        ExternalOptimizationTool::Cosmo => {
            names.push("COSMO_DIR".to_string());
        }
        ExternalOptimizationTool::Sdpa => {
            names.push("SDPA_DIR".to_string());
        }
        ExternalOptimizationTool::Csdp => {
            names.push("CSDP_DIR".to_string());
        }
        ExternalOptimizationTool::HighsCli
        | ExternalOptimizationTool::GlpkCli
        | ExternalOptimizationTool::ScipCli
        | ExternalOptimizationTool::CbcCli
        | ExternalOptimizationTool::ClpCli
        | ExternalOptimizationTool::GurobiCli
        | ExternalOptimizationTool::CplexCli
        | ExternalOptimizationTool::XpressCli
        | ExternalOptimizationTool::LindoCli => {
            if let Some(solver) = tool.linear_cli_solver() {
                for name in solver.command_env_vars() {
                    push_unique_env_name(&mut names, *name);
                }
                for name in solver.command_dir_env_vars() {
                    push_unique_env_name(&mut names, *name);
                }
            }
        }
        ExternalOptimizationTool::Nlopt => {
            names.push("NLOPT_DIR".to_string());
        }
        ExternalOptimizationTool::HighsRust => {
            names.push("HIGHS_DIR".to_string());
        }
        ExternalOptimizationTool::ScipRust => {
            names.push("SCIPOPTDIR".to_string());
            names.push("SCIP_DIR".to_string());
        }
        ExternalOptimizationTool::CbcRust => {
            names.push("CBC_DIR".to_string());
            names.push("COINOR_DIR".to_string());
        }
        _ => {}
    }
    names
}

pub fn external_optimization_command_dir_env_names(tool: ExternalOptimizationTool) -> Vec<String> {
    let mut names = Vec::new();
    if tool.language() == ExternalOptimizationLanguage::Native {
        for name in artifact_env_names(tool) {
            push_unique_env_name(&mut names, name);
        }
    }
    for name in match tool {
        ExternalOptimizationTool::ChocoSolver => &["CHOCO_SOLVER_HOME", "CHOCO_HOME"][..],
        ExternalOptimizationTool::Jacop => &["JACOP_HOME", "JACOP_DIR"],
        ExternalOptimizationTool::IbmCpOptimizer => {
            &["CPLEX_STUDIO_DIR", "CPLEX_HOME", "CP_OPTIMIZER_HOME"][..]
        }
        ExternalOptimizationTool::OptaPlanner => &["OPTAPLANNER_HOME", "OPTAPLANNER_DIR"],
        ExternalOptimizationTool::Timefold => &["TIMEFOLD_HOME", "TIMEFOLD_DIR"],
        ExternalOptimizationTool::JMetal => &["JMETAL_HOME", "JMETAL_DIR"],
        ExternalOptimizationTool::MoeaFramework => &["MOEA_FRAMEWORK_HOME", "MOEA_HOME"],
        ExternalOptimizationTool::Ecj => &["ECJ_HOME", "ECJ_DIR"],
        ExternalOptimizationTool::OjAlgo => &["OJALGO_HOME", "OJALGO_DIR"],
        ExternalOptimizationTool::OrToolsJava => &["ORTOOLS_JAVA_HOME", "ORTOOLS_HOME"],
        ExternalOptimizationTool::Cpmpy => &["CPMPY_HOME", "CPMPY_DIR"],
        ExternalOptimizationTool::PyCsp3 => &["PYCSP3_HOME", "PYCSP3_DIR"],
        ExternalOptimizationTool::Conjure => &["CONJURE_HOME", "CONJURE_DIR"],
        ExternalOptimizationTool::SavileRow => {
            &["SAVILE_ROW_HOME", "SAVILE_ROW_DIR", "SAVILEROW_HOME"][..]
        }
        ExternalOptimizationTool::Picat => &["PICAT_HOME", "PICAT_DIR"],
        ExternalOptimizationTool::Clingo => &["CLINGO_HOME", "CLINGO_DIR", "POTASSCO_HOME"],
        ExternalOptimizationTool::Clingcon => &["CLINGCON_HOME", "CLINGCON_DIR", "POTASSCO_HOME"],
        ExternalOptimizationTool::Sat4j => &["SAT4J_HOME", "SAT4J_DIR"],
        ExternalOptimizationTool::PySat => &["PYSAT_HOME", "PYSAT_DIR"],
        ExternalOptimizationTool::OpenWbo => &["OPEN_WBO_HOME", "OPEN_WBO_DIR", "OPENWBO_HOME"],
        ExternalOptimizationTool::Pyomo => &["PYOMO_HOME", "PYOMO_DIR"],
        ExternalOptimizationTool::Pulp => &["PULP_HOME", "PULP_DIR"],
        ExternalOptimizationTool::Cvxpy => &["CVXPY_HOME", "CVXPY_DIR"],
        ExternalOptimizationTool::Cvxopt => &["CVXOPT_HOME", "CVXOPT_DIR"],
        ExternalOptimizationTool::PyScipOpt => &["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
        ExternalOptimizationTool::PythonMip => &["PYTHON_MIP_HOME", "PYTHON_MIP_DIR"],
        ExternalOptimizationTool::GurobiPy => &["GUROBI_HOME"],
        ExternalOptimizationTool::Docplex => &["DOCPLEX_HOME", "CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        ExternalOptimizationTool::OrToolsPython => &["ORTOOLS_HOME", "ORTOOLS_PYTHON_HOME"],
        ExternalOptimizationTool::ScipyOptimize => &["SCIPY_HOME", "SCIPY_DIR"],
        ExternalOptimizationTool::Ampl => &["AMPL_HOME", "AMPL_DIR"],
        ExternalOptimizationTool::Gams => &["GAMS_DIR", "GAMSDIR", "GAMS_HOME"],
        ExternalOptimizationTool::Hexaly => &[
            "HEXALY_HOME",
            "HEXALY_DIR",
            "LOCALSOLVER_HOME",
            "LOCALSOLVER_DIR",
        ],
        ExternalOptimizationTool::Minotaur => &["MINOTAUR_DIR", "MINOTAUR_HOME"],
        ExternalOptimizationTool::Symphony => {
            &["SYMPHONY_DIR", "SYMPHONY_HOME", "COINOR_DIR", "COINOR_HOME"]
        }
        ExternalOptimizationTool::Ipopt => &["IPOPT_DIR", "IPOPT_HOME"],
        ExternalOptimizationTool::Bonmin => &["BONMIN_DIR", "BONMIN_HOME"],
        ExternalOptimizationTool::Couenne => &["COUENNE_DIR", "COUENNE_HOME"],
        ExternalOptimizationTool::Knitro => {
            &["KNITRO_HOME", "KNITRODIR", "KNITRO_DIR", "ARTELYS_HOME"]
        }
        ExternalOptimizationTool::Mosek => &["MOSEK_HOME", "MSKHOME"],
        ExternalOptimizationTool::Baron => &["BARON_DIR", "BARON_HOME"],
        ExternalOptimizationTool::Copt => &["COPT_HOME", "COPT_DIR"],
        ExternalOptimizationTool::Casadi => &["CASADI_DIR", "CASADI_HOME"],
        ExternalOptimizationTool::Osqp => &["OSQP_DIR", "OSQP_HOME"],
        ExternalOptimizationTool::Scs => &["SCS_DIR", "SCS_HOME"],
        ExternalOptimizationTool::Clarabel => &["CLARABEL_DIR", "CLARABEL_HOME"],
        ExternalOptimizationTool::Ecos => &["ECOS_DIR", "ECOS_HOME"],
        ExternalOptimizationTool::Qpoases => &["QPOASES_DIR", "QPOASES_HOME"],
        ExternalOptimizationTool::Proxqp => &["PROXQP_DIR", "PROXQP_HOME"],
        ExternalOptimizationTool::Cosmo => &["COSMO_DIR", "COSMO_HOME"],
        ExternalOptimizationTool::Sdpa => &["SDPA_DIR", "SDPA_HOME"],
        ExternalOptimizationTool::Csdp => &["CSDP_DIR", "CSDP_HOME"],
        ExternalOptimizationTool::HighsCli => &["HIGHS_DIR", "HIGHS_HOME"],
        ExternalOptimizationTool::GlpkCli => &["GLPK_DIR", "GLPK_HOME"],
        ExternalOptimizationTool::ScipCli => &["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
        ExternalOptimizationTool::CbcCli => &["CBC_DIR", "CBC_HOME", "COINOR_DIR", "COINOR_HOME"],
        ExternalOptimizationTool::ClpCli => &["CLP_DIR", "CLP_HOME", "COINOR_DIR", "COINOR_HOME"],
        ExternalOptimizationTool::GurobiCli => &["GUROBI_HOME"],
        ExternalOptimizationTool::CplexCli => &["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        ExternalOptimizationTool::XpressCli => &["XPRESSDIR", "XPRESS_DIR", "XPRESS_HOME"],
        ExternalOptimizationTool::LindoCli => {
            &["LINDO_HOME", "LINDO_DIR", "LINDOAPI_HOME", "LINDOAPI_DIR"]
        }
        ExternalOptimizationTool::Nlopt => &["NLOPT_DIR", "NLOPT_HOME"],
        ExternalOptimizationTool::HighsRust => &["HIGHS_DIR", "HIGHS_HOME"],
        ExternalOptimizationTool::ScipRust => &["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
        ExternalOptimizationTool::CbcRust => &["CBC_DIR", "CBC_HOME", "COINOR_DIR", "COINOR_HOME"],
        _ => &[],
    } {
        push_unique_env_name(&mut names, *name);
    }
    names
}

pub fn external_optimization_adapter_command(tool: ExternalOptimizationTool) -> Option<PathBuf> {
    configured_adapter_command(tool)
        .0
        .or_else(|| find_first_command_in_install_dirs(tool))
        .or_else(|| find_first_command(tool.adapter_command_aliases()))
}

pub fn external_optimization_adapter_command_with_options(
    opts: &ExternalOptimizationAdapterOptions,
) -> Option<PathBuf> {
    opts.command_path
        .as_ref()
        .cloned()
        .or_else(|| external_optimization_adapter_command(opts.tool))
}

pub fn probe_external_optimization_tool(
    opts: &ExternalOptimizationAdapterOptions,
) -> ExternalOptimizationProbe {
    if let Some(solver) = opts.tool.linear_cli_solver() {
        return probe_external_optimization_linear_cli_tool(opts, solver);
    }

    let (configured_command, saw_configured_command) = configured_adapter_command(opts.tool);
    let command = opts
        .command_path
        .as_ref()
        .cloned()
        .or(configured_command)
        .or_else(|| find_first_command_in_install_dirs(opts.tool))
        .or_else(|| find_first_command(opts.tool.adapter_command_aliases()));
    if let Some(command) = command {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::Ready,
            command: Some(command.clone()),
            message: format!(
                "{} adapter command is configured at {}",
                opts.tool.display_name(),
                command.display()
            ),
        };
    }
    if saw_configured_command || opts.command_path.is_some() {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::AdapterMissing,
            command: None,
            message: format!(
                "{} adapter command was configured but could not be resolved",
                opts.tool.display_name()
            ),
        };
    }

    match opts.tool.language() {
        ExternalOptimizationLanguage::Java => probe_java_tool(opts),
        ExternalOptimizationLanguage::Python => probe_python_tool(opts),
        ExternalOptimizationLanguage::Julia => probe_julia_tool(opts),
        ExternalOptimizationLanguage::Native => probe_native_tool(opts),
        ExternalOptimizationLanguage::Rust => probe_rust_tool(opts),
    }
}

pub fn run_external_optimization_adapter(
    input: &Value,
    opts: &ExternalOptimizationAdapterOptions,
) -> ExternalOptimizationAdapterRun {
    let Some(command) = external_optimization_adapter_command_with_options(opts) else {
        return ExternalOptimizationAdapterRun {
            tool: opts.tool,
            status: ExternalOptimizationAdapterStatus::Unavailable,
            output: None,
            elapsed_ms: 0.0,
            message: format!(
                "{} adapter command not configured; set {}",
                opts.tool.display_name(),
                adapter_env_names(opts.tool)[0]
            ),
        };
    };
    let started = Instant::now();
    let mut child = match Command::new(&command)
        .args(&opts.extra_args)
        .env("ORES_EXTERNAL_OPTIMIZATION_TOOL", opts.tool.as_str())
        .current_dir(
            opts.working_dir
                .as_deref()
                .unwrap_or_else(|| Path::new(".")),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return ExternalOptimizationAdapterRun {
                tool: opts.tool,
                status: ExternalOptimizationAdapterStatus::Unavailable,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: err.to_string(),
            };
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        let payload = serde_json::to_vec(input).unwrap_or_else(|_| b"null".to_vec());
        if let Err(err) = stdin.write_all(&payload) {
            return ExternalOptimizationAdapterRun {
                tool: opts.tool,
                status: ExternalOptimizationAdapterStatus::Failed,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: err.to_string(),
            };
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return ExternalOptimizationAdapterRun {
                tool: opts.tool,
                status: ExternalOptimizationAdapterStatus::Failed,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: err.to_string(),
            };
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if !output.status.success() {
        return ExternalOptimizationAdapterRun {
            tool: opts.tool,
            status: ExternalOptimizationAdapterStatus::Failed,
            output: None,
            elapsed_ms,
            message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        };
    }
    match serde_json::from_slice::<Value>(&output.stdout) {
        Ok(value) => ExternalOptimizationAdapterRun {
            tool: opts.tool,
            status: ExternalOptimizationAdapterStatus::Ok,
            output: Some(value),
            elapsed_ms,
            message: String::new(),
        },
        Err(err) => ExternalOptimizationAdapterRun {
            tool: opts.tool,
            status: ExternalOptimizationAdapterStatus::InvalidOutput,
            output: None,
            elapsed_ms,
            message: err.to_string(),
        },
    }
}

pub fn external_optimization_normalized_result_from_value(
    output: &Value,
) -> ExternalOptimizationNormalizedResult {
    ExternalOptimizationNormalizedResult {
        status: first_string_at_any_path(
            output,
            &[
                &["status"],
                &["solver_status"],
                &["termination_status"],
                &["result", "status"],
                &["result", "solver_status"],
                &["solution", "status"],
            ],
        ),
        objective: first_f64_at_any_path(
            output,
            &[
                &["objective"],
                &["objective_value"],
                &["best_objective"],
                &["cost"],
                &["value"],
                &["result", "objective"],
                &["result", "objective_value"],
                &["solution", "objective"],
                &["solution", "objective_value"],
            ],
        ),
        solution: first_numeric_vector_at_any_path(
            output,
            &[
                &["x"],
                &["values"],
                &["variables"],
                &["assignment"],
                &["solution"],
                &["result", "x"],
                &["result", "values"],
                &["result", "solution"],
                &["solution", "x"],
                &["solution", "values"],
                &["solution", "variables"],
            ],
        ),
    }
}

pub fn external_optimization_run_normalized_result(
    run: &ExternalOptimizationAdapterRun,
) -> ExternalOptimizationNormalizedResult {
    run.output
        .as_ref()
        .map(external_optimization_normalized_result_from_value)
        .unwrap_or_default()
}

pub fn run_external_optimization_comparison(
    input: &Value,
    invocations: &[ExternalOptimizationAdapterInvocation],
    objective_tolerance: f64,
    solution_tolerance: f64,
) -> ExternalOptimizationComparisonReport {
    let objective_tolerance = objective_tolerance.max(0.0);
    let solution_tolerance = solution_tolerance.max(0.0);
    let runs: Vec<ExternalOptimizationComparisonRun> = invocations
        .iter()
        .map(|invocation| {
            let run = run_external_optimization_adapter(input, &invocation.options);
            let normalized = external_optimization_run_normalized_result(&run);
            ExternalOptimizationComparisonRun {
                label: invocation.label.clone(),
                run,
                normalized,
            }
        })
        .collect();

    let all_successful = !runs.is_empty()
        && runs
            .iter()
            .all(|run| run.run.status == ExternalOptimizationAdapterStatus::Ok);
    let reference_status = runs.first().and_then(|run| run.normalized.status.clone());
    let reference_objective = runs.first().and_then(|run| run.normalized.objective);
    let reference_solution = runs.first().and_then(|run| run.normalized.solution.clone());
    let statuses_agree = all_successful
        && reference_status.as_ref().is_some_and(|reference| {
            runs.iter()
                .all(|run| run.normalized.status.as_ref() == Some(reference))
        });
    let objectives_agree = all_successful
        && reference_objective.is_some_and(|reference| {
            runs.iter().all(|run| {
                run.normalized
                    .objective
                    .is_some_and(|objective| (objective - reference).abs() <= objective_tolerance)
            })
        });
    let solutions_agree = all_successful
        && reference_solution.as_ref().is_some_and(|reference| {
            runs.iter().all(|run| {
                run.normalized.solution.as_ref().is_some_and(|solution| {
                    solution.len() == reference.len()
                        && solution
                            .iter()
                            .zip(reference.iter())
                            .all(|(actual, expected)| {
                                (actual - expected).abs() <= solution_tolerance
                            })
                })
            })
        });
    let agreement = all_successful && statuses_agree && objectives_agree && solutions_agree;

    ExternalOptimizationComparisonReport {
        objective_tolerance,
        solution_tolerance,
        all_successful,
        statuses_agree,
        objectives_agree,
        solutions_agree,
        agreement,
        reference_status,
        reference_objective,
        reference_solution,
        runs,
    }
}

pub fn external_optimization_comparison_report_to_json(
    report: &ExternalOptimizationComparisonReport,
) -> Value {
    let runs: Vec<Value> = report
        .runs
        .iter()
        .map(|run| {
            json!({
                "label": &run.label,
                "tool": run.run.tool.as_str(),
                "status": run.run.status.as_str(),
                "normalized": {
                    "status": &run.normalized.status,
                    "objective": run.normalized.objective,
                    "solution": &run.normalized.solution,
                },
                "elapsed_ms": run.run.elapsed_ms,
                "message": &run.run.message,
                "output": &run.run.output,
            })
        })
        .collect();
    json!({
        "kind": "external-optimization-comparison-report",
        "objective_tolerance": report.objective_tolerance,
        "solution_tolerance": report.solution_tolerance,
        "all_successful": report.all_successful,
        "statuses_agree": report.statuses_agree,
        "objectives_agree": report.objectives_agree,
        "solutions_agree": report.solutions_agree,
        "agreement": report.agreement,
        "reference_status": &report.reference_status,
        "reference_objective": report.reference_objective,
        "reference_solution": &report.reference_solution,
        "runs": runs,
    })
}

fn probe_java_tool(opts: &ExternalOptimizationAdapterOptions) -> ExternalOptimizationProbe {
    let artifact = first_configured_env_value(&artifact_env_names(opts.tool));
    if artifact.is_none() {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::NotConfigured,
            command: None,
            message: format!(
                "{} needs a local adapter command or Java classpath; set {} or {}",
                opts.tool.display_name(),
                adapter_env_names(opts.tool)[0],
                artifact_env_names(opts.tool)[0]
            ),
        };
    }
    let java = opts
        .java_command
        .as_ref()
        .cloned()
        .or_else(|| find_first_command(&["java"]));
    let Some(java) = java else {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::RuntimeMissing,
            command: None,
            message: format!(
                "{} classpath is configured, but `java` was not found",
                opts.tool.display_name()
            ),
        };
    };
    ExternalOptimizationProbe {
        tool: opts.tool,
        status: ExternalOptimizationProbeStatus::Ready,
        command: Some(java),
        message: format!(
            "{} Java runtime and artifact configuration are available",
            opts.tool.display_name()
        ),
    }
}

fn probe_rust_tool(opts: &ExternalOptimizationAdapterOptions) -> ExternalOptimizationProbe {
    if first_configured_env_value(&artifact_env_names(opts.tool)).is_some() {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::Ready,
            command: None,
            message: format!(
                "{} Rust crate/native binding configuration is available",
                opts.tool.display_name()
            ),
        };
    }
    if cargo_manifest_contains_any_crate(opts) {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::Ready,
            command: None,
            message: format!(
                "{} appears in the local Cargo manifest",
                opts.tool.display_name()
            ),
        };
    }
    ExternalOptimizationProbe {
        tool: opts.tool,
        status: ExternalOptimizationProbeStatus::NotConfigured,
        command: None,
        message: format!(
            "{} needs a local adapter command, crate dependency, or binding env; set {} or {}",
            opts.tool.display_name(),
            adapter_env_names(opts.tool)[0],
            artifact_env_names(opts.tool)[0]
        ),
    }
}

fn probe_python_tool(opts: &ExternalOptimizationAdapterOptions) -> ExternalOptimizationProbe {
    if first_configured_env_value(&artifact_env_names(opts.tool)).is_some() {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::Ready,
            command: None,
            message: format!(
                "{} Python package/runtime configuration is available",
                opts.tool.display_name()
            ),
        };
    }
    let Some(python) = find_first_command(&["python3", "python"]) else {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::NotConfigured,
            command: None,
            message: format!(
                "{} needs a local adapter command or Python env; set {} or {}",
                opts.tool.display_name(),
                adapter_env_names(opts.tool)[0],
                artifact_env_names(opts.tool)[0]
            ),
        };
    };
    if opts
        .tool
        .python_modules()
        .iter()
        .any(|module| python_can_import(&python, module))
    {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::Ready,
            command: Some(python),
            message: format!("{} Python module is importable", opts.tool.display_name()),
        };
    }
    ExternalOptimizationProbe {
        tool: opts.tool,
        status: ExternalOptimizationProbeStatus::NotConfigured,
        command: Some(python),
        message: format!(
            "{} needs a local adapter command or importable package; set {} or {}",
            opts.tool.display_name(),
            adapter_env_names(opts.tool)[0],
            artifact_env_names(opts.tool)[0]
        ),
    }
}

fn probe_julia_tool(opts: &ExternalOptimizationAdapterOptions) -> ExternalOptimizationProbe {
    if first_configured_env_value(&artifact_env_names(opts.tool)).is_some() {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::Ready,
            command: None,
            message: format!(
                "{} Julia project/runtime configuration is available",
                opts.tool.display_name()
            ),
        };
    }
    ExternalOptimizationProbe {
        tool: opts.tool,
        status: ExternalOptimizationProbeStatus::NotConfigured,
        command: find_first_command(&["julia"]),
        message: format!(
            "{} needs a local adapter command or Julia project; set {} or {}",
            opts.tool.display_name(),
            adapter_env_names(opts.tool)[0],
            artifact_env_names(opts.tool)[0]
        ),
    }
}

fn probe_native_tool(opts: &ExternalOptimizationAdapterOptions) -> ExternalOptimizationProbe {
    if first_configured_env_value(&artifact_env_names(opts.tool)).is_some() {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::Ready,
            command: None,
            message: format!(
                "{} native installation configuration is available",
                opts.tool.display_name()
            ),
        };
    }
    ExternalOptimizationProbe {
        tool: opts.tool,
        status: ExternalOptimizationProbeStatus::NotConfigured,
        command: None,
        message: format!(
            "{} needs a local adapter command or installation directory; set {} or {}",
            opts.tool.display_name(),
            adapter_env_names(opts.tool)[0],
            artifact_env_names(opts.tool)[0]
        ),
    }
}

fn probe_external_optimization_linear_cli_tool(
    opts: &ExternalOptimizationAdapterOptions,
    solver: ExternalLinearCliSolver,
) -> ExternalOptimizationProbe {
    let mut cli_opts = ExternalLinearCliOptions {
        solver,
        command_path: opts.command_path.clone(),
        time_limit_secs: opts.time_limit_secs,
        ..Default::default()
    };
    if cli_opts.time_limit_secs.is_none() {
        cli_opts.time_limit_secs = Some(2.0);
    }
    let probe = probe_external_linear_cli_solver(ExternalLinearCliKind::Lp, &cli_opts);
    let status = match probe.status {
        ExternalLinearCliProbeStatus::Ready => ExternalOptimizationProbeStatus::Ready,
        ExternalLinearCliProbeStatus::NotInstalled => {
            ExternalOptimizationProbeStatus::NotConfigured
        }
        ExternalLinearCliProbeStatus::BridgeUnsupported
        | ExternalLinearCliProbeStatus::SmokeFailed => {
            ExternalOptimizationProbeStatus::AdapterMissing
        }
    };
    ExternalOptimizationProbe {
        tool: opts.tool,
        status,
        command: probe.command,
        message: format!(
            "{} via external_linear_cli {} probe: {}",
            opts.tool.display_name(),
            probe.status.as_str(),
            probe.message
        ),
    }
}

fn python_can_import(python: &Path, module: &str) -> bool {
    Command::new(python)
        .arg("-c")
        .arg(format!("import {module}"))
        .output()
        .is_ok_and(|output| output.status.success())
}

fn cargo_manifest_contains_any_crate(opts: &ExternalOptimizationAdapterOptions) -> bool {
    let crates = opts.tool.cargo_crates();
    if crates.is_empty() {
        return false;
    }
    let manifest_dir = opts
        .cargo_manifest_dir
        .clone()
        .or_else(|| env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    let manifest_path = manifest_dir.join("Cargo.toml");
    let Ok(text) = fs::read_to_string(manifest_path) else {
        return false;
    };
    crates
        .iter()
        .any(|name| text.contains(&format!("{name} =")) || text.contains(&format!("\"{name}\"")))
}

fn configured_adapter_command(tool: ExternalOptimizationTool) -> (Option<PathBuf>, bool) {
    let mut saw_configured = false;
    for env_name in adapter_env_names(tool) {
        let Some(configured) = env::var_os(&env_name) else {
            continue;
        };
        saw_configured = true;
        let configured = PathBuf::from(configured);
        if let Some(path) = resolve_command_path(&configured) {
            return (Some(path), saw_configured);
        }
    }
    (None, saw_configured)
}

fn push_unique_env_name(names: &mut Vec<String>, name: impl Into<String>) {
    let name = name.into();
    if !names.iter().any(|existing| existing == &name) {
        names.push(name);
    }
}

fn find_first_command_in_install_dirs(tool: ExternalOptimizationTool) -> Option<PathBuf> {
    for env_name in external_optimization_command_dir_env_names(tool) {
        let Some(raw_value) = env::var_os(&env_name) else {
            continue;
        };
        if raw_value.to_string_lossy().trim().is_empty() {
            continue;
        }
        for root in env::split_paths(&raw_value) {
            if let Some(path) = find_command_in_install_dir(&root, tool.adapter_command_aliases()) {
                return Some(path);
            }
        }
    }
    None
}

fn find_command_in_install_dir(root: &Path, aliases: &[&str]) -> Option<PathBuf> {
    let mut candidate_dirs = vec![root.to_path_buf(), root.join("bin")];
    if let Ok(children) = fs::read_dir(root) {
        for child in children.flatten() {
            let child_path = child.path();
            if !child_path.is_dir() {
                continue;
            }
            let child_bin = child_path.join("bin");
            candidate_dirs.push(child_bin.clone());
            if let Ok(platform_dirs) = fs::read_dir(&child_bin) {
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
            if let Some(path) = resolve_command_path(&dir.join(alias)) {
                return Some(path);
            }
        }
    }
    None
}

fn first_configured_env_value(names: &[String]) -> Option<String> {
    names.iter().find_map(|name| {
        env::var(name)
            .ok()
            .and_then(|value| (!value.trim().is_empty()).then_some(value))
    })
}

fn env_key(tool: ExternalOptimizationTool) -> String {
    tool.as_str()
        .chars()
        .map(|ch| {
            if ch == '-' {
                '_'
            } else {
                ch.to_ascii_uppercase()
            }
        })
        .collect()
}

fn find_first_command(aliases: &[&str]) -> Option<PathBuf> {
    for alias in aliases {
        if let Some(path) = resolve_command_path(Path::new(alias)) {
            return Some(path);
        }
    }
    None
}

fn resolve_command_path(command: &Path) -> Option<PathBuf> {
    if command.components().count() > 1 {
        return command.is_file().then(|| command.to_path_buf());
    }
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(command);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn first_string_at_any_path(output: &Value, paths: &[&[&str]]) -> Option<String> {
    paths.iter().find_map(|path| {
        value_at_path(output, path).and_then(|value| {
            value
                .as_str()
                .map(|text| text.trim().to_ascii_lowercase())
                .filter(|text| !text.is_empty())
        })
    })
}

fn first_f64_at_any_path(output: &Value, paths: &[&[&str]]) -> Option<f64> {
    paths
        .iter()
        .find_map(|path| value_at_path(output, path).and_then(Value::as_f64))
}

fn first_numeric_vector_at_any_path(output: &Value, paths: &[&[&str]]) -> Option<Vec<f64>> {
    paths
        .iter()
        .find_map(|path| value_at_path(output, path).and_then(numeric_vector_from_value))
}

fn value_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn numeric_vector_from_value(value: &Value) -> Option<Vec<f64>> {
    let items = value.as_array()?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if let Some(value) = item.as_f64() {
            out.push(value);
            continue;
        }
        if let Some(value) = item.get("value").and_then(Value::as_f64) {
            out.push(value);
            continue;
        }
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use crate::des::general::external_optimization_tools::{
        adapter_env_names, artifact_env_names, external_optimization_command_dir_env_names,
        external_optimization_comparison_report_to_json,
        external_optimization_normalized_result_from_value, external_optimization_tool_specs,
        external_optimization_tools, find_command_in_install_dir,
        run_external_optimization_comparison, ExternalLinearCliSolver,
        ExternalOptimizationAdapterInvocation, ExternalOptimizationAdapterOptions,
        ExternalOptimizationAdapterStatus, ExternalOptimizationExactness,
        ExternalOptimizationFamily, ExternalOptimizationLanguage, ExternalOptimizationProbeStatus,
        ExternalOptimizationTool,
    };
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn registry_covers_requested_java_and_rust_ecosystems() {
        let specs = external_optimization_tool_specs();
        assert_eq!(external_optimization_tools().len(), 70);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.language == ExternalOptimizationLanguage::Java)
                .count(),
            12
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.language == ExternalOptimizationLanguage::Rust)
                .count(),
            8
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.language == ExternalOptimizationLanguage::Python)
                .count(),
            18
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.language == ExternalOptimizationLanguage::Julia)
                .count(),
            1
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.language == ExternalOptimizationLanguage::Native)
                .count(),
            31
        );
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::ChocoSolver
                && spec.family == ExternalOptimizationFamily::ConstraintProgramming
                && spec.exactness == ExternalOptimizationExactness::Exact
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::OptaPlanner
                && spec.family == ExternalOptimizationFamily::PlanningMetaheuristic
                && spec.exactness == ExternalOptimizationExactness::Heuristic
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Timefold
                && spec.family == ExternalOptimizationFamily::PlanningMetaheuristic
                && spec.exactness == ExternalOptimizationExactness::Heuristic
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Pyomo
                && spec.family == ExternalOptimizationFamily::LinearMip
                && spec.exactness == ExternalOptimizationExactness::ModelingLayer
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Cpmpy
                && spec.family == ExternalOptimizationFamily::ConstraintProgramming
                && spec.exactness == ExternalOptimizationExactness::ModelingLayer
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Clingo
                && spec.family == ExternalOptimizationFamily::ConstraintProgramming
                && spec.exactness == ExternalOptimizationExactness::Exact
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Cvxpy
                && spec.family == ExternalOptimizationFamily::ConvexOptimization
                && spec.exactness == ExternalOptimizationExactness::ModelingLayer
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::PyScipOpt
                && spec.family == ExternalOptimizationFamily::NativeSolverBinding
                && spec.exactness == ExternalOptimizationExactness::Numerical
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Hexaly
                && spec.family == ExternalOptimizationFamily::HybridOptimization
                && spec.exactness == ExternalOptimizationExactness::Heuristic
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Ipopt
                && spec.family == ExternalOptimizationFamily::NonlinearOptimization
                && spec.exactness == ExternalOptimizationExactness::Numerical
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Mosek
                && spec.family == ExternalOptimizationFamily::ConvexOptimization
                && spec.exactness == ExternalOptimizationExactness::Numerical
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Casadi
                && spec.family == ExternalOptimizationFamily::NonlinearOptimization
                && spec.exactness == ExternalOptimizationExactness::ModelingLayer
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::HighsCli
                && spec.family == ExternalOptimizationFamily::LinearMip
                && spec.exactness == ExternalOptimizationExactness::Exact
                && spec.adapter_command_aliases.is_empty()
        }));
        assert_eq!(
            ExternalOptimizationTool::HighsCli.linear_cli_solver(),
            Some(ExternalLinearCliSolver::Highs)
        );
        assert_eq!(
            ExternalOptimizationTool::LindoCli.linear_cli_solver(),
            Some(ExternalLinearCliSolver::Lindo)
        );
    }

    #[test]
    fn environment_names_are_stable() {
        assert_eq!(
            adapter_env_names(ExternalOptimizationTool::ChocoSolver)[0],
            "ORES_CHOCO_SOLVER_ADAPTER"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::ChocoSolver)[0],
            "ORES_CHOCO_SOLVER_CLASSPATH"
        );
        assert_eq!(
            adapter_env_names(ExternalOptimizationTool::GoodLp)[0],
            "ORES_GOOD_LP_ADAPTER"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::GoodLp)[0],
            "ORES_GOOD_LP_CRATE"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Pyomo)[0],
            "ORES_PYOMO_PYTHON"
        );
        assert_eq!(
            adapter_env_names(ExternalOptimizationTool::Cpmpy)[0],
            "ORES_CPMPY_ADAPTER"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Cpmpy)[0],
            "ORES_CPMPY_PYTHON"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Conjure)[0],
            "ORES_CONJURE_DIR"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Sat4j)[0],
            "ORES_SAT4J_CLASSPATH"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Cvxpy)[0],
            "ORES_CVXPY_PYTHON"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::PythonMip)[0],
            "ORES_PYTHON_MIP_PYTHON"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::GurobiPy)[0],
            "ORES_GUROBIPY_PYTHON"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Jump)[0],
            "ORES_JUMP_JULIA"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Ampl)[0],
            "ORES_AMPL_DIR"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Ipopt)[0],
            "ORES_IPOPT_DIR"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Casadi)[0],
            "ORES_CASADI_PYTHON"
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::HighsCli)[0],
            "ORES_HIGHS_CLI_DIR"
        );
        assert!(artifact_env_names(ExternalOptimizationTool::HighsCli)
            .contains(&"HIGHS_CMD".to_string()));
        assert!(artifact_env_names(ExternalOptimizationTool::GurobiCli)
            .contains(&"GUROBI_CL_CMD".to_string()));
        assert!(artifact_env_names(ExternalOptimizationTool::LindoCli)
            .contains(&"LINDOAPI_CMD".to_string()));
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::Ampl)
                .contains(&"AMPL_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::ChocoSolver)
                .contains(&"CHOCO_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::OptaPlanner)
                .contains(&"OPTAPLANNER_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::Pyomo)
                .contains(&"PYOMO_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::Cpmpy)
                .contains(&"CPMPY_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::OpenWbo)
                .contains(&"OPEN_WBO_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::GurobiPy)
                .contains(&"GUROBI_HOME".to_string())
        );
        assert!(
            !external_optimization_command_dir_env_names(ExternalOptimizationTool::GurobiPy)
                .contains(&"GRB_LICENSE_FILE".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::HighsRust)
                .contains(&"HIGHS_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::Mosek)
                .contains(&"MSKHOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::Symphony)
                .contains(&"COINOR_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::HighsCli)
                .contains(&"HIGHS_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::CplexCli)
                .contains(&"CPLEX_STUDIO_DIR".to_string())
        );
    }

    #[test]
    fn install_dir_lookup_handles_adapter_bin_layouts() {
        let root = std::env::temp_dir().join(format!(
            "des-external-optimization-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let command = root
            .join("hexaly")
            .join("bin")
            .join("macos")
            .join("hexaly-adapter");
        std::fs::create_dir_all(command.parent().unwrap()).unwrap();
        std::fs::write(&command, b"").unwrap();

        assert_eq!(
            find_command_in_install_dir(&root, &["hexaly-adapter"]),
            Some(command)
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn command_override_is_preferred() {
        let opts = ExternalOptimizationAdapterOptions {
            tool: ExternalOptimizationTool::Argmin,
            command_path: Some(PathBuf::from("/opt/local/bin/argmin-adapter")),
            ..Default::default()
        };
        assert_eq!(
            opts.command_path,
            Some(PathBuf::from("/opt/local/bin/argmin-adapter"))
        );
    }

    #[test]
    fn status_strings_are_stable() {
        for (status, expected) in [
            (ExternalOptimizationProbeStatus::Ready, "ready"),
            (
                ExternalOptimizationProbeStatus::NotConfigured,
                "not-configured",
            ),
            (
                ExternalOptimizationProbeStatus::RuntimeMissing,
                "runtime-missing",
            ),
            (
                ExternalOptimizationProbeStatus::AdapterMissing,
                "adapter-missing",
            ),
        ] {
            assert_eq!(status.as_str(), expected);
        }
        for (status, expected) in [
            (ExternalOptimizationAdapterStatus::Ok, "ok"),
            (
                ExternalOptimizationAdapterStatus::Unavailable,
                "unavailable",
            ),
            (ExternalOptimizationAdapterStatus::Failed, "failed"),
            (
                ExternalOptimizationAdapterStatus::InvalidOutput,
                "invalid-output",
            ),
        ] {
            assert_eq!(status.as_str(), expected);
        }
    }

    #[test]
    fn family_language_and_exactness_strings_are_stable() {
        assert_eq!(ExternalOptimizationLanguage::Java.as_str(), "java");
        assert_eq!(ExternalOptimizationLanguage::Python.as_str(), "python");
        assert_eq!(ExternalOptimizationLanguage::Julia.as_str(), "julia");
        assert_eq!(ExternalOptimizationLanguage::Native.as_str(), "native");
        assert_eq!(ExternalOptimizationLanguage::Rust.as_str(), "rust");
        assert_eq!(
            ExternalOptimizationFamily::EvolutionaryMultiObjective.as_str(),
            "evolutionary-multi-objective"
        );
        assert_eq!(
            ExternalOptimizationFamily::ConvexOptimization.as_str(),
            "convex-optimization"
        );
        assert_eq!(
            ExternalOptimizationFamily::HybridOptimization.as_str(),
            "hybrid-optimization"
        );
        assert_eq!(
            ExternalOptimizationExactness::ModelingLayer.as_str(),
            "modeling-layer"
        );
    }

    #[test]
    fn normalized_result_extracts_common_adapter_shapes() {
        let output = json!({
            "result": {
                "status": "OPTIMAL",
                "objective_value": 42.5,
                "solution": [
                    {"name": "x", "value": 1.0},
                    {"name": "y", "value": 0.0}
                ]
            }
        });
        let normalized = external_optimization_normalized_result_from_value(&output);
        assert_eq!(normalized.status.as_deref(), Some("optimal"));
        assert_eq!(normalized.objective, Some(42.5));
        assert_eq!(normalized.solution, Some(vec![1.0, 0.0]));
    }

    #[test]
    fn comparison_report_runs_echo_adapters_and_checks_agreement() {
        let input = json!({
            "status": "optimal",
            "objective": 12.0,
            "x": [1.0, 2.0, 3.0]
        });
        let report = run_external_optimization_comparison(
            &input,
            &[
                ExternalOptimizationAdapterInvocation {
                    label: "good-lp".to_string(),
                    options: ExternalOptimizationAdapterOptions {
                        tool: ExternalOptimizationTool::GoodLp,
                        command_path: Some(PathBuf::from("/bin/cat")),
                        ..Default::default()
                    },
                },
                ExternalOptimizationAdapterInvocation {
                    label: "lp-modeler".to_string(),
                    options: ExternalOptimizationAdapterOptions {
                        tool: ExternalOptimizationTool::LpModeler,
                        command_path: Some(PathBuf::from("/bin/cat")),
                        ..Default::default()
                    },
                },
            ],
            1e-9,
            1e-9,
        );
        assert!(report.agreement);
        assert_eq!(report.reference_status.as_deref(), Some("optimal"));
        assert_eq!(report.reference_objective, Some(12.0));
        assert_eq!(report.reference_solution, Some(vec![1.0, 2.0, 3.0]));
        let report_json = external_optimization_comparison_report_to_json(&report);
        assert_eq!(
            report_json["kind"],
            "external-optimization-comparison-report"
        );
        assert_eq!(report_json["agreement"], true);
    }
}
