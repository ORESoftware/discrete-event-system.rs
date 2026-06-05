//! Local adapter/probe surface for optimization ecosystems that are not plain
//! LP/MIP command-line solvers.
//!
//! Java CP/planning systems and Rust modeling/binding crates are usually wired
//! into an application through a small local wrapper. This module gives those
//! wrappers a stable JSON-in/JSON-out contract while keeping jars, native
//! libraries, and generated executables out of version control.

use super::external_gams_solver_probe::{probe_external_gams_solver, ExternalGamsSolver};
use super::external_linear_cli::{
    probe_external_linear_cli_solver, ExternalLinearCliKind, ExternalLinearCliOptions,
    ExternalLinearCliProbeStatus, ExternalLinearCliSolver,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalOptimizationTool {
    ChocoSolver,
    Jacop,
    IbmCpOptimizer,
    OptaPlanner,
    Timefold,
    FastDownward,
    LpgTd,
    Optic,
    Enhsp,
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
    CplexPython,
    XpressPython,
    Docplex,
    OrToolsPython,
    OrToolsGlop,
    OrToolsPdlp,
    OrToolsCpSat,
    ScipyOptimize,
    MosekPython,
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
    Z3,
    Cvc5,
    Yices,
    Bitwuzla,
    Boolector,
    MathSat,
    OptiMathSat,
    OpenSmt,
    SmtInterpol,
    Princess,
    HighsCli,
    GlpkCli,
    ScipCli,
    CbcCli,
    ClpCli,
    SoplexCli,
    QsoptExCli,
    LpSolveCli,
    GurobiCli,
    CplexCli,
    XpressCli,
    LindoCli,
    GoodLp,
    LpModeler,
    RustLinprog,
    MiniLp,
    Argmin,
    Nlopt,
    OsqpRust,
    ClarabelRust,
    GurobiRust,
    CplexRust,
    IpoptRust,
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
            ExternalOptimizationTool::FastDownward => "fast-downward",
            ExternalOptimizationTool::LpgTd => "lpg-td",
            ExternalOptimizationTool::Optic => "optic",
            ExternalOptimizationTool::Enhsp => "enhsp",
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
            ExternalOptimizationTool::CplexPython => "cplex-python",
            ExternalOptimizationTool::XpressPython => "xpress-python",
            ExternalOptimizationTool::Docplex => "docplex",
            ExternalOptimizationTool::OrToolsPython => "ortools-python",
            ExternalOptimizationTool::OrToolsGlop => "ortools-glop",
            ExternalOptimizationTool::OrToolsPdlp => "ortools-pdlp",
            ExternalOptimizationTool::OrToolsCpSat => "ortools-cp-sat",
            ExternalOptimizationTool::ScipyOptimize => "scipy-optimize",
            ExternalOptimizationTool::MosekPython => "mosek-python",
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
            ExternalOptimizationTool::Z3 => "z3",
            ExternalOptimizationTool::Cvc5 => "cvc5",
            ExternalOptimizationTool::Yices => "yices",
            ExternalOptimizationTool::Bitwuzla => "bitwuzla",
            ExternalOptimizationTool::Boolector => "boolector",
            ExternalOptimizationTool::MathSat => "mathsat",
            ExternalOptimizationTool::OptiMathSat => "optimathsat",
            ExternalOptimizationTool::OpenSmt => "opensmt",
            ExternalOptimizationTool::SmtInterpol => "smtinterpol",
            ExternalOptimizationTool::Princess => "princess",
            ExternalOptimizationTool::HighsCli => "highs-cli",
            ExternalOptimizationTool::GlpkCli => "glpk-cli",
            ExternalOptimizationTool::ScipCli => "scip-cli",
            ExternalOptimizationTool::CbcCli => "cbc-cli",
            ExternalOptimizationTool::ClpCli => "clp-cli",
            ExternalOptimizationTool::SoplexCli => "soplex-cli",
            ExternalOptimizationTool::QsoptExCli => "qsopt-ex-cli",
            ExternalOptimizationTool::LpSolveCli => "lp-solve-cli",
            ExternalOptimizationTool::GurobiCli => "gurobi-cli",
            ExternalOptimizationTool::CplexCli => "cplex-cli",
            ExternalOptimizationTool::XpressCli => "xpress-cli",
            ExternalOptimizationTool::LindoCli => "lindo-cli",
            ExternalOptimizationTool::GoodLp => "good-lp",
            ExternalOptimizationTool::LpModeler => "lp-modeler",
            ExternalOptimizationTool::RustLinprog => "rust-linprog",
            ExternalOptimizationTool::MiniLp => "minilp",
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::Nlopt => "nlopt",
            ExternalOptimizationTool::OsqpRust => "osqp-rust",
            ExternalOptimizationTool::ClarabelRust => "clarabel-rust",
            ExternalOptimizationTool::GurobiRust => "gurobi-rust",
            ExternalOptimizationTool::CplexRust => "cplex-rust",
            ExternalOptimizationTool::IpoptRust => "ipopt-rust",
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
            ExternalOptimizationTool::FastDownward => "Fast Downward",
            ExternalOptimizationTool::LpgTd => "LPG-td",
            ExternalOptimizationTool::Optic => "OPTIC",
            ExternalOptimizationTool::Enhsp => "ENHSP",
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
            ExternalOptimizationTool::CplexPython => "IBM ILOG CPLEX Python API",
            ExternalOptimizationTool::XpressPython => "FICO Xpress Python API",
            ExternalOptimizationTool::Docplex => "DOcplex",
            ExternalOptimizationTool::OrToolsPython => "Google OR-Tools Python",
            ExternalOptimizationTool::OrToolsGlop => "Google OR-Tools GLOP",
            ExternalOptimizationTool::OrToolsPdlp => "Google OR-Tools PDLP",
            ExternalOptimizationTool::OrToolsCpSat => "Google OR-Tools CP-SAT FlatZinc",
            ExternalOptimizationTool::ScipyOptimize => "SciPy optimize",
            ExternalOptimizationTool::MosekPython => "MOSEK Python API",
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
            ExternalOptimizationTool::Z3 => "Z3",
            ExternalOptimizationTool::Cvc5 => "cvc5",
            ExternalOptimizationTool::Yices => "Yices",
            ExternalOptimizationTool::Bitwuzla => "Bitwuzla",
            ExternalOptimizationTool::Boolector => "Boolector",
            ExternalOptimizationTool::MathSat => "MathSAT",
            ExternalOptimizationTool::OptiMathSat => "OptiMathSAT",
            ExternalOptimizationTool::OpenSmt => "OpenSMT",
            ExternalOptimizationTool::SmtInterpol => "SMTInterpol",
            ExternalOptimizationTool::Princess => "Princess",
            ExternalOptimizationTool::HighsCli => "HiGHS CLI",
            ExternalOptimizationTool::GlpkCli => "GLPK glpsol CLI",
            ExternalOptimizationTool::ScipCli => "SCIP CLI",
            ExternalOptimizationTool::CbcCli => "COIN-OR CBC CLI",
            ExternalOptimizationTool::ClpCli => "COIN-OR CLP CLI",
            ExternalOptimizationTool::SoplexCli => "SoPlex CLI",
            ExternalOptimizationTool::QsoptExCli => "QSopt_ex CLI",
            ExternalOptimizationTool::LpSolveCli => "lp_solve CLI",
            ExternalOptimizationTool::GurobiCli => "Gurobi Optimizer CLI",
            ExternalOptimizationTool::CplexCli => "IBM ILOG CPLEX Optimizer CLI",
            ExternalOptimizationTool::XpressCli => "FICO Xpress Optimizer CLI",
            ExternalOptimizationTool::LindoCli => "LINDO Systems CLI",
            ExternalOptimizationTool::GoodLp => "good_lp",
            ExternalOptimizationTool::LpModeler => "lp-modeler",
            ExternalOptimizationTool::RustLinprog => "rust-linprog",
            ExternalOptimizationTool::MiniLp => "minilp",
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::Nlopt => "NLopt Rust bindings",
            ExternalOptimizationTool::OsqpRust => "OSQP Rust bindings",
            ExternalOptimizationTool::ClarabelRust => "Clarabel Rust crate",
            ExternalOptimizationTool::GurobiRust => "Gurobi Rust bindings",
            ExternalOptimizationTool::CplexRust => "CPLEX Rust bindings",
            ExternalOptimizationTool::IpoptRust => "Ipopt Rust bindings",
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
            | ExternalOptimizationTool::CplexPython
            | ExternalOptimizationTool::XpressPython
            | ExternalOptimizationTool::Docplex
            | ExternalOptimizationTool::OrToolsPython
            | ExternalOptimizationTool::OrToolsGlop
            | ExternalOptimizationTool::OrToolsPdlp
            | ExternalOptimizationTool::ScipyOptimize
            | ExternalOptimizationTool::MosekPython
            | ExternalOptimizationTool::Casadi
            | ExternalOptimizationTool::Osqp
            | ExternalOptimizationTool::Scs
            | ExternalOptimizationTool::Clarabel
            | ExternalOptimizationTool::Ecos
            | ExternalOptimizationTool::Copt
            | ExternalOptimizationTool::Clingo
            | ExternalOptimizationTool::Cvc5
            | ExternalOptimizationTool::Proxqp
            | ExternalOptimizationTool::Bitwuzla => ExternalOptimizationLanguage::Python,
            ExternalOptimizationTool::Jump => ExternalOptimizationLanguage::Julia,
            ExternalOptimizationTool::Ampl
            | ExternalOptimizationTool::Gams
            | ExternalOptimizationTool::OrToolsCpSat
            | ExternalOptimizationTool::Hexaly
            | ExternalOptimizationTool::FastDownward
            | ExternalOptimizationTool::LpgTd
            | ExternalOptimizationTool::Optic
            | ExternalOptimizationTool::Enhsp
            | ExternalOptimizationTool::Minotaur
            | ExternalOptimizationTool::Symphony
            | ExternalOptimizationTool::Ipopt
            | ExternalOptimizationTool::Bonmin
            | ExternalOptimizationTool::Couenne
            | ExternalOptimizationTool::Knitro
            | ExternalOptimizationTool::Mosek
            | ExternalOptimizationTool::Baron
            | ExternalOptimizationTool::Qpoases
            | ExternalOptimizationTool::Cosmo
            | ExternalOptimizationTool::Sdpa
            | ExternalOptimizationTool::Csdp
            | ExternalOptimizationTool::Z3
            | ExternalOptimizationTool::Yices
            | ExternalOptimizationTool::Boolector
            | ExternalOptimizationTool::MathSat
            | ExternalOptimizationTool::OptiMathSat
            | ExternalOptimizationTool::OpenSmt
            | ExternalOptimizationTool::SmtInterpol
            | ExternalOptimizationTool::Princess
            | ExternalOptimizationTool::HighsCli
            | ExternalOptimizationTool::GlpkCli
            | ExternalOptimizationTool::ScipCli
            | ExternalOptimizationTool::CbcCli
            | ExternalOptimizationTool::ClpCli
            | ExternalOptimizationTool::SoplexCli
            | ExternalOptimizationTool::QsoptExCli
            | ExternalOptimizationTool::LpSolveCli
            | ExternalOptimizationTool::GurobiCli
            | ExternalOptimizationTool::CplexCli
            | ExternalOptimizationTool::XpressCli
            | ExternalOptimizationTool::LindoCli
            | ExternalOptimizationTool::Conjure
            | ExternalOptimizationTool::Picat
            | ExternalOptimizationTool::Clingcon
            | ExternalOptimizationTool::OpenWbo => ExternalOptimizationLanguage::Native,
            ExternalOptimizationTool::GoodLp
            | ExternalOptimizationTool::LpModeler
            | ExternalOptimizationTool::RustLinprog
            | ExternalOptimizationTool::MiniLp
            | ExternalOptimizationTool::Argmin
            | ExternalOptimizationTool::Nlopt
            | ExternalOptimizationTool::OsqpRust
            | ExternalOptimizationTool::ClarabelRust
            | ExternalOptimizationTool::GurobiRust
            | ExternalOptimizationTool::CplexRust
            | ExternalOptimizationTool::IpoptRust
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
            ExternalOptimizationTool::FastDownward
            | ExternalOptimizationTool::LpgTd
            | ExternalOptimizationTool::Optic
            | ExternalOptimizationTool::Enhsp => ExternalOptimizationFamily::AiPlanning,
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
            | ExternalOptimizationTool::OrToolsGlop
            | ExternalOptimizationTool::OrToolsPdlp
            | ExternalOptimizationTool::HighsCli
            | ExternalOptimizationTool::GlpkCli
            | ExternalOptimizationTool::ScipCli
            | ExternalOptimizationTool::CbcCli
            | ExternalOptimizationTool::ClpCli
            | ExternalOptimizationTool::SoplexCli
            | ExternalOptimizationTool::QsoptExCli
            | ExternalOptimizationTool::LpSolveCli
            | ExternalOptimizationTool::GurobiCli
            | ExternalOptimizationTool::CplexCli
            | ExternalOptimizationTool::XpressCli
            | ExternalOptimizationTool::LindoCli
            | ExternalOptimizationTool::GoodLp
            | ExternalOptimizationTool::LpModeler
            | ExternalOptimizationTool::RustLinprog
            | ExternalOptimizationTool::MiniLp => ExternalOptimizationFamily::LinearMip,
            ExternalOptimizationTool::Cvxpy
            | ExternalOptimizationTool::Cvxopt
            | ExternalOptimizationTool::MosekPython
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
            | ExternalOptimizationTool::Csdp
            | ExternalOptimizationTool::OsqpRust
            | ExternalOptimizationTool::ClarabelRust => {
                ExternalOptimizationFamily::ConvexOptimization
            }
            ExternalOptimizationTool::Z3
            | ExternalOptimizationTool::Cvc5
            | ExternalOptimizationTool::Yices
            | ExternalOptimizationTool::Bitwuzla
            | ExternalOptimizationTool::Boolector
            | ExternalOptimizationTool::MathSat
            | ExternalOptimizationTool::OptiMathSat
            | ExternalOptimizationTool::OpenSmt
            | ExternalOptimizationTool::SmtInterpol
            | ExternalOptimizationTool::Princess => ExternalOptimizationFamily::SmtOmt,
            ExternalOptimizationTool::OrToolsJava
            | ExternalOptimizationTool::OrToolsPython
            | ExternalOptimizationTool::OrToolsCpSat => ExternalOptimizationFamily::CpSatRouting,
            ExternalOptimizationTool::Hexaly => ExternalOptimizationFamily::HybridOptimization,
            ExternalOptimizationTool::Argmin
            | ExternalOptimizationTool::Nlopt
            | ExternalOptimizationTool::ScipyOptimize
            | ExternalOptimizationTool::Minotaur
            | ExternalOptimizationTool::Ipopt
            | ExternalOptimizationTool::IpoptRust
            | ExternalOptimizationTool::Bonmin
            | ExternalOptimizationTool::Couenne
            | ExternalOptimizationTool::Knitro
            | ExternalOptimizationTool::Baron
            | ExternalOptimizationTool::Casadi => ExternalOptimizationFamily::NonlinearOptimization,
            ExternalOptimizationTool::GurobiRust
            | ExternalOptimizationTool::CplexRust
            | ExternalOptimizationTool::HighsRust
            | ExternalOptimizationTool::ScipRust
            | ExternalOptimizationTool::CbcRust
            | ExternalOptimizationTool::PyScipOpt
            | ExternalOptimizationTool::GurobiPy
            | ExternalOptimizationTool::CplexPython
            | ExternalOptimizationTool::XpressPython => {
                ExternalOptimizationFamily::NativeSolverBinding
            }
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
            | ExternalOptimizationTool::OrToolsCpSat
            | ExternalOptimizationTool::RustLinprog
            | ExternalOptimizationTool::MiniLp
            | ExternalOptimizationTool::HighsRust
            | ExternalOptimizationTool::ScipRust
            | ExternalOptimizationTool::CbcRust
            | ExternalOptimizationTool::Picat
            | ExternalOptimizationTool::Clingo
            | ExternalOptimizationTool::Clingcon
            | ExternalOptimizationTool::Sat4j
            | ExternalOptimizationTool::PySat
            | ExternalOptimizationTool::OpenWbo
            | ExternalOptimizationTool::FastDownward
            | ExternalOptimizationTool::HighsCli
            | ExternalOptimizationTool::GlpkCli
            | ExternalOptimizationTool::ScipCli
            | ExternalOptimizationTool::CbcCli
            | ExternalOptimizationTool::ClpCli
            | ExternalOptimizationTool::SoplexCli
            | ExternalOptimizationTool::QsoptExCli
            | ExternalOptimizationTool::LpSolveCli
            | ExternalOptimizationTool::Symphony
            | ExternalOptimizationTool::GurobiCli
            | ExternalOptimizationTool::CplexCli
            | ExternalOptimizationTool::XpressCli
            | ExternalOptimizationTool::LindoCli
            | ExternalOptimizationTool::Z3
            | ExternalOptimizationTool::Cvc5
            | ExternalOptimizationTool::Yices
            | ExternalOptimizationTool::Bitwuzla
            | ExternalOptimizationTool::Boolector
            | ExternalOptimizationTool::MathSat
            | ExternalOptimizationTool::OptiMathSat
            | ExternalOptimizationTool::OpenSmt
            | ExternalOptimizationTool::SmtInterpol
            | ExternalOptimizationTool::Princess => ExternalOptimizationExactness::Exact,
            ExternalOptimizationTool::OptaPlanner
            | ExternalOptimizationTool::Timefold
            | ExternalOptimizationTool::Hexaly
            | ExternalOptimizationTool::LpgTd
            | ExternalOptimizationTool::Optic
            | ExternalOptimizationTool::Enhsp
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
            | ExternalOptimizationTool::OsqpRust
            | ExternalOptimizationTool::ClarabelRust
            | ExternalOptimizationTool::GurobiRust
            | ExternalOptimizationTool::CplexRust
            | ExternalOptimizationTool::IpoptRust
            | ExternalOptimizationTool::OrToolsGlop
            | ExternalOptimizationTool::OrToolsPdlp
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
            | ExternalOptimizationTool::CplexPython
            | ExternalOptimizationTool::XpressPython
            | ExternalOptimizationTool::MosekPython
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
            ExternalOptimizationTool::SoplexCli => Some(ExternalLinearCliSolver::Soplex),
            ExternalOptimizationTool::QsoptExCli => Some(ExternalLinearCliSolver::QsoptEx),
            ExternalOptimizationTool::LpSolveCli => Some(ExternalLinearCliSolver::LpSolve),
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
            ExternalOptimizationTool::FastDownward => {
                &["ores-fast-downward-adapter", "fast-downward-adapter"]
            }
            ExternalOptimizationTool::LpgTd => &["ores-lpg-td-adapter", "lpg-td-adapter"],
            ExternalOptimizationTool::Optic => &["ores-optic-adapter", "optic-adapter"],
            ExternalOptimizationTool::Enhsp => &["ores-enhsp-adapter", "enhsp-adapter"],
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
            ExternalOptimizationTool::CplexPython => {
                &["ores-cplex-python-adapter", "cplex-python-adapter"]
            }
            ExternalOptimizationTool::XpressPython => {
                &["ores-xpress-python-adapter", "xpress-python-adapter"]
            }
            ExternalOptimizationTool::Docplex => &["ores-docplex-adapter", "docplex-adapter"],
            ExternalOptimizationTool::OrToolsPython => {
                &["ores-ortools-python-adapter", "ortools-python-adapter"]
            }
            ExternalOptimizationTool::OrToolsGlop => {
                &["ores-ortools-glop-adapter", "ortools-glop-adapter"]
            }
            ExternalOptimizationTool::OrToolsPdlp => {
                &["ores-ortools-pdlp-adapter", "ortools-pdlp-adapter"]
            }
            ExternalOptimizationTool::ScipyOptimize => {
                &["ores-scipy-optimize-adapter", "scipy-optimize-adapter"]
            }
            ExternalOptimizationTool::MosekPython => {
                &["ores-mosek-python-adapter", "mosek-python-adapter"]
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
            ExternalOptimizationTool::Z3 => &["ores-z3-adapter", "z3-adapter", "z3"],
            ExternalOptimizationTool::Cvc5 => &["ores-cvc5-adapter", "cvc5-adapter", "cvc5"],
            ExternalOptimizationTool::Yices => {
                &["ores-yices-adapter", "yices-adapter", "yices-smt2", "yices"]
            }
            ExternalOptimizationTool::Bitwuzla => {
                &["ores-bitwuzla-adapter", "bitwuzla-adapter", "bitwuzla"]
            }
            ExternalOptimizationTool::Boolector => {
                &["ores-boolector-adapter", "boolector-adapter", "boolector"]
            }
            ExternalOptimizationTool::MathSat => &[
                "ores-mathsat-adapter",
                "mathsat-adapter",
                "mathsat",
                "mathsat5",
            ],
            ExternalOptimizationTool::OptiMathSat => &[
                "ores-optimathsat-adapter",
                "optimathsat-adapter",
                "optimathsat",
                "optimathsat5",
            ],
            ExternalOptimizationTool::OpenSmt => &[
                "ores-opensmt-adapter",
                "opensmt-adapter",
                "opensmt",
                "opensmt2",
            ],
            ExternalOptimizationTool::SmtInterpol => &[
                "ores-smtinterpol-adapter",
                "smtinterpol-adapter",
                "smtinterpol",
                "smtinterpol.sh",
            ],
            ExternalOptimizationTool::Princess => &[
                "ores-princess-adapter",
                "princess-adapter",
                "princess",
                "princess-smt",
            ],
            ExternalOptimizationTool::HighsCli
            | ExternalOptimizationTool::GlpkCli
            | ExternalOptimizationTool::ScipCli
            | ExternalOptimizationTool::CbcCli
            | ExternalOptimizationTool::ClpCli
            | ExternalOptimizationTool::SoplexCli
            | ExternalOptimizationTool::QsoptExCli
            | ExternalOptimizationTool::LpSolveCli
            | ExternalOptimizationTool::GurobiCli
            | ExternalOptimizationTool::CplexCli
            | ExternalOptimizationTool::XpressCli
            | ExternalOptimizationTool::LindoCli => &[],
            ExternalOptimizationTool::OrToolsCpSat => &["fzn-cp-sat"],
            ExternalOptimizationTool::GoodLp => &["ores-good-lp-adapter", "good-lp-adapter"],
            ExternalOptimizationTool::LpModeler => {
                &["ores-lp-modeler-adapter", "lp-modeler-adapter"]
            }
            ExternalOptimizationTool::RustLinprog => {
                &["ores-rust-linprog-adapter", "rust-linprog-adapter"]
            }
            ExternalOptimizationTool::MiniLp => &["ores-minilp-adapter", "minilp-adapter"],
            ExternalOptimizationTool::Argmin => &["ores-argmin-adapter", "argmin-adapter"],
            ExternalOptimizationTool::Nlopt => &["ores-nlopt-adapter", "nlopt-adapter"],
            ExternalOptimizationTool::OsqpRust => &["ores-osqp-rust-adapter", "osqp-rust-adapter"],
            ExternalOptimizationTool::ClarabelRust => {
                &["ores-clarabel-rust-adapter", "clarabel-rust-adapter"]
            }
            ExternalOptimizationTool::GurobiRust => {
                &["ores-gurobi-rust-adapter", "gurobi-rust-adapter"]
            }
            ExternalOptimizationTool::CplexRust => {
                &["ores-cplex-rust-adapter", "cplex-rust-adapter"]
            }
            ExternalOptimizationTool::IpoptRust => {
                &["ores-ipopt-rust-adapter", "ipopt-rust-adapter"]
            }
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
            ExternalOptimizationTool::MiniLp => &["minilp"],
            ExternalOptimizationTool::Argmin => &["argmin"],
            ExternalOptimizationTool::Nlopt => &["nlopt", "nlopt-rs", "nlopt-sys"],
            ExternalOptimizationTool::OsqpRust => &["osqp", "osqp-sys"],
            ExternalOptimizationTool::ClarabelRust => &["clarabel"],
            ExternalOptimizationTool::GurobiRust => &["grb", "gurobi"],
            ExternalOptimizationTool::CplexRust => &["cplex-rs", "cplex-rs-sys", "cplex_sys"],
            ExternalOptimizationTool::IpoptRust => &["ipopt", "ipopt-sys"],
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
            ExternalOptimizationTool::Clingo => &["clingo"],
            ExternalOptimizationTool::Cvc5 => &["cvc5"],
            ExternalOptimizationTool::PySat => &["pysat"],
            ExternalOptimizationTool::Pulp => &["pulp"],
            ExternalOptimizationTool::Cvxpy => &["cvxpy"],
            ExternalOptimizationTool::Cvxopt => &["cvxopt"],
            ExternalOptimizationTool::PyScipOpt => &["pyscipopt"],
            ExternalOptimizationTool::PythonMip => &["mip"],
            ExternalOptimizationTool::GurobiPy => &["gurobipy"],
            ExternalOptimizationTool::CplexPython => &["cplex"],
            ExternalOptimizationTool::XpressPython => &["xpress"],
            ExternalOptimizationTool::Docplex => &["docplex"],
            ExternalOptimizationTool::OrToolsPython
            | ExternalOptimizationTool::OrToolsGlop
            | ExternalOptimizationTool::OrToolsPdlp => &["ortools"],
            ExternalOptimizationTool::ScipyOptimize => &["scipy"],
            ExternalOptimizationTool::MosekPython => &["mosek"],
            ExternalOptimizationTool::Casadi => &["casadi"],
            ExternalOptimizationTool::Osqp => &["osqp"],
            ExternalOptimizationTool::Scs => &["scs"],
            ExternalOptimizationTool::Clarabel => &["clarabel"],
            ExternalOptimizationTool::Ecos => &["ecos"],
            ExternalOptimizationTool::Copt => &["coptpy"],
            ExternalOptimizationTool::Proxqp => &["proxsuite"],
            ExternalOptimizationTool::Bitwuzla => &["bitwuzla"],
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
            ExternalOptimizationTool::FastDownward => {
                "Classical PDDL planner for independent plan and scheduling cross-checks"
            }
            ExternalOptimizationTool::LpgTd => {
                "Temporal PDDL planner for schedule and plan validation adapters"
            }
            ExternalOptimizationTool::Optic => {
                "Temporal and numeric PDDL planner for plan optimization cross-checks"
            }
            ExternalOptimizationTool::Enhsp => {
                "Hybrid/numeric PDDL planner for planning-model validation"
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
            ExternalOptimizationTool::CplexPython => {
                "Official IBM ILOG CPLEX Python API for direct model, parameter, and solution checks"
            }
            ExternalOptimizationTool::XpressPython => {
                "Official FICO Xpress Python API for direct optimizer model and solution checks"
            }
            ExternalOptimizationTool::Docplex => {
                "IBM DOcplex object-oriented Python modeling API for CPLEX and CP Optimizer"
            }
            ExternalOptimizationTool::OrToolsPython => {
                "Python API surface for OR-Tools CP-SAT, routing, and linear solver validation"
            }
            ExternalOptimizationTool::OrToolsGlop => {
                "OR-Tools GLOP linear-programming engine for LP same-input cross-checks"
            }
            ExternalOptimizationTool::OrToolsPdlp => {
                "OR-Tools PDLP first-order linear-programming engine for large sparse LP cross-checks"
            }
            ExternalOptimizationTool::OrToolsCpSat => {
                "OR-Tools CP-SAT FlatZinc executable for MiniZinc-model cross-checks"
            }
            ExternalOptimizationTool::ScipyOptimize => {
                "SciPy numerical optimization routines for nonlinear and least-squares reference checks"
            }
            ExternalOptimizationTool::MosekPython => {
                "Official MOSEK Python API for LP, conic, convex, and mixed-integer model cross-checks"
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
                "Commercial LP/QP/QCP/MIP and conic optimization solver via CLI or Python API"
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
            ExternalOptimizationTool::Z3 => {
                "SMT/OMT solver for SMT-LIB Optimize, MaxSMT, and feasibility cross-checks"
            }
            ExternalOptimizationTool::Cvc5 => {
                "SMT solver with optimization support for SMT-LIB model and objective validation"
            }
            ExternalOptimizationTool::Yices => {
                "SMT solver for bit-vector, arithmetic, and finite-domain feasibility validation"
            }
            ExternalOptimizationTool::Bitwuzla => {
                "SMT solver for bit-vector and array-heavy optimization model cross-checks"
            }
            ExternalOptimizationTool::Boolector => {
                "SMT solver for bit-vector and array feasibility cross-checks"
            }
            ExternalOptimizationTool::MathSat => {
                "SMT solver for arithmetic, bit-vector, and array feasibility cross-checks"
            }
            ExternalOptimizationTool::OptiMathSat => {
                "Optimization-modulo-theories solver for SMT-LIB Optimize and MaxSMT checks"
            }
            ExternalOptimizationTool::OpenSmt => {
                "Open-source SMT solver for independent SMT-LIB feasibility checks"
            }
            ExternalOptimizationTool::SmtInterpol => {
                "Interpolating SMT solver for proof-oriented feasibility cross-checks"
            }
            ExternalOptimizationTool::Princess => {
                "SMT solver and theorem prover for integer-arithmetic model checks"
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
            ExternalOptimizationTool::SoplexCli => {
                "Native SoPlex command-line bridge for local LP cross-validation and rational-mode checks"
            }
            ExternalOptimizationTool::QsoptExCli => {
                "Native QSopt_ex exact-rational LP command-line bridge for local optimum validation"
            }
            ExternalOptimizationTool::LpSolveCli => {
                "Native lp_solve command-line bridge for local LP/MIP cross-validation"
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
            ExternalOptimizationTool::MiniLp => "Rust-first lightweight LP solver crate",
            ExternalOptimizationTool::Argmin => {
                "Rust nonlinear optimization algorithms for gradient and derivative-free runs"
            }
            ExternalOptimizationTool::Nlopt => {
                "Rust bindings to NLopt nonlinear optimization algorithms"
            }
            ExternalOptimizationTool::OsqpRust => {
                "Rust bindings to OSQP for local quadratic-program checks"
            }
            ExternalOptimizationTool::ClarabelRust => {
                "Rust-native Clarabel crate for conic and quadratic optimization checks"
            }
            ExternalOptimizationTool::GurobiRust => {
                "Rust bindings to Gurobi Optimizer using local, non-vendored solver libraries"
            }
            ExternalOptimizationTool::CplexRust => {
                "Rust bindings to IBM ILOG CPLEX using local, non-vendored solver libraries"
            }
            ExternalOptimizationTool::IpoptRust => {
                "Rust bindings to Ipopt nonlinear optimization using local native libraries"
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
    AiPlanning,
    EvolutionaryMultiObjective,
    LinearMip,
    CpSatRouting,
    ConvexOptimization,
    NonlinearOptimization,
    HybridOptimization,
    SmtOmt,
    NativeSolverBinding,
}

impl ExternalOptimizationFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationFamily::ConstraintProgramming => "constraint-programming",
            ExternalOptimizationFamily::PlanningMetaheuristic => "planning-metaheuristic",
            ExternalOptimizationFamily::AiPlanning => "ai-planning",
            ExternalOptimizationFamily::EvolutionaryMultiObjective => {
                "evolutionary-multi-objective"
            }
            ExternalOptimizationFamily::LinearMip => "linear-mip",
            ExternalOptimizationFamily::CpSatRouting => "cp-sat-routing",
            ExternalOptimizationFamily::ConvexOptimization => "convex-optimization",
            ExternalOptimizationFamily::NonlinearOptimization => "nonlinear-optimization",
            ExternalOptimizationFamily::HybridOptimization => "hybrid-optimization",
            ExternalOptimizationFamily::SmtOmt => "smt-omt",
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
        ExternalOptimizationTool::FastDownward,
        ExternalOptimizationTool::LpgTd,
        ExternalOptimizationTool::Optic,
        ExternalOptimizationTool::Enhsp,
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
        ExternalOptimizationTool::CplexPython,
        ExternalOptimizationTool::XpressPython,
        ExternalOptimizationTool::Docplex,
        ExternalOptimizationTool::OrToolsPython,
        ExternalOptimizationTool::OrToolsGlop,
        ExternalOptimizationTool::OrToolsPdlp,
        ExternalOptimizationTool::OrToolsCpSat,
        ExternalOptimizationTool::ScipyOptimize,
        ExternalOptimizationTool::MosekPython,
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
        ExternalOptimizationTool::Z3,
        ExternalOptimizationTool::Cvc5,
        ExternalOptimizationTool::Yices,
        ExternalOptimizationTool::Bitwuzla,
        ExternalOptimizationTool::Boolector,
        ExternalOptimizationTool::MathSat,
        ExternalOptimizationTool::OptiMathSat,
        ExternalOptimizationTool::OpenSmt,
        ExternalOptimizationTool::SmtInterpol,
        ExternalOptimizationTool::Princess,
        ExternalOptimizationTool::HighsCli,
        ExternalOptimizationTool::GlpkCli,
        ExternalOptimizationTool::ScipCli,
        ExternalOptimizationTool::CbcCli,
        ExternalOptimizationTool::ClpCli,
        ExternalOptimizationTool::SoplexCli,
        ExternalOptimizationTool::QsoptExCli,
        ExternalOptimizationTool::LpSolveCli,
        ExternalOptimizationTool::GurobiCli,
        ExternalOptimizationTool::CplexCli,
        ExternalOptimizationTool::XpressCli,
        ExternalOptimizationTool::LindoCli,
        ExternalOptimizationTool::GoodLp,
        ExternalOptimizationTool::LpModeler,
        ExternalOptimizationTool::RustLinprog,
        ExternalOptimizationTool::MiniLp,
        ExternalOptimizationTool::Argmin,
        ExternalOptimizationTool::Nlopt,
        ExternalOptimizationTool::OsqpRust,
        ExternalOptimizationTool::ClarabelRust,
        ExternalOptimizationTool::GurobiRust,
        ExternalOptimizationTool::CplexRust,
        ExternalOptimizationTool::IpoptRust,
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

pub fn external_optimization_ecosystem_reference_script() -> PathBuf {
    let root = env::var_os("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts")
        .join("optimization_ecosystem_reference.py")
}

pub fn external_optimization_ecosystem_reference_options(
    tool: ExternalOptimizationTool,
) -> ExternalOptimizationAdapterOptions {
    let python = env::var_os("PYTHON_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"));
    let script = external_optimization_ecosystem_reference_script();
    let working_dir = script
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    ExternalOptimizationAdapterOptions {
        tool,
        command_path: Some(python),
        working_dir,
        extra_args: vec![
            script.to_string_lossy().to_string(),
            "--tool".to_string(),
            tool.as_str().to_string(),
        ],
        ..Default::default()
    }
}

pub fn external_optimization_ecosystem_reference_invocation(
    label: impl Into<String>,
    tool: ExternalOptimizationTool,
) -> ExternalOptimizationAdapterInvocation {
    ExternalOptimizationAdapterInvocation {
        label: label.into(),
        options: external_optimization_ecosystem_reference_options(tool),
    }
}

pub fn run_external_optimization_ecosystem_reference(
    input: &Value,
    tool: ExternalOptimizationTool,
) -> ExternalOptimizationAdapterRun {
    if let Some(run) = run_native_external_optimization_ecosystem_reference(input, tool) {
        return run;
    }
    let options = external_optimization_ecosystem_reference_options(tool);
    run_external_optimization_adapter(input, &options)
}

pub fn run_external_optimization_ecosystem_reference_with_rust_builtin(
    input: &Value,
    tool: ExternalOptimizationTool,
) -> ExternalOptimizationAdapterRun {
    if let Some(run) = run_native_external_optimization_ecosystem_reference(input, tool) {
        return run;
    }
    ExternalOptimizationAdapterRun {
        tool,
        status: ExternalOptimizationAdapterStatus::Unavailable,
        output: Some(json!({
            "kind": "optimization-ecosystem-reference-result",
            "tool": tool.as_str(),
            "family": "unknown",
            "status": "unsupported",
            "objective": null,
            "x": null,
            "message": "no Rust builtin ecosystem reference for this tool/payload",
            "backend": "builtin-rust:unavailable",
        })),
        elapsed_ms: 0.0,
        message: "no Rust builtin ecosystem reference for this tool/payload".to_string(),
    }
}

#[derive(Clone, Debug)]
struct NativeExternalOptimizationEcosystemResult {
    status: &'static str,
    objective: Option<f64>,
    x: Option<Vec<f64>>,
    message: String,
    extra: Vec<(String, Value)>,
}

impl NativeExternalOptimizationEcosystemResult {
    fn optimal(objective: f64, x: Vec<f64>) -> Self {
        Self {
            status: "optimal",
            objective: Some(objective),
            x: Some(x),
            message: String::new(),
            extra: Vec::new(),
        }
    }

    fn status(status: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            objective: None,
            x: None,
            message: message.into(),
            extra: Vec::new(),
        }
    }

    fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.push((key.into(), value));
        self
    }
}

fn run_native_external_optimization_ecosystem_reference(
    input: &Value,
    tool: ExternalOptimizationTool,
) -> Option<ExternalOptimizationAdapterRun> {
    let started = Instant::now();
    input.as_object()?;
    let kind = input
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let family = native_external_optimization_ecosystem_family(tool, &kind)?;
    let result = match family {
        "constraint-programming" | "smt-omt" => {
            if native_ecosystem_kind_is_cp_job_shop(&kind) {
                native_solve_ecosystem_cp_job_shop(input)
            } else {
                native_solve_ecosystem_cp_assignment(input)
            }
        }
        "planning-metaheuristic" | "ai-planning" => {
            native_solve_ecosystem_planning_assignment(input)
        }
        "evolutionary-multiobjective" => native_solve_ecosystem_multiobjective(input),
        "nonlinear-optimization" => native_solve_ecosystem_nonlinear(input),
        "linear-mip" | "convex-optimization" | "native-solver-binding" | "hybrid-optimization" => {
            if native_ecosystem_kind_is_planning(&kind) {
                native_solve_ecosystem_planning_assignment(input)
            } else if native_ecosystem_kind_is_nonlinear(&kind) {
                native_solve_ecosystem_nonlinear(input)
            } else {
                native_solve_ecosystem_discrete_linear(input)
            }
        }
        _ => return None,
    };
    let mut output = serde_json::Map::new();
    output.insert(
        "kind".to_string(),
        Value::String("optimization-ecosystem-reference-result".to_string()),
    );
    output.insert("tool".to_string(), Value::String(tool.as_str().to_string()));
    output.insert("family".to_string(), Value::String(family.to_string()));
    output.insert(
        "status".to_string(),
        Value::String(result.status.to_string()),
    );
    output.insert(
        "objective".to_string(),
        result.objective.map_or(Value::Null, Value::from),
    );
    output.insert(
        "x".to_string(),
        result
            .x
            .map(|x| Value::Array(x.into_iter().map(Value::from).collect()))
            .unwrap_or(Value::Null),
    );
    output.insert("message".to_string(), Value::String(result.message));
    output.insert(
        "backend".to_string(),
        Value::String(format!("builtin-rust:{family}")),
    );
    for (key, value) in result.extra {
        output.insert(key, value);
    }
    Some(ExternalOptimizationAdapterRun {
        tool,
        status: ExternalOptimizationAdapterStatus::Ok,
        output: Some(Value::Object(output)),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        message: String::new(),
    })
}

fn native_ecosystem_kind_is_planning(kind: &str) -> bool {
    matches!(
        kind,
        "planning-assignment" | "ecosystem-planning-assignment"
    )
}

fn native_ecosystem_kind_is_nonlinear(kind: &str) -> bool {
    matches!(kind, "nonlinear-program" | "ecosystem-nonlinear")
}

fn native_ecosystem_kind_is_cp_job_shop(kind: &str) -> bool {
    matches!(kind, "cp-job-shop" | "ecosystem-cp-job-shop")
}

fn native_external_optimization_ecosystem_family(
    tool: ExternalOptimizationTool,
    payload_kind: &str,
) -> Option<&'static str> {
    match tool {
        ExternalOptimizationTool::ChocoSolver
        | ExternalOptimizationTool::Jacop
        | ExternalOptimizationTool::IbmCpOptimizer
        | ExternalOptimizationTool::OrToolsJava
        | ExternalOptimizationTool::OrToolsPython
        | ExternalOptimizationTool::OrToolsCpSat
        | ExternalOptimizationTool::Cpmpy
        | ExternalOptimizationTool::PyCsp3
        | ExternalOptimizationTool::Conjure
        | ExternalOptimizationTool::SavileRow
        | ExternalOptimizationTool::Picat
        | ExternalOptimizationTool::Clingo
        | ExternalOptimizationTool::Clingcon
        | ExternalOptimizationTool::Sat4j
        | ExternalOptimizationTool::PySat
        | ExternalOptimizationTool::OpenWbo => Some("constraint-programming"),
        ExternalOptimizationTool::JMetal
        | ExternalOptimizationTool::MoeaFramework
        | ExternalOptimizationTool::Ecj => Some("evolutionary-multiobjective"),
        ExternalOptimizationTool::OptaPlanner | ExternalOptimizationTool::Timefold => {
            Some("planning-metaheuristic")
        }
        ExternalOptimizationTool::FastDownward
        | ExternalOptimizationTool::LpgTd
        | ExternalOptimizationTool::Optic
        | ExternalOptimizationTool::Enhsp => Some("ai-planning"),
        ExternalOptimizationTool::Z3
        | ExternalOptimizationTool::Cvc5
        | ExternalOptimizationTool::Yices
        | ExternalOptimizationTool::Bitwuzla
        | ExternalOptimizationTool::Boolector
        | ExternalOptimizationTool::MathSat
        | ExternalOptimizationTool::OptiMathSat
        | ExternalOptimizationTool::OpenSmt
        | ExternalOptimizationTool::SmtInterpol
        | ExternalOptimizationTool::Princess => Some("smt-omt"),
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
        | ExternalOptimizationTool::Csdp
        | ExternalOptimizationTool::OsqpRust
        | ExternalOptimizationTool::ClarabelRust => Some("convex-optimization"),
        ExternalOptimizationTool::PyScipOpt
        | ExternalOptimizationTool::GurobiPy
        | ExternalOptimizationTool::CplexPython
        | ExternalOptimizationTool::XpressPython
        | ExternalOptimizationTool::GurobiRust
        | ExternalOptimizationTool::CplexRust => Some("native-solver-binding"),
        ExternalOptimizationTool::MosekPython => Some("convex-optimization"),
        ExternalOptimizationTool::Hexaly => Some("hybrid-optimization"),
        ExternalOptimizationTool::Argmin
        | ExternalOptimizationTool::Nlopt
        | ExternalOptimizationTool::ScipyOptimize
        | ExternalOptimizationTool::Minotaur
        | ExternalOptimizationTool::Ipopt
        | ExternalOptimizationTool::IpoptRust
        | ExternalOptimizationTool::Bonmin
        | ExternalOptimizationTool::Couenne
        | ExternalOptimizationTool::Knitro
        | ExternalOptimizationTool::Baron
        | ExternalOptimizationTool::Casadi => Some("nonlinear-optimization"),
        ExternalOptimizationTool::OjAlgo
        | ExternalOptimizationTool::Pyomo
        | ExternalOptimizationTool::Pulp
        | ExternalOptimizationTool::PythonMip
        | ExternalOptimizationTool::OrToolsGlop
        | ExternalOptimizationTool::OrToolsPdlp
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
        | ExternalOptimizationTool::SoplexCli
        | ExternalOptimizationTool::QsoptExCli
        | ExternalOptimizationTool::LpSolveCli
        | ExternalOptimizationTool::GurobiCli
        | ExternalOptimizationTool::CplexCli
        | ExternalOptimizationTool::XpressCli
        | ExternalOptimizationTool::LindoCli
        | ExternalOptimizationTool::GoodLp
        | ExternalOptimizationTool::LpModeler
        | ExternalOptimizationTool::RustLinprog
        | ExternalOptimizationTool::MiniLp
        | ExternalOptimizationTool::HighsRust
        | ExternalOptimizationTool::ScipRust
        | ExternalOptimizationTool::CbcRust => {
            if native_ecosystem_kind_is_nonlinear(payload_kind) {
                Some("nonlinear-optimization")
            } else {
                Some("linear-mip")
            }
        }
    }
}

fn native_value_as_f64(value: Option<&Value>, default: f64) -> f64 {
    match value {
        Some(Value::Bool(value)) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Some(Value::Number(value)) => value.as_f64().unwrap_or(default),
        _ => default,
    }
}

fn native_value_array_as_f64(value: Option<&Value>) -> Option<Vec<f64>> {
    value?.as_array().map(|items| {
        items
            .iter()
            .map(|item| native_value_as_f64(Some(item), 0.0))
            .collect()
    })
}

fn native_better(candidate: f64, incumbent: Option<f64>, sense: &str) -> bool {
    match incumbent {
        None => true,
        Some(incumbent) if sense == "max" => candidate > incumbent + 1e-12,
        Some(incumbent) => candidate < incumbent - 1e-12,
    }
}

fn native_row_feasible(lhs: f64, sense: &str, rhs: f64) -> bool {
    match sense.trim() {
        "<=" | "le" | "less-equal" => lhs <= rhs + 1e-9,
        ">=" | "ge" | "greater-equal" => lhs + 1e-9 >= rhs,
        "=" | "==" | "eq" | "equal" => (lhs - rhs).abs() <= 1e-9,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug)]
enum NativeNonlinearBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Clone, Debug)]
enum NativeNonlinearExpr {
    Constant(f64),
    Variable(String),
    Unary {
        sign: f64,
        expr: Box<NativeNonlinearExpr>,
    },
    Binary {
        op: NativeNonlinearBinaryOp,
        left: Box<NativeNonlinearExpr>,
        right: Box<NativeNonlinearExpr>,
    },
}

struct NativeNonlinearExprParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> NativeNonlinearExprParser<'a> {
    fn parse(input: &'a str) -> Result<NativeNonlinearExpr, String> {
        let mut parser = Self { input, pos: 0 };
        let expr = parser.parse_sum()?;
        parser.skip_ws();
        if parser.pos != parser.input.len() {
            return Err(format!(
                "unexpected nonlinear expression token at byte {}",
                parser.pos
            ));
        }
        Ok(expr)
    }

    fn parse_sum(&mut self) -> Result<NativeNonlinearExpr, String> {
        let mut expr = self.parse_product()?;
        loop {
            self.skip_ws();
            let op = if self.consume_byte(b'+') {
                NativeNonlinearBinaryOp::Add
            } else if self.consume_byte(b'-') {
                NativeNonlinearBinaryOp::Sub
            } else {
                break;
            };
            let right = self.parse_product()?;
            expr = NativeNonlinearExpr::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_product(&mut self) -> Result<NativeNonlinearExpr, String> {
        let mut expr = self.parse_power()?;
        loop {
            self.skip_ws();
            if self.input[self.pos..].starts_with("**") {
                break;
            }
            let op = if self.consume_byte(b'*') {
                NativeNonlinearBinaryOp::Mul
            } else if self.consume_byte(b'/') {
                NativeNonlinearBinaryOp::Div
            } else {
                break;
            };
            let right = self.parse_power()?;
            expr = NativeNonlinearExpr::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_power(&mut self) -> Result<NativeNonlinearExpr, String> {
        let mut expr = self.parse_unary()?;
        self.skip_ws();
        if self.consume_str("**") || self.consume_byte(b'^') {
            let right = self.parse_power()?;
            expr = NativeNonlinearExpr::Binary {
                op: NativeNonlinearBinaryOp::Pow,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<NativeNonlinearExpr, String> {
        self.skip_ws();
        if self.consume_byte(b'+') {
            return Ok(NativeNonlinearExpr::Unary {
                sign: 1.0,
                expr: Box::new(self.parse_unary()?),
            });
        }
        if self.consume_byte(b'-') {
            return Ok(NativeNonlinearExpr::Unary {
                sign: -1.0,
                expr: Box::new(self.parse_unary()?),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<NativeNonlinearExpr, String> {
        self.skip_ws();
        if self.consume_byte(b'(') {
            let expr = self.parse_sum()?;
            self.skip_ws();
            if !self.consume_byte(b')') {
                return Err(format!(
                    "missing ')' in nonlinear expression at byte {}",
                    self.pos
                ));
            }
            return Ok(expr);
        }
        match self.peek_byte() {
            Some(byte) if byte.is_ascii_digit() || byte == b'.' => self.parse_number(),
            Some(byte) if byte.is_ascii_alphabetic() || byte == b'_' => self.parse_identifier(),
            _ => Err(format!(
                "expected nonlinear expression term at byte {}",
                self.pos
            )),
        }
    }

    fn parse_number(&mut self) -> Result<NativeNonlinearExpr, String> {
        let start = self.pos;
        let mut has_digit = false;
        while self.peek_byte().is_some_and(|byte| byte.is_ascii_digit()) {
            has_digit = true;
            self.pos += 1;
        }
        if self.consume_byte(b'.') {
            while self.peek_byte().is_some_and(|byte| byte.is_ascii_digit()) {
                has_digit = true;
                self.pos += 1;
            }
        }
        if !has_digit {
            return Err(format!("invalid number at byte {start}"));
        }
        if self
            .peek_byte()
            .is_some_and(|byte| matches!(byte, b'e' | b'E'))
        {
            self.pos += 1;
            if self
                .peek_byte()
                .is_some_and(|byte| matches!(byte, b'+' | b'-'))
            {
                self.pos += 1;
            }
            let exponent_start = self.pos;
            while self.peek_byte().is_some_and(|byte| byte.is_ascii_digit()) {
                self.pos += 1;
            }
            if exponent_start == self.pos {
                return Err(format!("invalid exponent at byte {exponent_start}"));
            }
        }
        let raw = &self.input[start..self.pos];
        let value = raw
            .parse::<f64>()
            .map_err(|err| format!("parse nonlinear number {raw:?}: {err}"))?;
        if !value.is_finite() {
            return Err(format!("non-finite nonlinear number {raw:?}"));
        }
        Ok(NativeNonlinearExpr::Constant(value))
    }

    fn parse_identifier(&mut self) -> Result<NativeNonlinearExpr, String> {
        let start = self.pos;
        self.pos += 1;
        while self
            .peek_byte()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            self.pos += 1;
        }
        Ok(NativeNonlinearExpr::Variable(
            self.input[start..self.pos].to_string(),
        ))
    }

    fn skip_ws(&mut self) {
        while self
            .peek_byte()
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            self.pos += 1;
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }

    fn consume_byte(&mut self, expected: u8) -> bool {
        if self.peek_byte() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn consume_str(&mut self, expected: &str) -> bool {
        if self.input[self.pos..].starts_with(expected) {
            self.pos += expected.len();
            true
        } else {
            false
        }
    }
}

fn native_nonlinear_expression_source(value: Option<&Value>, default: &str) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
        None => default.to_string(),
    }
}

fn native_eval_nonlinear_expr(
    expr: &NativeNonlinearExpr,
    names: &[String],
    values: &[f64],
) -> Result<f64, String> {
    let value = match expr {
        NativeNonlinearExpr::Constant(value) => *value,
        NativeNonlinearExpr::Variable(name) => names
            .iter()
            .position(|candidate| candidate == name)
            .and_then(|index| values.get(index))
            .copied()
            .ok_or_else(|| format!("unknown nonlinear variable {name:?}"))?,
        NativeNonlinearExpr::Unary { sign, expr } => {
            *sign * native_eval_nonlinear_expr(expr, names, values)?
        }
        NativeNonlinearExpr::Binary { op, left, right } => {
            let left = native_eval_nonlinear_expr(left, names, values)?;
            let right = native_eval_nonlinear_expr(right, names, values)?;
            match op {
                NativeNonlinearBinaryOp::Add => left + right,
                NativeNonlinearBinaryOp::Sub => left - right,
                NativeNonlinearBinaryOp::Mul => left * right,
                NativeNonlinearBinaryOp::Div => left / right,
                NativeNonlinearBinaryOp::Pow => left.powf(right),
            }
        }
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err("non-finite nonlinear expression value".to_string())
    }
}

fn native_nonlinear_constraint_violation(lhs: f64, sense: &str, rhs: f64) -> Result<f64, String> {
    match sense.trim() {
        "<=" | "le" | "less-equal" => Ok((lhs - rhs).max(0.0)),
        ">=" | "ge" | "greater-equal" => Ok((rhs - lhs).max(0.0)),
        "=" | "==" | "eq" | "equal" => Ok((lhs - rhs).abs()),
        _ => Err(format!("unsupported nonlinear sense {sense:?}")),
    }
}

fn native_sorted_unique_finite_values(values: &mut Vec<f64>) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("nonlinear domain values must be finite".to_string());
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mut unique = Vec::with_capacity(values.len());
    for value in values.drain(..) {
        if unique
            .last()
            .is_none_or(|previous: &f64| (value - *previous).abs() > 1e-12)
        {
            unique.push(value);
        }
    }
    *values = unique;
    Ok(())
}

fn enumerate_float_domains<F>(domains: &[Vec<f64>], mut visit: F)
where
    F: FnMut(&[f64]),
{
    fn go<F>(domains: &[Vec<f64>], index: usize, assignment: &mut Vec<f64>, visit: &mut F)
    where
        F: FnMut(&[f64]),
    {
        if index == domains.len() {
            visit(assignment);
            return;
        }
        for value in &domains[index] {
            assignment.push(*value);
            go(domains, index + 1, assignment, visit);
            assignment.pop();
        }
    }
    let mut assignment = Vec::with_capacity(domains.len());
    go(domains, 0, &mut assignment, &mut visit);
}

fn native_integer_domains(payload: &Value, width: usize) -> Result<Vec<Vec<i64>>, String> {
    let domains = payload.get("domains").and_then(Value::as_array);
    let mut out = Vec::with_capacity(width);
    for index in 0..width {
        let raw = domains.and_then(|domains| domains.get(index));
        let (lb, ub) = raw
            .and_then(Value::as_array)
            .and_then(|items| {
                Some((
                    native_value_as_f64(items.first(), 0.0).round() as i64,
                    native_value_as_f64(items.get(1), 1.0).round() as i64,
                ))
            })
            .unwrap_or((0, 1));
        if ub < lb {
            return Err("empty domain".to_string());
        }
        if ub - lb > 20 {
            return Err("domain too large for reference enumeration".to_string());
        }
        out.push((lb..=ub).collect());
    }
    Ok(out)
}

fn enumerate_integer_domains<F>(domains: &[Vec<i64>], mut visit: F)
where
    F: FnMut(&[i64]),
{
    fn go<F>(domains: &[Vec<i64>], index: usize, assignment: &mut Vec<i64>, visit: &mut F)
    where
        F: FnMut(&[i64]),
    {
        if index == domains.len() {
            visit(assignment);
            return;
        }
        for value in &domains[index] {
            assignment.push(*value);
            go(domains, index + 1, assignment, visit);
            assignment.pop();
        }
    }
    let mut assignment = Vec::with_capacity(domains.len());
    go(domains, 0, &mut assignment, &mut visit);
}

fn native_solve_ecosystem_discrete_linear(
    payload: &Value,
) -> NativeExternalOptimizationEcosystemResult {
    let Some(objective) = native_value_array_as_f64(payload.get("objective")) else {
        return NativeExternalOptimizationEcosystemResult::status(
            "invalid",
            "missing objective vector",
        );
    };
    if objective.is_empty() {
        return NativeExternalOptimizationEcosystemResult::status(
            "invalid",
            "missing objective vector",
        );
    }
    let sense = payload
        .get("sense")
        .and_then(Value::as_str)
        .unwrap_or("min")
        .to_ascii_lowercase();
    let domains = match native_integer_domains(payload, objective.len()) {
        Ok(domains) => domains,
        Err(message) if message == "empty domain" => {
            return NativeExternalOptimizationEcosystemResult::status("infeasible", message)
        }
        Err(message) => {
            return NativeExternalOptimizationEcosystemResult::status("unsupported", message)
        }
    };
    let constraints = payload
        .get("constraints")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut best_x = None::<Vec<f64>>;
    let mut best_objective = None::<f64>;
    let mut invalid_message = None::<String>;
    enumerate_integer_domains(&domains, |assignment| {
        if invalid_message.is_some() {
            return;
        }
        for row in &constraints {
            let Some(coefs) = native_value_array_as_f64(row.get("coefs")) else {
                invalid_message = Some("constraint coefficient length mismatch".to_string());
                return;
            };
            if coefs.len() != objective.len() {
                invalid_message = Some("constraint coefficient length mismatch".to_string());
                return;
            }
            let lhs = coefs
                .iter()
                .zip(assignment.iter())
                .map(|(coef, value)| coef * *value as f64)
                .sum::<f64>();
            let row_sense = row.get("sense").and_then(Value::as_str).unwrap_or("<=");
            let rhs = native_value_as_f64(row.get("rhs"), 0.0);
            if !native_row_feasible(lhs, row_sense, rhs) {
                return;
            }
        }
        let value = objective
            .iter()
            .zip(assignment.iter())
            .map(|(coef, value)| coef * *value as f64)
            .sum::<f64>();
        if native_better(value, best_objective, &sense) {
            best_objective = Some(value);
            best_x = Some(assignment.iter().map(|value| *value as f64).collect());
        }
    });
    if let Some(message) = invalid_message {
        return NativeExternalOptimizationEcosystemResult::status("invalid", message);
    }
    match (best_objective, best_x) {
        (Some(objective), Some(x)) => {
            NativeExternalOptimizationEcosystemResult::optimal(objective, x)
        }
        _ => NativeExternalOptimizationEcosystemResult::status(
            "infeasible",
            "no feasible assignment",
        ),
    }
}

fn native_solve_ecosystem_nonlinear(payload: &Value) -> NativeExternalOptimizationEcosystemResult {
    let Some(variables) = payload.get("variables").and_then(Value::as_array) else {
        return NativeExternalOptimizationEcosystemResult::status("invalid", "missing variables");
    };
    if variables.is_empty() {
        return NativeExternalOptimizationEcosystemResult::status("invalid", "missing variables");
    }
    if variables.len() > 10 {
        return NativeExternalOptimizationEcosystemResult::status(
            "unsupported",
            "nonlinear reference enumeration supports at most 10 variables",
        );
    }

    let mut names = Vec::with_capacity(variables.len());
    let mut domains = Vec::with_capacity(variables.len());
    let mut point_count = 1usize;
    for (index, variable) in variables.iter().enumerate() {
        let name = variable
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("x{index}"));
        let lb = native_value_as_f64(variable.get("lb"), -5.0);
        let ub = native_value_as_f64(variable.get("ub"), 5.0);
        if !lb.is_finite() || !ub.is_finite() {
            return NativeExternalOptimizationEcosystemResult::status(
                "invalid",
                "nonlinear bounds must be finite",
            );
        }
        if ub < lb {
            return NativeExternalOptimizationEcosystemResult::status(
                "infeasible",
                "empty nonlinear domain",
            );
        }
        let midpoint = 0.5 * (lb + ub);
        let start = native_value_as_f64(variable.get("start"), midpoint);
        let mut domain = vec![lb, ub, midpoint, start];
        if let Err(message) = native_sorted_unique_finite_values(&mut domain) {
            return NativeExternalOptimizationEcosystemResult::status("invalid", message);
        }
        point_count = point_count.saturating_mul(domain.len().max(1));
        if point_count > 1_048_576 {
            return NativeExternalOptimizationEcosystemResult::status(
                "unsupported",
                "nonlinear reference grid too large",
            );
        }
        names.push(name);
        domains.push(domain);
    }

    let objective_source = native_nonlinear_expression_source(payload.get("objective"), "0");
    let objective_expr = match NativeNonlinearExprParser::parse(&objective_source) {
        Ok(expr) => expr,
        Err(message) => {
            return NativeExternalOptimizationEcosystemResult::status("invalid", message)
        }
    };
    let sense = payload
        .get("sense")
        .and_then(Value::as_str)
        .unwrap_or("min")
        .to_ascii_lowercase();
    let constraints: &[Value] = match payload.get("constraints") {
        Some(value) => match value.as_array() {
            Some(rows) => rows.as_slice(),
            None => {
                return NativeExternalOptimizationEcosystemResult::status(
                    "invalid",
                    "constraints must be a list",
                )
            }
        },
        None => &[],
    };
    let mut parsed_constraints = Vec::<(NativeNonlinearExpr, String, f64)>::new();
    for row in constraints {
        let expr_source = native_nonlinear_expression_source(row.get("expr"), "0");
        let expr = match NativeNonlinearExprParser::parse(&expr_source) {
            Ok(expr) => expr,
            Err(message) => {
                return NativeExternalOptimizationEcosystemResult::status("invalid", message)
            }
        };
        let row_sense = row
            .get("sense")
            .and_then(Value::as_str)
            .unwrap_or("<=")
            .to_string();
        if !matches!(
            row_sense.trim(),
            "<=" | "le"
                | "less-equal"
                | ">="
                | "ge"
                | "greater-equal"
                | "="
                | "=="
                | "eq"
                | "equal"
        ) {
            return NativeExternalOptimizationEcosystemResult::status(
                "invalid",
                format!("unsupported nonlinear sense {row_sense:?}"),
            );
        }
        parsed_constraints.push((expr, row_sense, native_value_as_f64(row.get("rhs"), 0.0)));
    }

    let mut best_x = None::<Vec<f64>>;
    let mut best_objective = None::<f64>;
    let mut best_violation = f64::INFINITY;
    let mut invalid_message = None::<String>;
    enumerate_float_domains(&domains, |point| {
        if invalid_message.is_some() {
            return;
        }
        let mut violation = 0.0f64;
        for (expr, row_sense, rhs) in &parsed_constraints {
            let lhs = match native_eval_nonlinear_expr(expr, &names, point) {
                Ok(value) => value,
                Err(message) => {
                    invalid_message = Some(message);
                    return;
                }
            };
            let row_violation = match native_nonlinear_constraint_violation(lhs, row_sense, *rhs) {
                Ok(value) => value,
                Err(message) => {
                    invalid_message = Some(message);
                    return;
                }
            };
            violation = violation.max(row_violation);
        }
        let objective = match native_eval_nonlinear_expr(&objective_expr, &names, point) {
            Ok(value) => value,
            Err(message) => {
                invalid_message = Some(message);
                return;
            }
        };
        if violation <= 1e-7 && native_better(objective, best_objective, &sense) {
            best_objective = Some(objective);
            best_x = Some(point.to_vec());
        }
        best_violation = best_violation.min(violation);
    });
    if let Some(message) = invalid_message {
        return NativeExternalOptimizationEcosystemResult::status("invalid", message);
    }
    match (best_objective, best_x) {
        (Some(objective), Some(x)) => {
            NativeExternalOptimizationEcosystemResult::optimal(objective, x)
                .with_extra("grid_points", Value::from(point_count as u64))
        }
        _ => NativeExternalOptimizationEcosystemResult::status(
            "infeasible",
            format!("best constraint violation {best_violation:.3e}"),
        ),
    }
}

fn native_solve_ecosystem_cp_job_shop(
    payload: &Value,
) -> NativeExternalOptimizationEcosystemResult {
    let Some(raw_jobs) = payload.get("jobs").and_then(Value::as_array) else {
        return NativeExternalOptimizationEcosystemResult::status("invalid", "missing jobs");
    };
    if raw_jobs.is_empty() {
        return NativeExternalOptimizationEcosystemResult::status("invalid", "missing jobs");
    }

    let mut operations = Vec::<NativeCpJobShopOperation>::new();
    let mut per_machine = BTreeMap::<String, Vec<usize>>::new();
    let mut job_operation_ids = Vec::<Vec<usize>>::new();
    for (job_index, raw_job) in raw_jobs.iter().enumerate() {
        let Some(raw_operations) = raw_job.get("operations").and_then(Value::as_array) else {
            return NativeExternalOptimizationEcosystemResult::status(
                "invalid",
                "each job needs operations",
            );
        };
        if raw_operations.is_empty() {
            return NativeExternalOptimizationEcosystemResult::status(
                "invalid",
                "each job needs operations",
            );
        }
        let mut ids = Vec::with_capacity(raw_operations.len());
        for (op_index, raw_op) in raw_operations.iter().enumerate() {
            let machine = raw_op
                .get("machine")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if machine.is_empty() {
                return NativeExternalOptimizationEcosystemResult::status(
                    "invalid",
                    "operation machine is required",
                );
            }
            let duration = native_value_as_f64(raw_op.get("duration"), 0.0);
            if !duration.is_finite() || duration < 0.0 {
                return NativeExternalOptimizationEcosystemResult::status(
                    "invalid",
                    "operation duration must be non-negative",
                );
            }
            let op_id = operations.len();
            operations.push(NativeCpJobShopOperation {
                job: job_index,
                op: op_index,
                machine: machine.clone(),
                duration,
            });
            per_machine.entry(machine).or_default().push(op_id);
            ids.push(op_id);
        }
        job_operation_ids.push(ids);
    }

    let total_ops = operations.len();
    let max_operations = native_value_as_f64(payload.get("max_operations"), 10.0)
        .round()
        .max(0.0) as usize;
    if total_ops > max_operations {
        return NativeExternalOptimizationEcosystemResult::status(
            "unsupported",
            "job-shop reference model is too large",
        );
    }

    let machine_orders = per_machine
        .values()
        .map(|ids| native_permutations(ids))
        .collect::<Vec<_>>();
    let order_count = machine_orders
        .iter()
        .map(Vec::len)
        .try_fold(1usize, |acc, len| acc.checked_mul(len));
    if order_count.is_none_or(|count| count > 1_048_576) {
        return NativeExternalOptimizationEcosystemResult::status(
            "unsupported",
            "job-shop machine-order search space is too large",
        );
    }

    let mut best_starts = None::<Vec<f64>>;
    let mut best_makespan = None::<f64>;
    native_enumerate_machine_orders(&machine_orders, |order_choice| {
        let Some(starts) = native_cp_job_shop_starts_for_machine_order(
            &operations,
            &job_operation_ids,
            order_choice,
        ) else {
            return;
        };
        let makespan = operations
            .iter()
            .enumerate()
            .map(|(idx, operation)| starts[idx] + operation.duration)
            .fold(0.0_f64, f64::max);
        if native_better(makespan, best_makespan, "min")
            || (best_makespan.is_some_and(|best| (makespan - best).abs() <= 1e-12)
                && best_starts
                    .as_ref()
                    .is_some_and(|best| native_lexicographic_less_f64(&starts, best)))
        {
            best_makespan = Some(makespan);
            best_starts = Some(starts);
        }
    });

    let (Some(makespan), Some(starts)) = (best_makespan, best_starts.clone()) else {
        return NativeExternalOptimizationEcosystemResult::status(
            "infeasible",
            "no acyclic machine/job ordering",
        );
    };

    let schedule = operations
        .iter()
        .enumerate()
        .map(|(idx, operation)| {
            let start = starts[idx];
            json!({
                "job": operation.job,
                "op": operation.op,
                "machine": operation.machine,
                "start": start,
                "finish": start + operation.duration
            })
        })
        .collect::<Vec<_>>();
    NativeExternalOptimizationEcosystemResult::optimal(makespan, starts)
        .with_extra("schedule", Value::Array(schedule))
}

#[derive(Clone, Debug)]
struct NativeCpJobShopOperation {
    job: usize,
    op: usize,
    machine: String,
    duration: f64,
}

fn native_permutations(values: &[usize]) -> Vec<Vec<usize>> {
    fn go(values: &mut Vec<usize>, index: usize, out: &mut Vec<Vec<usize>>) {
        if index == values.len() {
            out.push(values.clone());
            return;
        }
        for swap_index in index..values.len() {
            values.swap(index, swap_index);
            go(values, index + 1, out);
            values.swap(index, swap_index);
        }
    }

    let mut values = values.to_vec();
    let mut out = Vec::new();
    go(&mut values, 0, &mut out);
    out
}

fn native_enumerate_machine_orders<F>(machine_orders: &[Vec<Vec<usize>>], mut visit: F)
where
    F: FnMut(&[Vec<usize>]),
{
    fn go<F>(
        machine_orders: &[Vec<Vec<usize>>],
        index: usize,
        current: &mut Vec<Vec<usize>>,
        visit: &mut F,
    ) where
        F: FnMut(&[Vec<usize>]),
    {
        if index == machine_orders.len() {
            visit(current);
            return;
        }
        for order in &machine_orders[index] {
            current.push(order.clone());
            go(machine_orders, index + 1, current, visit);
            current.pop();
        }
    }

    let mut current = Vec::with_capacity(machine_orders.len());
    go(machine_orders, 0, &mut current, &mut visit);
}

fn native_cp_job_shop_starts_for_machine_order(
    operations: &[NativeCpJobShopOperation],
    job_operation_ids: &[Vec<usize>],
    machine_order: &[Vec<usize>],
) -> Option<Vec<f64>> {
    let mut successors = vec![Vec::<(usize, f64)>::new(); operations.len()];
    let mut indegree = vec![0usize; operations.len()];
    let mut add_arc = |before: usize, after: usize| {
        successors[before].push((after, operations[before].duration));
        indegree[after] = indegree[after].saturating_add(1);
    };
    for ids in job_operation_ids {
        for pair in ids.windows(2) {
            add_arc(pair[0], pair[1]);
        }
    }
    for order in machine_order {
        for pair in order.windows(2) {
            add_arc(pair[0], pair[1]);
        }
    }

    let mut starts = vec![0.0_f64; operations.len()];
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(idx, count)| (*count == 0).then_some(idx))
        .collect::<Vec<_>>();
    let mut topological_count = 0usize;
    while !ready.is_empty() {
        let (ready_index, current) = ready
            .iter()
            .enumerate()
            .min_by_key(|(_, op_id)| **op_id)
            .map(|(idx, op_id)| (idx, *op_id))?;
        ready.remove(ready_index);
        topological_count += 1;
        for (next, lag) in &successors[current] {
            starts[*next] = starts[*next].max(starts[current] + *lag);
            indegree[*next] = indegree[*next].saturating_sub(1);
            if indegree[*next] == 0 {
                ready.push(*next);
            }
        }
    }

    (topological_count == operations.len()).then_some(starts)
}

fn native_lexicographic_less_f64(left: &[f64], right: &[f64]) -> bool {
    for (left, right) in left.iter().zip(right.iter()) {
        if (left - right).abs() <= 1e-12 {
            continue;
        }
        return left < right;
    }
    left.len() < right.len()
}

fn native_solve_ecosystem_cp_assignment(
    payload: &Value,
) -> NativeExternalOptimizationEcosystemResult {
    let Some(cost_rows) = payload.get("costs").and_then(Value::as_array) else {
        return NativeExternalOptimizationEcosystemResult::status("invalid", "missing cost matrix");
    };
    if cost_rows.is_empty() {
        return NativeExternalOptimizationEcosystemResult::status("invalid", "missing cost matrix");
    }
    let rows = cost_rows
        .iter()
        .map(|row| native_value_array_as_f64(Some(row)).unwrap_or_default())
        .collect::<Vec<_>>();
    if rows.iter().any(Vec::is_empty) || rows.iter().any(|row| row.len() != rows[0].len()) {
        return NativeExternalOptimizationEcosystemResult::status("invalid", "ragged cost matrix");
    }
    let domains = if payload.get("domains").and_then(Value::as_array).is_some() {
        match native_integer_domains(payload, rows.len()) {
            Ok(domains) => domains,
            Err(message) if message == "empty domain" => {
                return NativeExternalOptimizationEcosystemResult::status("infeasible", message)
            }
            Err(message) => {
                return NativeExternalOptimizationEcosystemResult::status("unsupported", message)
            }
        }
    } else {
        let columns = rows[0].len() as i64;
        (0..rows.len())
            .map(|_| (0..columns).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    let all_different = payload
        .get("all_different")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let forbidden = payload
        .get("forbidden")
        .and_then(Value::as_array)
        .map(|pairs| {
            pairs
                .iter()
                .filter_map(|pair| {
                    let items = pair.as_array()?;
                    Some((
                        native_value_as_f64(items.first(), 0.0).round() as usize,
                        native_value_as_f64(items.get(1), 0.0).round() as i64,
                    ))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut best_x = None::<Vec<f64>>;
    let mut best_objective = None::<f64>;
    enumerate_integer_domains(&domains, |assignment| {
        if all_different {
            let mut sorted = assignment.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != assignment.len() {
                return;
            }
        }
        if assignment
            .iter()
            .enumerate()
            .any(|(row, value)| forbidden.iter().any(|pair| *pair == (row, *value)))
        {
            return;
        }
        if assignment
            .iter()
            .enumerate()
            .any(|(row, value)| *value < 0 || *value as usize >= rows[row].len())
        {
            return;
        }
        let value = assignment
            .iter()
            .enumerate()
            .map(|(row, column)| rows[row][*column as usize])
            .sum::<f64>();
        if native_better(value, best_objective, "min") {
            best_objective = Some(value);
            best_x = Some(assignment.iter().map(|value| *value as f64).collect());
        }
    });
    match (best_objective, best_x) {
        (Some(objective), Some(x)) => {
            NativeExternalOptimizationEcosystemResult::optimal(objective, x)
        }
        _ => NativeExternalOptimizationEcosystemResult::status(
            "infeasible",
            "no feasible assignment",
        ),
    }
}

fn native_solve_ecosystem_planning_assignment(
    payload: &Value,
) -> NativeExternalOptimizationEcosystemResult {
    let Some(durations) = native_value_array_as_f64(payload.get("task_durations")) else {
        return NativeExternalOptimizationEcosystemResult::status(
            "invalid",
            "missing task_durations",
        );
    };
    if durations.is_empty() {
        return NativeExternalOptimizationEcosystemResult::status(
            "invalid",
            "missing task_durations",
        );
    }
    let machines = native_value_as_f64(payload.get("machines"), 0.0).round() as i64;
    if machines <= 0 {
        return NativeExternalOptimizationEcosystemResult::status(
            "invalid",
            "machines must be positive",
        );
    }
    let capacities = match payload.get("capacities").and_then(Value::as_array) {
        Some(items) => {
            if items.len() != machines as usize {
                return NativeExternalOptimizationEcosystemResult::status(
                    "invalid",
                    "capacity length mismatch",
                );
            }
            items
                .iter()
                .map(|item| native_value_as_f64(Some(item), f64::INFINITY))
                .collect::<Vec<_>>()
        }
        None => vec![f64::INFINITY; machines as usize],
    };
    if durations.len() > 20 {
        return NativeExternalOptimizationEcosystemResult::status(
            "unsupported",
            "planning assignment too large for reference enumeration",
        );
    }
    let domains = (0..durations.len())
        .map(|_| (0..machines).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut best_x = None::<Vec<f64>>;
    let mut best_objective = None::<f64>;
    let mut best_loads = None::<Vec<f64>>;
    enumerate_integer_domains(&domains, |assignment| {
        let mut loads = vec![0.0; machines as usize];
        for (task, machine) in assignment.iter().enumerate() {
            loads[*machine as usize] += durations[task];
        }
        if loads
            .iter()
            .zip(capacities.iter())
            .any(|(load, capacity)| load > &(capacity + 1e-9))
        {
            return;
        }
        let value = loads.iter().copied().fold(0.0, f64::max);
        if native_better(value, best_objective, "min") {
            best_objective = Some(value);
            best_x = Some(assignment.iter().map(|value| *value as f64).collect());
            best_loads = Some(loads);
        }
    });
    match (best_objective, best_x, best_loads) {
        (Some(objective), Some(x), Some(loads)) => {
            NativeExternalOptimizationEcosystemResult::optimal(objective, x).with_extra(
                "loads",
                Value::Array(loads.into_iter().map(Value::from).collect()),
            )
        }
        _ => NativeExternalOptimizationEcosystemResult::status("infeasible", "no feasible plan"),
    }
}

fn native_dominates(a: &[f64], b: &[f64], senses: &[String]) -> bool {
    let mut at_least_one = false;
    for ((a, b), sense) in a.iter().zip(b.iter()).zip(senses.iter()) {
        if sense == "max" {
            if a < &(b - 1e-12) {
                return false;
            }
            at_least_one = at_least_one || a > &(b + 1e-12);
        } else {
            if a > &(b + 1e-12) {
                return false;
            }
            at_least_one = at_least_one || a < &(b - 1e-12);
        }
    }
    at_least_one
}

fn native_solve_ecosystem_multiobjective(
    payload: &Value,
) -> NativeExternalOptimizationEcosystemResult {
    let Some(candidates) = payload.get("candidates").and_then(Value::as_array) else {
        return NativeExternalOptimizationEcosystemResult::status("invalid", "missing candidates");
    };
    if candidates.is_empty() {
        return NativeExternalOptimizationEcosystemResult::status("invalid", "missing candidates");
    }
    let senses = payload
        .get("senses")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| item.as_str().unwrap_or("min").to_ascii_lowercase())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["min".to_string(), "min".to_string()]);
    let weights = native_value_array_as_f64(payload.get("weights")).unwrap_or_else(|| {
        std::iter::repeat(1.0)
            .take(senses.len())
            .collect::<Vec<_>>()
    });
    let mut parsed = Vec::<(Vec<f64>, Vec<f64>)>::new();
    for candidate in candidates {
        let x = native_value_array_as_f64(candidate.get("x")).unwrap_or_default();
        let objectives = native_value_array_as_f64(candidate.get("objectives")).unwrap_or_default();
        if objectives.len() != senses.len() {
            return NativeExternalOptimizationEcosystemResult::status(
                "invalid",
                "objective dimension mismatch",
            );
        }
        parsed.push((x, objectives));
    }
    let front = parsed
        .iter()
        .filter(|(_, objectives)| {
            !parsed
                .iter()
                .any(|(_, other)| native_dominates(other, objectives, &senses))
        })
        .cloned()
        .collect::<Vec<_>>();
    if front.is_empty() {
        return NativeExternalOptimizationEcosystemResult::status(
            "infeasible",
            "empty Pareto front",
        );
    }
    let scalar_score = |objectives: &[f64]| {
        objectives
            .iter()
            .zip(weights.iter())
            .zip(senses.iter())
            .map(|((value, weight), sense)| {
                if sense == "max" {
                    weight * -*value
                } else {
                    weight * *value
                }
            })
            .sum::<f64>()
    };
    let (best_x, best_objectives) = front
        .iter()
        .min_by(|(_, left), (_, right)| {
            scalar_score(left)
                .partial_cmp(&scalar_score(right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
        .unwrap_or_default();
    let pareto_front = front
        .into_iter()
        .map(|(x, objectives)| json!({ "x": x, "objectives": objectives }))
        .collect::<Vec<_>>();
    NativeExternalOptimizationEcosystemResult::optimal(scalar_score(&best_objectives), best_x)
        .with_extra("pareto_front", Value::Array(pareto_front))
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
        ExternalOptimizationTool::FastDownward => {
            names.push("FAST_DOWNWARD_HOME".to_string());
        }
        ExternalOptimizationTool::LpgTd => {
            names.push("LPG_TD_HOME".to_string());
            names.push("LPG_HOME".to_string());
        }
        ExternalOptimizationTool::Optic => {
            names.push("OPTIC_HOME".to_string());
        }
        ExternalOptimizationTool::Enhsp => {
            names.push("ENHSP_HOME".to_string());
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
        ExternalOptimizationTool::CplexPython => {
            names.push("CPLEX_PYTHON".to_string());
            names.push("CPLEX_STUDIO_DIR".to_string());
            names.push("CPLEX_HOME".to_string());
        }
        ExternalOptimizationTool::XpressPython => {
            names.push("XPRESS_PYTHON".to_string());
            names.push("XPRESSDIR".to_string());
            names.push("XPRESS_HOME".to_string());
        }
        ExternalOptimizationTool::Docplex => {
            names.push("DOCPLEX_PYTHON".to_string());
            names.push("CPLEX_STUDIO_DIR".to_string());
        }
        ExternalOptimizationTool::OrToolsPython
        | ExternalOptimizationTool::OrToolsGlop
        | ExternalOptimizationTool::OrToolsPdlp => {
            names.push("ORTOOLS_PYTHON".to_string());
        }
        ExternalOptimizationTool::OrToolsCpSat => {
            names.push("FZN_CP_SAT_CMD".to_string());
            names.push("ORTOOLS_CP_SAT_CMD".to_string());
            names.push("ORTOOLS_HOME".to_string());
            names.push("MINIZINC_HOME".to_string());
        }
        ExternalOptimizationTool::ScipyOptimize => {
            names.push("SCIPY_PYTHON".to_string());
        }
        ExternalOptimizationTool::MosekPython => {
            names.push("MOSEK_PYTHON".to_string());
            names.push("MOSEK_HOME".to_string());
            names.push("MSKHOME".to_string());
            names.push("MOSEKLM_LICENSE_FILE".to_string());
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
            names.push("COPT_PYTHON".to_string());
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
        ExternalOptimizationTool::Z3 => {
            names.push("Z3_HOME".to_string());
            names.push("Z3_DIR".to_string());
        }
        ExternalOptimizationTool::Cvc5 => {
            names.push("CVC5_HOME".to_string());
            names.push("CVC5_DIR".to_string());
        }
        ExternalOptimizationTool::Yices => {
            names.push("YICES_HOME".to_string());
            names.push("YICES_DIR".to_string());
        }
        ExternalOptimizationTool::Bitwuzla => {
            names.push("BITWUZLA_HOME".to_string());
            names.push("BITWUZLA_DIR".to_string());
        }
        ExternalOptimizationTool::Boolector => {
            names.push("BOOLECTOR_HOME".to_string());
            names.push("BOOLECTOR_DIR".to_string());
        }
        ExternalOptimizationTool::MathSat => {
            names.push("MATHSAT_HOME".to_string());
            names.push("MATHSAT_DIR".to_string());
        }
        ExternalOptimizationTool::OptiMathSat => {
            names.push("OPTIMATHSAT_HOME".to_string());
            names.push("OPTIMATHSAT_DIR".to_string());
        }
        ExternalOptimizationTool::OpenSmt => {
            names.push("OPENSMT_HOME".to_string());
            names.push("OPENSMT_DIR".to_string());
        }
        ExternalOptimizationTool::SmtInterpol => {
            names.push("SMTINTERPOL_HOME".to_string());
            names.push("SMTINTERPOL_DIR".to_string());
        }
        ExternalOptimizationTool::Princess => {
            names.push("PRINCESS_HOME".to_string());
            names.push("PRINCESS_DIR".to_string());
        }
        ExternalOptimizationTool::HighsCli
        | ExternalOptimizationTool::GlpkCli
        | ExternalOptimizationTool::ScipCli
        | ExternalOptimizationTool::CbcCli
        | ExternalOptimizationTool::ClpCli
        | ExternalOptimizationTool::SoplexCli
        | ExternalOptimizationTool::QsoptExCli
        | ExternalOptimizationTool::LpSolveCli
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
        ExternalOptimizationTool::OsqpRust => {
            names.push("OSQP_RS_CARGO_MANIFEST".to_string());
            names.push("OSQP_DIR".to_string());
            names.push("OSQP_HOME".to_string());
        }
        ExternalOptimizationTool::ClarabelRust => {
            names.push("CLARABEL_RS_CARGO_MANIFEST".to_string());
        }
        ExternalOptimizationTool::GurobiRust => {
            names.push("GUROBI_RUST_CARGO_MANIFEST".to_string());
            names.push("GUROBI_HOME".to_string());
            names.push("GRB_LICENSE_FILE".to_string());
        }
        ExternalOptimizationTool::CplexRust => {
            names.push("CPLEX_RUST_CARGO_MANIFEST".to_string());
            names.push("CPLEX_STUDIO_DIR".to_string());
            names.push("CPLEX_HOME".to_string());
        }
        ExternalOptimizationTool::IpoptRust => {
            names.push("IPOPT_RUST_CARGO_MANIFEST".to_string());
            names.push("IPOPT_DIR".to_string());
            names.push("IPOPT_HOME".to_string());
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
        ExternalOptimizationTool::FastDownward => &["FAST_DOWNWARD_HOME", "FAST_DOWNWARD_DIR"],
        ExternalOptimizationTool::LpgTd => &["LPG_TD_HOME", "LPG_HOME", "LPG_DIR"],
        ExternalOptimizationTool::Optic => &["OPTIC_HOME", "OPTIC_DIR"],
        ExternalOptimizationTool::Enhsp => &["ENHSP_HOME", "ENHSP_DIR"],
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
        ExternalOptimizationTool::CplexPython => &["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        ExternalOptimizationTool::XpressPython => &["XPRESSDIR", "XPRESS_DIR", "XPRESS_HOME"],
        ExternalOptimizationTool::Docplex => &["DOCPLEX_HOME", "CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        ExternalOptimizationTool::OrToolsPython
        | ExternalOptimizationTool::OrToolsGlop
        | ExternalOptimizationTool::OrToolsPdlp => &["ORTOOLS_HOME", "ORTOOLS_PYTHON_HOME"][..],
        ExternalOptimizationTool::OrToolsCpSat => &[
            "ORTOOLS_HOME",
            "ORTOOLS_DIR",
            "MINIZINC_HOME",
            "MINIZINC_DIR",
        ],
        ExternalOptimizationTool::ScipyOptimize => &["SCIPY_HOME", "SCIPY_DIR"],
        ExternalOptimizationTool::MosekPython => &["MOSEK_HOME", "MSKHOME"],
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
        ExternalOptimizationTool::Z3 => &["Z3_DIR", "Z3_HOME"],
        ExternalOptimizationTool::Cvc5 => &["CVC5_DIR", "CVC5_HOME"],
        ExternalOptimizationTool::Yices => &["YICES_DIR", "YICES_HOME"],
        ExternalOptimizationTool::Bitwuzla => &["BITWUZLA_DIR", "BITWUZLA_HOME"],
        ExternalOptimizationTool::Boolector => &["BOOLECTOR_DIR", "BOOLECTOR_HOME"],
        ExternalOptimizationTool::MathSat => &["MATHSAT_DIR", "MATHSAT_HOME"],
        ExternalOptimizationTool::OptiMathSat => &["OPTIMATHSAT_DIR", "OPTIMATHSAT_HOME"],
        ExternalOptimizationTool::OpenSmt => &["OPENSMT_DIR", "OPENSMT_HOME"],
        ExternalOptimizationTool::SmtInterpol => &["SMTINTERPOL_DIR", "SMTINTERPOL_HOME"],
        ExternalOptimizationTool::Princess => &["PRINCESS_DIR", "PRINCESS_HOME"],
        ExternalOptimizationTool::HighsCli => &["HIGHS_DIR", "HIGHS_HOME"],
        ExternalOptimizationTool::GlpkCli => &["GLPK_DIR", "GLPK_HOME"],
        ExternalOptimizationTool::ScipCli => &["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
        ExternalOptimizationTool::CbcCli => &["CBC_DIR", "CBC_HOME", "COINOR_DIR", "COINOR_HOME"],
        ExternalOptimizationTool::ClpCli => &["CLP_DIR", "CLP_HOME", "COINOR_DIR", "COINOR_HOME"],
        ExternalOptimizationTool::SoplexCli => &["SOPLEX_DIR", "SOPLEX_HOME"],
        ExternalOptimizationTool::QsoptExCli => {
            &["QSOPT_EX_DIR", "QSOPT_EX_HOME", "QSOPT_DIR", "QSOPT_HOME"]
        }
        ExternalOptimizationTool::LpSolveCli => &[
            "LP_SOLVE_DIR",
            "LPSOLVE_DIR",
            "LP_SOLVE_HOME",
            "LPSOLVE_HOME",
        ],
        ExternalOptimizationTool::GurobiCli => &["GUROBI_HOME"],
        ExternalOptimizationTool::CplexCli => &["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        ExternalOptimizationTool::XpressCli => &["XPRESSDIR", "XPRESS_DIR", "XPRESS_HOME"],
        ExternalOptimizationTool::LindoCli => {
            &["LINDO_HOME", "LINDO_DIR", "LINDOAPI_HOME", "LINDOAPI_DIR"]
        }
        ExternalOptimizationTool::Nlopt => &["NLOPT_DIR", "NLOPT_HOME"],
        ExternalOptimizationTool::OsqpRust => &["OSQP_DIR", "OSQP_HOME"],
        ExternalOptimizationTool::GurobiRust => &["GUROBI_HOME"],
        ExternalOptimizationTool::CplexRust => &["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        ExternalOptimizationTool::IpoptRust => &["IPOPT_DIR", "IPOPT_HOME"],
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

    if let Some(probe) = probe_gams_backed_native_tool(opts.tool) {
        return probe;
    }

    match opts.tool.language() {
        ExternalOptimizationLanguage::Java => probe_java_tool(opts),
        ExternalOptimizationLanguage::Python => probe_python_tool(opts),
        ExternalOptimizationLanguage::Julia => probe_julia_tool(opts),
        ExternalOptimizationLanguage::Native => probe_native_tool(opts),
        ExternalOptimizationLanguage::Rust => probe_rust_tool(opts),
    }
}

fn probe_gams_backed_native_tool(
    tool: ExternalOptimizationTool,
) -> Option<ExternalOptimizationProbe> {
    let solver = match tool {
        ExternalOptimizationTool::Knitro => ExternalGamsSolver::Knitro,
        ExternalOptimizationTool::Mosek => ExternalGamsSolver::Mosek,
        _ => return None,
    };
    let probe = probe_external_gams_solver(solver, 10_000);
    if !probe.ready {
        return None;
    }
    Some(ExternalOptimizationProbe {
        tool,
        status: ExternalOptimizationProbeStatus::Ready,
        command: probe.command,
        message: format!(
            "{} via GAMS ready probe: {}",
            tool.display_name(),
            probe.message
        ),
    })
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
    let timeout_ms = external_optimization_adapter_timeout_ms();
    let (output, timed_out) = match wait_for_external_optimization_adapter_output(child, timeout_ms)
    {
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
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("external optimization adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; external optimization adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
    if !output.status.success() {
        return ExternalOptimizationAdapterRun {
            tool: opts.tool,
            status: ExternalOptimizationAdapterStatus::Failed,
            output: None,
            elapsed_ms,
            message: stderr,
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

fn external_optimization_adapter_timeout_ms() -> u64 {
    env::var("EXTERNAL_OPTIMIZATION_ADAPTER_TIMEOUT_MS")
        .or_else(|_| env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_external_optimization_adapter_output(
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
            Err(err) => {
                return Err(format!(
                    "failed to poll external optimization adapter: {err}"
                ))
            }
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for external optimization adapter: {err}"))
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
    if !external_optimization_python_import_probes_enabled() {
        return ExternalOptimizationProbe {
            tool: opts.tool,
            status: ExternalOptimizationProbeStatus::NotConfigured,
            command: default_python_probe_command(),
            message: format!(
                "{} needs a local adapter command or explicit Python env; set {} or {}, or set EXTERNAL_OPTIMIZATION_PYTHON_IMPORT_PROBES=1 to probe importable Python packages",
                opts.tool.display_name(),
                adapter_env_names(opts.tool)[0],
                artifact_env_names(opts.tool)[0]
            ),
        };
    }
    let Some(python) = default_python_probe_command() else {
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

fn default_python_probe_command() -> Option<PathBuf> {
    python_probe_command_from_env(env::var_os("PYTHON_BIN"), env::var_os("PYTHON"))
        .or_else(|| find_first_command(&["python3", "python"]))
}

fn python_probe_command_from_env(
    python_bin: Option<OsString>,
    python: Option<OsString>,
) -> Option<PathBuf> {
    python_bin
        .filter(|value| !value.is_empty())
        .or_else(|| python.filter(|value| !value.is_empty()))
        .map(PathBuf::from)
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
    let Ok(child) = Command::new(python)
        .arg("-c")
        .arg(python_import_probe_code(module))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    wait_for_external_optimization_adapter_output(
        child,
        external_optimization_python_import_probe_timeout_ms(),
    )
    .is_ok_and(|(output, timed_out)| !timed_out && output.status.success())
}

fn external_optimization_python_import_probes_enabled() -> bool {
    [
        "ORES_EXTERNAL_OPTIMIZATION_PYTHON_IMPORT_PROBES",
        "EXTERNAL_OPTIMIZATION_PYTHON_IMPORT_PROBES",
        "EXTERNAL_OPTIMIZATION_PROBE_PYTHON_IMPORTS",
    ]
    .iter()
    .find_map(|name| env::var(name).ok())
    .is_some_and(|value| external_optimization_python_import_probe_value_enabled(&value))
}

fn external_optimization_python_import_probe_value_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().replace('_', "-").as_str(),
        "1" | "true" | "yes" | "y" | "on" | "python" | "python-imports" | "imports"
    )
}

fn external_optimization_python_import_probe_timeout_ms() -> u64 {
    env::var("EXTERNAL_OPTIMIZATION_PYTHON_IMPORT_TIMEOUT_MS")
        .or_else(|_| env::var("EXTERNAL_OPTIMIZATION_ADAPTER_TIMEOUT_MS"))
        .or_else(|_| env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2_000)
}

fn python_import_probe_code(module: &str) -> String {
    if module == "pycsp3" {
        return format!(
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module:?}) else 1)"
        );
    }
    format!("import {module}")
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
    use super::{
        external_optimization_python_import_probe_value_enabled, python_import_probe_code,
        python_probe_command_from_env, wait_for_external_optimization_adapter_output,
    };
    use crate::des::general::external_optimization_tools::{
        adapter_env_names, artifact_env_names, external_optimization_command_dir_env_names,
        external_optimization_comparison_report_to_json,
        external_optimization_ecosystem_reference_options,
        external_optimization_normalized_result_from_value, external_optimization_tool_specs,
        external_optimization_tools, find_command_in_install_dir,
        run_external_optimization_comparison, run_external_optimization_ecosystem_reference,
        ExternalLinearCliSolver, ExternalOptimizationAdapterInvocation,
        ExternalOptimizationAdapterOptions, ExternalOptimizationAdapterStatus,
        ExternalOptimizationExactness, ExternalOptimizationFamily, ExternalOptimizationLanguage,
        ExternalOptimizationNormalizedResult, ExternalOptimizationProbeStatus,
        ExternalOptimizationTool,
    };
    use serde_json::{json, Value};
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    #[test]
    fn registry_covers_requested_java_and_rust_ecosystems() {
        let specs = external_optimization_tool_specs();
        assert_eq!(external_optimization_tools().len(), 99);
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
            14
        );
        let minilp = specs
            .iter()
            .find(|spec| spec.tool == ExternalOptimizationTool::MiniLp)
            .expect("minilp spec");
        assert_eq!(minilp.language, ExternalOptimizationLanguage::Rust);
        assert_eq!(minilp.family, ExternalOptimizationFamily::LinearMip);
        assert!(minilp.cargo_crates.contains(&"minilp"));
        let osqp_rust = specs
            .iter()
            .find(|spec| spec.tool == ExternalOptimizationTool::OsqpRust)
            .expect("osqp-rust spec");
        assert_eq!(osqp_rust.language, ExternalOptimizationLanguage::Rust);
        assert_eq!(
            osqp_rust.family,
            ExternalOptimizationFamily::ConvexOptimization
        );
        assert!(osqp_rust.cargo_crates.contains(&"osqp"));
        let clarabel_rust = specs
            .iter()
            .find(|spec| spec.tool == ExternalOptimizationTool::ClarabelRust)
            .expect("clarabel-rust spec");
        assert_eq!(clarabel_rust.language, ExternalOptimizationLanguage::Rust);
        assert_eq!(
            clarabel_rust.family,
            ExternalOptimizationFamily::ConvexOptimization
        );
        assert!(clarabel_rust.cargo_crates.contains(&"clarabel"));
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.language == ExternalOptimizationLanguage::Python)
                .count(),
            28
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
            44
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
            spec.tool == ExternalOptimizationTool::FastDownward
                && spec.family == ExternalOptimizationFamily::AiPlanning
                && spec.exactness == ExternalOptimizationExactness::Exact
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::LpgTd
                && spec.family == ExternalOptimizationFamily::AiPlanning
                && spec.exactness == ExternalOptimizationExactness::Heuristic
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Optic
                && spec.family == ExternalOptimizationFamily::AiPlanning
                && spec.exactness == ExternalOptimizationExactness::Heuristic
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Enhsp
                && spec.family == ExternalOptimizationFamily::AiPlanning
                && spec.exactness == ExternalOptimizationExactness::Heuristic
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Pyomo
                && spec.family == ExternalOptimizationFamily::LinearMip
                && spec.exactness == ExternalOptimizationExactness::ModelingLayer
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::OrToolsGlop
                && spec.family == ExternalOptimizationFamily::LinearMip
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec.cargo_crates.is_empty()
                && spec.adapter_env_names[0] == "ORES_ORTOOLS_GLOP_ADAPTER"
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::OrToolsPdlp
                && spec.family == ExternalOptimizationFamily::LinearMip
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec
                    .artifact_env_names
                    .iter()
                    .any(|name| name == "ORTOOLS_PYTHON")
                && spec.notes.contains("PDLP")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::OrToolsCpSat
                && spec.language == ExternalOptimizationLanguage::Native
                && spec.family == ExternalOptimizationFamily::CpSatRouting
                && spec.exactness == ExternalOptimizationExactness::Exact
                && spec.adapter_command_aliases.contains(&"fzn-cp-sat")
                && spec
                    .artifact_env_names
                    .iter()
                    .any(|name| name == "FZN_CP_SAT_CMD")
                && spec.notes.contains("FlatZinc")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Cpmpy
                && spec.family == ExternalOptimizationFamily::ConstraintProgramming
                && spec.exactness == ExternalOptimizationExactness::ModelingLayer
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Clingo
                && spec.language == ExternalOptimizationLanguage::Python
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
            spec.tool == ExternalOptimizationTool::CplexPython
                && spec.language == ExternalOptimizationLanguage::Python
                && spec.family == ExternalOptimizationFamily::NativeSolverBinding
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec.tool.python_modules().contains(&"cplex")
                && spec.notes.contains("CPLEX Python API")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::XpressPython
                && spec.language == ExternalOptimizationLanguage::Python
                && spec.family == ExternalOptimizationFamily::NativeSolverBinding
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec.tool.python_modules().contains(&"xpress")
                && spec.notes.contains("Xpress Python API")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::MosekPython
                && spec.language == ExternalOptimizationLanguage::Python
                && spec.family == ExternalOptimizationFamily::ConvexOptimization
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec.tool.python_modules().contains(&"mosek")
                && spec.notes.contains("MOSEK Python API")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Copt
                && spec.language == ExternalOptimizationLanguage::Python
                && spec.family == ExternalOptimizationFamily::ConvexOptimization
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec.tool.python_modules().contains(&"coptpy")
                && spec.notes.contains("Python API")
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
            spec.tool == ExternalOptimizationTool::GurobiRust
                && spec.language == ExternalOptimizationLanguage::Rust
                && spec.family == ExternalOptimizationFamily::NativeSolverBinding
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec.tool.cargo_crates().contains(&"grb")
                && spec
                    .adapter_command_aliases
                    .contains(&"gurobi-rust-adapter")
                && spec.notes.contains("Gurobi Optimizer")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::CplexRust
                && spec.language == ExternalOptimizationLanguage::Rust
                && spec.family == ExternalOptimizationFamily::NativeSolverBinding
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec.tool.cargo_crates().contains(&"cplex-rs")
                && spec.adapter_command_aliases.contains(&"cplex-rust-adapter")
                && spec.notes.contains("CPLEX")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::IpoptRust
                && spec.language == ExternalOptimizationLanguage::Rust
                && spec.family == ExternalOptimizationFamily::NonlinearOptimization
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec.tool.cargo_crates().contains(&"ipopt")
                && spec.adapter_command_aliases.contains(&"ipopt-rust-adapter")
                && spec.notes.contains("Ipopt")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Mosek
                && spec.family == ExternalOptimizationFamily::ConvexOptimization
                && spec.exactness == ExternalOptimizationExactness::Numerical
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Z3
                && spec.family == ExternalOptimizationFamily::SmtOmt
                && spec.exactness == ExternalOptimizationExactness::Exact
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Cvc5
                && spec.language == ExternalOptimizationLanguage::Python
                && spec.family == ExternalOptimizationFamily::SmtOmt
                && spec.exactness == ExternalOptimizationExactness::Exact
                && spec.tool.python_modules().contains(&"cvc5")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Bitwuzla
                && spec.language == ExternalOptimizationLanguage::Python
                && spec.family == ExternalOptimizationFamily::SmtOmt
                && spec.exactness == ExternalOptimizationExactness::Exact
                && spec.tool.python_modules().contains(&"bitwuzla")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::Proxqp
                && spec.language == ExternalOptimizationLanguage::Python
                && spec.family == ExternalOptimizationFamily::ConvexOptimization
                && spec.exactness == ExternalOptimizationExactness::Numerical
                && spec.tool.python_modules().contains(&"proxsuite")
        }));
        assert!(specs.iter().any(|spec| {
            spec.tool == ExternalOptimizationTool::OptiMathSat
                && spec.family == ExternalOptimizationFamily::SmtOmt
                && spec.exactness == ExternalOptimizationExactness::Exact
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
        assert_eq!(
            ExternalOptimizationTool::LpSolveCli.linear_cli_solver(),
            Some(ExternalLinearCliSolver::LpSolve)
        );
        assert_eq!(
            ExternalOptimizationTool::SoplexCli.linear_cli_solver(),
            Some(ExternalLinearCliSolver::Soplex)
        );
        assert_eq!(
            ExternalOptimizationTool::QsoptExCli.linear_cli_solver(),
            Some(ExternalLinearCliSolver::QsoptEx)
        );
    }

    #[test]
    fn external_optimization_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_external_optimization_adapter_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }

    #[test]
    fn python_import_probes_are_explicit_opt_in() {
        for value in [
            "1",
            "true",
            "YES",
            "on",
            "python",
            "python_imports",
            "imports",
        ] {
            assert!(
                external_optimization_python_import_probe_value_enabled(value),
                "{value:?} should opt into Python import probes"
            );
        }

        for value in ["", "0", "false", "off", "auto", "rust", "native"] {
            assert!(
                !external_optimization_python_import_probe_value_enabled(value),
                "{value:?} should keep Python import probes disabled"
            );
        }
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
            artifact_env_names(ExternalOptimizationTool::GurobiRust)[0],
            "ORES_GUROBI_RUST_CRATE"
        );
        assert!(artifact_env_names(ExternalOptimizationTool::GurobiRust)
            .contains(&"GUROBI_RUST_CARGO_MANIFEST".to_string()));
        assert_eq!(
            adapter_env_names(ExternalOptimizationTool::CplexRust)[0],
            "ORES_CPLEX_RUST_ADAPTER"
        );
        assert!(artifact_env_names(ExternalOptimizationTool::CplexRust)
            .contains(&"CPLEX_RUST_CARGO_MANIFEST".to_string()));
        assert!(artifact_env_names(ExternalOptimizationTool::IpoptRust)
            .contains(&"IPOPT_RUST_CARGO_MANIFEST".to_string()));
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::MiniLp)[0],
            "ORES_MINILP_CRATE"
        );
        assert!(artifact_env_names(ExternalOptimizationTool::OsqpRust)
            .contains(&"OSQP_RS_CARGO_MANIFEST".to_string()));
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::OsqpRust)
                .contains(&"OSQP_HOME".to_string())
        );
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::ClarabelRust)[0],
            "ORES_CLARABEL_RUST_CRATE"
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
            artifact_env_names(ExternalOptimizationTool::Clingo)[0],
            "ORES_CLINGO_PYTHON"
        );
        assert!(ExternalOptimizationTool::Clingo
            .python_modules()
            .contains(&"clingo"));
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Cvc5)[0],
            "ORES_CVC5_PYTHON"
        );
        assert!(ExternalOptimizationTool::Cvc5
            .python_modules()
            .contains(&"cvc5"));
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Bitwuzla)[0],
            "ORES_BITWUZLA_PYTHON"
        );
        assert!(ExternalOptimizationTool::Bitwuzla
            .python_modules()
            .contains(&"bitwuzla"));
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Proxqp)[0],
            "ORES_PROXQP_PYTHON"
        );
        assert!(ExternalOptimizationTool::Proxqp
            .python_modules()
            .contains(&"proxsuite"));
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
            artifact_env_names(ExternalOptimizationTool::CplexPython)[0],
            "ORES_CPLEX_PYTHON_PYTHON"
        );
        assert!(artifact_env_names(ExternalOptimizationTool::CplexPython)
            .contains(&"CPLEX_STUDIO_DIR".to_string()));
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::XpressPython)[0],
            "ORES_XPRESS_PYTHON_PYTHON"
        );
        assert!(artifact_env_names(ExternalOptimizationTool::XpressPython)
            .contains(&"XPRESSDIR".to_string()));
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::MosekPython)[0],
            "ORES_MOSEK_PYTHON_PYTHON"
        );
        assert!(artifact_env_names(ExternalOptimizationTool::MosekPython)
            .contains(&"MOSEKLM_LICENSE_FILE".to_string()));
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Copt)[0],
            "ORES_COPT_PYTHON"
        );
        assert!(ExternalOptimizationTool::Copt
            .python_modules()
            .contains(&"coptpy"));
        assert!(
            artifact_env_names(ExternalOptimizationTool::Copt).contains(&"COPT_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::Copt)
                .contains(&"COPT_HOME".to_string())
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
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::Z3)[0],
            "ORES_Z3_DIR"
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::Z3)
                .contains(&"Z3_HOME".to_string())
        );
        assert!(artifact_env_names(ExternalOptimizationTool::HighsCli)
            .contains(&"HIGHS_CMD".to_string()));
        assert!(artifact_env_names(ExternalOptimizationTool::SoplexCli)
            .contains(&"SOPLEX_CMD".to_string()));
        assert!(artifact_env_names(ExternalOptimizationTool::QsoptExCli)
            .contains(&"QSOPT_EX_CMD".to_string()));
        assert!(artifact_env_names(ExternalOptimizationTool::LpSolveCli)
            .contains(&"LP_SOLVE_CMD".to_string()));
        assert!(artifact_env_names(ExternalOptimizationTool::GurobiCli)
            .contains(&"GUROBI_CL_CMD".to_string()));
        assert!(artifact_env_names(ExternalOptimizationTool::LindoCli)
            .contains(&"LINDOAPI_CMD".to_string()));
        assert_eq!(
            artifact_env_names(ExternalOptimizationTool::OrToolsCpSat)[0],
            "ORES_ORTOOLS_CP_SAT_DIR"
        );
        assert!(artifact_env_names(ExternalOptimizationTool::OrToolsCpSat)
            .contains(&"FZN_CP_SAT_CMD".to_string()));
        assert!(external_optimization_command_dir_env_names(
            ExternalOptimizationTool::OrToolsCpSat
        )
        .contains(&"ORTOOLS_HOME".to_string()));
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
        assert!(external_optimization_command_dir_env_names(
            ExternalOptimizationTool::FastDownward
        )
        .contains(&"FAST_DOWNWARD_HOME".to_string()));
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::LpgTd)
                .contains(&"LPG_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::Optic)
                .contains(&"OPTIC_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::Enhsp)
                .contains(&"ENHSP_HOME".to_string())
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
            external_optimization_command_dir_env_names(ExternalOptimizationTool::CplexPython)
                .contains(&"CPLEX_STUDIO_DIR".to_string())
        );
        assert!(external_optimization_command_dir_env_names(
            ExternalOptimizationTool::XpressPython
        )
        .contains(&"XPRESSDIR".to_string()));
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::MosekPython)
                .contains(&"MOSEK_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::GurobiRust)
                .contains(&"GUROBI_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::CplexRust)
                .contains(&"CPLEX_STUDIO_DIR".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::IpoptRust)
                .contains(&"IPOPT_HOME".to_string())
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
            external_optimization_command_dir_env_names(ExternalOptimizationTool::SoplexCli)
                .contains(&"SOPLEX_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::QsoptExCli)
                .contains(&"QSOPT_EX_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::LpSolveCli)
                .contains(&"LP_SOLVE_HOME".to_string())
        );
        assert!(
            external_optimization_command_dir_env_names(ExternalOptimizationTool::CplexCli)
                .contains(&"CPLEX_STUDIO_DIR".to_string())
        );
    }

    #[test]
    fn python_probe_command_honors_python_bin_precedence() {
        assert_eq!(
            python_probe_command_from_env(
                Some(OsString::from("/tmp/python-bin")),
                Some(OsString::from("/tmp/python")),
            ),
            Some(PathBuf::from("/tmp/python-bin")),
        );
        assert_eq!(
            python_probe_command_from_env(None, Some(OsString::from("/tmp/python"))),
            Some(PathBuf::from("/tmp/python")),
        );
        assert_eq!(
            python_probe_command_from_env(
                Some(OsString::new()),
                Some(OsString::from("/tmp/python"))
            ),
            Some(PathBuf::from("/tmp/python")),
        );
        assert_eq!(python_probe_command_from_env(None, None), None);
    }

    #[test]
    fn pycsp3_probe_uses_spec_lookup_to_avoid_import_side_effects() {
        let pycsp3_probe = python_import_probe_code("pycsp3");
        assert!(pycsp3_probe.contains("find_spec"));
        assert!(pycsp3_probe.contains("\"pycsp3\""));
        assert_eq!(python_import_probe_code("pyomo"), "import pyomo");
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
            ExternalOptimizationFamily::AiPlanning.as_str(),
            "ai-planning"
        );
        assert_eq!(ExternalOptimizationFamily::SmtOmt.as_str(), "smt-omt");
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

    #[test]
    fn ecosystem_reference_helper_runs_java_and_metaheuristic_families() {
        let cp_payload = json!({
            "kind": "ecosystem-cp-assignment",
            "costs": [[9, 2, 7], [6, 4, 3], [5, 8, 1]],
            "all_different": true
        });
        let choco = run_external_optimization_ecosystem_reference(
            &cp_payload,
            ExternalOptimizationTool::ChocoSolver,
        );
        assert_eq!(choco.status, ExternalOptimizationAdapterStatus::Ok);
        let choco_normalized = choco.output.as_ref().map_or_else(
            ExternalOptimizationNormalizedResult::default,
            external_optimization_normalized_result_from_value,
        );
        assert_eq!(choco_normalized.status.as_deref(), Some("optimal"));
        assert_eq!(choco_normalized.objective, Some(9.0));
        assert_eq!(choco_normalized.solution, Some(vec![1.0, 0.0, 2.0]));

        let cp_job_shop_payload = json!({
            "kind": "ecosystem-cp-job-shop",
            "jobs": [
                {"operations": [{"machine": "M1", "duration": 3}, {"machine": "M2", "duration": 2}]},
                {"operations": [{"machine": "M2", "duration": 2}, {"machine": "M1", "duration": 4}]},
                {"operations": [{"machine": "M1", "duration": 2}, {"machine": "M2", "duration": 3}]}
            ]
        });
        let choco_job_shop = run_external_optimization_ecosystem_reference(
            &cp_job_shop_payload,
            ExternalOptimizationTool::ChocoSolver,
        );
        assert_eq!(choco_job_shop.status, ExternalOptimizationAdapterStatus::Ok);
        assert_eq!(
            choco_job_shop
                .output
                .as_ref()
                .and_then(|output| output.get("backend"))
                .and_then(Value::as_str),
            Some("builtin-rust:constraint-programming")
        );
        let choco_job_shop_normalized = choco_job_shop.output.as_ref().map_or_else(
            ExternalOptimizationNormalizedResult::default,
            external_optimization_normalized_result_from_value,
        );
        assert_eq!(choco_job_shop_normalized.status.as_deref(), Some("optimal"));
        assert_eq!(choco_job_shop_normalized.objective, Some(9.0));
        assert_eq!(
            choco_job_shop_normalized.solution,
            Some(vec![0.0, 3.0, 0.0, 5.0, 3.0, 5.0])
        );

        let multiobjective_payload = json!({
            "kind": "ecosystem-multiobjective",
            "senses": ["min", "min"],
            "weights": [0.5, 0.5],
            "candidates": [
                {"x": [0], "objectives": [4, 1]},
                {"x": [1], "objectives": [2, 2]},
                {"x": [2], "objectives": [1, 4]}
            ]
        });
        for tool in [
            ExternalOptimizationTool::JMetal,
            ExternalOptimizationTool::MoeaFramework,
            ExternalOptimizationTool::Ecj,
        ] {
            let run = run_external_optimization_ecosystem_reference(&multiobjective_payload, tool);
            assert_eq!(run.status, ExternalOptimizationAdapterStatus::Ok);
            assert_eq!(
                run.output
                    .as_ref()
                    .and_then(|output| output.get("backend"))
                    .and_then(Value::as_str),
                Some("builtin-rust:evolutionary-multiobjective")
            );
            let normalized = run.output.as_ref().map_or_else(
                ExternalOptimizationNormalizedResult::default,
                external_optimization_normalized_result_from_value,
            );
            assert_eq!(normalized.status.as_deref(), Some("optimal"));
            assert_eq!(normalized.objective, Some(2.0));
            assert_eq!(normalized.solution, Some(vec![1.0]));
        }
    }

    #[test]
    fn ecosystem_reference_helper_runs_ortools_linear_engines() {
        let linear_payload = json!({
            "kind": "ecosystem-linear-binary",
            "sense": "max",
            "objective": [3, 2],
            "constraints": [{"coefs": [1, 1], "sense": "<=", "rhs": 1}],
            "domains": [[0, 1], [0, 1]]
        });
        for tool in [
            ExternalOptimizationTool::OrToolsGlop,
            ExternalOptimizationTool::OrToolsPdlp,
        ] {
            let run = run_external_optimization_ecosystem_reference(&linear_payload, tool);
            assert_eq!(run.status, ExternalOptimizationAdapterStatus::Ok);
            assert_eq!(
                run.output
                    .as_ref()
                    .and_then(|output| output.get("backend"))
                    .and_then(Value::as_str),
                Some("builtin-rust:linear-mip")
            );
            let normalized = run.output.as_ref().map_or_else(
                ExternalOptimizationNormalizedResult::default,
                external_optimization_normalized_result_from_value,
            );
            assert_eq!(normalized.status.as_deref(), Some("optimal"));
            assert_eq!(normalized.objective, Some(3.0));
            assert_eq!(normalized.solution, Some(vec![1.0, 0.0]));
        }
    }

    #[test]
    fn ecosystem_reference_helper_runs_solver_bindings_and_qsopt_in_rust() {
        let linear_payload = json!({
            "kind": "ecosystem-linear-binary",
            "sense": "max",
            "objective": [3, 2],
            "constraints": [{"coefs": [1, 1], "sense": "<=", "rhs": 1}],
            "domains": [[0, 1], [0, 1]]
        });
        for (tool, backend) in [
            (
                ExternalOptimizationTool::MosekPython,
                "builtin-rust:convex-optimization",
            ),
            (
                ExternalOptimizationTool::GurobiRust,
                "builtin-rust:native-solver-binding",
            ),
            (
                ExternalOptimizationTool::CplexRust,
                "builtin-rust:native-solver-binding",
            ),
            (
                ExternalOptimizationTool::QsoptExCli,
                "builtin-rust:linear-mip",
            ),
        ] {
            let run = run_external_optimization_ecosystem_reference(&linear_payload, tool);
            assert_eq!(run.status, ExternalOptimizationAdapterStatus::Ok);
            assert_eq!(
                run.output
                    .as_ref()
                    .and_then(|output| output.get("backend"))
                    .and_then(Value::as_str),
                Some(backend)
            );
            let normalized = run.output.as_ref().map_or_else(
                ExternalOptimizationNormalizedResult::default,
                external_optimization_normalized_result_from_value,
            );
            assert_eq!(normalized.status.as_deref(), Some("optimal"));
            assert_eq!(normalized.objective, Some(3.0));
            assert_eq!(normalized.solution, Some(vec![1.0, 0.0]));
        }
    }

    #[test]
    fn ecosystem_reference_helper_runs_planning_engines_in_rust() {
        let planning_payload = json!({
            "kind": "ecosystem-planning-assignment",
            "task_durations": [3, 4, 5],
            "machines": 2,
            "capacities": [8, 8]
        });
        for (tool, backend) in [
            (
                ExternalOptimizationTool::OptaPlanner,
                "builtin-rust:planning-metaheuristic",
            ),
            (
                ExternalOptimizationTool::Timefold,
                "builtin-rust:planning-metaheuristic",
            ),
            (
                ExternalOptimizationTool::FastDownward,
                "builtin-rust:ai-planning",
            ),
        ] {
            let run = run_external_optimization_ecosystem_reference(&planning_payload, tool);
            assert_eq!(run.status, ExternalOptimizationAdapterStatus::Ok);
            assert_eq!(
                run.output
                    .as_ref()
                    .and_then(|output| output.get("backend"))
                    .and_then(Value::as_str),
                Some(backend)
            );
            assert_eq!(
                run.output
                    .as_ref()
                    .and_then(|output| output.get("loads"))
                    .and_then(Value::as_array)
                    .map(Vec::len),
                Some(2)
            );
            let normalized = run.output.as_ref().map_or_else(
                ExternalOptimizationNormalizedResult::default,
                external_optimization_normalized_result_from_value,
            );
            assert_eq!(normalized.status.as_deref(), Some("optimal"));
            assert_eq!(normalized.objective, Some(7.0));
            assert_eq!(normalized.solution, Some(vec![0.0, 0.0, 1.0]));
        }
    }

    #[test]
    fn ecosystem_reference_helper_runs_nonlinear_engines_in_rust() {
        let nonlinear_payload = json!({
            "kind": "ecosystem-nonlinear",
            "variables": [
                {"name": "x", "lb": 0.0, "ub": 2.0, "start": 1.0},
                {"name": "y", "lb": 0.0, "ub": 4.0, "start": 2.0}
            ],
            "objective": "(x - 1)**2 + (y - 2)**2",
            "constraints": [{"expr": "x + y", "sense": ">=", "rhs": 1.0}]
        });
        for (tool, backend) in [
            (
                ExternalOptimizationTool::Argmin,
                "builtin-rust:nonlinear-optimization",
            ),
            (
                ExternalOptimizationTool::Nlopt,
                "builtin-rust:nonlinear-optimization",
            ),
            (
                ExternalOptimizationTool::ScipyOptimize,
                "builtin-rust:nonlinear-optimization",
            ),
            (
                ExternalOptimizationTool::IpoptRust,
                "builtin-rust:nonlinear-optimization",
            ),
            (
                ExternalOptimizationTool::Cvxpy,
                "builtin-rust:convex-optimization",
            ),
            (
                ExternalOptimizationTool::Hexaly,
                "builtin-rust:hybrid-optimization",
            ),
        ] {
            let run = run_external_optimization_ecosystem_reference(&nonlinear_payload, tool);
            assert_eq!(run.status, ExternalOptimizationAdapterStatus::Ok);
            assert_eq!(
                run.output
                    .as_ref()
                    .and_then(|output| output.get("backend"))
                    .and_then(Value::as_str),
                Some(backend)
            );
            let normalized = run.output.as_ref().map_or_else(
                ExternalOptimizationNormalizedResult::default,
                external_optimization_normalized_result_from_value,
            );
            assert_eq!(normalized.status.as_deref(), Some("optimal"));
            assert_eq!(normalized.objective, Some(0.0));
            assert_eq!(normalized.solution, Some(vec![1.0, 2.0]));
        }
    }

    #[test]
    fn ecosystem_reference_options_point_at_checked_in_script() {
        let opts =
            external_optimization_ecosystem_reference_options(ExternalOptimizationTool::Jacop);
        assert_eq!(opts.tool, ExternalOptimizationTool::Jacop);
        assert!(opts.command_path.is_some());
        assert!(opts
            .extra_args
            .iter()
            .any(|arg| arg.ends_with("optimization_ecosystem_reference.py")));
        assert!(opts.extra_args.iter().any(|arg| arg == "--tool"));
        assert!(opts.extra_args.iter().any(|arg| arg == "jacop"));
    }
}
