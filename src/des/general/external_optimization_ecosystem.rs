//! Optional probes for Java and Rust optimization ecosystems.
//!
//! Java CP/planning systems are usually consumed as jars on a classpath, while
//! Rust optimization libraries are usually compile-time crates or FFI bindings.
//! This module gives the crate a typed, non-vendored integration boundary for
//! both styles: callers point environment variables at local classpaths or
//! Cargo manifests, and probes report whether that integration is ready.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Broad ecosystem for an optional optimization integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalOptimizationEcosystem {
    Java,
    Python,
    Julia,
    Native,
    Rust,
}

impl ExternalOptimizationEcosystem {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationEcosystem::Java => "java",
            ExternalOptimizationEcosystem::Python => "python",
            ExternalOptimizationEcosystem::Julia => "julia",
            ExternalOptimizationEcosystem::Native => "native",
            ExternalOptimizationEcosystem::Rust => "rust",
        }
    }
}

/// Solver/modeling tool families known to the optional ecosystem bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalOptimizationTool {
    ChocoSolver,
    JaCoP,
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
    Argmin,
    NloptRs,
    GurobiRust,
    CplexRust,
    IpoptRust,
    HighsRust,
    ScipRust,
    CbcRust,
}

impl ExternalOptimizationTool {
    pub fn all() -> &'static [ExternalOptimizationTool] {
        &[
            ExternalOptimizationTool::ChocoSolver,
            ExternalOptimizationTool::JaCoP,
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
            ExternalOptimizationTool::Argmin,
            ExternalOptimizationTool::NloptRs,
            ExternalOptimizationTool::GurobiRust,
            ExternalOptimizationTool::CplexRust,
            ExternalOptimizationTool::IpoptRust,
            ExternalOptimizationTool::HighsRust,
            ExternalOptimizationTool::ScipRust,
            ExternalOptimizationTool::CbcRust,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationTool::ChocoSolver => "choco-solver",
            ExternalOptimizationTool::JaCoP => "jacop",
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
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::NloptRs => "nlopt-rs",
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
            ExternalOptimizationTool::JaCoP => "JaCoP",
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
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::NloptRs => "nlopt-rs",
            ExternalOptimizationTool::GurobiRust => "Gurobi Rust bindings",
            ExternalOptimizationTool::CplexRust => "CPLEX Rust bindings",
            ExternalOptimizationTool::IpoptRust => "Ipopt Rust bindings",
            ExternalOptimizationTool::HighsRust => "HiGHS Rust bindings",
            ExternalOptimizationTool::ScipRust => "SCIP Rust bindings",
            ExternalOptimizationTool::CbcRust => "CBC Rust bindings",
        }
    }

    pub fn ecosystem(self) -> ExternalOptimizationEcosystem {
        match self {
            ExternalOptimizationTool::ChocoSolver
            | ExternalOptimizationTool::JaCoP
            | ExternalOptimizationTool::IbmCpOptimizer
            | ExternalOptimizationTool::OptaPlanner
            | ExternalOptimizationTool::Timefold
            | ExternalOptimizationTool::JMetal
            | ExternalOptimizationTool::MoeaFramework
            | ExternalOptimizationTool::Ecj
            | ExternalOptimizationTool::OjAlgo
            | ExternalOptimizationTool::OrToolsJava
            | ExternalOptimizationTool::SavileRow
            | ExternalOptimizationTool::Sat4j => ExternalOptimizationEcosystem::Java,
            ExternalOptimizationTool::Cpmpy
            | ExternalOptimizationTool::PyCsp3
            | ExternalOptimizationTool::PySat
            | ExternalOptimizationTool::Pyomo
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
            | ExternalOptimizationTool::Bitwuzla => ExternalOptimizationEcosystem::Python,
            ExternalOptimizationTool::Jump => ExternalOptimizationEcosystem::Julia,
            ExternalOptimizationTool::Conjure
            | ExternalOptimizationTool::Picat
            | ExternalOptimizationTool::Clingcon
            | ExternalOptimizationTool::OpenWbo
            | ExternalOptimizationTool::Ampl
            | ExternalOptimizationTool::Gams
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
            | ExternalOptimizationTool::OrToolsCpSat => ExternalOptimizationEcosystem::Native,
            ExternalOptimizationTool::GoodLp
            | ExternalOptimizationTool::LpModeler
            | ExternalOptimizationTool::RustLinprog
            | ExternalOptimizationTool::Argmin
            | ExternalOptimizationTool::NloptRs
            | ExternalOptimizationTool::GurobiRust
            | ExternalOptimizationTool::CplexRust
            | ExternalOptimizationTool::IpoptRust
            | ExternalOptimizationTool::HighsRust
            | ExternalOptimizationTool::ScipRust
            | ExternalOptimizationTool::CbcRust => ExternalOptimizationEcosystem::Rust,
        }
    }

    pub fn env_var(self) -> &'static str {
        match self {
            ExternalOptimizationTool::ChocoSolver => "CHOCO_SOLVER_CLASSPATH",
            ExternalOptimizationTool::JaCoP => "JACOP_CLASSPATH",
            ExternalOptimizationTool::IbmCpOptimizer => "IBM_CP_OPTIMIZER_CLASSPATH",
            ExternalOptimizationTool::OptaPlanner => "OPTAPLANNER_CLASSPATH",
            ExternalOptimizationTool::Timefold => "TIMEFOLD_CLASSPATH",
            ExternalOptimizationTool::FastDownward => "FAST_DOWNWARD_DIR",
            ExternalOptimizationTool::LpgTd => "LPG_TD_DIR",
            ExternalOptimizationTool::Optic => "OPTIC_DIR",
            ExternalOptimizationTool::Enhsp => "ENHSP_DIR",
            ExternalOptimizationTool::JMetal => "JMETAL_CLASSPATH",
            ExternalOptimizationTool::MoeaFramework => "MOEA_FRAMEWORK_CLASSPATH",
            ExternalOptimizationTool::Ecj => "ECJ_CLASSPATH",
            ExternalOptimizationTool::OjAlgo => "OJALGO_CLASSPATH",
            ExternalOptimizationTool::OrToolsJava => "ORTOOLS_JAVA_CLASSPATH",
            ExternalOptimizationTool::Cpmpy => "CPMPY_PYTHON",
            ExternalOptimizationTool::PyCsp3 => "PYCSP3_PYTHON",
            ExternalOptimizationTool::Conjure => "CONJURE_DIR",
            ExternalOptimizationTool::SavileRow => "SAVILE_ROW_CLASSPATH",
            ExternalOptimizationTool::Picat => "PICAT_DIR",
            ExternalOptimizationTool::Clingo => "CLINGO_PYTHON",
            ExternalOptimizationTool::Clingcon => "CLINGCON_DIR",
            ExternalOptimizationTool::Sat4j => "SAT4J_CLASSPATH",
            ExternalOptimizationTool::PySat => "PYSAT_PYTHON",
            ExternalOptimizationTool::OpenWbo => "OPEN_WBO_DIR",
            ExternalOptimizationTool::Pyomo => "PYOMO_PYTHON",
            ExternalOptimizationTool::Pulp => "PULP_PYTHON",
            ExternalOptimizationTool::Cvxpy => "CVXPY_PYTHON",
            ExternalOptimizationTool::Cvxopt => "CVXOPT_PYTHON",
            ExternalOptimizationTool::PyScipOpt => "PYSCIPOPT_PYTHON",
            ExternalOptimizationTool::PythonMip => "PYTHON_MIP_PYTHON",
            ExternalOptimizationTool::GurobiPy => "GUROBIPY_PYTHON",
            ExternalOptimizationTool::CplexPython => "CPLEX_PYTHON",
            ExternalOptimizationTool::XpressPython => "XPRESS_PYTHON",
            ExternalOptimizationTool::Docplex => "DOCPLEX_PYTHON",
            ExternalOptimizationTool::OrToolsPython => "ORTOOLS_PYTHON_PYTHON",
            ExternalOptimizationTool::OrToolsGlop => "ORTOOLS_GLOP_PYTHON",
            ExternalOptimizationTool::OrToolsPdlp => "ORTOOLS_PDLP_PYTHON",
            ExternalOptimizationTool::OrToolsCpSat => "ORTOOLS_CP_SAT_DIR",
            ExternalOptimizationTool::ScipyOptimize => "SCIPY_OPTIMIZE_PYTHON",
            ExternalOptimizationTool::MosekPython => "MOSEK_PYTHON",
            ExternalOptimizationTool::Jump => "JUMP_JULIA",
            ExternalOptimizationTool::Ampl => "AMPL_DIR",
            ExternalOptimizationTool::Gams => "GAMS_DIR",
            ExternalOptimizationTool::Hexaly => "HEXALY_DIR",
            ExternalOptimizationTool::Minotaur => "MINOTAUR_DIR",
            ExternalOptimizationTool::Symphony => "SYMPHONY_DIR",
            ExternalOptimizationTool::Ipopt => "IPOPT_DIR",
            ExternalOptimizationTool::Bonmin => "BONMIN_DIR",
            ExternalOptimizationTool::Couenne => "COUENNE_DIR",
            ExternalOptimizationTool::Knitro => "KNITRO_HOME",
            ExternalOptimizationTool::Mosek => "MOSEK_HOME",
            ExternalOptimizationTool::Baron => "BARON_DIR",
            ExternalOptimizationTool::Copt => "COPT_PYTHON",
            ExternalOptimizationTool::Casadi => "CASADI_PYTHON",
            ExternalOptimizationTool::Osqp => "OSQP_PYTHON",
            ExternalOptimizationTool::Scs => "SCS_PYTHON",
            ExternalOptimizationTool::Clarabel => "CLARABEL_PYTHON",
            ExternalOptimizationTool::Ecos => "ECOS_PYTHON",
            ExternalOptimizationTool::Qpoases => "QPOASES_DIR",
            ExternalOptimizationTool::Proxqp => "PROXQP_PYTHON",
            ExternalOptimizationTool::Cosmo => "COSMO_DIR",
            ExternalOptimizationTool::Sdpa => "SDPA_DIR",
            ExternalOptimizationTool::Csdp => "CSDP_DIR",
            ExternalOptimizationTool::Z3 => "Z3_DIR",
            ExternalOptimizationTool::Cvc5 => "CVC5_PYTHON",
            ExternalOptimizationTool::Yices => "YICES_DIR",
            ExternalOptimizationTool::Bitwuzla => "BITWUZLA_PYTHON",
            ExternalOptimizationTool::Boolector => "BOOLECTOR_DIR",
            ExternalOptimizationTool::MathSat => "MATHSAT_DIR",
            ExternalOptimizationTool::OptiMathSat => "OPTIMATHSAT_DIR",
            ExternalOptimizationTool::OpenSmt => "OPENSMT_DIR",
            ExternalOptimizationTool::SmtInterpol => "SMTINTERPOL_DIR",
            ExternalOptimizationTool::Princess => "PRINCESS_DIR",
            ExternalOptimizationTool::HighsCli => "HIGHS_DIR",
            ExternalOptimizationTool::GlpkCli => "GLPK_DIR",
            ExternalOptimizationTool::ScipCli => "SCIP_DIR",
            ExternalOptimizationTool::CbcCli => "CBC_DIR",
            ExternalOptimizationTool::ClpCli => "CLP_DIR",
            ExternalOptimizationTool::SoplexCli => "SOPLEX_DIR",
            ExternalOptimizationTool::QsoptExCli => "QSOPT_EX_DIR",
            ExternalOptimizationTool::LpSolveCli => "LP_SOLVE_DIR",
            ExternalOptimizationTool::GurobiCli => "GUROBI_HOME",
            ExternalOptimizationTool::CplexCli => "CPLEX_STUDIO_DIR",
            ExternalOptimizationTool::XpressCli => "XPRESSDIR",
            ExternalOptimizationTool::LindoCli => "LINDO_HOME",
            ExternalOptimizationTool::GoodLp => "GOOD_LP_CARGO_MANIFEST",
            ExternalOptimizationTool::LpModeler => "LP_MODELER_CARGO_MANIFEST",
            ExternalOptimizationTool::RustLinprog => "RUST_LINPROG_CARGO_MANIFEST",
            ExternalOptimizationTool::Argmin => "ARGMIN_CARGO_MANIFEST",
            ExternalOptimizationTool::NloptRs => "NLOPT_RS_CARGO_MANIFEST",
            ExternalOptimizationTool::GurobiRust => "GUROBI_RUST_CARGO_MANIFEST",
            ExternalOptimizationTool::CplexRust => "CPLEX_RUST_CARGO_MANIFEST",
            ExternalOptimizationTool::IpoptRust => "IPOPT_RUST_CARGO_MANIFEST",
            ExternalOptimizationTool::HighsRust => "HIGHS_RS_CARGO_MANIFEST",
            ExternalOptimizationTool::ScipRust => "SCIP_RS_CARGO_MANIFEST",
            ExternalOptimizationTool::CbcRust => "CBC_RS_CARGO_MANIFEST",
        }
    }

    pub fn artifact_env_vars(self) -> Vec<&'static str> {
        let mut names = vec![self.env_var()];
        for name in match self {
            ExternalOptimizationTool::Cpmpy => &["CPMPY_PYTHON"][..],
            ExternalOptimizationTool::PyCsp3 => &["PYCSP3_PYTHON"],
            ExternalOptimizationTool::Conjure => &["CONJURE_HOME"],
            ExternalOptimizationTool::SavileRow => &["SAVILEROW_HOME"],
            ExternalOptimizationTool::Picat => &["PICAT_HOME"],
            ExternalOptimizationTool::Clingo => &["CLINGO_HOME"],
            ExternalOptimizationTool::Clingcon => &["CLINGCON_HOME"],
            ExternalOptimizationTool::Sat4j => &["SAT4J_HOME"],
            ExternalOptimizationTool::PySat => &["PYSAT_PYTHON"],
            ExternalOptimizationTool::OpenWbo => &["OPEN_WBO_HOME"],
            ExternalOptimizationTool::FastDownward => &["FAST_DOWNWARD_HOME"],
            ExternalOptimizationTool::LpgTd => &["LPG_TD_HOME", "LPG_HOME"],
            ExternalOptimizationTool::Optic => &["OPTIC_HOME"],
            ExternalOptimizationTool::Enhsp => &["ENHSP_HOME"],
            ExternalOptimizationTool::Pyomo => &["PYOMO_PYTHON"][..],
            ExternalOptimizationTool::Pulp => &["PULP_PYTHON"],
            ExternalOptimizationTool::Cvxpy => &["CVXPY_PYTHON"],
            ExternalOptimizationTool::Cvxopt => &["CVXOPT_PYTHON"],
            ExternalOptimizationTool::PyScipOpt => &["PYSCIPOPT_PYTHON"],
            ExternalOptimizationTool::PythonMip => &["PYTHON_MIP_PYTHON"],
            ExternalOptimizationTool::GurobiPy => &["GUROBIPY_PYTHON"],
            ExternalOptimizationTool::CplexPython => {
                &["CPLEX_PYTHON", "CPLEX_STUDIO_DIR", "CPLEX_HOME"]
            }
            ExternalOptimizationTool::XpressPython => {
                &["XPRESS_PYTHON", "XPRESSDIR", "XPRESS_HOME"]
            }
            ExternalOptimizationTool::Docplex => &["DOCPLEX_PYTHON"],
            ExternalOptimizationTool::OrToolsPython
            | ExternalOptimizationTool::OrToolsGlop
            | ExternalOptimizationTool::OrToolsPdlp => &["ORTOOLS_PYTHON"],
            ExternalOptimizationTool::OrToolsCpSat => &[
                "FZN_CP_SAT_CMD",
                "ORTOOLS_CP_SAT_CMD",
                "ORTOOLS_HOME",
                "MINIZINC_HOME",
            ],
            ExternalOptimizationTool::ScipyOptimize => &["SCIPY_PYTHON"],
            ExternalOptimizationTool::MosekPython => &[
                "MOSEK_PYTHON",
                "MOSEK_HOME",
                "MSKHOME",
                "MOSEKLM_LICENSE_FILE",
            ],
            ExternalOptimizationTool::Jump => &["JULIA_PROJECT"],
            ExternalOptimizationTool::Minotaur => &["MINOTAUR_HOME"],
            ExternalOptimizationTool::Symphony => &["SYMPHONY_HOME", "COINOR_HOME"],
            ExternalOptimizationTool::Ipopt => &["IPOPT_HOME"],
            ExternalOptimizationTool::Bonmin => &["BONMIN_HOME"],
            ExternalOptimizationTool::Couenne => &["COUENNE_HOME"],
            ExternalOptimizationTool::Knitro => &["KNITRODIR", "KNITRO_DIR", "ARTELYS_LICENSE"],
            ExternalOptimizationTool::Mosek => &["MSKHOME", "MOSEKLM_LICENSE_FILE"],
            ExternalOptimizationTool::Baron => &["BARON_HOME", "BARON_LICENSE"],
            ExternalOptimizationTool::Copt => &["COPT_DIR"],
            ExternalOptimizationTool::Casadi => &["CASADI_PYTHON"],
            ExternalOptimizationTool::Osqp => &["OSQP_PYTHON"],
            ExternalOptimizationTool::Scs => &["SCS_PYTHON"],
            ExternalOptimizationTool::Clarabel => &["CLARABEL_PYTHON"],
            ExternalOptimizationTool::Ecos => &["ECOS_PYTHON"],
            ExternalOptimizationTool::Qpoases => &["QPOASES_HOME"],
            ExternalOptimizationTool::Proxqp => &["PROXQP_HOME"],
            ExternalOptimizationTool::Cosmo => &["COSMO_HOME"],
            ExternalOptimizationTool::Sdpa => &["SDPA_HOME"],
            ExternalOptimizationTool::Csdp => &["CSDP_HOME"],
            ExternalOptimizationTool::Z3 => &["Z3_HOME", "Z3_CMD"],
            ExternalOptimizationTool::Cvc5 => &["CVC5_HOME", "CVC5_CMD"],
            ExternalOptimizationTool::Yices => &["YICES_HOME", "YICES_CMD"],
            ExternalOptimizationTool::Bitwuzla => &["BITWUZLA_HOME", "BITWUZLA_CMD"],
            ExternalOptimizationTool::Boolector => &["BOOLECTOR_HOME", "BOOLECTOR_CMD"],
            ExternalOptimizationTool::MathSat => &["MATHSAT_HOME", "MATHSAT_CMD"],
            ExternalOptimizationTool::OptiMathSat => &["OPTIMATHSAT_HOME", "OPTIMATHSAT_CMD"],
            ExternalOptimizationTool::OpenSmt => &["OPENSMT_HOME", "OPENSMT_CMD"],
            ExternalOptimizationTool::SmtInterpol => &["SMTINTERPOL_HOME", "SMTINTERPOL_CMD"],
            ExternalOptimizationTool::Princess => &["PRINCESS_HOME", "PRINCESS_CMD"],
            ExternalOptimizationTool::HighsCli => &["HIGHS_CMD", "HIGHS_HOME"],
            ExternalOptimizationTool::GlpkCli => &["GLPSOL_CMD", "GLPK_CMD", "GLPK_HOME"],
            ExternalOptimizationTool::ScipCli => &["SCIP_CMD", "SCIPOPTDIR", "SCIP_HOME"],
            ExternalOptimizationTool::CbcCli => &["CBC_CMD", "CBC_HOME", "COINOR_HOME"],
            ExternalOptimizationTool::ClpCli => &["CLP_CMD", "CLP_HOME", "COINOR_HOME"],
            ExternalOptimizationTool::SoplexCli => &["SOPLEX_CMD", "SOPLEX_HOME"],
            ExternalOptimizationTool::QsoptExCli => &["QSOPT_EX_CMD", "QSOPT_CMD", "QSOPT_EX_HOME"],
            ExternalOptimizationTool::LpSolveCli => &["LP_SOLVE_CMD", "LPSOLVE_CMD"],
            ExternalOptimizationTool::GurobiCli => &["GUROBI_CL_CMD", "GUROBI_CMD"],
            ExternalOptimizationTool::CplexCli => &["CPLEX_CMD", "CPLEX_HOME"],
            ExternalOptimizationTool::XpressCli => &["XPRESS_CMD", "XPRESS_HOME"],
            ExternalOptimizationTool::LindoCli => &["RUNLINDO_CMD", "LINDO_CMD", "LINDOAPI_CMD"],
            ExternalOptimizationTool::GoodLp => &["GOOD_LP_CARGO_MANIFEST"],
            ExternalOptimizationTool::LpModeler => &["LP_MODELER_CARGO_MANIFEST"],
            ExternalOptimizationTool::RustLinprog => &["RUST_LINPROG_CARGO_MANIFEST"],
            ExternalOptimizationTool::Argmin => &["ARGMIN_CARGO_MANIFEST"],
            ExternalOptimizationTool::NloptRs => &["NLOPT_RS_CARGO_MANIFEST"],
            ExternalOptimizationTool::GurobiRust => &[
                "GUROBI_RUST_CARGO_MANIFEST",
                "GUROBI_HOME",
                "GRB_LICENSE_FILE",
            ],
            ExternalOptimizationTool::CplexRust => &[
                "CPLEX_RUST_CARGO_MANIFEST",
                "CPLEX_STUDIO_DIR",
                "CPLEX_HOME",
            ],
            ExternalOptimizationTool::IpoptRust => {
                &["IPOPT_RUST_CARGO_MANIFEST", "IPOPT_DIR", "IPOPT_HOME"]
            }
            ExternalOptimizationTool::HighsRust => &["HIGHS_RS_CARGO_MANIFEST"],
            ExternalOptimizationTool::ScipRust => &["SCIP_RS_CARGO_MANIFEST"],
            ExternalOptimizationTool::CbcRust => &["CBC_RS_CARGO_MANIFEST"],
            _ => &[],
        } {
            push_unique_env_name(&mut names, *name);
        }
        names
    }

    pub fn install_dir_env_vars(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::ChocoSolver => &["CHOCO_SOLVER_HOME", "CHOCO_HOME"],
            ExternalOptimizationTool::JaCoP => &["JACOP_HOME", "JACOP_DIR"],
            ExternalOptimizationTool::IbmCpOptimizer => {
                &["CPLEX_STUDIO_DIR", "CPLEX_HOME", "CP_OPTIMIZER_HOME"]
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
                &["SAVILE_ROW_HOME", "SAVILE_ROW_DIR", "SAVILEROW_HOME"]
            }
            ExternalOptimizationTool::Picat => &["PICAT_HOME", "PICAT_DIR"],
            ExternalOptimizationTool::Clingo => &["CLINGO_HOME", "CLINGO_DIR", "POTASSCO_HOME"],
            ExternalOptimizationTool::Clingcon => {
                &["CLINGCON_HOME", "CLINGCON_DIR", "POTASSCO_HOME"]
            }
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
            ExternalOptimizationTool::Docplex => {
                &["DOCPLEX_HOME", "CPLEX_STUDIO_DIR", "CPLEX_HOME"]
            }
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
            ExternalOptimizationTool::CbcCli => {
                &["CBC_DIR", "CBC_HOME", "COINOR_DIR", "COINOR_HOME"]
            }
            ExternalOptimizationTool::ClpCli => {
                &["CLP_DIR", "CLP_HOME", "COINOR_DIR", "COINOR_HOME"]
            }
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
            ExternalOptimizationTool::NloptRs => &["NLOPT_DIR", "NLOPT_HOME"],
            ExternalOptimizationTool::GurobiRust => &["GUROBI_HOME"],
            ExternalOptimizationTool::CplexRust => &["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
            ExternalOptimizationTool::IpoptRust => &["IPOPT_DIR", "IPOPT_HOME"],
            ExternalOptimizationTool::HighsRust => &["HIGHS_DIR", "HIGHS_HOME"],
            ExternalOptimizationTool::ScipRust => &["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
            ExternalOptimizationTool::CbcRust => {
                &["CBC_DIR", "CBC_HOME", "COINOR_DIR", "COINOR_HOME"]
            }
            _ => &[],
        }
    }

    pub fn java_probe_classes(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::ChocoSolver => &["org.chocosolver.solver.Model"],
            ExternalOptimizationTool::JaCoP => &["org.jacop.core.Store"],
            ExternalOptimizationTool::IbmCpOptimizer => &["ilog.cp.IloCP"],
            ExternalOptimizationTool::OptaPlanner => {
                &["org.optaplanner.core.api.solver.SolverFactory"]
            }
            ExternalOptimizationTool::Timefold => {
                &["ai.timefold.solver.core.api.solver.SolverFactory"]
            }
            ExternalOptimizationTool::JMetal => &["org.uma.jmetal.algorithm.Algorithm"],
            ExternalOptimizationTool::MoeaFramework => &["org.moeaframework.Executor"],
            ExternalOptimizationTool::Ecj => &["ec.Evolve"],
            ExternalOptimizationTool::OjAlgo => &["org.ojalgo.optimisation.ExpressionsBasedModel"],
            ExternalOptimizationTool::OrToolsJava => &[
                "com.google.ortools.Loader",
                "com.google.ortools.sat.CpModel",
            ],
            ExternalOptimizationTool::SavileRow => &["savilerow.SavileRow", "SavileRow"],
            ExternalOptimizationTool::Sat4j => &[
                "org.sat4j.minisat.SolverFactory",
                "org.sat4j.pb.SolverFactory",
            ],
            _ => &[],
        }
    }

    pub fn python_modules(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::Pyomo => &["pyomo.environ"],
            ExternalOptimizationTool::Pulp => &["pulp"],
            ExternalOptimizationTool::Cvxpy => &["cvxpy"],
            ExternalOptimizationTool::Cvxopt => &["cvxopt"],
            ExternalOptimizationTool::PyScipOpt => &["pyscipopt"],
            ExternalOptimizationTool::PythonMip => &["mip"],
            ExternalOptimizationTool::GurobiPy => &["gurobipy"],
            ExternalOptimizationTool::CplexPython => &["cplex"],
            ExternalOptimizationTool::XpressPython => &["xpress"],
            ExternalOptimizationTool::Docplex => &["docplex.mp.model"],
            ExternalOptimizationTool::OrToolsPython => &["ortools.sat.python.cp_model"],
            ExternalOptimizationTool::OrToolsGlop | ExternalOptimizationTool::OrToolsPdlp => {
                &["ortools.linear_solver.pywraplp"]
            }
            ExternalOptimizationTool::ScipyOptimize => &["scipy.optimize"],
            ExternalOptimizationTool::MosekPython => &["mosek"],
            ExternalOptimizationTool::Cpmpy => &["cpmpy"],
            ExternalOptimizationTool::PyCsp3 => &["pycsp3"],
            ExternalOptimizationTool::PySat => &["pysat"],
            ExternalOptimizationTool::Clingo => &["clingo"],
            ExternalOptimizationTool::Casadi => &["casadi"],
            ExternalOptimizationTool::Copt => &["coptpy"],
            ExternalOptimizationTool::Osqp => &["osqp"],
            ExternalOptimizationTool::Scs => &["scs"],
            ExternalOptimizationTool::Clarabel => &["clarabel"],
            ExternalOptimizationTool::Ecos => &["ecos"],
            ExternalOptimizationTool::Cvc5 => &["cvc5"],
            ExternalOptimizationTool::Proxqp => &["proxsuite"],
            ExternalOptimizationTool::Bitwuzla => &["bitwuzla"],
            _ => &[],
        }
    }

    pub fn native_command_aliases(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::Ampl => &["ampl"],
            ExternalOptimizationTool::Gams => &["gams"],
            ExternalOptimizationTool::Hexaly => &["hexaly", "localsolver"],
            ExternalOptimizationTool::HighsCli => &["highs"],
            ExternalOptimizationTool::GlpkCli => &["glpsol"],
            ExternalOptimizationTool::ScipCli => &["scip"],
            ExternalOptimizationTool::CbcCli => &["cbc"],
            ExternalOptimizationTool::ClpCli => &["clp"],
            ExternalOptimizationTool::SoplexCli => &["soplex"],
            ExternalOptimizationTool::QsoptExCli => &["qsopt_ex", "qsopt-ex", "qsopt", "esolver"],
            ExternalOptimizationTool::LpSolveCli => &["lp_solve", "lp-solve", "lpsolve"],
            ExternalOptimizationTool::GurobiCli => &["gurobi_cl"],
            ExternalOptimizationTool::CplexCli => &["cplex"],
            ExternalOptimizationTool::XpressCli => &["optimizer", "xpress"],
            ExternalOptimizationTool::LindoCli => &["runlindo", "lindo", "lindoapi"],
            ExternalOptimizationTool::Conjure => &["conjure"],
            ExternalOptimizationTool::Picat => &["picat"],
            ExternalOptimizationTool::Clingo => &["clingo"],
            ExternalOptimizationTool::Clingcon => &["clingcon"],
            ExternalOptimizationTool::OpenWbo => &["open-wbo", "open-wbo_static"],
            ExternalOptimizationTool::OrToolsCpSat => &["fzn-cp-sat"],
            ExternalOptimizationTool::FastDownward => &["fast-downward.py", "fast-downward"],
            ExternalOptimizationTool::LpgTd => &["lpg-td", "lpg"],
            ExternalOptimizationTool::Optic => &["optic", "optic-clp"],
            ExternalOptimizationTool::Enhsp => &["enhsp", "enhsp.jar"],
            ExternalOptimizationTool::Minotaur => &["minotaur"],
            ExternalOptimizationTool::Symphony => &["symphony"],
            ExternalOptimizationTool::Ipopt => &["ipopt"],
            ExternalOptimizationTool::Bonmin => &["bonmin"],
            ExternalOptimizationTool::Couenne => &["couenne"],
            ExternalOptimizationTool::Knitro => &["knitro", "knitroampl"],
            ExternalOptimizationTool::Mosek => &["mosek"],
            ExternalOptimizationTool::Baron => &["baron"],
            ExternalOptimizationTool::Copt => &["copt_cmd", "copt"],
            ExternalOptimizationTool::Qpoases => &["qpoases"],
            ExternalOptimizationTool::Proxqp => &["proxqp"],
            ExternalOptimizationTool::Cosmo => &["cosmo"],
            ExternalOptimizationTool::Sdpa => &["sdpa", "sdpa_gmp", "sdpa_dd"],
            ExternalOptimizationTool::Csdp => &["csdp"],
            ExternalOptimizationTool::Z3 => &["z3"],
            ExternalOptimizationTool::Cvc5 => &["cvc5"],
            ExternalOptimizationTool::Yices => &["yices-smt2", "yices"],
            ExternalOptimizationTool::Bitwuzla => &["bitwuzla"],
            ExternalOptimizationTool::Boolector => &["boolector"],
            ExternalOptimizationTool::MathSat => &["mathsat"],
            ExternalOptimizationTool::OptiMathSat => &["optimathsat", "optimathsat5"],
            ExternalOptimizationTool::OpenSmt => &["opensmt", "opensmt2"],
            ExternalOptimizationTool::SmtInterpol => &["smtinterpol", "smtinterpol.sh"],
            ExternalOptimizationTool::Princess => &["princess", "princess-smt"],
            _ => &[],
        }
    }

    pub fn rust_dependency_names(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::GoodLp => &["good_lp"],
            ExternalOptimizationTool::LpModeler => &["lp-modeler", "lp_modeler"],
            ExternalOptimizationTool::RustLinprog => &["rust-linprog", "linprog"],
            ExternalOptimizationTool::Argmin => &["argmin"],
            ExternalOptimizationTool::NloptRs => &["nlopt", "nlopt-sys"],
            ExternalOptimizationTool::GurobiRust => &["grb", "gurobi"],
            ExternalOptimizationTool::CplexRust => &["cplex-rs", "cplex-rs-sys", "cplex_sys"],
            ExternalOptimizationTool::IpoptRust => &["ipopt", "ipopt-sys"],
            ExternalOptimizationTool::HighsRust => &["highs", "highs-sys"],
            ExternalOptimizationTool::ScipRust => &["russcip", "scip", "scip-sys"],
            ExternalOptimizationTool::CbcRust => &["coin_cbc", "cbc", "cbc-sys"],
            _ => &[],
        }
    }
}

/// Probe status for an optional ecosystem integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalOptimizationProbeStatus {
    Ready,
    NotConfigured,
    RuntimeMissing,
    ArtifactMissing,
    ProbeFailed,
}

impl ExternalOptimizationProbeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationProbeStatus::Ready => "ready",
            ExternalOptimizationProbeStatus::NotConfigured => "not-configured",
            ExternalOptimizationProbeStatus::RuntimeMissing => "runtime-missing",
            ExternalOptimizationProbeStatus::ArtifactMissing => "artifact-missing",
            ExternalOptimizationProbeStatus::ProbeFailed => "probe-failed",
        }
    }
}

/// Probe result for one optional Java/Rust optimization integration.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalOptimizationProbe {
    pub tool: ExternalOptimizationTool,
    pub ecosystem: ExternalOptimizationEcosystem,
    pub status: ExternalOptimizationProbeStatus,
    pub command: Option<PathBuf>,
    pub env_var: &'static str,
    pub artifact: Option<String>,
    pub elapsed_ms: f64,
    pub message: String,
}

/// Probe one Java classpath or Rust Cargo-manifest integration.
pub fn probe_external_optimization_tool(
    tool: ExternalOptimizationTool,
) -> ExternalOptimizationProbe {
    match tool.ecosystem() {
        ExternalOptimizationEcosystem::Java => probe_java_tool(tool),
        ExternalOptimizationEcosystem::Python => probe_python_tool(tool),
        ExternalOptimizationEcosystem::Julia => probe_julia_tool(tool),
        ExternalOptimizationEcosystem::Native => probe_native_tool(tool),
        ExternalOptimizationEcosystem::Rust => probe_rust_tool(tool),
    }
}

/// Probe all optional Java/Rust optimization integrations known to the bridge.
pub fn probe_external_optimization_tools() -> Vec<ExternalOptimizationProbe> {
    ExternalOptimizationTool::all()
        .iter()
        .copied()
        .map(probe_external_optimization_tool)
        .collect()
}

fn optimization_ecosystem_probe_timeout_ms() -> u64 {
    env::var("EXTERNAL_OPTIMIZATION_ECOSYSTEM_PROBE_TIMEOUT_MS")
        .or_else(|_| env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30_000)
}

fn wait_for_optimization_ecosystem_probe_output(
    mut child: Child,
    timeout_ms: u64,
) -> Result<(Output, bool), String> {
    let timeout = Duration::from_millis(timeout_ms);
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                return child
                    .wait_with_output()
                    .map(|output| (output, false))
                    .map_err(|err| format!("failed to wait for ecosystem probe: {err}"));
            }
            Ok(None) => {}
            Err(err) => return Err(format!("failed to poll ecosystem probe: {err}")),
        }

        if started.elapsed() >= timeout {
            let _ = child.kill();
            return child
                .wait_with_output()
                .map(|output| (output, true))
                .map_err(|err| format!("failed to collect timed-out ecosystem probe: {err}"));
        }

        thread::sleep(Duration::from_millis(2));
    }
}

fn run_optimization_ecosystem_probe_command(
    command: &mut Command,
    timeout_ms: u64,
) -> Result<(Output, bool), String> {
    let child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("failed to start ecosystem probe: {err}"))?;
    wait_for_optimization_ecosystem_probe_output(child, timeout_ms)
}

fn probe_java_tool(tool: ExternalOptimizationTool) -> ExternalOptimizationProbe {
    let t0 = Instant::now();
    let primary_env_var = tool.env_var();
    let classpath =
        configured_java_classpath(tool).or_else(|| java_classpath_from_install_dirs(tool));
    let Some((classpath, env_var)) = classpath else {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::NotConfigured,
            None,
            primary_env_var,
            None,
            elapsed_ms(t0),
            format!(
                "set {primary_env_var} to a local jar/classpath or one of {:?} to an installation root for {}",
                tool.install_dir_env_vars(),
                tool.display_name()
            ),
        );
    };
    let javap = find_first_command(&["javap"]);
    let Some(javap) = javap else {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::RuntimeMissing,
            None,
            env_var,
            Some(classpath.to_string_lossy().to_string()),
            elapsed_ms(t0),
            "javap was not found on PATH; install a local JDK to probe Java solver jars"
                .to_string(),
        );
    };

    let mut last_error = String::new();
    let timeout_ms = optimization_ecosystem_probe_timeout_ms();
    for class_name in tool.java_probe_classes() {
        let mut command = Command::new(&javap);
        command.arg("-classpath").arg(&classpath).arg(class_name);
        match run_optimization_ecosystem_probe_command(&mut command, timeout_ms) {
            Ok((output, false)) if output.status.success() => {
                return probe_result(
                    tool,
                    ExternalOptimizationProbeStatus::Ready,
                    Some(javap),
                    env_var,
                    Some(classpath.to_string_lossy().to_string()),
                    elapsed_ms(t0),
                    format!(
                        "found Java API class {class_name} for {}",
                        tool.display_name()
                    ),
                );
            }
            Ok((output, timed_out)) => {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                last_error = if timed_out {
                    if stderr.is_empty() {
                        format!("javap probe timed out after {timeout_ms}ms")
                    } else {
                        format!("{stderr}; javap probe timed out after {timeout_ms}ms")
                    }
                } else {
                    stderr
                };
            }
            Err(err) => {
                last_error = err.to_string();
            }
        }
    }

    probe_result(
        tool,
        ExternalOptimizationProbeStatus::ArtifactMissing,
        Some(javap),
        env_var,
        Some(classpath.to_string_lossy().to_string()),
        elapsed_ms(t0),
        format!(
            "{} classpath did not expose any of {:?}: {}",
            tool.display_name(),
            tool.java_probe_classes(),
            last_error
        ),
    )
}

fn probe_python_tool(tool: ExternalOptimizationTool) -> ExternalOptimizationProbe {
    let t0 = Instant::now();
    let primary_env_var = tool.env_var();
    let configured_python = configured_python_command(tool)
        .or_else(|| python_command_from_install_dirs(tool))
        .or_else(|| default_python_probe_command().map(|python| (python, primary_env_var, None)));
    let Some((python, env_var, source_artifact)) = configured_python else {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::NotConfigured,
            None,
            primary_env_var,
            None,
            elapsed_ms(t0),
            format!(
                "set {primary_env_var} to a Python interpreter or one of {:?} to a Python package/venv root for {}",
                tool.install_dir_env_vars(),
                tool.display_name()
            ),
        );
    };
    let python_path_roots = python_module_search_paths_from_install_dirs(tool);
    let python_path = joined_python_path(&python_path_roots);
    let timeout_ms = optimization_ecosystem_probe_timeout_ms();
    let mut last_error = String::new();
    for module in tool.python_modules() {
        let mut command = Command::new(&python);
        command.arg("-c").arg(python_import_probe_code(module));
        if let Some(python_path) = python_path.as_ref() {
            command.env("PYTHONPATH", python_path);
        }
        match run_optimization_ecosystem_probe_command(&mut command, timeout_ms) {
            Ok((output, false)) if output.status.success() => {
                return probe_result(
                    tool,
                    ExternalOptimizationProbeStatus::Ready,
                    Some(python),
                    env_var,
                    Some(source_artifact.unwrap_or_else(|| module.to_string())),
                    elapsed_ms(t0),
                    format!(
                        "Python interpreter can import module '{module}' for {}",
                        tool.display_name()
                    ),
                );
            }
            Ok((output, timed_out)) => {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                last_error = if timed_out {
                    if stderr.is_empty() {
                        format!("Python import probe timed out after {timeout_ms}ms")
                    } else {
                        format!("{stderr}; Python import probe timed out after {timeout_ms}ms")
                    }
                } else {
                    stderr
                };
            }
            Err(err) => {
                last_error = err;
            }
        }
    }
    probe_result(
        tool,
        ExternalOptimizationProbeStatus::NotConfigured,
        Some(python),
        env_var,
        None,
        elapsed_ms(t0),
        if last_error.is_empty() {
            format!(
                "{} Python modules {:?} are not importable; set {env_var}",
                tool.display_name(),
                tool.python_modules()
            )
        } else {
            format!(
                "{} Python modules {:?} are not importable; set {env_var}: {last_error}",
                tool.display_name(),
                tool.python_modules()
            )
        },
    )
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

fn python_import_probe_code(module: &str) -> String {
    if module == "pycsp3" {
        return format!(
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module:?}) else 1)"
        );
    }
    format!("import {module}")
}

fn probe_julia_tool(tool: ExternalOptimizationTool) -> ExternalOptimizationProbe {
    let t0 = Instant::now();
    let env_var = tool.env_var();
    if let Some(project) = env::var_os(env_var).filter(|value| !value.is_empty()) {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::Ready,
            find_first_command(&["julia"]),
            env_var,
            Some(project.to_string_lossy().to_string()),
            elapsed_ms(t0),
            format!(
                "configured Julia project/runtime for {}",
                tool.display_name()
            ),
        );
    }
    probe_result(
        tool,
        ExternalOptimizationProbeStatus::NotConfigured,
        find_first_command(&["julia"]),
        env_var,
        None,
        elapsed_ms(t0),
        format!(
            "set {env_var} to a Julia project or environment with {}",
            tool.display_name()
        ),
    )
}

fn probe_native_tool(tool: ExternalOptimizationTool) -> ExternalOptimizationProbe {
    let t0 = Instant::now();
    let env_var = tool.env_var();
    if let Some(dir) = env::var_os(env_var).filter(|value| !value.is_empty()) {
        let command =
            find_command_in_install_dir(&PathBuf::from(&dir), tool.native_command_aliases())
                .or_else(|| find_first_command(tool.native_command_aliases()));
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::Ready,
            command,
            env_var,
            Some(dir.to_string_lossy().to_string()),
            elapsed_ms(t0),
            format!("configured native installation for {}", tool.display_name()),
        );
    }
    if let Some((command, source_env_var)) =
        command_from_install_dirs(tool, tool.native_command_aliases())
    {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::Ready,
            Some(command),
            source_env_var,
            None,
            elapsed_ms(t0),
            format!("found native command for {}", tool.display_name()),
        );
    }
    if let Some(command) = find_first_command(tool.native_command_aliases()) {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::Ready,
            Some(command),
            env_var,
            None,
            elapsed_ms(t0),
            format!("found native command for {}", tool.display_name()),
        );
    }
    probe_result(
        tool,
        ExternalOptimizationProbeStatus::NotConfigured,
        None,
        env_var,
        None,
        elapsed_ms(t0),
        format!(
            "set {env_var} to a local installation directory for {}",
            tool.display_name()
        ),
    )
}

fn probe_rust_tool(tool: ExternalOptimizationTool) -> ExternalOptimizationProbe {
    let t0 = Instant::now();
    let primary_env_var = tool.env_var();
    let manifest = configured_rust_manifest(tool).or_else(|| rust_manifest_from_install_dirs(tool));
    let Some((manifest, env_var)) = manifest else {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::NotConfigured,
            None,
            primary_env_var,
            None,
            elapsed_ms(t0),
            format!(
                "set {primary_env_var} to a Cargo.toml or one of {:?} to a crate root that uses {}",
                tool.install_dir_env_vars(),
                tool.display_name()
            ),
        );
    };
    let cargo = find_first_command(&["cargo"]);
    let Some(cargo) = cargo else {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::RuntimeMissing,
            None,
            env_var,
            Some(manifest.display().to_string()),
            elapsed_ms(t0),
            "cargo was not found on PATH; Rust crate integrations build through Cargo".to_string(),
        );
    };

    let raw = match fs::read_to_string(&manifest) {
        Ok(raw) => raw,
        Err(err) => {
            return probe_result(
                tool,
                ExternalOptimizationProbeStatus::ArtifactMissing,
                Some(cargo),
                env_var,
                Some(manifest.display().to_string()),
                elapsed_ms(t0),
                format!(
                    "failed to read Cargo manifest '{}': {err}",
                    manifest.display()
                ),
            );
        }
    };
    let dependency = tool
        .rust_dependency_names()
        .iter()
        .copied()
        .find(|name| cargo_manifest_mentions_dependency(&raw, name));
    match dependency {
        Some(name) => probe_result(
            tool,
            ExternalOptimizationProbeStatus::Ready,
            Some(cargo),
            env_var,
            Some(manifest.display().to_string()),
            elapsed_ms(t0),
            format!(
                "Cargo manifest '{}' references dependency '{}'",
                manifest.display(),
                name
            ),
        ),
        None => probe_result(
            tool,
            ExternalOptimizationProbeStatus::ArtifactMissing,
            Some(cargo),
            env_var,
            Some(manifest.display().to_string()),
            elapsed_ms(t0),
            format!(
                "Cargo manifest '{}' did not reference any of {:?}",
                manifest.display(),
                tool.rust_dependency_names()
            ),
        ),
    }
}

fn cargo_manifest_mentions_dependency(raw: &str, dependency: &str) -> bool {
    raw.lines().any(|line| {
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if trimmed.is_empty() {
            return false;
        }
        trimmed.starts_with(&format!("{dependency} "))
            || trimmed.starts_with(&format!("{dependency}="))
            || trimmed.starts_with(&format!("{dependency} ="))
            || trimmed.starts_with(&format!("\"{dependency}\""))
            || trimmed.starts_with(&format!("'{dependency}'"))
            || trimmed.contains(&format!("package = \"{dependency}\""))
            || trimmed.contains(&format!("package = '{dependency}'"))
    })
}

fn push_unique_env_name(names: &mut Vec<&'static str>, name: &'static str) {
    if !names.contains(&name) {
        names.push(name);
    }
}

fn configured_java_classpath(tool: ExternalOptimizationTool) -> Option<(OsString, &'static str)> {
    for env_var in tool.artifact_env_vars() {
        if let Some(value) = env::var_os(env_var).filter(|value| !value.is_empty()) {
            return Some((value, env_var));
        }
    }
    None
}

fn java_classpath_from_install_dirs(
    tool: ExternalOptimizationTool,
) -> Option<(OsString, &'static str)> {
    for env_var in tool.install_dir_env_vars() {
        let Some(value) = env::var_os(env_var).filter(|value| !value.is_empty()) else {
            continue;
        };
        let root = PathBuf::from(value);
        if let Some(classpath) = find_java_classpath_in_install_dir(&root) {
            return Some((classpath, env_var));
        }
    }
    None
}

fn find_java_classpath_in_install_dir(root: &Path) -> Option<OsString> {
    let mut jars = Vec::new();
    if is_jar_file(root) {
        jars.push(root.to_path_buf());
    }
    for dir in [
        root.to_path_buf(),
        root.join("lib"),
        root.join("share").join("java"),
        root.join("build").join("libs"),
        root.join("target"),
        root.join("target").join("dependency"),
    ] {
        collect_jar_files(&dir, &mut jars);
    }
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let child = entry.path();
            if !child.is_dir() {
                continue;
            }
            collect_jar_files(&child.join("lib"), &mut jars);
            collect_jar_files(&child.join("build").join("libs"), &mut jars);
            collect_jar_files(&child.join("target"), &mut jars);
            collect_jar_files(&child.join("target").join("dependency"), &mut jars);
        }
    }
    env::join_paths(jars).ok()
}

fn collect_jar_files(dir: &Path, jars: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if is_jar_file(&path) && !jars.contains(&path) {
            jars.push(path);
        }
    }
}

fn is_jar_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("jar"))
}

fn configured_python_command(
    tool: ExternalOptimizationTool,
) -> Option<(PathBuf, &'static str, Option<String>)> {
    for env_var in tool.artifact_env_vars() {
        let Some(value) = env::var_os(env_var).filter(|value| !value.is_empty()) else {
            continue;
        };
        let path = PathBuf::from(&value);
        if let Some(command) = find_command_path(&path) {
            return Some((command, env_var, Some(value.to_string_lossy().to_string())));
        }
    }
    None
}

fn python_command_from_install_dirs(
    tool: ExternalOptimizationTool,
) -> Option<(PathBuf, &'static str, Option<String>)> {
    for env_var in tool.install_dir_env_vars() {
        let Some(value) = env::var_os(env_var).filter(|value| !value.is_empty()) else {
            continue;
        };
        let root = PathBuf::from(&value);
        if let Some(command) = find_python_in_install_dir(&root) {
            return Some((command, env_var, Some(root.display().to_string())));
        }
    }
    None
}

fn find_python_in_install_dir(root: &Path) -> Option<PathBuf> {
    if executable_file(root) {
        return Some(root.to_path_buf());
    }
    for relative in [
        "bin/python3",
        "bin/python",
        ".venv/bin/python3",
        ".venv/bin/python",
        "venv/bin/python3",
        "venv/bin/python",
    ] {
        let candidate = root.join(relative);
        if executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn python_module_search_paths_from_install_dirs(tool: ExternalOptimizationTool) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for env_var in tool.install_dir_env_vars() {
        let Some(value) = env::var_os(env_var).filter(|value| !value.is_empty()) else {
            continue;
        };
        let root = PathBuf::from(value);
        for base in [root.clone(), root.join("src"), root.join("lib")] {
            if python_base_contains_probe_module(&base, tool.python_modules())
                && !paths.contains(&base)
            {
                paths.push(base);
            }
        }
    }
    paths
}

fn python_base_contains_probe_module(base: &Path, modules: &[&str]) -> bool {
    modules.iter().any(|module| {
        let first = module.split('.').next().unwrap_or(module);
        base.join(first).exists()
    })
}

fn joined_python_path(extra_roots: &[PathBuf]) -> Option<OsString> {
    if extra_roots.is_empty() {
        return None;
    }
    let mut paths = extra_roots.to_vec();
    if let Some(current) = env::var_os("PYTHONPATH") {
        paths.extend(env::split_paths(&current));
    }
    env::join_paths(paths).ok()
}

fn command_from_install_dirs(
    tool: ExternalOptimizationTool,
    aliases: &[&str],
) -> Option<(PathBuf, &'static str)> {
    for env_var in tool.install_dir_env_vars() {
        let Some(value) = env::var_os(env_var).filter(|value| !value.is_empty()) else {
            continue;
        };
        if let Some(command) = find_command_in_install_dir(&PathBuf::from(value), aliases) {
            return Some((command, env_var));
        }
    }
    None
}

fn find_command_in_install_dir(root: &Path, aliases: &[&str]) -> Option<PathBuf> {
    for candidate_dir in [root.to_path_buf(), root.join("bin")] {
        if let Some(path) = find_command_in_dir(&candidate_dir, aliases) {
            return Some(path);
        }
    }
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let child = entry.path();
            if !child.is_dir() {
                continue;
            }
            let child_bin = child.join("bin");
            if let Some(path) = find_command_in_dir(&child_bin, aliases) {
                return Some(path);
            }
            if let Ok(platforms) = fs::read_dir(&child_bin) {
                for platform in platforms.flatten() {
                    let platform_dir = platform.path();
                    if !platform_dir.is_dir() {
                        continue;
                    }
                    if let Some(path) = find_command_in_dir(&platform_dir, aliases) {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
}

fn find_command_in_dir(dir: &Path, aliases: &[&str]) -> Option<PathBuf> {
    aliases
        .iter()
        .map(|alias| dir.join(alias))
        .find(|candidate| executable_file(candidate))
}

fn configured_rust_manifest(tool: ExternalOptimizationTool) -> Option<(PathBuf, &'static str)> {
    for env_var in tool.artifact_env_vars() {
        let Some(value) = env::var_os(env_var).filter(|value| !value.is_empty()) else {
            continue;
        };
        let path = PathBuf::from(value);
        if let Some(manifest) = manifest_path(&path) {
            return Some((manifest, env_var));
        }
    }
    None
}

fn rust_manifest_from_install_dirs(
    tool: ExternalOptimizationTool,
) -> Option<(PathBuf, &'static str)> {
    for env_var in tool.install_dir_env_vars() {
        let Some(value) = env::var_os(env_var).filter(|value| !value.is_empty()) else {
            continue;
        };
        let root = PathBuf::from(value);
        if let Some(manifest) = manifest_path(&root).or_else(|| child_manifest_path(&root)) {
            return Some((manifest, env_var));
        }
    }
    None
}

fn manifest_path(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        return Some(path.to_path_buf());
    }
    let candidate = path.join("Cargo.toml");
    candidate.is_file().then_some(candidate)
}

fn child_manifest_path(root: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let child = entry.path();
        if !child.is_dir() {
            continue;
        }
        let candidate = child.join("Cargo.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn probe_result(
    tool: ExternalOptimizationTool,
    status: ExternalOptimizationProbeStatus,
    command: Option<PathBuf>,
    env_var: &'static str,
    artifact: Option<String>,
    elapsed_ms: f64,
    message: String,
) -> ExternalOptimizationProbe {
    ExternalOptimizationProbe {
        tool,
        ecosystem: tool.ecosystem(),
        status,
        command,
        env_var,
        artifact,
        elapsed_ms,
        message,
    }
}

fn find_first_command(aliases: &[&str]) -> Option<PathBuf> {
    aliases.iter().find_map(|alias| find_command(alias))
}

fn find_command_path(path: &Path) -> Option<PathBuf> {
    if path.components().count() > 1 {
        executable_file(path).then(|| path.to_path_buf())
    } else {
        path.to_str().and_then(find_command)
    }
}

fn find_command(alias: &str) -> Option<PathBuf> {
    let alias_path = Path::new(alias);
    if alias_path.components().count() > 1 {
        return executable_file(alias_path).then(|| alias_path.to_path_buf());
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(alias))
        .find(|candidate| executable_file(candidate))
}

fn executable_file(path: impl AsRef<Path>) -> bool {
    path.as_ref().is_file()
}

fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn ecosystem_probe_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_optimization_ecosystem_probe_output(child, 10).expect("timeout output");
        let stderr = String::from_utf8_lossy(&output.stderr);

        assert!(timed_out);
        assert!(!output.status.success());
        assert!(stderr.is_empty());
    }

    #[test]
    fn ecosystem_tool_metadata_covers_supported_languages() {
        assert_eq!(ExternalOptimizationTool::all().len(), 96);
        assert_eq!(
            ExternalOptimizationTool::ChocoSolver.ecosystem(),
            ExternalOptimizationEcosystem::Java
        );
        assert_eq!(
            ExternalOptimizationTool::SavileRow.ecosystem(),
            ExternalOptimizationEcosystem::Java
        );
        assert_eq!(
            ExternalOptimizationTool::Pyomo.ecosystem(),
            ExternalOptimizationEcosystem::Python
        );
        assert_eq!(
            ExternalOptimizationTool::Cpmpy.ecosystem(),
            ExternalOptimizationEcosystem::Python
        );
        assert_eq!(
            ExternalOptimizationTool::Jump.ecosystem(),
            ExternalOptimizationEcosystem::Julia
        );
        assert_eq!(
            ExternalOptimizationTool::Ampl.ecosystem(),
            ExternalOptimizationEcosystem::Native
        );
        assert_eq!(
            ExternalOptimizationTool::FastDownward.ecosystem(),
            ExternalOptimizationEcosystem::Native
        );
        assert_eq!(
            ExternalOptimizationTool::Conjure.ecosystem(),
            ExternalOptimizationEcosystem::Native
        );
        assert_eq!(
            ExternalOptimizationTool::HighsCli.ecosystem(),
            ExternalOptimizationEcosystem::Native
        );
        assert_eq!(
            ExternalOptimizationTool::Ipopt.ecosystem(),
            ExternalOptimizationEcosystem::Native
        );
        assert_eq!(
            ExternalOptimizationTool::Z3.ecosystem(),
            ExternalOptimizationEcosystem::Native
        );
        assert_eq!(
            ExternalOptimizationTool::Casadi.ecosystem(),
            ExternalOptimizationEcosystem::Python
        );
        assert_eq!(
            ExternalOptimizationTool::OrToolsPdlp.ecosystem(),
            ExternalOptimizationEcosystem::Python
        );
        assert_eq!(
            ExternalOptimizationTool::OrToolsCpSat.ecosystem(),
            ExternalOptimizationEcosystem::Native
        );
        assert_eq!(
            ExternalOptimizationTool::GoodLp.ecosystem(),
            ExternalOptimizationEcosystem::Rust
        );
        assert_eq!(
            ExternalOptimizationTool::GurobiRust.ecosystem(),
            ExternalOptimizationEcosystem::Rust
        );
        assert_eq!(
            ExternalOptimizationTool::ChocoSolver.env_var(),
            "CHOCO_SOLVER_CLASSPATH"
        );
        assert_eq!(ExternalOptimizationTool::Cvxpy.env_var(), "CVXPY_PYTHON");
        assert_eq!(ExternalOptimizationTool::Hexaly.env_var(), "HEXALY_DIR");
        assert_eq!(
            ExternalOptimizationTool::FastDownward.env_var(),
            "FAST_DOWNWARD_DIR"
        );
        assert_eq!(ExternalOptimizationTool::Mosek.env_var(), "MOSEK_HOME");
        assert_eq!(
            ExternalOptimizationTool::MosekPython.env_var(),
            "MOSEK_PYTHON"
        );
        assert_eq!(ExternalOptimizationTool::Copt.env_var(), "COPT_PYTHON");
        assert_eq!(
            ExternalOptimizationTool::GurobiRust.env_var(),
            "GUROBI_RUST_CARGO_MANIFEST"
        );
        assert_eq!(
            ExternalOptimizationTool::CplexRust.env_var(),
            "CPLEX_RUST_CARGO_MANIFEST"
        );
        assert_eq!(
            ExternalOptimizationTool::IpoptRust.env_var(),
            "IPOPT_RUST_CARGO_MANIFEST"
        );
        assert_eq!(ExternalOptimizationTool::Z3.env_var(), "Z3_DIR");
        assert_eq!(
            ExternalOptimizationTool::OptiMathSat.env_var(),
            "OPTIMATHSAT_DIR"
        );
        assert_eq!(ExternalOptimizationTool::Casadi.env_var(), "CASADI_PYTHON");
        assert!(ExternalOptimizationTool::OjAlgo
            .java_probe_classes()
            .contains(&"org.ojalgo.optimisation.ExpressionsBasedModel"));
        assert!(ExternalOptimizationTool::Timefold
            .java_probe_classes()
            .contains(&"ai.timefold.solver.core.api.solver.SolverFactory"));
        assert!(ExternalOptimizationTool::Sat4j
            .java_probe_classes()
            .contains(&"org.sat4j.minisat.SolverFactory"));
        assert!(ExternalOptimizationTool::GurobiPy
            .python_modules()
            .contains(&"gurobipy"));
        assert!(ExternalOptimizationTool::CplexPython
            .python_modules()
            .contains(&"cplex"));
        assert_eq!(
            ExternalOptimizationTool::CplexPython.env_var(),
            "CPLEX_PYTHON"
        );
        assert!(ExternalOptimizationTool::CplexPython
            .artifact_env_vars()
            .contains(&"CPLEX_STUDIO_DIR"));
        assert!(ExternalOptimizationTool::CplexPython
            .install_dir_env_vars()
            .contains(&"CPLEX_HOME"));
        assert!(ExternalOptimizationTool::XpressPython
            .python_modules()
            .contains(&"xpress"));
        assert_eq!(
            ExternalOptimizationTool::XpressPython.env_var(),
            "XPRESS_PYTHON"
        );
        assert!(ExternalOptimizationTool::XpressPython
            .artifact_env_vars()
            .contains(&"XPRESSDIR"));
        assert!(ExternalOptimizationTool::XpressPython
            .install_dir_env_vars()
            .contains(&"XPRESS_HOME"));
        assert!(ExternalOptimizationTool::MosekPython
            .python_modules()
            .contains(&"mosek"));
        assert!(ExternalOptimizationTool::MosekPython
            .artifact_env_vars()
            .contains(&"MOSEKLM_LICENSE_FILE"));
        assert!(ExternalOptimizationTool::MosekPython
            .install_dir_env_vars()
            .contains(&"MOSEK_HOME"));
        assert_eq!(
            ExternalOptimizationTool::Copt.ecosystem(),
            ExternalOptimizationEcosystem::Python
        );
        assert!(ExternalOptimizationTool::Copt
            .python_modules()
            .contains(&"coptpy"));
        assert!(ExternalOptimizationTool::Copt
            .install_dir_env_vars()
            .contains(&"COPT_HOME"));
        assert!(ExternalOptimizationTool::OrToolsGlop
            .python_modules()
            .contains(&"ortools.linear_solver.pywraplp"));
        assert_eq!(
            ExternalOptimizationTool::OrToolsPdlp.env_var(),
            "ORTOOLS_PDLP_PYTHON"
        );
        assert!(ExternalOptimizationTool::OrToolsPdlp
            .artifact_env_vars()
            .contains(&"ORTOOLS_PYTHON"));
        assert_eq!(
            ExternalOptimizationTool::OrToolsCpSat.env_var(),
            "ORTOOLS_CP_SAT_DIR"
        );
        assert!(ExternalOptimizationTool::OrToolsCpSat
            .artifact_env_vars()
            .contains(&"FZN_CP_SAT_CMD"));
        assert!(ExternalOptimizationTool::OrToolsCpSat
            .install_dir_env_vars()
            .contains(&"ORTOOLS_HOME"));
        assert!(ExternalOptimizationTool::OrToolsCpSat
            .native_command_aliases()
            .contains(&"fzn-cp-sat"));
        assert!(ExternalOptimizationTool::Cpmpy
            .python_modules()
            .contains(&"cpmpy"));
        assert_eq!(ExternalOptimizationTool::Clingo.env_var(), "CLINGO_PYTHON");
        assert!(ExternalOptimizationTool::Clingo
            .python_modules()
            .contains(&"clingo"));
        assert_eq!(ExternalOptimizationTool::Cvc5.env_var(), "CVC5_PYTHON");
        assert!(ExternalOptimizationTool::Cvc5
            .python_modules()
            .contains(&"cvc5"));
        assert_eq!(
            ExternalOptimizationTool::Bitwuzla.env_var(),
            "BITWUZLA_PYTHON"
        );
        assert!(ExternalOptimizationTool::Bitwuzla
            .python_modules()
            .contains(&"bitwuzla"));
        assert_eq!(ExternalOptimizationTool::Proxqp.env_var(), "PROXQP_PYTHON");
        assert!(ExternalOptimizationTool::Proxqp
            .python_modules()
            .contains(&"proxsuite"));
        assert!(ExternalOptimizationTool::Osqp
            .python_modules()
            .contains(&"osqp"));
        assert!(ExternalOptimizationTool::Hexaly
            .native_command_aliases()
            .contains(&"hexaly"));
        assert!(ExternalOptimizationTool::Ipopt
            .native_command_aliases()
            .contains(&"ipopt"));
        assert!(ExternalOptimizationTool::Sdpa
            .native_command_aliases()
            .contains(&"sdpa_gmp"));
        assert!(ExternalOptimizationTool::OpenWbo
            .native_command_aliases()
            .contains(&"open-wbo"));
        assert!(ExternalOptimizationTool::FastDownward
            .native_command_aliases()
            .contains(&"fast-downward.py"));
        assert!(ExternalOptimizationTool::LpgTd
            .native_command_aliases()
            .contains(&"lpg-td"));
        assert!(ExternalOptimizationTool::Optic
            .native_command_aliases()
            .contains(&"optic"));
        assert!(ExternalOptimizationTool::Enhsp
            .native_command_aliases()
            .contains(&"enhsp"));
        assert!(ExternalOptimizationTool::Z3
            .native_command_aliases()
            .contains(&"z3"));
        assert!(ExternalOptimizationTool::Yices
            .native_command_aliases()
            .contains(&"yices-smt2"));
        assert!(ExternalOptimizationTool::OptiMathSat
            .native_command_aliases()
            .contains(&"optimathsat"));
        assert!(ExternalOptimizationTool::SmtInterpol
            .native_command_aliases()
            .contains(&"smtinterpol"));
        assert_eq!(
            ExternalOptimizationTool::QsoptExCli.env_var(),
            "QSOPT_EX_DIR"
        );
        assert!(ExternalOptimizationTool::QsoptExCli
            .artifact_env_vars()
            .contains(&"QSOPT_EX_CMD"));
        assert!(ExternalOptimizationTool::QsoptExCli
            .install_dir_env_vars()
            .contains(&"QSOPT_EX_HOME"));
        assert!(ExternalOptimizationTool::QsoptExCli
            .native_command_aliases()
            .contains(&"esolver"));
        assert!(ExternalOptimizationTool::HighsCli
            .native_command_aliases()
            .contains(&"highs"));
        assert!(ExternalOptimizationTool::LindoCli
            .native_command_aliases()
            .contains(&"runlindo"));
        assert!(ExternalOptimizationTool::HighsRust
            .rust_dependency_names()
            .contains(&"highs-sys"));
        assert!(ExternalOptimizationTool::GurobiRust
            .rust_dependency_names()
            .contains(&"grb"));
        assert!(ExternalOptimizationTool::GurobiRust
            .artifact_env_vars()
            .contains(&"GRB_LICENSE_FILE"));
        assert!(ExternalOptimizationTool::CplexRust
            .rust_dependency_names()
            .contains(&"cplex-rs"));
        assert!(ExternalOptimizationTool::CplexRust
            .install_dir_env_vars()
            .contains(&"CPLEX_STUDIO_DIR"));
        assert!(ExternalOptimizationTool::IpoptRust
            .rust_dependency_names()
            .contains(&"ipopt-sys"));
        assert!(ExternalOptimizationTool::IpoptRust
            .install_dir_env_vars()
            .contains(&"IPOPT_HOME"));
        assert!(ExternalOptimizationTool::ChocoSolver
            .install_dir_env_vars()
            .contains(&"CHOCO_HOME"));
        assert!(ExternalOptimizationTool::Pyomo
            .artifact_env_vars()
            .contains(&"PYOMO_PYTHON"));
        assert!(ExternalOptimizationTool::PySat
            .artifact_env_vars()
            .contains(&"PYSAT_PYTHON"));
        assert!(ExternalOptimizationTool::Docplex
            .install_dir_env_vars()
            .contains(&"CPLEX_STUDIO_DIR"));
        assert!(ExternalOptimizationTool::Conjure
            .install_dir_env_vars()
            .contains(&"CONJURE_HOME"));
        assert!(ExternalOptimizationTool::FastDownward
            .install_dir_env_vars()
            .contains(&"FAST_DOWNWARD_HOME"));
        assert!(ExternalOptimizationTool::Symphony
            .install_dir_env_vars()
            .contains(&"COINOR_HOME"));
        assert!(ExternalOptimizationTool::Z3
            .install_dir_env_vars()
            .contains(&"Z3_HOME"));
        assert!(ExternalOptimizationTool::OptiMathSat
            .install_dir_env_vars()
            .contains(&"OPTIMATHSAT_HOME"));
        assert!(ExternalOptimizationTool::Cvc5
            .artifact_env_vars()
            .contains(&"CVC5_CMD"));
        assert!(ExternalOptimizationTool::Mosek
            .artifact_env_vars()
            .contains(&"MOSEKLM_LICENSE_FILE"));
        assert!(ExternalOptimizationTool::CplexCli
            .install_dir_env_vars()
            .contains(&"CPLEX_STUDIO_DIR"));
        assert!(ExternalOptimizationTool::GurobiCli
            .artifact_env_vars()
            .contains(&"GUROBI_CL_CMD"));
        assert!(ExternalOptimizationTool::HighsRust
            .install_dir_env_vars()
            .contains(&"HIGHS_HOME"));
        assert_eq!(ExternalOptimizationEcosystem::Python.as_str(), "python");
    }

    #[test]
    fn cargo_manifest_dependency_probe_handles_common_forms() {
        let raw = r#"
            [dependencies]
            good_lp = "1"
            highs-wrapper = { package = "highs", version = "0.1" }
            cplex = { package = "cplex-rs", version = "0.1" }
            cplex_sys = "0.1"
            grb = "3"
            ipopt-sys = "0.6"
        "#;
        assert!(cargo_manifest_mentions_dependency(raw, "good_lp"));
        assert!(cargo_manifest_mentions_dependency(raw, "highs"));
        assert!(cargo_manifest_mentions_dependency(raw, "cplex-rs"));
        assert!(cargo_manifest_mentions_dependency(raw, "cplex_sys"));
        assert!(cargo_manifest_mentions_dependency(raw, "grb"));
        assert!(cargo_manifest_mentions_dependency(raw, "ipopt-sys"));
        assert!(!cargo_manifest_mentions_dependency(raw, "argmin"));
    }

    #[test]
    fn install_dir_lookup_handles_ecosystem_layouts() {
        let root = std::env::temp_dir().join(format!(
            "des-external-optimization-ecosystem-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let jar = root.join("share").join("java").join("choco-solver.jar");
        std::fs::create_dir_all(jar.parent().unwrap()).unwrap();
        std::fs::write(&jar, b"").unwrap();
        let classpath = find_java_classpath_in_install_dir(&root).unwrap();
        assert!(env::split_paths(&classpath).any(|path| path == jar));

        let native = root.join("hexaly").join("bin").join("macos").join("hexaly");
        std::fs::create_dir_all(native.parent().unwrap()).unwrap();
        std::fs::write(&native, b"").unwrap();
        assert_eq!(
            find_command_in_install_dir(&root, &["hexaly"]),
            Some(native)
        );

        let python = root.join(".venv").join("bin").join("python3");
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, b"").unwrap();
        assert_eq!(find_python_in_install_dir(&root), Some(python));

        let manifest = root.join("crate").join("Cargo.toml");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(&manifest, b"[dependencies]\nhighs = \"1\"\n").unwrap();
        assert_eq!(child_manifest_path(&root), Some(manifest));

        std::fs::remove_dir_all(root).unwrap();
    }
}
