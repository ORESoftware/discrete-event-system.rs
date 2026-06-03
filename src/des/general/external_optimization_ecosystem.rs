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
use std::process::Command;
use std::time::Instant;

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
    JMetal,
    MoeaFramework,
    Ecj,
    OjAlgo,
    OrToolsJava,
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
    GoodLp,
    LpModeler,
    RustLinprog,
    Argmin,
    NloptRs,
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
            ExternalOptimizationTool::JMetal,
            ExternalOptimizationTool::MoeaFramework,
            ExternalOptimizationTool::Ecj,
            ExternalOptimizationTool::OjAlgo,
            ExternalOptimizationTool::OrToolsJava,
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
            ExternalOptimizationTool::GoodLp,
            ExternalOptimizationTool::LpModeler,
            ExternalOptimizationTool::RustLinprog,
            ExternalOptimizationTool::Argmin,
            ExternalOptimizationTool::NloptRs,
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
            ExternalOptimizationTool::JMetal => "jmetal",
            ExternalOptimizationTool::MoeaFramework => "moea-framework",
            ExternalOptimizationTool::Ecj => "ecj",
            ExternalOptimizationTool::OjAlgo => "ojalgo",
            ExternalOptimizationTool::OrToolsJava => "ortools-java",
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
            ExternalOptimizationTool::GoodLp => "good-lp",
            ExternalOptimizationTool::LpModeler => "lp-modeler",
            ExternalOptimizationTool::RustLinprog => "rust-linprog",
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::NloptRs => "nlopt-rs",
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
            ExternalOptimizationTool::JMetal => "jMetal",
            ExternalOptimizationTool::MoeaFramework => "MOEA Framework",
            ExternalOptimizationTool::Ecj => "ECJ",
            ExternalOptimizationTool::OjAlgo => "ojAlgo",
            ExternalOptimizationTool::OrToolsJava => "Google OR-Tools Java",
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
            ExternalOptimizationTool::GoodLp => "good_lp",
            ExternalOptimizationTool::LpModeler => "lp-modeler",
            ExternalOptimizationTool::RustLinprog => "rust-linprog",
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::NloptRs => "nlopt-rs",
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
            | ExternalOptimizationTool::OrToolsJava => ExternalOptimizationEcosystem::Java,
            ExternalOptimizationTool::Pyomo
            | ExternalOptimizationTool::Pulp
            | ExternalOptimizationTool::Cvxpy
            | ExternalOptimizationTool::Cvxopt
            | ExternalOptimizationTool::PyScipOpt
            | ExternalOptimizationTool::PythonMip
            | ExternalOptimizationTool::GurobiPy
            | ExternalOptimizationTool::Docplex
            | ExternalOptimizationTool::OrToolsPython
            | ExternalOptimizationTool::ScipyOptimize => ExternalOptimizationEcosystem::Python,
            ExternalOptimizationTool::Jump => ExternalOptimizationEcosystem::Julia,
            ExternalOptimizationTool::Ampl
            | ExternalOptimizationTool::Gams
            | ExternalOptimizationTool::Hexaly => ExternalOptimizationEcosystem::Native,
            ExternalOptimizationTool::GoodLp
            | ExternalOptimizationTool::LpModeler
            | ExternalOptimizationTool::RustLinprog
            | ExternalOptimizationTool::Argmin
            | ExternalOptimizationTool::NloptRs
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
            ExternalOptimizationTool::JMetal => "JMETAL_CLASSPATH",
            ExternalOptimizationTool::MoeaFramework => "MOEA_FRAMEWORK_CLASSPATH",
            ExternalOptimizationTool::Ecj => "ECJ_CLASSPATH",
            ExternalOptimizationTool::OjAlgo => "OJALGO_CLASSPATH",
            ExternalOptimizationTool::OrToolsJava => "ORTOOLS_JAVA_CLASSPATH",
            ExternalOptimizationTool::Pyomo => "PYOMO_PYTHON",
            ExternalOptimizationTool::Pulp => "PULP_PYTHON",
            ExternalOptimizationTool::Cvxpy => "CVXPY_PYTHON",
            ExternalOptimizationTool::Cvxopt => "CVXOPT_PYTHON",
            ExternalOptimizationTool::PyScipOpt => "PYSCIPOPT_PYTHON",
            ExternalOptimizationTool::PythonMip => "PYTHON_MIP_PYTHON",
            ExternalOptimizationTool::GurobiPy => "GUROBIPY_PYTHON",
            ExternalOptimizationTool::Docplex => "DOCPLEX_PYTHON",
            ExternalOptimizationTool::OrToolsPython => "ORTOOLS_PYTHON_PYTHON",
            ExternalOptimizationTool::ScipyOptimize => "SCIPY_OPTIMIZE_PYTHON",
            ExternalOptimizationTool::Jump => "JUMP_JULIA",
            ExternalOptimizationTool::Ampl => "AMPL_DIR",
            ExternalOptimizationTool::Gams => "GAMS_DIR",
            ExternalOptimizationTool::Hexaly => "HEXALY_DIR",
            ExternalOptimizationTool::GoodLp => "GOOD_LP_CARGO_MANIFEST",
            ExternalOptimizationTool::LpModeler => "LP_MODELER_CARGO_MANIFEST",
            ExternalOptimizationTool::RustLinprog => "RUST_LINPROG_CARGO_MANIFEST",
            ExternalOptimizationTool::Argmin => "ARGMIN_CARGO_MANIFEST",
            ExternalOptimizationTool::NloptRs => "NLOPT_RS_CARGO_MANIFEST",
            ExternalOptimizationTool::HighsRust => "HIGHS_RS_CARGO_MANIFEST",
            ExternalOptimizationTool::ScipRust => "SCIP_RS_CARGO_MANIFEST",
            ExternalOptimizationTool::CbcRust => "CBC_RS_CARGO_MANIFEST",
        }
    }

    pub fn artifact_env_vars(self) -> Vec<&'static str> {
        let mut names = vec![self.env_var()];
        for name in match self {
            ExternalOptimizationTool::Pyomo => &["PYOMO_PYTHON"][..],
            ExternalOptimizationTool::Pulp => &["PULP_PYTHON"],
            ExternalOptimizationTool::Cvxpy => &["CVXPY_PYTHON"],
            ExternalOptimizationTool::Cvxopt => &["CVXOPT_PYTHON"],
            ExternalOptimizationTool::PyScipOpt => &["PYSCIPOPT_PYTHON"],
            ExternalOptimizationTool::PythonMip => &["PYTHON_MIP_PYTHON"],
            ExternalOptimizationTool::GurobiPy => &["GUROBIPY_PYTHON"],
            ExternalOptimizationTool::Docplex => &["DOCPLEX_PYTHON"],
            ExternalOptimizationTool::OrToolsPython => &["ORTOOLS_PYTHON"],
            ExternalOptimizationTool::ScipyOptimize => &["SCIPY_PYTHON"],
            ExternalOptimizationTool::Jump => &["JULIA_PROJECT"],
            ExternalOptimizationTool::GoodLp => &["GOOD_LP_CARGO_MANIFEST"],
            ExternalOptimizationTool::LpModeler => &["LP_MODELER_CARGO_MANIFEST"],
            ExternalOptimizationTool::RustLinprog => &["RUST_LINPROG_CARGO_MANIFEST"],
            ExternalOptimizationTool::Argmin => &["ARGMIN_CARGO_MANIFEST"],
            ExternalOptimizationTool::NloptRs => &["NLOPT_RS_CARGO_MANIFEST"],
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
            ExternalOptimizationTool::JMetal => &["JMETAL_HOME", "JMETAL_DIR"],
            ExternalOptimizationTool::MoeaFramework => &["MOEA_FRAMEWORK_HOME", "MOEA_HOME"],
            ExternalOptimizationTool::Ecj => &["ECJ_HOME", "ECJ_DIR"],
            ExternalOptimizationTool::OjAlgo => &["OJALGO_HOME", "OJALGO_DIR"],
            ExternalOptimizationTool::OrToolsJava => &["ORTOOLS_JAVA_HOME", "ORTOOLS_HOME"],
            ExternalOptimizationTool::Pyomo => &["PYOMO_HOME", "PYOMO_DIR"],
            ExternalOptimizationTool::Pulp => &["PULP_HOME", "PULP_DIR"],
            ExternalOptimizationTool::Cvxpy => &["CVXPY_HOME", "CVXPY_DIR"],
            ExternalOptimizationTool::Cvxopt => &["CVXOPT_HOME", "CVXOPT_DIR"],
            ExternalOptimizationTool::PyScipOpt => &["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
            ExternalOptimizationTool::PythonMip => &["PYTHON_MIP_HOME", "PYTHON_MIP_DIR"],
            ExternalOptimizationTool::GurobiPy => &["GUROBI_HOME"],
            ExternalOptimizationTool::Docplex => {
                &["DOCPLEX_HOME", "CPLEX_STUDIO_DIR", "CPLEX_HOME"]
            }
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
            ExternalOptimizationTool::NloptRs => &["NLOPT_DIR", "NLOPT_HOME"],
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
            ExternalOptimizationTool::Docplex => &["docplex.mp.model"],
            ExternalOptimizationTool::OrToolsPython => &["ortools.sat.python.cp_model"],
            ExternalOptimizationTool::ScipyOptimize => &["scipy.optimize"],
            _ => &[],
        }
    }

    pub fn native_command_aliases(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::Ampl => &["ampl"],
            ExternalOptimizationTool::Gams => &["gams"],
            ExternalOptimizationTool::Hexaly => &["hexaly", "localsolver"],
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
    for class_name in tool.java_probe_classes() {
        match Command::new(&javap)
            .arg("-classpath")
            .arg(&classpath)
            .arg(class_name)
            .output()
        {
            Ok(output) if output.status.success() => {
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
            Ok(output) => {
                last_error = String::from_utf8_lossy(&output.stderr).trim().to_string();
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
        .or_else(|| {
            find_first_command(&["python3", "python"]).map(|python| (python, primary_env_var, None))
        });
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
    for module in tool.python_modules() {
        let mut command = Command::new(&python);
        command.arg("-c").arg(format!("import {module}"));
        if let Some(python_path) = python_path.as_ref() {
            command.env("PYTHONPATH", python_path);
        }
        match command.output() {
            Ok(output) if output.status.success() => {
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
            Ok(_) | Err(_) => {}
        }
    }
    probe_result(
        tool,
        ExternalOptimizationProbeStatus::NotConfigured,
        Some(python),
        env_var,
        None,
        elapsed_ms(t0),
        format!(
            "{} Python modules {:?} are not importable; set {env_var}",
            tool.display_name(),
            tool.python_modules()
        ),
    )
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
    fn ecosystem_tool_metadata_covers_supported_languages() {
        assert_eq!(ExternalOptimizationTool::all().len(), 32);
        assert_eq!(
            ExternalOptimizationTool::ChocoSolver.ecosystem(),
            ExternalOptimizationEcosystem::Java
        );
        assert_eq!(
            ExternalOptimizationTool::Pyomo.ecosystem(),
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
            ExternalOptimizationTool::GoodLp.ecosystem(),
            ExternalOptimizationEcosystem::Rust
        );
        assert_eq!(
            ExternalOptimizationTool::ChocoSolver.env_var(),
            "CHOCO_SOLVER_CLASSPATH"
        );
        assert_eq!(ExternalOptimizationTool::Cvxpy.env_var(), "CVXPY_PYTHON");
        assert_eq!(ExternalOptimizationTool::Hexaly.env_var(), "HEXALY_DIR");
        assert!(ExternalOptimizationTool::OjAlgo
            .java_probe_classes()
            .contains(&"org.ojalgo.optimisation.ExpressionsBasedModel"));
        assert!(ExternalOptimizationTool::Timefold
            .java_probe_classes()
            .contains(&"ai.timefold.solver.core.api.solver.SolverFactory"));
        assert!(ExternalOptimizationTool::GurobiPy
            .python_modules()
            .contains(&"gurobipy"));
        assert!(ExternalOptimizationTool::Hexaly
            .native_command_aliases()
            .contains(&"hexaly"));
        assert!(ExternalOptimizationTool::HighsRust
            .rust_dependency_names()
            .contains(&"highs-sys"));
        assert!(ExternalOptimizationTool::ChocoSolver
            .install_dir_env_vars()
            .contains(&"CHOCO_HOME"));
        assert!(ExternalOptimizationTool::Pyomo
            .artifact_env_vars()
            .contains(&"PYOMO_PYTHON"));
        assert!(ExternalOptimizationTool::Docplex
            .install_dir_env_vars()
            .contains(&"CPLEX_STUDIO_DIR"));
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
        "#;
        assert!(cargo_manifest_mentions_dependency(raw, "good_lp"));
        assert!(cargo_manifest_mentions_dependency(raw, "highs"));
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
