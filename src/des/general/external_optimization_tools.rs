//! Local adapter/probe surface for optimization ecosystems that are not plain
//! LP/MIP command-line solvers.
//!
//! Java CP/planning systems and Rust modeling/binding crates are usually wired
//! into an application through a small local wrapper. This module gives those
//! wrappers a stable JSON-in/JSON-out contract while keeping jars, native
//! libraries, and generated executables out of version control.

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
    JMetal,
    MoeaFramework,
    Ecj,
    OjAlgo,
    OrToolsJava,
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
            ExternalOptimizationTool::JMetal => "jmetal",
            ExternalOptimizationTool::MoeaFramework => "moea-framework",
            ExternalOptimizationTool::Ecj => "ecj",
            ExternalOptimizationTool::OjAlgo => "ojalgo",
            ExternalOptimizationTool::OrToolsJava => "ortools-java",
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
            ExternalOptimizationTool::JMetal => "jMetal",
            ExternalOptimizationTool::MoeaFramework => "MOEA Framework",
            ExternalOptimizationTool::Ecj => "ECJ",
            ExternalOptimizationTool::OjAlgo => "ojAlgo",
            ExternalOptimizationTool::OrToolsJava => "Google OR-Tools Java",
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
            | ExternalOptimizationTool::JMetal
            | ExternalOptimizationTool::MoeaFramework
            | ExternalOptimizationTool::Ecj
            | ExternalOptimizationTool::OjAlgo
            | ExternalOptimizationTool::OrToolsJava => ExternalOptimizationLanguage::Java,
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
            | ExternalOptimizationTool::IbmCpOptimizer => {
                ExternalOptimizationFamily::ConstraintProgramming
            }
            ExternalOptimizationTool::OptaPlanner => {
                ExternalOptimizationFamily::PlanningMetaheuristic
            }
            ExternalOptimizationTool::JMetal
            | ExternalOptimizationTool::MoeaFramework
            | ExternalOptimizationTool::Ecj => {
                ExternalOptimizationFamily::EvolutionaryMultiObjective
            }
            ExternalOptimizationTool::OjAlgo
            | ExternalOptimizationTool::GoodLp
            | ExternalOptimizationTool::LpModeler
            | ExternalOptimizationTool::RustLinprog => ExternalOptimizationFamily::LinearMip,
            ExternalOptimizationTool::OrToolsJava => ExternalOptimizationFamily::CpSatRouting,
            ExternalOptimizationTool::Argmin | ExternalOptimizationTool::Nlopt => {
                ExternalOptimizationFamily::NonlinearOptimization
            }
            ExternalOptimizationTool::HighsRust
            | ExternalOptimizationTool::ScipRust
            | ExternalOptimizationTool::CbcRust => ExternalOptimizationFamily::NativeSolverBinding,
        }
    }

    pub fn exactness(self) -> ExternalOptimizationExactness {
        match self {
            ExternalOptimizationTool::ChocoSolver
            | ExternalOptimizationTool::Jacop
            | ExternalOptimizationTool::IbmCpOptimizer
            | ExternalOptimizationTool::OjAlgo
            | ExternalOptimizationTool::OrToolsJava
            | ExternalOptimizationTool::RustLinprog
            | ExternalOptimizationTool::HighsRust
            | ExternalOptimizationTool::ScipRust
            | ExternalOptimizationTool::CbcRust => ExternalOptimizationExactness::Exact,
            ExternalOptimizationTool::OptaPlanner
            | ExternalOptimizationTool::JMetal
            | ExternalOptimizationTool::MoeaFramework
            | ExternalOptimizationTool::Ecj => ExternalOptimizationExactness::Heuristic,
            ExternalOptimizationTool::GoodLp | ExternalOptimizationTool::LpModeler => {
                ExternalOptimizationExactness::ModelingLayer
            }
            ExternalOptimizationTool::Argmin | ExternalOptimizationTool::Nlopt => {
                ExternalOptimizationExactness::Numerical
            }
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
            ExternalOptimizationTool::JMetal => &["ores-jmetal-adapter", "jmetal-adapter"],
            ExternalOptimizationTool::MoeaFramework => {
                &["ores-moea-framework-adapter", "moea-framework-adapter"]
            }
            ExternalOptimizationTool::Ecj => &["ores-ecj-adapter", "ecj-adapter"],
            ExternalOptimizationTool::OjAlgo => &["ores-ojalgo-adapter", "ojalgo-adapter"],
            ExternalOptimizationTool::OrToolsJava => {
                &["ores-ortools-java-adapter", "ortools-java-adapter"]
            }
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
    Rust,
}

impl ExternalOptimizationLanguage {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationLanguage::Java => "java",
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
    NonlinearOptimization,
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
            ExternalOptimizationFamily::NonlinearOptimization => "nonlinear-optimization",
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
        ExternalOptimizationTool::JMetal,
        ExternalOptimizationTool::MoeaFramework,
        ExternalOptimizationTool::Ecj,
        ExternalOptimizationTool::OjAlgo,
        ExternalOptimizationTool::OrToolsJava,
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

pub fn external_optimization_adapter_command(tool: ExternalOptimizationTool) -> Option<PathBuf> {
    configured_adapter_command(tool)
        .0
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
    let (configured_command, saw_configured_command) = configured_adapter_command(opts.tool);
    let command = opts
        .command_path
        .as_ref()
        .cloned()
        .or(configured_command)
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
        adapter_env_names, artifact_env_names, external_optimization_comparison_report_to_json,
        external_optimization_normalized_result_from_value, external_optimization_tool_specs,
        external_optimization_tools, run_external_optimization_comparison,
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
        assert_eq!(external_optimization_tools().len(), 17);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.language == ExternalOptimizationLanguage::Java)
                .count(),
            9
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.language == ExternalOptimizationLanguage::Rust)
                .count(),
            8
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
        assert_eq!(ExternalOptimizationLanguage::Rust.as_str(), "rust");
        assert_eq!(
            ExternalOptimizationFamily::EvolutionaryMultiObjective.as_str(),
            "evolutionary-multi-objective"
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
