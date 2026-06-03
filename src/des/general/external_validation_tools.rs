//! Local adapter/probe surface for external model-validation, output-validation,
//! benchmark, proof-checking, and simulation engines.
//!
//! These tools are intentionally represented as local adapters: the crate knows
//! stable names, capabilities, command aliases, and environment-variable hooks,
//! but it does not vendor jars, native libraries, solver binaries, benchmark
//! corpora, or simulator installations.

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalValidationFamily {
    ConstraintModeling,
    SmtSolver,
    SatSolver,
    ProofChecker,
    FormalModelChecker,
    BenchmarkLibrary,
    NonlinearGlobalSolver,
    ConvexConicSolver,
    SimulationEngine,
    OutputDataValidator,
}

impl ExternalValidationFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalValidationFamily::ConstraintModeling => "constraint-modeling",
            ExternalValidationFamily::SmtSolver => "smt-solver",
            ExternalValidationFamily::SatSolver => "sat-solver",
            ExternalValidationFamily::ProofChecker => "proof-checker",
            ExternalValidationFamily::FormalModelChecker => "formal-model-checker",
            ExternalValidationFamily::BenchmarkLibrary => "benchmark-library",
            ExternalValidationFamily::NonlinearGlobalSolver => "nonlinear-global-solver",
            ExternalValidationFamily::ConvexConicSolver => "convex-conic-solver",
            ExternalValidationFamily::SimulationEngine => "simulation-engine",
            ExternalValidationFamily::OutputDataValidator => "output-data-validator",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalValidationRuntime {
    NativeCli,
    Java,
    Python,
    Rust,
    Dataset,
    GenericAdapter,
}

impl ExternalValidationRuntime {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalValidationRuntime::NativeCli => "native-cli",
            ExternalValidationRuntime::Java => "java",
            ExternalValidationRuntime::Python => "python",
            ExternalValidationRuntime::Rust => "rust",
            ExternalValidationRuntime::Dataset => "dataset",
            ExternalValidationRuntime::GenericAdapter => "generic-adapter",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalValidationCapability {
    CompileModel,
    SolveModel,
    CheckSolution,
    CheckSatisfiability,
    CheckProof,
    CheckInvariant,
    CheckReachability,
    RunBenchmark,
    RunSimulation,
    ValidateOutput,
}

impl ExternalValidationCapability {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalValidationCapability::CompileModel => "compile-model",
            ExternalValidationCapability::SolveModel => "solve-model",
            ExternalValidationCapability::CheckSolution => "check-solution",
            ExternalValidationCapability::CheckSatisfiability => "check-satisfiability",
            ExternalValidationCapability::CheckProof => "check-proof",
            ExternalValidationCapability::CheckInvariant => "check-invariant",
            ExternalValidationCapability::CheckReachability => "check-reachability",
            ExternalValidationCapability::RunBenchmark => "run-benchmark",
            ExternalValidationCapability::RunSimulation => "run-simulation",
            ExternalValidationCapability::ValidateOutput => "validate-output",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalValidationArtifactKind {
    None,
    JavaClasspath,
    PythonPackage,
    RustCrate,
    NativeInstallDir,
    BenchmarkDataDir,
    SchemaOrSpecPath,
}

impl ExternalValidationArtifactKind {
    pub fn env_suffix(self) -> Option<&'static str> {
        match self {
            ExternalValidationArtifactKind::None => None,
            ExternalValidationArtifactKind::JavaClasspath => Some("CLASSPATH"),
            ExternalValidationArtifactKind::PythonPackage => Some("PYTHON"),
            ExternalValidationArtifactKind::RustCrate => Some("CRATE"),
            ExternalValidationArtifactKind::NativeInstallDir => Some("DIR"),
            ExternalValidationArtifactKind::BenchmarkDataDir => Some("DATA_DIR"),
            ExternalValidationArtifactKind::SchemaOrSpecPath => Some("SPEC"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExternalValidationToolSpec {
    pub id: &'static str,
    pub display_name: &'static str,
    pub env_key: &'static str,
    pub family: ExternalValidationFamily,
    pub runtime: ExternalValidationRuntime,
    pub artifact_kind: ExternalValidationArtifactKind,
    pub command_aliases: &'static [&'static str],
    pub capabilities: &'static [ExternalValidationCapability],
    pub input_formats: &'static [&'static str],
    pub notes: &'static str,
}

const MINIZINC_CAPS: &[ExternalValidationCapability] = &[
    ExternalValidationCapability::CompileModel,
    ExternalValidationCapability::SolveModel,
    ExternalValidationCapability::CheckSolution,
];
const SMT_CAPS: &[ExternalValidationCapability] =
    &[ExternalValidationCapability::CheckSatisfiability];
const SAT_CAPS: &[ExternalValidationCapability] =
    &[ExternalValidationCapability::CheckSatisfiability];
const PROOF_CAPS: &[ExternalValidationCapability] = &[ExternalValidationCapability::CheckProof];
const MODEL_CHECK_CAPS: &[ExternalValidationCapability] = &[
    ExternalValidationCapability::CheckInvariant,
    ExternalValidationCapability::CheckReachability,
];
const PROGRAM_VERIFIER_CAPS: &[ExternalValidationCapability] = &[
    ExternalValidationCapability::CheckInvariant,
    ExternalValidationCapability::CheckReachability,
    ExternalValidationCapability::CheckProof,
];
const SECURITY_PROTOCOL_CAPS: &[ExternalValidationCapability] = &[
    ExternalValidationCapability::CheckInvariant,
    ExternalValidationCapability::CheckReachability,
];
const BENCHMARK_CAPS: &[ExternalValidationCapability] =
    &[ExternalValidationCapability::RunBenchmark];
const NONLINEAR_CAPS: &[ExternalValidationCapability] = &[ExternalValidationCapability::SolveModel];
const CONVEX_CONIC_CAPS: &[ExternalValidationCapability] =
    &[ExternalValidationCapability::SolveModel];
const SOLVE_AND_VALIDATE_CAPS: &[ExternalValidationCapability] = &[
    ExternalValidationCapability::SolveModel,
    ExternalValidationCapability::CheckSolution,
];
const PLAN_VALIDATOR_CAPS: &[ExternalValidationCapability] =
    &[ExternalValidationCapability::CheckSolution];
const SIMULATION_CAPS: &[ExternalValidationCapability] =
    &[ExternalValidationCapability::RunSimulation];
const OUTPUT_VALIDATOR_CAPS: &[ExternalValidationCapability] =
    &[ExternalValidationCapability::ValidateOutput];

const EMPTY_ALIASES: &[&str] = &[];
const MZN_FORMATS: &[&str] = &["mzn", "dzn", "fzn", "ozn"];
const CP_MODEL_FORMATS: &[&str] = &[
    "mzn",
    "dzn",
    "fzn",
    "essence",
    "essence-param",
    "eprime",
    "xcsp3",
    "xml",
    "py",
    "json",
];
const ASP_FORMATS: &[&str] = &["lp", "asp", "clingo", "dlv", "json"];
const ALGEBRAIC_MODEL_FORMATS: &[&str] =
    &["lp", "mps", "nl", "mod", "dat", "gms", "jl", "py", "json"];
const PDDL_FORMATS: &[&str] = &["pddl", "plan", "sas", "json"];
const SMT_FORMATS: &[&str] = &["smt2"];
const SAT_FORMATS: &[&str] = &["cnf", "wcnf", "opb"];
const PROOF_FORMATS: &[&str] = &["drat", "lrat", "grat", "frat", "opb", "pbp", "rup"];
const TLA_FORMATS: &[&str] = &["tla", "cfg"];
const ALLOY_FORMATS: &[&str] = &["als"];
const PROMELA_FORMATS: &[&str] = &["pml"];
const SMV_FORMATS: &[&str] = &["smv"];
const PRISM_FORMATS: &[&str] = &["prism", "pm", "tra", "lab"];
const UPPAAL_FORMATS: &[&str] = &["xml", "q"];
const BENCHMARK_FORMATS: &[&str] = &["mps", "lp", "nl", "osil", "json", "dzn"];
const MILP_FORMATS: &[&str] = &["mps", "lp", "osil", "json"];
const NLP_FORMATS: &[&str] = &["nl", "osil", "mod", "json"];
const CONIC_FORMATS: &[&str] = &["mps", "lp", "qps", "cone", "json", "yaml"];
const SDP_FORMATS: &[&str] = &["sdpa", "dat-s", "csdp", "json"];
const SOFTWARE_MODEL_FORMATS: &[&str] = &["c", "cpp", "h", "json"];
const PROGRAM_VERIFIER_FORMATS: &[&str] = &[
    "dfy", "why", "mlw", "bpl", "sil", "fst", "c", "cpp", "h", "rs", "java", "adb", "ads", "ll",
    "bc", "v", "thy", "lean", "pvs", "lisp", "json",
];
const SECURITY_PROTOCOL_FORMATS: &[&str] = &["spthy", "pv", "cv", "scyther", "vp", "sapic", "json"];
const REWRITE_MODEL_FORMATS: &[&str] = &["maude", "mcrl2", "json"];
const SIMPY_FORMATS: &[&str] = &["py", "json"];
const R_SIM_FORMATS: &[&str] = &["r", "json"];
const JAVA_SIM_FORMATS: &[&str] = &["jar", "xml", "json"];
const AGENT_SIM_FORMATS: &[&str] = &["py", "java", "nlogo", "xml", "json"];
const DISTRIBUTED_SIM_FORMATS: &[&str] = &["xml", "json", "cpp", "cc", "py"];
const PROCESS_SIM_FORMATS: &[&str] = &["json", "xml", "mo", "cape-open"];
const NETWORK_SIM_FORMATS: &[&str] = &["cc", "cpp", "ini", "ned", "xml", "json"];
const TRAFFIC_SIM_FORMATS: &[&str] = &["net.xml", "rou.xml", "xml", "json"];
const BUILDING_SIM_FORMATS: &[&str] = &["idf", "osm", "epw", "json"];
const MODELICA_FORMATS: &[&str] = &["mo", "mos", "fmu", "ssp", "json"];
const ROBOTICS_FORMATS: &[&str] = &["urdf", "sdf", "mjcf", "xml", "json"];
const MATLAB_SIM_FORMATS: &[&str] = &["slx", "mdl", "m", "json"];
const POWER_GRID_FORMATS: &[&str] = &["dss", "glm", "json", "csv", "xlsx"];
const BIO_SIM_FORMATS: &[&str] = &["sbml", "antimony", "cps", "json"];
const OUTPUT_FORMATS: &[&str] = &["json", "jsonschema", "csv", "parquet", "avro", "protobuf"];
const API_OUTPUT_FORMATS: &[&str] = &["openapi", "yaml", "json"];
const XML_OUTPUT_FORMATS: &[&str] = &["xml", "xsd", "rng", "sch"];
const CUE_OUTPUT_FORMATS: &[&str] = &["cue", "json", "yaml"];
const YAML_OUTPUT_FORMATS: &[&str] = &["yaml", "yml", "json"];
const GRAPHQL_OUTPUT_FORMATS: &[&str] = &["graphql", "gql", "json"];
const DBT_OUTPUT_FORMATS: &[&str] = &["sql", "yml", "yaml", "json"];
const PARQUET_OUTPUT_FORMATS: &[&str] = &["parquet", "arrow", "json", "csv"];

pub const EXTERNAL_VALIDATION_TOOLS: &[ExternalValidationToolSpec] = &[
    ExternalValidationToolSpec {
        id: "minizinc",
        display_name: "MiniZinc",
        env_key: "MINIZINC",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["minizinc"],
        capabilities: MINIZINC_CAPS,
        input_formats: MZN_FORMATS,
        notes: "MiniZinc/FlatZinc compiler, solver launcher, and solution-checker bridge",
    },
    ExternalValidationToolSpec {
        id: "flatzinc",
        display_name: "FlatZinc solver",
        env_key: "FLATZINC",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["fzn-gecode", "fzn-chuffed", "fzn-or-tools"],
        capabilities: &[ExternalValidationCapability::SolveModel],
        input_formats: &["fzn"],
        notes: "Generic FlatZinc solver executable adapter for compiled CP models",
    },
    ExternalValidationToolSpec {
        id: "minizinc-solution-checker",
        display_name: "MiniZinc solution checker",
        env_key: "MINIZINC_SOLUTION_CHECKER",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["minizinc"],
        capabilities: &[ExternalValidationCapability::CheckSolution],
        input_formats: MZN_FORMATS,
        notes: "MiniZinc checker-model path for independent solution validation",
    },
    ExternalValidationToolSpec {
        id: "choco-solver",
        display_name: "Choco Solver",
        env_key: "CHOCO_SOLVER",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-choco-solver-adapter", "choco-solver-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: CP_MODEL_FORMATS,
        notes: "Java-native finite-domain CP solver adapter for scheduling, timetabling, and combinatorial checks",
    },
    ExternalValidationToolSpec {
        id: "jacop",
        display_name: "JaCoP",
        env_key: "JACOP",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-jacop-adapter", "jacop-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: CP_MODEL_FORMATS,
        notes: "Java CP solver adapter for independent finite-domain and scheduling-model validation",
    },
    ExternalValidationToolSpec {
        id: "ibm-cp-optimizer",
        display_name: "IBM ILOG CP Optimizer",
        env_key: "IBM_CP_OPTIMIZER",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-ibm-cp-optimizer-adapter", "cpoptimizer-adapter", "cpoptimizer"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: CP_MODEL_FORMATS,
        notes: "Industrial CP Optimizer adapter hook for interval scheduling and CP solution cross-checks",
    },
    ExternalValidationToolSpec {
        id: "ortools-java",
        display_name: "Google OR-Tools Java",
        env_key: "ORTOOLS_JAVA",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-ortools-java-adapter", "ortools-java-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: CP_MODEL_FORMATS,
        notes: "Java OR-Tools CP-SAT, routing, and linear-solver adapter for independent JVM-side validation",
    },
    ExternalValidationToolSpec {
        id: "ojalgo",
        display_name: "ojAlgo",
        env_key: "OJALGO",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-ojalgo-adapter", "ojalgo-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Java numerical optimization and self-contained LP/MIP-style validation adapter",
    },
    ExternalValidationToolSpec {
        id: "optaplanner",
        display_name: "OptaPlanner",
        env_key: "OPTAPLANNER",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-optaplanner-adapter", "optaplanner-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: PDDL_FORMATS,
        notes: "Java planning/metaheuristic adapter for timetabling, routing, rostering, and plan-output validation",
    },
    ExternalValidationToolSpec {
        id: "timefold",
        display_name: "Timefold Solver",
        env_key: "TIMEFOLD",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-timefold-adapter", "timefold-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: PDDL_FORMATS,
        notes: "Open-source Java/Kotlin planning solver adapter for metaheuristic schedule cross-checks",
    },
    ExternalValidationToolSpec {
        id: "jmetal",
        display_name: "jMetal",
        env_key: "JMETAL",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-jmetal-adapter", "jmetal-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Java evolutionary multi-objective optimization adapter for Pareto-front validation",
    },
    ExternalValidationToolSpec {
        id: "moea-framework",
        display_name: "MOEA Framework",
        env_key: "MOEA_FRAMEWORK",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-moea-framework-adapter", "moea-framework-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Java multi-objective evolutionary optimizer adapter for Pareto-front and scalarization checks",
    },
    ExternalValidationToolSpec {
        id: "ecj",
        display_name: "ECJ",
        env_key: "ECJ",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ores-ecj-adapter", "ecj-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Java evolutionary computation framework adapter for heuristic-result validation",
    },
    ExternalValidationToolSpec {
        id: "good-lp",
        display_name: "good_lp",
        env_key: "GOOD_LP",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-good-lp-adapter", "good-lp-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Rust LP/MIP modeling layer adapter for cross-checking solver-backed linear models",
    },
    ExternalValidationToolSpec {
        id: "lp-modeler",
        display_name: "lp-modeler",
        env_key: "LP_MODELER",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-lp-modeler-adapter", "lp-modeler-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Rust linear-programming DSL adapter for independent LP/MIP model export validation",
    },
    ExternalValidationToolSpec {
        id: "rust-linprog",
        display_name: "rust-linprog",
        env_key: "RUST_LINPROG",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-rust-linprog-adapter", "rust-linprog-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Rust-first lightweight LP adapter for validation-scale simplex cross-checks",
    },
    ExternalValidationToolSpec {
        id: "argmin",
        display_name: "argmin",
        env_key: "ARGMIN",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-argmin-adapter", "argmin-adapter"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "Rust nonlinear optimization crate adapter for gradient and derivative-free validation runs",
    },
    ExternalValidationToolSpec {
        id: "nlopt-rs",
        display_name: "NLopt Rust bindings",
        env_key: "NLOPT_RS",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-nlopt-rs-adapter", "nlopt-rs-adapter", "nlopt-adapter"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "Rust bindings to NLopt nonlinear algorithms for local nonlinear model cross-checks",
    },
    ExternalValidationToolSpec {
        id: "gurobi-rust",
        display_name: "Gurobi Rust bindings",
        env_key: "GUROBI_RUST",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-gurobi-rust-adapter", "gurobi-rust-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Rust binding adapter for Gurobi Optimizer using local, non-vendored solver libraries",
    },
    ExternalValidationToolSpec {
        id: "cplex-rust",
        display_name: "CPLEX Rust bindings",
        env_key: "CPLEX_RUST",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-cplex-rust-adapter", "cplex-rust-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Rust binding adapter for IBM ILOG CPLEX using local, non-vendored solver libraries",
    },
    ExternalValidationToolSpec {
        id: "ipopt-rust",
        display_name: "Ipopt Rust bindings",
        env_key: "IPOPT_RUST",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-ipopt-rust-adapter", "ipopt-rust-adapter"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "Rust binding adapter for Ipopt nonlinear optimization using local native libraries",
    },
    ExternalValidationToolSpec {
        id: "highs-rust",
        display_name: "HiGHS Rust bindings",
        env_key: "HIGHS_RUST",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-highs-rust-adapter", "highs-rust-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Rust HiGHS binding adapter for LP/MIP/QP cross-checks without vendoring solver binaries",
    },
    ExternalValidationToolSpec {
        id: "scip-rust",
        display_name: "SCIP Rust bindings",
        env_key: "SCIP_RUST",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-scip-rust-adapter", "scip-rust-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Rust SCIP binding adapter for MIP/CP-stack validation using local SCIP installations",
    },
    ExternalValidationToolSpec {
        id: "cbc-rust",
        display_name: "CBC Rust bindings",
        env_key: "CBC_RUST",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-cbc-rust-adapter", "cbc-rust-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Rust COIN-OR CBC binding adapter for MIP solution and model validation",
    },
    ExternalValidationToolSpec {
        id: "gecode",
        display_name: "Gecode FlatZinc",
        env_key: "GECODE",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["fzn-gecode"],
        capabilities: &[ExternalValidationCapability::SolveModel],
        input_formats: &["fzn"],
        notes: "Gecode FlatZinc backend for CP cross-checks",
    },
    ExternalValidationToolSpec {
        id: "chuffed",
        display_name: "Chuffed FlatZinc",
        env_key: "CHUFFED",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["fzn-chuffed"],
        capabilities: &[ExternalValidationCapability::SolveModel],
        input_formats: &["fzn"],
        notes: "Lazy-clause-generation FlatZinc backend for CP cross-checks",
    },
    ExternalValidationToolSpec {
        id: "ortools-cp-sat",
        display_name: "Google OR-Tools CP-SAT FlatZinc",
        env_key: "ORTOOLS_CP_SAT",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["fzn-cp-sat"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MZN_FORMATS,
        notes: "Native OR-Tools CP-SAT FlatZinc backend, usually exposed by Homebrew or MiniZinc solver bundles",
    },
    ExternalValidationToolSpec {
        id: "cpmpy",
        display_name: "CPMpy",
        env_key: "CPMPY",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["cpmpy-adapter", "cpm-py-adapter"],
        capabilities: MINIZINC_CAPS,
        input_formats: CP_MODEL_FORMATS,
        notes: "Python CP modeling layer for solver-agnostic constraint model cross-checks",
    },
    ExternalValidationToolSpec {
        id: "pycsp3",
        display_name: "PyCSP3",
        env_key: "PYCSP3",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["pycsp3", "pycsp3-adapter"],
        capabilities: MINIZINC_CAPS,
        input_formats: CP_MODEL_FORMATS,
        notes: "Python/XCSP3 modeling layer for constraint-problem validation",
    },
    ExternalValidationToolSpec {
        id: "conjure",
        display_name: "Conjure",
        env_key: "CONJURE",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["conjure"],
        capabilities: MINIZINC_CAPS,
        input_formats: CP_MODEL_FORMATS,
        notes: "Essence constraint-modeling frontend for generating independent solver models",
    },
    ExternalValidationToolSpec {
        id: "savile-row",
        display_name: "Savile Row",
        env_key: "SAVILE_ROW",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["savilerow", "savile-row", "SavileRow"],
        capabilities: MINIZINC_CAPS,
        input_formats: CP_MODEL_FORMATS,
        notes: "Constraint-model reformulation tool for Essence Prime and SAT/SMT/MIP backends",
    },
    ExternalValidationToolSpec {
        id: "picat",
        display_name: "Picat",
        env_key: "PICAT",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["picat"],
        capabilities: MINIZINC_CAPS,
        input_formats: CP_MODEL_FORMATS,
        notes: "Logic-based CP/MIP/SAT programming language for independent model checks",
    },
    ExternalValidationToolSpec {
        id: "clingo",
        display_name: "clingo",
        env_key: "CLINGO",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["clingo"],
        capabilities: &[
            ExternalValidationCapability::CompileModel,
            ExternalValidationCapability::SolveModel,
            ExternalValidationCapability::CheckSatisfiability,
        ],
        input_formats: ASP_FORMATS,
        notes: "Answer-set programming solver for combinatorial logic-model cross-checks",
    },
    ExternalValidationToolSpec {
        id: "clingcon",
        display_name: "clingcon",
        env_key: "CLINGCON",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["clingcon"],
        capabilities: &[
            ExternalValidationCapability::CompileModel,
            ExternalValidationCapability::SolveModel,
            ExternalValidationCapability::CheckSatisfiability,
        ],
        input_formats: ASP_FORMATS,
        notes: "Answer-set and finite-domain constraint solver in the Potassco ecosystem",
    },
    ExternalValidationToolSpec {
        id: "pyomo",
        display_name: "Pyomo",
        env_key: "PYOMO",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["pyomo", "pyomo-adapter"],
        capabilities: MINIZINC_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Python algebraic modeling layer for LP/MIP/NLP validation adapters",
    },
    ExternalValidationToolSpec {
        id: "pulp",
        display_name: "PuLP",
        env_key: "PULP",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["pulp-adapter"],
        capabilities: MINIZINC_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Python LP/MIP modeling layer for independent model checks",
    },
    ExternalValidationToolSpec {
        id: "pyscipopt",
        display_name: "PySCIPOpt",
        env_key: "PYSCIPOPT",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["pyscipopt-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Python SCIP binding adapter for MIP, MINLP, solution-pool, and plugin-backed validation",
    },
    ExternalValidationToolSpec {
        id: "python-mip",
        display_name: "Python-MIP",
        env_key: "PYTHON_MIP",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["python-mip-adapter", "mip-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Python-MIP adapter for CBC/Gurobi-backed LP/MIP model and solution validation",
    },
    ExternalValidationToolSpec {
        id: "gurobipy",
        display_name: "gurobipy",
        env_key: "GUROBIPY",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["gurobipy-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Official Gurobi Python API adapter for model, attribute, parameter, and solution validation",
    },
    ExternalValidationToolSpec {
        id: "cplex-python",
        display_name: "IBM ILOG CPLEX Python API",
        env_key: "CPLEX_PYTHON",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["cplex-python-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Official CPLEX Python API adapter for model, parameter, and solution validation",
    },
    ExternalValidationToolSpec {
        id: "xpress-python",
        display_name: "FICO Xpress Python API",
        env_key: "XPRESS_PYTHON",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["xpress-python-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Official FICO Xpress Python API adapter for model, parameter, and solution validation",
    },
    ExternalValidationToolSpec {
        id: "docplex",
        display_name: "DOcplex",
        env_key: "DOCPLEX",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["docplex-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "IBM DOcplex adapter for CPLEX and CP Optimizer model validation",
    },
    ExternalValidationToolSpec {
        id: "ortools-python",
        display_name: "Google OR-Tools Python",
        env_key: "ORTOOLS_PYTHON",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["ortools-python-adapter", "ortools-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Python OR-Tools adapter for CP-SAT, routing, and linear-solver validation",
    },
    ExternalValidationToolSpec {
        id: "ortools-glop",
        display_name: "Google OR-Tools GLOP",
        env_key: "ORTOOLS_GLOP",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["ortools-glop-adapter", "glop-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: &["lp", "mps", "proto", "json"],
        notes: "OR-Tools GLOP linear-programming adapter for independent LP solution checks",
    },
    ExternalValidationToolSpec {
        id: "ortools-pdlp",
        display_name: "Google OR-Tools PDLP",
        env_key: "ORTOOLS_PDLP",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["ortools-pdlp-adapter", "pdlp-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: &["lp", "mps", "proto", "json"],
        notes: "OR-Tools PDLP first-order linear-programming adapter for large sparse LP validation",
    },
    ExternalValidationToolSpec {
        id: "scipy-optimize",
        display_name: "SciPy optimize",
        env_key: "SCIPY_OPTIMIZE",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["scipy-optimize-adapter", "scipy-adapter"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "SciPy optimize adapter for lightweight nonlinear and constrained numerical cross-checks",
    },
    ExternalValidationToolSpec {
        id: "highs-cli",
        display_name: "HiGHS CLI",
        env_key: "HIGHS_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["highs"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Native HiGHS command-line adapter for LP/MIP/QP model and solution cross-checks",
    },
    ExternalValidationToolSpec {
        id: "glpk-cli",
        display_name: "GLPK glpsol CLI",
        env_key: "GLPK_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["glpsol"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "GLPK glpsol command-line adapter for LP/MIP validation and model export checks",
    },
    ExternalValidationToolSpec {
        id: "scip-cli",
        display_name: "SCIP CLI",
        env_key: "SCIP_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["scip"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "SCIP command-line adapter for MIP, constraint-integer, and validation-scale model checks",
    },
    ExternalValidationToolSpec {
        id: "cbc-cli",
        display_name: "COIN-OR CBC CLI",
        env_key: "CBC_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["cbc"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "COIN-OR CBC command-line adapter for MIP solution and model validation",
    },
    ExternalValidationToolSpec {
        id: "clp-cli",
        display_name: "COIN-OR CLP CLI",
        env_key: "CLP_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["clp"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: &["lp", "mps", "json"],
        notes: "COIN-OR CLP command-line adapter for LP-only model and solution validation",
    },
    ExternalValidationToolSpec {
        id: "soplex-cli",
        display_name: "SoPlex CLI",
        env_key: "SOPLEX_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["soplex"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: &["lp", "mps", "json"],
        notes: "ZIB SoPlex command-line adapter for LP validation, including rational solve modes",
    },
    ExternalValidationToolSpec {
        id: "qsopt-ex-cli",
        display_name: "QSopt_ex CLI",
        env_key: "QSOPT_EX_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["qsopt_ex", "qsopt-ex", "qsopt", "esolver"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: &["lp", "mps", "json"],
        notes: "QSopt_ex exact rational LP solver adapter hook for independently validating LP optima",
    },
    ExternalValidationToolSpec {
        id: "lp-solve-cli",
        display_name: "lp_solve CLI",
        env_key: "LP_SOLVE_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["lp_solve", "lp-solve", "lpsolve"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: &["lp", "json"],
        notes: "lp_solve command-line adapter for lightweight LP/MIP validation cross-checks",
    },
    ExternalValidationToolSpec {
        id: "gurobi-cli",
        display_name: "Gurobi Optimizer CLI",
        env_key: "GUROBI_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["gurobi_cl"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Commercial Gurobi command-line adapter using local non-vendored executables",
    },
    ExternalValidationToolSpec {
        id: "cplex-cli",
        display_name: "IBM ILOG CPLEX CLI",
        env_key: "CPLEX_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["cplex"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Commercial IBM ILOG CPLEX command-line adapter using local installations",
    },
    ExternalValidationToolSpec {
        id: "xpress-cli",
        display_name: "FICO Xpress CLI",
        env_key: "XPRESS_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["optimizer", "xpress"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Commercial FICO Xpress command-line adapter using local installations",
    },
    ExternalValidationToolSpec {
        id: "lindo-cli",
        display_name: "LINDO Systems CLI",
        env_key: "LINDO_CLI",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["runlindo", "lindo", "lindoapi"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: MILP_FORMATS,
        notes: "Commercial LINDO Systems command-line adapter using local installations",
    },
    ExternalValidationToolSpec {
        id: "ampl",
        display_name: "AMPL",
        env_key: "AMPL",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["ampl"],
        capabilities: MINIZINC_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Algebraic modeling language and solver launcher for optimization model validation",
    },
    ExternalValidationToolSpec {
        id: "gams",
        display_name: "GAMS",
        env_key: "GAMS",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["gams"],
        capabilities: MINIZINC_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Algebraic modeling system for optimization model cross-checks",
    },
    ExternalValidationToolSpec {
        id: "hexaly",
        display_name: "Hexaly Optimizer",
        env_key: "HEXALY",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["hexaly", "localsolver", "localsolver-studio"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Hexaly/LocalSolver adapter hook for routing, scheduling, nonlinear, and CP-style validation",
    },
    ExternalValidationToolSpec {
        id: "jump",
        display_name: "JuMP",
        env_key: "JUMP",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["jump-adapter", "julia"],
        capabilities: MINIZINC_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Julia/JuMP adapter hook for solver-independent optimization model checks",
    },
    ExternalValidationToolSpec {
        id: "neos",
        display_name: "NEOS Server adapter",
        env_key: "NEOS",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["neos-adapter", "kestrel"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Local adapter hook for NEOS/Kestrel-style remote solver validation",
    },
    ExternalValidationToolSpec {
        id: "pddl-val",
        display_name: "VAL PDDL validator",
        env_key: "PDDL_VAL",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["Validate", "validate", "val"],
        capabilities: PLAN_VALIDATOR_CAPS,
        input_formats: PDDL_FORMATS,
        notes: "PDDL plan validator for independent plan/output validation",
    },
    ExternalValidationToolSpec {
        id: "fast-downward",
        display_name: "Fast Downward",
        env_key: "FAST_DOWNWARD",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["fast-downward.py", "fast-downward"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: PDDL_FORMATS,
        notes: "Classical planning solver for PDDL validation scenarios",
    },
    ExternalValidationToolSpec {
        id: "lpg-td",
        display_name: "LPG-td",
        env_key: "LPG_TD",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["lpg-td", "lpg"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: PDDL_FORMATS,
        notes: "Temporal planning solver for PDDL schedule/plan cross-checks",
    },
    ExternalValidationToolSpec {
        id: "optic",
        display_name: "OPTIC",
        env_key: "OPTIC",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["optic", "optic-clp"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: PDDL_FORMATS,
        notes: "Temporal/continuous-effects PDDL planner for plan validation scenarios",
    },
    ExternalValidationToolSpec {
        id: "enhsp",
        display_name: "ENHSP",
        env_key: "ENHSP",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["enhsp", "enhsp.jar"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: PDDL_FORMATS,
        notes: "Hybrid numeric PDDL planner adapter for planning-model validation",
    },
    ExternalValidationToolSpec {
        id: "z3",
        display_name: "Z3",
        env_key: "Z3",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["z3"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "SMT-LIB solver for arithmetic, arrays, bit-vectors, and unsat cores",
    },
    ExternalValidationToolSpec {
        id: "cvc5",
        display_name: "cvc5",
        env_key: "CVC5",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["cvc5"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "SMT/SyGuS solver for independent satisfiability and proof-oriented checks",
    },
    ExternalValidationToolSpec {
        id: "yices",
        display_name: "Yices",
        env_key: "YICES",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["yices-smt2", "yices"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "SMT-LIB solver for arithmetic and bit-vector validation",
    },
    ExternalValidationToolSpec {
        id: "bitwuzla",
        display_name: "Bitwuzla",
        env_key: "BITWUZLA",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["bitwuzla"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "SMT solver focused on bit-vectors, arrays, floating point, and strings",
    },
    ExternalValidationToolSpec {
        id: "boolector",
        display_name: "Boolector",
        env_key: "BOOLECTOR",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["boolector"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "SMT solver for bit-vector and array encodings",
    },
    ExternalValidationToolSpec {
        id: "mathsat",
        display_name: "MathSAT",
        env_key: "MATHSAT",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["mathsat"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "SMT solver for arithmetic, bit-vector, and array model validation",
    },
    ExternalValidationToolSpec {
        id: "optimathsat",
        display_name: "OptiMathSAT",
        env_key: "OPTIMATHSAT",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["optimathsat", "optimathsat5"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "Optimization-modulo-theories solver for SMT-LIB Optimize validation",
    },
    ExternalValidationToolSpec {
        id: "opensmt",
        display_name: "OpenSMT",
        env_key: "OPENSMT",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["opensmt", "opensmt2"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "Open-source SMT solver for independent SMT-LIB satisfiability checks",
    },
    ExternalValidationToolSpec {
        id: "smtinterpol",
        display_name: "SMTInterpol",
        env_key: "SMTINTERPOL",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["smtinterpol", "smtinterpol.sh"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "Interpolating SMT solver useful for proof-oriented model validation",
    },
    ExternalValidationToolSpec {
        id: "princess",
        display_name: "Princess",
        env_key: "PRINCESS",
        family: ExternalValidationFamily::SmtSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["princess", "princess-smt"],
        capabilities: SMT_CAPS,
        input_formats: SMT_FORMATS,
        notes: "SMT solver and theorem prover for integer arithmetic model checks",
    },
    ExternalValidationToolSpec {
        id: "kissat",
        display_name: "Kissat",
        env_key: "KISSAT",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["kissat"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "CDCL SAT solver for DIMACS CNF cross-checks",
    },
    ExternalValidationToolSpec {
        id: "cadical",
        display_name: "CaDiCaL",
        env_key: "CADICAL",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["cadical"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "SAT solver with proof-generation and checker ecosystem support",
    },
    ExternalValidationToolSpec {
        id: "cryptominisat",
        display_name: "CryptoMiniSat",
        env_key: "CRYPTOMINISAT",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["cryptominisat5", "cryptominisat"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "SAT solver for CNF/XOR-heavy Boolean encodings",
    },
    ExternalValidationToolSpec {
        id: "minisat",
        display_name: "MiniSat",
        env_key: "MINISAT",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["minisat"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "Classic CDCL SAT solver for DIMACS CNF smoke-model validation",
    },
    ExternalValidationToolSpec {
        id: "glucose",
        display_name: "Glucose",
        env_key: "GLUCOSE",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["glucose", "glucose-syrup"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "CDCL SAT solver family for independent DIMACS satisfiability checks",
    },
    ExternalValidationToolSpec {
        id: "maplesat",
        display_name: "MapleSAT",
        env_key: "MAPLESAT",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["maplesat", "maple-sat", "maple-lcm"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "Maple-family SAT solver for CDCL branching and restart cross-checks",
    },
    ExternalValidationToolSpec {
        id: "varisat",
        display_name: "Varisat",
        env_key: "VARISAT",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["varisat"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "Rust SAT solver and checker for DIMACS/LRAT-oriented validation",
    },
    ExternalValidationToolSpec {
        id: "sat4j",
        display_name: "SAT4J",
        env_key: "SAT4J",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["sat4j", "sat4j-sat"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "Java SAT and pseudo-Boolean solver library for JVM-side validation",
    },
    ExternalValidationToolSpec {
        id: "pysat",
        display_name: "PySAT",
        env_key: "PYSAT",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["pysat-adapter", "python-sat-adapter"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "Python SAT toolkit wrapping multiple SAT solvers and cardinality encodings",
    },
    ExternalValidationToolSpec {
        id: "open-wbo",
        display_name: "Open-WBO",
        env_key: "OPEN_WBO",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["open-wbo", "open-wbo_static"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "MaxSAT solver for weighted-CNF objective and feasibility cross-checks",
    },
    ExternalValidationToolSpec {
        id: "maxhs",
        display_name: "MaxHS",
        env_key: "MAXHS",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["maxhs"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "Core-guided MaxSAT solver for weighted-CNF objective validation",
    },
    ExternalValidationToolSpec {
        id: "roundingsat",
        display_name: "RoundingSat",
        env_key: "ROUNDINGSAT",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["roundingsat"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "Pseudo-Boolean optimizer and proof-producing SAT/PB solver for OPB checks",
    },
    ExternalValidationToolSpec {
        id: "drat-trim",
        display_name: "DRAT-trim",
        env_key: "DRAT_TRIM",
        family: ExternalValidationFamily::ProofChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["drat-trim"],
        capabilities: PROOF_CAPS,
        input_formats: PROOF_FORMATS,
        notes: "DRAT proof checker for validating UNSAT certificates",
    },
    ExternalValidationToolSpec {
        id: "lrat-check",
        display_name: "LRAT checker",
        env_key: "LRAT_CHECK",
        family: ExternalValidationFamily::ProofChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["lrat-check", "cake_lpr"],
        capabilities: PROOF_CAPS,
        input_formats: PROOF_FORMATS,
        notes: "LRAT/LPR proof checker for machine-checkable SAT certificates",
    },
    ExternalValidationToolSpec {
        id: "frat",
        display_name: "FRAT checker",
        env_key: "FRAT",
        family: ExternalValidationFamily::ProofChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["frat-rs", "frat-trim", "frat"],
        capabilities: PROOF_CAPS,
        input_formats: PROOF_FORMATS,
        notes: "FRAT proof checker for SAT/UNSAT certificate validation",
    },
    ExternalValidationToolSpec {
        id: "veripb",
        display_name: "VeriPB",
        env_key: "VERIPB",
        family: ExternalValidationFamily::ProofChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["veripb", "veripb-checker"],
        capabilities: PROOF_CAPS,
        input_formats: PROOF_FORMATS,
        notes: "Pseudo-Boolean proof checker for OPB/PBP/RUP optimization certificates",
    },
    ExternalValidationToolSpec {
        id: "tlc",
        display_name: "TLC",
        env_key: "TLC",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["tlc", "tlc2"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: TLA_FORMATS,
        notes: "Explicit-state model checker for TLA+ specifications",
    },
    ExternalValidationToolSpec {
        id: "apalache",
        display_name: "Apalache",
        env_key: "APALACHE",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["apalache-mc", "apalache"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: TLA_FORMATS,
        notes: "Symbolic model checker for bounded and inductive TLA+ checks",
    },
    ExternalValidationToolSpec {
        id: "alloy",
        display_name: "Alloy/Kodkod",
        env_key: "ALLOY",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["alloy"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: ALLOY_FORMATS,
        notes: "Relational model finder for Alloy specifications via Kodkod",
    },
    ExternalValidationToolSpec {
        id: "kodkod",
        display_name: "Kodkod",
        env_key: "KODKOD",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["kodkod"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: ALLOY_FORMATS,
        notes: "Standalone relational model finder backend for Alloy-style validation",
    },
    ExternalValidationToolSpec {
        id: "spin",
        display_name: "SPIN",
        env_key: "SPIN",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["spin"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: PROMELA_FORMATS,
        notes: "Promela model checker for asynchronous protocol validation",
    },
    ExternalValidationToolSpec {
        id: "nuxmv",
        display_name: "nuXmv",
        env_key: "NUXMV",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["nuXmv", "nuxmv"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: SMV_FORMATS,
        notes: "Symbolic model checker for finite/infinite-state transition systems",
    },
    ExternalValidationToolSpec {
        id: "prism",
        display_name: "PRISM",
        env_key: "PRISM",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["prism"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: PRISM_FORMATS,
        notes: "Probabilistic model checker for stochastic systems",
    },
    ExternalValidationToolSpec {
        id: "storm",
        display_name: "Storm",
        env_key: "STORM",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["storm"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: PRISM_FORMATS,
        notes: "Probabilistic model checker compatible with PRISM-style models",
    },
    ExternalValidationToolSpec {
        id: "uppaal",
        display_name: "UPPAAL",
        env_key: "UPPAAL",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["verifyta"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: UPPAAL_FORMATS,
        notes: "Timed-automata model checker for real-time systems",
    },
    ExternalValidationToolSpec {
        id: "cbmc",
        display_name: "CBMC",
        env_key: "CBMC",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["cbmc"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: SOFTWARE_MODEL_FORMATS,
        notes: "Bounded model checker for C/C++ reference implementations and generated code",
    },
    ExternalValidationToolSpec {
        id: "ebmc",
        display_name: "EBMC",
        env_key: "EBMC",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["ebmc"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "CBMC-family bounded model checker for hardware, C, and SystemC reference artifacts",
    },
    ExternalValidationToolSpec {
        id: "dafny",
        display_name: "Dafny",
        env_key: "DAFNY",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["dafny"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Verification-aware programming language and static verifier for contracts",
    },
    ExternalValidationToolSpec {
        id: "frama-c",
        display_name: "Frama-C/WP",
        env_key: "FRAMA_C",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["frama-c"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "C program analyzer with ACSL contracts and WP deductive verification",
    },
    ExternalValidationToolSpec {
        id: "why3",
        display_name: "Why3",
        env_key: "WHY3",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["why3"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Deductive program-verification platform and prover orchestrator",
    },
    ExternalValidationToolSpec {
        id: "kani",
        display_name: "Kani Rust Verifier",
        env_key: "KANI",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["kani"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Bounded model checker for Rust proof harnesses",
    },
    ExternalValidationToolSpec {
        id: "esbmc",
        display_name: "ESBMC",
        env_key: "ESBMC",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["esbmc"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "SMT-based bounded model checker for software reference implementations",
    },
    ExternalValidationToolSpec {
        id: "cpachecker",
        display_name: "CPAchecker",
        env_key: "CPACHECKER",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["cpachecker"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Configurable software-verification framework for C programs",
    },
    ExternalValidationToolSpec {
        id: "jbmc",
        display_name: "JBMC",
        env_key: "JBMC",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["jbmc"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Bounded model checker for Java bytecode and Java reference exports",
    },
    ExternalValidationToolSpec {
        id: "klee",
        display_name: "KLEE",
        env_key: "KLEE",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["klee"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "LLVM symbolic-execution engine for generated C/C++ and bitcode safety checks",
    },
    ExternalValidationToolSpec {
        id: "java-pathfinder",
        display_name: "Java Pathfinder",
        env_key: "JAVA_PATHFINDER",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["jpf", "jpf-core"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Java model checker for state-space and concurrency validation",
    },
    ExternalValidationToolSpec {
        id: "key",
        display_name: "KeY",
        env_key: "KEY",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["key", "key-cli"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Deductive Java verifier for contracts and symbolic execution checks",
    },
    ExternalValidationToolSpec {
        id: "boogie",
        display_name: "Boogie",
        env_key: "BOOGIE",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["boogie"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Intermediate verification-language checker used by Dafny and related frontends",
    },
    ExternalValidationToolSpec {
        id: "viper",
        display_name: "Viper",
        env_key: "VIPER",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["viper", "silicon", "carbon"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Permission-based verification infrastructure for heap-manipulating programs",
    },
    ExternalValidationToolSpec {
        id: "fstar",
        display_name: "F*",
        env_key: "FSTAR",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["fstar", "fstar.exe"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Dependent-type program verifier for proof-carrying reference artifacts",
    },
    ExternalValidationToolSpec {
        id: "gnatprove",
        display_name: "GNATprove",
        env_key: "GNATPROVE",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["gnatprove"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "SPARK/Ada proof tool for contract and absence-of-run-time-error validation",
    },
    ExternalValidationToolSpec {
        id: "seahorn",
        display_name: "SeaHorn",
        env_key: "SEAHORN",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["sea", "seahorn"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "LLVM-based software model checker for C/C++ safety properties",
    },
    ExternalValidationToolSpec {
        id: "smack",
        display_name: "SMACK",
        env_key: "SMACK",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["smack"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "LLVM-to-Boogie verifier bridge for program-level cross-checks",
    },
    ExternalValidationToolSpec {
        id: "ultimate-automizer",
        display_name: "Ultimate Automizer",
        env_key: "ULTIMATE_AUTOMIZER",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["Ultimate", "ultimate", "Ultimate.py"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Automata-based software verifier for C and concurrent program checks",
    },
    ExternalValidationToolSpec {
        id: "goblint",
        display_name: "Goblint",
        env_key: "GOBLINT",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["goblint"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Static analyzer and verifier for C concurrency and race properties",
    },
    ExternalValidationToolSpec {
        id: "prusti",
        display_name: "Prusti",
        env_key: "PRUSTI",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["prusti-rustc", "cargo-prusti"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Rust verifier for contracts and borrow-aware program properties",
    },
    ExternalValidationToolSpec {
        id: "mirai",
        display_name: "MIRAI",
        env_key: "MIRAI",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["cargo-mirai", "mirai"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Rust abstract interpreter for contracts, panics, and safety invariants",
    },
    ExternalValidationToolSpec {
        id: "creusot",
        display_name: "Creusot",
        env_key: "CREUSOT",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["cargo-creusot", "creusot"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Rust-to-Why3 deductive verifier for functional correctness checks",
    },
    ExternalValidationToolSpec {
        id: "coq",
        display_name: "Coq",
        env_key: "COQ",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["coqc"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Interactive proof assistant for independently checked proof artifacts",
    },
    ExternalValidationToolSpec {
        id: "isabelle",
        display_name: "Isabelle",
        env_key: "ISABELLE",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["isabelle"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Interactive theorem prover for proof and model-validation artifacts",
    },
    ExternalValidationToolSpec {
        id: "lean",
        display_name: "Lean",
        env_key: "LEAN",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["lean"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Interactive theorem prover for independently checked proof artifacts",
    },
    ExternalValidationToolSpec {
        id: "pvs",
        display_name: "PVS",
        env_key: "PVS",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["pvs"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Prototype Verification System theorem prover for specification proofs",
    },
    ExternalValidationToolSpec {
        id: "acl2",
        display_name: "ACL2",
        env_key: "ACL2",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["acl2"],
        capabilities: PROGRAM_VERIFIER_CAPS,
        input_formats: PROGRAM_VERIFIER_FORMATS,
        notes: "Automated theorem prover and applicative Common Lisp logic",
    },
    ExternalValidationToolSpec {
        id: "tamarin",
        display_name: "Tamarin Prover",
        env_key: "TAMARIN",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["tamarin-prover"],
        capabilities: SECURITY_PROTOCOL_CAPS,
        input_formats: SECURITY_PROTOCOL_FORMATS,
        notes: "Security-protocol verifier for symbolic attack finding and proofs",
    },
    ExternalValidationToolSpec {
        id: "proverif",
        display_name: "ProVerif",
        env_key: "PROVERIF",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["proverif"],
        capabilities: SECURITY_PROTOCOL_CAPS,
        input_formats: SECURITY_PROTOCOL_FORMATS,
        notes: "Automatic cryptographic-protocol verifier in the symbolic model",
    },
    ExternalValidationToolSpec {
        id: "cryptoverif",
        display_name: "CryptoVerif",
        env_key: "CRYPTOVERIF",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["cryptoverif"],
        capabilities: SECURITY_PROTOCOL_CAPS,
        input_formats: SECURITY_PROTOCOL_FORMATS,
        notes: "Computational security-protocol verifier",
    },
    ExternalValidationToolSpec {
        id: "deepsec",
        display_name: "DeepSec",
        env_key: "DEEPSEC",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["deepsec"],
        capabilities: SECURITY_PROTOCOL_CAPS,
        input_formats: SECURITY_PROTOCOL_FORMATS,
        notes: "Security-protocol equivalence and reachability checker",
    },
    ExternalValidationToolSpec {
        id: "scyther",
        display_name: "Scyther",
        env_key: "SCYTHER",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["scyther"],
        capabilities: SECURITY_PROTOCOL_CAPS,
        input_formats: SECURITY_PROTOCOL_FORMATS,
        notes: "Security-protocol model checker for role/claim specifications",
    },
    ExternalValidationToolSpec {
        id: "verifpal",
        display_name: "Verifpal",
        env_key: "VERIFPAL",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["verifpal"],
        capabilities: SECURITY_PROTOCOL_CAPS,
        input_formats: SECURITY_PROTOCOL_FORMATS,
        notes: "Human-oriented symbolic cryptographic-protocol verifier",
    },
    ExternalValidationToolSpec {
        id: "sapic-plus",
        display_name: "SAPIC+",
        env_key: "SAPIC_PLUS",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["sapic", "sapic-plus"],
        capabilities: SECURITY_PROTOCOL_CAPS,
        input_formats: SECURITY_PROTOCOL_FORMATS,
        notes: "Security-protocol frontend with exports to ProVerif, Tamarin, and DeepSec",
    },
    ExternalValidationToolSpec {
        id: "mcrl2",
        display_name: "mCRL2",
        env_key: "MCRL2",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["mcrl22lps", "lps2pbes", "pbes2bool"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: REWRITE_MODEL_FORMATS,
        notes: "Process-algebra toolset for transition-system and modal-mu-calculus checks",
    },
    ExternalValidationToolSpec {
        id: "maude",
        display_name: "Maude",
        env_key: "MAUDE",
        family: ExternalValidationFamily::FormalModelChecker,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["maude"],
        capabilities: MODEL_CHECK_CAPS,
        input_formats: REWRITE_MODEL_FORMATS,
        notes: "Rewriting-logic engine with search and LTL model-checking support",
    },
    ExternalValidationToolSpec {
        id: "miplib",
        display_name: "MIPLIB",
        env_key: "MIPLIB",
        family: ExternalValidationFamily::BenchmarkLibrary,
        runtime: ExternalValidationRuntime::Dataset,
        artifact_kind: ExternalValidationArtifactKind::BenchmarkDataDir,
        command_aliases: EMPTY_ALIASES,
        capabilities: BENCHMARK_CAPS,
        input_formats: BENCHMARK_FORMATS,
        notes: "Mixed-integer programming benchmark corpus",
    },
    ExternalValidationToolSpec {
        id: "qplib",
        display_name: "QPLIB",
        env_key: "QPLIB",
        family: ExternalValidationFamily::BenchmarkLibrary,
        runtime: ExternalValidationRuntime::Dataset,
        artifact_kind: ExternalValidationArtifactKind::BenchmarkDataDir,
        command_aliases: EMPTY_ALIASES,
        capabilities: BENCHMARK_CAPS,
        input_formats: BENCHMARK_FORMATS,
        notes: "Quadratic programming benchmark corpus",
    },
    ExternalValidationToolSpec {
        id: "minlplib",
        display_name: "MINLPLib",
        env_key: "MINLPLIB",
        family: ExternalValidationFamily::BenchmarkLibrary,
        runtime: ExternalValidationRuntime::Dataset,
        artifact_kind: ExternalValidationArtifactKind::BenchmarkDataDir,
        command_aliases: EMPTY_ALIASES,
        capabilities: BENCHMARK_CAPS,
        input_formats: BENCHMARK_FORMATS,
        notes: "Continuous and mixed-integer nonlinear programming benchmark corpus",
    },
    ExternalValidationToolSpec {
        id: "netlib-lp",
        display_name: "Netlib LP",
        env_key: "NETLIB_LP",
        family: ExternalValidationFamily::BenchmarkLibrary,
        runtime: ExternalValidationRuntime::Dataset,
        artifact_kind: ExternalValidationArtifactKind::BenchmarkDataDir,
        command_aliases: EMPTY_ALIASES,
        capabilities: BENCHMARK_CAPS,
        input_formats: BENCHMARK_FORMATS,
        notes: "Classic linear programming benchmark corpus",
    },
    ExternalValidationToolSpec {
        id: "csplib",
        display_name: "CSPLib",
        env_key: "CSPLIB",
        family: ExternalValidationFamily::BenchmarkLibrary,
        runtime: ExternalValidationRuntime::Dataset,
        artifact_kind: ExternalValidationArtifactKind::BenchmarkDataDir,
        command_aliases: EMPTY_ALIASES,
        capabilities: BENCHMARK_CAPS,
        input_formats: BENCHMARK_FORMATS,
        notes: "Constraint-satisfaction benchmark problem library",
    },
    ExternalValidationToolSpec {
        id: "or-library",
        display_name: "OR-Library",
        env_key: "OR_LIBRARY",
        family: ExternalValidationFamily::BenchmarkLibrary,
        runtime: ExternalValidationRuntime::Dataset,
        artifact_kind: ExternalValidationArtifactKind::BenchmarkDataDir,
        command_aliases: EMPTY_ALIASES,
        capabilities: BENCHMARK_CAPS,
        input_formats: BENCHMARK_FORMATS,
        notes: "Operations-research benchmark instance collection",
    },
    ExternalValidationToolSpec {
        id: "tsplib",
        display_name: "TSPLIB",
        env_key: "TSPLIB",
        family: ExternalValidationFamily::BenchmarkLibrary,
        runtime: ExternalValidationRuntime::Dataset,
        artifact_kind: ExternalValidationArtifactKind::BenchmarkDataDir,
        command_aliases: EMPTY_ALIASES,
        capabilities: BENCHMARK_CAPS,
        input_formats: BENCHMARK_FORMATS,
        notes: "Traveling-salesperson benchmark instance library",
    },
    ExternalValidationToolSpec {
        id: "vrplib",
        display_name: "VRPLIB",
        env_key: "VRPLIB",
        family: ExternalValidationFamily::BenchmarkLibrary,
        runtime: ExternalValidationRuntime::Dataset,
        artifact_kind: ExternalValidationArtifactKind::BenchmarkDataDir,
        command_aliases: EMPTY_ALIASES,
        capabilities: BENCHMARK_CAPS,
        input_formats: BENCHMARK_FORMATS,
        notes: "Vehicle-routing benchmark instance library",
    },
    ExternalValidationToolSpec {
        id: "minizinc-challenge",
        display_name: "MiniZinc Challenge",
        env_key: "MINIZINC_CHALLENGE",
        family: ExternalValidationFamily::BenchmarkLibrary,
        runtime: ExternalValidationRuntime::Dataset,
        artifact_kind: ExternalValidationArtifactKind::BenchmarkDataDir,
        command_aliases: EMPTY_ALIASES,
        capabilities: BENCHMARK_CAPS,
        input_formats: MZN_FORMATS,
        notes: "MiniZinc Challenge model/data/checker benchmark archive",
    },
    ExternalValidationToolSpec {
        id: "ipopt",
        display_name: "Ipopt",
        env_key: "IPOPT",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["ipopt"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "Interior-point nonlinear programming solver for NLP validation",
    },
    ExternalValidationToolSpec {
        id: "bonmin",
        display_name: "Bonmin",
        env_key: "BONMIN",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["bonmin"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "COIN-OR mixed-integer nonlinear programming solver",
    },
    ExternalValidationToolSpec {
        id: "minotaur",
        display_name: "MINOTAUR",
        env_key: "MINOTAUR",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["minotaur"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "Open-source mixed-integer nonlinear optimization toolkit",
    },
    ExternalValidationToolSpec {
        id: "couenne",
        display_name: "Couenne",
        env_key: "COUENNE",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["couenne"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "COIN-OR global optimization solver for nonconvex MINLP",
    },
    ExternalValidationToolSpec {
        id: "symphony",
        display_name: "COIN-OR SYMPHONY",
        env_key: "SYMPHONY",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["symphony"],
        capabilities: NONLINEAR_CAPS,
        input_formats: MILP_FORMATS,
        notes: "COIN-OR MILP solver and branch-and-cut framework",
    },
    ExternalValidationToolSpec {
        id: "knitro",
        display_name: "Artelys Knitro",
        env_key: "KNITRO",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["knitro", "knitroampl"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "Commercial nonlinear and mixed-integer nonlinear optimization solver",
    },
    ExternalValidationToolSpec {
        id: "mosek",
        display_name: "MOSEK",
        env_key: "MOSEK",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["mosek"],
        capabilities: NONLINEAR_CAPS,
        input_formats: &["mps", "ptf", "opf", "task", "json"],
        notes: "Commercial conic, quadratic, and nonlinear optimization solver",
    },
    ExternalValidationToolSpec {
        id: "baron",
        display_name: "BARON",
        env_key: "BARON",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["baron"],
        capabilities: NONLINEAR_CAPS,
        input_formats: NLP_FORMATS,
        notes: "Commercial global optimization solver for nonlinear models",
    },
    ExternalValidationToolSpec {
        id: "copt",
        display_name: "COPT",
        env_key: "COPT",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["copt_cmd", "copt"],
        capabilities: NONLINEAR_CAPS,
        input_formats: &["mps", "lp", "json"],
        notes: "Commercial LP/QP/QCP/MIP solver CLI for independent checks",
    },
    ExternalValidationToolSpec {
        id: "nlopt",
        display_name: "NLopt",
        env_key: "NLOPT",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["ores-nlopt-adapter", "nlopt-adapter"],
        capabilities: NONLINEAR_CAPS,
        input_formats: &["json", "nl"],
        notes: "NLopt derivative-free and gradient nonlinear optimization adapter using local installations",
    },
    ExternalValidationToolSpec {
        id: "nlopt-cli",
        display_name: "NLopt CLI",
        env_key: "NLOPT_CLI",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["nlopt-adapter"],
        capabilities: NONLINEAR_CAPS,
        input_formats: &["json"],
        notes: "Local adapter around NLopt derivative-free and gradient algorithms",
    },
    ExternalValidationToolSpec {
        id: "casadi",
        display_name: "CasADi",
        env_key: "CASADI",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["casadi-adapter"],
        capabilities: NONLINEAR_CAPS,
        input_formats: &["py", "json"],
        notes: "CasADi symbolic/numeric optimization adapter for NLP validation",
    },
    ExternalValidationToolSpec {
        id: "osqp",
        display_name: "OSQP",
        env_key: "OSQP",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["osqp-adapter", "osqp"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: CONIC_FORMATS,
        notes: "Operator-splitting QP solver for convex quadratic-program cross-checks",
    },
    ExternalValidationToolSpec {
        id: "scs",
        display_name: "SCS",
        env_key: "SCS",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["scs-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: CONIC_FORMATS,
        notes: "Splitting conic solver adapter for cone-program validation",
    },
    ExternalValidationToolSpec {
        id: "clarabel",
        display_name: "Clarabel",
        env_key: "CLARABEL",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["clarabel-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: CONIC_FORMATS,
        notes: "Interior-point conic solver with quadratic objective support",
    },
    ExternalValidationToolSpec {
        id: "ecos",
        display_name: "ECOS",
        env_key: "ECOS",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["ecos-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: CONIC_FORMATS,
        notes: "Embedded conic solver adapter for SOCP-style reference checks",
    },
    ExternalValidationToolSpec {
        id: "qpoases",
        display_name: "qpOASES",
        env_key: "QPOASES",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["qpoases-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: &["qps", "json"],
        notes: "Active-set quadratic-programming adapter for small and medium QP checks",
    },
    ExternalValidationToolSpec {
        id: "proxqp",
        display_name: "ProxQP",
        env_key: "PROXQP",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["proxqp-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: &["qps", "json"],
        notes: "Proximal QP solver adapter for numerical QP cross-checks",
    },
    ExternalValidationToolSpec {
        id: "cosmo",
        display_name: "COSMO",
        env_key: "COSMO",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["cosmo-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: CONIC_FORMATS,
        notes: "Conic-splitting solver adapter for cone-program validation",
    },
    ExternalValidationToolSpec {
        id: "sdpa",
        display_name: "SDPA",
        env_key: "SDPA",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["sdpa", "sdpa_gmp", "sdpa_dd"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: SDP_FORMATS,
        notes: "Semidefinite-programming solver family using SDPA-format models",
    },
    ExternalValidationToolSpec {
        id: "csdp",
        display_name: "CSDP",
        env_key: "CSDP",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["csdp"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: SDP_FORMATS,
        notes: "C implementation of a semidefinite-programming interior-point solver",
    },
    ExternalValidationToolSpec {
        id: "cvxpy",
        display_name: "CVXPY",
        env_key: "CVXPY",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["cvxpy-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: &["py", "json"],
        notes: "Python convex modeling layer used to dispatch OSQP/SCS/Clarabel/ECOS checks",
    },
    ExternalValidationToolSpec {
        id: "cvxopt",
        display_name: "CVXOPT",
        env_key: "CVXOPT",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["cvxopt-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: &["py", "json", "mps", "cone"],
        notes: "Python convex/numerical optimization package for LP, QP, and cone-program validation",
    },
    ExternalValidationToolSpec {
        id: "simpy",
        display_name: "SimPy",
        env_key: "SIMPY",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["simpy-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: SIMPY_FORMATS,
        notes: "Python process-based discrete-event simulation cross-check engine",
    },
    ExternalValidationToolSpec {
        id: "salabim",
        display_name: "salabim",
        env_key: "SALABIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["salabim-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: SIMPY_FORMATS,
        notes: "Python discrete-event simulation package",
    },
    ExternalValidationToolSpec {
        id: "ciw",
        display_name: "Ciw",
        env_key: "CIW",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["ciw-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: SIMPY_FORMATS,
        notes: "Python queueing-network and discrete-event simulation package",
    },
    ExternalValidationToolSpec {
        id: "simulus",
        display_name: "simulus",
        env_key: "SIMULUS",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["simulus-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: SIMPY_FORMATS,
        notes: "Python discrete-event simulation engine with process and event APIs",
    },
    ExternalValidationToolSpec {
        id: "simmer",
        display_name: "simmer",
        env_key: "SIMMER",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["simmer-adapter", "Rscript"],
        capabilities: SIMULATION_CAPS,
        input_formats: R_SIM_FORMATS,
        notes: "R discrete-event simulation adapter",
    },
    ExternalValidationToolSpec {
        id: "jaamsim",
        display_name: "JaamSim",
        env_key: "JAAMSIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["jaamsim"],
        capabilities: SIMULATION_CAPS,
        input_formats: JAVA_SIM_FORMATS,
        notes: "Java discrete-event simulation engine",
    },
    ExternalValidationToolSpec {
        id: "desmo-j",
        display_name: "DESMO-J",
        env_key: "DESMO_J",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["desmoj-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: JAVA_SIM_FORMATS,
        notes: "Java object-oriented discrete-event simulation framework",
    },
    ExternalValidationToolSpec {
        id: "simsharp",
        display_name: "SimSharp",
        env_key: "SIMSHARP",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["simsharp-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: JAVA_SIM_FORMATS,
        notes: ".NET discrete-event simulation adapter for cross-language queueing checks",
    },
    ExternalValidationToolSpec {
        id: "ns3",
        display_name: "ns-3",
        env_key: "NS3",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["ns3", "waf"],
        capabilities: SIMULATION_CAPS,
        input_formats: NETWORK_SIM_FORMATS,
        notes: "Discrete-event network simulator for internet systems",
    },
    ExternalValidationToolSpec {
        id: "omnetpp",
        display_name: "OMNeT++",
        env_key: "OMNETPP",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["opp_run", "omnetpp"],
        capabilities: SIMULATION_CAPS,
        input_formats: NETWORK_SIM_FORMATS,
        notes: "Component-based network and distributed-system simulation framework",
    },
    ExternalValidationToolSpec {
        id: "sumo",
        display_name: "SUMO",
        env_key: "SUMO",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["sumo", "sumo-gui"],
        capabilities: SIMULATION_CAPS,
        input_formats: TRAFFIC_SIM_FORMATS,
        notes: "Road traffic simulation engine for transport model validation",
    },
    ExternalValidationToolSpec {
        id: "matsim",
        display_name: "MATSim",
        env_key: "MATSIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["matsim"],
        capabilities: SIMULATION_CAPS,
        input_formats: TRAFFIC_SIM_FORMATS,
        notes: "Agent-based transport simulation framework",
    },
    ExternalValidationToolSpec {
        id: "energyplus",
        display_name: "EnergyPlus",
        env_key: "ENERGYPLUS",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["energyplus"],
        capabilities: SIMULATION_CAPS,
        input_formats: BUILDING_SIM_FORMATS,
        notes: "Whole-building energy simulation engine",
    },
    ExternalValidationToolSpec {
        id: "openstudio",
        display_name: "OpenStudio",
        env_key: "OPENSTUDIO",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["openstudio"],
        capabilities: SIMULATION_CAPS,
        input_formats: BUILDING_SIM_FORMATS,
        notes: "OpenStudio building-energy workflow and EnergyPlus wrapper",
    },
    ExternalValidationToolSpec {
        id: "openmodelica",
        display_name: "OpenModelica",
        env_key: "OPENMODELICA",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["omc"],
        capabilities: SIMULATION_CAPS,
        input_formats: MODELICA_FORMATS,
        notes: "Modelica compiler/simulator for hybrid physical models",
    },
    ExternalValidationToolSpec {
        id: "fmi-fmu",
        display_name: "FMI/FMU",
        env_key: "FMI_FMU",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["fmpy", "fmucheck", "fmu-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: &["fmu", "json"],
        notes: "Functional Mock-up Unit import/export and co-simulation adapter",
    },
    ExternalValidationToolSpec {
        id: "omsimulator",
        display_name: "OMSimulator",
        env_key: "OMSIMULATOR",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["OMSimulator", "omsimulator"],
        capabilities: SIMULATION_CAPS,
        input_formats: MODELICA_FORMATS,
        notes: "OpenModelica co-simulation/master simulation tool",
    },
    ExternalValidationToolSpec {
        id: "simulink",
        display_name: "Simulink/SimEvents",
        env_key: "SIMULINK",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["matlab", "simulink-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: MATLAB_SIM_FORMATS,
        notes: "MATLAB Simulink/SimEvents adapter for hybrid and discrete-event models",
    },
    ExternalValidationToolSpec {
        id: "ptolemy-ii",
        display_name: "Ptolemy II",
        env_key: "PTOLEMY_II",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["ptolemy", "vergil", "ptolemy-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: JAVA_SIM_FORMATS,
        notes: "Heterogeneous actor-model simulation framework for hybrid/discrete systems",
    },
    ExternalValidationToolSpec {
        id: "gem5",
        display_name: "gem5",
        env_key: "GEM5",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["gem5.opt", "gem5"],
        capabilities: SIMULATION_CAPS,
        input_formats: DISTRIBUTED_SIM_FORMATS,
        notes: "Computer-system simulator for architecture, memory, and scheduling validation",
    },
    ExternalValidationToolSpec {
        id: "gridlabd",
        display_name: "GridLAB-D",
        env_key: "GRIDLABD",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["gridlabd"],
        capabilities: SIMULATION_CAPS,
        input_formats: POWER_GRID_FORMATS,
        notes: "Power-distribution and transactive-energy simulation engine",
    },
    ExternalValidationToolSpec {
        id: "opendss",
        display_name: "OpenDSS",
        env_key: "OPENDSS",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["OpenDSSCmd", "opendsscmd", "dss"],
        capabilities: SIMULATION_CAPS,
        input_formats: POWER_GRID_FORMATS,
        notes: "Electric-power distribution system simulator for grid validation",
    },
    ExternalValidationToolSpec {
        id: "pandapower",
        display_name: "pandapower",
        env_key: "PANDAPOWER",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["pandapower-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: POWER_GRID_FORMATS,
        notes: "Python power-system analysis and simulation adapter",
    },
    ExternalValidationToolSpec {
        id: "copasi",
        display_name: "COPASI",
        env_key: "COPASI",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["CopasiSE", "copasi"],
        capabilities: SIMULATION_CAPS,
        input_formats: BIO_SIM_FORMATS,
        notes: "Biochemical network simulation and SBML validation adapter",
    },
    ExternalValidationToolSpec {
        id: "tellurium",
        display_name: "Tellurium",
        env_key: "TELLURIUM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["tellurium-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: BIO_SIM_FORMATS,
        notes: "Python systems-biology simulation adapter for Antimony/SBML models",
    },
    ExternalValidationToolSpec {
        id: "gazebo",
        display_name: "Gazebo",
        env_key: "GAZEBO",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["gz", "gazebo"],
        capabilities: SIMULATION_CAPS,
        input_formats: ROBOTICS_FORMATS,
        notes: "Robotics and physics simulation engine",
    },
    ExternalValidationToolSpec {
        id: "webots",
        display_name: "Webots",
        env_key: "WEBOTS",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["webots"],
        capabilities: SIMULATION_CAPS,
        input_formats: ROBOTICS_FORMATS,
        notes: "Robot simulator for controller and physical-world validation",
    },
    ExternalValidationToolSpec {
        id: "mujoco",
        display_name: "MuJoCo",
        env_key: "MUJOCO",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["mujoco-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: ROBOTICS_FORMATS,
        notes: "Physics simulator for robot/control model validation",
    },
    ExternalValidationToolSpec {
        id: "drake",
        display_name: "Drake",
        env_key: "DRAKE",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["drake-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: ROBOTICS_FORMATS,
        notes: "Model-based design, optimization, and simulation toolkit",
    },
    ExternalValidationToolSpec {
        id: "pybullet",
        display_name: "PyBullet",
        env_key: "PYBULLET",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["pybullet-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: ROBOTICS_FORMATS,
        notes: "Python robotics and rigid-body simulation adapter",
    },
    ExternalValidationToolSpec {
        id: "carla",
        display_name: "CARLA",
        env_key: "CARLA",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["carla", "CarlaUE4", "CarlaUE4.sh"],
        capabilities: SIMULATION_CAPS,
        input_formats: ROBOTICS_FORMATS,
        notes: "Autonomous-driving simulator for perception, routing, and control validation",
    },
    ExternalValidationToolSpec {
        id: "isaac-sim",
        display_name: "NVIDIA Isaac Sim",
        env_key: "ISAAC_SIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["isaac-sim", "isaacsim"],
        capabilities: SIMULATION_CAPS,
        input_formats: ROBOTICS_FORMATS,
        notes: "Robotics and synthetic-data simulation adapter for physical validation",
    },
    ExternalValidationToolSpec {
        id: "airsim",
        display_name: "AirSim",
        env_key: "AIRSIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["airsim-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: ROBOTICS_FORMATS,
        notes: "Vehicle and drone simulator adapter for autonomy model validation",
    },
    ExternalValidationToolSpec {
        id: "mesa",
        display_name: "Mesa",
        env_key: "MESA",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["mesa-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: AGENT_SIM_FORMATS,
        notes: "Python agent-based modeling framework",
    },
    ExternalValidationToolSpec {
        id: "agentpy",
        display_name: "AgentPy",
        env_key: "AGENTPY",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["agentpy-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: AGENT_SIM_FORMATS,
        notes: "Python agent-based simulation toolkit for stochastic ABM validation",
    },
    ExternalValidationToolSpec {
        id: "repast",
        display_name: "Repast",
        env_key: "REPAST",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["repast-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: AGENT_SIM_FORMATS,
        notes: "Agent-based modeling and simulation toolkit family",
    },
    ExternalValidationToolSpec {
        id: "mason",
        display_name: "MASON",
        env_key: "MASON",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["mason-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: AGENT_SIM_FORMATS,
        notes: "Java multi-agent simulation toolkit with DES support",
    },
    ExternalValidationToolSpec {
        id: "netlogo",
        display_name: "NetLogo",
        env_key: "NETLOGO",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["netlogo-headless.sh", "netlogo-headless"],
        capabilities: SIMULATION_CAPS,
        input_formats: AGENT_SIM_FORMATS,
        notes: "Agent-based modeling platform for complex-system validation",
    },
    ExternalValidationToolSpec {
        id: "simgrid",
        display_name: "SimGrid",
        env_key: "SIMGRID",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["simgrid-mc", "teshsuite"],
        capabilities: SIMULATION_CAPS,
        input_formats: DISTRIBUTED_SIM_FORMATS,
        notes: "Distributed-system simulation framework for clusters, grids, clouds, and HPC",
    },
    ExternalValidationToolSpec {
        id: "cloudsim",
        display_name: "CloudSim",
        env_key: "CLOUDSIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["cloudsim-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: DISTRIBUTED_SIM_FORMATS,
        notes: "Java cloud/datacenter simulation adapter for resource-scheduling checks",
    },
    ExternalValidationToolSpec {
        id: "batsim",
        display_name: "Batsim",
        env_key: "BATSIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["batsim"],
        capabilities: SIMULATION_CAPS,
        input_formats: DISTRIBUTED_SIM_FORMATS,
        notes: "Batch-scheduler and distributed-platform simulation adapter",
    },
    ExternalValidationToolSpec {
        id: "neqsim",
        display_name: "NeqSim",
        env_key: "NEQSIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["neqsim-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: PROCESS_SIM_FORMATS,
        notes: "Open-source process simulation library for fluid and unit-operation models",
    },
    ExternalValidationToolSpec {
        id: "dwsim",
        display_name: "DWSIM",
        env_key: "DWSIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["dwsim-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: PROCESS_SIM_FORMATS,
        notes: "Open-source chemical-process simulation adapter",
    },
    ExternalValidationToolSpec {
        id: "cape-open",
        display_name: "CAPE-OPEN",
        env_key: "CAPE_OPEN",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["cape-open-adapter", "capeopen-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: PROCESS_SIM_FORMATS,
        notes: "CAPE-OPEN process-simulation interoperability adapter",
    },
    ExternalValidationToolSpec {
        id: "plant-simulation",
        display_name: "Tecnomatix Plant Simulation",
        env_key: "PLANT_SIMULATION",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["plant-simulation-adapter", "plantsim-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: &["json", "xml", "csv", "spp"],
        notes: "Commercial factory/logistics discrete-event simulation adapter",
    },
    ExternalValidationToolSpec {
        id: "extendsim",
        display_name: "ExtendSim",
        env_key: "EXTENDSIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["extendsim-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: &["json", "xml", "csv"],
        notes: "Commercial discrete-event and continuous simulation adapter",
    },
    ExternalValidationToolSpec {
        id: "gpss-world",
        display_name: "GPSS World",
        env_key: "GPSS_WORLD",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["gpss-adapter", "gpss-world-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: &["gps", "json", "txt"],
        notes: "GPSS-family discrete-event simulation adapter for queueing models",
    },
    ExternalValidationToolSpec {
        id: "anylogic",
        display_name: "AnyLogic",
        env_key: "ANYLOGIC",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["anylogic-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: JAVA_SIM_FORMATS,
        notes:
            "Commercial multimethod simulation adapter for DES, agent, and system-dynamics checks",
    },
    ExternalValidationToolSpec {
        id: "simio",
        display_name: "Simio",
        env_key: "SIMIO",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["simio-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: &["json", "xml", "csv"],
        notes: "Commercial discrete-event simulation adapter for digital-twin and process checks",
    },
    ExternalValidationToolSpec {
        id: "simul8",
        display_name: "Simul8",
        env_key: "SIMUL8",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["simul8-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: &["json", "xml", "csv"],
        notes: "Commercial discrete-event simulation adapter for process-flow validation",
    },
    ExternalValidationToolSpec {
        id: "arena",
        display_name: "Arena",
        env_key: "ARENA",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["arena-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: &["json", "xml", "csv"],
        notes: "Commercial Arena/SIMAN simulation adapter for manufacturing and queueing checks",
    },
    ExternalValidationToolSpec {
        id: "flexsim",
        display_name: "FlexSim",
        env_key: "FLEXSIM",
        family: ExternalValidationFamily::SimulationEngine,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["flexsim-adapter"],
        capabilities: SIMULATION_CAPS,
        input_formats: &["json", "xml", "csv"],
        notes: "Commercial 3D discrete-event simulation adapter for logistics and factory checks",
    },
    ExternalValidationToolSpec {
        id: "json-schema",
        display_name: "JSON Schema",
        env_key: "JSON_SCHEMA",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["check-jsonschema", "jsonschema"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Schema validation for JSON run artifacts and traces",
    },
    ExternalValidationToolSpec {
        id: "check-jsonschema",
        display_name: "check-jsonschema",
        env_key: "CHECK_JSONSCHEMA",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["check-jsonschema"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Python CLI for JSON Schema, YAML, and policy-style artifact validation",
    },
    ExternalValidationToolSpec {
        id: "ajv",
        display_name: "Ajv",
        env_key: "AJV",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["ajv"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "JavaScript JSON Schema validator and OpenAPI-compatible schema checker",
    },
    ExternalValidationToolSpec {
        id: "cue",
        display_name: "CUE",
        env_key: "CUE",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["cue"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: CUE_OUTPUT_FORMATS,
        notes: "CUE schema, constraint, and data validation adapter",
    },
    ExternalValidationToolSpec {
        id: "yamllint",
        display_name: "yamllint",
        env_key: "YAMLLINT",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["yamllint"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: YAML_OUTPUT_FORMATS,
        notes: "YAML syntax and convention validator for config and API artifacts",
    },
    ExternalValidationToolSpec {
        id: "csv-validator",
        display_name: "CSV Validator",
        env_key: "CSV_VALIDATOR",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["csv-validator", "csvlint"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: &["csv", "json"],
        notes: "CSV/table schema validation adapter for tabular run artifacts",
    },
    ExternalValidationToolSpec {
        id: "openapi-validator",
        display_name: "OpenAPI validator",
        env_key: "OPENAPI_VALIDATOR",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["openapi-generator-cli", "swagger-cli", "openapi"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: API_OUTPUT_FORMATS,
        notes: "OpenAPI/Swagger document and response-shape validation adapter",
    },
    ExternalValidationToolSpec {
        id: "openapi-spec-validator",
        display_name: "openapi-spec-validator",
        env_key: "OPENAPI_SPEC_VALIDATOR",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["openapi-spec-validator"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: API_OUTPUT_FORMATS,
        notes: "Python OpenAPI specification validator adapter",
    },
    ExternalValidationToolSpec {
        id: "spectral",
        display_name: "Spectral",
        env_key: "SPECTRAL",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["spectral"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: API_OUTPUT_FORMATS,
        notes: "OpenAPI/API description linter and validation adapter",
    },
    ExternalValidationToolSpec {
        id: "redocly-cli",
        display_name: "Redocly CLI",
        env_key: "REDOCLY_CLI",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["redocly", "redocly-cli"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: API_OUTPUT_FORMATS,
        notes: "OpenAPI linting and bundle validation adapter",
    },
    ExternalValidationToolSpec {
        id: "asyncapi-cli",
        display_name: "AsyncAPI CLI",
        env_key: "ASYNCAPI_CLI",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["asyncapi"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: API_OUTPUT_FORMATS,
        notes: "AsyncAPI document validation adapter for event/message interfaces",
    },
    ExternalValidationToolSpec {
        id: "graphql-schema",
        display_name: "GraphQL schema validator",
        env_key: "GRAPHQL_SCHEMA",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["graphql-schema-linter", "graphql-inspector"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: GRAPHQL_OUTPUT_FORMATS,
        notes: "GraphQL schema and operation validation adapter",
    },
    ExternalValidationToolSpec {
        id: "xmllint",
        display_name: "xmllint",
        env_key: "XMLLINT",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["xmllint"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: XML_OUTPUT_FORMATS,
        notes: "XML well-formedness and XSD validation adapter",
    },
    ExternalValidationToolSpec {
        id: "xml-schema",
        display_name: "XML Schema",
        env_key: "XML_SCHEMA",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["xmlschema", "xsd-validator"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: XML_OUTPUT_FORMATS,
        notes: "XSD/XML Schema validation adapter for structured XML run artifacts",
    },
    ExternalValidationToolSpec {
        id: "schematron",
        display_name: "Schematron",
        env_key: "SCHEMATRON",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["schematron-adapter", "jing", "saxon"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: XML_OUTPUT_FORMATS,
        notes: "Rule-based XML validation adapter for cross-field output constraints",
    },
    ExternalValidationToolSpec {
        id: "jing",
        display_name: "Jing",
        env_key: "JING",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["jing"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: XML_OUTPUT_FORMATS,
        notes: "RELAX NG and XML structural validation adapter for run artifacts",
    },
    ExternalValidationToolSpec {
        id: "saxon",
        display_name: "Saxon",
        env_key: "SAXON",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["saxon", "saxon-he", "saxon9he"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: XML_OUTPUT_FORMATS,
        notes: "Saxon-backed XML, XPath, and Schematron-style validation adapter",
    },
    ExternalValidationToolSpec {
        id: "pydantic",
        display_name: "Pydantic",
        env_key: "PYDANTIC",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["pydantic-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Python data-model validation adapter for structured run artifacts",
    },
    ExternalValidationToolSpec {
        id: "zod",
        display_name: "Zod",
        env_key: "ZOD",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["zod-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "TypeScript schema validation adapter for JSON/API run artifacts",
    },
    ExternalValidationToolSpec {
        id: "valibot",
        display_name: "Valibot",
        env_key: "VALIBOT",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["valibot-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "TypeScript schema validation adapter for structured artifacts",
    },
    ExternalValidationToolSpec {
        id: "marshmallow",
        display_name: "marshmallow",
        env_key: "MARSHMALLOW",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["marshmallow-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Python object/schema validation adapter for output payloads",
    },
    ExternalValidationToolSpec {
        id: "cerberus",
        display_name: "Cerberus",
        env_key: "CERBERUS",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["cerberus-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Python lightweight schema validation adapter for dictionaries and JSON",
    },
    ExternalValidationToolSpec {
        id: "python-xmlschema",
        display_name: "python-xmlschema",
        env_key: "PYTHON_XMLSCHEMA",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["xmlschema-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: XML_OUTPUT_FORMATS,
        notes: "Python XML Schema validation package adapter",
    },
    ExternalValidationToolSpec {
        id: "protobuf-conformance",
        display_name: "Protobuf conformance",
        env_key: "PROTOBUF_CONFORMANCE",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["conformance-test-runner", "protoc"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: &["proto", "pb", "json"],
        notes: "Protocol Buffers schema/conformance validation adapter",
    },
    ExternalValidationToolSpec {
        id: "protoc",
        display_name: "protoc",
        env_key: "PROTOC",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::SchemaOrSpecPath,
        command_aliases: &["protoc"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: &["proto", "pb", "json"],
        notes: "Protocol Buffers compiler and descriptor validation adapter",
    },
    ExternalValidationToolSpec {
        id: "avro-tools",
        display_name: "Avro tools",
        env_key: "AVRO_TOOLS",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["avro-tools"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: &["avro", "avsc", "json"],
        notes: "Avro schema and data-file validation adapter",
    },
    ExternalValidationToolSpec {
        id: "apache-avro",
        display_name: "Apache Avro",
        env_key: "APACHE_AVRO",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["avro-tools", "avro"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: &["avro", "avsc", "json"],
        notes: "Apache Avro schema and data-file validation adapter",
    },
    ExternalValidationToolSpec {
        id: "great-expectations",
        display_name: "Great Expectations",
        env_key: "GREAT_EXPECTATIONS",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["great_expectations", "gx"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Expectation-suite validation for tabular outputs and reports",
    },
    ExternalValidationToolSpec {
        id: "pandera",
        display_name: "Pandera",
        env_key: "PANDERA",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["pandera-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Python dataframe/schema validation adapter",
    },
    ExternalValidationToolSpec {
        id: "dbt",
        display_name: "dbt",
        env_key: "DBT",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["dbt"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: DBT_OUTPUT_FORMATS,
        notes: "dbt test adapter for relational output contracts and data models",
    },
    ExternalValidationToolSpec {
        id: "whylogs",
        display_name: "whylogs",
        env_key: "WHYLOGS",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["whylogs-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Data-profile and constraint validation adapter for logged outputs",
    },
    ExternalValidationToolSpec {
        id: "soda-core",
        display_name: "Soda Core",
        env_key: "SODA_CORE",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["soda"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Data-quality checks for tabular outputs and databases",
    },
    ExternalValidationToolSpec {
        id: "evidently",
        display_name: "Evidently",
        env_key: "EVIDENTLY",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["evidently-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Statistical output validation and drift checks",
    },
    ExternalValidationToolSpec {
        id: "deepchecks",
        display_name: "Deepchecks",
        env_key: "DEEPCHECKS",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["deepchecks-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Model/data validation checks for ML-style outputs",
    },
    ExternalValidationToolSpec {
        id: "frictionless",
        display_name: "Frictionless Data",
        env_key: "FRICTIONLESS",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["frictionless"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Data Package and tabular-data validation for CSV/JSON outputs",
    },
    ExternalValidationToolSpec {
        id: "parquet-tools",
        display_name: "Parquet tools",
        env_key: "PARQUET_TOOLS",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::NativeCli,
        artifact_kind: ExternalValidationArtifactKind::None,
        command_aliases: &["parquet-tools"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: PARQUET_OUTPUT_FORMATS,
        notes: "Parquet file metadata and schema validation adapter",
    },
    ExternalValidationToolSpec {
        id: "apache-arrow",
        display_name: "Apache Arrow",
        env_key: "APACHE_ARROW",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["arrow-adapter", "pyarrow-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: PARQUET_OUTPUT_FORMATS,
        notes: "Arrow/Parquet schema and columnar output validation adapter",
    },
    ExternalValidationToolSpec {
        id: "deequ",
        display_name: "Deequ",
        env_key: "DEEQU",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
        command_aliases: &["deequ-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Spark-based data-quality validation adapter for large output tables",
    },
    ExternalValidationToolSpec {
        id: "tensorflow-data-validation",
        display_name: "TensorFlow Data Validation",
        env_key: "TENSORFLOW_DATA_VALIDATION",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["tfdv-adapter"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Schema, anomaly, and statistics validation for ML/data pipeline outputs",
    },
    ExternalValidationToolSpec {
        id: "openrefine",
        display_name: "OpenRefine",
        env_key: "OPENREFINE",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::GenericAdapter,
        artifact_kind: ExternalValidationArtifactKind::NativeInstallDir,
        command_aliases: &["openrefine-adapter", "refine"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Data-cleaning and reconciliation adapter for inspecting messy tabular outputs",
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalValidationProbeStatus {
    Ready,
    NotConfigured,
    RuntimeMissing,
    AdapterMissing,
    ArtifactMissing,
}

impl ExternalValidationProbeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalValidationProbeStatus::Ready => "ready",
            ExternalValidationProbeStatus::NotConfigured => "not-configured",
            ExternalValidationProbeStatus::RuntimeMissing => "runtime-missing",
            ExternalValidationProbeStatus::AdapterMissing => "adapter-missing",
            ExternalValidationProbeStatus::ArtifactMissing => "artifact-missing",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalValidationRunStatus {
    Ok,
    Unavailable,
    Failed,
    InvalidOutput,
}

impl ExternalValidationRunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalValidationRunStatus::Ok => "ok",
            ExternalValidationRunStatus::Unavailable => "unavailable",
            ExternalValidationRunStatus::Failed => "failed",
            ExternalValidationRunStatus::InvalidOutput => "invalid-output",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternalValidationAdapterOptions {
    pub tool_id: String,
    pub command_path: Option<PathBuf>,
    pub working_dir: Option<PathBuf>,
    pub time_limit_secs: Option<f64>,
    pub extra_args: Vec<String>,
}

impl ExternalValidationAdapterOptions {
    pub fn for_tool(tool: &ExternalValidationToolSpec) -> Self {
        Self {
            tool_id: tool.id.to_string(),
            command_path: None,
            working_dir: None,
            time_limit_secs: None,
            extra_args: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalValidationProbe {
    pub tool_id: String,
    pub status: ExternalValidationProbeStatus,
    pub command: Option<PathBuf>,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalValidationRun {
    pub tool_id: String,
    pub status: ExternalValidationRunStatus,
    pub output: Option<Value>,
    pub elapsed_ms: f64,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalValidationTextFormat {
    SmtLib2,
    DimacsCnf,
    DimacsWcnf,
    MiniZinc,
    FlatZinc,
    TlaPlus,
    PrismModel,
    Json,
}

impl ExternalValidationTextFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalValidationTextFormat::SmtLib2 => "smtlib2",
            ExternalValidationTextFormat::DimacsCnf => "dimacs-cnf",
            ExternalValidationTextFormat::DimacsWcnf => "dimacs-wcnf",
            ExternalValidationTextFormat::MiniZinc => "minizinc",
            ExternalValidationTextFormat::FlatZinc => "flatzinc",
            ExternalValidationTextFormat::TlaPlus => "tla-plus",
            ExternalValidationTextFormat::PrismModel => "prism-model",
            ExternalValidationTextFormat::Json => "json",
        }
    }

    pub fn file_extension(self) -> &'static str {
        match self {
            ExternalValidationTextFormat::SmtLib2 => "smt2",
            ExternalValidationTextFormat::DimacsCnf => "cnf",
            ExternalValidationTextFormat::DimacsWcnf => "wcnf",
            ExternalValidationTextFormat::MiniZinc => "mzn",
            ExternalValidationTextFormat::FlatZinc => "fzn",
            ExternalValidationTextFormat::TlaPlus => "tla",
            ExternalValidationTextFormat::PrismModel => "pm",
            ExternalValidationTextFormat::Json => "json",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalValidationTextVerdict {
    Sat,
    Unsat,
    Unknown,
    Valid,
    Invalid,
    Success,
    Failure,
}

impl ExternalValidationTextVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalValidationTextVerdict::Sat => "sat",
            ExternalValidationTextVerdict::Unsat => "unsat",
            ExternalValidationTextVerdict::Unknown => "unknown",
            ExternalValidationTextVerdict::Valid => "valid",
            ExternalValidationTextVerdict::Invalid => "invalid",
            ExternalValidationTextVerdict::Success => "success",
            ExternalValidationTextVerdict::Failure => "failure",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sat" | "satisfiable" => Some(ExternalValidationTextVerdict::Sat),
            "unsat" | "unsatisfiable" => Some(ExternalValidationTextVerdict::Unsat),
            "unknown" => Some(ExternalValidationTextVerdict::Unknown),
            "valid" => Some(ExternalValidationTextVerdict::Valid),
            "invalid" => Some(ExternalValidationTextVerdict::Invalid),
            "success" => Some(ExternalValidationTextVerdict::Success),
            "failure" | "failed" => Some(ExternalValidationTextVerdict::Failure),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternalValidationTextCliOptions {
    pub tool_id: String,
    pub input_format: ExternalValidationTextFormat,
    pub command_path: Option<PathBuf>,
    pub working_dir: Option<PathBuf>,
    pub extra_args: Vec<String>,
    pub use_default_args: bool,
}

impl ExternalValidationTextCliOptions {
    pub fn for_tool(
        tool: &ExternalValidationToolSpec,
        input_format: ExternalValidationTextFormat,
    ) -> Self {
        Self {
            tool_id: tool.id.to_string(),
            input_format,
            command_path: None,
            working_dir: None,
            extra_args: Vec::new(),
            use_default_args: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternalValidationFileCliOptions {
    pub tool_id: String,
    pub input_format: ExternalValidationTextFormat,
    pub command_path: Option<PathBuf>,
    pub working_dir: Option<PathBuf>,
    pub extra_args: Vec<String>,
    pub use_default_args: bool,
    pub append_input_path: bool,
    pub file_extension: Option<String>,
}

impl ExternalValidationFileCliOptions {
    pub fn for_tool(
        tool: &ExternalValidationToolSpec,
        input_format: ExternalValidationTextFormat,
    ) -> Self {
        Self {
            tool_id: tool.id.to_string(),
            input_format,
            command_path: None,
            working_dir: None,
            extra_args: Vec::new(),
            use_default_args: true,
            append_input_path: true,
            file_extension: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalValidationArtifact {
    pub key: String,
    pub contents: String,
    pub file_name: Option<String>,
    pub file_extension: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ExternalValidationArtifactCliOptions {
    pub tool_id: String,
    pub input_format: ExternalValidationTextFormat,
    pub command_path: Option<PathBuf>,
    pub working_dir: Option<PathBuf>,
    pub extra_args: Vec<String>,
    pub use_default_args: bool,
}

impl ExternalValidationArtifactCliOptions {
    pub fn for_tool(
        tool: &ExternalValidationToolSpec,
        input_format: ExternalValidationTextFormat,
    ) -> Self {
        Self {
            tool_id: tool.id.to_string(),
            input_format,
            command_path: None,
            working_dir: None,
            extra_args: Vec::new(),
            use_default_args: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ExternalValidationCliInvocation {
    Text {
        label: String,
        options: ExternalValidationTextCliOptions,
    },
    File {
        label: String,
        options: ExternalValidationFileCliOptions,
    },
    Artifact {
        label: String,
        artifacts: Vec<ExternalValidationArtifact>,
        options: ExternalValidationArtifactCliOptions,
    },
}

impl ExternalValidationCliInvocation {
    pub fn label(&self) -> &str {
        match self {
            ExternalValidationCliInvocation::Text { label, .. }
            | ExternalValidationCliInvocation::File { label, .. }
            | ExternalValidationCliInvocation::Artifact { label, .. } => label,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalValidationConsensusRun {
    pub label: String,
    pub run: ExternalValidationRun,
    pub verdict: Option<ExternalValidationTextVerdict>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalValidationConsensusReport {
    pub expected_verdict: Option<ExternalValidationTextVerdict>,
    pub agreed_verdict: Option<ExternalValidationTextVerdict>,
    pub all_successful: bool,
    pub all_successful_verdicts_agree: bool,
    pub expected_matches: bool,
    pub agreement: bool,
    pub runs: Vec<ExternalValidationConsensusRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiniZincValidationRequest {
    pub model: String,
    pub data: Option<String>,
    pub solver: Option<String>,
    pub checker_model: Option<String>,
}

pub fn minizinc_validation_request_to_json(request: &MiniZincValidationRequest) -> Value {
    json!({
        "kind": "minizinc-validation",
        "format": "mzn",
        "model": &request.model,
        "data": &request.data,
        "solver": &request.solver,
        "checker_model": &request.checker_model,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SmtSort {
    Bool,
    Int,
    Real,
    BitVector(u32),
    Custom(String),
}

impl SmtSort {
    pub fn as_smtlib(&self) -> String {
        match self {
            Self::Bool => "Bool".to_string(),
            Self::Int => "Int".to_string(),
            Self::Real => "Real".to_string(),
            Self::BitVector(width) => format!("(_ BitVec {width})"),
            Self::Custom(sort) => sort.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmtDeclaration {
    pub name: String,
    pub sort: SmtSort,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmtLibValidationScript {
    pub logic: Option<String>,
    pub declarations: Vec<SmtDeclaration>,
    pub assertions: Vec<String>,
    pub check_sat_assumptions: Vec<String>,
    pub get_model: bool,
}

pub fn smtlib_validation_script_to_string(script: &SmtLibValidationScript) -> String {
    let mut out = String::new();
    if let Some(logic) = &script.logic {
        out.push_str("(set-logic ");
        out.push_str(logic.trim());
        out.push_str(")\n");
    }
    for declaration in &script.declarations {
        out.push_str("(declare-const ");
        out.push_str(declaration.name.trim());
        out.push(' ');
        out.push_str(&declaration.sort.as_smtlib());
        out.push_str(")\n");
    }
    for assertion in &script.assertions {
        let trimmed = assertion.trim();
        if trimmed.starts_with("(assert ") {
            out.push_str(trimmed);
        } else {
            out.push_str("(assert ");
            if trimmed.starts_with('(') || !trimmed.chars().any(char::is_whitespace) {
                out.push_str(trimmed);
            } else {
                out.push('(');
                out.push_str(trimmed);
                out.push(')');
            }
            out.push(')');
        }
        out.push('\n');
    }
    if script.check_sat_assumptions.is_empty() {
        out.push_str("(check-sat)\n");
    } else {
        out.push_str("(check-sat-assuming (");
        out.push_str(&script.check_sat_assumptions.join(" "));
        out.push_str("))\n");
    }
    if script.get_model {
        out.push_str("(get-model)\n");
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimacsCnf {
    pub num_vars: usize,
    pub clauses: Vec<Vec<i32>>,
    pub comments: Vec<String>,
}

pub fn dimacs_cnf_to_string(cnf: &DimacsCnf) -> String {
    let mut out = String::new();
    for comment in &cnf.comments {
        out.push_str("c ");
        out.push_str(comment.trim());
        out.push('\n');
    }
    out.push_str(&format!("p cnf {} {}\n", cnf.num_vars, cnf.clauses.len()));
    for clause in &cnf.clauses {
        for literal in clause {
            out.push_str(&format!("{literal} "));
        }
        out.push_str("0\n");
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimacsWeightedClause {
    pub weight: u64,
    pub literals: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimacsWcnf {
    pub num_vars: usize,
    pub clauses: Vec<DimacsWeightedClause>,
    pub top_weight: Option<u64>,
    pub comments: Vec<String>,
}

pub fn dimacs_wcnf_to_string(wcnf: &DimacsWcnf) -> String {
    let mut out = String::new();
    for comment in &wcnf.comments {
        out.push_str("c ");
        out.push_str(comment.trim());
        out.push('\n');
    }
    out.push_str(&format!("p wcnf {} {}", wcnf.num_vars, wcnf.clauses.len()));
    if let Some(top_weight) = wcnf.top_weight {
        out.push_str(&format!(" {top_weight}"));
    }
    out.push('\n');
    for clause in &wcnf.clauses {
        out.push_str(&format!("{} ", clause.weight));
        for literal in &clause.literals {
            out.push_str(&format!("{literal} "));
        }
        out.push_str("0\n");
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TlaValidationModule {
    pub module_name: String,
    pub extends: Vec<String>,
    pub constants: Vec<String>,
    pub variables: Vec<String>,
    pub init: String,
    pub next: String,
    pub invariants: Vec<String>,
    pub temporal_properties: Vec<String>,
}

pub fn tla_validation_module_to_string(module: &TlaValidationModule) -> String {
    let mut out = String::new();
    out.push_str("---- MODULE ");
    out.push_str(module.module_name.trim());
    out.push_str(" ----\n");
    if !module.extends.is_empty() {
        out.push_str("EXTENDS ");
        out.push_str(&module.extends.join(", "));
        out.push_str("\n\n");
    }
    if !module.constants.is_empty() {
        out.push_str("CONSTANTS ");
        out.push_str(&module.constants.join(", "));
        out.push('\n');
    }
    if !module.variables.is_empty() {
        out.push_str("VARIABLES ");
        out.push_str(&module.variables.join(", "));
        out.push_str("\n\n");
    }
    out.push_str("Init == ");
    out.push_str(module.init.trim());
    out.push_str("\n\nNext == ");
    out.push_str(module.next.trim());
    out.push_str("\n\n");
    for (idx, invariant) in module.invariants.iter().enumerate() {
        out.push_str(&format!("Invariant{} == {}\n", idx + 1, invariant.trim()));
    }
    for (idx, property) in module.temporal_properties.iter().enumerate() {
        out.push_str(&format!(
            "TemporalProperty{} == {}\n",
            idx + 1,
            property.trim()
        ));
    }
    if !module.variables.is_empty() {
        let frame = if module.variables.len() == 1 {
            module.variables[0].clone()
        } else {
            format!("<<{}>>", module.variables.join(", "))
        };
        out.push_str("\nSpec == Init /\\ [][Next]_");
        out.push_str(&frame);
        out.push('\n');
    }
    out.push_str("====\n");
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrismModule {
    pub name: String,
    pub variables: Vec<String>,
    pub commands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrismValidationModel {
    pub model_type: String,
    pub declarations: Vec<String>,
    pub modules: Vec<PrismModule>,
    pub labels: Vec<String>,
    pub properties: Vec<String>,
}

pub fn prism_validation_model_to_string(model: &PrismValidationModel) -> String {
    let mut out = String::new();
    out.push_str(model.model_type.trim());
    out.push_str("\n\n");
    for declaration in &model.declarations {
        out.push_str(declaration.trim());
        out.push('\n');
    }
    if !model.declarations.is_empty() {
        out.push('\n');
    }
    for module in &model.modules {
        out.push_str("module ");
        out.push_str(module.name.trim());
        out.push('\n');
        for variable in &module.variables {
            out.push_str("  ");
            out.push_str(variable.trim());
            out.push('\n');
        }
        for command in &module.commands {
            out.push_str("  ");
            out.push_str(command.trim());
            out.push('\n');
        }
        out.push_str("endmodule\n\n");
    }
    for label in &model.labels {
        out.push_str(label.trim());
        out.push('\n');
    }
    out
}

pub fn prism_validation_properties_to_string(model: &PrismValidationModel) -> String {
    let mut out = String::new();
    for property in &model.properties {
        out.push_str(property.trim());
        out.push('\n');
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct JsonSchemaValidationRequest {
    pub schema: Value,
    pub instance: Value,
    pub draft: Option<String>,
}

pub fn json_schema_validation_request_to_json(request: &JsonSchemaValidationRequest) -> Value {
    json!({
        "kind": "json-schema-validation",
        "schema": &request.schema,
        "instance": &request.instance,
        "draft": &request.draft,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimulationMetricExpectation {
    pub name: String,
    pub target: f64,
    pub tolerance: f64,
    pub comparison: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimulationValidationRequest {
    pub engine_id: String,
    pub model_format: String,
    pub model: Value,
    pub scenario: Option<Value>,
    pub expected_trace_properties: Vec<String>,
    pub metric_expectations: Vec<SimulationMetricExpectation>,
}

pub fn simulation_validation_request_to_json(request: &SimulationValidationRequest) -> Value {
    let metrics: Vec<Value> = request
        .metric_expectations
        .iter()
        .map(|metric| {
            json!({
                "name": &metric.name,
                "target": metric.target,
                "tolerance": metric.tolerance,
                "comparison": &metric.comparison,
            })
        })
        .collect();
    json!({
        "kind": "simulation-validation",
        "engine": &request.engine_id,
        "model_format": &request.model_format,
        "model": &request.model,
        "scenario": &request.scenario,
        "expected_trace_properties": &request.expected_trace_properties,
        "metric_expectations": metrics,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalSimulationValidationReferenceOptions {
    pub engine_id: Option<String>,
}

impl Default for ExternalSimulationValidationReferenceOptions {
    fn default() -> Self {
        ExternalSimulationValidationReferenceOptions { engine_id: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalSimulationValidationStatus {
    Ok,
    Unavailable,
    Failed,
    Unknown,
}

impl ExternalSimulationValidationStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalSimulationValidationStatus::Ok => "ok",
            ExternalSimulationValidationStatus::Unavailable => "unavailable",
            ExternalSimulationValidationStatus::Failed => "failed",
            ExternalSimulationValidationStatus::Unknown => "unknown",
        }
    }

    pub fn from_label(label: &str) -> Self {
        match label {
            "ok" => ExternalSimulationValidationStatus::Ok,
            "unavailable" => ExternalSimulationValidationStatus::Unavailable,
            "failed" => ExternalSimulationValidationStatus::Failed,
            _ => ExternalSimulationValidationStatus::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalSimulationValidationVerdict {
    Valid,
    Invalid,
    Failure,
    Unknown,
}

impl ExternalSimulationValidationVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalSimulationValidationVerdict::Valid => "valid",
            ExternalSimulationValidationVerdict::Invalid => "invalid",
            ExternalSimulationValidationVerdict::Failure => "failure",
            ExternalSimulationValidationVerdict::Unknown => "unknown",
        }
    }

    pub fn from_label(label: &str) -> Self {
        match label {
            "valid" => ExternalSimulationValidationVerdict::Valid,
            "invalid" => ExternalSimulationValidationVerdict::Invalid,
            "failure" => ExternalSimulationValidationVerdict::Failure,
            _ => ExternalSimulationValidationVerdict::Unknown,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalSimulationValidationReferenceRun {
    pub engine_id: String,
    pub simulator: String,
    pub status: ExternalSimulationValidationStatus,
    pub verdict: ExternalSimulationValidationVerdict,
    pub metrics: BTreeMap<String, f64>,
    pub checks: Vec<Value>,
    pub trace: Vec<Value>,
    pub raw: Value,
    pub elapsed_ms: f64,
    pub message: String,
}

#[derive(Debug, Deserialize)]
struct SimulationValidationReferenceOutput {
    status: String,
    verdict: String,
    #[serde(default)]
    simulator: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    metrics: BTreeMap<String, f64>,
    #[serde(default)]
    checks: Vec<Value>,
    #[serde(default)]
    trace: Vec<Value>,
}

pub fn external_simulation_validation_tool_specs() -> Vec<&'static ExternalValidationToolSpec> {
    EXTERNAL_VALIDATION_TOOLS
        .iter()
        .filter(|tool| tool.family == ExternalValidationFamily::SimulationEngine)
        .collect()
}

pub fn external_simulation_validation_engine_manifest() -> Value {
    Value::Array(
        external_simulation_validation_tool_specs()
            .into_iter()
            .map(|tool| {
                json!({
                    "id": tool.id,
                    "displayName": tool.display_name,
                    "runtime": tool.runtime.as_str(),
                    "artifactKind": tool.artifact_kind.env_suffix().unwrap_or("none"),
                    "commandAliases": tool.command_aliases,
                    "inputFormats": tool.input_formats,
                    "notes": tool.notes,
                })
            })
            .collect(),
    )
}

pub fn external_simulation_validation_reference_script() -> PathBuf {
    let root = env::var_os("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts")
        .join("simulation_validation_reference.py")
}

const EVENT_SIMULATION_ENGINES: &[&str] = &[
    "simpy",
    "salabim",
    "simmer",
    "jaamsim",
    "anylogic",
    "simio",
    "simul8",
    "arena",
    "flexsim",
    "plant-simulation",
    "extendsim",
    "gpss-world",
    "simulink",
    "ptolemy-ii",
];

const MOBILITY_SIMULATION_ENGINES: &[&str] = &[
    "ns-3", "ns3", "omnetpp", "omnet++", "sumo", "matsim", "carla",
];

const ENERGY_SIMULATION_ENGINES: &[&str] = &[
    "energyplus",
    "openstudio",
    "openmodelica",
    "fmi-fmu",
    "fmi",
    "fmu",
    "omsimulator",
    "simulink",
    "gridlabd",
    "opendss",
    "pandapower",
];

const PHYSICS_SIMULATION_ENGINES: &[&str] = &[
    "gazebo",
    "webots",
    "mujoco",
    "drake",
    "pybullet",
    "carla",
    "isaac-sim",
    "airsim",
];

const AGENT_SIMULATION_ENGINES: &[&str] = &[
    "mesa",
    "repast",
    "repast-simphony",
    "mason",
    "netlogo",
    "agentpy",
];

const DISTRIBUTED_SIMULATION_ENGINES: &[&str] =
    &["simgrid", "cloudsim", "batsim", "gem5", "ptolemy-ii"];

const PROCESS_SIMULATION_ENGINES: &[&str] =
    &["neqsim", "dwsim", "cape-open", "copasi", "tellurium"];

#[derive(Clone, Debug)]
struct RustSimulationJob {
    arrival: f64,
    start: f64,
    departure: f64,
    wait: f64,
}

#[derive(Clone, Debug)]
struct RustSimulationTraceEvent {
    time: f64,
    event: &'static str,
    job: usize,
    server: Option<usize>,
}

#[derive(Clone, Debug)]
struct RustMobilityVehicle {
    depart: f64,
    arrival: f64,
    travel_time: f64,
}

#[derive(Clone, Debug)]
struct RustMobilityTraceEvent {
    time: f64,
    event: &'static str,
    vehicle: usize,
    segment: Option<usize>,
}

#[derive(Clone, Debug)]
struct RustAgentSimulation {
    trace: Vec<Value>,
    metrics: BTreeMap<String, f64>,
    interactions: Vec<Value>,
}

fn finite_simulation_float(value: Option<&Value>, default: Option<f64>) -> Result<f64, String> {
    let Some(value) = value else {
        return default.ok_or_else(|| "expected finite number".to_string());
    };
    let out = match value {
        Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| "expected finite number".to_string())?,
        Value::String(text) => text
            .parse::<f64>()
            .map_err(|_| "expected finite number".to_string())?,
        _ => return Err("expected finite number".to_string()),
    };
    if out.is_finite() {
        Ok(out)
    } else {
        Err("expected finite number".to_string())
    }
}

fn simulation_i64(value: Option<&Value>, default: i64) -> Result<i64, String> {
    let Some(value) = value else {
        return Ok(default);
    };
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_u64().and_then(|value| i64::try_from(value).ok()))
            .or_else(|| number.as_f64().map(|value| value as i64))
            .ok_or_else(|| "expected integer".to_string()),
        Value::String(text) => text
            .parse::<i64>()
            .map_err(|_| "expected integer".to_string()),
        _ => Err("expected integer".to_string()),
    }
}

fn simulation_usize(value: Option<&Value>, default: usize) -> Result<usize, String> {
    let Some(value) = value else {
        return Ok(default);
    };
    let raw = match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_u64().and_then(|value| i64::try_from(value).ok()))
            .or_else(|| number.as_f64().map(|value| value as i64))
            .ok_or_else(|| "expected integer".to_string())?,
        Value::String(text) => text
            .parse::<i64>()
            .map_err(|_| "expected integer".to_string())?,
        _ => return Err("expected integer".to_string()),
    };
    usize::try_from(raw).map_err(|_| "expected non-negative integer".to_string())
}

fn simulation_float_array(value: Option<&Value>) -> Result<Option<Vec<f64>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .ok_or_else(|| "expected numeric array".to_string())?
        .iter()
        .map(|item| finite_simulation_float(Some(item), None))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(values))
}

fn simulation_object_or_empty(
    value: Option<&Value>,
) -> Result<&serde_json::Map<String, Value>, String> {
    static EMPTY: std::sync::OnceLock<serde_json::Map<String, Value>> = std::sync::OnceLock::new();
    match value {
        Some(Value::Object(object)) => Ok(object),
        Some(_) => Err("simulation model must be an object".to_string()),
        None => Ok(EMPTY.get_or_init(serde_json::Map::new)),
    }
}

fn python_round_to_i64(value: f64) -> i64 {
    let floor = value.floor();
    let frac = value - floor;
    if (frac - 0.5).abs() <= 1e-12 {
        let floor_i = floor as i64;
        if floor_i % 2 == 0 {
            floor_i
        } else {
            floor_i + 1
        }
    } else {
        value.round() as i64
    }
}

fn value_truthy(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|value| value != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(values)) => !values.is_empty(),
        Some(Value::Object(values)) => !values.is_empty(),
        Some(Value::Null) | None => false,
    }
}

fn stream_endpoint_matches(
    stream: &serde_json::Map<String, Value>,
    key: &str,
    alias: &str,
) -> bool {
    match stream.get(key) {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.is_empty() || text == alias,
        _ => false,
    }
}

fn normalize_event_simulation_model(model: &Value) -> Result<(usize, Vec<f64>, Vec<f64>), String> {
    let model = model
        .as_object()
        .ok_or_else(|| "simulation model must be an object".to_string())?;
    let servers = simulation_usize(model.get("servers"), 1)?;
    if servers == 0 {
        return Err("servers must be positive".to_string());
    }

    let arrivals = simulation_float_array(model.get("arrival_times"))?;
    let services = simulation_float_array(model.get("service_times"))?;
    let (arrivals, services) = match (arrivals, services) {
        (Some(arrivals), Some(services)) => (arrivals, services),
        _ => {
            let jobs = simulation_usize(model.get("jobs"), 5)?;
            let interarrival = finite_simulation_float(model.get("interarrival_time"), Some(1.0))?;
            let service_time = finite_simulation_float(model.get("service_time"), Some(1.0))?;
            (
                (0..jobs)
                    .map(|idx| idx as f64 * interarrival)
                    .collect::<Vec<_>>(),
                vec![service_time; jobs],
            )
        }
    };
    if arrivals.len() != services.len() {
        return Err("arrival_times and service_times length mismatch".to_string());
    }
    if services.iter().any(|service| *service < 0.0) {
        return Err("service times must be non-negative".to_string());
    }
    if arrivals.windows(2).any(|window| window[0] > window[1]) {
        return Err("arrival_times must be sorted".to_string());
    }
    Ok((servers, arrivals, services))
}

fn simulate_event_network_with_rust(
    model: &Value,
) -> Result<(Vec<RustSimulationJob>, Vec<Value>, BTreeMap<String, f64>), String> {
    let (servers, arrivals, services) = normalize_event_simulation_model(model)?;
    let mut available_at = vec![0.0; servers];
    let mut jobs = Vec::with_capacity(arrivals.len());
    let mut trace = Vec::with_capacity(arrivals.len() * 3);
    for (job, (&arrival, &service)) in arrivals.iter().zip(&services).enumerate() {
        let server = available_at
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let start = arrival.max(available_at[server]);
        let departure = start + service;
        available_at[server] = departure;
        let wait = start - arrival;
        jobs.push(RustSimulationJob {
            arrival,
            start,
            departure,
            wait,
        });
        trace.push(RustSimulationTraceEvent {
            time: arrival,
            event: "arrival",
            job,
            server: None,
        });
        trace.push(RustSimulationTraceEvent {
            time: start,
            event: "service_start",
            job,
            server: Some(server),
        });
        trace.push(RustSimulationTraceEvent {
            time: departure,
            event: "departure",
            job,
            server: Some(server),
        });
    }
    trace.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.event.cmp(b.event))
            .then_with(|| a.job.cmp(&b.job))
    });
    let trace = trace
        .into_iter()
        .map(|event| {
            let mut item = serde_json::Map::new();
            item.insert("time".to_string(), json!(event.time));
            item.insert("event".to_string(), json!(event.event));
            item.insert("job".to_string(), json!(event.job));
            if let Some(server) = event.server {
                item.insert("server".to_string(), json!(server));
            }
            Value::Object(item)
        })
        .collect::<Vec<_>>();

    let waits = jobs.iter().map(|job| job.wait).collect::<Vec<_>>();
    let sojourns = jobs
        .iter()
        .map(|job| job.departure - job.arrival)
        .collect::<Vec<_>>();
    let mut metrics = BTreeMap::new();
    metrics.insert("jobs_completed".to_string(), jobs.len() as f64);
    metrics.insert(
        "mean_wait".to_string(),
        if waits.is_empty() {
            0.0
        } else {
            waits.iter().sum::<f64>() / waits.len() as f64
        },
    );
    metrics.insert(
        "max_wait".to_string(),
        waits.iter().copied().fold(0.0_f64, f64::max),
    );
    metrics.insert(
        "mean_sojourn".to_string(),
        if sojourns.is_empty() {
            0.0
        } else {
            sojourns.iter().sum::<f64>() / sojourns.len() as f64
        },
    );
    metrics.insert(
        "makespan".to_string(),
        jobs.iter().map(|job| job.departure).fold(0.0_f64, f64::max),
    );
    let service_sum = services.iter().sum::<f64>();
    let horizon = available_at
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1.0);
    metrics.insert(
        "utilization_lower_bound".to_string(),
        service_sum / (servers as f64 * horizon),
    );
    Ok((jobs, trace, metrics))
}

fn simulate_mobility_network_with_rust(
    model: &Value,
) -> Result<(Vec<RustMobilityVehicle>, Vec<Value>, BTreeMap<String, f64>), String> {
    let model = model
        .as_object()
        .ok_or_else(|| "simulation model must be an object".to_string())?;
    let routes = model
        .get("routes")
        .and_then(Value::as_array)
        .filter(|routes| !routes.is_empty())
        .ok_or_else(|| "mobility model needs non-empty routes".to_string())?;

    let mut vehicles = Vec::with_capacity(routes.len());
    let mut trace = Vec::new();
    for (vehicle, route) in routes.iter().enumerate() {
        let route = route
            .as_object()
            .ok_or_else(|| "each mobility route must be an object".to_string())?;
        let depart = finite_simulation_float(route.get("depart"), Some(0.0))?;
        let segments = route
            .get("segments")
            .or_else(|| route.get("travel_times"))
            .and_then(Value::as_array)
            .filter(|segments| !segments.is_empty())
            .ok_or_else(|| "each mobility route needs segments or travel_times".to_string())?;
        let mut travel_time = 0.0;
        let mut time = depart;
        trace.push(RustMobilityTraceEvent {
            time,
            event: "vehicle_depart",
            vehicle,
            segment: None,
        });
        for (segment, value) in segments.iter().enumerate() {
            let segment_time = if let Some(segment) = value.as_object() {
                finite_simulation_float(segment.get("travel_time"), Some(0.0))?
            } else {
                finite_simulation_float(Some(value), None)?
            };
            if segment_time < 0.0 {
                return Err("travel times must be non-negative".to_string());
            }
            travel_time += segment_time;
            time += segment_time;
            trace.push(RustMobilityTraceEvent {
                time,
                event: "segment_arrive",
                vehicle,
                segment: Some(segment),
            });
        }
        vehicles.push(RustMobilityVehicle {
            depart,
            arrival: time,
            travel_time,
        });
        trace.push(RustMobilityTraceEvent {
            time,
            event: "vehicle_arrive",
            vehicle,
            segment: None,
        });
    }
    trace.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.event.cmp(b.event))
            .then_with(|| a.vehicle.cmp(&b.vehicle))
    });
    let trace = trace
        .into_iter()
        .map(|event| {
            let mut item = serde_json::Map::new();
            item.insert("time".to_string(), json!(event.time));
            item.insert("event".to_string(), json!(event.event));
            item.insert("vehicle".to_string(), json!(event.vehicle));
            if let Some(segment) = event.segment {
                item.insert("segment".to_string(), json!(segment));
            }
            Value::Object(item)
        })
        .collect::<Vec<_>>();

    let travel_times = vehicles
        .iter()
        .map(|vehicle| vehicle.travel_time)
        .collect::<Vec<_>>();
    let mut metrics = BTreeMap::new();
    metrics.insert("vehicles_completed".to_string(), vehicles.len() as f64);
    metrics.insert(
        "mean_travel_time".to_string(),
        travel_times.iter().sum::<f64>() / travel_times.len() as f64,
    );
    metrics.insert(
        "max_travel_time".to_string(),
        travel_times
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    metrics.insert(
        "min_travel_time".to_string(),
        travel_times.iter().copied().fold(f64::INFINITY, f64::min),
    );
    metrics.insert(
        "last_arrival".to_string(),
        vehicles
            .iter()
            .map(|vehicle| vehicle.arrival)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    Ok((vehicles, trace, metrics))
}

fn simulate_energy_balance_with_rust(
    model: &Value,
    scenario: &Value,
) -> Result<(Vec<Value>, BTreeMap<String, f64>), String> {
    let model = model
        .as_object()
        .ok_or_else(|| "simulation model must be an object".to_string())?;
    let scenario = simulation_object_or_empty(Some(scenario))?;
    let zone_values = model
        .get("zones")
        .and_then(Value::as_array)
        .filter(|zones| !zones.is_empty())
        .cloned()
        .unwrap_or_else(|| vec![Value::Object(model.clone())]);
    let horizon = finite_simulation_float(
        scenario.get("horizon"),
        Some(finite_simulation_float(model.get("horizon"), Some(4.0))?),
    )?;
    let step = finite_simulation_float(
        scenario.get("step"),
        Some(finite_simulation_float(model.get("step"), Some(1.0))?),
    )?;
    if horizon <= 0.0 || step <= 0.0 {
        return Err("energy horizon and step must be positive".to_string());
    }
    let steps = python_round_to_i64(horizon / step).max(1) as usize;
    let mut trace = Vec::new();
    let mut final_errors = Vec::new();
    let mut energy_kwh = 0.0;
    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;
    for (zone, value) in zone_values.iter().enumerate() {
        let zone_model = value
            .as_object()
            .ok_or_else(|| "energy zone must be an object".to_string())?;
        let mut temp = finite_simulation_float(zone_model.get("initial_temp"), Some(20.0))?;
        let setpoint = finite_simulation_float(zone_model.get("setpoint"), Some(21.0))?;
        let outdoor = finite_simulation_float(zone_model.get("outdoor_temp"), Some(10.0))?;
        let ua = finite_simulation_float(zone_model.get("ua"), Some(0.2))?;
        let heat_capacity = finite_simulation_float(zone_model.get("heat_capacity"), Some(5.0))?;
        let hvac_power = finite_simulation_float(zone_model.get("hvac_power"), Some(4.0))?;
        let internal_gain = finite_simulation_float(zone_model.get("internal_gain"), Some(0.0))?;
        if heat_capacity <= 0.0 {
            return Err("heat_capacity must be positive".to_string());
        }
        for step_idx in 0..steps {
            let error = setpoint - temp;
            let hvac = (error * hvac_power).clamp(-hvac_power, hvac_power);
            temp += ((ua * (outdoor - temp)) + internal_gain + hvac) * step / heat_capacity;
            energy_kwh += hvac.abs() * step;
            min_temp = min_temp.min(temp);
            max_temp = max_temp.max(temp);
            trace.push(json!({
                "time": (step_idx + 1) as f64 * step,
                "event": "zone_temperature",
                "zone": zone,
                "temperature": temp,
                "hvac": hvac,
            }));
        }
        final_errors.push((temp - setpoint).abs());
    }
    let mut metrics = BTreeMap::new();
    metrics.insert("energy_kwh".to_string(), energy_kwh);
    metrics.insert(
        "max_abs_setpoint_error".to_string(),
        final_errors.iter().copied().fold(0.0_f64, f64::max),
    );
    metrics.insert(
        "min_temperature".to_string(),
        if min_temp.is_finite() { min_temp } else { 0.0 },
    );
    metrics.insert(
        "max_temperature".to_string(),
        if max_temp.is_finite() { max_temp } else { 0.0 },
    );
    metrics.insert("zones".to_string(), zone_values.len() as f64);
    Ok((trace, metrics))
}

fn simulate_physics_trajectory_with_rust(
    model: &Value,
    scenario: &Value,
) -> Result<(Vec<Value>, BTreeMap<String, f64>), String> {
    let model = model
        .as_object()
        .ok_or_else(|| "simulation model must be an object".to_string())?;
    let scenario = simulation_object_or_empty(Some(scenario))?;
    let dt = finite_simulation_float(
        scenario.get("dt"),
        Some(finite_simulation_float(model.get("dt"), Some(0.1))?),
    )?;
    let steps = simulation_i64(scenario.get("steps").or_else(|| model.get("steps")), 10)?;
    if dt <= 0.0 || steps <= 0 {
        return Err("trajectory dt and steps must be positive".to_string());
    }
    let mut position = finite_simulation_float(model.get("initial_position"), Some(0.0))?;
    let mut velocity = finite_simulation_float(model.get("initial_velocity"), Some(0.0))?;
    let acceleration = finite_simulation_float(
        model.get("acceleration"),
        Some(finite_simulation_float(
            model.get("acceleration_command"),
            Some(0.0),
        )?),
    )?;
    let floor = match model.get("floor") {
        Some(value) => finite_simulation_float(Some(value), None)?,
        None => f64::NEG_INFINITY,
    };
    let mut trace = vec![json!({
        "time": 0.0,
        "event": "state",
        "position": position,
        "velocity": velocity,
    })];
    let mut positions = vec![position];
    let mut path_length = 0.0;
    for step_idx in 0..steps as usize {
        let previous = position;
        velocity += acceleration * dt;
        position += velocity * dt;
        if position < floor {
            position = floor;
            velocity = velocity.max(0.0);
        }
        path_length += (position - previous).abs();
        positions.push(position);
        trace.push(json!({
            "time": (step_idx + 1) as f64 * dt,
            "event": "state",
            "position": position,
            "velocity": velocity,
        }));
    }
    let mut metrics = BTreeMap::new();
    metrics.insert("final_position".to_string(), position);
    metrics.insert("final_velocity".to_string(), velocity);
    metrics.insert(
        "max_position".to_string(),
        positions.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    metrics.insert(
        "min_position".to_string(),
        positions.iter().copied().fold(f64::INFINITY, f64::min),
    );
    metrics.insert("path_length".to_string(), path_length);
    Ok((trace, metrics))
}

fn simulate_agent_based_with_rust(
    model: &Value,
    scenario: &Value,
) -> Result<RustAgentSimulation, String> {
    let model = model
        .as_object()
        .ok_or_else(|| "simulation model must be an object".to_string())?;
    let scenario = simulation_object_or_empty(Some(scenario))?;
    let agents = model
        .get("agents")
        .and_then(Value::as_array)
        .filter(|agents| !agents.is_empty())
        .ok_or_else(|| "agent-based model needs non-empty agents".to_string())?;
    let steps = simulation_i64(scenario.get("steps").or_else(|| model.get("steps")), 1)?;
    if steps < 0 {
        return Err("agent-based steps must be non-negative".to_string());
    }
    let interactions = match model.get("interactions").or_else(|| model.get("edges")) {
        Some(Value::Array(values)) => values.clone(),
        Some(Value::Null) | None => Vec::new(),
        Some(_) => return Err("agent-based interactions must be an array".to_string()),
    };
    let mut trace = Vec::with_capacity(steps as usize + 1);
    for step in 0..=steps as usize {
        trace.push(json!({
            "time": step as f64,
            "event": "step",
            "agents": agents.len(),
        }));
    }
    let stateful_agents = agents
        .iter()
        .filter(|agent| {
            agent
                .as_object()
                .is_some_and(|agent| value_truthy(agent.get("state").or_else(|| agent.get("type"))))
        })
        .count();
    let mut metrics = BTreeMap::new();
    metrics.insert("agents".to_string(), agents.len() as f64);
    metrics.insert("steps".to_string(), steps as f64);
    metrics.insert("interactions".to_string(), interactions.len() as f64);
    metrics.insert("stateful_agents".to_string(), stateful_agents as f64);
    Ok(RustAgentSimulation {
        trace,
        metrics,
        interactions,
    })
}

fn simulate_distributed_system_with_rust(
    model: &Value,
) -> Result<(Vec<Value>, BTreeMap<String, f64>), String> {
    let model = model
        .as_object()
        .ok_or_else(|| "simulation model must be an object".to_string())?;
    let hosts = model
        .get("hosts")
        .and_then(Value::as_array)
        .filter(|hosts| !hosts.is_empty())
        .ok_or_else(|| "distributed-system model needs non-empty hosts".to_string())?;
    let links = model
        .get("links")
        .and_then(Value::as_array)
        .ok_or_else(|| "distributed-system links must be an array".to_string())?;
    let tasks = model
        .get("tasks")
        .or_else(|| model.get("workloads"))
        .and_then(Value::as_array)
        .ok_or_else(|| "distributed-system tasks/workloads must be an array".to_string())?;
    let mut total_capacity = 0.0;
    let mut min_bandwidth = f64::INFINITY;
    let mut total_work = 0.0;
    for host in hosts {
        let host = host
            .as_object()
            .ok_or_else(|| "distributed-system host must be an object".to_string())?;
        total_capacity += finite_simulation_float(
            host.get("capacity"),
            Some(finite_simulation_float(host.get("cores"), Some(1.0))?),
        )?;
    }
    for link in links {
        let link = link
            .as_object()
            .ok_or_else(|| "distributed-system link must be an object".to_string())?;
        min_bandwidth =
            min_bandwidth.min(finite_simulation_float(link.get("bandwidth"), Some(0.0))?);
    }
    for task in tasks {
        let task = task
            .as_object()
            .ok_or_else(|| "distributed-system task must be an object".to_string())?;
        total_work += finite_simulation_float(
            task.get("work"),
            Some(finite_simulation_float(task.get("duration"), Some(0.0))?),
        )?;
    }
    let mut metrics = BTreeMap::new();
    metrics.insert("hosts".to_string(), hosts.len() as f64);
    metrics.insert("links".to_string(), links.len() as f64);
    metrics.insert("tasks".to_string(), tasks.len() as f64);
    metrics.insert("total_capacity".to_string(), total_capacity);
    metrics.insert(
        "min_bandwidth".to_string(),
        if min_bandwidth.is_finite() {
            min_bandwidth
        } else {
            0.0
        },
    );
    metrics.insert("total_work".to_string(), total_work);
    Ok((
        vec![json!({
            "time": 0.0,
            "event": "distributed_model_loaded",
            "hosts": hosts.len(),
            "tasks": tasks.len(),
        })],
        metrics,
    ))
}

fn simulate_process_flow_with_rust(
    model: &Value,
) -> Result<(Vec<Value>, BTreeMap<String, f64>), String> {
    let model = model
        .as_object()
        .ok_or_else(|| "simulation model must be an object".to_string())?;
    let units = model
        .get("units")
        .and_then(Value::as_array)
        .filter(|units| !units.is_empty())
        .ok_or_else(|| "process-flow model needs non-empty units".to_string())?;
    let streams = model
        .get("streams")
        .and_then(Value::as_array)
        .ok_or_else(|| "process-flow streams must be an array".to_string())?;
    let mut inlet = 0.0;
    let mut outlet = 0.0;
    let mut min_flow = f64::INFINITY;
    for stream in streams {
        let stream = stream
            .as_object()
            .ok_or_else(|| "process-flow stream must be an object".to_string())?;
        let flow = if stream.contains_key("flow") {
            finite_simulation_float(stream.get("flow"), None)?
        } else {
            finite_simulation_float(stream.get("mass_flow"), Some(0.0))?
        };
        min_flow = min_flow.min(flow);
        if stream_endpoint_matches(stream, "to", "sink") {
            outlet += flow;
        }
        if stream_endpoint_matches(stream, "from", "source") {
            inlet += flow;
        }
    }
    let mut metrics = BTreeMap::new();
    metrics.insert("units".to_string(), units.len() as f64);
    metrics.insert("streams".to_string(), streams.len() as f64);
    metrics.insert("inlet_flow".to_string(), inlet);
    metrics.insert("outlet_flow".to_string(), outlet);
    metrics.insert("mass_balance_error".to_string(), (inlet - outlet).abs());
    metrics.insert(
        "min_stream_flow".to_string(),
        if min_flow.is_finite() { min_flow } else { 0.0 },
    );
    Ok((
        vec![json!({
            "time": 0.0,
            "event": "process_model_loaded",
            "units": units.len(),
            "streams": streams.len(),
        })],
        metrics,
    ))
}

fn check_event_trace_property(name: &str, jobs: &[RustSimulationJob]) -> Value {
    let passed = match name {
        "queue_length_never_negative" => jobs.iter().all(|job| job.wait >= -1e-9),
        "departures_after_arrivals" => jobs.iter().all(|job| job.departure + 1e-9 >= job.arrival),
        "service_starts_after_arrivals" => jobs.iter().all(|job| job.start + 1e-9 >= job.arrival),
        "single_station_fcfs" => jobs
            .windows(2)
            .all(|window| window[0].start <= window[1].start + 1e-9),
        _ => {
            return json!({
                "name": name,
                "passed": false,
                "message": "unknown trace property",
            })
        }
    };
    json!({
        "name": name,
        "passed": passed,
        "message": "",
    })
}

fn check_mobility_trace_property(name: &str, vehicles: &[RustMobilityVehicle]) -> Value {
    let passed = match name {
        "departures_before_arrivals" => vehicles
            .iter()
            .all(|vehicle| vehicle.arrival + 1e-9 >= vehicle.depart),
        "travel_times_nonnegative" => vehicles.iter().all(|vehicle| vehicle.travel_time >= -1e-9),
        "vehicles_complete" => vehicles.iter().all(|vehicle| vehicle.arrival.is_finite()),
        _ => {
            return json!({
                "name": name,
                "passed": false,
                "message": "unknown mobility property",
            })
        }
    };
    json!({
        "name": name,
        "passed": passed,
        "message": "",
    })
}

fn check_energy_trace_property(
    name: &str,
    trace: &[Value],
    metrics: &BTreeMap<String, f64>,
) -> Value {
    let passed = match name {
        "energy_nonnegative" => metrics.get("energy_kwh").copied().unwrap_or(0.0) >= -1e-9,
        "temperatures_finite" => trace.iter().all(|event| {
            event
                .get("temperature")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                .is_finite()
        }),
        "temperature_within_bounds" => {
            metrics.get("min_temperature").copied().unwrap_or(0.0) >= -100.0
                && metrics.get("max_temperature").copied().unwrap_or(0.0) <= 100.0
        }
        _ => {
            return json!({
                "name": name,
                "passed": false,
                "message": "unknown energy property",
            })
        }
    };
    json!({
        "name": name,
        "passed": passed,
        "message": "",
    })
}

fn check_physics_trace_property(
    name: &str,
    trace: &[Value],
    metrics: &BTreeMap<String, f64>,
) -> Value {
    let passed = match name {
        "positions_finite" => trace.iter().all(|event| {
            event
                .get("position")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                .is_finite()
        }),
        "velocities_finite" => trace.iter().all(|event| {
            event
                .get("velocity")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                .is_finite()
        }),
        "path_length_nonnegative" => metrics.get("path_length").copied().unwrap_or(0.0) >= -1e-9,
        "stays_above_floor" => {
            let floor = trace
                .iter()
                .filter_map(|event| event.get("position").and_then(Value::as_f64))
                .fold(0.0_f64, f64::min);
            floor >= -1e-9
        }
        _ => {
            return json!({
                "name": name,
                "passed": false,
                "message": "unknown physics property",
            })
        }
    };
    json!({
        "name": name,
        "passed": passed,
        "message": "",
    })
}

fn check_agent_trace_property(name: &str, simulation: &RustAgentSimulation) -> Value {
    let passed = match name {
        "agents_nonempty" => simulation.metrics.get("agents").copied().unwrap_or(0.0) > 0.0,
        "states_present" => {
            simulation
                .metrics
                .get("stateful_agents")
                .copied()
                .unwrap_or(0.0)
                == simulation.metrics.get("agents").copied().unwrap_or(0.0)
        }
        "steps_nonnegative" => simulation.metrics.get("steps").copied().unwrap_or(0.0) >= 0.0,
        "interactions_reference_agents" => {
            let count = simulation.metrics.get("agents").copied().unwrap_or(0.0) as i64;
            simulation.interactions.iter().all(|edge| {
                let Some(edge) = edge.as_object() else {
                    return false;
                };
                let Ok(src) = simulation_i64(edge.get("source").or_else(|| edge.get("from")), -1)
                else {
                    return false;
                };
                let Ok(dst) = simulation_i64(edge.get("target").or_else(|| edge.get("to")), -1)
                else {
                    return false;
                };
                src >= 0 && dst >= 0 && src < count && dst < count
            })
        }
        _ => {
            return json!({
                "name": name,
                "passed": false,
                "message": "unknown agent-based property",
            })
        }
    };
    json!({
        "name": name,
        "passed": passed,
        "message": "",
    })
}

fn check_distributed_trace_property(name: &str, metrics: &BTreeMap<String, f64>) -> Value {
    let passed = match name {
        "hosts_have_capacity" => metrics.get("total_capacity").copied().unwrap_or(0.0) > 0.0,
        "links_nonnegative" => metrics.get("min_bandwidth").copied().unwrap_or(0.0) >= 0.0,
        "tasks_schedulable" => {
            metrics.get("tasks").copied().unwrap_or(0.0) == 0.0
                || metrics.get("total_capacity").copied().unwrap_or(0.0) > 0.0
        }
        _ => {
            return json!({
                "name": name,
                "passed": false,
                "message": "unknown distributed-system property",
            })
        }
    };
    json!({
        "name": name,
        "passed": passed,
        "message": "",
    })
}

fn check_process_trace_property(name: &str, metrics: &BTreeMap<String, f64>) -> Value {
    let passed = match name {
        "units_present" => metrics.get("units").copied().unwrap_or(0.0) > 0.0,
        "streams_nonnegative" => metrics.get("min_stream_flow").copied().unwrap_or(0.0) >= -1e-9,
        "mass_balance_closed" => {
            metrics
                .get("mass_balance_error")
                .copied()
                .unwrap_or(f64::INFINITY)
                <= 1e-9
        }
        _ => {
            return json!({
                "name": name,
                "passed": false,
                "message": "unknown process-flow property",
            })
        }
    };
    json!({
        "name": name,
        "passed": passed,
        "message": "",
    })
}

fn check_simulation_metric_expectation(
    expectation: &Value,
    metrics: &BTreeMap<String, f64>,
) -> Value {
    let name = expectation
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("");
    let comparison = expectation
        .get("comparison")
        .and_then(Value::as_str)
        .unwrap_or("within-absolute");
    let target = match finite_simulation_float(expectation.get("target"), Some(0.0)) {
        Ok(target) => target,
        Err(message) => {
            return json!({
                "name": name,
                "passed": false,
                "actual": null,
                "target": 0.0,
                "message": message,
            })
        }
    };
    let tolerance = match finite_simulation_float(expectation.get("tolerance"), Some(0.0)) {
        Ok(tolerance) => tolerance.abs(),
        Err(message) => {
            return json!({
                "name": name,
                "passed": false,
                "actual": null,
                "target": target,
                "message": message,
            })
        }
    };
    let Some(actual) = metrics.get(name).copied() else {
        return json!({
            "name": name,
            "passed": false,
            "actual": null,
            "target": target,
            "message": "metric missing",
        });
    };
    let passed = match comparison {
        "within-absolute" => (actual - target).abs() <= tolerance,
        "less-equal" | "at-most" | "<=" => actual <= target + tolerance,
        "greater-equal" | "at-least" | ">=" => actual + tolerance >= target,
        "equal" | "==" => (actual - target).abs() <= tolerance,
        _ => {
            return json!({
                "name": name,
                "passed": false,
                "actual": actual,
                "target": target,
                "message": format!("unknown comparison {comparison:?}"),
            })
        }
    };
    json!({
        "name": name,
        "passed": passed,
        "actual": actual,
        "target": target,
        "tolerance": tolerance,
        "comparison": comparison,
        "message": "",
    })
}

fn simulation_reference_run(
    engine_id: String,
    simulator: String,
    status: ExternalSimulationValidationStatus,
    verdict: ExternalSimulationValidationVerdict,
    message: String,
    metrics: BTreeMap<String, f64>,
    checks: Vec<Value>,
    trace: Vec<Value>,
    elapsed_ms: f64,
) -> ExternalSimulationValidationReferenceRun {
    let raw = json!({
        "status": status.as_str(),
        "verdict": verdict.as_str(),
        "simulator": simulator,
        "message": message,
        "metrics": metrics,
        "checks": checks,
        "trace": trace,
    });
    ExternalSimulationValidationReferenceRun {
        engine_id,
        simulator: raw["simulator"]
            .as_str()
            .unwrap_or("simulation")
            .to_string(),
        status,
        verdict,
        metrics: raw["metrics"]
            .as_object()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|(key, value)| value.as_f64().map(|value| (key.clone(), value)))
                    .collect()
            })
            .unwrap_or_default(),
        checks: raw["checks"].as_array().cloned().unwrap_or_default(),
        trace: raw["trace"].as_array().cloned().unwrap_or_default(),
        message: raw["message"].as_str().unwrap_or("").to_string(),
        raw,
        elapsed_ms,
    }
}

fn failed_rust_simulation_reference_run(
    engine_id: String,
    message: String,
    elapsed_ms: f64,
) -> ExternalSimulationValidationReferenceRun {
    simulation_reference_run(
        engine_id.clone(),
        engine_id,
        ExternalSimulationValidationStatus::Failed,
        ExternalSimulationValidationVerdict::Failure,
        message,
        BTreeMap::new(),
        Vec::new(),
        Vec::new(),
        elapsed_ms,
    )
}

fn run_event_simulation_validation_with_rust_reference(
    payload: &Value,
    engine_id: String,
    started: Instant,
) -> ExternalSimulationValidationReferenceRun {
    let engine = engine_id.to_ascii_lowercase();
    let default_model = json!({});
    let model = payload.get("model").unwrap_or(&default_model);
    let (jobs, trace, metrics) = match simulate_event_network_with_rust(model) {
        Ok(simulation) => simulation,
        Err(message) => {
            return failed_rust_simulation_reference_run(
                engine_id,
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let simulator = if EVENT_SIMULATION_ENGINES.contains(&engine.as_str()) {
        format!("rust:single-station-des-for-{engine}")
    } else {
        "rust:single-station-des".to_string()
    };

    let mut checks = Vec::new();
    if let Some(properties) = payload
        .get("expected_trace_properties")
        .and_then(Value::as_array)
    {
        for property in properties {
            checks.push(check_event_trace_property(
                property.as_str().unwrap_or(""),
                &jobs,
            ));
        }
    }
    if let Some(expectations) = payload.get("metric_expectations").and_then(Value::as_array) {
        for expectation in expectations {
            checks.push(check_simulation_metric_expectation(expectation, &metrics));
        }
    }
    let verdict = if checks
        .iter()
        .all(|check| check.get("passed").and_then(Value::as_bool) == Some(true))
    {
        ExternalSimulationValidationVerdict::Valid
    } else {
        ExternalSimulationValidationVerdict::Invalid
    };
    simulation_reference_run(
        engine_id,
        simulator,
        ExternalSimulationValidationStatus::Ok,
        verdict,
        String::new(),
        metrics,
        checks,
        trace,
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn run_mobility_simulation_validation_with_rust_reference(
    payload: &Value,
    engine_id: String,
    started: Instant,
) -> ExternalSimulationValidationReferenceRun {
    let engine = engine_id.to_ascii_lowercase();
    let default_model = json!({});
    let model = payload.get("model").unwrap_or(&default_model);
    let (vehicles, trace, metrics) = match simulate_mobility_network_with_rust(model) {
        Ok(simulation) => simulation,
        Err(message) => {
            return failed_rust_simulation_reference_run(
                engine_id,
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let simulator = if MOBILITY_SIMULATION_ENGINES.contains(&engine.as_str()) {
        format!("rust:mobility-network-for-{engine}")
    } else {
        "rust:mobility-network".to_string()
    };

    let mut checks = Vec::new();
    if let Some(properties) = payload
        .get("expected_trace_properties")
        .and_then(Value::as_array)
    {
        for property in properties {
            checks.push(check_mobility_trace_property(
                property.as_str().unwrap_or(""),
                &vehicles,
            ));
        }
    }
    if let Some(expectations) = payload.get("metric_expectations").and_then(Value::as_array) {
        for expectation in expectations {
            checks.push(check_simulation_metric_expectation(expectation, &metrics));
        }
    }
    let verdict = if checks
        .iter()
        .all(|check| check.get("passed").and_then(Value::as_bool) == Some(true))
    {
        ExternalSimulationValidationVerdict::Valid
    } else {
        ExternalSimulationValidationVerdict::Invalid
    };
    simulation_reference_run(
        engine_id,
        simulator,
        ExternalSimulationValidationStatus::Ok,
        verdict,
        String::new(),
        metrics,
        checks,
        trace,
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn run_energy_simulation_validation_with_rust_reference(
    payload: &Value,
    engine_id: String,
    started: Instant,
) -> ExternalSimulationValidationReferenceRun {
    let engine = engine_id.to_ascii_lowercase();
    let default_model = json!({});
    let default_scenario = json!({});
    let model = payload.get("model").unwrap_or(&default_model);
    let scenario = payload.get("scenario").unwrap_or(&default_scenario);
    let (trace, metrics) = match simulate_energy_balance_with_rust(model, scenario) {
        Ok(simulation) => simulation,
        Err(message) => {
            return failed_rust_simulation_reference_run(
                engine_id,
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let simulator = if ENERGY_SIMULATION_ENGINES.contains(&engine.as_str()) {
        format!("rust:energy-balance-for-{engine}")
    } else {
        "rust:energy-balance".to_string()
    };
    let mut checks = Vec::new();
    if let Some(properties) = payload
        .get("expected_trace_properties")
        .and_then(Value::as_array)
    {
        for property in properties {
            checks.push(check_energy_trace_property(
                property.as_str().unwrap_or(""),
                &trace,
                &metrics,
            ));
        }
    }
    if let Some(expectations) = payload.get("metric_expectations").and_then(Value::as_array) {
        for expectation in expectations {
            checks.push(check_simulation_metric_expectation(expectation, &metrics));
        }
    }
    let verdict = if checks
        .iter()
        .all(|check| check.get("passed").and_then(Value::as_bool) == Some(true))
    {
        ExternalSimulationValidationVerdict::Valid
    } else {
        ExternalSimulationValidationVerdict::Invalid
    };
    simulation_reference_run(
        engine_id,
        simulator,
        ExternalSimulationValidationStatus::Ok,
        verdict,
        String::new(),
        metrics,
        checks,
        trace,
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn run_physics_simulation_validation_with_rust_reference(
    payload: &Value,
    engine_id: String,
    started: Instant,
) -> ExternalSimulationValidationReferenceRun {
    let engine = engine_id.to_ascii_lowercase();
    let default_model = json!({});
    let default_scenario = json!({});
    let model = payload.get("model").unwrap_or(&default_model);
    let scenario = payload.get("scenario").unwrap_or(&default_scenario);
    let (trace, metrics) = match simulate_physics_trajectory_with_rust(model, scenario) {
        Ok(simulation) => simulation,
        Err(message) => {
            return failed_rust_simulation_reference_run(
                engine_id,
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let simulator = if PHYSICS_SIMULATION_ENGINES.contains(&engine.as_str()) {
        format!("rust:physics-trajectory-for-{engine}")
    } else {
        "rust:physics-trajectory".to_string()
    };
    let mut checks = Vec::new();
    if let Some(properties) = payload
        .get("expected_trace_properties")
        .and_then(Value::as_array)
    {
        for property in properties {
            checks.push(check_physics_trace_property(
                property.as_str().unwrap_or(""),
                &trace,
                &metrics,
            ));
        }
    }
    if let Some(expectations) = payload.get("metric_expectations").and_then(Value::as_array) {
        for expectation in expectations {
            checks.push(check_simulation_metric_expectation(expectation, &metrics));
        }
    }
    let verdict = if checks
        .iter()
        .all(|check| check.get("passed").and_then(Value::as_bool) == Some(true))
    {
        ExternalSimulationValidationVerdict::Valid
    } else {
        ExternalSimulationValidationVerdict::Invalid
    };
    simulation_reference_run(
        engine_id,
        simulator,
        ExternalSimulationValidationStatus::Ok,
        verdict,
        String::new(),
        metrics,
        checks,
        trace,
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn run_agent_simulation_validation_with_rust_reference(
    payload: &Value,
    engine_id: String,
    started: Instant,
) -> ExternalSimulationValidationReferenceRun {
    let engine = engine_id.to_ascii_lowercase();
    let default_model = json!({});
    let default_scenario = json!({});
    let model = payload.get("model").unwrap_or(&default_model);
    let scenario = payload.get("scenario").unwrap_or(&default_scenario);
    let simulation = match simulate_agent_based_with_rust(model, scenario) {
        Ok(simulation) => simulation,
        Err(message) => {
            return failed_rust_simulation_reference_run(
                engine_id,
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let simulator = if AGENT_SIMULATION_ENGINES.contains(&engine.as_str()) {
        format!("rust:agent-based-for-{engine}")
    } else {
        "rust:agent-based".to_string()
    };
    let mut checks = Vec::new();
    if let Some(properties) = payload
        .get("expected_trace_properties")
        .and_then(Value::as_array)
    {
        for property in properties {
            checks.push(check_agent_trace_property(
                property.as_str().unwrap_or(""),
                &simulation,
            ));
        }
    }
    if let Some(expectations) = payload.get("metric_expectations").and_then(Value::as_array) {
        for expectation in expectations {
            checks.push(check_simulation_metric_expectation(
                expectation,
                &simulation.metrics,
            ));
        }
    }
    let verdict = if checks
        .iter()
        .all(|check| check.get("passed").and_then(Value::as_bool) == Some(true))
    {
        ExternalSimulationValidationVerdict::Valid
    } else {
        ExternalSimulationValidationVerdict::Invalid
    };
    simulation_reference_run(
        engine_id,
        simulator,
        ExternalSimulationValidationStatus::Ok,
        verdict,
        String::new(),
        simulation.metrics,
        checks,
        simulation.trace,
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn run_distributed_simulation_validation_with_rust_reference(
    payload: &Value,
    engine_id: String,
    started: Instant,
) -> ExternalSimulationValidationReferenceRun {
    let engine = engine_id.to_ascii_lowercase();
    let default_model = json!({});
    let model = payload.get("model").unwrap_or(&default_model);
    let (trace, metrics) = match simulate_distributed_system_with_rust(model) {
        Ok(simulation) => simulation,
        Err(message) => {
            return failed_rust_simulation_reference_run(
                engine_id,
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let simulator = if DISTRIBUTED_SIMULATION_ENGINES.contains(&engine.as_str()) {
        format!("rust:distributed-system-for-{engine}")
    } else {
        "rust:distributed-system".to_string()
    };
    let mut checks = Vec::new();
    if let Some(properties) = payload
        .get("expected_trace_properties")
        .and_then(Value::as_array)
    {
        for property in properties {
            checks.push(check_distributed_trace_property(
                property.as_str().unwrap_or(""),
                &metrics,
            ));
        }
    }
    if let Some(expectations) = payload.get("metric_expectations").and_then(Value::as_array) {
        for expectation in expectations {
            checks.push(check_simulation_metric_expectation(expectation, &metrics));
        }
    }
    let verdict = if checks
        .iter()
        .all(|check| check.get("passed").and_then(Value::as_bool) == Some(true))
    {
        ExternalSimulationValidationVerdict::Valid
    } else {
        ExternalSimulationValidationVerdict::Invalid
    };
    simulation_reference_run(
        engine_id,
        simulator,
        ExternalSimulationValidationStatus::Ok,
        verdict,
        String::new(),
        metrics,
        checks,
        trace,
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn run_process_simulation_validation_with_rust_reference(
    payload: &Value,
    engine_id: String,
    started: Instant,
) -> ExternalSimulationValidationReferenceRun {
    let engine = engine_id.to_ascii_lowercase();
    let default_model = json!({});
    let model = payload.get("model").unwrap_or(&default_model);
    let (trace, metrics) = match simulate_process_flow_with_rust(model) {
        Ok(simulation) => simulation,
        Err(message) => {
            return failed_rust_simulation_reference_run(
                engine_id,
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let simulator = if PROCESS_SIMULATION_ENGINES.contains(&engine.as_str()) {
        format!("rust:process-flow-for-{engine}")
    } else {
        "rust:process-flow".to_string()
    };
    let mut checks = Vec::new();
    if let Some(properties) = payload
        .get("expected_trace_properties")
        .and_then(Value::as_array)
    {
        for property in properties {
            checks.push(check_process_trace_property(
                property.as_str().unwrap_or(""),
                &metrics,
            ));
        }
    }
    if let Some(expectations) = payload.get("metric_expectations").and_then(Value::as_array) {
        for expectation in expectations {
            checks.push(check_simulation_metric_expectation(expectation, &metrics));
        }
    }
    let verdict = if checks
        .iter()
        .all(|check| check.get("passed").and_then(Value::as_bool) == Some(true))
    {
        ExternalSimulationValidationVerdict::Valid
    } else {
        ExternalSimulationValidationVerdict::Invalid
    };
    simulation_reference_run(
        engine_id,
        simulator,
        ExternalSimulationValidationStatus::Ok,
        verdict,
        String::new(),
        metrics,
        checks,
        trace,
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

pub fn run_simulation_validation_with_external_reference(
    request: &SimulationValidationRequest,
    options: &ExternalSimulationValidationReferenceOptions,
) -> ExternalSimulationValidationReferenceRun {
    let payload = simulation_validation_request_to_json(request);
    run_simulation_validation_json_with_external_reference(&payload, options)
}

pub fn run_simulation_validation_json_with_external_reference(
    payload: &Value,
    options: &ExternalSimulationValidationReferenceOptions,
) -> ExternalSimulationValidationReferenceRun {
    let engine_id = options
        .engine_id
        .clone()
        .or_else(|| {
            payload
                .get("engine")
                .or_else(|| payload.get("engine_id"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "builtin".to_string());
    let started = Instant::now();
    let model_format = payload
        .get("model_format")
        .and_then(Value::as_str)
        .unwrap_or("json-event-network");
    if model_format == "json-event-network" {
        return run_event_simulation_validation_with_rust_reference(payload, engine_id, started);
    } else if model_format == "json-mobility-network" {
        return run_mobility_simulation_validation_with_rust_reference(payload, engine_id, started);
    } else if model_format == "json-energy-balance" {
        return run_energy_simulation_validation_with_rust_reference(payload, engine_id, started);
    } else if model_format == "json-physics-trajectory" {
        return run_physics_simulation_validation_with_rust_reference(payload, engine_id, started);
    } else if model_format == "json-agent-based" {
        return run_agent_simulation_validation_with_rust_reference(payload, engine_id, started);
    } else if model_format == "json-distributed-system" {
        return run_distributed_simulation_validation_with_rust_reference(
            payload, engine_id, started,
        );
    } else if model_format == "json-process-flow" {
        return run_process_simulation_validation_with_rust_reference(payload, engine_id, started);
    }

    let python = env::var_os("PYTHON_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"));
    let script = external_simulation_validation_reference_script();
    let working_dir = script
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    let mut command = Command::new(python);
    if let Some(working_dir) = working_dir {
        command.current_dir(working_dir);
    }
    command.arg(script).arg("--engine").arg(&engine_id);

    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            return ExternalSimulationValidationReferenceRun {
                engine_id,
                simulator: "unavailable".to_string(),
                status: ExternalSimulationValidationStatus::Unavailable,
                verdict: ExternalSimulationValidationVerdict::Unknown,
                metrics: BTreeMap::new(),
                checks: Vec::new(),
                trace: Vec::new(),
                raw: json!({"status": "unavailable", "verdict": "unknown", "message": e.to_string()}),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: e.to_string(),
            }
        }
    };

    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(e) = stdin.write_all(payload.to_string().as_bytes()) {
            return ExternalSimulationValidationReferenceRun {
                engine_id,
                simulator: "failed".to_string(),
                status: ExternalSimulationValidationStatus::Failed,
                verdict: ExternalSimulationValidationVerdict::Failure,
                metrics: BTreeMap::new(),
                checks: Vec::new(),
                trace: Vec::new(),
                raw: json!({"status": "failed", "verdict": "failure", "message": e.to_string()}),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: e.to_string(),
            };
        }
    }

    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(e) => {
            return ExternalSimulationValidationReferenceRun {
                engine_id,
                simulator: "failed".to_string(),
                status: ExternalSimulationValidationStatus::Failed,
                verdict: ExternalSimulationValidationVerdict::Failure,
                metrics: BTreeMap::new(),
                checks: Vec::new(),
                trace: Vec::new(),
                raw: json!({"status": "failed", "verdict": "failure", "message": e.to_string()}),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: e.to_string(),
            }
        }
    };

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let raw = match serde_json::from_str::<Value>(stdout.trim()) {
        Ok(raw) => raw,
        Err(e) => {
            return ExternalSimulationValidationReferenceRun {
                engine_id,
                simulator: "failed".to_string(),
                status: ExternalSimulationValidationStatus::Failed,
                verdict: ExternalSimulationValidationVerdict::Failure,
                metrics: BTreeMap::new(),
                checks: Vec::new(),
                trace: Vec::new(),
                raw: json!({
                    "status": "failed",
                    "verdict": "failure",
                    "stdout": stdout.trim(),
                    "stderr": stderr,
                    "message": e.to_string(),
                }),
                elapsed_ms,
                message: e.to_string(),
            };
        }
    };
    let parsed = serde_json::from_value::<SimulationValidationReferenceOutput>(raw.clone()).ok();
    let status = parsed
        .as_ref()
        .map(|parsed| ExternalSimulationValidationStatus::from_label(&parsed.status))
        .unwrap_or(ExternalSimulationValidationStatus::Unknown);
    let verdict = parsed
        .as_ref()
        .map(|parsed| ExternalSimulationValidationVerdict::from_label(&parsed.verdict))
        .unwrap_or(ExternalSimulationValidationVerdict::Unknown);
    let message = parsed
        .as_ref()
        .map(|parsed| parsed.message.clone())
        .filter(|message| !message.is_empty())
        .unwrap_or(stderr);

    ExternalSimulationValidationReferenceRun {
        engine_id,
        simulator: parsed
            .as_ref()
            .map(|parsed| parsed.simulator.clone())
            .filter(|simulator| !simulator.is_empty())
            .unwrap_or_else(|| "simulation".to_string()),
        status,
        verdict,
        metrics: parsed
            .as_ref()
            .map(|parsed| parsed.metrics.clone())
            .unwrap_or_default(),
        checks: parsed
            .as_ref()
            .map(|parsed| parsed.checks.clone())
            .unwrap_or_default(),
        trace: parsed
            .as_ref()
            .map(|parsed| parsed.trace.clone())
            .unwrap_or_default(),
        raw,
        elapsed_ms,
        message,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalBenchmarkManifestEntry {
    pub name: String,
    pub family: String,
    pub format: String,
    pub path: PathBuf,
    pub objective_sense: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalBenchmarkManifest {
    pub suite: String,
    pub version: Option<String>,
    pub entries: Vec<ExternalBenchmarkManifestEntry>,
}

pub fn external_benchmark_manifest_to_json(manifest: &ExternalBenchmarkManifest) -> Value {
    let entries: Vec<Value> = manifest
        .entries
        .iter()
        .map(|entry| {
            json!({
                "name": &entry.name,
                "family": &entry.family,
                "format": &entry.format,
                "path": entry.path.to_string_lossy(),
                "objective_sense": &entry.objective_sense,
                "tags": &entry.tags,
            })
        })
        .collect();
    json!({
        "kind": "external-benchmark-manifest",
        "suite": &manifest.suite,
        "version": &manifest.version,
        "entries": entries,
    })
}

pub fn external_validation_default_text_cli_args(
    tool_id: &str,
    input_format: ExternalValidationTextFormat,
) -> &'static [&'static str] {
    let normalized = normalize_tool_id(tool_id);
    match (normalized.as_str(), input_format) {
        ("z3", ExternalValidationTextFormat::SmtLib2) => &["-in", "-smt2"],
        ("cvc5", ExternalValidationTextFormat::SmtLib2) => &["--lang=smt2", "-"],
        ("bitwuzla", ExternalValidationTextFormat::SmtLib2) => &["--smt2", "-"],
        ("boolector", ExternalValidationTextFormat::SmtLib2) => &["--smt2", "-"],
        ("mathsat" | "optimathsat", ExternalValidationTextFormat::SmtLib2) => &["-input=smt2"],
        ("opensmt", ExternalValidationTextFormat::SmtLib2) => &["--smt2", "-"],
        (
            "kissat" | "cadical" | "cryptominisat" | "minisat" | "glucose" | "maplesat" | "varisat"
            | "open-wbo" | "maxhs" | "roundingsat",
            ExternalValidationTextFormat::DimacsCnf | ExternalValidationTextFormat::DimacsWcnf,
        ) => &["-"],
        (
            "minizinc",
            ExternalValidationTextFormat::MiniZinc | ExternalValidationTextFormat::FlatZinc,
        ) => &["-"],
        ("flatzinc" | "gecode" | "chuffed", ExternalValidationTextFormat::FlatZinc) => &["-"],
        _ => &[],
    }
}

pub fn external_validation_default_file_cli_args(
    tool_id: &str,
    input_format: ExternalValidationTextFormat,
    input_path: &Path,
) -> Vec<String> {
    let normalized = normalize_tool_id(tool_id);
    let path = input_path.to_string_lossy().to_string();
    match (normalized.as_str(), input_format) {
        ("z3", ExternalValidationTextFormat::SmtLib2) => vec!["-smt2".to_string(), path],
        ("cvc5", ExternalValidationTextFormat::SmtLib2) => {
            vec!["--lang=smt2".to_string(), path]
        }
        ("yices", ExternalValidationTextFormat::SmtLib2) => vec![path],
        ("bitwuzla", ExternalValidationTextFormat::SmtLib2) => {
            vec!["--smt2".to_string(), path]
        }
        ("boolector", ExternalValidationTextFormat::SmtLib2) => {
            vec!["--smt2".to_string(), path]
        }
        ("mathsat" | "optimathsat", ExternalValidationTextFormat::SmtLib2) => {
            vec!["-input=smt2".to_string(), path]
        }
        ("opensmt", ExternalValidationTextFormat::SmtLib2) => {
            vec!["--smt2".to_string(), path]
        }
        (
            "kissat" | "cadical" | "cryptominisat" | "minisat" | "glucose" | "maplesat" | "varisat"
            | "open-wbo" | "maxhs" | "roundingsat",
            ExternalValidationTextFormat::DimacsCnf | ExternalValidationTextFormat::DimacsWcnf,
        ) => vec![path],
        (
            "minizinc",
            ExternalValidationTextFormat::MiniZinc | ExternalValidationTextFormat::FlatZinc,
        ) => vec![path],
        ("flatzinc" | "gecode" | "chuffed", ExternalValidationTextFormat::FlatZinc) => vec![path],
        ("tlc" | "apalache", ExternalValidationTextFormat::TlaPlus) => vec![path],
        ("prism" | "storm", ExternalValidationTextFormat::PrismModel) => vec![path],
        ("json-schema", ExternalValidationTextFormat::Json) => vec![path],
        _ => Vec::new(),
    }
}

pub fn infer_external_validation_text_verdict(
    input_format: ExternalValidationTextFormat,
    stdout: &str,
    stderr: &str,
    exit_success: bool,
) -> ExternalValidationTextVerdict {
    let first_token = stdout
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .and_then(|line| line.split_whitespace().next())
        .unwrap_or("")
        .trim_matches(|ch: char| ch == '=' || ch == ';')
        .to_ascii_lowercase();
    if matches!(
        input_format,
        ExternalValidationTextFormat::SmtLib2
            | ExternalValidationTextFormat::DimacsCnf
            | ExternalValidationTextFormat::DimacsWcnf
    ) {
        return match first_token.as_str() {
            "sat" | "satisfiable" => ExternalValidationTextVerdict::Sat,
            "unsat" | "unsatisfiable" => ExternalValidationTextVerdict::Unsat,
            "unknown" => ExternalValidationTextVerdict::Unknown,
            _ if exit_success => ExternalValidationTextVerdict::Success,
            _ => ExternalValidationTextVerdict::Failure,
        };
    }

    let combined = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    if !exit_success {
        return ExternalValidationTextVerdict::Failure;
    }
    if combined.contains("=====unsatisfiable=====")
        || combined.contains("unsatisfiable")
        || combined.contains("no solution")
    {
        return ExternalValidationTextVerdict::Unsat;
    }
    if combined.contains("unknown") {
        return ExternalValidationTextVerdict::Unknown;
    }
    if combined.contains("counterexample")
        || combined.contains("violated")
        || combined.contains("violation")
        || combined.contains("invalid")
    {
        return ExternalValidationTextVerdict::Invalid;
    }
    if combined.contains("satisfied")
        || combined.contains("no error")
        || combined.contains("valid")
        || combined.contains("true")
    {
        return ExternalValidationTextVerdict::Valid;
    }
    ExternalValidationTextVerdict::Success
}

pub fn external_validation_text_cli_command(
    opts: &ExternalValidationTextCliOptions,
) -> Option<PathBuf> {
    let tool = find_external_validation_tool(&opts.tool_id)?;
    opts.command_path
        .as_ref()
        .cloned()
        .or_else(|| external_validation_adapter_command(tool))
}

pub fn run_external_validation_text_cli(
    input_text: &str,
    opts: &ExternalValidationTextCliOptions,
) -> ExternalValidationRun {
    let Some(tool) = find_external_validation_tool(&opts.tool_id) else {
        return ExternalValidationRun {
            tool_id: opts.tool_id.clone(),
            status: ExternalValidationRunStatus::Unavailable,
            output: None,
            elapsed_ms: 0.0,
            message: format!("unknown external validation tool '{}'", opts.tool_id),
        };
    };
    let Some(command) = external_validation_text_cli_command(opts) else {
        return ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Unavailable,
            output: None,
            elapsed_ms: 0.0,
            message: format!(
                "{} text CLI command not configured; set {}",
                tool.display_name,
                external_validation_adapter_env_names(tool)[0]
            ),
        };
    };

    let mut args: Vec<String> = if opts.use_default_args {
        external_validation_default_text_cli_args(tool.id, opts.input_format)
            .iter()
            .map(|arg| (*arg).to_string())
            .collect()
    } else {
        Vec::new()
    };
    args.extend(opts.extra_args.iter().cloned());

    let started = Instant::now();
    let mut child = match Command::new(&command)
        .args(&args)
        .env("ORES_EXTERNAL_VALIDATION_TOOL", tool.id)
        .env(
            "ORES_EXTERNAL_VALIDATION_FORMAT",
            opts.input_format.as_str(),
        )
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
            return ExternalValidationRun {
                tool_id: tool.id.to_string(),
                status: ExternalValidationRunStatus::Unavailable,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: err.to_string(),
            };
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(input_text.as_bytes()) {
            return ExternalValidationRun {
                tool_id: tool.id.to_string(),
                status: ExternalValidationRunStatus::Failed,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: err.to_string(),
            };
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return ExternalValidationRun {
                tool_id: tool.id.to_string(),
                status: ExternalValidationRunStatus::Failed,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: err.to_string(),
            };
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let verdict = infer_external_validation_text_verdict(
        opts.input_format,
        &stdout,
        &stderr,
        output.status.success(),
    );
    let payload = json!({
        "kind": "external-validation-text-cli-run",
        "tool": tool.id,
        "format": opts.input_format.as_str(),
        "command": command.to_string_lossy(),
        "args": args,
        "exit_success": output.status.success(),
        "exit_code": output.status.code(),
        "verdict": verdict.as_str(),
        "stdout": stdout,
        "stderr": stderr,
    });
    if output.status.success() {
        ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Ok,
            output: Some(payload),
            elapsed_ms,
            message: String::new(),
        }
    } else {
        let message = stderr
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("external validation text CLI exited unsuccessfully")
            .to_string();
        ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Failed,
            output: Some(payload),
            elapsed_ms,
            message,
        }
    }
}

pub fn external_validation_file_cli_command(
    opts: &ExternalValidationFileCliOptions,
) -> Option<PathBuf> {
    let tool = find_external_validation_tool(&opts.tool_id)?;
    opts.command_path
        .as_ref()
        .cloned()
        .or_else(|| external_validation_adapter_command(tool))
}

pub fn external_validation_file_cli_args(
    opts: &ExternalValidationFileCliOptions,
    input_path: &Path,
) -> Vec<String> {
    let path = input_path.to_string_lossy().to_string();
    let mut used_input_path = false;
    let mut args = Vec::new();
    if opts.use_default_args {
        let default_args =
            external_validation_default_file_cli_args(&opts.tool_id, opts.input_format, input_path);
        used_input_path = !default_args.is_empty();
        args.extend(default_args);
    }
    for arg in &opts.extra_args {
        if arg.contains("{input}") {
            used_input_path = true;
            args.push(arg.replace("{input}", &path));
        } else {
            args.push(arg.clone());
        }
    }
    if opts.append_input_path && !used_input_path {
        args.push(path);
    }
    args
}

pub fn run_external_validation_file_cli(
    input_text: &str,
    opts: &ExternalValidationFileCliOptions,
) -> ExternalValidationRun {
    let Some(tool) = find_external_validation_tool(&opts.tool_id) else {
        return ExternalValidationRun {
            tool_id: opts.tool_id.clone(),
            status: ExternalValidationRunStatus::Unavailable,
            output: None,
            elapsed_ms: 0.0,
            message: format!("unknown external validation tool '{}'", opts.tool_id),
        };
    };
    let Some(command) = external_validation_file_cli_command(opts) else {
        return ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Unavailable,
            output: None,
            elapsed_ms: 0.0,
            message: format!(
                "{} file CLI command not configured; set {}",
                tool.display_name,
                external_validation_adapter_env_names(tool)[0]
            ),
        };
    };

    let extension = opts
        .file_extension
        .as_deref()
        .unwrap_or_else(|| opts.input_format.file_extension())
        .trim_start_matches('.');
    let input_path = env::temp_dir().join(format!(
        "ores-validation-{}.{}",
        uuid::Uuid::new_v4().simple(),
        extension
    ));
    let started = Instant::now();
    if let Err(err) = fs::write(&input_path, input_text) {
        return ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Failed,
            output: None,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            message: format!("failed to write validation temp file: {err}"),
        };
    }
    let args = external_validation_file_cli_args(opts, &input_path);
    let output = Command::new(&command)
        .args(&args)
        .env("ORES_EXTERNAL_VALIDATION_TOOL", tool.id)
        .env(
            "ORES_EXTERNAL_VALIDATION_FORMAT",
            opts.input_format.as_str(),
        )
        .current_dir(
            opts.working_dir
                .as_deref()
                .unwrap_or_else(|| Path::new(".")),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let remove_result = fs::remove_file(&input_path);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let output = match output {
        Ok(output) => output,
        Err(err) => {
            return ExternalValidationRun {
                tool_id: tool.id.to_string(),
                status: ExternalValidationRunStatus::Unavailable,
                output: None,
                elapsed_ms,
                message: err.to_string(),
            };
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let verdict = infer_external_validation_text_verdict(
        opts.input_format,
        &stdout,
        &stderr,
        output.status.success(),
    );
    let payload = json!({
        "kind": "external-validation-file-cli-run",
        "tool": tool.id,
        "format": opts.input_format.as_str(),
        "command": command.to_string_lossy(),
        "args": args,
        "input_path": input_path.to_string_lossy(),
        "temp_file_removed": remove_result.is_ok(),
        "exit_success": output.status.success(),
        "exit_code": output.status.code(),
        "verdict": verdict.as_str(),
        "stdout": stdout,
        "stderr": stderr,
    });
    if output.status.success() {
        ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Ok,
            output: Some(payload),
            elapsed_ms,
            message: String::new(),
        }
    } else {
        let message = stderr
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("external validation file CLI exited unsuccessfully")
            .to_string();
        ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Failed,
            output: Some(payload),
            elapsed_ms,
            message,
        }
    }
}

pub fn external_validation_artifact_cli_command(
    opts: &ExternalValidationArtifactCliOptions,
) -> Option<PathBuf> {
    let tool = find_external_validation_tool(&opts.tool_id)?;
    opts.command_path
        .as_ref()
        .cloned()
        .or_else(|| external_validation_adapter_command(tool))
}

pub fn external_validation_default_artifact_cli_args(
    tool_id: &str,
    input_format: ExternalValidationTextFormat,
    artifact_paths: &BTreeMap<String, PathBuf>,
) -> Vec<String> {
    let normalized = normalize_tool_id(tool_id);
    match (normalized.as_str(), input_format) {
        ("z3", ExternalValidationTextFormat::SmtLib2) => {
            artifact_path_by_any(artifact_paths, &["model", "input", "smt2", "script"])
                .map(|path| vec!["-smt2".to_string(), path])
                .unwrap_or_default()
        }
        (
            "drat-trim" | "lrat-check",
            ExternalValidationTextFormat::DimacsCnf | ExternalValidationTextFormat::DimacsWcnf,
        ) => {
            let Some(cnf) = artifact_path_by_any(artifact_paths, &["cnf", "model", "input"]) else {
                return Vec::new();
            };
            let Some(proof) = artifact_path_by_any(artifact_paths, &["proof", "drat", "lrat"])
            else {
                return Vec::new();
            };
            vec![cnf, proof]
        }
        (
            "minizinc",
            ExternalValidationTextFormat::MiniZinc | ExternalValidationTextFormat::FlatZinc,
        ) => {
            let Some(model) = artifact_path_by_any(artifact_paths, &["model", "mzn", "fzn"]) else {
                return Vec::new();
            };
            let mut args = vec![model];
            if let Some(data) = artifact_path_by_any(artifact_paths, &["data", "dzn"]) {
                args.push(data);
            }
            args
        }
        ("minizinc-solution-checker", ExternalValidationTextFormat::MiniZinc) => {
            let mut args = Vec::new();
            for keys in [
                &["model", "checker", "mzn"][..],
                &["data", "dzn"][..],
                &["solution", "output", "sol"][..],
            ] {
                if let Some(path) = artifact_path_by_any(artifact_paths, keys) {
                    args.push(path);
                }
            }
            args
        }
        ("prism", ExternalValidationTextFormat::PrismModel) => {
            let Some(model) = artifact_path_by_any(artifact_paths, &["model", "prism", "pm"])
            else {
                return Vec::new();
            };
            let mut args = vec![model];
            if let Some(properties) =
                artifact_path_by_any(artifact_paths, &["properties", "property", "props", "pctl"])
            {
                args.push(properties);
            }
            args
        }
        ("storm", ExternalValidationTextFormat::PrismModel) => {
            let Some(model) = artifact_path_by_any(artifact_paths, &["model", "prism", "pm"])
            else {
                return Vec::new();
            };
            let mut args = vec!["--prism".to_string(), model];
            if let Some(properties) =
                artifact_path_by_any(artifact_paths, &["properties", "property", "props", "pctl"])
            {
                args.push("--prop".to_string());
                args.push(properties);
            }
            args
        }
        _ => Vec::new(),
    }
}

pub fn external_validation_artifact_cli_args(
    opts: &ExternalValidationArtifactCliOptions,
    artifact_paths: &BTreeMap<String, PathBuf>,
) -> Vec<String> {
    let mut args = if opts.use_default_args {
        external_validation_default_artifact_cli_args(
            &opts.tool_id,
            opts.input_format,
            artifact_paths,
        )
    } else {
        Vec::new()
    };
    args.extend(
        opts.extra_args
            .iter()
            .map(|arg| replace_artifact_placeholders(arg, artifact_paths)),
    );
    args
}

pub fn run_external_validation_artifact_cli(
    artifacts: &[ExternalValidationArtifact],
    opts: &ExternalValidationArtifactCliOptions,
) -> ExternalValidationRun {
    let Some(tool) = find_external_validation_tool(&opts.tool_id) else {
        return ExternalValidationRun {
            tool_id: opts.tool_id.clone(),
            status: ExternalValidationRunStatus::Unavailable,
            output: None,
            elapsed_ms: 0.0,
            message: format!("unknown external validation tool '{}'", opts.tool_id),
        };
    };
    let Some(command) = external_validation_artifact_cli_command(opts) else {
        return ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Unavailable,
            output: None,
            elapsed_ms: 0.0,
            message: format!(
                "{} artifact CLI command not configured; set {}",
                tool.display_name,
                external_validation_adapter_env_names(tool)[0]
            ),
        };
    };

    let started = Instant::now();
    let temp_dir = env::temp_dir().join(format!(
        "ores-validation-artifacts-{}",
        uuid::Uuid::new_v4().simple()
    ));
    if let Err(err) = fs::create_dir(&temp_dir) {
        return ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Failed,
            output: None,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            message: format!("failed to create validation artifact temp dir: {err}"),
        };
    }

    let mut artifact_paths = BTreeMap::new();
    let mut written_paths = Vec::new();
    for artifact in artifacts {
        let file_name = validation_artifact_file_name(artifact, opts.input_format);
        let path = temp_dir.join(file_name);
        if let Err(err) = fs::write(&path, &artifact.contents) {
            cleanup_validation_artifact_workspace(&written_paths, &temp_dir);
            return ExternalValidationRun {
                tool_id: tool.id.to_string(),
                status: ExternalValidationRunStatus::Failed,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: format!(
                    "failed to write validation artifact '{}': {err}",
                    artifact.key
                ),
            };
        }
        written_paths.push(path.clone());
        artifact_paths.insert(artifact.key.clone(), path);
    }

    let args = external_validation_artifact_cli_args(opts, &artifact_paths);
    let output = Command::new(&command)
        .args(&args)
        .env("ORES_EXTERNAL_VALIDATION_TOOL", tool.id)
        .env(
            "ORES_EXTERNAL_VALIDATION_FORMAT",
            opts.input_format.as_str(),
        )
        .current_dir(
            opts.working_dir
                .as_deref()
                .unwrap_or_else(|| temp_dir.as_path()),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let cleanup_ok = cleanup_validation_artifact_workspace(&written_paths, &temp_dir);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let output = match output {
        Ok(output) => output,
        Err(err) => {
            return ExternalValidationRun {
                tool_id: tool.id.to_string(),
                status: ExternalValidationRunStatus::Unavailable,
                output: None,
                elapsed_ms,
                message: err.to_string(),
            };
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let verdict = infer_external_validation_text_verdict(
        opts.input_format,
        &stdout,
        &stderr,
        output.status.success(),
    );
    let artifact_payload: Vec<Value> = artifact_paths
        .iter()
        .map(|(key, path)| {
            json!({
                "key": key,
                "path": path.to_string_lossy(),
            })
        })
        .collect();
    let payload = json!({
        "kind": "external-validation-artifact-cli-run",
        "tool": tool.id,
        "format": opts.input_format.as_str(),
        "command": command.to_string_lossy(),
        "args": args,
        "artifact_paths": artifact_payload,
        "temp_dir": temp_dir.to_string_lossy(),
        "temp_dir_removed": cleanup_ok,
        "exit_success": output.status.success(),
        "exit_code": output.status.code(),
        "verdict": verdict.as_str(),
        "stdout": stdout,
        "stderr": stderr,
    });
    if output.status.success() {
        ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Ok,
            output: Some(payload),
            elapsed_ms,
            message: String::new(),
        }
    } else {
        let message = stderr
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("external validation artifact CLI exited unsuccessfully")
            .to_string();
        ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Failed,
            output: Some(payload),
            elapsed_ms,
            message,
        }
    }
}

pub fn external_validation_run_verdict(
    run: &ExternalValidationRun,
) -> Option<ExternalValidationTextVerdict> {
    run.output
        .as_ref()
        .and_then(|output| output["verdict"].as_str())
        .and_then(ExternalValidationTextVerdict::parse)
}

pub fn run_external_validation_consensus(
    input_text: &str,
    invocations: &[ExternalValidationCliInvocation],
    expected_verdict: Option<ExternalValidationTextVerdict>,
) -> ExternalValidationConsensusReport {
    let runs: Vec<ExternalValidationConsensusRun> = invocations
        .iter()
        .map(|invocation| {
            let run = match invocation {
                ExternalValidationCliInvocation::Text { options, .. } => {
                    run_external_validation_text_cli(input_text, options)
                }
                ExternalValidationCliInvocation::File { options, .. } => {
                    run_external_validation_file_cli(input_text, options)
                }
                ExternalValidationCliInvocation::Artifact {
                    artifacts, options, ..
                } => run_external_validation_artifact_cli(artifacts, options),
            };
            let verdict = external_validation_run_verdict(&run);
            ExternalValidationConsensusRun {
                label: invocation.label().to_string(),
                run,
                verdict,
            }
        })
        .collect();

    let all_successful = !runs.is_empty()
        && runs
            .iter()
            .all(|run| run.run.status == ExternalValidationRunStatus::Ok && run.verdict.is_some());
    let first_verdict = runs.first().and_then(|run| run.verdict);
    let all_successful_verdicts_agree =
        all_successful && runs.iter().all(|run| run.verdict == first_verdict);
    let agreed_verdict = all_successful_verdicts_agree
        .then_some(first_verdict)
        .flatten();
    let expected_matches = expected_verdict
        .map(|expected| Some(expected) == agreed_verdict)
        .unwrap_or(true);
    let agreement = all_successful && all_successful_verdicts_agree && expected_matches;

    ExternalValidationConsensusReport {
        expected_verdict,
        agreed_verdict,
        all_successful,
        all_successful_verdicts_agree,
        expected_matches,
        agreement,
        runs,
    }
}

pub fn external_validation_consensus_report_to_json(
    report: &ExternalValidationConsensusReport,
) -> Value {
    let runs: Vec<Value> = report
        .runs
        .iter()
        .map(|run| {
            json!({
                "label": &run.label,
                "tool": &run.run.tool_id,
                "status": run.run.status.as_str(),
                "verdict": run.verdict.map(ExternalValidationTextVerdict::as_str),
                "elapsed_ms": run.run.elapsed_ms,
                "message": &run.run.message,
                "output": &run.run.output,
            })
        })
        .collect();
    json!({
        "kind": "external-validation-consensus-report",
        "expected_verdict": report.expected_verdict.map(ExternalValidationTextVerdict::as_str),
        "agreed_verdict": report.agreed_verdict.map(ExternalValidationTextVerdict::as_str),
        "all_successful": report.all_successful,
        "all_successful_verdicts_agree": report.all_successful_verdicts_agree,
        "expected_matches": report.expected_matches,
        "agreement": report.agreement,
        "runs": runs,
    })
}

fn artifact_path_by_any(
    artifact_paths: &BTreeMap<String, PathBuf>,
    keys: &[&str],
) -> Option<String> {
    keys.iter().find_map(|key| {
        artifact_paths
            .get(*key)
            .map(|path| path.to_string_lossy().to_string())
    })
}

fn replace_artifact_placeholders(arg: &str, artifact_paths: &BTreeMap<String, PathBuf>) -> String {
    let mut out = arg.to_string();
    for (key, path) in artifact_paths {
        out = out.replace(&format!("{{{key}}}"), &path.to_string_lossy());
        out = out.replace(&format!("{{artifact:{key}}}"), &path.to_string_lossy());
    }
    if let Some(input) = artifact_path_by_any(artifact_paths, &["input", "model", "cnf"]) {
        out = out.replace("{input}", &input);
    }
    out
}

fn validation_artifact_file_name(
    artifact: &ExternalValidationArtifact,
    input_format: ExternalValidationTextFormat,
) -> String {
    if let Some(file_name) = &artifact.file_name {
        return sanitize_validation_artifact_file_name(file_name);
    }
    let key = sanitize_validation_artifact_file_stem(&artifact.key);
    let extension = artifact
        .file_extension
        .as_deref()
        .unwrap_or_else(|| input_format.file_extension())
        .trim_start_matches('.');
    format!("{key}.{extension}")
}

fn sanitize_validation_artifact_file_name(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => ch,
            _ => '_',
        })
        .collect();
    if sanitized.is_empty() {
        "artifact.txt".to_string()
    } else {
        sanitized
    }
}

fn sanitize_validation_artifact_file_stem(value: &str) -> String {
    let sanitized = sanitize_validation_artifact_file_name(value);
    let trimmed = sanitized.trim_matches('.').to_string();
    if trimmed.is_empty() {
        "artifact".to_string()
    } else {
        trimmed
    }
}

fn cleanup_validation_artifact_workspace(paths: &[PathBuf], temp_dir: &Path) -> bool {
    let files_removed = paths
        .iter()
        .all(|path| !path.exists() || fs::remove_file(path).is_ok());
    let dir_removed = !temp_dir.exists() || fs::remove_dir(temp_dir).is_ok();
    files_removed && dir_removed
}

pub fn external_validation_tool_specs() -> &'static [ExternalValidationToolSpec] {
    EXTERNAL_VALIDATION_TOOLS
}

pub fn find_external_validation_tool(id: &str) -> Option<&'static ExternalValidationToolSpec> {
    let normalized = normalize_tool_id(id);
    EXTERNAL_VALIDATION_TOOLS
        .iter()
        .find(|tool| tool.id == normalized || tool.env_key.eq_ignore_ascii_case(&normalized))
}

pub fn external_validation_adapter_env_names(tool: &ExternalValidationToolSpec) -> Vec<String> {
    vec![
        format!("ORES_{}_ADAPTER", tool.env_key),
        format!("DES_{}_ADAPTER", tool.env_key),
        format!("{}_ADAPTER", tool.env_key),
    ]
}

pub fn external_validation_artifact_env_names(tool: &ExternalValidationToolSpec) -> Vec<String> {
    let mut names = Vec::new();
    if let Some(suffix) = tool.artifact_kind.env_suffix() {
        names.push(format!("ORES_{}_{}", tool.env_key, suffix));
        names.push(format!("DES_{}_{}", tool.env_key, suffix));
        names.push(format!("{}_{}", tool.env_key, suffix));
    }
    match tool.id {
        "tlc" => names.push("TLA_TOOLS_JAR".to_string()),
        "minizinc" | "flatzinc" | "minizinc-solution-checker" => {
            names.push("MINIZINC_HOME".to_string());
        }
        "ortools-cp-sat" => {
            names.push("FZN_CP_SAT_CMD".to_string());
            names.push("ORTOOLS_HOME".to_string());
            names.push("MINIZINC_HOME".to_string());
        }
        "choco-solver" => names.push("CHOCO_SOLVER_HOME".to_string()),
        "jacop" => names.push("JACOP_HOME".to_string()),
        "ibm-cp-optimizer" => names.push("CPLEX_STUDIO_DIR".to_string()),
        "ortools-java" => names.push("ORTOOLS_JAVA_HOME".to_string()),
        "ojalgo" => names.push("OJALGO_HOME".to_string()),
        "optaplanner" => names.push("OPTAPLANNER_HOME".to_string()),
        "timefold" => names.push("TIMEFOLD_HOME".to_string()),
        "jmetal" => names.push("JMETAL_HOME".to_string()),
        "moea-framework" => names.push("MOEA_FRAMEWORK_HOME".to_string()),
        "ecj" => names.push("ECJ_HOME".to_string()),
        "good-lp" => names.push("GOOD_LP_CRATE".to_string()),
        "lp-modeler" => names.push("LP_MODELER_CRATE".to_string()),
        "rust-linprog" => names.push("RUST_LINPROG_CRATE".to_string()),
        "argmin" => names.push("ARGMIN_CRATE".to_string()),
        "nlopt-rs" => names.push("NLOPT_DIR".to_string()),
        "gurobi-rust" => {
            names.push("GUROBI_RUST_CARGO_MANIFEST".to_string());
            names.push("GUROBI_HOME".to_string());
            names.push("GRB_LICENSE_FILE".to_string());
        }
        "cplex-rust" => {
            names.push("CPLEX_RUST_CARGO_MANIFEST".to_string());
            names.push("CPLEX_STUDIO_DIR".to_string());
            names.push("CPLEX_HOME".to_string());
        }
        "ipopt-rust" => {
            names.push("IPOPT_RUST_CARGO_MANIFEST".to_string());
            names.push("IPOPT_DIR".to_string());
            names.push("IPOPT_HOME".to_string());
        }
        "highs-rust" => names.push("HIGHS_DIR".to_string()),
        "scip-rust" => names.push("SCIPOPTDIR".to_string()),
        "cbc-rust" => names.push("CBC_DIR".to_string()),
        "cpmpy" => names.push("CPMPY_PYTHON".to_string()),
        "pycsp3" => names.push("PYCSP3_PYTHON".to_string()),
        "conjure" => names.push("CONJURE_HOME".to_string()),
        "savile-row" => names.push("SAVILEROW_HOME".to_string()),
        "picat" => names.push("PICAT_HOME".to_string()),
        "clingo" => names.push("CLINGO_HOME".to_string()),
        "clingcon" => names.push("CLINGCON_HOME".to_string()),
        "pyomo" => names.push("PYOMO_PYTHON".to_string()),
        "pulp" => names.push("PULP_PYTHON".to_string()),
        "pyscipopt" => names.push("PYSCIPOPT_PYTHON".to_string()),
        "python-mip" => names.push("PYTHON_MIP_PYTHON".to_string()),
        "gurobipy" => names.push("GUROBIPY_PYTHON".to_string()),
        "cplex-python" => {
            names.push("CPLEX_PYTHON".to_string());
            names.push("CPLEX_STUDIO_DIR".to_string());
        }
        "xpress-python" => {
            names.push("XPRESS_PYTHON".to_string());
            names.push("XPRESSDIR".to_string());
        }
        "docplex" => names.push("DOCPLEX_PYTHON".to_string()),
        "ortools-python" | "ortools-glop" | "ortools-pdlp" => {
            names.push("ORTOOLS_PYTHON".to_string())
        }
        "scipy-optimize" => names.push("SCIPY_OPTIMIZE_PYTHON".to_string()),
        "highs-cli" => names.push("HIGHS_CMD".to_string()),
        "glpk-cli" => names.push("GLPSOL_CMD".to_string()),
        "scip-cli" => names.push("SCIP_CMD".to_string()),
        "cbc-cli" => names.push("CBC_CMD".to_string()),
        "clp-cli" => names.push("CLP_CMD".to_string()),
        "soplex-cli" => names.push("SOPLEX_CMD".to_string()),
        "qsopt-ex-cli" => names.push("QSOPT_EX_CMD".to_string()),
        "lp-solve-cli" => names.push("LP_SOLVE_CMD".to_string()),
        "gurobi-cli" => names.push("GUROBI_CL_CMD".to_string()),
        "cplex-cli" => names.push("CPLEX_CMD".to_string()),
        "xpress-cli" => names.push("XPRESS_CMD".to_string()),
        "lindo-cli" => names.push("LINDOAPI_CMD".to_string()),
        "ampl" => names.push("AMPL_HOME".to_string()),
        "gams" => names.push("GAMS_HOME".to_string()),
        "hexaly" => names.push("HEXALY_HOME".to_string()),
        "jump" => names.push("JULIA_PROJECT".to_string()),
        "neos" => names.push("NEOS_EMAIL".to_string()),
        "pddl-val" => names.push("VAL_HOME".to_string()),
        "fast-downward" => names.push("FAST_DOWNWARD_HOME".to_string()),
        "lpg-td" => names.push("LPG_HOME".to_string()),
        "optic" => names.push("OPTIC_HOME".to_string()),
        "enhsp" => names.push("ENHSP_HOME".to_string()),
        "sat4j" => names.push("SAT4J_HOME".to_string()),
        "pysat" => names.push("PYSAT_PYTHON".to_string()),
        "open-wbo" => names.push("OPEN_WBO_HOME".to_string()),
        "maxhs" => names.push("MAXHS_HOME".to_string()),
        "roundingsat" => names.push("ROUNDINGSAT_HOME".to_string()),
        "frat" => names.push("FRAT_HOME".to_string()),
        "veripb" => names.push("VERIPB_HOME".to_string()),
        "ipopt" => names.push("IPOPT_DIR".to_string()),
        "bonmin" => names.push("BONMIN_DIR".to_string()),
        "couenne" => names.push("COUENNE_DIR".to_string()),
        "knitro" => names.push("ARTELYS_LICENSE".to_string()),
        "mosek" => names.push("MOSEKLM_LICENSE_FILE".to_string()),
        "baron" => names.push("BARON_LICENSE".to_string()),
        "copt" => names.push("COPT_HOME".to_string()),
        "nlopt" => names.push("NLOPT_DIR".to_string()),
        "cvxopt" => names.push("CVXOPT_PYTHON".to_string()),
        "java-pathfinder" => names.push("JPF_HOME".to_string()),
        "key" => names.push("KEY_HOME".to_string()),
        "viper" => names.push("VIPER_HOME".to_string()),
        "fstar" => names.push("FSTAR_HOME".to_string()),
        "gnatprove" => names.push("GNATPROVE_HOME".to_string()),
        "seahorn" => names.push("SEAHORN_DIR".to_string()),
        "smack" => names.push("SMACK_HOME".to_string()),
        "klee" => names.push("KLEE_HOME".to_string()),
        "ultimate-automizer" => names.push("ULTIMATE_HOME".to_string()),
        "sumo" => names.push("SUMO_HOME".to_string()),
        "omnetpp" => names.push("OMNETPP_ROOT".to_string()),
        "ciw" => names.push("CIW_PYTHON".to_string()),
        "simulus" => names.push("SIMULUS_PYTHON".to_string()),
        "desmo-j" => names.push("DESMOJ_HOME".to_string()),
        "simsharp" => names.push("SIMSHARP_HOME".to_string()),
        "energyplus" => names.push("ENERGYPLUS_HOME".to_string()),
        "openmodelica" => names.push("OPENMODELICAHOME".to_string()),
        "simulink" => names.push("MATLAB_ROOT".to_string()),
        "ptolemy-ii" => names.push("PTII".to_string()),
        "gem5" => names.push("GEM5_ROOT".to_string()),
        "gridlabd" => names.push("GRIDLABD_HOME".to_string()),
        "opendss" => names.push("OPENDSS_HOME".to_string()),
        "copasi" => names.push("COPASI_HOME".to_string()),
        "gazebo" => names.push("GZ_SIM_RESOURCE_PATH".to_string()),
        "webots" => names.push("WEBOTS_HOME".to_string()),
        "mujoco" => names.push("MUJOCO_GL".to_string()),
        "carla" => names.push("CARLA_ROOT".to_string()),
        "isaac-sim" => names.push("ISAACSIM_PATH".to_string()),
        "plant-simulation" => names.push("PLANT_SIMULATION_HOME".to_string()),
        "extendsim" => names.push("EXTENDSIM_HOME".to_string()),
        "gpss-world" => names.push("GPSS_WORLD_HOME".to_string()),
        "dbt" => names.push("DBT_PROFILES_DIR".to_string()),
        _ => {}
    }
    names
}

pub fn external_validation_command_dir_env_names(tool: &ExternalValidationToolSpec) -> Vec<String> {
    let mut names = Vec::new();
    if tool.artifact_kind == ExternalValidationArtifactKind::NativeInstallDir {
        for name in external_validation_artifact_env_names(tool) {
            push_unique_env_name(&mut names, name);
        }
    }
    for name in match tool.id {
        "minizinc" | "flatzinc" | "minizinc-solution-checker" => &["MINIZINC_HOME"][..],
        "ortools-cp-sat" => &["ORTOOLS_HOME", "ORTOOLS_DIR", "MINIZINC_HOME"],
        "choco-solver" => &["CHOCO_SOLVER_HOME", "CHOCO_HOME"],
        "jacop" => &["JACOP_HOME", "JACOP_DIR"],
        "ibm-cp-optimizer" => &["CPLEX_STUDIO_DIR", "CPLEX_HOME", "CP_OPTIMIZER_HOME"],
        "ortools-java" => &["ORTOOLS_JAVA_HOME", "ORTOOLS_HOME"],
        "ojalgo" => &["OJALGO_HOME", "OJALGO_DIR"],
        "optaplanner" => &["OPTAPLANNER_HOME", "OPTAPLANNER_DIR"],
        "timefold" => &["TIMEFOLD_HOME", "TIMEFOLD_DIR"],
        "jmetal" => &["JMETAL_HOME", "JMETAL_DIR"],
        "moea-framework" => &["MOEA_FRAMEWORK_HOME", "MOEA_HOME"],
        "ecj" => &["ECJ_HOME", "ECJ_DIR"],
        "nlopt-rs" => &["NLOPT_DIR", "NLOPT_HOME"],
        "gurobi-rust" => &["GUROBI_HOME"],
        "cplex-rust" => &["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        "ipopt-rust" => &["IPOPT_DIR", "IPOPT_HOME"],
        "highs-rust" => &["HIGHS_DIR", "HIGHS_HOME"],
        "scip-rust" => &["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
        "cbc-rust" => &["CBC_DIR", "CBC_HOME", "COINOR_DIR", "COINOR_HOME"],
        "cpmpy" => &["CPMPY_HOME", "CPMPY_DIR"],
        "pycsp3" => &["PYCSP3_HOME", "PYCSP3_DIR"],
        "conjure" => &["CONJURE_HOME", "CONJURE_DIR"],
        "savile-row" => &["SAVILE_ROW_HOME", "SAVILE_ROW_DIR", "SAVILEROW_HOME"],
        "picat" => &["PICAT_HOME", "PICAT_DIR"],
        "clingo" => &["CLINGO_HOME", "CLINGO_DIR", "POTASSCO_HOME"],
        "clingcon" => &["CLINGCON_HOME", "CLINGCON_DIR", "POTASSCO_HOME"],
        "pyomo" => &["PYOMO_HOME", "PYOMO_DIR"],
        "pulp" => &["PULP_HOME", "PULP_DIR"],
        "pyscipopt" => &["PYSCIPOPT_HOME", "PYSCIPOPT_DIR", "SCIPOPTDIR", "SCIP_DIR"],
        "python-mip" => &["PYTHON_MIP_HOME", "MIP_HOME", "MIP_DIR"],
        "gurobipy" => &["GUROBI_HOME"],
        "cplex-python" => &["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        "xpress-python" => &["XPRESSDIR", "XPRESS_DIR", "XPRESS_HOME"],
        "docplex" => &["DOCPLEX_HOME", "CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        "ortools-python" | "ortools-glop" | "ortools-pdlp" => &["ORTOOLS_HOME", "ORTOOLS_DIR"],
        "scipy-optimize" => &["SCIPY_HOME", "SCIPY_DIR"],
        "highs-cli" => &["HIGHS_DIR", "HIGHS_HOME"],
        "glpk-cli" => &["GLPK_DIR", "GLPK_HOME"],
        "scip-cli" => &["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
        "cbc-cli" => &["CBC_DIR", "CBC_HOME", "COINOR_DIR", "COINOR_HOME"],
        "clp-cli" => &["CLP_DIR", "CLP_HOME", "COINOR_DIR", "COINOR_HOME"],
        "soplex-cli" => &["SOPLEX_DIR", "SOPLEX_HOME"],
        "qsopt-ex-cli" => &["QSOPT_EX_DIR", "QSOPT_EX_HOME", "QSOPT_DIR", "QSOPT_HOME"],
        "lp-solve-cli" => &[
            "LP_SOLVE_DIR",
            "LPSOLVE_DIR",
            "LP_SOLVE_HOME",
            "LPSOLVE_HOME",
        ],
        "gurobi-cli" => &["GUROBI_HOME"],
        "cplex-cli" => &["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
        "xpress-cli" => &["XPRESSDIR", "XPRESS_DIR", "XPRESS_HOME"],
        "lindo-cli" => &["LINDO_HOME", "LINDO_DIR", "LINDOAPI_HOME", "LINDOAPI_DIR"],
        "ampl" => &["AMPL_HOME", "AMPL_DIR"],
        "gams" => &["GAMS_HOME", "GAMS_DIR"],
        "hexaly" => &["HEXALY_HOME", "HEXALY_DIR", "LOCALSOLVER_HOME"],
        "jump" => &["JUMP_HOME", "JULIA_HOME", "JULIA_DIR"],
        "neos" => &["NEOS_HOME", "NEOS_DIR"],
        "pddl-val" => &["VAL_HOME", "VAL_DIR", "PDDL_VAL_HOME", "PDDL_VAL_DIR"],
        "fast-downward" => &["FAST_DOWNWARD_HOME", "FAST_DOWNWARD_DIR"],
        "lpg-td" => &["LPG_TD_HOME", "LPG_HOME", "LPG_DIR"],
        "optic" => &["OPTIC_HOME", "OPTIC_DIR"],
        "enhsp" => &["ENHSP_HOME", "ENHSP_DIR"],
        "minisat" => &["MINISAT_HOME", "MINISAT_DIR"],
        "glucose" => &["GLUCOSE_HOME", "GLUCOSE_DIR"],
        "maplesat" => &["MAPLESAT_HOME", "MAPLESAT_DIR"],
        "varisat" => &["VARISAT_HOME", "VARISAT_DIR"],
        "sat4j" => &["SAT4J_HOME", "SAT4J_DIR"],
        "pysat" => &["PYSAT_HOME", "PYSAT_DIR"],
        "open-wbo" => &["OPEN_WBO_HOME", "OPEN_WBO_DIR", "OPENWBO_HOME"],
        "maxhs" => &["MAXHS_HOME", "MAXHS_DIR"],
        "roundingsat" => &["ROUNDINGSAT_HOME", "ROUNDINGSAT_DIR"],
        "frat" => &["FRAT_HOME", "FRAT_DIR"],
        "veripb" => &["VERIPB_HOME", "VERIPB_DIR"],
        "tlc" => &["TLC_HOME", "TLA_TOOLS_DIR"],
        "apalache" => &["APALACHE_HOME", "APALACHE_DIR"],
        "alloy" => &["ALLOY_HOME", "ALLOY_DIR"],
        "kodkod" => &["KODKOD_HOME", "KODKOD_DIR"],
        "spin" => &["SPIN_HOME", "SPIN_DIR"],
        "nuxmv" => &["NUXMV_HOME", "NUXMV_DIR"],
        "prism" => &["PRISM_HOME", "PRISM_DIR"],
        "storm" => &["STORM_HOME", "STORM_DIR"],
        "uppaal" => &["UPPAAL_HOME", "UPPAAL_DIR"],
        "cbmc" => &["CBMC_HOME", "CBMC_DIR"],
        "ebmc" => &["EBMC_HOME", "EBMC_DIR"],
        "dafny" => &["DAFNY_HOME", "DAFNY_DIR"],
        "frama-c" => &["FRAMA_C_HOME", "FRAMA_C_DIR"],
        "why3" => &["WHY3_HOME", "WHY3_DIR"],
        "esbmc" => &["ESBMC_HOME", "ESBMC_DIR"],
        "klee" => &["KLEE_HOME", "KLEE_DIR"],
        "mirai" => &["MIRAI_HOME", "MIRAI_DIR"],
        "jbmc" => &["JBMC_HOME", "JBMC_DIR"],
        "java-pathfinder" => &["JPF_HOME", "JAVA_PATHFINDER_HOME"],
        "key" => &["KEY_HOME"],
        "boogie" => &["BOOGIE_HOME", "BOOGIE_DIR"],
        "goblint" => &["GOBLINT_HOME", "GOBLINT_DIR"],
        "coq" => &["COQ_HOME", "COQ_DIR"],
        "lean" => &["LEAN_HOME", "LEAN_DIR"],
        "acl2" => &["ACL2_HOME", "ACL2_DIR"],
        "tamarin" => &["TAMARIN_HOME", "TAMARIN_DIR"],
        "proverif" => &["PROVERIF_HOME", "PROVERIF_DIR"],
        "cryptoverif" => &["CRYPTOVERIF_HOME", "CRYPTOVERIF_DIR"],
        "deepsec" => &["DEEPSEC_HOME", "DEEPSEC_DIR"],
        "scyther" => &["SCYTHER_HOME", "SCYTHER_DIR"],
        "verifpal" => &["VERIFPAL_HOME", "VERIFPAL_DIR"],
        "sapic-plus" => &["SAPIC_PLUS_HOME", "SAPIC_PLUS_DIR"],
        "maude" => &["MAUDE_HOME", "MAUDE_DIR"],
        "ipopt" => &["IPOPT_DIR", "IPOPT_HOME"],
        "bonmin" => &["BONMIN_DIR", "BONMIN_HOME"],
        "minotaur" => &["MINOTAUR_DIR", "MINOTAUR_HOME"],
        "couenne" => &["COUENNE_DIR", "COUENNE_HOME"],
        "symphony" => &["SYMPHONY_DIR", "SYMPHONY_HOME", "COINOR_DIR", "COINOR_HOME"],
        "knitro" => &["KNITRO_HOME", "KNITRODIR", "KNITRO_DIR", "ARTELYS_HOME"],
        "mosek" => &["MOSEK_HOME", "MSKHOME"],
        "baron" => &["BARON_DIR", "BARON_HOME"],
        "copt" => &["COPT_HOME", "COPT_DIR"],
        "nlopt" => &["NLOPT_DIR", "NLOPT_HOME"],
        "nlopt-cli" => &["NLOPT_DIR", "NLOPT_HOME"],
        "casadi" => &["CASADI_DIR", "CASADI_HOME"],
        "cvxopt" => &["CVXOPT_DIR", "CVXOPT_HOME"],
        "osqp" => &["OSQP_DIR", "OSQP_HOME"],
        "scs" => &["SCS_DIR", "SCS_HOME"],
        "clarabel" => &["CLARABEL_DIR", "CLARABEL_HOME"],
        "ecos" => &["ECOS_DIR", "ECOS_HOME"],
        "qpoases" => &["QPOASES_DIR", "QPOASES_HOME"],
        "proxqp" => &["PROXQP_DIR", "PROXQP_HOME"],
        "cosmo" => &["COSMO_DIR", "COSMO_HOME"],
        "sdpa" => &["SDPA_DIR", "SDPA_HOME"],
        "csdp" => &["CSDP_DIR", "CSDP_HOME"],
        "openmodelica" => &["OPENMODELICAHOME", "OPENMODELICA_HOME"],
        "simulink" => &["MATLAB_ROOT", "MATLAB_HOME"],
        "gazebo" => &["GAZEBO_HOME", "GZ_HOME"],
        "ciw" => &["CIW_HOME", "CIW_DIR"],
        "simulus" => &["SIMULUS_HOME", "SIMULUS_DIR"],
        "jaamsim" => &["JAAMSIM_HOME", "JAAMSIM_DIR"],
        "desmo-j" => &["DESMO_J_HOME", "DESMO_J_DIR", "DESMOJ_HOME"],
        "simsharp" => &["SIMSHARP_HOME", "SIMSHARP_DIR"],
        "matsim" => &["MATSIM_HOME", "MATSIM_DIR"],
        "ptolemy-ii" => &["PTII", "PTOLEMY_HOME", "PTOLEMY_II_HOME"],
        "repast" => &["REPAST_HOME", "REPAST_DIR"],
        "mason" => &["MASON_HOME", "MASON_DIR"],
        "cloudsim" => &["CLOUDSIM_HOME", "CLOUDSIM_DIR"],
        "neqsim" => &["NEQSIM_HOME", "NEQSIM_DIR"],
        "anylogic" => &["ANYLOGIC_HOME", "ANYLOGIC_DIR"],
        _ => &[],
    } {
        push_unique_env_name(&mut names, *name);
    }
    names
}

pub fn external_validation_adapter_command(tool: &ExternalValidationToolSpec) -> Option<PathBuf> {
    configured_adapter_command(tool)
        .0
        .or_else(|| find_first_command_in_install_dirs(tool))
        .or_else(|| find_first_command(tool.command_aliases))
}

pub fn external_validation_adapter_command_with_options(
    opts: &ExternalValidationAdapterOptions,
) -> Option<PathBuf> {
    let tool = find_external_validation_tool(&opts.tool_id)?;
    opts.command_path
        .as_ref()
        .cloned()
        .or_else(|| external_validation_adapter_command(tool))
}

pub fn probe_external_validation_tool(
    tool: &ExternalValidationToolSpec,
) -> ExternalValidationProbe {
    let (configured_command, saw_configured_command) = configured_adapter_command(tool);
    if let Some(command) = configured_command
        .or_else(|| find_first_command_in_install_dirs(tool))
        .or_else(|| find_first_command(tool.command_aliases))
    {
        return ExternalValidationProbe {
            tool_id: tool.id.to_string(),
            status: ExternalValidationProbeStatus::Ready,
            command: Some(command.clone()),
            message: format!(
                "{} command is configured at {}",
                tool.display_name,
                command.display()
            ),
        };
    }
    if saw_configured_command {
        return ExternalValidationProbe {
            tool_id: tool.id.to_string(),
            status: ExternalValidationProbeStatus::AdapterMissing,
            command: None,
            message: format!(
                "{} adapter was configured but could not be resolved",
                tool.display_name
            ),
        };
    }

    let artifact = first_configured_env_value(&external_validation_artifact_env_names(tool));
    if let Some(value) = artifact {
        return probe_configured_artifact(tool, value);
    }
    if tool.artifact_kind == ExternalValidationArtifactKind::JavaClasspath {
        if let Some(classpath) = java_classpath_from_install_dirs(tool) {
            return probe_configured_artifact(tool, classpath.to_string_lossy().to_string());
        }
    }

    ExternalValidationProbe {
        tool_id: tool.id.to_string(),
        status: ExternalValidationProbeStatus::NotConfigured,
        command: None,
        message: format!(
            "{} is not configured; set {} or {}",
            tool.display_name,
            external_validation_adapter_env_names(tool)[0],
            external_validation_artifact_hint(tool)
        ),
    }
}

pub fn run_external_validation_adapter(
    input: &Value,
    opts: &ExternalValidationAdapterOptions,
) -> ExternalValidationRun {
    let Some(tool) = find_external_validation_tool(&opts.tool_id) else {
        return ExternalValidationRun {
            tool_id: opts.tool_id.clone(),
            status: ExternalValidationRunStatus::Unavailable,
            output: None,
            elapsed_ms: 0.0,
            message: format!("unknown external validation tool '{}'", opts.tool_id),
        };
    };
    let Some(command) = external_validation_adapter_command_with_options(opts) else {
        return ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Unavailable,
            output: None,
            elapsed_ms: 0.0,
            message: format!(
                "{} adapter command not configured; set {}",
                tool.display_name,
                external_validation_adapter_env_names(tool)[0]
            ),
        };
    };
    run_json_adapter(input, tool, command, opts)
}

fn run_json_adapter(
    input: &Value,
    tool: &ExternalValidationToolSpec,
    command: PathBuf,
    opts: &ExternalValidationAdapterOptions,
) -> ExternalValidationRun {
    let started = Instant::now();
    let mut child = match Command::new(&command)
        .args(&opts.extra_args)
        .env("ORES_EXTERNAL_VALIDATION_TOOL", tool.id)
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
            return ExternalValidationRun {
                tool_id: tool.id.to_string(),
                status: ExternalValidationRunStatus::Unavailable,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: err.to_string(),
            };
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        let payload = serde_json::to_vec(input).unwrap_or_else(|_| b"null".to_vec());
        if let Err(err) = stdin.write_all(&payload) {
            return ExternalValidationRun {
                tool_id: tool.id.to_string(),
                status: ExternalValidationRunStatus::Failed,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: err.to_string(),
            };
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return ExternalValidationRun {
                tool_id: tool.id.to_string(),
                status: ExternalValidationRunStatus::Failed,
                output: None,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: err.to_string(),
            };
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if !output.status.success() {
        return ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Failed,
            output: None,
            elapsed_ms,
            message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        };
    }
    match serde_json::from_slice::<Value>(&output.stdout) {
        Ok(value) => ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Ok,
            output: Some(value),
            elapsed_ms,
            message: String::new(),
        },
        Err(err) => ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::InvalidOutput,
            output: None,
            elapsed_ms,
            message: err.to_string(),
        },
    }
}

fn probe_configured_artifact(
    tool: &ExternalValidationToolSpec,
    value: String,
) -> ExternalValidationProbe {
    match tool.artifact_kind {
        ExternalValidationArtifactKind::None
        | ExternalValidationArtifactKind::PythonPackage
        | ExternalValidationArtifactKind::RustCrate => ExternalValidationProbe {
            tool_id: tool.id.to_string(),
            status: ExternalValidationProbeStatus::Ready,
            command: None,
            message: format!("{} artifact marker is configured", tool.display_name),
        },
        ExternalValidationArtifactKind::JavaClasspath => {
            let java = find_first_command(&["java"]);
            if java.is_none() {
                return ExternalValidationProbe {
                    tool_id: tool.id.to_string(),
                    status: ExternalValidationProbeStatus::RuntimeMissing,
                    command: None,
                    message: format!(
                        "{} classpath is configured, but `java` was not found",
                        tool.display_name
                    ),
                };
            }
            ExternalValidationProbe {
                tool_id: tool.id.to_string(),
                status: ExternalValidationProbeStatus::Ready,
                command: java,
                message: format!("{} Java classpath is configured", tool.display_name),
            }
        }
        ExternalValidationArtifactKind::NativeInstallDir
        | ExternalValidationArtifactKind::BenchmarkDataDir
        | ExternalValidationArtifactKind::SchemaOrSpecPath => {
            let path = PathBuf::from(value);
            if path.exists() {
                ExternalValidationProbe {
                    tool_id: tool.id.to_string(),
                    status: ExternalValidationProbeStatus::Ready,
                    command: None,
                    message: format!("{} artifact path is configured", tool.display_name),
                }
            } else {
                ExternalValidationProbe {
                    tool_id: tool.id.to_string(),
                    status: ExternalValidationProbeStatus::ArtifactMissing,
                    command: None,
                    message: format!(
                        "{} artifact path does not exist: {}",
                        tool.display_name,
                        path.display()
                    ),
                }
            }
        }
    }
}

fn external_validation_artifact_hint(tool: &ExternalValidationToolSpec) -> String {
    external_validation_artifact_env_names(tool)
        .first()
        .cloned()
        .unwrap_or_else(|| "a local adapter command".to_string())
}

fn push_unique_env_name(names: &mut Vec<String>, name: impl Into<String>) {
    let name = name.into();
    if !names.iter().any(|existing| existing == &name) {
        names.push(name);
    }
}

fn configured_adapter_command(tool: &ExternalValidationToolSpec) -> (Option<PathBuf>, bool) {
    let mut saw_configured = false;
    for env_name in external_validation_adapter_env_names(tool) {
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

fn find_first_command_in_install_dirs(tool: &ExternalValidationToolSpec) -> Option<PathBuf> {
    for env_name in external_validation_command_dir_env_names(tool) {
        let Some(raw_value) = env::var_os(&env_name) else {
            continue;
        };
        if raw_value.to_string_lossy().trim().is_empty() {
            continue;
        }
        for root in env::split_paths(&raw_value) {
            if let Some(path) = find_command_in_install_dir(&root, tool.command_aliases) {
                return Some(path);
            }
        }
    }
    None
}

fn java_classpath_from_install_dirs(tool: &ExternalValidationToolSpec) -> Option<OsString> {
    for env_name in external_validation_command_dir_env_names(tool) {
        let Some(raw_value) = env::var_os(&env_name) else {
            continue;
        };
        if raw_value.to_string_lossy().trim().is_empty() {
            continue;
        }
        for root in env::split_paths(&raw_value) {
            if let Some(classpath) = find_java_classpath_in_install_dir(&root) {
                return Some(classpath);
            }
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
    if let Ok(children) = fs::read_dir(root) {
        for child in children.flatten() {
            let child_path = child.path();
            if !child_path.is_dir() {
                continue;
            }
            collect_jar_files(&child_path.join("lib"), &mut jars);
            collect_jar_files(&child_path.join("build").join("libs"), &mut jars);
            collect_jar_files(&child_path.join("target"), &mut jars);
            collect_jar_files(&child_path.join("target").join("dependency"), &mut jars);
        }
    }
    if jars.is_empty() {
        return None;
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
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("jar"))
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

fn normalize_tool_id(id: &str) -> String {
    id.trim().to_ascii_lowercase().replace('_', "-")
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

#[cfg(test)]
mod tests {
    use crate::des::general::external_validation_tools::{
        dimacs_cnf_to_string, dimacs_wcnf_to_string, external_benchmark_manifest_to_json,
        external_simulation_validation_engine_manifest, external_simulation_validation_tool_specs,
        external_validation_adapter_env_names, external_validation_artifact_cli_args,
        external_validation_artifact_env_names, external_validation_command_dir_env_names,
        external_validation_consensus_report_to_json,
        external_validation_default_artifact_cli_args, external_validation_default_file_cli_args,
        external_validation_default_text_cli_args, external_validation_file_cli_args,
        external_validation_tool_specs, find_command_in_install_dir, find_external_validation_tool,
        find_java_classpath_in_install_dir, infer_external_validation_text_verdict, is_jar_file,
        json_schema_validation_request_to_json, minizinc_validation_request_to_json,
        prism_validation_model_to_string, prism_validation_properties_to_string,
        run_external_validation_artifact_cli, run_external_validation_consensus,
        run_external_validation_file_cli, run_external_validation_text_cli,
        run_simulation_validation_json_with_external_reference,
        run_simulation_validation_with_external_reference, simulation_validation_request_to_json,
        smtlib_validation_script_to_string, tla_validation_module_to_string, DimacsCnf, DimacsWcnf,
        DimacsWeightedClause, ExternalBenchmarkManifest, ExternalBenchmarkManifestEntry,
        ExternalSimulationValidationReferenceOptions, ExternalSimulationValidationStatus,
        ExternalSimulationValidationVerdict, ExternalValidationArtifact,
        ExternalValidationArtifactCliOptions, ExternalValidationArtifactKind,
        ExternalValidationCapability, ExternalValidationCliInvocation, ExternalValidationFamily,
        ExternalValidationFileCliOptions, ExternalValidationProbeStatus,
        ExternalValidationRunStatus, ExternalValidationRuntime, ExternalValidationTextCliOptions,
        ExternalValidationTextFormat, ExternalValidationTextVerdict, JsonSchemaValidationRequest,
        MiniZincValidationRequest, PrismModule, PrismValidationModel, SimulationMetricExpectation,
        SimulationValidationRequest, SmtDeclaration, SmtLibValidationScript, SmtSort,
        TlaValidationModule,
    };
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    #[test]
    fn registry_covers_recommended_validation_layers() {
        let tools = external_validation_tool_specs();
        assert_eq!(tools.len(), 264);
        assert!(tools
            .iter()
            .any(|tool| tool.id == "minizinc" && tool.input_formats.contains(&"mzn")));
        assert!(tools.iter().any(|tool| {
            tool.id == "choco-solver"
                && tool.runtime == ExternalValidationRuntime::Java
                && tool
                    .capabilities
                    .contains(&ExternalValidationCapability::CheckSolution)
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "jacop" && tool.input_formats.contains(&"xcsp3") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "ibm-cp-optimizer"
                && tool.artifact_kind == ExternalValidationArtifactKind::JavaClasspath
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "ortools-java" && tool.runtime == ExternalValidationRuntime::Java
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "ojalgo" && tool.input_formats.contains(&"mps") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "optaplanner"
                && tool
                    .capabilities
                    .contains(&ExternalValidationCapability::SolveModel)
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "timefold" && tool.input_formats.contains(&"pddl") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "jmetal" && tool.input_formats.contains(&"json") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "moea-framework" && tool.runtime == ExternalValidationRuntime::Java
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "ecj" && tool.family == ExternalValidationFamily::ConstraintModeling
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "good-lp"
                && tool.runtime == ExternalValidationRuntime::Rust
                && tool.input_formats.contains(&"lp")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "lp-modeler"
                && tool.artifact_kind == ExternalValidationArtifactKind::RustCrate
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "rust-linprog" && tool.runtime == ExternalValidationRuntime::Rust
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "argmin" && tool.family == ExternalValidationFamily::NonlinearGlobalSolver
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "nlopt-rs" && tool.input_formats.contains(&"nl") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "gurobi-rust"
                && tool.runtime == ExternalValidationRuntime::Rust
                && tool.artifact_kind == ExternalValidationArtifactKind::RustCrate
                && tool.command_aliases.contains(&"gurobi-rust-adapter")
                && tool.input_formats.contains(&"mps")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cplex-rust"
                && tool.runtime == ExternalValidationRuntime::Rust
                && tool.artifact_kind == ExternalValidationArtifactKind::RustCrate
                && tool.command_aliases.contains(&"cplex-rust-adapter")
                && tool.input_formats.contains(&"lp")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "ipopt-rust"
                && tool.runtime == ExternalValidationRuntime::Rust
                && tool.family == ExternalValidationFamily::NonlinearGlobalSolver
                && tool.input_formats.contains(&"nl")
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "nlopt" && tool.input_formats.contains(&"json") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "highs-rust" && tool.input_formats.contains(&"mps") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "scip-rust" && tool.input_formats.contains(&"json") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "cbc-rust" && tool.input_formats.contains(&"lp") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cpmpy" && tool.family == ExternalValidationFamily::ConstraintModeling
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "conjure" && tool.input_formats.contains(&"essence") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "clingo" && tool.input_formats.contains(&"asp") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "pyomo" && tool.family == ExternalValidationFamily::ConstraintModeling
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "pulp" && tool.input_formats.contains(&"lp") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "pyscipopt"
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.input_formats.contains(&"mps")
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "python-mip" && tool.input_formats.contains(&"osil") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "gurobipy"
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cplex-python"
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.input_formats.contains(&"mps")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "xpress-python"
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.input_formats.contains(&"lp")
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "docplex" && tool.input_formats.contains(&"lp") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "ortools-python" && tool.runtime == ExternalValidationRuntime::Python
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "ortools-cp-sat"
                && tool.runtime == ExternalValidationRuntime::NativeCli
                && tool.command_aliases.contains(&"fzn-cp-sat")
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "ortools-glop" && tool.input_formats.contains(&"proto") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "ortools-pdlp" && tool.input_formats.contains(&"proto") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "scipy-optimize"
                && tool.family == ExternalValidationFamily::NonlinearGlobalSolver
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "highs-cli" && tool.command_aliases.contains(&"highs") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "glpk-cli" && tool.command_aliases.contains(&"glpsol") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "scip-cli" && tool.command_aliases.contains(&"scip") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "cbc-cli" && tool.command_aliases.contains(&"cbc") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "clp-cli" && tool.command_aliases.contains(&"clp") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "soplex-cli" && tool.command_aliases.contains(&"soplex") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "qsopt-ex-cli"
                && tool.command_aliases.contains(&"qsopt_ex")
                && tool.command_aliases.contains(&"esolver")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "lp-solve-cli" && tool.command_aliases.contains(&"lp_solve")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "gurobi-cli" && tool.command_aliases.contains(&"gurobi_cl")
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "cplex-cli" && tool.command_aliases.contains(&"cplex") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "xpress-cli" && tool.command_aliases.contains(&"optimizer")
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "lindo-cli" && tool.command_aliases.contains(&"runlindo") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "ampl" && tool.input_formats.contains(&"mod") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "gams" && tool.input_formats.contains(&"gms") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "hexaly" && tool.command_aliases.contains(&"localsolver") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "jump" && tool.command_aliases.contains(&"julia") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "neos"
                && tool
                    .capabilities
                    .contains(&ExternalValidationCapability::SolveModel)
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "pddl-val"
                && tool
                    .capabilities
                    .contains(&ExternalValidationCapability::CheckSolution)
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "fast-downward" && tool.input_formats.contains(&"pddl") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "lpg-td" && tool.input_formats.contains(&"pddl") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "optic" && tool.input_formats.contains(&"pddl") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "enhsp" && tool.input_formats.contains(&"pddl") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "z3" && tool.family == ExternalValidationFamily::SmtSolver }));
        assert!(tools.iter().any(|tool| {
            tool.id == "optimathsat"
                && tool.family == ExternalValidationFamily::SmtSolver
                && tool.input_formats.contains(&"smt2")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "sat4j" && tool.family == ExternalValidationFamily::SatSolver
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "minisat" && tool.family == ExternalValidationFamily::SatSolver
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "varisat" && tool.runtime == ExternalValidationRuntime::Rust
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "open-wbo" && tool.input_formats.contains(&"wcnf") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "maxhs" && tool.input_formats.contains(&"wcnf") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "roundingsat" && tool.input_formats.contains(&"opb") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "drat-trim" && tool.family == ExternalValidationFamily::ProofChecker
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "veripb" && tool.family == ExternalValidationFamily::ProofChecker
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "frat" && tool.input_formats.contains(&"frat") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "tlc" && tool.runtime == ExternalValidationRuntime::Java }));
        assert!(tools.iter().any(|tool| {
            tool.id == "kodkod" && tool.family == ExternalValidationFamily::FormalModelChecker
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "miplib" && tool.family == ExternalValidationFamily::BenchmarkLibrary
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "osqp" && tool.family == ExternalValidationFamily::ConvexConicSolver
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cvxopt" && tool.family == ExternalValidationFamily::ConvexConicSolver
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cbmc" && tool.family == ExternalValidationFamily::FormalModelChecker
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "ebmc" && tool.family == ExternalValidationFamily::FormalModelChecker
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "klee" && tool.family == ExternalValidationFamily::FormalModelChecker
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "java-pathfinder"
                && tool.family == ExternalValidationFamily::FormalModelChecker
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "boogie" && tool.family == ExternalValidationFamily::FormalModelChecker
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "creusot" && tool.runtime == ExternalValidationRuntime::Rust
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "mirai" && tool.runtime == ExternalValidationRuntime::Rust }));
        assert!(tools.iter().any(|tool| {
            tool.id == "dafny" && tool.family == ExternalValidationFamily::FormalModelChecker
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "tamarin" && tool.family == ExternalValidationFamily::FormalModelChecker
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "minotaur" && tool.family == ExternalValidationFamily::NonlinearGlobalSolver
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "sdpa" && tool.family == ExternalValidationFamily::ConvexConicSolver
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "energyplus" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "gridlabd" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "copasi" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "carla" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "plant-simulation"
                && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "mesa" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "agentpy" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "ciw" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "desmo-j" && tool.runtime == ExternalValidationRuntime::Java
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "simgrid" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cloudsim" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "batsim" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cape-open" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "anylogic" && tool.family == ExternalValidationFamily::SimulationEngine
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "great-expectations"
                && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "check-jsonschema"
                && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cue" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "zod" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "dbt" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "apache-arrow"
                && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "spectral" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "schematron" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "xml-schema" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "jing" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "saxon" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "csv-validator"
                && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "protoc" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "apache-avro" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "frictionless"
                && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
    }

    #[test]
    fn representative_environment_names_are_stable() {
        let minizinc = find_external_validation_tool("minizinc").unwrap();
        assert_eq!(
            external_validation_adapter_env_names(minizinc)[0],
            "ORES_MINIZINC_ADAPTER"
        );
        let miplib = find_external_validation_tool("miplib").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(miplib)[0],
            "ORES_MIPLIB_DATA_DIR"
        );
        let tlc = find_external_validation_tool("tlc").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(tlc)[0],
            "ORES_TLC_CLASSPATH"
        );
        let jpf = find_external_validation_tool("java-pathfinder").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(jpf)[0],
            "ORES_JAVA_PATHFINDER_CLASSPATH"
        );
        let carla = find_external_validation_tool("carla").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(carla)[0],
            "ORES_CARLA_DIR"
        );
        let check_jsonschema = find_external_validation_tool("check-jsonschema").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(check_jsonschema)[0],
            "ORES_CHECK_JSONSCHEMA_PYTHON"
        );
        let sumo = find_external_validation_tool("sumo").unwrap();
        assert!(
            external_validation_command_dir_env_names(sumo).contains(&"ORES_SUMO_DIR".to_string())
        );
        assert!(external_validation_command_dir_env_names(sumo).contains(&"SUMO_HOME".to_string()));
        let minizinc = find_external_validation_tool("minizinc").unwrap();
        assert!(external_validation_command_dir_env_names(minizinc)
            .contains(&"MINIZINC_HOME".to_string()));
        let choco = find_external_validation_tool("choco_solver").unwrap();
        assert_eq!(choco.id, "choco-solver");
        assert_eq!(
            external_validation_artifact_env_names(choco)[0],
            "ORES_CHOCO_SOLVER_CLASSPATH"
        );
        assert!(external_validation_command_dir_env_names(choco)
            .contains(&"CHOCO_SOLVER_HOME".to_string()));
        let cp_optimizer = find_external_validation_tool("ibm_cp_optimizer").unwrap();
        assert_eq!(cp_optimizer.id, "ibm-cp-optimizer");
        assert!(external_validation_command_dir_env_names(cp_optimizer)
            .contains(&"CPLEX_STUDIO_DIR".to_string()));
        let timefold = find_external_validation_tool("timefold").unwrap();
        assert!(external_validation_command_dir_env_names(timefold)
            .contains(&"TIMEFOLD_HOME".to_string()));
        let moea = find_external_validation_tool("moea_framework").unwrap();
        assert_eq!(moea.id, "moea-framework");
        assert!(external_validation_command_dir_env_names(moea)
            .contains(&"MOEA_FRAMEWORK_HOME".to_string()));
        let good_lp = find_external_validation_tool("good_lp").unwrap();
        assert_eq!(good_lp.id, "good-lp");
        assert_eq!(
            external_validation_artifact_env_names(good_lp)[0],
            "ORES_GOOD_LP_CRATE"
        );
        let argmin = find_external_validation_tool("argmin").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(argmin)[0],
            "ORES_ARGMIN_CRATE"
        );
        let nlopt_rs = find_external_validation_tool("nlopt_rs").unwrap();
        assert_eq!(nlopt_rs.id, "nlopt-rs");
        assert!(
            external_validation_command_dir_env_names(nlopt_rs).contains(&"NLOPT_HOME".to_string())
        );
        let gurobi_rust = find_external_validation_tool("gurobi_rust").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(gurobi_rust)[0],
            "ORES_GUROBI_RUST_CRATE"
        );
        assert!(external_validation_artifact_env_names(gurobi_rust)
            .contains(&"GRB_LICENSE_FILE".to_string()));
        assert!(external_validation_command_dir_env_names(gurobi_rust)
            .contains(&"GUROBI_HOME".to_string()));
        let cplex_rust = find_external_validation_tool("cplex_rust").unwrap();
        assert!(external_validation_artifact_env_names(cplex_rust)
            .contains(&"CPLEX_RUST_CARGO_MANIFEST".to_string()));
        assert!(external_validation_command_dir_env_names(cplex_rust)
            .contains(&"CPLEX_STUDIO_DIR".to_string()));
        let ipopt_rust = find_external_validation_tool("ipopt_rust").unwrap();
        assert!(external_validation_artifact_env_names(ipopt_rust)
            .contains(&"IPOPT_RUST_CARGO_MANIFEST".to_string()));
        assert!(external_validation_command_dir_env_names(ipopt_rust)
            .contains(&"IPOPT_HOME".to_string()));
        let nlopt = find_external_validation_tool("nlopt").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(nlopt)[0],
            "ORES_NLOPT_DIR"
        );
        let highs_rust = find_external_validation_tool("highs_rust").unwrap();
        assert!(external_validation_command_dir_env_names(highs_rust)
            .contains(&"HIGHS_HOME".to_string()));
        let scip_rust = find_external_validation_tool("scip_rust").unwrap();
        assert!(external_validation_command_dir_env_names(scip_rust)
            .contains(&"SCIPOPTDIR".to_string()));
        let cbc_rust = find_external_validation_tool("cbc_rust").unwrap();
        assert!(external_validation_command_dir_env_names(cbc_rust)
            .contains(&"COINOR_HOME".to_string()));
        let cpmpy = find_external_validation_tool("cpmpy").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(cpmpy)[0],
            "ORES_CPMPY_PYTHON"
        );
        assert!(
            external_validation_command_dir_env_names(cpmpy).contains(&"CPMPY_HOME".to_string())
        );
        let pyomo = find_external_validation_tool("pyomo").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(pyomo)[0],
            "ORES_PYOMO_PYTHON"
        );
        let pyscipopt = find_external_validation_tool("pyscipopt").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(pyscipopt)[0],
            "ORES_PYSCIPOPT_PYTHON"
        );
        assert!(external_validation_command_dir_env_names(pyscipopt)
            .contains(&"SCIPOPTDIR".to_string()));
        let ortools_glop = find_external_validation_tool("ortools_glop").unwrap();
        assert_eq!(ortools_glop.id, "ortools-glop");
        assert!(external_validation_artifact_env_names(ortools_glop)
            .contains(&"ORTOOLS_PYTHON".to_string()));
        let ortools_cp_sat = find_external_validation_tool("ortools_cp_sat").unwrap();
        assert_eq!(ortools_cp_sat.id, "ortools-cp-sat");
        assert!(external_validation_artifact_env_names(ortools_cp_sat)
            .contains(&"FZN_CP_SAT_CMD".to_string()));
        assert!(external_validation_command_dir_env_names(ortools_cp_sat)
            .contains(&"ORTOOLS_HOME".to_string()));
        let scipy_optimize = find_external_validation_tool("scipy_optimize").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(scipy_optimize)[0],
            "ORES_SCIPY_OPTIMIZE_PYTHON"
        );
        let gurobipy = find_external_validation_tool("gurobipy").unwrap();
        assert!(external_validation_command_dir_env_names(gurobipy)
            .contains(&"GUROBI_HOME".to_string()));
        let cplex_python = find_external_validation_tool("cplex_python").unwrap();
        assert_eq!(cplex_python.id, "cplex-python");
        assert!(external_validation_artifact_env_names(cplex_python)
            .contains(&"CPLEX_STUDIO_DIR".to_string()));
        assert!(external_validation_command_dir_env_names(cplex_python)
            .contains(&"CPLEX_HOME".to_string()));
        let xpress_python = find_external_validation_tool("xpress_python").unwrap();
        assert_eq!(xpress_python.id, "xpress-python");
        assert!(external_validation_artifact_env_names(xpress_python)
            .contains(&"XPRESSDIR".to_string()));
        assert!(external_validation_command_dir_env_names(xpress_python)
            .contains(&"XPRESS_HOME".to_string()));
        let highs_cli = find_external_validation_tool("highs_cli").unwrap();
        assert_eq!(highs_cli.id, "highs-cli");
        assert!(
            external_validation_artifact_env_names(highs_cli).contains(&"HIGHS_CMD".to_string())
        );
        let glpk_cli = find_external_validation_tool("glpk_cli").unwrap();
        assert!(
            external_validation_artifact_env_names(glpk_cli).contains(&"GLPSOL_CMD".to_string())
        );
        let soplex_cli = find_external_validation_tool("soplex_cli").unwrap();
        assert_eq!(soplex_cli.id, "soplex-cli");
        assert!(
            external_validation_artifact_env_names(soplex_cli).contains(&"SOPLEX_CMD".to_string())
        );
        let qsopt_ex_cli = find_external_validation_tool("qsopt_ex_cli").unwrap();
        assert_eq!(qsopt_ex_cli.id, "qsopt-ex-cli");
        assert!(external_validation_artifact_env_names(qsopt_ex_cli)
            .contains(&"QSOPT_EX_CMD".to_string()));
        let lp_solve_cli = find_external_validation_tool("lp_solve_cli").unwrap();
        assert_eq!(lp_solve_cli.id, "lp-solve-cli");
        assert!(external_validation_artifact_env_names(lp_solve_cli)
            .contains(&"LP_SOLVE_CMD".to_string()));
        let gurobi_cli = find_external_validation_tool("gurobi_cli").unwrap();
        assert!(external_validation_artifact_env_names(gurobi_cli)
            .contains(&"GUROBI_CL_CMD".to_string()));
        let cplex_cli = find_external_validation_tool("cplex_cli").unwrap();
        assert!(external_validation_command_dir_env_names(cplex_cli)
            .contains(&"CPLEX_STUDIO_DIR".to_string()));
        let xpress_cli = find_external_validation_tool("xpress_cli").unwrap();
        assert!(external_validation_command_dir_env_names(xpress_cli)
            .contains(&"XPRESSDIR".to_string()));
        let lindo_cli = find_external_validation_tool("lindo_cli").unwrap();
        assert!(
            external_validation_artifact_env_names(lindo_cli).contains(&"LINDOAPI_CMD".to_string())
        );
        let ampl = find_external_validation_tool("ampl").unwrap();
        assert!(external_validation_command_dir_env_names(ampl).contains(&"AMPL_HOME".to_string()));
        let hexaly = find_external_validation_tool("hexaly").unwrap();
        assert!(
            external_validation_command_dir_env_names(hexaly).contains(&"HEXALY_HOME".to_string())
        );
        let cvxopt = find_external_validation_tool("cvxopt").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(cvxopt)[0],
            "ORES_CVXOPT_PYTHON"
        );
        let pddl_val = find_external_validation_tool("pddl_val").unwrap();
        assert_eq!(pddl_val.id, "pddl-val");
        assert!(
            external_validation_command_dir_env_names(pddl_val).contains(&"VAL_HOME".to_string())
        );
        let fast_downward = find_external_validation_tool("fast_downward").unwrap();
        assert_eq!(fast_downward.id, "fast-downward");
        assert!(external_validation_command_dir_env_names(fast_downward)
            .contains(&"FAST_DOWNWARD_HOME".to_string()));
        let conjure = find_external_validation_tool("conjure").unwrap();
        assert!(external_validation_command_dir_env_names(conjure)
            .contains(&"ORES_CONJURE_DIR".to_string()));
        assert!(external_validation_command_dir_env_names(conjure)
            .contains(&"CONJURE_HOME".to_string()));
        let open_wbo = find_external_validation_tool("open_wbo").unwrap();
        assert_eq!(open_wbo.id, "open-wbo");
        assert!(external_validation_command_dir_env_names(open_wbo)
            .contains(&"OPEN_WBO_HOME".to_string()));
        let maxhs = find_external_validation_tool("maxhs").unwrap();
        assert!(
            external_validation_command_dir_env_names(maxhs).contains(&"MAXHS_HOME".to_string())
        );
        let roundingsat = find_external_validation_tool("roundingsat").unwrap();
        assert!(external_validation_command_dir_env_names(roundingsat)
            .contains(&"ROUNDINGSAT_HOME".to_string()));
        let veripb = find_external_validation_tool("veripb").unwrap();
        assert!(
            external_validation_command_dir_env_names(veripb).contains(&"VERIPB_HOME".to_string())
        );
        let klee = find_external_validation_tool("klee").unwrap();
        assert!(external_validation_command_dir_env_names(klee).contains(&"KLEE_HOME".to_string()));
        let copt = find_external_validation_tool("copt").unwrap();
        assert!(external_validation_command_dir_env_names(copt).contains(&"COPT_HOME".to_string()));
        let mosek = find_external_validation_tool("mosek").unwrap();
        assert!(
            external_validation_command_dir_env_names(mosek).contains(&"MOSEK_HOME".to_string())
        );
        assert!(!external_validation_command_dir_env_names(mosek)
            .contains(&"MOSEKLM_LICENSE_FILE".to_string()));
        let prism = find_external_validation_tool("prism").unwrap();
        assert!(
            external_validation_command_dir_env_names(prism).contains(&"PRISM_HOME".to_string())
        );
        let jpf = find_external_validation_tool("java-pathfinder").unwrap();
        assert!(external_validation_command_dir_env_names(jpf).contains(&"JPF_HOME".to_string()));
        let ptolemy = find_external_validation_tool("ptolemy-ii").unwrap();
        assert!(external_validation_command_dir_env_names(ptolemy).contains(&"PTII".to_string()));
        let jaamsim = find_external_validation_tool("jaamsim").unwrap();
        assert!(external_validation_command_dir_env_names(jaamsim)
            .contains(&"JAAMSIM_HOME".to_string()));
        let desmoj = find_external_validation_tool("desmo_j").unwrap();
        assert_eq!(desmoj.id, "desmo-j");
        assert!(
            external_validation_command_dir_env_names(desmoj).contains(&"DESMOJ_HOME".to_string())
        );
        let proverif = find_external_validation_tool("proverif").unwrap();
        assert!(external_validation_command_dir_env_names(proverif)
            .contains(&"PROVERIF_HOME".to_string()));
    }

    #[test]
    fn install_dir_lookup_handles_validation_tool_bin_layouts() {
        let root = std::env::temp_dir().join(format!(
            "des-external-validation-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let command = root
            .join("webots")
            .join("bin")
            .join("x86-64_osx")
            .join("webots");
        std::fs::create_dir_all(command.parent().unwrap()).unwrap();
        std::fs::write(&command, b"").unwrap();

        assert_eq!(
            find_command_in_install_dir(&root, &["webots"]),
            Some(command)
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn install_dir_lookup_builds_java_classpath_layouts() {
        let root = std::env::temp_dir().join(format!(
            "des-external-validation-java-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let share_jar = root.join("share").join("java").join("jaamsim.jar");
        let nested_jar = root.join("ptolemy").join("lib").join("ptolemy.jar");
        let non_jar = root.join("lib").join("README.txt");
        std::fs::create_dir_all(share_jar.parent().unwrap()).unwrap();
        std::fs::create_dir_all(nested_jar.parent().unwrap()).unwrap();
        std::fs::create_dir_all(non_jar.parent().unwrap()).unwrap();
        std::fs::write(&share_jar, b"").unwrap();
        std::fs::write(&nested_jar, b"").unwrap();
        std::fs::write(&non_jar, b"").unwrap();

        let classpath = find_java_classpath_in_install_dir(&root).unwrap();
        let paths: Vec<PathBuf> = std::env::split_paths(&classpath).collect();
        assert!(paths.contains(&share_jar));
        assert!(paths.contains(&nested_jar));
        assert!(!paths.contains(&non_jar));
        assert!(is_jar_file(&share_jar));
        assert!(!is_jar_file(&non_jar));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_install_dir_does_not_make_java_classpath() {
        let root = std::env::temp_dir().join(format!(
            "des-external-validation-empty-java-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();

        assert!(find_java_classpath_in_install_dir(&root).is_none());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ids_can_be_found_with_hyphen_or_underscore_spelling() {
        assert_eq!(
            find_external_validation_tool("json_schema").unwrap().id,
            "json-schema"
        );
        assert_eq!(
            find_external_validation_tool("MINIZINC").unwrap().id,
            "minizinc"
        );
        assert_eq!(
            find_external_validation_tool("tensorflow_data_validation")
                .unwrap()
                .id,
            "tensorflow-data-validation"
        );
        assert_eq!(
            find_external_validation_tool("java_pathfinder").unwrap().id,
            "java-pathfinder"
        );
        assert_eq!(
            find_external_validation_tool("apache_arrow").unwrap().id,
            "apache-arrow"
        );
        assert!(find_external_validation_tool("not-a-tool").is_none());
    }

    #[test]
    fn status_and_capability_strings_are_stable() {
        assert_eq!(ExternalValidationProbeStatus::Ready.as_str(), "ready");
        assert_eq!(
            ExternalValidationProbeStatus::ArtifactMissing.as_str(),
            "artifact-missing"
        );
        assert_eq!(ExternalValidationRunStatus::Ok.as_str(), "ok");
        assert_eq!(
            ExternalValidationRunStatus::InvalidOutput.as_str(),
            "invalid-output"
        );
        assert_eq!(
            ExternalValidationCapability::CheckSatisfiability.as_str(),
            "check-satisfiability"
        );
        assert_eq!(
            ExternalValidationFamily::FormalModelChecker.as_str(),
            "formal-model-checker"
        );
        assert_eq!(
            ExternalValidationFamily::ConvexConicSolver.as_str(),
            "convex-conic-solver"
        );
    }

    #[test]
    fn minizinc_and_json_schema_requests_are_stable_json_contracts() {
        let minizinc = minizinc_validation_request_to_json(&MiniZincValidationRequest {
            model: "var 0..10: x; constraint x >= 3; solve satisfy;".to_string(),
            data: Some("x_max = 10;".to_string()),
            solver: Some("chuffed".to_string()),
            checker_model: Some("constraint x >= 3;".to_string()),
        });
        assert_eq!(minizinc["kind"], "minizinc-validation");
        assert_eq!(minizinc["format"], "mzn");
        assert_eq!(minizinc["solver"], "chuffed");
        assert!(minizinc["model"]
            .as_str()
            .unwrap()
            .contains("solve satisfy"));

        let schema = json_schema_validation_request_to_json(&JsonSchemaValidationRequest {
            schema: json!({
                "type": "object",
                "required": ["objective"],
                "properties": {"objective": {"type": "number"}}
            }),
            instance: json!({"objective": 42.0}),
            draft: Some("2020-12".to_string()),
        });
        assert_eq!(schema["kind"], "json-schema-validation");
        assert_eq!(schema["draft"], "2020-12");
        assert_eq!(schema["instance"]["objective"], 42.0);
    }

    #[test]
    fn smtlib_and_dimacs_exporters_emit_solver_ready_text() {
        let smtlib = smtlib_validation_script_to_string(&SmtLibValidationScript {
            logic: Some("QF_LIA".to_string()),
            declarations: vec![
                SmtDeclaration {
                    name: "x".to_string(),
                    sort: SmtSort::Int,
                },
                SmtDeclaration {
                    name: "flag".to_string(),
                    sort: SmtSort::Bool,
                },
            ],
            assertions: vec![">= x 0".to_string(), "(assert flag)".to_string()],
            check_sat_assumptions: Vec::new(),
            get_model: true,
        });
        assert!(smtlib.contains("(set-logic QF_LIA)"));
        assert!(smtlib.contains("(declare-const x Int)"));
        assert!(smtlib.contains("(assert (>= x 0))"));
        assert!(smtlib.contains("(assert flag)"));
        assert!(smtlib.contains("(check-sat)"));
        assert!(smtlib.contains("(get-model)"));

        let cnf = dimacs_cnf_to_string(&DimacsCnf {
            num_vars: 3,
            clauses: vec![vec![1, -2], vec![2, 3]],
            comments: vec!["tiny satisfiable cross-check".to_string()],
        });
        assert!(cnf.starts_with("c tiny satisfiable cross-check\np cnf 3 2\n"));
        assert!(cnf.contains("1 -2 0\n"));
        assert!(cnf.contains("2 3 0\n"));

        let wcnf = dimacs_wcnf_to_string(&DimacsWcnf {
            num_vars: 2,
            clauses: vec![DimacsWeightedClause {
                weight: 7,
                literals: vec![1, -2],
            }],
            top_weight: Some(100),
            comments: Vec::new(),
        });
        assert_eq!(wcnf, "p wcnf 2 1 100\n7 1 -2 0\n");
    }

    #[test]
    fn formal_model_exporters_emit_tla_and_prism_text() {
        let tla = tla_validation_module_to_string(&TlaValidationModule {
            module_name: "Counter".to_string(),
            extends: vec!["Naturals".to_string(), "TLC".to_string()],
            constants: vec!["Limit".to_string()],
            variables: vec!["x".to_string()],
            init: "x = 0".to_string(),
            next: "x' = x + 1".to_string(),
            invariants: vec!["x <= Limit".to_string()],
            temporal_properties: vec!["[]Invariant1".to_string()],
        });
        assert!(tla.contains("---- MODULE Counter ----"));
        assert!(tla.contains("EXTENDS Naturals, TLC"));
        assert!(tla.contains("Invariant1 == x <= Limit"));
        assert!(tla.contains("Spec == Init /\\ [][Next]_x"));

        let prism = PrismValidationModel {
            model_type: "dtmc".to_string(),
            declarations: vec!["const double p = 0.5;".to_string()],
            modules: vec![PrismModule {
                name: "coin".to_string(),
                variables: vec!["s : [0..1] init 0;".to_string()],
                commands: vec!["[] s=0 -> p:(s'=0) + (1-p):(s'=1);".to_string()],
            }],
            labels: vec!["label \"done\" = s=1;".to_string()],
            properties: vec!["P>=0.4 [ F \"done\" ]".to_string()],
        };
        let model_text = prism_validation_model_to_string(&prism);
        assert!(model_text.contains("dtmc"));
        assert!(model_text.contains("module coin"));
        assert!(model_text.contains("endmodule"));
        assert!(model_text.contains("label \"done\" = s=1;"));
        assert_eq!(
            prism_validation_properties_to_string(&prism),
            "P>=0.4 [ F \"done\" ]\n"
        );
    }

    #[test]
    fn simulation_and_benchmark_requests_are_stable_json_contracts() {
        let simulation = simulation_validation_request_to_json(&SimulationValidationRequest {
            engine_id: "simpy".to_string(),
            model_format: "json-event-network".to_string(),
            model: json!({"servers": 2, "arrival_rate": 1.5}),
            scenario: Some(json!({"seed": 7, "horizon": 1000.0})),
            expected_trace_properties: vec!["queue_length_never_negative".to_string()],
            metric_expectations: vec![SimulationMetricExpectation {
                name: "mean_wait".to_string(),
                target: 2.0,
                tolerance: 0.25,
                comparison: "within-absolute".to_string(),
            }],
        });
        assert_eq!(simulation["kind"], "simulation-validation");
        assert_eq!(simulation["engine"], "simpy");
        assert_eq!(simulation["scenario"]["seed"], 7);
        assert_eq!(
            simulation["metric_expectations"][0]["comparison"],
            "within-absolute"
        );

        let manifest = external_benchmark_manifest_to_json(&ExternalBenchmarkManifest {
            suite: "miplib".to_string(),
            version: Some("2017".to_string()),
            entries: vec![ExternalBenchmarkManifestEntry {
                name: "sample".to_string(),
                family: "mip".to_string(),
                format: "mps".to_string(),
                path: PathBuf::from("MIPLIB/sample.mps"),
                objective_sense: Some("min".to_string()),
                tags: vec!["smoke".to_string()],
            }],
        });
        assert_eq!(manifest["kind"], "external-benchmark-manifest");
        assert_eq!(manifest["suite"], "miplib");
        assert_eq!(manifest["entries"][0]["path"], "MIPLIB/sample.mps");
        assert_eq!(manifest["entries"][0]["tags"][0], "smoke");
    }

    #[test]
    fn simulation_validation_manifest_covers_external_engine_families() {
        let specs = external_simulation_validation_tool_specs();
        let manifest = external_simulation_validation_engine_manifest();

        assert!(specs.len() >= 40);
        for id in [
            "simpy",
            "sumo",
            "energyplus",
            "openmodelica",
            "fmi-fmu",
            "mujoco",
            "mesa",
            "simgrid",
            "neqsim",
            "anylogic",
            "arena",
        ] {
            assert!(specs.iter().any(|spec| spec.id == id), "missing {id}");
        }
        assert_eq!(manifest.as_array().map(Vec::len), Some(specs.len()));
    }

    #[test]
    fn simulation_validation_reference_wrapper_runs_valid_and_invalid_cases() {
        let valid = SimulationValidationRequest {
            engine_id: "simpy".to_string(),
            model_format: "json-event-network".to_string(),
            model: json!({
                "servers": 1,
                "arrival_times": [0.0, 1.0, 2.0],
                "service_times": [1.0, 1.0, 1.0]
            }),
            scenario: Some(json!({"horizon": 10.0})),
            expected_trace_properties: vec![
                "queue_length_never_negative".to_string(),
                "departures_after_arrivals".to_string(),
            ],
            metric_expectations: vec![SimulationMetricExpectation {
                name: "jobs_completed".to_string(),
                target: 3.0,
                tolerance: 1e-9,
                comparison: "equal".to_string(),
            }],
        };
        let valid_run = run_simulation_validation_with_external_reference(
            &valid,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(valid_run.status, ExternalSimulationValidationStatus::Ok);
        assert_eq!(
            valid_run.verdict,
            ExternalSimulationValidationVerdict::Valid
        );
        assert!(valid_run
            .simulator
            .starts_with("rust:single-station-des-for-simpy"));
        assert_eq!(valid_run.metrics.get("jobs_completed").copied(), Some(3.0));
        assert_eq!(valid_run.trace.len(), 9);

        let mut invalid = valid;
        invalid.metric_expectations = vec![SimulationMetricExpectation {
            name: "mean_wait".to_string(),
            target: 2.0,
            tolerance: 0.1,
            comparison: "within-absolute".to_string(),
        }];
        let invalid_run = run_simulation_validation_with_external_reference(
            &invalid,
            &ExternalSimulationValidationReferenceOptions {
                engine_id: Some("arena".to_string()),
            },
        );
        assert_eq!(invalid_run.status, ExternalSimulationValidationStatus::Ok);
        assert_eq!(
            invalid_run.verdict,
            ExternalSimulationValidationVerdict::Invalid
        );
        assert!(invalid_run
            .simulator
            .starts_with("rust:single-station-des-for-arena"));
        assert!(invalid_run.simulator.contains("arena"));
        assert!(invalid_run
            .checks
            .iter()
            .any(|check| check["name"] == "mean_wait" && check["passed"] == false));
    }

    #[test]
    fn simulation_validation_reference_wrapper_runs_mobility_in_rust() {
        let payload = json!({
            "kind": "simulation-validation",
            "engine": "sumo",
            "model_format": "json-mobility-network",
            "model": {
                "routes": [
                    {"depart": 0.0, "travel_times": [2.0, 3.0]},
                    {"depart": 1.0, "segments": [{"travel_time": 1.5}, {"travel_time": 2.5}]}
                ]
            },
            "expected_trace_properties": [
                "departures_before_arrivals",
                "travel_times_nonnegative",
                "vehicles_complete"
            ],
            "metric_expectations": [
                {"name": "vehicles_completed", "target": 2.0, "tolerance": 1e-9, "comparison": "equal"},
                {"name": "mean_travel_time", "target": 4.5, "tolerance": 1e-9, "comparison": "within-absolute"}
            ]
        });
        let run = run_simulation_validation_json_with_external_reference(
            &payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(run.status, ExternalSimulationValidationStatus::Ok);
        assert_eq!(run.verdict, ExternalSimulationValidationVerdict::Valid);
        assert!(run.simulator.starts_with("rust:mobility-network-for-sumo"));
        assert_eq!(run.metrics.get("mean_travel_time").copied(), Some(4.5));
        assert_eq!(run.metrics.get("vehicles_completed").copied(), Some(2.0));
        assert_eq!(run.trace.len(), 8);
    }

    #[test]
    fn simulation_validation_reference_wrapper_runs_remaining_formats_in_rust() {
        let energy_payload = json!({
            "kind": "simulation-validation",
            "engine": "energyplus",
            "model_format": "json-energy-balance",
            "model": {
                "initial_temp": 20.0,
                "setpoint": 21.0,
                "outdoor_temp": 10.0,
                "ua": 0.1,
                "heat_capacity": 10.0,
                "hvac_power": 2.0,
                "internal_gain": 0.1
            },
            "scenario": {"horizon": 2.0, "step": 1.0},
            "expected_trace_properties": [
                "energy_nonnegative",
                "temperatures_finite",
                "temperature_within_bounds"
            ],
            "metric_expectations": [
                {"name": "zones", "target": 1.0, "tolerance": 1e-9, "comparison": "equal"},
                {"name": "energy_kwh", "target": 0.0, "tolerance": 10.0, "comparison": "greater-equal"}
            ]
        });
        let energy_run = run_simulation_validation_json_with_external_reference(
            &energy_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(energy_run.status, ExternalSimulationValidationStatus::Ok);
        assert_eq!(
            energy_run.verdict,
            ExternalSimulationValidationVerdict::Valid
        );
        assert!(energy_run
            .simulator
            .starts_with("rust:energy-balance-for-energyplus"));
        assert_eq!(energy_run.metrics.get("zones").copied(), Some(1.0));

        let physics_payload = json!({
            "kind": "simulation-validation",
            "engine": "mujoco",
            "model_format": "json-physics-trajectory",
            "model": {
                "initial_position": 0.0,
                "initial_velocity": 0.0,
                "acceleration": 1.0,
                "floor": 0.0
            },
            "scenario": {"dt": 0.5, "steps": 4},
            "expected_trace_properties": [
                "positions_finite",
                "velocities_finite",
                "path_length_nonnegative",
                "stays_above_floor"
            ],
            "metric_expectations": [
                {"name": "final_position", "target": 2.5, "tolerance": 1e-9, "comparison": "within-absolute"},
                {"name": "final_velocity", "target": 2.0, "tolerance": 1e-9, "comparison": "within-absolute"}
            ]
        });
        let physics_run = run_simulation_validation_json_with_external_reference(
            &physics_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(physics_run.status, ExternalSimulationValidationStatus::Ok);
        assert_eq!(
            physics_run.verdict,
            ExternalSimulationValidationVerdict::Valid
        );
        assert!(physics_run
            .simulator
            .starts_with("rust:physics-trajectory-for-mujoco"));
        assert_eq!(
            physics_run.metrics.get("final_position").copied(),
            Some(2.5)
        );
        assert_eq!(
            physics_run.metrics.get("final_velocity").copied(),
            Some(2.0)
        );

        let agent_payload = json!({
            "kind": "simulation-validation",
            "engine": "mesa",
            "model_format": "json-agent-based",
            "model": {
                "agents": [{"state": "idle"}, {"state": "busy"}],
                "interactions": [{"source": 0, "target": 1}]
            },
            "scenario": {"steps": 2},
            "expected_trace_properties": [
                "agents_nonempty",
                "states_present",
                "interactions_reference_agents"
            ]
        });
        let agent_run = run_simulation_validation_json_with_external_reference(
            &agent_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(agent_run.status, ExternalSimulationValidationStatus::Ok);
        assert_eq!(
            agent_run.verdict,
            ExternalSimulationValidationVerdict::Valid
        );
        assert!(agent_run.simulator.starts_with("rust:agent-based-for-mesa"));
        assert_eq!(agent_run.metrics.get("agents").copied(), Some(2.0));

        let distributed_payload = json!({
            "kind": "simulation-validation",
            "engine": "simgrid",
            "model_format": "json-distributed-system",
            "model": {
                "hosts": [{"capacity": 4}],
                "links": [{"bandwidth": 10}],
                "tasks": [{"work": 3}]
            },
            "expected_trace_properties": [
                "hosts_have_capacity",
                "links_nonnegative",
                "tasks_schedulable"
            ]
        });
        let distributed_run = run_simulation_validation_json_with_external_reference(
            &distributed_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(
            distributed_run.status,
            ExternalSimulationValidationStatus::Ok
        );
        assert_eq!(
            distributed_run.verdict,
            ExternalSimulationValidationVerdict::Valid
        );
        assert!(distributed_run
            .simulator
            .starts_with("rust:distributed-system-for-simgrid"));
        assert_eq!(distributed_run.metrics.get("hosts").copied(), Some(1.0));

        let process_payload = json!({
            "kind": "simulation-validation",
            "engine": "neqsim",
            "model_format": "json-process-flow",
            "model": {
                "units": [{"name": "mixer"}],
                "streams": [
                    {"from": "source", "to": "mixer", "flow": 5},
                    {"from": "mixer", "to": "sink", "flow": 5}
                ]
            },
            "expected_trace_properties": [
                "units_present",
                "streams_nonnegative",
                "mass_balance_closed"
            ]
        });
        let process_run = run_simulation_validation_json_with_external_reference(
            &process_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(process_run.status, ExternalSimulationValidationStatus::Ok);
        assert_eq!(
            process_run.verdict,
            ExternalSimulationValidationVerdict::Valid
        );
        assert!(process_run
            .simulator
            .starts_with("rust:process-flow-for-neqsim"));
        assert_eq!(
            process_run.metrics.get("mass_balance_error").copied(),
            Some(0.0)
        );
    }

    #[test]
    fn text_cli_profiles_and_verdict_inference_are_stable() {
        assert_eq!(
            external_validation_default_text_cli_args("z3", ExternalValidationTextFormat::SmtLib2),
            &["-in", "-smt2"]
        );
        assert_eq!(
            external_validation_default_text_cli_args(
                "mathsat",
                ExternalValidationTextFormat::SmtLib2
            ),
            &["-input=smt2"]
        );
        assert_eq!(
            external_validation_default_text_cli_args(
                "kissat",
                ExternalValidationTextFormat::DimacsCnf
            ),
            &["-"]
        );
        assert_eq!(
            external_validation_default_text_cli_args("tlc", ExternalValidationTextFormat::TlaPlus),
            &[] as &[&str]
        );
        assert_eq!(
            infer_external_validation_text_verdict(
                ExternalValidationTextFormat::SmtLib2,
                "sat\n(model)\n",
                "",
                true
            ),
            ExternalValidationTextVerdict::Sat
        );
        assert_eq!(
            infer_external_validation_text_verdict(
                ExternalValidationTextFormat::DimacsCnf,
                "UNSATISFIABLE\n",
                "",
                true
            ),
            ExternalValidationTextVerdict::Unsat
        );
        assert_eq!(
            infer_external_validation_text_verdict(
                ExternalValidationTextFormat::TlaPlus,
                "Invariant violated. Counterexample follows.",
                "",
                true
            ),
            ExternalValidationTextVerdict::Invalid
        );
        assert_eq!(
            infer_external_validation_text_verdict(
                ExternalValidationTextFormat::PrismModel,
                "Property is satisfied: true",
                "",
                true
            ),
            ExternalValidationTextVerdict::Valid
        );
        assert_eq!(
            infer_external_validation_text_verdict(
                ExternalValidationTextFormat::Json,
                "",
                "schema parser failed",
                false
            ),
            ExternalValidationTextVerdict::Failure
        );
    }

    #[test]
    fn text_cli_runner_feeds_stdin_and_returns_normalized_payload() {
        let run = run_external_validation_text_cli(
            "sat\n",
            &ExternalValidationTextCliOptions {
                tool_id: "z3".to_string(),
                input_format: ExternalValidationTextFormat::SmtLib2,
                command_path: Some(PathBuf::from("/bin/cat")),
                working_dir: None,
                extra_args: Vec::new(),
                use_default_args: false,
            },
        );
        assert_eq!(run.status, ExternalValidationRunStatus::Ok);
        let output = run.output.expect("text CLI output payload");
        assert_eq!(output["kind"], "external-validation-text-cli-run");
        assert_eq!(output["tool"], "z3");
        assert_eq!(output["format"], "smtlib2");
        assert_eq!(output["verdict"], "sat");
        assert_eq!(output["stdout"], "sat\n");
        assert_eq!(output["exit_success"], true);
    }

    #[test]
    fn file_cli_profiles_and_placeholder_args_are_stable() {
        let input = PathBuf::from("/tmp/model.smt2");
        assert_eq!(
            external_validation_default_file_cli_args(
                "z3",
                ExternalValidationTextFormat::SmtLib2,
                &input,
            ),
            vec!["-smt2".to_string(), "/tmp/model.smt2".to_string()]
        );
        assert_eq!(
            external_validation_default_file_cli_args(
                "optimathsat",
                ExternalValidationTextFormat::SmtLib2,
                &input,
            ),
            vec!["-input=smt2".to_string(), "/tmp/model.smt2".to_string()]
        );
        assert_eq!(
            external_validation_default_file_cli_args(
                "kissat",
                ExternalValidationTextFormat::DimacsCnf,
                &input,
            ),
            vec!["/tmp/model.smt2".to_string()]
        );
        let args = external_validation_file_cli_args(
            &ExternalValidationFileCliOptions {
                tool_id: "z3".to_string(),
                input_format: ExternalValidationTextFormat::SmtLib2,
                command_path: None,
                working_dir: None,
                extra_args: vec!["--model={input}".to_string()],
                use_default_args: false,
                append_input_path: true,
                file_extension: None,
            },
            &input,
        );
        assert_eq!(args, vec!["--model=/tmp/model.smt2".to_string()]);
        assert_eq!(
            ExternalValidationTextFormat::MiniZinc.file_extension(),
            "mzn"
        );
    }

    #[test]
    fn file_cli_runner_writes_temp_file_and_returns_normalized_payload() {
        let run = run_external_validation_file_cli(
            "sat\n",
            &ExternalValidationFileCliOptions {
                tool_id: "z3".to_string(),
                input_format: ExternalValidationTextFormat::SmtLib2,
                command_path: Some(PathBuf::from("/bin/cat")),
                working_dir: None,
                extra_args: Vec::new(),
                use_default_args: false,
                append_input_path: true,
                file_extension: None,
            },
        );
        assert_eq!(run.status, ExternalValidationRunStatus::Ok);
        let output = run.output.expect("file CLI output payload");
        assert_eq!(output["kind"], "external-validation-file-cli-run");
        assert_eq!(output["tool"], "z3");
        assert_eq!(output["format"], "smtlib2");
        assert_eq!(output["verdict"], "sat");
        assert_eq!(output["stdout"], "sat\n");
        assert_eq!(output["exit_success"], true);
        assert_eq!(output["temp_file_removed"], true);
        let input_path = output["input_path"].as_str().unwrap();
        assert!(!PathBuf::from(input_path).exists());
    }

    #[test]
    fn artifact_cli_profiles_and_placeholders_are_stable() {
        let mut paths = BTreeMap::new();
        paths.insert("model".to_string(), PathBuf::from("/tmp/model.pm"));
        paths.insert("properties".to_string(), PathBuf::from("/tmp/model.pctl"));
        paths.insert("proof".to_string(), PathBuf::from("/tmp/proof.drat"));
        paths.insert("cnf".to_string(), PathBuf::from("/tmp/problem.cnf"));

        assert_eq!(
            external_validation_default_artifact_cli_args(
                "prism",
                ExternalValidationTextFormat::PrismModel,
                &paths,
            ),
            vec!["/tmp/model.pm".to_string(), "/tmp/model.pctl".to_string()]
        );
        assert_eq!(
            external_validation_default_artifact_cli_args(
                "drat-trim",
                ExternalValidationTextFormat::DimacsCnf,
                &paths,
            ),
            vec![
                "/tmp/problem.cnf".to_string(),
                "/tmp/proof.drat".to_string()
            ]
        );
        let args = external_validation_artifact_cli_args(
            &ExternalValidationArtifactCliOptions {
                tool_id: "prism".to_string(),
                input_format: ExternalValidationTextFormat::PrismModel,
                command_path: None,
                working_dir: None,
                extra_args: vec![
                    "--model={model}".to_string(),
                    "--props={properties}".to_string(),
                ],
                use_default_args: false,
            },
            &paths,
        );
        assert_eq!(
            args,
            vec![
                "--model=/tmp/model.pm".to_string(),
                "--props=/tmp/model.pctl".to_string()
            ]
        );
    }

    #[test]
    fn artifact_cli_runner_writes_workspace_and_returns_normalized_payload() {
        let run = run_external_validation_artifact_cli(
            &[ExternalValidationArtifact {
                key: "model".to_string(),
                contents: "sat\n".to_string(),
                file_name: Some("model.smt2".to_string()),
                file_extension: None,
            }],
            &ExternalValidationArtifactCliOptions {
                tool_id: "z3".to_string(),
                input_format: ExternalValidationTextFormat::SmtLib2,
                command_path: Some(PathBuf::from("/bin/cat")),
                working_dir: None,
                extra_args: vec!["{model}".to_string()],
                use_default_args: false,
            },
        );
        assert_eq!(run.status, ExternalValidationRunStatus::Ok);
        let output = run.output.expect("artifact CLI output payload");
        assert_eq!(output["kind"], "external-validation-artifact-cli-run");
        assert_eq!(output["tool"], "z3");
        assert_eq!(output["verdict"], "sat");
        assert_eq!(output["stdout"], "sat\n");
        assert_eq!(output["temp_dir_removed"], true);
        let temp_dir = output["temp_dir"].as_str().unwrap();
        assert!(!PathBuf::from(temp_dir).exists());
    }

    #[test]
    fn consensus_runner_reports_agreement_across_text_and_file_runs() {
        let report = run_external_validation_consensus(
            "sat\n",
            &[
                ExternalValidationCliInvocation::Text {
                    label: "stdin-cat".to_string(),
                    options: ExternalValidationTextCliOptions {
                        tool_id: "z3".to_string(),
                        input_format: ExternalValidationTextFormat::SmtLib2,
                        command_path: Some(PathBuf::from("/bin/cat")),
                        working_dir: None,
                        extra_args: Vec::new(),
                        use_default_args: false,
                    },
                },
                ExternalValidationCliInvocation::File {
                    label: "file-cat".to_string(),
                    options: ExternalValidationFileCliOptions {
                        tool_id: "z3".to_string(),
                        input_format: ExternalValidationTextFormat::SmtLib2,
                        command_path: Some(PathBuf::from("/bin/cat")),
                        working_dir: None,
                        extra_args: Vec::new(),
                        use_default_args: false,
                        append_input_path: true,
                        file_extension: None,
                    },
                },
                ExternalValidationCliInvocation::Artifact {
                    label: "artifact-cat".to_string(),
                    artifacts: vec![ExternalValidationArtifact {
                        key: "model".to_string(),
                        contents: "sat\n".to_string(),
                        file_name: Some("model.smt2".to_string()),
                        file_extension: None,
                    }],
                    options: ExternalValidationArtifactCliOptions {
                        tool_id: "z3".to_string(),
                        input_format: ExternalValidationTextFormat::SmtLib2,
                        command_path: Some(PathBuf::from("/bin/cat")),
                        working_dir: None,
                        extra_args: vec!["{model}".to_string()],
                        use_default_args: false,
                    },
                },
            ],
            Some(ExternalValidationTextVerdict::Sat),
        );
        assert!(report.agreement);
        assert!(report.all_successful);
        assert!(report.all_successful_verdicts_agree);
        assert_eq!(
            report.agreed_verdict,
            Some(ExternalValidationTextVerdict::Sat)
        );
        assert_eq!(report.runs.len(), 3);

        let json = external_validation_consensus_report_to_json(&report);
        assert_eq!(json["kind"], "external-validation-consensus-report");
        assert_eq!(json["agreement"], true);
        assert_eq!(json["agreed_verdict"], "sat");
        assert_eq!(json["runs"][0]["label"], "stdin-cat");
    }

    #[test]
    fn consensus_runner_reports_disagreement_without_hiding_successful_runs() {
        let report = run_external_validation_consensus(
            "sat\n",
            &[
                ExternalValidationCliInvocation::Text {
                    label: "sat-cat".to_string(),
                    options: ExternalValidationTextCliOptions {
                        tool_id: "z3".to_string(),
                        input_format: ExternalValidationTextFormat::SmtLib2,
                        command_path: Some(PathBuf::from("/bin/cat")),
                        working_dir: None,
                        extra_args: Vec::new(),
                        use_default_args: false,
                    },
                },
                ExternalValidationCliInvocation::Text {
                    label: "unsat-echo".to_string(),
                    options: ExternalValidationTextCliOptions {
                        tool_id: "z3".to_string(),
                        input_format: ExternalValidationTextFormat::SmtLib2,
                        command_path: Some(PathBuf::from("/bin/echo")),
                        working_dir: None,
                        extra_args: vec!["unsat".to_string()],
                        use_default_args: false,
                    },
                },
            ],
            None,
        );
        assert!(!report.agreement);
        assert!(report.all_successful);
        assert!(!report.all_successful_verdicts_agree);
        assert_eq!(report.agreed_verdict, None);
        assert_eq!(
            report.runs[0].verdict,
            Some(ExternalValidationTextVerdict::Sat)
        );
        assert_eq!(
            report.runs[1].verdict,
            Some(ExternalValidationTextVerdict::Unsat)
        );
    }

    #[test]
    fn artifact_kind_suffixes_match_probe_contract() {
        assert_eq!(
            ExternalValidationArtifactKind::BenchmarkDataDir.env_suffix(),
            Some("DATA_DIR")
        );
        assert_eq!(
            ExternalValidationArtifactKind::SchemaOrSpecPath.env_suffix(),
            Some("SPEC")
        );
        assert_eq!(ExternalValidationArtifactKind::None.env_suffix(), None);
    }
}
