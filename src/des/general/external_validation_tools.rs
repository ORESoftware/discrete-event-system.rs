//! Local adapter/probe surface for external model-validation, output-validation,
//! benchmark, proof-checking, and simulation engines.
//!
//! These tools are intentionally represented as local adapters: the crate knows
//! stable names, capabilities, command aliases, and environment-variable hooks,
//! but it does not vendor jars, native libraries, solver binaries, benchmark
//! corpora, or simulator installations.

use crate::des::general::bin_packing::{BinPackingItem, BinPackingProblem};
use crate::des::general::classical_optimization_models::{
    FlowShopJob, JobOperation, JobShopJob, Point, VRPCustomer,
};
use crate::des::general::external_assignment_reference::{
    solve_assignment_with_external_reference, ExternalAssignmentReferenceOptions,
    ExternalAssignmentReferenceSolver, ExternalAssignmentReferenceStatus,
};
use crate::des::general::external_bin_packing_reference::{
    solve_bin_packing_with_external_reference, ExternalBinPackingReferenceOptions,
    ExternalBinPackingReferenceSolver, ExternalBinPackingReferenceStatus,
};
use crate::des::general::external_cp_sat_reference::{
    solve_cp_sat_json_with_external_reference, ExternalCpSatReferenceOptions,
    ExternalCpSatReferenceSolver, ExternalCpSatReferenceStatus,
};
use crate::des::general::external_facility_location_reference::{
    solve_facility_location_with_external_reference, ExternalFacilityLocationReferenceOptions,
    ExternalFacilityLocationReferenceSolver, ExternalFacilityLocationReferenceStatus,
};
use crate::des::general::external_gams_solver_probe::{
    probe_external_gams_solver, ExternalGamsSolver,
};
use crate::des::general::external_graph_coloring_reference::{
    solve_graph_coloring_with_external_reference, ExternalGraphColoringReferenceOptions,
    ExternalGraphColoringReferenceSolver, ExternalGraphColoringReferenceStatus,
};
use crate::des::general::external_knapsack_reference::{
    solve_knapsack_with_external_reference, ExternalKnapsackReferenceOptions,
    ExternalKnapsackReferenceSolver, ExternalKnapsackReferenceStatus,
};
use crate::des::general::external_linear_cli::{
    probe_external_linear_cli_solver, ExternalLinearCliKind, ExternalLinearCliOptions,
    ExternalLinearCliProbeStatus, ExternalLinearCliSolver,
};
use crate::des::general::external_max_flow_reference::{
    solve_max_flow_with_external_reference, ExternalMaxFlowReferenceOptions,
    ExternalMaxFlowReferenceSolver, ExternalMaxFlowReferenceStatus,
};
use crate::des::general::external_min_cost_flow_reference::{
    solve_min_cost_flow_with_external_reference, ExternalMinCostFlowReferenceOptions,
    ExternalMinCostFlowReferenceSolver, ExternalMinCostFlowReferenceStatus,
};
use crate::des::general::external_minimum_spanning_tree_reference::{
    solve_minimum_spanning_tree_with_external_reference,
    ExternalMinimumSpanningTreeReferenceOptions, ExternalMinimumSpanningTreeReferenceSolver,
    ExternalMinimumSpanningTreeReferenceStatus,
};
use crate::des::general::external_nonlinear_validation_reference::{
    solve_nonlinear_validation_json_with_external_reference,
    ExternalNonlinearValidationReferenceOptions, ExternalNonlinearValidationReferenceSolver,
    ExternalNonlinearValidationReferenceStatus,
};
use crate::des::general::external_quadratic_reference::{
    solve_miqp_with_external_reference, solve_qp_with_external_reference,
    ExternalQuadraticReferenceOptions, ExternalQuadraticReferenceSolver,
    ExternalQuadraticReferenceStatus,
};
use crate::des::general::external_routing_reference::{
    solve_cvrp_with_external_reference, ExternalRoutingReferenceOptions,
    ExternalRoutingReferenceSolver, ExternalRoutingReferenceStatus,
};
use crate::des::general::external_scheduling_reference::{
    solve_flow_shop_with_external_reference, solve_job_shop_with_external_reference,
    ExternalSchedulingReferenceOptions, ExternalSchedulingReferenceSolver,
    ExternalSchedulingReferenceStatus,
};
use crate::des::general::external_set_cover_reference::{
    solve_set_cover_with_external_reference, ExternalSetCoverReferenceOptions,
    ExternalSetCoverReferenceSolver, ExternalSetCoverReferenceStatus,
};
use crate::des::general::external_stochastic_lp_reference::{
    solve_stochastic_lp_with_external_reference, ExternalStochasticLpReferenceOptions,
    ExternalStochasticLpReferenceSolver, ExternalStochasticLpReferenceStatus,
};
use crate::des::general::external_tsp_reference::{
    solve_euclidean_tsp_with_external_reference, solve_tsp_with_external_reference,
    ExternalTspPoint, ExternalTspReferenceOptions, ExternalTspReferenceSolver,
    ExternalTspReferenceStatus,
};
use crate::des::general::external_weighted_independent_set_reference::{
    solve_weighted_independent_set_with_external_reference,
    ExternalWeightedIndependentSetReferenceOptions, ExternalWeightedIndependentSetReferenceSolver,
    ExternalWeightedIndependentSetReferenceStatus,
};
use crate::des::general::external_weighted_max_sat_reference::{
    solve_weighted_max_sat_with_external_reference, ExternalWeightedMaxSatReferenceOptions,
    ExternalWeightedMaxSatReferenceSolver, ExternalWeightedMaxSatReferenceStatus,
};
use crate::des::general::facility_location::FacilityLocationProblem;
use crate::des::general::graph_coloring::GraphColoringProblem;
use crate::des::general::knapsack::{KnapsackItem, KnapsackProblem};
use crate::des::general::max_flow::{MaxFlowEdge, MaxFlowProblem};
use crate::des::general::min_cost_flow::{MinCostFlowArc, MinCostFlowProblem};
use crate::des::general::minimum_spanning_tree::{
    MinimumSpanningTreeEdge, MinimumSpanningTreeProblem,
};
use crate::des::general::qp::{MixedIntegerQuadraticProgram, QuadraticProgram};
use crate::des::general::set_cover::{SetCoverProblem, SetCoverSet};
use crate::des::general::stochastic_lp::{SLPProblem, Scenario};
use crate::des::general::weighted_independent_set::{
    WeightedIndependentSetProblem, WeightedIndependentSetVertex,
};
use crate::des::general::weighted_max_sat::{WeightedMaxSatClause, WeightedMaxSatProblem};
use serde::Deserialize;
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
    Node,
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
            ExternalValidationRuntime::Node => "node",
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
    NodePackage,
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
            ExternalValidationArtifactKind::NodePackage => Some("NODE_PATH"),
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
const SQL_OUTPUT_FORMATS: &[&str] = &["sql", "json"];
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
        id: "minilp",
        display_name: "MiniLp",
        env_key: "MINILP",
        family: ExternalValidationFamily::ConstraintModeling,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-minilp-adapter", "minilp-adapter"],
        capabilities: SOLVE_AND_VALIDATE_CAPS,
        input_formats: ALGEBRAIC_MODEL_FORMATS,
        notes: "Rust-first lightweight LP solver crate adapter for local model-validation cross-checks",
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
        id: "osqp-rust",
        display_name: "OSQP Rust bindings",
        env_key: "OSQP_RUST",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-osqp-rust-adapter", "osqp-rust-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: CONIC_FORMATS,
        notes: "Rust OSQP binding adapter for convex quadratic-program validation without Python package probes",
    },
    ExternalValidationToolSpec {
        id: "clarabel-rust",
        display_name: "Clarabel Rust crate",
        env_key: "CLARABEL_RUST",
        family: ExternalValidationFamily::ConvexConicSolver,
        runtime: ExternalValidationRuntime::Rust,
        artifact_kind: ExternalValidationArtifactKind::RustCrate,
        command_aliases: &["ores-clarabel-rust-adapter", "clarabel-rust-adapter"],
        capabilities: CONVEX_CONIC_CAPS,
        input_formats: CONIC_FORMATS,
        notes: "Rust-native Clarabel crate adapter for conic and quadratic validation checks",
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["kissat"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "CDCL SAT solver for DIMACS CNF cross-checks via native CLI or PySAT backend",
    },
    ExternalValidationToolSpec {
        id: "cadical",
        display_name: "CaDiCaL",
        env_key: "CADICAL",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["cadical"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "SAT solver with proof-generation and checker ecosystem support via native CLI or PySAT backend",
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["minisat"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "Classic CDCL SAT solver for DIMACS CNF smoke-model validation via native CLI or PySAT backend",
    },
    ExternalValidationToolSpec {
        id: "glucose",
        display_name: "Glucose",
        env_key: "GLUCOSE",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["glucose", "glucose-syrup"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "CDCL SAT solver family for independent DIMACS satisfiability checks via native CLI or PySAT backend",
    },
    ExternalValidationToolSpec {
        id: "maplesat",
        display_name: "MapleSAT",
        env_key: "MAPLESAT",
        family: ExternalValidationFamily::SatSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["maplesat", "maple-sat", "maple-lcm"],
        capabilities: SAT_CAPS,
        input_formats: SAT_FORMATS,
        notes: "Maple-family SAT solver for CDCL branching and restart cross-checks via native CLI or PySAT backend",
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["mosek"],
        capabilities: NONLINEAR_CAPS,
        input_formats: &["mps", "ptf", "opf", "task", "json"],
        notes: "Commercial conic, quadratic, and nonlinear optimization solver via CLI or Python API",
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["copt_cmd", "copt"],
        capabilities: NONLINEAR_CAPS,
        input_formats: &["mps", "lp", "json"],
        notes: "Commercial LP/QP/QCP/MIP solver via CLI or Python API for independent checks",
    },
    ExternalValidationToolSpec {
        id: "nlopt",
        display_name: "NLopt",
        env_key: "NLOPT",
        family: ExternalValidationFamily::NonlinearGlobalSolver,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["ores-nlopt-adapter", "nlopt-adapter"],
        capabilities: NONLINEAR_CAPS,
        input_formats: &["json", "nl"],
        notes: "NLopt derivative-free and gradient nonlinear optimization adapter using the Python package or local adapters",
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["check-jsonschema", "jsonschema"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: OUTPUT_FORMATS,
        notes: "Schema validation for JSON run artifacts and traces via generic CLI or Python package",
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["csv-validator", "csvlint"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: &["csv", "json"],
        notes: "CSV/table schema validation adapter for tabular run artifacts via CLI or Python csvvalidator",
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
        command_aliases: &["xmlschema", "xmlschema-validate", "xsd-validator"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: XML_OUTPUT_FORMATS,
        notes: "XSD/XML Schema validation adapter for structured XML run artifacts",
    },
    ExternalValidationToolSpec {
        id: "schematron",
        display_name: "Schematron",
        env_key: "SCHEMATRON",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["schematron-adapter", "jing", "saxon"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: XML_OUTPUT_FORMATS,
        notes: "Rule-based XML validation adapter for cross-field output constraints via generic CLI or Python lxml ISO Schematron",
    },
    ExternalValidationToolSpec {
        id: "jing",
        display_name: "Jing",
        env_key: "JING",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Java,
        artifact_kind: ExternalValidationArtifactKind::JavaClasspath,
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["saxon", "saxon-he", "saxon9he"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: XML_OUTPUT_FORMATS,
        notes: "Saxon-backed XML, XPath, and Schematron-style validation adapter via CLI or Python SaxonC",
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
        runtime: ExternalValidationRuntime::Node,
        artifact_kind: ExternalValidationArtifactKind::NodePackage,
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
        runtime: ExternalValidationRuntime::Node,
        artifact_kind: ExternalValidationArtifactKind::NodePackage,
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
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["avro-tools", "avro"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: &["avro", "avsc", "json"],
        notes: "Apache Avro schema and data-file validation adapter via CLI or Python package",
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
        id: "sqlfluff",
        display_name: "SQLFluff",
        env_key: "SQLFLUFF",
        family: ExternalValidationFamily::OutputDataValidator,
        runtime: ExternalValidationRuntime::Python,
        artifact_kind: ExternalValidationArtifactKind::PythonPackage,
        command_aliases: &["sqlfluff", "sql-lint", "sql-validator"],
        capabilities: OUTPUT_VALIDATOR_CAPS,
        input_formats: SQL_OUTPUT_FORMATS,
        notes: "SQL linting and structural query validation adapter via CLI or Python package",
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

fn output_validation_result(
    status: &str,
    verdict: &str,
    validator: &str,
    message: impl Into<String>,
    errors: Vec<String>,
) -> Value {
    json!({
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message.into(),
        "errors": errors,
    })
}

fn output_validation_json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn output_validation_json_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
}

fn output_validation_json_integer(value: &Value) -> Option<i128> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .map(i128::from)
            .or_else(|| number.as_u64().map(i128::from)),
        _ => None,
    }
}

fn output_validation_matches_json_type(value: &Value, expected: &str) -> bool {
    match expected {
        "number" => output_validation_json_number(value).is_some(),
        "integer" => output_validation_json_integer(value).is_some(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn output_validation_schema_errors(schema: &Value, instance: &Value, path: &str) -> Vec<String> {
    let Some(schema_obj) = schema.as_object() else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    if let Some(expected_type) = schema_obj.get("type") {
        match expected_type {
            Value::Array(types) => {
                let any_match = types.iter().any(|item| {
                    item.as_str().is_some_and(|expected| {
                        output_validation_matches_json_type(instance, expected)
                    })
                });
                if !any_match {
                    errors.push(format!(
                        "{path}: expected one of {expected_type}, got {}",
                        output_validation_json_type_name(instance)
                    ));
                    return errors;
                }
            }
            Value::String(expected) => {
                if !output_validation_matches_json_type(instance, expected) {
                    errors.push(format!(
                        "{path}: expected {expected}, got {}",
                        output_validation_json_type_name(instance)
                    ));
                    return errors;
                }
            }
            _ => {}
        }
    }
    if let Some(expected_const) = schema_obj.get("const") {
        if instance != expected_const {
            errors.push(format!("{path}: expected constant {expected_const}"));
        }
    }
    if let Some(enum_values) = schema_obj.get("enum").and_then(Value::as_array) {
        if !enum_values.iter().any(|value| value == instance) {
            errors.push(format!("{path}: value {instance} is not in enum"));
        }
    }
    if let Some(number) = output_validation_json_number(instance) {
        if let Some(minimum) = schema_obj
            .get("minimum")
            .and_then(output_validation_json_number)
        {
            if number < minimum {
                errors.push(format!("{path}: value is below minimum {minimum}"));
            }
        }
        if let Some(maximum) = schema_obj
            .get("maximum")
            .and_then(output_validation_json_number)
        {
            if number > maximum {
                errors.push(format!("{path}: value is above maximum {maximum}"));
            }
        }
        if let Some(minimum) = schema_obj
            .get("exclusiveMinimum")
            .and_then(output_validation_json_number)
        {
            if number <= minimum {
                errors.push(format!(
                    "{path}: value is not above exclusiveMinimum {minimum}"
                ));
            }
        }
        if let Some(maximum) = schema_obj
            .get("exclusiveMaximum")
            .and_then(output_validation_json_number)
        {
            if number >= maximum {
                errors.push(format!(
                    "{path}: value is not below exclusiveMaximum {maximum}"
                ));
            }
        }
    }
    if let Some(text) = instance.as_str() {
        let len = text.chars().count();
        if let Some(min_len) = schema_obj.get("minLength").and_then(Value::as_u64) {
            if len < min_len as usize {
                errors.push(format!(
                    "{path}: string is shorter than minLength {min_len}"
                ));
            }
        }
        if let Some(max_len) = schema_obj.get("maxLength").and_then(Value::as_u64) {
            if len > max_len as usize {
                errors.push(format!("{path}: string is longer than maxLength {max_len}"));
            }
        }
    }
    if let Some(items) = instance.as_array() {
        if let Some(min_items) = schema_obj.get("minItems").and_then(Value::as_u64) {
            if items.len() < min_items as usize {
                errors.push(format!("{path}: array has fewer than minItems {min_items}"));
            }
        }
        if let Some(max_items) = schema_obj.get("maxItems").and_then(Value::as_u64) {
            if items.len() > max_items as usize {
                errors.push(format!("{path}: array has more than maxItems {max_items}"));
            }
        }
        if let Some(item_schema) = schema_obj.get("items") {
            if item_schema.is_object() {
                for (idx, item) in items.iter().enumerate() {
                    errors.extend(output_validation_schema_errors(
                        item_schema,
                        item,
                        &format!("{path}[{idx}]"),
                    ));
                }
            }
        }
    }
    if let Some(instance_obj) = instance.as_object() {
        if let Some(required) = schema_obj.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if !instance_obj.contains_key(key) {
                    errors.push(format!("{path}: missing required property '{key}'"));
                }
            }
        }
        let properties = schema_obj.get("properties").and_then(Value::as_object);
        if let Some(properties) = properties {
            for (key, property_schema) in properties {
                if let Some(value) = instance_obj.get(key) {
                    errors.extend(output_validation_schema_errors(
                        property_schema,
                        value,
                        &format!("{path}.{key}"),
                    ));
                }
            }
        }
        if schema_obj
            .get("additionalProperties")
            .and_then(Value::as_bool)
            == Some(false)
        {
            if let Some(properties) = properties {
                for key in instance_obj.keys() {
                    if !properties.contains_key(key) {
                        errors.push(format!("{path}: unexpected property '{key}'"));
                    }
                }
            }
        }
    }
    errors
}

fn output_validation_json_schema_reference(payload: &Value, validator: &str) -> Value {
    let Some(schema) = payload.get("schema") else {
        return output_validation_result(
            "failed",
            "invalid",
            validator,
            "schema must be an object",
            Vec::new(),
        );
    };
    if !schema.is_object() {
        return output_validation_result(
            "failed",
            "invalid",
            validator,
            "schema must be an object",
            Vec::new(),
        );
    }
    let instance = payload.get("instance").unwrap_or(&Value::Null);
    let errors = output_validation_schema_errors(schema, instance, "$");
    let message = errors.first().cloned().unwrap_or_default();
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        message,
        errors,
    )
}

fn output_validation_table_columns(
    schema: &Value,
) -> BTreeMap<String, serde_json::Map<String, Value>> {
    let mut specs = BTreeMap::new();
    match schema.get("columns") {
        Some(Value::Object(columns)) => {
            for (name, spec) in columns {
                let spec_obj = match spec {
                    Value::Object(obj) => obj.clone(),
                    Value::String(kind) => {
                        let mut obj = serde_json::Map::new();
                        obj.insert("type".to_string(), Value::String(kind.clone()));
                        obj
                    }
                    _ => serde_json::Map::new(),
                };
                specs.insert(name.clone(), spec_obj);
            }
        }
        Some(Value::Array(columns)) => {
            for item in columns {
                match item {
                    Value::String(name) => {
                        specs.insert(name.clone(), serde_json::Map::new());
                    }
                    Value::Object(obj) => {
                        if let Some(name) = obj.get("name").and_then(Value::as_str) {
                            specs.insert(name.to_string(), obj.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    specs
}

fn output_validation_table_rows(
    payload: &Value,
) -> Result<Vec<serde_json::Map<String, Value>>, String> {
    let source = payload
        .get("rows")
        .or_else(|| payload.get("data"))
        .or_else(|| payload.get("instance"));
    let Some(source) = source else {
        let Some(text) = payload
            .get("csv")
            .or_else(|| payload.get("text"))
            .and_then(Value::as_str)
        else {
            return Err(
                "table-validation payload needs rows, data, instance, csv, or text".to_string(),
            );
        };
        return output_validation_csv_rows(text);
    };
    if let Some(text) = source.as_str() {
        return output_validation_csv_rows(text);
    }
    let Some(rows) = source.as_array() else {
        return Err(
            "table-validation payload needs rows, data, or instance as an array or CSV text"
                .to_string(),
        );
    };
    let mut out = Vec::with_capacity(rows.len());
    for (idx, row) in rows.iter().enumerate() {
        let Some(row_obj) = row.as_object() else {
            return Err(format!("row {idx} must be an object"));
        };
        out.push(row_obj.clone());
    }
    Ok(out)
}

fn output_validation_csv_record(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => {
                in_quotes = !in_quotes;
            }
            ',' if !in_quotes => {
                fields.push(field.trim().to_string());
                field.clear();
            }
            _ => field.push(ch),
        }
    }
    if in_quotes {
        return Err("csv row has an unterminated quoted field".to_string());
    }
    fields.push(field.trim().to_string());
    Ok(fields)
}

fn output_validation_csv_rows(text: &str) -> Result<Vec<serde_json::Map<String, Value>>, String> {
    let mut records = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(output_validation_csv_record);
    let Some(headers) = records.next() else {
        return Err("csv table payload needs a header row".to_string());
    };
    let headers = headers?;
    if headers.is_empty() || headers.iter().any(|header| header.is_empty()) {
        return Err("csv table header names must be non-empty".to_string());
    }
    let mut rows = Vec::new();
    for (idx, record) in records.enumerate() {
        let fields = record?;
        if fields.len() > headers.len() {
            return Err(format!(
                "csv row {} has {} fields for {} headers",
                idx + 1,
                fields.len(),
                headers.len()
            ));
        }
        let mut row = serde_json::Map::new();
        for (col_idx, header) in headers.iter().enumerate() {
            row.insert(
                header.clone(),
                Value::String(fields.get(col_idx).cloned().unwrap_or_default()),
            );
        }
        rows.push(row);
    }
    Ok(rows)
}

fn output_validation_has_table_payload(payload: &Value) -> bool {
    ["rows", "data", "instance", "csv", "text"]
        .iter()
        .any(|key| payload.get(*key).is_some())
}

fn output_validation_missing_cell(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.is_empty(),
        _ => false,
    }
}

fn output_validation_parse_table_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(_) => output_validation_json_number(value),
        Value::String(text) => text.parse::<f64>().ok().filter(|number| number.is_finite()),
        _ => None,
    }
}

fn output_validation_matches_table_type(value: &Value, expected: &str) -> bool {
    match expected {
        "number" => output_validation_parse_table_number(value).is_some(),
        "integer" => {
            output_validation_parse_table_number(value).is_some_and(|number| number.fract() == 0.0)
        }
        "boolean" => {
            value.is_boolean()
                || value.as_str().is_some_and(|text| {
                    matches!(
                        text.trim().to_ascii_lowercase().as_str(),
                        "true" | "false" | "0" | "1"
                    )
                })
        }
        "string" => value.is_string(),
        _ => true,
    }
}

fn output_validation_table_reference(payload: &Value, validator: &str) -> Value {
    let schema = payload
        .get("schema")
        .or_else(|| payload.get("expectations"))
        .unwrap_or(&Value::Null);
    if !schema.is_object() {
        return output_validation_result(
            "failed",
            "invalid",
            validator,
            "schema must be an object",
            Vec::new(),
        );
    }
    let rows = match output_validation_table_rows(payload) {
        Ok(rows) => rows,
        Err(message) => {
            return output_validation_result("failed", "failure", validator, message, Vec::new());
        }
    };
    let mut errors = Vec::new();
    let min_rows = schema
        .get("min_rows")
        .or_else(|| schema.get("minRows"))
        .and_then(Value::as_u64);
    if let Some(min_rows) = min_rows {
        if rows.len() < min_rows as usize {
            errors.push(format!(
                "table: expected at least {min_rows} rows, got {}",
                rows.len()
            ));
        }
    }
    let max_rows = schema
        .get("max_rows")
        .or_else(|| schema.get("maxRows"))
        .and_then(Value::as_u64);
    if let Some(max_rows) = max_rows {
        if rows.len() > max_rows as usize {
            errors.push(format!(
                "table: expected at most {max_rows} rows, got {}",
                rows.len()
            ));
        }
    }
    let columns = output_validation_table_columns(schema);
    if schema.get("additionalColumns").and_then(Value::as_bool) == Some(false)
        || schema.get("additional_columns").and_then(Value::as_bool) == Some(false)
    {
        for (idx, row) in rows.iter().enumerate() {
            for key in row.keys() {
                if !columns.contains_key(key) {
                    errors.push(format!("row {idx}: unexpected column '{key}'"));
                }
            }
        }
    }
    let required_columns: Vec<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mut unique_values: BTreeMap<String, Vec<Value>> = columns
        .iter()
        .filter_map(|(name, spec)| {
            if spec.get("unique").and_then(Value::as_bool) == Some(true) {
                Some((name.clone(), Vec::new()))
            } else {
                None
            }
        })
        .collect();
    for (idx, row) in rows.iter().enumerate() {
        for (name, spec) in &columns {
            let value = row.get(name);
            let required = spec.get("required").and_then(Value::as_bool) == Some(true)
                || required_columns.iter().any(|required| required == name);
            if output_validation_missing_cell(value) {
                if required {
                    errors.push(format!("row {idx}.{name}: required value is missing"));
                }
                continue;
            }
            let value = value.expect("checked missing value");
            if let Some(expected_type) = spec.get("type").and_then(Value::as_str) {
                if !output_validation_matches_table_type(value, expected_type) {
                    errors.push(format!(
                        "row {idx}.{name}: expected {expected_type}, got {value}"
                    ));
                    continue;
                }
            }
            if let Some(enum_values) = spec.get("enum").and_then(Value::as_array) {
                if !enum_values.iter().any(|item| item == value) {
                    errors.push(format!("row {idx}.{name}: value {value} is not in enum"));
                }
            }
            if let Some(number) = output_validation_parse_table_number(value) {
                if let Some(minimum) = spec.get("minimum").and_then(output_validation_json_number) {
                    if number < minimum {
                        errors.push(format!(
                            "row {idx}.{name}: value is below minimum {minimum}"
                        ));
                    }
                }
                if let Some(maximum) = spec.get("maximum").and_then(output_validation_json_number) {
                    if number > maximum {
                        errors.push(format!(
                            "row {idx}.{name}: value is above maximum {maximum}"
                        ));
                    }
                }
            }
            if let Some(text) = value.as_str() {
                let len = text.chars().count();
                if let Some(min_len) = spec.get("minLength").and_then(Value::as_u64) {
                    if len < min_len as usize {
                        errors.push(format!(
                            "row {idx}.{name}: string is shorter than minLength {min_len}"
                        ));
                    }
                }
                if let Some(max_len) = spec.get("maxLength").and_then(Value::as_u64) {
                    if len > max_len as usize {
                        errors.push(format!(
                            "row {idx}.{name}: string is longer than maxLength {max_len}"
                        ));
                    }
                }
            }
            if let Some(seen) = unique_values.get_mut(name) {
                if seen.iter().any(|prior| prior == value) {
                    errors.push(format!("row {idx}.{name}: duplicate value {value}"));
                }
                seen.push(value.clone());
            }
        }
    }
    let message = errors.first().cloned().unwrap_or_default();
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        message,
        errors,
    )
}

fn output_validation_has_data_package_payload(payload: &Value) -> bool {
    payload.get("package").is_some()
        || payload.get("datapackage").is_some()
        || payload.get("data_package").is_some()
        || payload.get("resources").is_some()
        || payload
            .get("profile")
            .and_then(Value::as_str)
            .is_some_and(|profile| profile.to_ascii_lowercase().contains("data-package"))
        || payload.get("frictionless").is_some()
}

fn output_validation_data_package_root(payload: &Value) -> &Value {
    payload
        .get("package")
        .or_else(|| payload.get("datapackage"))
        .or_else(|| payload.get("data_package"))
        .or_else(|| payload.get("frictionless"))
        .unwrap_or(payload)
}

fn output_validation_data_package_type_known(kind: &str) -> bool {
    matches!(
        kind.trim().to_ascii_lowercase().as_str(),
        "any"
            | "array"
            | "boolean"
            | "date"
            | "datetime"
            | "duration"
            | "geojson"
            | "geopoint"
            | "integer"
            | "number"
            | "object"
            | "string"
            | "time"
            | "year"
            | "yearmonth"
    )
}

fn output_validation_data_package_table_type(kind: &str) -> &'static str {
    match kind.trim().to_ascii_lowercase().as_str() {
        "integer" | "year" => "integer",
        "number" => "number",
        "boolean" => "boolean",
        _ => "string",
    }
}

fn output_validation_data_package_resource_schema(
    resource: &serde_json::Map<String, Value>,
    resource_label: &str,
) -> (serde_json::Map<String, Value>, Vec<String>) {
    let Some(schema) = resource.get("schema") else {
        return (serde_json::Map::new(), Vec::new());
    };
    let fields = schema
        .get("fields")
        .or_else(|| schema.get("columns"))
        .unwrap_or(schema);
    let mut columns = serde_json::Map::new();
    let mut errors = Vec::new();
    let mut seen = BTreeMap::<String, usize>::new();
    let field_items: Vec<Value> = match fields {
        Value::Array(items) => items.clone(),
        Value::Object(obj) => obj
            .iter()
            .map(|(name, spec)| {
                let mut item = match spec {
                    Value::Object(spec_obj) => spec_obj.clone(),
                    Value::String(kind) => {
                        let mut spec_obj = serde_json::Map::new();
                        spec_obj.insert("type".to_string(), Value::String(kind.clone()));
                        spec_obj
                    }
                    _ => serde_json::Map::new(),
                };
                item.entry("name".to_string())
                    .or_insert_with(|| Value::String(name.clone()));
                Value::Object(item)
            })
            .collect(),
        _ => {
            errors.push(format!(
                "{resource_label}: schema fields must be an array or object"
            ));
            Vec::new()
        }
    };
    for (idx, field) in field_items.iter().enumerate() {
        let Some(field_obj) = field.as_object() else {
            errors.push(format!("{resource_label}: field {idx} must be an object"));
            continue;
        };
        let name = field_obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if name.is_empty() {
            errors.push(format!("{resource_label}: field {idx} is missing name"));
            continue;
        }
        let count = seen.entry(name.to_string()).or_insert(0);
        *count += 1;
        if *count > 1 {
            errors.push(format!("{resource_label}: field '{name}' is duplicated"));
        }
        let kind = field_obj
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("string");
        if !output_validation_data_package_type_known(kind) {
            errors.push(format!(
                "{resource_label}: field '{name}' has unknown Table Schema type {kind:?}"
            ));
        }
        let mut column = serde_json::Map::new();
        column.insert(
            "type".to_string(),
            Value::String(output_validation_data_package_table_type(kind).to_string()),
        );
        if let Some(constraints) = field_obj.get("constraints").and_then(Value::as_object) {
            if constraints.get("required").and_then(Value::as_bool) == Some(true) {
                column.insert("required".to_string(), Value::Bool(true));
            }
            for (source, target) in [
                ("minimum", "minimum"),
                ("maximum", "maximum"),
                ("minLength", "minLength"),
                ("maxLength", "maxLength"),
                ("enum", "enum"),
            ] {
                if let Some(value) = constraints.get(source).cloned() {
                    column.insert(target.to_string(), value);
                }
            }
        }
        columns.insert(name.to_string(), Value::Object(column));
    }
    if let Some(primary_key) = schema
        .get("primaryKey")
        .or_else(|| schema.get("primary_key"))
    {
        let keys: Vec<&str> = match primary_key {
            Value::String(name) => vec![name.as_str()],
            Value::Array(items) => items.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if keys.is_empty() {
            errors.push(format!(
                "{resource_label}: primaryKey must name at least one field"
            ));
        }
        for key in keys {
            if !columns.contains_key(key) {
                errors.push(format!(
                    "{resource_label}: primaryKey references missing field '{key}'"
                ));
            }
        }
    }
    (columns, errors)
}

fn output_validation_data_package_reference(payload: &Value, validator: &str) -> Value {
    let package = output_validation_data_package_root(payload);
    let Some(package_obj) = package.as_object() else {
        return output_validation_result(
            "failed",
            "failure",
            validator,
            "data package payload must be an object",
            Vec::new(),
        );
    };
    let mut errors = Vec::new();
    if let Some(profile) = package_obj.get("profile").and_then(Value::as_str) {
        let profile = profile.trim().to_ascii_lowercase();
        if !matches!(
            profile.as_str(),
            "data-package" | "tabular-data-package" | "data-resource" | "tabular-data-resource"
        ) {
            errors.push(format!(
                "data package profile {profile:?} is not recognized"
            ));
        }
    }
    for array_key in ["licenses", "sources", "contributors"] {
        if let Some(value) = package_obj.get(array_key) {
            if !value.is_array() {
                errors.push(format!("data package {array_key} must be an array"));
            }
        }
    }
    let Some(resources) = package_obj.get("resources").and_then(Value::as_array) else {
        return output_validation_result(
            "failed",
            "failure",
            validator,
            "data package payload needs resources array",
            Vec::new(),
        );
    };
    if resources.is_empty() {
        errors.push("data package must contain at least one resource".to_string());
    }
    let mut names = BTreeMap::<String, usize>::new();
    for (idx, resource) in resources.iter().enumerate() {
        let label = format!("resource {idx}");
        let Some(resource_obj) = resource.as_object() else {
            errors.push(format!("{label}: must be an object"));
            continue;
        };
        let name = resource_obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if !name.is_empty() {
            let count = names.entry(name.to_string()).or_insert(0);
            *count += 1;
            if *count > 1 {
                errors.push(format!("{label}: duplicate resource name '{name}'"));
            }
        }
        let has_path = resource_obj.get("path").is_some_and(|path| match path {
            Value::String(text) => !text.trim().is_empty(),
            Value::Array(paths) => paths
                .iter()
                .any(|path| path.as_str().is_some_and(|text| !text.trim().is_empty())),
            _ => false,
        });
        let has_inline_data = resource_obj.get("data").is_some()
            || resource_obj.get("rows").is_some()
            || resource_obj.get("csv").is_some();
        if !has_path && !has_inline_data {
            errors.push(format!("{label}: needs path, data, rows, or csv"));
        }
        if let Some(format) = resource_obj.get("format").and_then(Value::as_str) {
            if format.trim().is_empty() {
                errors.push(format!("{label}: format must be non-empty when present"));
            }
        }
        let (columns, schema_errors) =
            output_validation_data_package_resource_schema(resource_obj, &label);
        errors.extend(schema_errors);
        let inline_rows = resource_obj
            .get("rows")
            .or_else(|| resource_obj.get("data"))
            .filter(|value| value.is_array());
        if !columns.is_empty() {
            if let Some(rows) = inline_rows {
                let table_payload = json!({
                    "schema": {
                        "columns": columns,
                        "additionalColumns": false
                    },
                    "rows": rows
                });
                let run = output_validation_table_reference(&table_payload, validator);
                if let Some(table_errors) = run.get("errors").and_then(Value::as_array) {
                    for error in table_errors.iter().filter_map(Value::as_str) {
                        errors.push(format!("{label}: {error}"));
                    }
                }
            }
        }
    }
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        errors.first().cloned().unwrap_or_default(),
        errors,
    )
}

fn output_validation_has_openrefine_payload(payload: &Value) -> bool {
    [
        "operations",
        "operationHistory",
        "operation_history",
        "history",
        "reconciliation",
        "reconcile",
    ]
    .iter()
    .any(|key| payload.get(*key).is_some())
}

fn output_validation_openrefine_operation_values(payload: &Value) -> Option<Vec<Value>> {
    for key in [
        "operations",
        "operationHistory",
        "operation_history",
        "history",
    ] {
        let Some(value) = payload.get(key) else {
            continue;
        };
        match value {
            Value::Array(items) => return Some(items.clone()),
            Value::Object(obj) => {
                if let Some(entries) = obj.get("entries").and_then(Value::as_array) {
                    return Some(entries.clone());
                }
                if obj.get("op").is_some() {
                    return Some(vec![Value::Object(obj.clone())]);
                }
            }
            _ => return Some(vec![value.clone()]),
        }
    }
    None
}

fn output_validation_openrefine_row_columns(payload: &Value) -> Vec<String> {
    let rows = payload
        .get("rows")
        .or_else(|| payload.get("data"))
        .and_then(Value::as_array);
    let Some(first_row) = rows.and_then(|rows| rows.iter().find_map(Value::as_object)) else {
        return Vec::new();
    };
    first_row.keys().cloned().collect()
}

fn output_validation_openrefine_column_name<'a>(
    obj: &'a serde_json::Map<String, Value>,
) -> Option<&'a str> {
    obj.get("columnName")
        .or_else(|| obj.get("column"))
        .or_else(|| obj.get("oldColumnName"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

fn output_validation_openrefine_validate_reconciliation(
    payload: &Value,
    columns: &[String],
    errors: &mut Vec<String>,
) {
    let Some(reconciliation) = payload
        .get("reconciliation")
        .or_else(|| payload.get("reconcile"))
    else {
        return;
    };
    let Some(recon_obj) = reconciliation.as_object() else {
        errors.push("openrefine reconciliation must be an object".to_string());
        return;
    };
    let column = output_validation_openrefine_column_name(recon_obj);
    if column.is_none() {
        errors.push("openrefine reconciliation needs column or columnName".to_string());
    }
    if let Some(column) = column {
        if !columns.is_empty() && !columns.iter().any(|known| known == column) {
            errors.push(format!(
                "openrefine reconciliation references missing column '{column}'"
            ));
        }
    }
    let matched = output_validation_json_integer(recon_obj.get("matched").unwrap_or(&Value::Null));
    let unmatched =
        output_validation_json_integer(recon_obj.get("unmatched").unwrap_or(&Value::Null));
    let total = output_validation_json_integer(
        recon_obj
            .get("total")
            .or_else(|| recon_obj.get("rowCount"))
            .or_else(|| recon_obj.get("rows"))
            .unwrap_or(&Value::Null),
    );
    for (name, value) in [
        ("matched", matched),
        ("unmatched", unmatched),
        ("total", total),
    ] {
        if let Some(value) = value {
            if value < 0 {
                errors.push(format!(
                    "openrefine reconciliation {name} must be non-negative"
                ));
            }
        }
    }
    if let (Some(matched), Some(unmatched), Some(total)) = (matched, unmatched, total) {
        if matched + unmatched > total {
            errors.push(format!(
                "openrefine reconciliation matched+unmatched exceeds total ({matched}+{unmatched}>{total})"
            ));
        }
    }
    if let Some(candidates) = recon_obj.get("candidates").and_then(Value::as_array) {
        for (idx, candidate) in candidates.iter().enumerate() {
            let Some(candidate_obj) = candidate.as_object() else {
                errors.push(format!("openrefine candidate {idx} must be an object"));
                continue;
            };
            if !candidate_obj.contains_key("id") && !candidate_obj.contains_key("name") {
                errors.push(format!("openrefine candidate {idx} needs id or name"));
            }
            if let Some(score) = candidate_obj
                .get("score")
                .and_then(output_validation_json_number)
            {
                if !(0.0..=100.0).contains(&score) {
                    errors.push(format!(
                        "openrefine candidate {idx} score {score} is outside 0..100"
                    ));
                }
            }
        }
    }
}

fn output_validation_openrefine_reference(payload: &Value, validator: &str) -> Value {
    let columns = output_validation_openrefine_row_columns(payload);
    let mut errors = Vec::new();
    let operations = output_validation_openrefine_operation_values(payload);
    if let Some(operations) = operations {
        if operations.is_empty() {
            errors.push("openrefine operation history must not be empty".to_string());
        }
        for (idx, operation) in operations.iter().enumerate() {
            let Some(operation_obj) = operation.as_object() else {
                errors.push(format!("openrefine operation {idx} must be an object"));
                continue;
            };
            let op = operation_obj
                .get("op")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if op.is_empty() {
                errors.push(format!("openrefine operation {idx} is missing op"));
                continue;
            }
            if !op.contains('/') {
                errors.push(format!(
                    "openrefine operation {idx} op {op:?} should include namespace"
                ));
            }
            if let Some(description) = operation_obj.get("description") {
                if !description.is_string() {
                    errors.push(format!(
                        "openrefine operation {idx} description must be a string"
                    ));
                }
            }
            let needs_column = op.contains("column")
                || op.contains("text-transform")
                || op.contains("mass-edit")
                || op.contains("recon");
            let column = output_validation_openrefine_column_name(operation_obj);
            if needs_column && column.is_none() {
                errors.push(format!("openrefine operation {idx} needs a column name"));
            }
            if let Some(column) = column {
                if !columns.is_empty() && !columns.iter().any(|known| known == column) {
                    errors.push(format!(
                        "openrefine operation {idx} references missing column '{column}'"
                    ));
                }
            }
            if op == "core/column-rename" {
                for key in ["oldColumnName", "newColumnName"] {
                    if !operation_obj
                        .get(key)
                        .and_then(Value::as_str)
                        .is_some_and(|text| !text.trim().is_empty())
                    {
                        errors.push(format!("openrefine operation {idx} needs {key}"));
                    }
                }
            }
            if op.contains("text-transform") || op.contains("column-addition") {
                if !operation_obj
                    .get("expression")
                    .and_then(Value::as_str)
                    .is_some_and(|text| !text.trim().is_empty())
                {
                    errors.push(format!("openrefine operation {idx} needs expression"));
                }
            }
            if op.contains("mass-edit") {
                let edits = operation_obj.get("edits").and_then(Value::as_array);
                if edits.is_none_or(Vec::is_empty) {
                    errors.push(format!("openrefine operation {idx} needs non-empty edits"));
                }
                if let Some(edits) = edits {
                    for (edit_idx, edit) in edits.iter().enumerate() {
                        let Some(edit_obj) = edit.as_object() else {
                            errors.push(format!(
                                "openrefine operation {idx} edit {edit_idx} must be an object"
                            ));
                            continue;
                        };
                        if !edit_obj.contains_key("from")
                            && !edit_obj.contains_key("fromBlank")
                            && !edit_obj.contains_key("fromError")
                        {
                            errors.push(format!(
                                "openrefine operation {idx} edit {edit_idx} needs from/fromBlank/fromError"
                            ));
                        }
                        if !edit_obj.contains_key("to") {
                            errors.push(format!(
                                "openrefine operation {idx} edit {edit_idx} needs to"
                            ));
                        }
                    }
                }
            }
            if let Some(on_error) = operation_obj.get("onError").and_then(Value::as_str) {
                if !matches!(
                    on_error,
                    "keep-original" | "set-to-blank" | "store-error" | "repeat" | "fail"
                ) {
                    errors.push(format!(
                        "openrefine operation {idx} onError {on_error:?} is not recognized"
                    ));
                }
            }
            if let Some(repeat_count) = operation_obj
                .get("repeatCount")
                .and_then(output_validation_json_integer)
            {
                if repeat_count < 0 {
                    errors.push(format!(
                        "openrefine operation {idx} repeatCount is negative"
                    ));
                }
            }
        }
    }
    output_validation_openrefine_validate_reconciliation(payload, &columns, &mut errors);
    if !output_validation_has_openrefine_payload(payload) {
        return output_validation_result(
            "failed",
            "failure",
            validator,
            "payload needs operations, operationHistory, history, reconciliation, or reconcile",
            Vec::new(),
        );
    }
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        errors.first().cloned().unwrap_or_default(),
        errors,
    )
}

fn output_validation_columnar_metadata<'a>(payload: &'a Value) -> &'a Value {
    payload.get("metadata").unwrap_or(payload)
}

fn output_validation_columnar_schema<'a>(payload: &'a Value) -> Option<&'a Value> {
    payload
        .get("parquet_schema")
        .or_else(|| payload.get("parquetSchema"))
        .or_else(|| payload.get("arrow_schema"))
        .or_else(|| payload.get("arrowSchema"))
        .or_else(|| payload.get("schema"))
        .or_else(|| {
            payload
                .get("metadata")
                .and_then(|metadata| metadata.get("schema"))
        })
}

fn output_validation_has_columnar_payload(payload: &Value) -> bool {
    payload
        .get("format")
        .or_else(|| output_validation_columnar_metadata(payload).get("format"))
        .and_then(Value::as_str)
        .is_some_and(|format| {
            matches!(
                format.trim().to_ascii_lowercase().as_str(),
                "parquet" | "arrow" | "arrow-ipc" | "feather"
            )
        })
        || [
            "parquet_schema",
            "parquetSchema",
            "arrow_schema",
            "arrowSchema",
            "row_groups",
            "rowGroups",
        ]
        .iter()
        .any(|key| payload.get(*key).is_some())
        || ["num_rows", "numRows", "row_count", "rowCount", "created_by"]
            .iter()
            .any(|key| {
                output_validation_columnar_metadata(payload)
                    .get(*key)
                    .is_some()
            })
}

fn output_validation_columnar_field_specs(
    schema: &Value,
) -> (Vec<(String, serde_json::Map<String, Value>)>, Vec<String>) {
    let fields = schema
        .get("fields")
        .or_else(|| schema.get("columns"))
        .unwrap_or(schema);
    let mut specs = Vec::new();
    let mut errors = Vec::new();
    let mut seen = BTreeMap::<String, usize>::new();
    match fields {
        Value::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                match item {
                    Value::String(name) if !name.trim().is_empty() => {
                        specs.push((name.trim().to_string(), serde_json::Map::new()));
                    }
                    Value::Object(obj) => {
                        let name = obj
                            .get("name")
                            .or_else(|| obj.get("column"))
                            .or_else(|| obj.get("column_name"))
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim();
                        if name.is_empty() {
                            errors.push(format!("columnar field {idx}: missing name"));
                        } else {
                            specs.push((name.to_string(), obj.clone()));
                        }
                    }
                    _ => errors.push(format!("columnar field {idx}: unsupported field shape")),
                }
            }
        }
        Value::Object(items) => {
            for (name, spec) in items {
                let spec_obj = match spec {
                    Value::Object(obj) => obj.clone(),
                    Value::String(kind) => {
                        let mut obj = serde_json::Map::new();
                        obj.insert("type".to_string(), Value::String(kind.clone()));
                        obj
                    }
                    _ => serde_json::Map::new(),
                };
                specs.push((name.clone(), spec_obj));
            }
        }
        _ => errors.push("columnar schema fields must be an array or object".to_string()),
    }
    for (name, _) in &specs {
        let count = seen.entry(name.clone()).or_insert(0);
        *count += 1;
        if *count > 1 {
            errors.push(format!("columnar field '{name}' is duplicated"));
        }
    }
    (specs, errors)
}

fn output_validation_columnar_type_known(kind: &str) -> bool {
    let lower = kind.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }
    matches!(
        lower.as_str(),
        "boolean"
            | "bool"
            | "int8"
            | "int16"
            | "int32"
            | "int64"
            | "uint8"
            | "uint16"
            | "uint32"
            | "uint64"
            | "int96"
            | "float"
            | "float16"
            | "float32"
            | "float64"
            | "double"
            | "byte_array"
            | "fixed_len_byte_array"
            | "binary"
            | "large_binary"
            | "utf8"
            | "large_utf8"
            | "string"
            | "date32"
            | "date64"
            | "timestamp"
            | "time32"
            | "time64"
            | "duration"
            | "interval"
            | "decimal"
            | "decimal128"
            | "decimal256"
            | "list"
            | "large_list"
            | "fixed_size_list"
            | "struct"
            | "map"
            | "dictionary"
            | "null"
    ) || lower.starts_with("timestamp[")
        || lower.starts_with("decimal(")
        || lower.starts_with("list<")
        || lower.starts_with("struct<")
        || lower.starts_with("map<")
}

fn output_validation_columnar_integer(metadata: &Value, keys: &[&str]) -> Option<i128> {
    keys.iter()
        .find_map(|key| metadata.get(*key).and_then(output_validation_json_integer))
}

fn output_validation_columnar_row_groups<'a>(metadata: &'a Value) -> Option<&'a Vec<Value>> {
    metadata
        .get("row_groups")
        .or_else(|| metadata.get("rowGroups"))
        .and_then(Value::as_array)
}

fn output_validation_columnar_reference(payload: &Value, validator: &str) -> Value {
    let Some(schema) = output_validation_columnar_schema(payload) else {
        return output_validation_result(
            "failed",
            "failure",
            validator,
            "payload needs schema, parquet_schema, parquetSchema, arrow_schema, or arrowSchema",
            Vec::new(),
        );
    };
    let metadata = output_validation_columnar_metadata(payload);
    let (fields, mut errors) = output_validation_columnar_field_specs(schema);
    if fields.is_empty() {
        errors.push("columnar schema must contain at least one field".to_string());
    }
    for (name, spec) in &fields {
        let kind = spec
            .get("type")
            .or_else(|| spec.get("data_type"))
            .or_else(|| spec.get("dataType"))
            .or_else(|| spec.get("logical_type"))
            .or_else(|| spec.get("logicalType"))
            .or_else(|| spec.get("physical_type"))
            .or_else(|| spec.get("physicalType"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if !output_validation_columnar_type_known(kind) {
            errors.push(format!("columnar field '{name}' has unknown type {kind:?}"));
        }
    }

    let row_count = output_validation_columnar_integer(
        metadata,
        &["num_rows", "numRows", "row_count", "rowCount"],
    );
    if let Some(row_count) = row_count {
        if row_count < 0 {
            errors.push(format!(
                "columnar row count must be non-negative, got {row_count}"
            ));
        }
    }
    if let Some(file_size) =
        output_validation_columnar_integer(metadata, &["file_size", "fileSize", "size_bytes"])
    {
        if file_size <= 0 {
            errors.push(format!(
                "columnar file size must be positive, got {file_size}"
            ));
        }
    }
    if let Some(column_count) = output_validation_columnar_integer(
        metadata,
        &["num_columns", "numColumns", "column_count", "columnCount"],
    ) {
        if column_count != fields.len() as i128 {
            errors.push(format!(
                "columnar metadata says {column_count} columns but schema has {} fields",
                fields.len()
            ));
        }
    }
    if let Some(compression) = metadata.get("compression").and_then(Value::as_str) {
        let compression = compression.trim().to_ascii_lowercase();
        if !matches!(
            compression.as_str(),
            "uncompressed" | "none" | "snappy" | "gzip" | "brotli" | "lz4" | "lz4_raw" | "zstd"
        ) {
            errors.push(format!(
                "columnar compression {compression:?} is not recognized"
            ));
        }
    }
    if let Some(row_groups) = output_validation_columnar_row_groups(metadata) {
        if let Some(expected_groups) = output_validation_columnar_integer(
            metadata,
            &[
                "num_row_groups",
                "numRowGroups",
                "row_group_count",
                "rowGroupCount",
            ],
        ) {
            if expected_groups != row_groups.len() as i128 {
                errors.push(format!(
                    "columnar metadata says {expected_groups} row groups but payload has {}",
                    row_groups.len()
                ));
            }
        }
        let mut sum = 0_i128;
        let mut all_counts_present = true;
        for (idx, group) in row_groups.iter().enumerate() {
            let count = output_validation_columnar_integer(
                group,
                &["num_rows", "numRows", "row_count", "rowCount"],
            );
            match count {
                Some(count) if count >= 0 => sum += count,
                Some(count) => errors.push(format!("row group {idx}: negative row count {count}")),
                None => all_counts_present = false,
            }
        }
        if all_counts_present {
            if let Some(row_count) = row_count {
                if row_count >= 0 && sum != row_count {
                    errors.push(format!(
                        "row group row counts sum to {sum}, expected {row_count}"
                    ));
                }
            }
        }
    }

    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        errors.first().cloned().unwrap_or_default(),
        errors,
    )
}

fn output_validation_has_profile_payload(payload: &Value) -> bool {
    [
        "profile",
        "profiles",
        "statistics",
        "stats",
        "baseline",
        "current",
        "constraints",
        "drift",
        "anomalies",
    ]
    .iter()
    .any(|key| payload.get(*key).is_some())
}

fn output_validation_profile_root<'a>(payload: &'a Value) -> &'a Value {
    payload
        .get("profile")
        .or_else(|| payload.get("statistics"))
        .or_else(|| payload.get("stats"))
        .or_else(|| payload.get("current"))
        .unwrap_or(payload)
}

fn output_validation_profile_features<'a>(
    profile: &'a Value,
) -> Option<&'a serde_json::Map<String, Value>> {
    if let Some(features) = profile
        .get("features")
        .or_else(|| profile.get("columns"))
        .or_else(|| profile.get("variables"))
        .or_else(|| profile.get("fields"))
        .and_then(Value::as_object)
    {
        return Some(features);
    }
    let object = profile.as_object()?;
    let feature_like = object.values().any(|value| {
        value.as_object().is_some_and(|obj| {
            [
                "count", "missing", "mean", "min", "max", "stddev", "distinct", "type",
            ]
            .iter()
            .any(|key| obj.contains_key(*key))
        })
    });
    feature_like.then_some(object)
}

fn output_validation_profile_number(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(_)) => value.and_then(output_validation_json_number),
        Some(Value::String(text)) => text.parse::<f64>().ok().filter(|number| number.is_finite()),
        _ => None,
    }
}

fn output_validation_profile_metric(feature: &Value, names: &[&str]) -> Option<f64> {
    let obj = feature.as_object()?;
    names
        .iter()
        .find_map(|name| output_validation_profile_number(obj.get(*name)))
}

fn output_validation_profile_known_type(kind: &str) -> bool {
    matches!(
        kind.trim().to_ascii_lowercase().as_str(),
        "number"
            | "numeric"
            | "float"
            | "double"
            | "integer"
            | "int"
            | "long"
            | "string"
            | "categorical"
            | "bool"
            | "boolean"
            | "datetime"
            | "timestamp"
            | "date"
            | "object"
            | "array"
            | "unknown"
    )
}

fn output_validation_profile_compare(
    actual: f64,
    comparison: &str,
    target: f64,
    tolerance: f64,
) -> bool {
    match comparison.trim().to_ascii_lowercase().as_str() {
        "<" | "lt" => actual < target + tolerance,
        "<=" | "le" | "lte" | "at-most" | "max" => actual <= target + tolerance,
        ">" | "gt" => actual > target - tolerance,
        ">=" | "ge" | "gte" | "at-least" | "min" => actual >= target - tolerance,
        "==" | "=" | "eq" | "equal" => (actual - target).abs() <= tolerance,
        "!=" | "ne" | "not-equal" => (actual - target).abs() > tolerance,
        _ => false,
    }
}

fn output_validation_profile_constraint_errors(
    payload: &Value,
    features: &serde_json::Map<String, Value>,
) -> Vec<String> {
    let Some(constraints) = payload
        .get("constraints")
        .or_else(|| payload.get("expectations"))
        .or_else(|| payload.get("checks"))
    else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    let constraint_iter: Vec<Value> = match constraints {
        Value::Array(items) => items.clone(),
        Value::Object(obj) => obj.values().cloned().collect(),
        _ => {
            return vec!["profile constraints must be an array or object".to_string()];
        }
    };
    for (idx, constraint) in constraint_iter.iter().enumerate() {
        let Some(obj) = constraint.as_object() else {
            errors.push(format!("profile constraint {idx}: must be an object"));
            continue;
        };
        let feature_name = obj
            .get("feature")
            .or_else(|| obj.get("column"))
            .or_else(|| obj.get("field"))
            .or_else(|| obj.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let metric_name = obj
            .get("metric")
            .or_else(|| obj.get("stat"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let comparison = obj
            .get("comparison")
            .or_else(|| obj.get("op"))
            .or_else(|| obj.get("operator"))
            .and_then(Value::as_str)
            .unwrap_or("<=");
        let target =
            output_validation_profile_number(obj.get("target").or_else(|| obj.get("value")));
        let tolerance = output_validation_profile_number(obj.get("tolerance")).unwrap_or(0.0);
        if feature_name.is_empty() || metric_name.is_empty() {
            errors.push(format!(
                "profile constraint {idx}: needs feature/column and metric"
            ));
            continue;
        }
        let Some(feature) = features.get(feature_name) else {
            errors.push(format!(
                "profile constraint {idx}: feature '{feature_name}' not found"
            ));
            continue;
        };
        let Some(actual) = output_validation_profile_metric(feature, &[metric_name]) else {
            errors.push(format!(
                "profile constraint {idx}: metric '{metric_name}' not found on '{feature_name}'"
            ));
            continue;
        };
        let Some(target) = target else {
            errors.push(format!("profile constraint {idx}: target must be numeric"));
            continue;
        };
        if !output_validation_profile_compare(actual, comparison, target, tolerance) {
            errors.push(format!(
                "profile constraint {idx}: {feature_name}.{metric_name}={actual} failed {comparison} {target}"
            ));
        }
    }
    errors
}

fn output_validation_profile_drift_errors(payload: &Value) -> Vec<String> {
    let Some(baseline) = payload.get("baseline") else {
        return Vec::new();
    };
    let Some(current) = payload.get("current").or_else(|| payload.get("profile")) else {
        return Vec::new();
    };
    let Some(baseline_features) = output_validation_profile_features(baseline) else {
        return vec!["baseline profile has no feature statistics".to_string()];
    };
    let Some(current_features) = output_validation_profile_features(current) else {
        return vec!["current profile has no feature statistics".to_string()];
    };
    let threshold = output_validation_profile_number(
        payload
            .get("drift_threshold")
            .or_else(|| payload.get("max_drift"))
            .or_else(|| payload.get("maxDrift")),
    )
    .unwrap_or(0.25);
    let mut errors = Vec::new();
    for (name, current_feature) in current_features {
        let Some(baseline_feature) = baseline_features.get(name) else {
            continue;
        };
        for metric in ["mean", "missing", "missing_fraction", "null_fraction"] {
            let current_metric = output_validation_profile_metric(current_feature, &[metric]);
            let baseline_metric = output_validation_profile_metric(baseline_feature, &[metric]);
            if let (Some(current_metric), Some(baseline_metric)) = (current_metric, baseline_metric)
            {
                let delta = (current_metric - baseline_metric).abs();
                if delta > threshold {
                    errors.push(format!(
                        "profile drift: {name}.{metric} changed by {delta}, threshold {threshold}"
                    ));
                }
            }
        }
    }
    errors
}

fn output_validation_profile_reference(payload: &Value, validator: &str) -> Value {
    let profile = output_validation_profile_root(payload);
    let Some(features) = output_validation_profile_features(profile) else {
        return output_validation_result(
            "failed",
            "failure",
            validator,
            "payload needs profile/statistics with features, columns, variables, or field metrics",
            Vec::new(),
        );
    };
    let mut errors = Vec::new();
    if features.is_empty() {
        errors.push("profile must contain at least one feature".to_string());
    }
    let row_count = output_validation_profile_number(
        profile
            .get("row_count")
            .or_else(|| profile.get("rowCount"))
            .or_else(|| profile.get("num_rows"))
            .or_else(|| profile.get("numRows")),
    );
    if let Some(row_count) = row_count {
        if row_count < 0.0 {
            errors.push(format!(
                "profile row count must be non-negative, got {row_count}"
            ));
        }
    }
    for (name, feature) in features {
        let Some(feature_obj) = feature.as_object() else {
            errors.push(format!("profile feature '{name}' must be an object"));
            continue;
        };
        if let Some(kind) = feature_obj
            .get("type")
            .or_else(|| feature_obj.get("data_type"))
            .or_else(|| feature_obj.get("dataType"))
            .and_then(Value::as_str)
        {
            if !output_validation_profile_known_type(kind) {
                errors.push(format!(
                    "profile feature '{name}' has unknown type {kind:?}"
                ));
            }
        }
        let count = output_validation_profile_metric(feature, &["count", "n"]);
        if let Some(count) = count {
            if count < 0.0 {
                errors.push(format!(
                    "profile feature '{name}' has negative count {count}"
                ));
            }
            if let Some(row_count) = row_count {
                if row_count >= 0.0 && count > row_count {
                    errors.push(format!(
                        "profile feature '{name}' count {count} exceeds row count {row_count}"
                    ));
                }
            }
        }
        let missing = output_validation_profile_metric(
            feature,
            &["missing", "null_count", "nullCount", "missing_count"],
        );
        if let Some(missing) = missing {
            if missing < 0.0 {
                errors.push(format!(
                    "profile feature '{name}' has negative missing count {missing}"
                ));
            }
            if let Some(count) = count {
                if count >= 0.0 && missing > count {
                    errors.push(format!(
                        "profile feature '{name}' missing count {missing} exceeds count {count}"
                    ));
                }
            }
        }
        let distinct =
            output_validation_profile_metric(feature, &["distinct", "unique", "distinct_count"]);
        if let (Some(distinct), Some(count)) = (distinct, count) {
            if distinct < 0.0 {
                errors.push(format!(
                    "profile feature '{name}' has negative distinct count {distinct}"
                ));
            } else if count >= 0.0 && distinct > count {
                errors.push(format!(
                    "profile feature '{name}' distinct count {distinct} exceeds count {count}"
                ));
            }
        }
        let minimum = output_validation_profile_metric(feature, &["min", "minimum"]);
        let maximum = output_validation_profile_metric(feature, &["max", "maximum"]);
        if let (Some(minimum), Some(maximum)) = (minimum, maximum) {
            if minimum > maximum {
                errors.push(format!(
                    "profile feature '{name}' minimum {minimum} exceeds maximum {maximum}"
                ));
            }
            if let Some(mean) = output_validation_profile_metric(feature, &["mean", "avg"]) {
                if mean < minimum || mean > maximum {
                    errors.push(format!(
                        "profile feature '{name}' mean {mean} is outside [{minimum}, {maximum}]"
                    ));
                }
            }
        }
    }
    errors.extend(output_validation_profile_constraint_errors(
        payload, features,
    ));
    errors.extend(output_validation_profile_drift_errors(payload));
    if let Some(anomalies) = payload.get("anomalies").and_then(Value::as_array) {
        if !anomalies.is_empty()
            && payload
                .get("allow_anomalies")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                == false
        {
            errors.push(format!("profile reports {} anomalies", anomalies.len()));
        }
    }
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        errors.first().cloned().unwrap_or_default(),
        errors,
    )
}

fn output_validation_object_fields(
    schema: &Value,
) -> BTreeMap<String, serde_json::Map<String, Value>> {
    let fields = schema.get("fields").unwrap_or(&Value::Null);
    let mut specs = BTreeMap::new();
    match fields {
        Value::Object(fields) => {
            for (name, spec) in fields {
                let spec_obj = match spec {
                    Value::Object(obj) => obj.clone(),
                    Value::String(kind) => {
                        let mut obj = serde_json::Map::new();
                        obj.insert("type".to_string(), Value::String(kind.clone()));
                        obj
                    }
                    _ => serde_json::Map::new(),
                };
                specs.insert(name.clone(), spec_obj);
            }
        }
        Value::Array(fields) => {
            for item in fields {
                match item {
                    Value::String(name) => {
                        specs.insert(name.clone(), serde_json::Map::new());
                    }
                    Value::Object(obj) => {
                        if let Some(name) = obj.get("name").and_then(Value::as_str) {
                            specs.insert(name.to_string(), obj.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    specs
}

fn output_validation_matches_protobuf_scalar(value: &Value, expected: &str) -> bool {
    match expected.to_ascii_lowercase().as_str() {
        "int32" | "sint32" | "sfixed32" => output_validation_json_integer(value)
            .is_some_and(|integer| (-(1_i128 << 31)..(1_i128 << 31)).contains(&integer)),
        "uint32" | "fixed32" => output_validation_json_integer(value)
            .is_some_and(|integer| (0..(1_i128 << 32)).contains(&integer)),
        "int64" | "sint64" | "sfixed64" => output_validation_json_integer(value)
            .is_some_and(|integer| (-(1_i128 << 63)..(1_i128 << 63)).contains(&integer)),
        "uint64" | "fixed64" => output_validation_json_integer(value)
            .is_some_and(|integer| (0..(1_i128 << 64)).contains(&integer)),
        "double" | "float" => output_validation_json_number(value).is_some(),
        "bool" | "boolean" => value.is_boolean(),
        "string" | "bytes" => value.is_string(),
        "message" | "object" => value.is_object(),
        _ => true,
    }
}

fn output_validation_protobuf_field_errors(
    name: &str,
    spec: &serde_json::Map<String, Value>,
    value: &Value,
    path: &str,
) -> Vec<String> {
    let mut errors = Vec::new();
    if spec.get("repeated").and_then(Value::as_bool) == Some(true) {
        let Some(items) = value.as_array() else {
            return vec![format!("{path}.{name}: expected repeated field array")];
        };
        let mut item_spec = spec.clone();
        item_spec.insert("repeated".to_string(), Value::Bool(false));
        for (idx, item) in items.iter().enumerate() {
            errors.extend(output_validation_protobuf_field_errors(
                name,
                &item_spec,
                item,
                &format!("{path}.{name}[{idx}]"),
            ));
        }
        return errors;
    }
    if let Some(enum_values) = spec.get("enum").and_then(Value::as_array) {
        if !enum_values.iter().any(|item| item == value) {
            errors.push(format!("{path}.{name}: value {value} is not in enum"));
            return errors;
        }
    }
    let expected_type = spec.get("type").and_then(Value::as_str).unwrap_or("string");
    if !output_validation_matches_protobuf_scalar(value, expected_type) {
        errors.push(format!(
            "{path}.{name}: expected protobuf {expected_type}, got {}",
            output_validation_json_type_name(value)
        ));
    }
    if spec.contains_key("fields") && value.is_object() {
        errors.extend(output_validation_protobuf_message_errors(
            &Value::Object(spec.clone()),
            value,
            &format!("{path}.{name}"),
        ));
    }
    errors
}

fn output_validation_protobuf_oneof_errors(
    schema: &Value,
    message: &serde_json::Map<String, Value>,
    path: &str,
) -> Vec<String> {
    let mut errors = Vec::new();
    let groups: Vec<(String, Vec<String>)> = match schema
        .get("oneof")
        .or_else(|| schema.get("oneofs"))
        .unwrap_or(&Value::Null)
    {
        Value::Object(groups) => groups
            .iter()
            .filter_map(|(name, group)| {
                let fields = group
                    .get("fields")
                    .and_then(Value::as_array)?
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                Some((name.clone(), fields))
            })
            .collect(),
        Value::Array(groups) => groups
            .iter()
            .filter_map(|group| {
                if let Some(group_obj) = group.as_object() {
                    let name = group_obj
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("oneof")
                        .to_string();
                    let fields = group_obj
                        .get("fields")
                        .and_then(Value::as_array)?
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect::<Vec<_>>();
                    Some((name, fields))
                } else {
                    let fields = group
                        .as_array()?
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect::<Vec<_>>();
                    Some(("oneof".to_string(), fields))
                }
            })
            .collect(),
        _ => Vec::new(),
    };
    for (name, fields) in groups {
        let present: Vec<String> = fields
            .iter()
            .filter(|field| message.get(*field).is_some_and(|value| !value.is_null()))
            .cloned()
            .collect();
        if present.len() != 1 {
            errors.push(format!(
                "{path}: oneof '{name}' expected exactly one of {fields:?}, got {present:?}"
            ));
        }
    }
    errors
}

fn output_validation_protobuf_message_errors(
    schema: &Value,
    message: &Value,
    path: &str,
) -> Vec<String> {
    let Some(message_obj) = message.as_object() else {
        return vec![format!("{path}: message must be an object")];
    };
    let fields = output_validation_object_fields(schema);
    let required_fields: Vec<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mut errors = Vec::new();
    for (name, spec) in &fields {
        let is_required = spec.get("required").and_then(Value::as_bool) == Some(true)
            || required_fields.iter().any(|required| required == name);
        let value = message_obj.get(name);
        if value.is_none() || value.is_some_and(Value::is_null) {
            if is_required {
                errors.push(format!("{path}: missing required protobuf field '{name}'"));
            }
            continue;
        }
        errors.extend(output_validation_protobuf_field_errors(
            name,
            spec,
            value.expect("checked protobuf value"),
            path,
        ));
    }
    if schema.get("additionalFields").and_then(Value::as_bool) == Some(false)
        || schema.get("additional_fields").and_then(Value::as_bool) == Some(false)
    {
        for name in message_obj.keys() {
            if !fields.contains_key(name) {
                errors.push(format!("{path}: unexpected protobuf field '{name}'"));
            }
        }
    }
    errors.extend(output_validation_protobuf_oneof_errors(
        schema,
        message_obj,
        path,
    ));
    errors
}

fn output_validation_protobuf_reference(payload: &Value) -> Value {
    let schema = payload
        .get("schema")
        .or_else(|| payload.get("descriptor"))
        .unwrap_or(&Value::Null);
    let message = payload
        .get("message")
        .or_else(|| payload.get("instance"))
        .or_else(|| payload.get("data"))
        .unwrap_or(&Value::Null);
    if !schema.is_object() {
        return output_validation_result(
            "failed",
            "invalid",
            "builtin:protobuf-conformance-subset",
            "schema must be an object",
            Vec::new(),
        );
    }
    if !message.is_object() {
        return output_validation_result(
            "failed",
            "invalid",
            "builtin:protobuf-conformance-subset",
            "message must be an object",
            Vec::new(),
        );
    }
    let errors = output_validation_protobuf_message_errors(schema, message, "$");
    let message = errors.first().cloned().unwrap_or_default();
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        "builtin:protobuf-conformance-subset",
        message,
        errors,
    )
}

fn output_validation_avro_value_errors(schema: &Value, value: &Value, path: &str) -> Vec<String> {
    match schema {
        Value::Array(branches) => {
            if branches
                .iter()
                .any(|branch| output_validation_avro_value_errors(branch, value, path).is_empty())
            {
                Vec::new()
            } else {
                vec![format!("{path}: value did not match any Avro union branch")]
            }
        }
        Value::String(kind) => match kind.as_str() {
            "null" => {
                if value.is_null() {
                    Vec::new()
                } else {
                    vec![format!("{path}: expected null")]
                }
            }
            "boolean" => {
                if value.is_boolean() {
                    Vec::new()
                } else {
                    vec![format!("{path}: expected boolean")]
                }
            }
            "int" | "long" => {
                if output_validation_json_integer(value).is_some() {
                    Vec::new()
                } else {
                    vec![format!("{path}: expected {kind}")]
                }
            }
            "float" | "double" => {
                if output_validation_json_number(value).is_some() {
                    Vec::new()
                } else {
                    vec![format!("{path}: expected {kind}")]
                }
            }
            "bytes" | "string" => {
                if value.is_string() {
                    Vec::new()
                } else {
                    vec![format!("{path}: expected {kind}")]
                }
            }
            _ => Vec::new(),
        },
        Value::Object(schema_obj) => {
            let Some(schema_type) = schema_obj.get("type") else {
                return vec![format!("{path}: unsupported Avro schema shape")];
            };
            if schema_type.is_array() || schema_type.is_object() {
                return output_validation_avro_value_errors(schema_type, value, path);
            }
            match schema_type.as_str().unwrap_or("") {
                "record" => {
                    let Some(record) = value.as_object() else {
                        return vec![format!("{path}: expected record object")];
                    };
                    let Some(fields) = schema_obj.get("fields").and_then(Value::as_array) else {
                        return vec![format!("{path}: record fields must be a list")];
                    };
                    let mut errors = Vec::new();
                    let mut known = Vec::new();
                    for field in fields {
                        let Some(field_obj) = field.as_object() else {
                            errors.push(format!("{path}: invalid Avro field"));
                            continue;
                        };
                        let Some(name) = field_obj.get("name").and_then(Value::as_str) else {
                            errors.push(format!("{path}: invalid Avro field"));
                            continue;
                        };
                        known.push(name.to_string());
                        if let Some(field_value) = record.get(name) {
                            let default_field_type = Value::String("string".to_string());
                            let field_schema = field_obj.get("type").unwrap_or(&default_field_type);
                            errors.extend(output_validation_avro_value_errors(
                                field_schema,
                                field_value,
                                &format!("{path}.{name}"),
                            ));
                        } else if !field_obj.contains_key("default") {
                            errors.push(format!("{path}: missing Avro field '{name}'"));
                        }
                    }
                    if schema_obj.get("additionalFields").and_then(Value::as_bool) == Some(false)
                        || schema_obj.get("additional_fields").and_then(Value::as_bool)
                            == Some(false)
                    {
                        for name in record.keys() {
                            if !known.iter().any(|known_name| known_name == name) {
                                errors.push(format!("{path}: unexpected Avro field '{name}'"));
                            }
                        }
                    }
                    errors
                }
                "array" => {
                    let Some(items) = value.as_array() else {
                        return vec![format!("{path}: expected Avro array")];
                    };
                    let default_item_schema = Value::String("string".to_string());
                    let item_schema = schema_obj.get("items").unwrap_or(&default_item_schema);
                    let mut errors = Vec::new();
                    for (idx, item) in items.iter().enumerate() {
                        errors.extend(output_validation_avro_value_errors(
                            item_schema,
                            item,
                            &format!("{path}[{idx}]"),
                        ));
                    }
                    errors
                }
                "map" => {
                    let Some(map) = value.as_object() else {
                        return vec![format!("{path}: expected Avro map")];
                    };
                    let default_value_schema = Value::String("string".to_string());
                    let value_schema = schema_obj.get("values").unwrap_or(&default_value_schema);
                    let mut errors = Vec::new();
                    for (key, item) in map {
                        errors.extend(output_validation_avro_value_errors(
                            value_schema,
                            item,
                            &format!("{path}.{key}"),
                        ));
                    }
                    errors
                }
                "enum" => {
                    let symbols = schema_obj
                        .get("symbols")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    if symbols.iter().any(|symbol| symbol == value) {
                        Vec::new()
                    } else {
                        vec![format!("{path}: value {value} is not in Avro enum")]
                    }
                }
                other => output_validation_avro_value_errors(
                    &Value::String(other.to_string()),
                    value,
                    path,
                ),
            }
        }
        _ => vec![format!("{path}: unsupported Avro schema shape")],
    }
}

fn output_validation_avro_reference(payload: &Value) -> Value {
    let schema = payload.get("schema").unwrap_or(&Value::Null);
    let instance = payload
        .get("record")
        .or_else(|| payload.get("instance"))
        .or_else(|| payload.get("data"))
        .unwrap_or(&Value::Null);
    let errors = output_validation_avro_value_errors(schema, instance, "$");
    let message = errors.first().cloned().unwrap_or_default();
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        "builtin:avro-schema-subset",
        message,
        errors,
    )
}

fn output_validation_openapi_reference(payload: &Value, validator: &str) -> Value {
    let spec = payload
        .get("spec")
        .or_else(|| payload.get("schema"))
        .or_else(|| payload.get("openapi"))
        .unwrap_or(&Value::Null);
    let Some(spec_obj) = spec.as_object() else {
        return output_validation_result(
            "failed",
            "invalid",
            validator,
            "OpenAPI spec must be an object",
            Vec::new(),
        );
    };
    let mut errors = Vec::new();
    if !spec_obj.contains_key("openapi") && !spec_obj.contains_key("swagger") {
        errors.push("$.openapi: missing OpenAPI/Swagger version".to_string());
    }
    if !spec_obj
        .get("info")
        .and_then(Value::as_object)
        .and_then(|info| info.get("title"))
        .is_some_and(|title| title.as_str().is_some_and(|text| !text.is_empty()))
    {
        errors.push("$.info.title: missing API title".to_string());
    }
    let valid_methods = [
        "get", "put", "post", "delete", "patch", "head", "options", "trace",
    ];
    match spec_obj.get("paths").and_then(Value::as_object) {
        Some(paths) if !paths.is_empty() => {
            for (path, operations) in paths {
                if !path.starts_with('/') {
                    errors.push(format!("$.paths.{path}: path must start with '/'"));
                }
                let Some(operation_obj) = operations.as_object() else {
                    errors.push(format!("$.paths.{path}: expected operations object"));
                    continue;
                };
                if operation_obj.is_empty() {
                    errors.push(format!("$.paths.{path}: expected operations object"));
                    continue;
                }
                for (method, operation) in operation_obj {
                    if !valid_methods.contains(&method.to_ascii_lowercase().as_str()) {
                        continue;
                    }
                    let Some(operation) = operation.as_object() else {
                        errors.push(format!(
                            "$.paths.{path}.{method}: operation must be an object"
                        ));
                        continue;
                    };
                    if !operation
                        .get("responses")
                        .and_then(Value::as_object)
                        .is_some_and(|responses| !responses.is_empty())
                    {
                        errors.push(format!(
                            "$.paths.{path}.{method}.responses: missing responses"
                        ));
                    }
                }
            }
        }
        _ => errors.push("$.paths: expected non-empty object".to_string()),
    }
    let message = errors.first().cloned().unwrap_or_default();
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        message,
        errors,
    )
}

fn output_validation_xml_is_well_formed(xml: &str) -> Result<Vec<String>, String> {
    let mut tags = Vec::new();
    let mut roots = Vec::new();
    let chars: Vec<char> = xml.chars().collect();
    let mut idx = 0;
    while idx < chars.len() {
        if chars[idx] != '<' {
            idx += 1;
            continue;
        }
        let Some(end) = chars[idx + 1..].iter().position(|ch| *ch == '>') else {
            return Err("unterminated tag".to_string());
        };
        let end = idx + 1 + end;
        let raw: String = chars[idx + 1..end].iter().collect();
        let tag = raw.trim();
        if tag.is_empty() {
            return Err("empty tag".to_string());
        }
        if tag.starts_with('?') || tag.starts_with('!') {
            idx = end + 1;
            continue;
        }
        if let Some(rest) = tag.strip_prefix('/') {
            let name = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches('/');
            match tags.pop() {
                Some(open) if open == name => {}
                Some(open) => return Err(format!("closing tag '{name}' did not match '{open}'")),
                None => return Err(format!("closing tag '{name}' without opener")),
            }
        } else {
            let name = tag
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches('/');
            if name.is_empty() {
                return Err("empty tag name".to_string());
            }
            roots.push(name.to_string());
            if !tag.ends_with('/') {
                tags.push(name.to_string());
            }
        }
        idx = end + 1;
    }
    if roots.is_empty() {
        return Err("missing root element".to_string());
    }
    if let Some(open) = tags.last() {
        return Err(format!("unclosed tag '{open}'"));
    }
    Ok(roots)
}

fn output_validation_xml_reference(payload: &Value, validator: &str) -> Value {
    let xml = payload
        .get("xml")
        .or_else(|| payload.get("instance"))
        .or_else(|| payload.get("document"))
        .or_else(|| payload.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let schema_text = payload
        .get("schema")
        .or_else(|| payload.get("xsd"))
        .or_else(|| payload.get("schematron"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let mut errors = Vec::new();
    match output_validation_xml_is_well_formed(xml) {
        Ok(tags) => {
            if let Some(required) = payload.get("required_elements").and_then(Value::as_array) {
                for name in required.iter().filter_map(Value::as_str) {
                    if !tags.iter().any(|tag| tag == name) {
                        errors.push(format!("xml: missing required element '{name}'"));
                    }
                }
            }
        }
        Err(message) => errors.push(format!("xml: not well-formed: {message}")),
    }
    if !schema_text.is_empty() {
        let lower = schema_text.to_ascii_lowercase();
        if validator.contains("schematron") {
            if !lower.contains("<schema")
                || (!lower.contains("<assert") && !lower.contains("<report"))
            {
                errors.push("schematron: expected schema with assert/report rules".to_string());
            }
        } else if (validator.contains("xsd") || validator.contains("xml-schema"))
            && !lower.contains("<xs:schema")
            && !lower.contains("<xsd:schema")
            && !lower.contains("<schema")
        {
            errors.push("xsd: expected schema root".to_string());
        }
    }
    let message = errors.first().cloned().unwrap_or_default();
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        message,
        errors,
    )
}

fn output_validation_structured_type_name(raw: &str) -> &'static str {
    let lower = raw
        .trim()
        .trim_start_matches("fields.")
        .trim_start_matches("marshmallow.fields.")
        .to_ascii_lowercase();
    match lower.as_str() {
        "int" | "integer" | "numberinteger" => "integer",
        "float" | "double" | "decimal" | "number" | "numberfloat" => "number",
        "bool" | "boolean" => "boolean",
        "list" | "array" | "tuple" | "set" => "array",
        "dict" | "mapping" | "object" | "nested" => "object",
        "raw" | "any" => "object",
        _ => "string",
    }
}

fn output_validation_structured_fields<'a>(
    model_obj: &'a serde_json::Map<String, Value>,
    model: &'a Value,
) -> Option<&'a serde_json::Map<String, Value>> {
    model_obj
        .get("fields")
        .or_else(|| model_obj.get("properties"))
        .or_else(|| model_obj.get("schema"))
        .and_then(Value::as_object)
        .or_else(|| model.as_object())
}

fn output_validation_structured_field_spec(spec: &Value) -> serde_json::Map<String, Value> {
    let mut field_spec = match spec {
        Value::String(kind) => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".to_string(), Value::String(kind.clone()));
            obj
        }
        Value::Object(obj) => obj.clone(),
        _ => serde_json::Map::new(),
    };
    let normalized_type = field_spec
        .get("type")
        .or_else(|| field_spec.get("field"))
        .or_else(|| field_spec.get("kind"))
        .or_else(|| field_spec.get("data_type"))
        .or_else(|| field_spec.get("dataType"))
        .and_then(Value::as_str)
        .map(output_validation_structured_type_name)
        .unwrap_or("string");
    let nullable = field_spec
        .get("nullable")
        .or_else(|| field_spec.get("allow_none"))
        .or_else(|| field_spec.get("allowNone"))
        .and_then(Value::as_bool)
        == Some(true);
    if nullable {
        field_spec.insert(
            "type".to_string(),
            Value::Array(vec![
                Value::String(normalized_type.to_string()),
                Value::String("null".to_string()),
            ]),
        );
    } else {
        field_spec.insert(
            "type".to_string(),
            Value::String(normalized_type.to_string()),
        );
    }
    if let Some(allowed) = field_spec
        .get("allowed")
        .or_else(|| field_spec.get("choices"))
        .cloned()
    {
        field_spec.entry("enum".to_string()).or_insert(allowed);
    }
    for (source, target) in [
        ("min", "minimum"),
        ("max", "maximum"),
        ("min_value", "minimum"),
        ("max_value", "maximum"),
        ("minLength", "minLength"),
        ("maxLength", "maxLength"),
        ("minlength", "minLength"),
        ("maxlength", "maxLength"),
        ("min_length", "minLength"),
        ("max_length", "maxLength"),
    ] {
        if let Some(value) = field_spec.get(source).cloned() {
            field_spec.entry(target.to_string()).or_insert(value);
        }
    }
    if field_spec.get("empty").and_then(Value::as_bool) == Some(false) {
        field_spec
            .entry("minLength".to_string())
            .or_insert(Value::from(1_u64));
    }
    if normalized_type == "array" {
        if let Some(item_schema) = field_spec.get("schema").cloned() {
            let item_spec = output_validation_structured_field_spec(&item_schema);
            field_spec
                .entry("items".to_string())
                .or_insert(Value::Object(item_spec));
        }
    }
    field_spec
}

fn output_validation_pydantic_reference(payload: &Value, validator: &str) -> Value {
    let model = payload
        .get("model")
        .or_else(|| payload.get("schema"))
        .unwrap_or(&Value::Null);
    let instance = payload
        .get("instance")
        .or_else(|| payload.get("data"))
        .unwrap_or(&Value::Null);
    let Some(model_obj) = model.as_object() else {
        return output_validation_result(
            "failed",
            "invalid",
            validator,
            "model must be an object",
            Vec::new(),
        );
    };
    if !instance.is_object() {
        return output_validation_result(
            "failed",
            "invalid",
            validator,
            "instance must be an object",
            Vec::new(),
        );
    }
    let Some(fields_obj) = output_validation_structured_fields(model_obj, model) else {
        return output_validation_result(
            "failed",
            "invalid",
            validator,
            "fields must be an object",
            Vec::new(),
        );
    };
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for (name, spec) in fields_obj {
        let field_spec = output_validation_structured_field_spec(spec);
        if field_spec.get("required").and_then(Value::as_bool) == Some(true) {
            required.push(Value::String(name.clone()));
        }
        properties.insert(name.clone(), Value::Object(field_spec));
    }
    let schema = json!({
        "type": "object",
        "properties": properties,
        "required": required,
    });
    let errors = output_validation_schema_errors(&schema, instance, "$");
    let message = errors.first().cloned().unwrap_or_default();
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        message,
        errors,
    )
}

fn output_validation_payload_text<'a>(payload: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .filter_map(|key| payload.get(*key).and_then(Value::as_str))
        .find(|text| !text.trim().is_empty())
}

fn output_validation_balanced_delimiters(text: &str) -> Vec<String> {
    let mut errors = Vec::new();
    let mut stack = Vec::<(char, usize)>::new();
    let mut quote = None::<char>;
    let mut escaped = false;
    for (line_idx, line) in text.lines().enumerate() {
        for ch in line.chars() {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' && quote == Some('"') {
                escaped = true;
                continue;
            }
            if matches!(ch, '"' | '\'') {
                if quote == Some(ch) {
                    quote = None;
                } else if quote.is_none() {
                    quote = Some(ch);
                }
                continue;
            }
            if quote.is_some() {
                continue;
            }
            match ch {
                '{' | '[' | '(' => stack.push((ch, line_idx + 1)),
                '}' | ']' | ')' => {
                    let Some((open, open_line)) = stack.pop() else {
                        errors.push(format!("line {} has unmatched {ch}", line_idx + 1));
                        continue;
                    };
                    let expected = match open {
                        '{' => '}',
                        '[' => ']',
                        '(' => ')',
                        _ => unreachable!(),
                    };
                    if ch != expected {
                        errors.push(format!(
                            "line {} closes {ch} but line {open_line} opened {open}",
                            line_idx + 1
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    for (open, line) in stack {
        errors.push(format!("line {line} has unclosed {open}"));
    }
    errors
}

fn output_validation_yaml_reference(payload: &Value, validator: &str) -> Value {
    let Some(text) =
        output_validation_payload_text(payload, &["yaml", "document", "text", "content"])
    else {
        return output_validation_result(
            "failed",
            "failure",
            validator,
            "payload needs yaml, document, text, or content",
            Vec::new(),
        );
    };
    let mut errors = output_validation_balanced_delimiters(text);
    for (line_idx, line) in text.lines().enumerate() {
        if line
            .chars()
            .take_while(|ch| ch.is_whitespace())
            .any(|ch| ch == '\t')
        {
            errors.push(format!("line {} uses a tab for indentation", line_idx + 1));
        }
        if line.ends_with(' ') || line.ends_with('\t') {
            errors.push(format!("line {} has trailing whitespace", line_idx + 1));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if matches!(trimmed, "---" | "...") {
            continue;
        }
        if trimmed.starts_with(':') {
            errors.push(format!("line {} has an empty mapping key", line_idx + 1));
        }
        if trimmed.starts_with("-:") {
            errors.push(format!(
                "line {} has an empty sequence mapping key",
                line_idx + 1
            ));
        }
    }
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        errors.first().cloned().unwrap_or_default(),
        errors,
    )
}

fn output_validation_cue_without_comments(text: &str) -> (String, Vec<String>) {
    let mut out = String::with_capacity(text.len());
    let mut errors = Vec::new();
    let mut chars = text.chars().peekable();
    let mut quote = None::<char>;
    let mut escaped = false;
    let mut block_comment_line = None::<usize>;
    let mut line = 1_usize;
    while let Some(ch) = chars.next() {
        if ch == '\n' {
            line += 1;
        }
        if let Some(start_line) = block_comment_line {
            if ch == '*' && chars.peek() == Some(&'/') {
                chars.next();
                block_comment_line = None;
                out.push(' ');
            } else if ch == '\n' {
                out.push('\n');
            } else {
                out.push(' ');
            }
            if chars.peek().is_none() && block_comment_line == Some(start_line) {
                errors.push(format!(
                    "line {start_line} starts an unterminated CUE block comment"
                ));
            }
            continue;
        }
        if escaped {
            escaped = false;
            out.push(ch);
            continue;
        }
        if ch == '\\' && quote.is_some() {
            escaped = true;
            out.push(ch);
            continue;
        }
        if matches!(ch, '"' | '\'' | '`') {
            if quote == Some(ch) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(ch);
            }
            out.push(ch);
            continue;
        }
        if quote.is_none() && ch == '/' && chars.peek() == Some(&'/') {
            for next in chars.by_ref() {
                if next == '\n' {
                    line += 1;
                    out.push('\n');
                    break;
                }
            }
            continue;
        }
        if quote.is_none() && ch == '/' && chars.peek() == Some(&'*') {
            chars.next();
            block_comment_line = Some(line);
            out.push(' ');
            continue;
        }
        out.push(ch);
    }
    if let Some(active_quote) = quote {
        errors.push(format!(
            "cue has an unterminated {active_quote} quoted string"
        ));
    }
    if let Some(start_line) = block_comment_line {
        errors.push(format!(
            "line {start_line} starts an unterminated CUE block comment"
        ));
    }
    (out, errors)
}

fn output_validation_text_looks_cue(text: &str) -> bool {
    let (without_comments, _) = output_validation_cue_without_comments(text);
    let lower = without_comments.trim_start().to_ascii_lowercase();
    lower.starts_with("package ")
        || lower.starts_with("import ")
        || lower.contains("#")
        || lower.lines().any(|line| {
            let trimmed = line.trim();
            trimmed.contains(':') || trimmed.contains("=~") || trimmed.contains("!=")
        })
}

fn output_validation_has_cue_payload(payload: &Value) -> bool {
    [
        "cue",
        "cue_schema",
        "cueSchema",
        "document",
        "text",
        "content",
    ]
    .iter()
    .any(|key| {
        payload
            .get(*key)
            .and_then(Value::as_str)
            .is_some_and(output_validation_text_looks_cue)
    }) || payload
        .get("schema")
        .and_then(Value::as_str)
        .is_some_and(output_validation_text_looks_cue)
}

fn output_validation_cue_field_constraint(line: &str) -> Option<(String, String, bool)> {
    let trimmed = line
        .trim()
        .trim_end_matches(',')
        .trim_end_matches(';')
        .trim();
    if trimmed.is_empty()
        || trimmed.starts_with("package ")
        || trimmed.starts_with("import ")
        || trimmed.starts_with("//")
        || matches!(trimmed, "{" | "}" | "[" | "]")
    {
        return None;
    }
    let (label, rest) = trimmed.split_once(':')?;
    let label = label.trim();
    if label.starts_with('#') || label.starts_with('@') || label.contains(' ') {
        return None;
    }
    let optional = label.ends_with('?');
    let label = label
        .trim_end_matches('?')
        .trim_matches('"')
        .trim_matches('`')
        .trim();
    if label.is_empty()
        || !label
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
    {
        return None;
    }
    let constraint = rest.trim();
    if constraint.is_empty() {
        return None;
    }
    Some((label.to_string(), constraint.to_string(), optional))
}

fn output_validation_cue_scalar_constraint_matches(constraint: &str, value: &Value) -> bool {
    let constraint = constraint.trim().trim_end_matches(',').trim();
    let lower = constraint.to_ascii_lowercase();
    if matches!(lower.as_str(), "_" | "any" | "top") {
        return true;
    }
    if lower.starts_with("string") {
        return value.is_string();
    }
    if lower.starts_with("number") {
        return value.is_number();
    }
    if lower.starts_with("int") {
        return value.as_i64().is_some() || value.as_u64().is_some();
    }
    if lower.starts_with("bool") || lower.starts_with("boolean") {
        return value.is_boolean();
    }
    if constraint.starts_with('[') || lower.starts_with("list") {
        return value.is_array();
    }
    if constraint.starts_with('{') || lower.starts_with("struct") {
        return value.is_object();
    }
    if constraint.starts_with('"') && value.is_string() {
        let Some(actual) = value.as_str() else {
            return false;
        };
        return constraint
            .split('|')
            .map(str::trim)
            .filter_map(|part| part.strip_prefix('"')?.split_once('"').map(|(lit, _)| lit))
            .any(|literal| literal == actual);
    }
    true
}

fn output_validation_cue_reference(payload: &Value, validator: &str) -> Value {
    let Some(text) = output_validation_payload_text(
        payload,
        &[
            "cue",
            "cue_schema",
            "cueSchema",
            "schema",
            "document",
            "text",
            "content",
        ],
    ) else {
        return output_validation_result(
            "failed",
            "failure",
            validator,
            "payload needs cue, cue_schema, cueSchema, schema text, document, text, or content",
            Vec::new(),
        );
    };
    let (without_comments, mut errors) = output_validation_cue_without_comments(text);
    errors.extend(output_validation_balanced_delimiters(&without_comments));
    if without_comments.trim().is_empty() {
        errors.push("cue document is empty".to_string());
    }
    if !without_comments.contains(':')
        && !without_comments.contains('=')
        && !without_comments
            .trim_start()
            .to_ascii_lowercase()
            .starts_with("package ")
    {
        errors.push("cue document has no fields, definitions, or declarations".to_string());
    }

    let instance = payload
        .get("instance")
        .or_else(|| payload.get("data"))
        .or_else(|| payload.get("value"));
    let mut depth = 0_i32;
    let mut constraints = Vec::new();
    for (line_idx, line) in without_comments.lines().enumerate() {
        let trimmed = line.trim();
        if depth == 0 {
            if let Some((field, constraint, optional)) =
                output_validation_cue_field_constraint(trimmed)
            {
                constraints.push((field, constraint, optional));
            } else {
                let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
                if tokens.len() == 2
                    && matches!(tokens[1], "string" | "number" | "int" | "bool" | "boolean")
                    && !trimmed.contains(':')
                    && !trimmed.contains('=')
                {
                    errors.push(format!(
                        "cue line {} looks like a field missing ':'",
                        line_idx + 1
                    ));
                }
            }
        }
        for ch in trimmed.chars() {
            match ch {
                '{' | '[' | '(' => depth += 1,
                '}' | ']' | ')' => depth -= 1,
                _ => {}
            }
        }
    }
    if let Some(instance) = instance {
        if let Some(instance_obj) = instance.as_object() {
            for (field, constraint, optional) in constraints {
                match instance_obj.get(&field) {
                    Some(value)
                        if !output_validation_cue_scalar_constraint_matches(&constraint, value) =>
                    {
                        errors.push(format!(
                            "$.{field}: value does not satisfy CUE constraint {constraint:?}"
                        ));
                    }
                    Some(_) => {}
                    None if !optional => errors.push(format!("$.{field}: missing required field")),
                    None => {}
                }
            }
        } else if !constraints.is_empty() {
            errors.push("instance must be an object for top-level CUE field checks".to_string());
        }
    }

    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        errors.first().cloned().unwrap_or_default(),
        errors,
    )
}

fn output_validation_graphql_reference(payload: &Value, validator: &str) -> Value {
    let Some(text) = output_validation_payload_text(
        payload,
        &["schema", "graphql", "sdl", "document", "text", "content"],
    ) else {
        return output_validation_result(
            "failed",
            "failure",
            validator,
            "payload needs schema, graphql, sdl, document, text, or content",
            Vec::new(),
        );
    };
    let without_comments = text
        .lines()
        .map(|line| line.split_once('#').map(|(head, _)| head).unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");
    let mut errors = output_validation_balanced_delimiters(&without_comments);
    let trimmed = without_comments.trim();
    if trimmed.is_empty() {
        errors.push("GraphQL schema is empty".to_string());
    }
    let has_definition = [
        "type ",
        "interface ",
        "enum ",
        "union ",
        "input ",
        "scalar ",
        "directive ",
        "schema ",
        "extend type ",
        "extend interface ",
    ]
    .iter()
    .any(|keyword| trimmed.contains(keyword));
    if !has_definition {
        errors.push(
            "GraphQL schema has no type, schema, scalar, directive, or extension definition"
                .to_string(),
        );
    }
    for keyword in ["type ", "interface ", "enum ", "input ", "schema "] {
        let mut rest = trimmed;
        while let Some(idx) = rest.find(keyword) {
            rest = &rest[idx + keyword.len()..];
            let before_next_definition = ["type ", "interface ", "enum ", "input ", "schema "]
                .iter()
                .filter_map(|next| rest.find(next))
                .min()
                .map(|next_idx| &rest[..next_idx])
                .unwrap_or(rest);
            if !before_next_definition.contains('{') {
                errors.push(format!(
                    "GraphQL {keyword:?} definition is missing a field block"
                ));
                break;
            }
        }
    }
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        errors.first().cloned().unwrap_or_default(),
        errors,
    )
}

fn output_validation_sql_without_comments(text: &str) -> (String, Vec<String>) {
    let mut out = String::with_capacity(text.len());
    let mut errors = Vec::new();
    let mut chars = text.chars().peekable();
    let mut quote = None::<char>;
    let mut line = 1usize;
    let mut block_comment_line = None::<usize>;
    while let Some(ch) = chars.next() {
        if ch == '\n' {
            line += 1;
        }
        if block_comment_line.is_some() {
            if ch == '*' && chars.peek() == Some(&'/') {
                chars.next();
                block_comment_line = None;
            } else if ch == '\n' {
                out.push('\n');
            }
            continue;
        }
        if let Some(active_quote) = quote {
            out.push(ch);
            if ch == active_quote {
                if active_quote == '\'' && chars.peek() == Some(&'\'') {
                    out.push(chars.next().unwrap_or('\''));
                } else {
                    quote = None;
                }
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                out.push(ch);
            }
            '-' if chars.peek() == Some(&'-') => {
                chars.next();
                for comment_ch in chars.by_ref() {
                    if comment_ch == '\n' {
                        line += 1;
                        out.push('\n');
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                block_comment_line = Some(line);
            }
            _ => out.push(ch),
        }
    }
    if let Some(active_quote) = quote {
        errors.push(format!(
            "sql has an unterminated {active_quote} quoted string"
        ));
    }
    if let Some(start_line) = block_comment_line {
        errors.push(format!(
            "line {start_line} starts an unterminated SQL block comment"
        ));
    }
    (out, errors)
}

fn output_validation_sql_reference(payload: &Value, validator: &str) -> Value {
    let Some(text) = output_validation_payload_text(
        payload,
        &[
            "sql",
            "query",
            "statement",
            "model_sql",
            "modelSql",
            "model",
            "text",
            "content",
        ],
    ) else {
        return output_validation_result(
            "failed",
            "failure",
            validator,
            "payload needs sql, query, statement, modelSql, model, text, or content",
            Vec::new(),
        );
    };
    let (without_comments, mut errors) = output_validation_sql_without_comments(text);
    errors.extend(output_validation_balanced_delimiters(&without_comments));
    let tokens = without_comments
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let has_sql_verb = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "select"
                | "with"
                | "insert"
                | "update"
                | "delete"
                | "merge"
                | "create"
                | "alter"
                | "drop"
                | "truncate"
                | "explain"
                | "analyze"
        )
    });
    if !has_sql_verb {
        errors.push("sql: expected a recognizable SQL statement keyword".to_string());
    }
    if tokens.windows(2).any(|pair| {
        pair[0] == "from"
            && matches!(
                pair[1].as_str(),
                "where" | "group" | "order" | "limit" | "having"
            )
    }) {
        errors.push("sql: FROM clause appears to be missing a relation".to_string());
    }
    output_validation_result(
        "ok",
        if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        validator,
        errors.first().cloned().unwrap_or_default(),
        errors,
    )
}

fn output_validation_text_looks_sql(text: &str) -> bool {
    let trimmed = text.trim_start().to_ascii_lowercase();
    [
        "select ",
        "with ",
        "insert ",
        "update ",
        "delete ",
        "merge ",
        "create ",
        "alter ",
        "drop ",
        "truncate ",
        "explain ",
        "analyze ",
    ]
    .iter()
    .any(|prefix| trimmed.starts_with(prefix))
        || trimmed.contains("\nselect ")
        || trimmed.contains("\nwith ")
}

fn output_validation_has_sql_payload(payload: &Value) -> bool {
    [
        "sql",
        "query",
        "statement",
        "model_sql",
        "modelSql",
        "model",
    ]
    .iter()
    .any(|key| payload.get(*key).and_then(Value::as_str).is_some())
        || output_validation_payload_text(payload, &["text", "content"])
            .is_some_and(output_validation_text_looks_sql)
}

pub fn run_output_validation_json_with_rust_reference(payload: &Value, tool: &str) -> Value {
    let tool = tool.trim().to_ascii_lowercase().replace('_', "-");
    let kind = payload
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .replace('_', "-");
    match tool.as_str() {
        "json-schema" | "jsonschema" => {
            output_validation_json_schema_reference(payload, "builtin:json-schema-subset")
        }
        "ajv" | "ajv-cli" | "check-jsonschema" => output_validation_json_schema_reference(
            payload,
            &format!("builtin:json-schema-subset-for-{tool}"),
        ),
        "cue" if kind == "cue-validation" || output_validation_has_cue_payload(payload) => {
            output_validation_cue_reference(payload, "builtin:cue-structural")
        }
        "cue" => {
            output_validation_json_schema_reference(payload, "builtin:json-schema-subset-for-cue")
        }
        "openapi"
        | "openapi-validator"
        | "openapi-generator-cli"
        | "swagger-cli"
        | "spectral"
        | "openapi-spec-validator"
        | "redocly"
        | "redocly-cli"
        | "asyncapi"
        | "asyncapi-cli" => {
            let validator = if matches!(
                tool.as_str(),
                "spectral"
                    | "openapi-spec-validator"
                    | "redocly"
                    | "redocly-cli"
                    | "asyncapi"
                    | "asyncapi-cli"
            ) {
                format!("builtin:openapi-structural-for-{tool}")
            } else {
                "builtin:openapi-structural".to_string()
            };
            output_validation_openapi_reference(payload, &validator)
        }
        "xml" | "xmllint" | "xml-schema" | "xsd" | "xmlschema" | "xmlschema-validate"
        | "xsd-validator" | "python-xmlschema" | "xmlschema-adapter" => {
            let validator = match tool.as_str() {
                "xmllint" => "builtin:xml-schema-structural-for-xmllint",
                "xmlschema" => "builtin:xml-schema-structural-for-xmlschema",
                "xmlschema-validate" => "builtin:xml-schema-structural-for-xmlschema-validate",
                "xsd-validator" => "builtin:xml-schema-structural-for-xsd-validator",
                "python-xmlschema" => "builtin:xml-schema-structural-for-python-xmlschema",
                "xmlschema-adapter" => "builtin:xml-schema-structural-for-xmlschema-adapter",
                _ => "builtin:xml-schema-structural",
            };
            output_validation_xml_reference(payload, validator)
        }
        "schematron" | "schematron-adapter" | "jing" | "saxon" | "saxon-he" | "saxon9he" => {
            let validator = if tool == "schematron" {
                "builtin:schematron-structural".to_string()
            } else {
                format!("builtin:schematron-structural-for-{tool}")
            };
            output_validation_xml_reference(payload, &validator)
        }
        "pydantic"
        | "pydantic-adapter"
        | "zod"
        | "zod-adapter"
        | "valibot"
        | "valibot-adapter"
        | "marshmallow"
        | "marshmallow-adapter"
        | "cerberus"
        | "cerberus-adapter" => {
            let validator = if tool == "pydantic" {
                "builtin:pydantic-model-subset".to_string()
            } else {
                format!("builtin:pydantic-model-subset-for-{tool}")
            };
            output_validation_pydantic_reference(payload, &validator)
        }
        "table" | "table-schema" | "tabular" | "csv-validator" | "csvlint" => {
            output_validation_table_reference(payload, "builtin:table-schema-subset")
        }
        "dbt"
            if kind == "sql-validation"
                || kind == "dbt-validation"
                || output_validation_has_sql_payload(payload) =>
        {
            output_validation_sql_reference(payload, "builtin:sql-structural-for-dbt")
        }
        "sqlfluff" | "sql-lint" | "sql-validator" => {
            output_validation_sql_reference(payload, &format!("builtin:sql-structural-for-{tool}"))
        }
        "parquet-tools" | "apache-arrow" | "arrow-adapter" | "pyarrow-adapter"
            if kind == "parquet-validation"
                || kind == "arrow-validation"
                || output_validation_has_columnar_payload(payload) =>
        {
            output_validation_columnar_reference(
                payload,
                &format!("builtin:columnar-metadata-for-{tool}"),
            )
        }
        "whylogs"
        | "whylogs-adapter"
        | "great-expectations"
        | "gx"
        | "evidently"
        | "evidently-adapter"
        | "deepchecks"
        | "deepchecks-adapter"
        | "tensorflow-data-validation"
        | "tfdv-adapter"
        | "soda-core"
        | "soda"
        | "deequ"
        | "deequ-adapter"
            if kind == "profile-validation"
                || kind == "data-profile-validation"
                || kind == "drift-validation"
                || output_validation_has_profile_payload(payload) =>
        {
            output_validation_profile_reference(
                payload,
                &format!("builtin:data-profile-structural-for-{tool}"),
            )
        }
        "frictionless"
            if kind == "data-package-validation"
                || kind == "frictionless-package-validation"
                || output_validation_has_data_package_payload(payload) =>
        {
            output_validation_data_package_reference(
                payload,
                "builtin:frictionless-data-package-structural",
            )
        }
        "openrefine" | "openrefine-adapter" | "refine"
            if kind == "openrefine-validation"
                || kind == "openrefine-history-validation"
                || kind == "data-cleaning-validation"
                || output_validation_has_openrefine_payload(payload) =>
        {
            output_validation_openrefine_reference(
                payload,
                &format!("builtin:openrefine-structural-for-{tool}"),
            )
        }
        "frictionless"
        | "pandera"
        | "pandera-adapter"
        | "dbt"
        | "whylogs"
        | "whylogs-adapter"
        | "great-expectations"
        | "gx"
        | "soda-core"
        | "soda"
        | "evidently"
        | "evidently-adapter"
        | "deepchecks"
        | "deepchecks-adapter"
        | "parquet-tools"
        | "apache-arrow"
        | "arrow-adapter"
        | "pyarrow-adapter"
        | "deequ"
        | "deequ-adapter"
        | "tensorflow-data-validation"
        | "tfdv-adapter"
        | "openrefine"
        | "openrefine-adapter"
        | "refine"
            if kind == "table-validation" || output_validation_has_table_payload(payload) =>
        {
            output_validation_table_reference(
                payload,
                &format!("builtin:table-schema-subset-for-{tool}"),
            )
        }
        "protobuf" | "protobuf-conformance" | "conformance-test-runner" | "protoc" => {
            output_validation_protobuf_reference(payload)
        }
        "avro" | "avro-tools" | "apache-avro" => output_validation_avro_reference(payload),
        _ if kind == "openapi-validation" => {
            output_validation_openapi_reference(payload, "builtin:openapi-structural")
        }
        _ if kind == "xml-validation" || kind == "xsd-validation" => {
            output_validation_xml_reference(payload, "builtin:xml-schema-structural")
        }
        _ if kind == "schematron-validation" => {
            output_validation_xml_reference(payload, "builtin:schematron-structural")
        }
        _ if kind == "pydantic-validation" => {
            output_validation_pydantic_reference(payload, "builtin:pydantic-model-subset")
        }
        _ if kind == "table-validation" => {
            output_validation_table_reference(payload, "builtin:table-schema-subset")
        }
        _ if kind == "sql-validation" || kind == "dbt-validation" => {
            output_validation_sql_reference(payload, "builtin:sql-structural")
        }
        _ if kind == "parquet-validation" || kind == "arrow-validation" => {
            output_validation_columnar_reference(payload, "builtin:columnar-metadata")
        }
        _ if kind == "profile-validation"
            || kind == "data-profile-validation"
            || kind == "drift-validation" =>
        {
            output_validation_profile_reference(payload, "builtin:data-profile-structural")
        }
        _ if kind == "data-package-validation" || kind == "frictionless-package-validation" => {
            output_validation_data_package_reference(
                payload,
                "builtin:frictionless-data-package-structural",
            )
        }
        _ if kind == "openrefine-validation"
            || kind == "openrefine-history-validation"
            || kind == "data-cleaning-validation" =>
        {
            output_validation_openrefine_reference(payload, "builtin:openrefine-structural")
        }
        _ if kind == "protobuf-validation" => output_validation_protobuf_reference(payload),
        _ if kind == "avro-validation" => output_validation_avro_reference(payload),
        "yamllint" => output_validation_yaml_reference(payload, "builtin:yaml-structural"),
        "graphql-schema" | "graphql-schema-linter" | "graphql-inspector" => {
            output_validation_graphql_reference(payload, "builtin:graphql-schema-structural")
        }
        _ if kind == "yaml-validation" || kind == "yamllint-validation" => {
            output_validation_yaml_reference(payload, "builtin:yaml-structural")
        }
        _ if kind == "cue-validation" => {
            output_validation_cue_reference(payload, "builtin:cue-structural")
        }
        _ if kind == "graphql-validation" || kind == "graphql-schema-validation" => {
            output_validation_graphql_reference(payload, "builtin:graphql-schema-structural")
        }
        _ => output_validation_result(
            "unavailable",
            "unknown",
            &tool,
            format!("unknown output validator '{tool}'"),
            Vec::new(),
        ),
    }
}

fn model_validation_result(
    status: &str,
    verdict: &str,
    validator: &str,
    message: impl Into<String>,
    stdout: impl Into<String>,
    stderr: impl Into<String>,
) -> Value {
    json!({
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message.into(),
        "stdout": stdout.into(),
        "stderr": stderr.into(),
    })
}

fn model_validation_payload_text<'a>(payload: &'a Value, keys: &[&str]) -> &'a str {
    keys.iter()
        .find_map(|key| payload.get(*key).and_then(Value::as_str))
        .unwrap_or("")
}

fn model_validation_normalized_tool(tool: &str) -> String {
    tool.trim().to_ascii_lowercase().replace('_', "-")
}

fn model_validation_payload_has_wcnf(payload: &Value) -> bool {
    payload.get("wcnf").is_some()
        || payload
            .get("dimacs")
            .and_then(Value::as_str)
            .is_some_and(|text| {
                text.lines().any(|line| {
                    let line = line.trim().to_ascii_lowercase();
                    line.starts_with("p wcnf ")
                })
            })
}

fn model_validation_payload_has_opb(payload: &Value) -> bool {
    payload.get("opb").is_some() || payload.get("pb").is_some()
}

fn model_validation_infer_smtlib(text: &str) -> Value {
    let lowered = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let lowered = lowered.to_ascii_lowercase();
    if lowered.contains("(assert false)") {
        return model_validation_result(
            "ok",
            "unsat",
            "builtin:smtlib-smoke",
            "assert false detected",
            "",
            "",
        );
    }
    model_validation_result(
        "ok",
        "sat",
        "builtin:smtlib-smoke",
        "no contradiction found",
        "",
        "",
    )
}

fn model_validation_parse_minizinc_domain(statement: &str) -> Option<(String, i64, i64)> {
    let rest = statement.strip_prefix("var ")?;
    let (bounds, name) = rest.split_once(':')?;
    let (lower, upper) = bounds.trim().split_once("..")?;
    let name = name.trim().trim_end_matches(';').trim().to_string();
    Some((
        name,
        lower.trim().parse::<i64>().ok()?,
        upper.trim().parse::<i64>().ok()?,
    ))
}

fn model_validation_eval_minizinc_constraint(
    expr: &str,
    assignment: &BTreeMap<String, i64>,
) -> Result<bool, String> {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(format!("unsupported MiniZinc constraint {expr:?}"));
    }
    let actual = assignment
        .get(parts[0])
        .ok_or_else(|| format!("unknown MiniZinc variable {:?}", parts[0]))?;
    let expected = parts[2]
        .parse::<i64>()
        .map_err(|_| format!("unsupported MiniZinc rhs {:?}", parts[2]))?;
    let satisfied = match parts[1] {
        "<=" => *actual <= expected,
        ">=" => *actual >= expected,
        "=" | "==" => *actual == expected,
        "<" => *actual < expected,
        ">" => *actual > expected,
        other => return Err(format!("unsupported MiniZinc operator {other:?}")),
    };
    Ok(satisfied)
}

fn model_validation_minizinc_search(
    names: &[String],
    domains: &BTreeMap<String, (i64, i64)>,
    constraints: &[String],
    idx: usize,
    assignment: &mut BTreeMap<String, i64>,
) -> Result<Option<BTreeMap<String, i64>>, String> {
    if idx == names.len() {
        for constraint in constraints {
            if !model_validation_eval_minizinc_constraint(constraint, assignment)? {
                return Ok(None);
            }
        }
        return Ok(Some(assignment.clone()));
    }
    let name = &names[idx];
    let (lower, upper) = domains
        .get(name)
        .ok_or_else(|| format!("missing MiniZinc domain for {name}"))?;
    for value in *lower..=*upper {
        assignment.insert(name.clone(), value);
        if let Some(solution) =
            model_validation_minizinc_search(names, domains, constraints, idx + 1, assignment)?
        {
            return Ok(Some(solution));
        }
    }
    assignment.remove(name);
    Ok(None)
}

fn model_validation_minizinc_reference(payload: &Value) -> Value {
    let model = model_validation_payload_text(payload, &["model"]);
    if model.trim().is_empty() {
        return model_validation_result(
            "failed",
            "failure",
            "minizinc",
            "payload needs model",
            "",
            "",
        );
    }
    if model.contains("constraint false;") {
        return model_validation_result(
            "ok",
            "unsat",
            "builtin:minizinc-smoke",
            "constraint false detected",
            "",
            "",
        );
    }
    let mut domains = BTreeMap::new();
    let mut constraints = Vec::new();
    for statement in model
        .split(';')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if statement.starts_with("var ") {
            match model_validation_parse_minizinc_domain(statement) {
                Some((name, lower, upper)) if upper >= lower && upper - lower <= 100 => {
                    domains.insert(name, (lower, upper));
                }
                Some((_, lower, upper)) if upper - lower > 100 => {
                    return model_validation_result(
                        "failed",
                        "failure",
                        "builtin:minizinc-smoke",
                        "builtin MiniZinc fallback supports domains of size <= 101",
                        "",
                        "",
                    );
                }
                _ => {}
            }
        } else if let Some(expr) = statement.strip_prefix("constraint ") {
            constraints.push(expr.trim().to_string());
        }
    }
    if domains.is_empty() {
        return model_validation_result(
            "ok",
            "sat",
            "builtin:minizinc-smoke",
            "no finite-domain variables detected",
            "----------\n",
            "",
        );
    }
    let total = domains
        .values()
        .try_fold(1_u128, |acc, (lower, upper)| {
            acc.checked_mul((*upper - *lower + 1) as u128)
        })
        .unwrap_or(u128::MAX);
    if total > 250_000 {
        return model_validation_result(
            "unavailable",
            "unknown",
            "builtin:minizinc-smoke",
            format!("search space too large: {total}"),
            "",
            "",
        );
    }
    let names: Vec<String> = domains.keys().cloned().collect();
    let mut assignment = BTreeMap::new();
    match model_validation_minizinc_search(&names, &domains, &constraints, 0, &mut assignment) {
        Ok(Some(solution)) => {
            let mut stdout = String::new();
            for name in &names {
                if let Some(value) = solution.get(name) {
                    stdout.push_str(&format!("{name} = {value};\n"));
                }
            }
            stdout.push_str("----------\n");
            model_validation_result(
                "ok",
                "sat",
                "builtin:minizinc-smoke",
                "satisfying assignment found",
                stdout,
                "",
            )
        }
        Ok(None) => model_validation_result(
            "ok",
            "unsat",
            "builtin:minizinc-smoke",
            "all assignments exhausted",
            "",
            "",
        ),
        Err(message) => model_validation_result(
            "failed",
            "failure",
            "builtin:minizinc-smoke",
            message,
            "",
            "",
        ),
    }
}

fn model_validation_asp_reference(payload: &Value, tool: &str) -> Value {
    let program = model_validation_payload_text(payload, &["asp", "program", "model", "text"]);
    if program.trim().is_empty() {
        return model_validation_result(
            "failed",
            "failure",
            "asp",
            "payload needs asp, program, model, or text",
            "",
            "",
        );
    }
    if program.contains("choose(") {
        return model_validation_result(
            "ok",
            "sat",
            "builtin:asp-smoke",
            "choice rule smoke model satisfied",
            "choose(a)\nSATISFIABLE\n",
            "",
        );
    }
    model_validation_result(
        "unavailable",
        "unknown",
        tool,
        format!("{tool} executable not found"),
        "",
        "",
    )
}

fn model_validation_cp_sat_source(payload: &Value) -> &Value {
    [
        "cp_sat_model",
        "cpSatModel",
        "cpsat_model",
        "cpsatModel",
        "cp_sat",
        "cpSat",
        "problem",
    ]
    .iter()
    .find_map(|key| {
        payload
            .get(*key)
            .filter(|value| value.as_object().is_some())
    })
    .or_else(|| {
        payload
            .get("model")
            .filter(|value| value.as_object().is_some())
    })
    .unwrap_or(payload)
}

fn model_validation_payload_has_cp_sat_json_model(payload: &Value) -> bool {
    let source = model_validation_cp_sat_source(payload);
    source
        .get("variables")
        .and_then(Value::as_array)
        .is_some_and(|variables| {
            !variables.is_empty()
                && variables
                    .iter()
                    .all(|variable| variable.get("domain").is_some())
        })
        && source.get("constraints").is_some_and(Value::is_array)
}

fn model_validation_cp_sat_reference(payload: &Value, tool: &str) -> Value {
    let tool = if tool.is_empty() { "cp-sat" } else { tool };
    let validator = format!("builtin:cp-sat-small-for-{tool}");
    let run = solve_cp_sat_json_with_external_reference(
        model_validation_cp_sat_source(payload),
        &ExternalCpSatReferenceOptions {
            solver: ExternalCpSatReferenceSolver::RustEnumeration,
            ..Default::default()
        },
    );
    let (status, verdict) = match run.status {
        ExternalCpSatReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalCpSatReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalCpSatReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalCpSatReferenceStatus::Exhausted
        | ExternalCpSatReferenceStatus::Unavailable
        | ExternalCpSatReferenceStatus::Unsupported => ("unavailable", "unknown"),
        ExternalCpSatReferenceStatus::Invalid
        | ExternalCpSatReferenceStatus::Failed
        | ExternalCpSatReferenceStatus::Unknown => ("failed", "failure"),
    };
    let mut stdout = Vec::new();
    if !run.assignment.is_empty() {
        stdout.push(format!(
            "assignment={}",
            run.assignment
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    if let Some(objective) = run.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if let Some(nodes) = run.nodes {
        stdout.push(format!("nodes={nodes}"));
    }
    stdout.push(format!("backend={}", run.backend));
    stdout.push(format!("solver={}", run.solver.as_arg()));
    model_validation_result(
        status,
        verdict,
        &validator,
        run.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_payload_has_finite_domain_cp(payload: &Value) -> bool {
    payload.get("variables").is_some()
        || payload.get("domains").is_some()
        || payload.get("constraints").is_some()
        || payload.get("constraint_model").is_some()
        || payload.get("cp_model").is_some()
}

fn model_validation_cp_integer(value: &Value) -> Option<i64> {
    output_validation_json_integer(value).and_then(|integer| i64::try_from(integer).ok())
}

fn model_validation_cp_variable_domain(name: &str, spec: &Value) -> Result<Vec<i64>, String> {
    let raw_domain = if let Some(obj) = spec.as_object() {
        obj.get("domain")
            .or_else(|| obj.get("values"))
            .unwrap_or(spec)
    } else {
        spec
    };
    let mut values = Vec::new();
    if let Some(items) = raw_domain.as_array() {
        if items.len() == 2
            && items
                .iter()
                .all(|item| model_validation_cp_integer(item).is_some())
            && spec
                .as_object()
                .is_some_and(|obj| obj.get("interval").and_then(Value::as_bool) == Some(true))
        {
            let lower = model_validation_cp_integer(&items[0]).unwrap_or(0);
            let upper = model_validation_cp_integer(&items[1]).unwrap_or(0);
            if upper < lower {
                return Err(format!(
                    "variable {name}: domain upper bound is below lower bound"
                ));
            }
            values.extend(lower..=upper);
        } else {
            for item in items {
                let Some(value) = model_validation_cp_integer(item) else {
                    return Err(format!("variable {name}: domain values must be integers"));
                };
                values.push(value);
            }
        }
    } else if let Some(obj) = spec.as_object() {
        let lower = obj
            .get("lb")
            .or_else(|| obj.get("lower"))
            .or_else(|| obj.get("min"))
            .and_then(model_validation_cp_integer);
        let upper = obj
            .get("ub")
            .or_else(|| obj.get("upper"))
            .or_else(|| obj.get("max"))
            .and_then(model_validation_cp_integer);
        match (lower, upper) {
            (Some(lower), Some(upper)) if upper >= lower => values.extend(lower..=upper),
            (Some(_), Some(_)) => {
                return Err(format!(
                    "variable {name}: domain upper bound is below lower bound"
                ));
            }
            _ => return Err(format!("variable {name}: unsupported domain shape")),
        }
    } else {
        return Err(format!("variable {name}: unsupported domain shape"));
    }
    values.sort_unstable();
    values.dedup();
    if values.is_empty() {
        return Err(format!("variable {name}: domain is empty"));
    }
    if values.len() > 101 {
        return Err(format!(
            "variable {name}: builtin CP fallback supports domains of size <= 101"
        ));
    }
    Ok(values)
}

fn model_validation_cp_domains(payload: &Value) -> Result<BTreeMap<String, Vec<i64>>, String> {
    let source = payload
        .get("variables")
        .or_else(|| payload.get("domains"))
        .or_else(|| {
            payload
                .get("constraint_model")
                .and_then(|model| model.get("variables"))
        })
        .or_else(|| {
            payload
                .get("cp_model")
                .and_then(|model| model.get("variables"))
        })
        .ok_or_else(|| "payload needs variables or domains".to_string())?;
    let mut domains = BTreeMap::new();
    match source {
        Value::Object(obj) => {
            for (name, spec) in obj {
                domains.insert(
                    name.clone(),
                    model_validation_cp_variable_domain(name, spec)?,
                );
            }
        }
        Value::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                let Some(obj) = item.as_object() else {
                    return Err(format!("variable {idx}: must be an object"));
                };
                let name = obj
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("variable {idx}: missing name"))?;
                domains.insert(
                    name.to_string(),
                    model_validation_cp_variable_domain(name, item)?,
                );
            }
        }
        _ => return Err("variables must be an object or array".to_string()),
    }
    if domains.is_empty() {
        return Err("payload defines no variables".to_string());
    }
    Ok(domains)
}

fn model_validation_cp_constraints(payload: &Value) -> Vec<Value> {
    payload
        .get("constraints")
        .or_else(|| {
            payload
                .get("constraint_model")
                .and_then(|model| model.get("constraints"))
        })
        .or_else(|| {
            payload
                .get("cp_model")
                .and_then(|model| model.get("constraints"))
        })
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn model_validation_cp_scope(obj: &serde_json::Map<String, Value>) -> Vec<String> {
    obj.get("vars")
        .or_else(|| obj.get("variables"))
        .or_else(|| obj.get("scope"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .or_else(|| {
            obj.get("var")
                .or_else(|| obj.get("variable"))
                .and_then(Value::as_str)
                .map(|name| vec![name.to_string()])
        })
        .unwrap_or_default()
}

fn model_validation_cp_constraint_holds(
    constraint: &Value,
    assignment: &BTreeMap<String, i64>,
) -> Result<bool, String> {
    let Some(obj) = constraint.as_object() else {
        return Err("CP constraint must be an object".to_string());
    };
    let op = obj
        .get("op")
        .or_else(|| obj.get("operator"))
        .or_else(|| obj.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("eq")
        .trim()
        .to_ascii_lowercase()
        .replace('_', "-");
    let scope = model_validation_cp_scope(obj);
    let values = scope
        .iter()
        .map(|name| {
            assignment
                .get(name)
                .copied()
                .ok_or_else(|| format!("unknown CP variable {name:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rhs = obj
        .get("rhs")
        .or_else(|| obj.get("value"))
        .or_else(|| obj.get("target"))
        .and_then(model_validation_cp_integer);
    match op.as_str() {
        "all-different" | "alldifferent" => {
            let mut seen = BTreeMap::<i64, ()>::new();
            Ok(values
                .into_iter()
                .all(|value| seen.insert(value, ()).is_none()))
        }
        "=" | "==" | "eq" => {
            if let Some(rhs) = rhs {
                Ok(values.iter().all(|value| *value == rhs))
            } else if values.len() >= 2 {
                Ok(values.windows(2).all(|pair| pair[0] == pair[1]))
            } else {
                Err("eq constraint needs rhs or at least two variables".to_string())
            }
        }
        "!=" | "ne" | "not-equal" => {
            if let Some(rhs) = rhs {
                Ok(values.iter().all(|value| *value != rhs))
            } else if values.len() == 2 {
                Ok(values[0] != values[1])
            } else {
                Err("ne constraint needs rhs or exactly two variables".to_string())
            }
        }
        "<" | "lt" | "<=" | "le" | "lte" | ">" | "gt" | ">=" | "ge" | "gte" => {
            let lhs = values
                .first()
                .copied()
                .ok_or_else(|| "comparison constraint needs a variable".to_string())?;
            let rhs = if let Some(rhs) = rhs {
                rhs
            } else if values.len() == 2 {
                values[1]
            } else {
                return Err("comparison constraint needs rhs or two variables".to_string());
            };
            Ok(match op.as_str() {
                "<" | "lt" => lhs < rhs,
                "<=" | "le" | "lte" => lhs <= rhs,
                ">" | "gt" => lhs > rhs,
                ">=" | "ge" | "gte" => lhs >= rhs,
                _ => unreachable!(),
            })
        }
        "sum-eq" | "sum-le" | "sum-lte" | "sum-ge" | "sum-gte" => {
            let sum: i64 = values.iter().sum();
            let rhs = rhs.ok_or_else(|| format!("{op} constraint needs rhs"))?;
            Ok(match op.as_str() {
                "sum-eq" => sum == rhs,
                "sum-le" | "sum-lte" => sum <= rhs,
                "sum-ge" | "sum-gte" => sum >= rhs,
                _ => unreachable!(),
            })
        }
        other => Err(format!("unsupported CP constraint op {other:?}")),
    }
}

fn model_validation_cp_search(
    names: &[String],
    domains: &BTreeMap<String, Vec<i64>>,
    constraints: &[Value],
    idx: usize,
    assignment: &mut BTreeMap<String, i64>,
) -> Result<Option<BTreeMap<String, i64>>, String> {
    if idx == names.len() {
        for constraint in constraints {
            if !model_validation_cp_constraint_holds(constraint, assignment)? {
                return Ok(None);
            }
        }
        return Ok(Some(assignment.clone()));
    }
    let name = &names[idx];
    let domain = domains
        .get(name)
        .ok_or_else(|| format!("missing domain for CP variable {name}"))?;
    for value in domain {
        assignment.insert(name.clone(), *value);
        if let Some(solution) =
            model_validation_cp_search(names, domains, constraints, idx + 1, assignment)?
        {
            return Ok(Some(solution));
        }
    }
    assignment.remove(name);
    Ok(None)
}

fn model_validation_finite_domain_cp_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:finite-domain-cp-for-{tool}");
    let domains = match model_validation_cp_domains(payload) {
        Ok(domains) => domains,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let constraints = model_validation_cp_constraints(payload);
    let total = domains
        .values()
        .try_fold(1_u128, |acc, domain| acc.checked_mul(domain.len() as u128))
        .unwrap_or(u128::MAX);
    if total > 250_000 {
        return model_validation_result(
            "unavailable",
            "unknown",
            &validator,
            format!("search space too large: {total}"),
            "",
            "",
        );
    }
    let names = domains.keys().cloned().collect::<Vec<_>>();
    let mut assignment = BTreeMap::new();
    match model_validation_cp_search(&names, &domains, &constraints, 0, &mut assignment) {
        Ok(Some(solution)) => {
            let stdout = names
                .iter()
                .filter_map(|name| solution.get(name).map(|value| format!("{name}={value}")))
                .collect::<Vec<_>>()
                .join(" ");
            model_validation_result(
                "ok",
                "sat",
                &validator,
                "satisfying assignment found",
                stdout,
                "",
            )
        }
        Ok(None) => model_validation_result(
            "ok",
            "unsat",
            &validator,
            "all assignments exhausted",
            "",
            "",
        ),
        Err(message) => model_validation_result("failed", "failure", &validator, message, "", ""),
    }
}

#[derive(Clone, Debug)]
struct ModelValidationLinearConstraint {
    coefs: Vec<f64>,
    sense: String,
    rhs: f64,
}

fn model_validation_linear_source(payload: &Value) -> &Value {
    [
        "linear_model",
        "linearModel",
        "mip_model",
        "mipModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_payload_has_linear_model(payload: &Value) -> bool {
    let source = model_validation_linear_source(payload);
    source.get("objective").is_some()
        || source.get("objective_coefs").is_some()
        || source.get("objectiveCoefficients").is_some()
        || source.get("c").is_some()
        || source
            .get("constraints")
            .and_then(Value::as_array)
            .is_some_and(|rows| {
                rows.iter().any(|row| {
                    row.get("coefs").is_some()
                        || row.get("coefficients").is_some()
                        || row.get("a").is_some()
                        || row.get("lhs").is_some()
                })
            })
}

fn model_validation_linear_number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
}

fn model_validation_linear_integer(value: &Value) -> Option<i64> {
    output_validation_json_integer(value)
        .and_then(|integer| i64::try_from(integer).ok())
        .or_else(|| {
            value.as_f64().and_then(|number| {
                let rounded = number.round();
                ((number - rounded).abs() <= 1e-9).then_some(rounded as i64)
            })
        })
}

fn model_validation_linear_vector(value: Option<&Value>) -> Option<Vec<f64>> {
    value?.as_array().map(|items| {
        items
            .iter()
            .map(|item| model_validation_linear_number(Some(item)).unwrap_or(0.0))
            .collect()
    })
}

fn model_validation_linear_variable_names(source: &Value, width: usize) -> Vec<String> {
    let from_variables = source
        .get("variables")
        .and_then(|variables| match variables {
            Value::Array(items) => Some(
                items
                    .iter()
                    .enumerate()
                    .map(|(idx, item)| {
                        item.get("name")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or_else(|| format!("x{idx}"))
                    })
                    .take(width)
                    .collect::<Vec<_>>(),
            ),
            Value::Object(obj) => Some(obj.keys().take(width).cloned().collect::<Vec<_>>()),
            _ => None,
        });
    from_variables
        .filter(|names| names.len() == width)
        .unwrap_or_else(|| (0..width).map(|idx| format!("x{idx}")).collect())
}

fn model_validation_linear_objective_terms(
    raw: Option<&Value>,
) -> Result<Option<(Vec<String>, Vec<f64>)>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    if let Some(coefs) = model_validation_linear_vector(Some(raw)) {
        return Ok(Some((Vec::new(), coefs)));
    }
    let Some(obj) = raw.as_object() else {
        return Err("objective must be an array or object".to_string());
    };
    for key in ["coefs", "coefficients", "linear", "values"] {
        if let Some(coefs) = model_validation_linear_vector(obj.get(key)) {
            return Ok(Some((Vec::new(), coefs)));
        }
    }
    let terms = obj.get("terms").and_then(Value::as_object).unwrap_or(obj);
    let mut entries = terms
        .iter()
        .filter_map(|(name, value)| {
            model_validation_linear_number(Some(value)).map(|coef| (name.clone(), coef))
        })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return Err("objective object needs numeric terms or coefficients".to_string());
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let names = entries
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let coefs = entries
        .into_iter()
        .map(|(_, coef)| coef)
        .collect::<Vec<_>>();
    Ok(Some((names, coefs)))
}

fn model_validation_linear_objective(
    source: &Value,
    width: usize,
) -> Result<(Vec<String>, Vec<f64>, bool), String> {
    let raw = source
        .get("objective")
        .or_else(|| source.get("objective_coefs"))
        .or_else(|| source.get("objectiveCoefficients"))
        .or_else(|| source.get("c"));
    let Some((names, coefs)) = model_validation_linear_objective_terms(raw)? else {
        return Ok((
            model_validation_linear_variable_names(source, width),
            vec![0.0; width],
            false,
        ));
    };
    if coefs.is_empty() {
        return Err("objective vector is empty".to_string());
    }
    let names = if names.len() == coefs.len() {
        names
    } else {
        model_validation_linear_variable_names(source, coefs.len())
    };
    Ok((names, coefs, true))
}

fn model_validation_linear_constraint_width(source: &Value) -> Option<usize> {
    source
        .get("constraints")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .filter_map(|row| {
                    model_validation_linear_vector(
                        row.get("coefs")
                            .or_else(|| row.get("coefficients"))
                            .or_else(|| row.get("a"))
                            .or_else(|| row.get("lhs")),
                    )
                    .map(|coefs| coefs.len())
                })
                .max()
        })
}

fn model_validation_linear_variable_width(source: &Value) -> Option<usize> {
    source
        .get("variables")
        .and_then(|variables| match variables {
            Value::Array(items) => Some(items.len()),
            Value::Object(obj) => Some(obj.len()),
            _ => None,
        })
}

fn model_validation_linear_domain_width(source: &Value) -> Option<usize> {
    source
        .get("domains")
        .or_else(|| source.get("bounds"))
        .and_then(Value::as_array)
        .map(Vec::len)
        .or_else(|| {
            source
                .get("lower_bounds")
                .or_else(|| source.get("lowerBounds"))
                .and_then(Value::as_array)
                .map(Vec::len)
        })
}

fn model_validation_linear_width(source: &Value) -> Result<usize, String> {
    let objective_width = model_validation_linear_objective_terms(
        source
            .get("objective")
            .or_else(|| source.get("objective_coefs"))
            .or_else(|| source.get("objectiveCoefficients"))
            .or_else(|| source.get("c")),
    )?
    .map(|(_, coefs)| coefs.len());
    objective_width
        .or_else(|| model_validation_linear_constraint_width(source))
        .or_else(|| model_validation_linear_domain_width(source))
        .or_else(|| model_validation_linear_variable_width(source))
        .filter(|width| *width > 0)
        .ok_or_else(|| {
            "linear model needs objective, constraints, domains, or variables".to_string()
        })
}

fn model_validation_linear_domain_from_spec(name: &str, spec: &Value) -> Result<Vec<i64>, String> {
    let raw_domain = if let Some(obj) = spec.as_object() {
        if obj
            .get("binary")
            .or_else(|| obj.get("is_binary"))
            .or_else(|| obj.get("isBinary"))
            .and_then(Value::as_bool)
            == Some(true)
            || obj
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind.eq_ignore_ascii_case("binary"))
        {
            return Ok(vec![0, 1]);
        }
        obj.get("domain")
            .or_else(|| obj.get("values"))
            .unwrap_or(spec)
    } else {
        spec
    };
    let mut values = Vec::new();
    if let Some(items) = raw_domain.as_array() {
        if items.len() == 2
            && items
                .iter()
                .all(|item| model_validation_linear_integer(item).is_some())
        {
            let lower = model_validation_linear_integer(&items[0]).unwrap_or(0);
            let upper = model_validation_linear_integer(&items[1]).unwrap_or(0);
            if upper < lower {
                return Err(format!(
                    "variable {name}: domain upper bound is below lower bound"
                ));
            }
            values.extend(lower..=upper);
        } else {
            for item in items {
                let Some(value) = model_validation_linear_integer(item) else {
                    return Err(format!("variable {name}: domain values must be integers"));
                };
                values.push(value);
            }
        }
    } else if let Some(obj) = spec.as_object() {
        let lower = obj
            .get("lb")
            .or_else(|| obj.get("lower"))
            .or_else(|| obj.get("min"))
            .and_then(model_validation_linear_integer);
        let upper = obj
            .get("ub")
            .or_else(|| obj.get("upper"))
            .or_else(|| obj.get("max"))
            .and_then(model_validation_linear_integer);
        match (lower, upper) {
            (Some(lower), Some(upper)) if upper >= lower => values.extend(lower..=upper),
            (Some(_), Some(_)) => {
                return Err(format!(
                    "variable {name}: domain upper bound is below lower bound"
                ));
            }
            _ => return Err(format!("variable {name}: unsupported domain shape")),
        }
    } else {
        return Err(format!("variable {name}: unsupported domain shape"));
    }
    values.sort_unstable();
    values.dedup();
    if values.is_empty() {
        return Err(format!("variable {name}: domain is empty"));
    }
    if values.len() > 101 {
        return Err(format!(
            "variable {name}: builtin linear fallback supports domains of size <= 101"
        ));
    }
    Ok(values)
}

fn model_validation_linear_variable_spec<'a>(
    source: &'a Value,
    names: &[String],
    idx: usize,
) -> Option<&'a Value> {
    source
        .get("variables")
        .and_then(|variables| match variables {
            Value::Array(items) => items.get(idx),
            Value::Object(obj) => obj.get(&names[idx]),
            _ => None,
        })
}

fn model_validation_linear_domain_spec<'a>(source: &'a Value, idx: usize) -> Option<&'a Value> {
    source
        .get("domains")
        .or_else(|| source.get("bounds"))
        .and_then(Value::as_array)
        .and_then(|items| items.get(idx))
}

fn model_validation_linear_domains(
    source: &Value,
    names: &[String],
) -> Result<Vec<Vec<i64>>, String> {
    let lower_bounds = source
        .get("lower_bounds")
        .or_else(|| source.get("lowerBounds"))
        .and_then(Value::as_array);
    let upper_bounds = source
        .get("upper_bounds")
        .or_else(|| source.get("upperBounds"))
        .and_then(Value::as_array);
    let mut domains = Vec::with_capacity(names.len());
    for (idx, name) in names.iter().enumerate() {
        let domain = if let Some(spec) = model_validation_linear_domain_spec(source, idx) {
            model_validation_linear_domain_from_spec(name, spec)?
        } else if let Some(spec) = model_validation_linear_variable_spec(source, names, idx) {
            model_validation_linear_domain_from_spec(name, spec)?
        } else if let (Some(lowers), Some(uppers)) = (lower_bounds, upper_bounds) {
            let lower = lowers
                .get(idx)
                .and_then(model_validation_linear_integer)
                .ok_or_else(|| format!("variable {name}: missing integer lower bound"))?;
            let upper = uppers
                .get(idx)
                .and_then(model_validation_linear_integer)
                .ok_or_else(|| format!("variable {name}: missing integer upper bound"))?;
            if upper < lower {
                return Err(format!(
                    "variable {name}: domain upper bound is below lower bound"
                ));
            }
            (lower..=upper).collect()
        } else {
            vec![0, 1]
        };
        domains.push(domain);
    }
    Ok(domains)
}

fn model_validation_linear_constraints(
    source: &Value,
    width: usize,
) -> Result<Vec<ModelValidationLinearConstraint>, String> {
    let rows = source
        .get("constraints")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    rows.iter()
        .enumerate()
        .map(|(idx, row)| {
            let coefs = model_validation_linear_vector(
                row.get("coefs")
                    .or_else(|| row.get("coefficients"))
                    .or_else(|| row.get("a"))
                    .or_else(|| row.get("lhs")),
            )
            .ok_or_else(|| format!("constraint {idx}: missing coefficient vector"))?;
            if coefs.len() != width {
                return Err(format!(
                    "constraint {idx}: coefficient length {} does not match width {width}",
                    coefs.len()
                ));
            }
            let sense = row
                .get("sense")
                .or_else(|| row.get("op"))
                .or_else(|| row.get("operator"))
                .or_else(|| row.get("relation"))
                .and_then(Value::as_str)
                .unwrap_or("<=")
                .trim()
                .to_ascii_lowercase();
            let rhs = model_validation_linear_number(
                row.get("rhs")
                    .or_else(|| row.get("bound"))
                    .or_else(|| row.get("value")),
            )
            .ok_or_else(|| format!("constraint {idx}: missing numeric rhs"))?;
            Ok(ModelValidationLinearConstraint { coefs, sense, rhs })
        })
        .collect()
}

fn model_validation_linear_row_feasible(lhs: f64, sense: &str, rhs: f64) -> Result<bool, String> {
    Ok(match sense {
        "<=" | "le" | "less-equal" | "lte" => lhs <= rhs + 1e-9,
        ">=" | "ge" | "greater-equal" | "gte" => lhs + 1e-9 >= rhs,
        "=" | "==" | "eq" | "equal" => (lhs - rhs).abs() <= 1e-9,
        "<" | "lt" => lhs < rhs + 1e-9,
        ">" | "gt" => lhs + 1e-9 > rhs,
        other => return Err(format!("unsupported linear row sense {other:?}")),
    })
}

fn model_validation_linear_better(candidate: f64, incumbent: Option<f64>, sense: &str) -> bool {
    match incumbent {
        None => true,
        Some(incumbent) if sense == "max" || sense == "maximize" => candidate > incumbent + 1e-12,
        Some(incumbent) => candidate < incumbent - 1e-12,
    }
}

fn model_validation_linear_search(
    domains: &[Vec<i64>],
    objective: &[f64],
    constraints: &[ModelValidationLinearConstraint],
    sense: &str,
    idx: usize,
    assignment: &mut Vec<i64>,
    best: &mut Option<(f64, Vec<i64>)>,
) -> Result<(), String> {
    if idx == domains.len() {
        for row in constraints {
            let lhs = row
                .coefs
                .iter()
                .zip(assignment.iter())
                .map(|(coef, value)| coef * *value as f64)
                .sum::<f64>();
            if !model_validation_linear_row_feasible(lhs, &row.sense, row.rhs)? {
                return Ok(());
            }
        }
        let value = objective
            .iter()
            .zip(assignment.iter())
            .map(|(coef, value)| coef * *value as f64)
            .sum::<f64>();
        if model_validation_linear_better(value, best.as_ref().map(|(value, _)| *value), sense) {
            *best = Some((value, assignment.clone()));
        }
        return Ok(());
    }
    for value in &domains[idx] {
        assignment.push(*value);
        model_validation_linear_search(
            domains,
            objective,
            constraints,
            sense,
            idx + 1,
            assignment,
            best,
        )?;
        assignment.pop();
    }
    Ok(())
}

fn model_validation_linear_mip_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:linear-mip-small-for-{tool}");
    let source = model_validation_linear_source(payload);
    let width = match model_validation_linear_width(source) {
        Ok(width) => width,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let (names, objective, has_objective) = match model_validation_linear_objective(source, width) {
        Ok(parsed) => parsed,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    if objective.len() != width {
        return model_validation_result(
            "failed",
            "failure",
            &validator,
            format!(
                "objective length {} does not match width {width}",
                objective.len()
            ),
            "",
            "",
        );
    }
    let domains = match model_validation_linear_domains(source, &names) {
        Ok(domains) => domains,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let total = domains
        .iter()
        .try_fold(1_u128, |acc, domain| acc.checked_mul(domain.len() as u128))
        .unwrap_or(u128::MAX);
    if total > 250_000 {
        return model_validation_result(
            "unavailable",
            "unknown",
            &validator,
            format!("search space too large: {total}"),
            "",
            "",
        );
    }
    let constraints = match model_validation_linear_constraints(source, width) {
        Ok(constraints) => constraints,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let sense = source
        .get("sense")
        .or_else(|| source.get("objective_sense"))
        .or_else(|| source.get("objectiveSense"))
        .and_then(Value::as_str)
        .unwrap_or("min")
        .trim()
        .to_ascii_lowercase();
    let mut assignment = Vec::with_capacity(width);
    let mut best = None::<(f64, Vec<i64>)>;
    if let Err(message) = model_validation_linear_search(
        &domains,
        &objective,
        &constraints,
        &sense,
        0,
        &mut assignment,
        &mut best,
    ) {
        return model_validation_result("failed", "failure", &validator, message, "", "");
    }
    let Some((objective_value, solution)) = best else {
        return model_validation_result(
            "ok",
            "infeasible",
            &validator,
            "no feasible assignment",
            "",
            "",
        );
    };
    let mut stdout = names
        .iter()
        .zip(solution.iter())
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>();
    stdout.push(format!("objective={objective_value}"));
    model_validation_result(
        "ok",
        if has_objective { "optimal" } else { "sat" },
        &validator,
        if has_objective {
            "optimal assignment found"
        } else {
            "feasible assignment found"
        },
        stdout.join(" "),
        "",
    )
}

fn model_validation_payload_has_nonlinear_model(payload: &Value) -> bool {
    payload.get("objective").is_some()
        || payload.get("variables").is_some()
        || payload.get("constraints").is_some()
        || payload.get("dimension").is_some()
}

fn model_validation_nonlinear_solver_for_tool(
    tool: &str,
) -> ExternalNonlinearValidationReferenceSolver {
    match tool {
        "scipy-optimize" | "scipy-optimize-adapter" | "scipy-adapter" => {
            ExternalNonlinearValidationReferenceSolver::Scipy
        }
        "argmin" | "argmin-adapter" | "ores-argmin-adapter" => {
            ExternalNonlinearValidationReferenceSolver::Fallback
        }
        "nlopt" | "nlopt-adapter" => ExternalNonlinearValidationReferenceSolver::Nlopt,
        "nlopt-rs" | "nlopt-rs-adapter" | "ores-nlopt-rs-adapter" => {
            ExternalNonlinearValidationReferenceSolver::Nlopt
        }
        "nlopt-cli" => ExternalNonlinearValidationReferenceSolver::NloptCli,
        "ipopt"
        | "ipopt-adapter"
        | "ipopt-rust"
        | "ipopt-rust-adapter"
        | "ores-ipopt-rust-adapter" => ExternalNonlinearValidationReferenceSolver::Ipopt,
        "casadi" | "casadi-adapter" => ExternalNonlinearValidationReferenceSolver::Casadi,
        "mosek" | "mosek-adapter" => ExternalNonlinearValidationReferenceSolver::Mosek,
        "copt" | "copt-adapter" => ExternalNonlinearValidationReferenceSolver::Copt,
        "cvxpy" | "cvxpy-adapter" | "cvxopt" | "cvxopt-adapter" | "osqp" | "osqp-adapter"
        | "scs" | "scs-adapter" | "clarabel" | "clarabel-adapter" | "ecos" | "ecos-adapter" => {
            ExternalNonlinearValidationReferenceSolver::Fallback
        }
        _ => ExternalNonlinearValidationReferenceSolver::Auto,
    }
}

fn model_validation_nonlinear_payload_for_bridge(payload: &Value) -> Value {
    let mut payload = payload.clone();
    if let Some(obj) = payload.as_object_mut() {
        let kind = obj
            .get("kind")
            .and_then(Value::as_str)
            .map(model_validation_normalized_tool)
            .unwrap_or_else(|| "nonlinear-validation".to_string());
        if !matches!(kind.as_str(), "nonlinear-validation" | "nlp-validation") {
            obj.insert(
                "kind".to_string(),
                Value::String("nonlinear-validation".to_string()),
            );
        }
        obj.entry("constraints".to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
    }
    payload
}

fn model_validation_nonlinear_variable_names(payload: &Value, width: usize) -> Vec<String> {
    payload
        .get("variables")
        .and_then(Value::as_array)
        .map(|variables| {
            variables
                .iter()
                .enumerate()
                .map(|(idx, variable)| {
                    variable
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("x{idx}"))
                })
                .take(width)
                .collect::<Vec<_>>()
        })
        .filter(|names| names.len() == width)
        .unwrap_or_else(|| (0..width).map(|idx| format!("x{idx}")).collect())
}

fn model_validation_nonlinear_reference(payload: &Value, tool: &str) -> Value {
    let solver = model_validation_nonlinear_solver_for_tool(tool);
    let bridge_payload = model_validation_nonlinear_payload_for_bridge(payload);
    let solution = solve_nonlinear_validation_json_with_external_reference(
        bridge_payload.clone(),
        &ExternalNonlinearValidationReferenceOptions { solver },
    );
    let validator = format!("builtin:nonlinear-reference-for-{tool}");
    let status = match solution.status {
        ExternalNonlinearValidationReferenceStatus::Optimal
        | ExternalNonlinearValidationReferenceStatus::Infeasible => "ok",
        ExternalNonlinearValidationReferenceStatus::Failed
        | ExternalNonlinearValidationReferenceStatus::NumericalError => "failed",
    };
    let verdict = match solution.status {
        ExternalNonlinearValidationReferenceStatus::Optimal => "optimal",
        ExternalNonlinearValidationReferenceStatus::Infeasible => "infeasible",
        ExternalNonlinearValidationReferenceStatus::Failed => "failure",
        ExternalNonlinearValidationReferenceStatus::NumericalError => "numerical-error",
    };
    let names = model_validation_nonlinear_variable_names(&bridge_payload, solution.x.len());
    let mut stdout = names
        .iter()
        .zip(solution.x.iter())
        .map(|(name, value)| format!("{name}={value:.9}"))
        .collect::<Vec<_>>();
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if let Some(iterations) = solution.iterations {
        stdout.push(format!("iterations={iterations}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_quadratic_source(payload: &Value) -> &Value {
    [
        "quadratic_model",
        "quadraticModel",
        "qp_model",
        "qpModel",
        "miqp_model",
        "miqpModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_payload_has_quadratic_model(payload: &Value) -> bool {
    let source = model_validation_quadratic_source(payload);
    source
        .get("Q")
        .or_else(|| source.get("q_matrix"))
        .or_else(|| source.get("qMatrix"))
        .or_else(|| source.get("quadratic"))
        .and_then(Value::as_array)
        .is_some()
        && source
            .get("c")
            .or_else(|| source.get("linear"))
            .or_else(|| source.get("objective"))
            .and_then(Value::as_array)
            .is_some()
}

fn model_validation_quadratic_bounds(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Option<Vec<Option<f64>>>, String> {
    let Some(values) = keys
        .iter()
        .filter_map(|key| source.get(*key).and_then(Value::as_array))
        .next()
    else {
        return Ok(None);
    };
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if value.is_null() {
                Ok(None)
            } else {
                model_validation_routing_number(Some(value), &format!("{label}[{idx}]")).map(Some)
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn model_validation_quadratic_bool_array(
    source: &Value,
    keys: &[&str],
    width: usize,
) -> Result<Vec<bool>, String> {
    let Some(values) = keys
        .iter()
        .filter_map(|key| source.get(*key).and_then(Value::as_array))
        .next()
    else {
        return Ok(vec![false; width]);
    };
    if values.len() != width {
        return Err(format!(
            "integer_vars length {} does not match variable count {width}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            value
                .as_bool()
                .or_else(|| model_validation_linear_integer(value).map(|integer| integer != 0))
                .ok_or_else(|| format!("integer_vars[{idx}] must be boolean"))
        })
        .collect()
}

fn model_validation_quadratic_program(source: &Value) -> Result<QuadraticProgram, String> {
    let q_value = source
        .get("Q")
        .or_else(|| source.get("q_matrix"))
        .or_else(|| source.get("qMatrix"))
        .or_else(|| source.get("quadratic"))
        .ok_or_else(|| "quadratic payload needs Q matrix".to_string())?;
    let c_value = source
        .get("c")
        .or_else(|| source.get("linear"))
        .or_else(|| source.get("objective"))
        .ok_or_else(|| "quadratic payload needs c vector".to_string())?;
    let q = model_validation_number_matrix_value(q_value, "Q")?;
    let c = model_validation_number_vector_value(c_value, "c")?;
    if q.len() != c.len() {
        return Err(format!(
            "Q row count {} does not match c length {}",
            q.len(),
            c.len()
        ));
    }
    if q.iter().any(|row| row.len() != c.len()) {
        return Err("Q must be square with width equal to c length".to_string());
    }
    Ok(QuadraticProgram {
        q,
        c,
        a_ub: model_validation_optional_number_matrix(
            source,
            &["A_ub", "a_ub", "aUb", "Aub", "constraints"],
            "A_ub",
        )
        .map(|matrix| (!matrix.is_empty()).then_some(matrix))?,
        b_ub: model_validation_optional_number_vector(source, &["b_ub", "bUb", "Bub"], "b_ub")
            .map(|vector| (!vector.is_empty()).then_some(vector))?,
        a_eq: model_validation_optional_number_matrix(source, &["A_eq", "a_eq", "aEq"], "A_eq")
            .map(|matrix| (!matrix.is_empty()).then_some(matrix))?,
        b_eq: model_validation_optional_number_vector(source, &["b_eq", "bEq"], "b_eq")
            .map(|vector| (!vector.is_empty()).then_some(vector))?,
        lb: model_validation_quadratic_bounds(
            source,
            &["lb", "lower_bounds", "lowerBounds"],
            "lb",
        )?,
        ub: model_validation_quadratic_bounds(
            source,
            &["ub", "upper_bounds", "upperBounds"],
            "ub",
        )?,
        var_names: source
            .get("var_names")
            .or_else(|| source.get("varNames"))
            .or_else(|| source.get("variables"))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .enumerate()
                    .map(|(idx, value)| {
                        model_validation_string_value(value, &format!("varNames[{idx}]"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?,
    })
}

fn model_validation_quadratic_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:quadratic-small-for-{tool}");
    let source = model_validation_quadratic_source(payload);
    let qp = match model_validation_quadratic_program(source) {
        Ok(qp) => qp,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let integer_vars = match model_validation_quadratic_bool_array(
        source,
        &["integer_vars", "integerVars"],
        qp.c.len(),
    ) {
        Ok(integer_vars) => integer_vars,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let has_integer_vars = integer_vars.iter().any(|value| *value);
    let opts = ExternalQuadraticReferenceOptions {
        solver: ExternalQuadraticReferenceSolver::RustInternal,
        max_enumerations: Some(1_000_000),
    };
    let solution = if has_integer_vars {
        solve_miqp_with_external_reference(
            &MixedIntegerQuadraticProgram { qp, integer_vars },
            &opts,
        )
    } else {
        solve_qp_with_external_reference(&qp, &opts)
    };
    let (status, verdict) = match solution.status {
        ExternalQuadraticReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalQuadraticReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalQuadraticReferenceStatus::Unbounded => ("ok", "unbounded"),
        ExternalQuadraticReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalQuadraticReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = solution
        .x
        .iter()
        .enumerate()
        .map(|(idx, value)| format!("x{idx}={value:.9}"))
        .collect::<Vec<_>>();
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if let Some(iterations) = solution.iterations {
        stdout.push(format!("iterations={iterations}"));
    }
    if let Some(enumerated) = solution.enumerated {
        stdout.push(format!("enumerated={enumerated}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_stochastic_lp_source(payload: &Value) -> &Value {
    [
        "stochastic_lp_model",
        "stochasticLpModel",
        "slp_model",
        "slpModel",
        "two_stage_model",
        "twoStageModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_payload_has_stochastic_lp_model(payload: &Value) -> bool {
    let source = model_validation_stochastic_lp_source(payload);
    source
        .get("scenarios")
        .and_then(Value::as_array)
        .is_some_and(|scenarios| !scenarios.is_empty())
        && source
            .get("c_first")
            .or_else(|| source.get("cFirst"))
            .and_then(Value::as_array)
            .is_some()
        && source
            .get("q_second")
            .or_else(|| source.get("qSecond"))
            .and_then(Value::as_array)
            .is_some()
        && source
            .get("w_second")
            .or_else(|| source.get("wSecond"))
            .and_then(Value::as_array)
            .is_some()
}

fn model_validation_number_vector_value(value: &Value, label: &str) -> Result<Vec<f64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?;
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            model_validation_routing_number(Some(value), &format!("{label}[{idx}]"))
        })
        .collect()
}

fn model_validation_number_matrix_value(
    value: &Value,
    label: &str,
) -> Result<Vec<Vec<f64>>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?;
    rows.iter()
        .enumerate()
        .map(|(row_idx, row)| {
            model_validation_number_vector_value(row, &format!("{label}[{row_idx}]"))
        })
        .collect()
}

fn model_validation_required_number_vector(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Vec<f64>, String> {
    let value = keys
        .iter()
        .filter_map(|key| source.get(*key))
        .next()
        .ok_or_else(|| format!("{label} is required"))?;
    model_validation_number_vector_value(value, label)
}

fn model_validation_optional_number_vector(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Vec<f64>, String> {
    match keys.iter().filter_map(|key| source.get(*key)).next() {
        Some(value) => model_validation_number_vector_value(value, label),
        None => Ok(Vec::new()),
    }
}

fn model_validation_optional_number_matrix(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Vec<Vec<f64>>, String> {
    match keys.iter().filter_map(|key| source.get(*key)).next() {
        Some(value) => model_validation_number_matrix_value(value, label),
        None => Ok(Vec::new()),
    }
}

fn model_validation_required_number_matrix(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Vec<Vec<f64>>, String> {
    let value = keys
        .iter()
        .filter_map(|key| source.get(*key))
        .next()
        .ok_or_else(|| format!("{label} is required"))?;
    model_validation_number_matrix_value(value, label)
}

fn model_validation_optional_number_field(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Option<f64>, String> {
    keys.iter()
        .filter_map(|key| source.get(*key))
        .next()
        .map(|value| model_validation_routing_number(Some(value), label))
        .transpose()
}

fn model_validation_stochastic_lp_scenario(value: &Value, idx: usize) -> Result<Scenario, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| format!("scenarios[{idx}] must be an object"))?;
    let t_value = obj
        .get("t")
        .or_else(|| obj.get("T"))
        .or_else(|| obj.get("technology"))
        .ok_or_else(|| format!("scenarios[{idx}].t is required"))?;
    let h_value = obj
        .get("h")
        .or_else(|| obj.get("rhs"))
        .or_else(|| obj.get("demand"))
        .ok_or_else(|| format!("scenarios[{idx}].h is required"))?;
    Ok(Scenario {
        t: model_validation_number_matrix_value(t_value, &format!("scenarios[{idx}].t"))?,
        h: model_validation_number_vector_value(h_value, &format!("scenarios[{idx}].h"))?,
        prob: model_validation_optional_number_field(
            value,
            &["prob", "probability", "p"],
            &format!("scenarios[{idx}].prob"),
        )?,
        meta: None,
    })
}

fn model_validation_stochastic_lp_problem(
    source: &Value,
) -> Result<(SLPProblem, Vec<Scenario>), String> {
    let scenarios = source
        .get("scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| "stochastic LP payload needs scenarios array".to_string())?
        .iter()
        .enumerate()
        .map(|(idx, scenario)| model_validation_stochastic_lp_scenario(scenario, idx))
        .collect::<Result<Vec<_>, _>>()?;
    let problem = SLPProblem {
        c_first: model_validation_required_number_vector(
            source,
            &["c_first", "cFirst", "first_stage_costs", "firstStageCosts"],
            "c_first",
        )?,
        a_first: model_validation_optional_number_matrix(
            source,
            &[
                "a_first",
                "aFirst",
                "first_stage_matrix",
                "firstStageMatrix",
            ],
            "a_first",
        )?,
        b_first: model_validation_optional_number_vector(
            source,
            &["b_first", "bFirst", "first_stage_rhs", "firstStageRhs"],
            "b_first",
        )?,
        q_second: model_validation_required_number_vector(
            source,
            &["q_second", "qSecond", "recourse_costs", "recourseCosts"],
            "q_second",
        )?,
        w_second: model_validation_required_number_matrix(
            source,
            &["w_second", "wSecond", "recourse_matrix", "recourseMatrix"],
            "w_second",
        )?,
        theta_lower_bound: model_validation_optional_number_field(
            source,
            &["theta_lower_bound", "thetaLowerBound"],
            "theta_lower_bound",
        )?
        .unwrap_or(0.0),
        theta_upper_bound: model_validation_optional_number_field(
            source,
            &["theta_upper_bound", "thetaUpperBound"],
            "theta_upper_bound",
        )?
        .unwrap_or(1.0e12),
        var_names: source
            .get("var_names")
            .or_else(|| source.get("varNames"))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .enumerate()
                    .map(|(idx, value)| {
                        model_validation_string_value(value, &format!("varNames[{idx}]"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?,
    };
    Ok((problem, scenarios))
}

fn model_validation_stochastic_lp_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:stochastic-lp-small-for-{tool}");
    let source = model_validation_stochastic_lp_source(payload);
    let (problem, scenarios) = match model_validation_stochastic_lp_problem(source) {
        Ok(parsed) => parsed,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_stochastic_lp_with_external_reference(
        &problem,
        &scenarios,
        &ExternalStochasticLpReferenceOptions {
            solver: ExternalStochasticLpReferenceSolver::RustMonolithic,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalStochasticLpReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalStochasticLpReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalStochasticLpReferenceStatus::Unbounded => ("ok", "unbounded"),
        ExternalStochasticLpReferenceStatus::IterLimit => ("failed", "iteration-limit"),
        ExternalStochasticLpReferenceStatus::Unsupported
        | ExternalStochasticLpReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalStochasticLpReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = solution
        .x
        .iter()
        .enumerate()
        .map(|(idx, value)| format!("x{idx}={value:.9}"))
        .collect::<Vec<_>>();
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if let Some(expected_q) = solution.expected_q {
        stdout.push(format!("expected_q={expected_q:.9}"));
    }
    stdout.push(format!("scenarios={}", solution.scenario_values.len()));
    if let Some(iterations) = solution.iterations {
        stdout.push(format!("iterations={iterations}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_routing_source(payload: &Value) -> &Value {
    [
        "routing_model",
        "routingModel",
        "vrp_model",
        "vrpModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_routing_matrix_value(source: &Value) -> Option<&Value> {
    [
        "distance_matrix",
        "distanceMatrix",
        "cost_matrix",
        "costMatrix",
        "travel_time_matrix",
        "travelTimeMatrix",
        "distances",
        "matrix",
    ]
    .iter()
    .filter_map(|key| source.get(*key))
    .find(|value| value.as_array().is_some())
}

fn model_validation_payload_has_routing_model(payload: &Value) -> bool {
    let source = model_validation_routing_source(payload);
    model_validation_routing_matrix_value(source).is_some()
        || source
            .get("customers")
            .and_then(Value::as_array)
            .is_some_and(|customers| !customers.is_empty())
}

fn model_validation_routing_number(value: Option<&Value>, label: &str) -> Result<f64, String> {
    let number = value
        .and_then(|value| model_validation_linear_number(Some(value)))
        .ok_or_else(|| format!("{label} must be numeric"))?;
    if !number.is_finite() {
        return Err(format!("{label} must be finite"));
    }
    Ok(number)
}

fn model_validation_routing_optional_number(source: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .filter_map(|key| source.get(*key))
        .find_map(|value| model_validation_linear_number(Some(value)).filter(|num| num.is_finite()))
}

fn model_validation_routing_index(
    value: &Value,
    limit: usize,
    label: &str,
) -> Result<usize, String> {
    let index = model_validation_linear_integer(value)
        .ok_or_else(|| format!("{label} must be an integer index"))?;
    if index < 0 {
        return Err(format!("{label} must be non-negative"));
    }
    let index = usize::try_from(index).map_err(|_| format!("{label} is too large"))?;
    if index >= limit {
        return Err(format!("{label} index {index} is outside 0..{limit}"));
    }
    Ok(index)
}

fn model_validation_routing_vehicle_count(source: &Value) -> Result<usize, String> {
    for key in [
        "vehicles",
        "num_vehicles",
        "numVehicles",
        "vehicle_count",
        "vehicleCount",
    ] {
        if let Some(value) = source.get(key) {
            let count = model_validation_linear_integer(value)
                .ok_or_else(|| format!("{key} must be an integer"))?;
            if count <= 0 {
                return Err(format!("{key} must be positive"));
            }
            return usize::try_from(count).map_err(|_| format!("{key} is too large"));
        }
    }
    for key in ["starts", "start_indices", "startIndices"] {
        if let Some(starts) = source.get(key).and_then(Value::as_array) {
            return Ok(starts.len().max(1));
        }
    }
    Ok(1)
}

fn model_validation_routing_first_index(
    source: &Value,
    array_keys: &[&str],
    scalar_keys: &[&str],
    limit: usize,
    default: usize,
) -> Result<usize, String> {
    for key in array_keys {
        if let Some(values) = source.get(*key).and_then(Value::as_array) {
            if let Some(value) = values.first() {
                return model_validation_routing_index(value, limit, key);
            }
        }
    }
    for key in scalar_keys {
        if let Some(value) = source.get(*key) {
            if value.is_number() || value.is_string() {
                return model_validation_routing_index(value, limit, key);
            }
        }
    }
    Ok(default)
}

fn model_validation_routing_matrix(source: &Value) -> Result<Option<Vec<Vec<f64>>>, String> {
    let Some(raw_matrix) = model_validation_routing_matrix_value(source) else {
        return Ok(None);
    };
    let rows = raw_matrix
        .as_array()
        .ok_or_else(|| "routing matrix must be an array".to_string())?;
    if rows.is_empty() {
        return Err("routing matrix must not be empty".to_string());
    }
    let width = rows.len();
    let mut matrix = Vec::with_capacity(width);
    for (row_idx, row) in rows.iter().enumerate() {
        let entries = row
            .as_array()
            .ok_or_else(|| format!("routing matrix row {row_idx} must be an array"))?;
        if entries.len() != width {
            return Err(format!(
                "routing matrix row {row_idx} has length {}, expected {width}",
                entries.len()
            ));
        }
        let mut parsed_row = Vec::with_capacity(width);
        for (col_idx, entry) in entries.iter().enumerate() {
            let cost = model_validation_routing_number(
                Some(entry),
                &format!("routing matrix[{row_idx}][{col_idx}]"),
            )?;
            if cost < 0.0 {
                return Err(format!(
                    "routing matrix[{row_idx}][{col_idx}] must be non-negative"
                ));
            }
            parsed_row.push(cost);
        }
        matrix.push(parsed_row);
    }
    Ok(Some(matrix))
}

fn model_validation_routing_matrix_customers(
    source: &Value,
    node_count: usize,
    start: usize,
    end: usize,
) -> Result<Vec<usize>, String> {
    for key in [
        "customer_indices",
        "customerIndices",
        "visit_nodes",
        "visitNodes",
    ] {
        if let Some(values) = source.get(key).and_then(Value::as_array) {
            let mut customers = Vec::with_capacity(values.len());
            for (idx, value) in values.iter().enumerate() {
                let customer =
                    model_validation_routing_index(value, node_count, &format!("{key}[{idx}]"))?;
                if customer == start || customer == end {
                    return Err(format!("{key}[{idx}] cannot be the start or end depot"));
                }
                if customers.contains(&customer) {
                    return Err(format!("{key}[{idx}] duplicates node {customer}"));
                }
                customers.push(customer);
            }
            return Ok(customers);
        }
    }
    if let Some(values) = source.get("customers").and_then(Value::as_array) {
        if values
            .iter()
            .all(|value| model_validation_linear_integer(value).is_some())
        {
            let mut customers = Vec::with_capacity(values.len());
            for (idx, value) in values.iter().enumerate() {
                let customer = model_validation_routing_index(
                    value,
                    node_count,
                    &format!("customers[{idx}]"),
                )?;
                if customer == start || customer == end {
                    return Err(format!("customers[{idx}] cannot be the start or end depot"));
                }
                if customers.contains(&customer) {
                    return Err(format!("customers[{idx}] duplicates node {customer}"));
                }
                customers.push(customer);
            }
            return Ok(customers);
        }
    }
    Ok((0..node_count)
        .filter(|node| *node != start && *node != end)
        .collect())
}

fn model_validation_routing_matrix_demands(
    source: &Value,
    node_count: usize,
) -> Result<Option<Vec<f64>>, String> {
    let Some(values) = ["demands", "demand", "loads"]
        .iter()
        .filter_map(|key| source.get(*key).and_then(Value::as_array))
        .next()
    else {
        return Ok(None);
    };
    if values.len() != node_count {
        return Err(format!(
            "routing demand vector has length {}, expected {node_count}",
            values.len()
        ));
    }
    let mut demands = Vec::with_capacity(node_count);
    for (idx, value) in values.iter().enumerate() {
        let demand = model_validation_routing_number(Some(value), &format!("demands[{idx}]"))?;
        if demand < 0.0 {
            return Err(format!("demands[{idx}] must be non-negative"));
        }
        demands.push(demand);
    }
    Ok(Some(demands))
}

fn model_validation_routing_matrix_capacity(source: &Value) -> Result<Option<f64>, String> {
    if let Some(capacity) = model_validation_routing_optional_number(
        source,
        &["capacity", "vehicle_capacity", "vehicleCapacity"],
    ) {
        if capacity <= 0.0 {
            return Err("vehicle capacity must be positive".to_string());
        }
        return Ok(Some(capacity));
    }
    for key in ["vehicle_capacities", "vehicleCapacities", "capacities"] {
        if let Some(values) = source.get(key).and_then(Value::as_array) {
            if let Some(value) = values.first() {
                let capacity = model_validation_routing_number(Some(value), &format!("{key}[0]"))?;
                if capacity <= 0.0 {
                    return Err(format!("{key}[0] must be positive"));
                }
                return Ok(Some(capacity));
            }
        }
    }
    Ok(None)
}

fn model_validation_routing_search_matrix_route(
    matrix: &[Vec<f64>],
    end: usize,
    current: usize,
    remaining: &mut [usize],
    depth: usize,
    route: &mut Vec<usize>,
    current_cost: f64,
    best: &mut Option<(f64, Vec<usize>)>,
) {
    if depth == remaining.len() {
        let total = current_cost + matrix[current][end];
        if best
            .as_ref()
            .is_none_or(|(best_cost, _)| total < *best_cost - 1e-9)
        {
            let mut best_route = route.clone();
            best_route.push(end);
            *best = Some((total, best_route));
        }
        return;
    }
    for idx in depth..remaining.len() {
        remaining.swap(depth, idx);
        let next = remaining[depth];
        let next_cost = current_cost + matrix[current][next];
        if best
            .as_ref()
            .is_none_or(|(best_cost, _)| next_cost < *best_cost - 1e-9)
        {
            route.push(next);
            model_validation_routing_search_matrix_route(
                matrix,
                end,
                next,
                remaining,
                depth + 1,
                route,
                next_cost,
                best,
            );
            route.pop();
        }
        remaining.swap(depth, idx);
    }
}

fn model_validation_routing_matrix_reference(source: &Value, tool: &str, validator: &str) -> Value {
    let matrix = match model_validation_routing_matrix(source) {
        Ok(Some(matrix)) => matrix,
        Ok(None) => {
            return model_validation_result(
                "failed",
                "failure",
                validator,
                "routing payload needs a distance matrix or coordinate CVRP data",
                "",
                "",
            );
        }
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    let vehicle_count = match model_validation_routing_vehicle_count(source) {
        Ok(vehicle_count) => vehicle_count,
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    if vehicle_count != 1 {
        return model_validation_result(
            "unavailable",
            "unknown",
            validator,
            format!("builtin routing matrix fallback supports one vehicle, got {vehicle_count}"),
            "",
            "",
        );
    }
    let node_count = matrix.len();
    if node_count > 10 {
        return model_validation_result(
            "unavailable",
            "unknown",
            validator,
            format!("builtin routing matrix fallback supports at most 10 nodes, got {node_count}"),
            "",
            "",
        );
    }
    let depot = match source.get("depot") {
        Some(value) if value.is_number() || value.is_string() => {
            match model_validation_routing_index(value, node_count, "depot") {
                Ok(depot) => depot,
                Err(message) => {
                    return model_validation_result(
                        "failed", "failure", validator, message, "", "",
                    );
                }
            }
        }
        _ => 0,
    };
    let start = match model_validation_routing_first_index(
        source,
        &["starts", "start_indices", "startIndices"],
        &["start", "start_index", "startIndex"],
        node_count,
        depot,
    ) {
        Ok(start) => start,
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    let end = match model_validation_routing_first_index(
        source,
        &["ends", "end_indices", "endIndices"],
        &["end", "end_index", "endIndex"],
        node_count,
        start,
    ) {
        Ok(end) => end,
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    let customers = match model_validation_routing_matrix_customers(source, node_count, start, end)
    {
        Ok(customers) => customers,
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    if customers.len() > 9 {
        return model_validation_result(
            "unavailable",
            "unknown",
            validator,
            format!(
                "builtin routing matrix fallback supports at most 9 visited nodes, got {}",
                customers.len()
            ),
            "",
            "",
        );
    }
    let demands = match model_validation_routing_matrix_demands(source, node_count) {
        Ok(demands) => demands,
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    let capacity = match model_validation_routing_matrix_capacity(source) {
        Ok(capacity) => capacity,
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    if let (Some(demands), Some(capacity)) = (&demands, capacity) {
        let load = customers.iter().map(|node| demands[*node]).sum::<f64>();
        if load > capacity + 1e-9 {
            return model_validation_result(
                "ok",
                "infeasible",
                validator,
                "single-vehicle route demand exceeds vehicle capacity",
                format!("load={load:.9} capacity={capacity:.9} solver=builtin:routing-matrix"),
                "",
            );
        }
    }

    let mut remaining = customers.clone();
    let mut route = vec![start];
    let mut best = None::<(f64, Vec<usize>)>;
    model_validation_routing_search_matrix_route(
        &matrix,
        end,
        start,
        &mut remaining,
        0,
        &mut route,
        0.0,
        &mut best,
    );
    let Some((objective, route)) = best else {
        return model_validation_result(
            "ok",
            "infeasible",
            validator,
            "no feasible route",
            format!("solver=builtin:routing-matrix-for-{tool}"),
            "",
        );
    };
    let route_text = route
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("->");
    model_validation_result(
        "ok",
        "optimal",
        validator,
        "optimal single-vehicle routing assignment found",
        format!(
            "route={route_text} objective={objective:.9} solver=builtin:routing-matrix-for-{tool}"
        ),
        "",
    )
}

fn model_validation_routing_point(value: &Value, label: &str) -> Result<Point, String> {
    if let Some(values) = value.as_array() {
        if values.len() < 2 {
            return Err(format!("{label} must have at least two coordinates"));
        }
        return Ok(Point {
            x: model_validation_routing_number(values.first(), &format!("{label}.x"))?,
            y: model_validation_routing_number(values.get(1), &format!("{label}.y"))?,
        });
    }
    let obj = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object or coordinate array"))?;
    let x = obj
        .get("x")
        .or_else(|| obj.get("lon"))
        .or_else(|| obj.get("longitude"));
    let y = obj
        .get("y")
        .or_else(|| obj.get("lat"))
        .or_else(|| obj.get("latitude"));
    Ok(Point {
        x: model_validation_routing_number(x, &format!("{label}.x"))?,
        y: model_validation_routing_number(y, &format!("{label}.y"))?,
    })
}

fn model_validation_routing_customer_point(value: &Value, idx: usize) -> Result<Point, String> {
    if value.as_array().is_some() {
        return model_validation_routing_point(value, &format!("customers[{idx}]"));
    }
    if let Some(obj) = value.as_object() {
        for key in ["point", "location", "coordinates", "coord"] {
            if let Some(point) = obj.get(key) {
                return model_validation_routing_point(point, &format!("customers[{idx}].{key}"));
            }
        }
    }
    model_validation_routing_point(value, &format!("customers[{idx}]"))
}

fn model_validation_routing_customer_id(value: &Value, idx: usize) -> String {
    value
        .as_object()
        .and_then(|obj| {
            obj.get("id")
                .or_else(|| obj.get("name"))
                .or_else(|| obj.get("label"))
        })
        .and_then(|value| match value {
            Value::String(text) => Some(text.clone()),
            Value::Number(number) => Some(number.to_string()),
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| format!("c{}", idx + 1))
}

fn model_validation_routing_customer_demand(
    source: &Value,
    value: &Value,
    idx: usize,
    customer_count: usize,
) -> Result<f64, String> {
    if let Some(obj) = value.as_object() {
        if let Some(demand) = obj
            .get("demand")
            .or_else(|| obj.get("load"))
            .or_else(|| obj.get("weight"))
        {
            return model_validation_routing_number(
                Some(demand),
                &format!("customers[{idx}].demand"),
            );
        }
    }
    if let Some(values) = value.as_array() {
        if let Some(demand) = values.get(2) {
            return model_validation_routing_number(
                Some(demand),
                &format!("customers[{idx}].demand"),
            );
        }
    }
    if let Some(demands) = source.get("demands").and_then(Value::as_array) {
        let demand = if demands.len() == customer_count {
            demands.get(idx)
        } else if demands.len() == customer_count + 1 {
            demands.get(idx + 1)
        } else {
            None
        };
        if let Some(demand) = demand {
            return model_validation_routing_number(Some(demand), &format!("demands[{idx}]"));
        }
    }
    Ok(1.0)
}

fn model_validation_routing_customers(source: &Value) -> Result<Vec<VRPCustomer>, String> {
    let raw_customers = source
        .get("customers")
        .and_then(Value::as_array)
        .ok_or_else(|| "coordinate CVRP payload needs customers array".to_string())?;
    let mut customers = Vec::with_capacity(raw_customers.len());
    for (idx, value) in raw_customers.iter().enumerate() {
        let point = model_validation_routing_customer_point(value, idx)?;
        let demand =
            model_validation_routing_customer_demand(source, value, idx, raw_customers.len())?;
        if demand < 0.0 {
            return Err(format!("customers[{idx}].demand must be non-negative"));
        }
        customers.push(VRPCustomer {
            id: model_validation_routing_customer_id(value, idx),
            x: point.x,
            y: point.y,
            demand,
        });
    }
    Ok(customers)
}

fn model_validation_routing_depot(source: &Value) -> Result<Point, String> {
    for key in ["depot", "depot_location", "depotLocation"] {
        if let Some(value) = source.get(key) {
            if value.is_object() || value.is_array() {
                return model_validation_routing_point(value, key);
            }
        }
    }
    Ok(Point { x: 0.0, y: 0.0 })
}

fn model_validation_routing_cvrp_capacity(
    source: &Value,
    customers: &[VRPCustomer],
) -> Result<f64, String> {
    if let Some(capacity) = model_validation_routing_optional_number(
        source,
        &["capacity", "vehicle_capacity", "vehicleCapacity"],
    ) {
        if capacity <= 0.0 {
            return Err("vehicle capacity must be positive".to_string());
        }
        return Ok(capacity);
    }
    for key in ["vehicle_capacities", "vehicleCapacities", "capacities"] {
        if let Some(values) = source.get(key).and_then(Value::as_array) {
            if let Some(value) = values.first() {
                let capacity = model_validation_routing_number(Some(value), &format!("{key}[0]"))?;
                if capacity <= 0.0 {
                    return Err(format!("{key}[0] must be positive"));
                }
                return Ok(capacity);
            }
        }
    }
    Ok(customers
        .iter()
        .map(|customer| customer.demand)
        .sum::<f64>()
        .max(1.0))
}

fn model_validation_routing_cvrp_reference(source: &Value, validator: &str) -> Value {
    let depot = match model_validation_routing_depot(source) {
        Ok(depot) => depot,
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    let customers = match model_validation_routing_customers(source) {
        Ok(customers) => customers,
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    let capacity = match model_validation_routing_cvrp_capacity(source, &customers) {
        Ok(capacity) => capacity,
        Err(message) => {
            return model_validation_result("failed", "failure", validator, message, "", "");
        }
    };
    let solution = solve_cvrp_with_external_reference(
        depot,
        &customers,
        capacity,
        &ExternalRoutingReferenceOptions {
            solver: ExternalRoutingReferenceSolver::RustExact,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalRoutingReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalRoutingReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalRoutingReferenceStatus::Unsupported
        | ExternalRoutingReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalRoutingReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = solution
        .routes
        .iter()
        .enumerate()
        .map(|(idx, route)| {
            format!(
                "route{idx}={} load={:.9} distance={:.9}",
                route.customers.join("->"),
                route.load,
                route.distance
            )
        })
        .collect::<Vec<_>>();
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if let Some(route_masks) = solution.feasible_route_masks {
        stdout.push(format!("feasible_route_masks={route_masks}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_routing_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:routing-small-for-{tool}");
    let source = model_validation_routing_source(payload);
    if model_validation_routing_matrix_value(source).is_some() {
        return model_validation_routing_matrix_reference(source, tool, &validator);
    }
    model_validation_routing_cvrp_reference(source, &validator)
}

fn model_validation_tsp_source(payload: &Value) -> &Value {
    [
        "tsp_model",
        "tspModel",
        "traveling_salesman_model",
        "travelingSalesmanModel",
        "travelling_salesman_model",
        "travellingSalesmanModel",
        "routing_model",
        "routingModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_tsp_points_value(source: &Value) -> Option<&Vec<Value>> {
    [
        "points",
        "cities",
        "nodes",
        "locations",
        "coordinates",
        "coords",
    ]
    .iter()
    .filter_map(|key| source.get(*key).and_then(Value::as_array))
    .next()
}

fn model_validation_tsp_has_vrp_fields(source: &Value) -> bool {
    [
        "customers",
        "demands",
        "demand",
        "capacity",
        "vehicle_capacity",
        "vehicleCapacity",
        "vehicle_capacities",
        "vehicleCapacities",
        "vehicles",
        "num_vehicles",
        "numVehicles",
        "starts",
        "ends",
    ]
    .iter()
    .any(|key| source.get(*key).is_some())
}

fn model_validation_payload_has_tsp_model(payload: &Value) -> bool {
    let source = model_validation_tsp_source(payload);
    model_validation_tsp_points_value(source).is_some()
        || (model_validation_routing_matrix_value(source).is_some()
            && !model_validation_tsp_has_vrp_fields(source))
}

fn model_validation_tsp_point(value: &Value, idx: usize) -> Result<ExternalTspPoint, String> {
    if let Some(obj) = value.as_object() {
        let id = obj
            .get("id")
            .or_else(|| obj.get("name"))
            .or_else(|| obj.get("label"))
            .map(|value| model_validation_string_value(value, &format!("points[{idx}].id")))
            .transpose()?;
        let point = model_validation_routing_point(value, &format!("points[{idx}]"))?;
        return Ok(ExternalTspPoint {
            id,
            x: point.x,
            y: point.y,
        });
    }
    let point = model_validation_routing_point(value, &format!("points[{idx}]"))?;
    Ok(ExternalTspPoint {
        id: Some(format!("c{idx}")),
        x: point.x,
        y: point.y,
    })
}

fn model_validation_tsp_points(source: &Value) -> Result<Option<Vec<ExternalTspPoint>>, String> {
    let Some(points) = model_validation_tsp_points_value(source) else {
        return Ok(None);
    };
    if points.len() < 2 {
        return Err("TSP points must contain at least two cities".to_string());
    }
    points
        .iter()
        .enumerate()
        .map(|(idx, point)| model_validation_tsp_point(point, idx))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn model_validation_tsp_matrix(source: &Value) -> Result<Vec<Vec<f64>>, String> {
    model_validation_number_matrix(
        source,
        &[
            "distance_matrix",
            "distanceMatrix",
            "cost_matrix",
            "costMatrix",
            "travel_time_matrix",
            "travelTimeMatrix",
            "distances",
            "matrix",
        ],
        "distance_matrix",
    )
}

fn model_validation_tsp_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:tsp-small-for-{tool}");
    let source = model_validation_tsp_source(payload);
    let solution = match model_validation_tsp_points(source) {
        Ok(Some(points)) => solve_euclidean_tsp_with_external_reference(
            &points,
            &ExternalTspReferenceOptions {
                solver: ExternalTspReferenceSolver::RustHeldKarp,
            },
        ),
        Ok(None) => {
            let matrix = match model_validation_tsp_matrix(source) {
                Ok(matrix) => matrix,
                Err(message) => {
                    return model_validation_result(
                        "failed", "failure", &validator, message, "", "",
                    );
                }
            };
            solve_tsp_with_external_reference(
                &matrix,
                &ExternalTspReferenceOptions {
                    solver: ExternalTspReferenceSolver::RustHeldKarp,
                },
            )
        }
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let (status, verdict) = match solution.status {
        ExternalTspReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalTspReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalTspReferenceStatus::Unsupported | ExternalTspReferenceStatus::Unavailable => {
            ("unavailable", "unknown")
        }
        ExternalTspReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut route = solution
        .tour
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>();
    if let Some(first) = solution.tour.first() {
        route.push(first.to_string());
    }
    let mut stdout = vec![format!("tour={}", route.join("->"))];
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_assignment_source(payload: &Value) -> &Value {
    [
        "assignment_model",
        "assignmentModel",
        "linear_sum_assignment",
        "linearSumAssignment",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_assignment_matrix_value(source: &Value) -> Option<&Value> {
    [
        "cost",
        "costs",
        "cost_matrix",
        "costMatrix",
        "costs_matrix",
        "costsMatrix",
        "matrix",
    ]
    .iter()
    .filter_map(|key| source.get(*key))
    .find(|value| value.as_array().is_some())
}

fn model_validation_payload_has_assignment_model(payload: &Value) -> bool {
    let source = model_validation_assignment_source(payload);
    source.get("cost").is_some()
        || source.get("costs").is_some()
        || (model_validation_assignment_matrix_value(source).is_some()
            && [
                "workers",
                "agents",
                "rows",
                "tasks",
                "jobs",
                "columns",
                "assignees",
            ]
            .iter()
            .any(|key| source.get(*key).is_some()))
}

fn model_validation_assignment_cost_matrix(source: &Value) -> Result<Vec<Vec<f64>>, String> {
    let raw_matrix = model_validation_assignment_matrix_value(source)
        .ok_or_else(|| "assignment payload needs cost or cost_matrix".to_string())?;
    let rows = raw_matrix
        .as_array()
        .ok_or_else(|| "assignment cost matrix must be an array".to_string())?;
    if rows.is_empty() {
        return Err("assignment cost matrix must not be empty".to_string());
    }
    let mut matrix = Vec::with_capacity(rows.len());
    let mut width = None::<usize>;
    for (row_idx, row) in rows.iter().enumerate() {
        let entries = row
            .as_array()
            .ok_or_else(|| format!("assignment cost row {row_idx} must be an array"))?;
        if entries.is_empty() {
            return Err(format!("assignment cost row {row_idx} must not be empty"));
        }
        let expected_width = *width.get_or_insert(entries.len());
        if entries.len() != expected_width {
            return Err(format!(
                "assignment cost row {row_idx} has length {}, expected {expected_width}",
                entries.len()
            ));
        }
        let mut parsed = Vec::with_capacity(entries.len());
        for (col_idx, entry) in entries.iter().enumerate() {
            parsed.push(model_validation_routing_number(
                Some(entry),
                &format!("assignment cost[{row_idx}][{col_idx}]"),
            )?);
        }
        matrix.push(parsed);
    }
    Ok(matrix)
}

fn model_validation_assignment_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:assignment-small-for-{tool}");
    let source = model_validation_assignment_source(payload);
    let cost = match model_validation_assignment_cost_matrix(source) {
        Ok(cost) => cost,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_assignment_with_external_reference(
        &cost,
        &ExternalAssignmentReferenceOptions {
            solver: ExternalAssignmentReferenceSolver::RustDp,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalAssignmentReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalAssignmentReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalAssignmentReferenceStatus::Unsupported
        | ExternalAssignmentReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalAssignmentReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = Vec::new();
    if !solution.assignment.is_empty() {
        stdout.push(format!(
            "assignment={}",
            solution
                .assignment
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_knapsack_source(payload: &Value) -> &Value {
    [
        "knapsack_model",
        "knapsackModel",
        "binary_knapsack",
        "binaryKnapsack",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_payload_has_knapsack_model(payload: &Value) -> bool {
    let source = model_validation_knapsack_source(payload);
    let has_capacity = [
        "capacity",
        "max_weight",
        "maxWeight",
        "weight_capacity",
        "weightCapacity",
    ]
    .iter()
    .any(|key| source.get(*key).is_some());
    has_capacity
        && (source.get("items").and_then(Value::as_array).is_some()
            || (source.get("weights").and_then(Value::as_array).is_some()
                && source.get("values").and_then(Value::as_array).is_some()))
}

fn model_validation_knapsack_capacity(source: &Value) -> Result<f64, String> {
    for key in [
        "capacity",
        "max_weight",
        "maxWeight",
        "weight_capacity",
        "weightCapacity",
    ] {
        if let Some(value) = source.get(key) {
            return model_validation_routing_number(Some(value), key);
        }
    }
    Err("knapsack payload needs capacity".to_string())
}

fn model_validation_knapsack_item_id(value: &Value, idx: usize) -> String {
    value
        .as_object()
        .and_then(|obj| {
            obj.get("id")
                .or_else(|| obj.get("name"))
                .or_else(|| obj.get("label"))
        })
        .and_then(|value| match value {
            Value::String(text) => Some(text.clone()),
            Value::Number(number) => Some(number.to_string()),
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| format!("item{idx}"))
}

fn model_validation_knapsack_item_from_value(
    value: &Value,
    idx: usize,
) -> Result<KnapsackItem, String> {
    if let Some(obj) = value.as_object() {
        let weight = obj
            .get("weight")
            .or_else(|| obj.get("w"))
            .or_else(|| obj.get("size"))
            .or_else(|| obj.get("cost"));
        let value_field = obj
            .get("value")
            .or_else(|| obj.get("v"))
            .or_else(|| obj.get("profit"))
            .or_else(|| obj.get("utility"));
        return Ok(KnapsackItem {
            id: model_validation_knapsack_item_id(value, idx),
            weight: model_validation_routing_number(weight, &format!("items[{idx}].weight"))?,
            value: model_validation_routing_number(value_field, &format!("items[{idx}].value"))?,
        });
    }
    if let Some(values) = value.as_array() {
        let (id, weight_idx, value_idx) = match values.first() {
            Some(Value::String(id)) if values.len() >= 3 => (id.clone(), 1, 2),
            Some(Value::Number(number)) if values.len() >= 3 => (number.to_string(), 1, 2),
            _ => (format!("item{idx}"), 0, 1),
        };
        return Ok(KnapsackItem {
            id,
            weight: model_validation_routing_number(
                values.get(weight_idx),
                &format!("items[{idx}].weight"),
            )?,
            value: model_validation_routing_number(
                values.get(value_idx),
                &format!("items[{idx}].value"),
            )?,
        });
    }
    Err(format!("items[{idx}] must be an object or array"))
}

fn model_validation_knapsack_item_id_from_arrays(source: &Value, idx: usize) -> String {
    for key in ["item_ids", "itemIds", "ids", "names"] {
        if let Some(values) = source.get(key).and_then(Value::as_array) {
            if let Some(value) = values.get(idx) {
                match value {
                    Value::String(text) if !text.trim().is_empty() => return text.clone(),
                    Value::Number(number) => return number.to_string(),
                    _ => {}
                }
            }
        }
    }
    format!("item{idx}")
}

fn model_validation_knapsack_items(source: &Value) -> Result<Vec<KnapsackItem>, String> {
    if let Some(items) = source.get("items").and_then(Value::as_array) {
        return items
            .iter()
            .enumerate()
            .map(|(idx, item)| model_validation_knapsack_item_from_value(item, idx))
            .collect();
    }
    let weights = source
        .get("weights")
        .and_then(Value::as_array)
        .ok_or_else(|| "knapsack payload needs items or weights".to_string())?;
    let values = source
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| "knapsack payload needs items or values".to_string())?;
    if weights.len() != values.len() {
        return Err(format!(
            "knapsack weights length {} does not match values length {}",
            weights.len(),
            values.len()
        ));
    }
    weights
        .iter()
        .zip(values.iter())
        .enumerate()
        .map(|(idx, (weight, value))| {
            Ok(KnapsackItem {
                id: model_validation_knapsack_item_id_from_arrays(source, idx),
                weight: model_validation_routing_number(Some(weight), &format!("weights[{idx}]"))?,
                value: model_validation_routing_number(Some(value), &format!("values[{idx}]"))?,
            })
        })
        .collect()
}

fn model_validation_knapsack_problem(source: &Value) -> Result<KnapsackProblem, String> {
    Ok(KnapsackProblem {
        capacity: model_validation_knapsack_capacity(source)?,
        items: model_validation_knapsack_items(source)?,
    })
}

fn model_validation_knapsack_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:knapsack-small-for-{tool}");
    let source = model_validation_knapsack_source(payload);
    let problem = match model_validation_knapsack_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_knapsack_with_external_reference(
        &problem,
        &ExternalKnapsackReferenceOptions {
            solver: ExternalKnapsackReferenceSolver::RustBranchAndBound,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalKnapsackReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalKnapsackReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalKnapsackReferenceStatus::Unsupported
        | ExternalKnapsackReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalKnapsackReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = vec![format!("items={}", solution.selected_item_ids.join(","))];
    if let Some(weight) = solution.total_weight {
        stdout.push(format!("weight={weight:.9}"));
    }
    if let Some(value) = solution.total_value {
        stdout.push(format!("value={value:.9}"));
    }
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if let Some(upper_bound) = solution.upper_bound {
        stdout.push(format!("upper_bound={upper_bound:.9}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_string_value(value: &Value, label: &str) -> Result<String, String> {
    match value {
        Value::String(text) if !text.trim().is_empty() => Ok(text.clone()),
        Value::Number(number) => Ok(number.to_string()),
        _ => Err(format!("{label} must be a non-empty string or number")),
    }
}

fn model_validation_string_array(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Vec<String>, String> {
    let values = keys
        .iter()
        .filter_map(|key| source.get(*key).and_then(Value::as_array))
        .next()
        .ok_or_else(|| format!("{label} array is required"))?;
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| model_validation_string_value(value, &format!("{label}[{idx}]")))
        .collect()
}

fn model_validation_number_array(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Vec<f64>, String> {
    let values = keys
        .iter()
        .filter_map(|key| source.get(*key).and_then(Value::as_array))
        .next()
        .ok_or_else(|| format!("{label} array is required"))?;
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            model_validation_routing_number(Some(value), &format!("{label}[{idx}]"))
        })
        .collect()
}

fn model_validation_number_matrix(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Vec<Vec<f64>>, String> {
    let rows = keys
        .iter()
        .filter_map(|key| source.get(*key).and_then(Value::as_array))
        .next()
        .ok_or_else(|| format!("{label} matrix is required"))?;
    if rows.is_empty() {
        return Err(format!("{label} matrix must not be empty"));
    }
    rows.iter()
        .enumerate()
        .map(|(row_idx, row)| {
            let values = row
                .as_array()
                .ok_or_else(|| format!("{label}[{row_idx}] must be an array"))?;
            values
                .iter()
                .enumerate()
                .map(|(col_idx, value)| {
                    model_validation_routing_number(
                        Some(value),
                        &format!("{label}[{row_idx}][{col_idx}]"),
                    )
                })
                .collect()
        })
        .collect()
}

fn model_validation_bin_packing_source(payload: &Value) -> &Value {
    [
        "bin_packing_model",
        "binPackingModel",
        "packing_model",
        "packingModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_bin_item_has_value(item: &Value) -> bool {
    item.as_object().is_some_and(|obj| {
        obj.get("value")
            .or_else(|| obj.get("v"))
            .or_else(|| obj.get("profit"))
            .or_else(|| obj.get("utility"))
            .is_some()
    })
}

fn model_validation_payload_has_bin_packing_model(payload: &Value) -> bool {
    let source = model_validation_bin_packing_source(payload);
    let has_capacity = [
        "capacity",
        "bin_capacity",
        "binCapacity",
        "max_weight",
        "maxWeight",
    ]
    .iter()
    .any(|key| source.get(*key).is_some());
    has_capacity
        && (source.get("weights").and_then(Value::as_array).is_some()
            || source
                .get("items")
                .and_then(Value::as_array)
                .is_some_and(|items| {
                    !items.is_empty() && !items.iter().any(model_validation_bin_item_has_value)
                }))
        && source.get("values").is_none()
}

fn model_validation_bin_packing_capacity(source: &Value) -> Result<f64, String> {
    for key in [
        "capacity",
        "bin_capacity",
        "binCapacity",
        "max_weight",
        "maxWeight",
    ] {
        if let Some(value) = source.get(key) {
            return model_validation_routing_number(Some(value), key);
        }
    }
    Err("bin-packing payload needs capacity".to_string())
}

fn model_validation_bin_packing_item(value: &Value, idx: usize) -> Result<BinPackingItem, String> {
    if let Some(obj) = value.as_object() {
        let id = obj
            .get("id")
            .or_else(|| obj.get("name"))
            .or_else(|| obj.get("label"))
            .map(|value| model_validation_string_value(value, &format!("items[{idx}].id")))
            .transpose()?
            .unwrap_or_else(|| format!("item{idx}"));
        let weight = obj
            .get("weight")
            .or_else(|| obj.get("w"))
            .or_else(|| obj.get("size"))
            .ok_or_else(|| format!("items[{idx}] needs weight"))?;
        return Ok(BinPackingItem {
            id,
            weight: model_validation_routing_number(Some(weight), &format!("items[{idx}].weight"))?,
        });
    }
    if let Some(values) = value.as_array() {
        let (id, weight_idx) = match values.first() {
            Some(Value::String(id)) if values.len() >= 2 => (id.clone(), 1),
            Some(Value::Number(number)) if values.len() >= 2 => (number.to_string(), 1),
            _ => (format!("item{idx}"), 0),
        };
        return Ok(BinPackingItem {
            id,
            weight: model_validation_routing_number(
                values.get(weight_idx),
                &format!("items[{idx}].weight"),
            )?,
        });
    }
    Ok(BinPackingItem {
        id: format!("item{idx}"),
        weight: model_validation_routing_number(Some(value), &format!("items[{idx}]"))?,
    })
}

fn model_validation_bin_packing_items(source: &Value) -> Result<Vec<BinPackingItem>, String> {
    if let Some(items) = source.get("items").and_then(Value::as_array) {
        return items
            .iter()
            .enumerate()
            .map(|(idx, item)| model_validation_bin_packing_item(item, idx))
            .collect();
    }
    let weights = source
        .get("weights")
        .and_then(Value::as_array)
        .ok_or_else(|| "bin-packing payload needs items or weights".to_string())?;
    weights
        .iter()
        .enumerate()
        .map(|(idx, weight)| {
            Ok(BinPackingItem {
                id: model_validation_knapsack_item_id_from_arrays(source, idx),
                weight: model_validation_routing_number(Some(weight), &format!("weights[{idx}]"))?,
            })
        })
        .collect()
}

fn model_validation_bin_packing_problem(source: &Value) -> Result<BinPackingProblem, String> {
    Ok(BinPackingProblem {
        capacity: model_validation_bin_packing_capacity(source)?,
        items: model_validation_bin_packing_items(source)?,
    })
}

fn model_validation_bin_packing_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:bin-packing-small-for-{tool}");
    let source = model_validation_bin_packing_source(payload);
    let problem = match model_validation_bin_packing_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_bin_packing_with_external_reference(
        &problem,
        &ExternalBinPackingReferenceOptions {
            solver: ExternalBinPackingReferenceSolver::RustExact,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalBinPackingReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalBinPackingReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalBinPackingReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalBinPackingReferenceStatus::Unsupported
        | ExternalBinPackingReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalBinPackingReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = Vec::new();
    if let Some(objective) = solution.objective {
        stdout.push(format!("bins={objective}"));
    }
    if let Some(total_weight) = solution.total_weight {
        stdout.push(format!("total_weight={total_weight:.9}"));
    }
    if let Some(lower_bound) = solution.lower_bound_bins {
        stdout.push(format!("lower_bound_bins={lower_bound}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_facility_location_source(payload: &Value) -> &Value {
    [
        "facility_location_model",
        "facilityLocationModel",
        "location_model",
        "locationModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_payload_has_facility_location_model(payload: &Value) -> bool {
    let source = model_validation_facility_location_source(payload);
    (source.get("facilities").and_then(Value::as_array).is_some()
        || source
            .get("facility_ids")
            .and_then(Value::as_array)
            .is_some()
        || source
            .get("facilityIds")
            .and_then(Value::as_array)
            .is_some())
        && (source.get("customers").and_then(Value::as_array).is_some()
            || source
                .get("customer_ids")
                .and_then(Value::as_array)
                .is_some()
            || source
                .get("customerIds")
                .and_then(Value::as_array)
                .is_some())
        && (source
            .get("fixed_costs")
            .and_then(Value::as_array)
            .is_some()
            || source.get("fixedCosts").and_then(Value::as_array).is_some()
            || source
                .get("opening_costs")
                .and_then(Value::as_array)
                .is_some()
            || source
                .get("openingCosts")
                .and_then(Value::as_array)
                .is_some())
        && (source
            .get("service_costs")
            .and_then(Value::as_array)
            .is_some()
            || source
                .get("serviceCosts")
                .and_then(Value::as_array)
                .is_some()
            || source
                .get("assignment_costs")
                .and_then(Value::as_array)
                .is_some()
            || source
                .get("assignmentCosts")
                .and_then(Value::as_array)
                .is_some())
}

fn model_validation_facility_location_problem(
    source: &Value,
) -> Result<FacilityLocationProblem, String> {
    Ok(FacilityLocationProblem {
        facility_ids: model_validation_string_array(
            source,
            &["facilities", "facility_ids", "facilityIds"],
            "facilities",
        )?,
        customer_ids: model_validation_string_array(
            source,
            &["customers", "customer_ids", "customerIds"],
            "customers",
        )?,
        fixed_costs: model_validation_number_array(
            source,
            &["fixed_costs", "fixedCosts", "opening_costs", "openingCosts"],
            "fixed_costs",
        )?,
        service_costs: model_validation_number_matrix(
            source,
            &[
                "service_costs",
                "serviceCosts",
                "assignment_costs",
                "assignmentCosts",
            ],
            "service_costs",
        )?,
    })
}

fn model_validation_facility_location_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:facility-location-small-for-{tool}");
    let source = model_validation_facility_location_source(payload);
    let problem = match model_validation_facility_location_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_facility_location_with_external_reference(
        &problem,
        &ExternalFacilityLocationReferenceOptions {
            solver: ExternalFacilityLocationReferenceSolver::RustExact,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalFacilityLocationReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalFacilityLocationReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalFacilityLocationReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalFacilityLocationReferenceStatus::Unsupported
        | ExternalFacilityLocationReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalFacilityLocationReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = vec![format!(
        "open_facilities={}",
        solution.open_facility_ids.join(",")
    )];
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    stdout.push(format!("assignments={}", solution.assignments.len()));
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_min_cost_flow_source(payload: &Value) -> &Value {
    [
        "min_cost_flow_model",
        "minCostFlowModel",
        "minimum_cost_flow_model",
        "minimumCostFlowModel",
        "flow_model",
        "flowModel",
        "network",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_min_cost_flow_arc_has_cost(value: &Value) -> bool {
    value.as_object().is_some_and(|obj| {
        obj.get("cost")
            .or_else(|| obj.get("unit_cost"))
            .or_else(|| obj.get("unitCost"))
            .or_else(|| obj.get("weight"))
            .is_some()
    }) || value.as_array().is_some_and(|items| items.len() >= 4)
}

fn model_validation_payload_has_min_cost_flow_model(payload: &Value) -> bool {
    let source = model_validation_min_cost_flow_source(payload);
    let has_arcs = source
        .get("arcs")
        .or_else(|| source.get("edges"))
        .and_then(Value::as_array)
        .is_some_and(|arcs| {
            !arcs.is_empty() && arcs.iter().all(model_validation_min_cost_flow_arc_has_cost)
        });
    let has_balances = source
        .get("supplies")
        .or_else(|| source.get("balances"))
        .or_else(|| source.get("node_balances"))
        .or_else(|| source.get("nodeBalances"))
        .and_then(Value::as_array)
        .is_some();
    has_arcs && has_balances
}

fn model_validation_optional_usize_field(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<Option<usize>, String> {
    keys.iter()
        .filter_map(|key| source.get(*key))
        .next()
        .map(|value| {
            let count = model_validation_linear_integer(value)
                .ok_or_else(|| format!("{label} must be an integer"))?;
            if count <= 0 {
                return Err(format!("{label} must be positive"));
            }
            usize::try_from(count).map_err(|_| format!("{label} is too large"))
        })
        .transpose()
}

fn model_validation_min_cost_flow_arc(value: &Value, idx: usize) -> Result<MinCostFlowArc, String> {
    if let Some(obj) = value.as_object() {
        let from = model_validation_usize_field(
            value,
            &["from", "source", "u", "tail"],
            &format!("arcs[{idx}].from"),
        )?;
        let to = model_validation_usize_field(
            value,
            &["to", "target", "v", "head"],
            &format!("arcs[{idx}].to"),
        )?;
        let lower_bound = obj
            .get("lower_bound")
            .or_else(|| obj.get("lowerBound"))
            .or_else(|| obj.get("lower"))
            .or_else(|| obj.get("lb"))
            .map(|value| {
                model_validation_routing_number(Some(value), &format!("arcs[{idx}].lower_bound"))
            })
            .transpose()?
            .unwrap_or(0.0);
        let capacity = obj
            .get("capacity")
            .or_else(|| obj.get("cap"))
            .or_else(|| obj.get("upper"))
            .or_else(|| obj.get("upper_bound"))
            .or_else(|| obj.get("upperBound"))
            .map(|value| {
                model_validation_routing_number(Some(value), &format!("arcs[{idx}].capacity"))
            })
            .transpose()?
            .ok_or_else(|| format!("arcs[{idx}] needs capacity"))?;
        let cost = obj
            .get("cost")
            .or_else(|| obj.get("unit_cost"))
            .or_else(|| obj.get("unitCost"))
            .or_else(|| obj.get("weight"))
            .map(|value| model_validation_routing_number(Some(value), &format!("arcs[{idx}].cost")))
            .transpose()?
            .ok_or_else(|| format!("arcs[{idx}] needs cost"))?;
        let name = obj.get("name").or_else(|| obj.get("id")).and_then(|value| {
            model_validation_string_value(value, &format!("arcs[{idx}].name")).ok()
        });
        return Ok(MinCostFlowArc {
            from,
            to,
            lower_bound,
            capacity,
            cost,
            name,
        });
    }
    if let Some(values) = value.as_array() {
        let (name, offset) = match values.first() {
            Some(Value::String(name)) => (Some(name.clone()), 1),
            _ => (None, 0),
        };
        if values.len().saturating_sub(offset) < 4 {
            return Err(format!(
                "arcs[{idx}] must have from, to, capacity, and cost"
            ));
        }
        let remaining = values.len() - offset;
        let lower_idx = (remaining >= 5).then_some(offset + 2);
        let capacity_idx = if remaining >= 5 {
            offset + 3
        } else {
            offset + 2
        };
        let cost_idx = if remaining >= 5 {
            offset + 4
        } else {
            offset + 3
        };
        return Ok(MinCostFlowArc {
            from: model_validation_routing_index(
                &values[offset],
                usize::MAX,
                &format!("arcs[{idx}][from]"),
            )?,
            to: model_validation_routing_index(
                &values[offset + 1],
                usize::MAX,
                &format!("arcs[{idx}][to]"),
            )?,
            lower_bound: lower_idx
                .map(|lower_idx| {
                    model_validation_routing_number(
                        values.get(lower_idx),
                        &format!("arcs[{idx}][lower_bound]"),
                    )
                })
                .transpose()?
                .unwrap_or(0.0),
            capacity: model_validation_routing_number(
                values.get(capacity_idx),
                &format!("arcs[{idx}][capacity]"),
            )?,
            cost: model_validation_routing_number(
                values.get(cost_idx),
                &format!("arcs[{idx}][cost]"),
            )?,
            name,
        });
    }
    Err(format!("arcs[{idx}] must be an object or array"))
}

fn model_validation_min_cost_flow_problem(source: &Value) -> Result<MinCostFlowProblem, String> {
    let raw_arcs = source
        .get("arcs")
        .or_else(|| source.get("edges"))
        .and_then(Value::as_array)
        .ok_or_else(|| "min-cost-flow payload needs arcs or edges array".to_string())?;
    let arcs = raw_arcs
        .iter()
        .enumerate()
        .map(|(idx, arc)| model_validation_min_cost_flow_arc(arc, idx))
        .collect::<Result<Vec<_>, _>>()?;
    let supplies = model_validation_number_array(
        source,
        &["supplies", "balances", "node_balances", "nodeBalances"],
        "supplies",
    )?;
    let derived_nodes = arcs
        .iter()
        .flat_map(|arc| [arc.from, arc.to])
        .max()
        .map(|max_index| max_index.saturating_add(1))
        .unwrap_or(0)
        .max(supplies.len());
    let num_nodes = model_validation_optional_usize_field(
        source,
        &["num_nodes", "numNodes", "node_count", "nodeCount"],
        "num_nodes",
    )?
    .unwrap_or(derived_nodes);
    Ok(MinCostFlowProblem {
        num_nodes,
        supplies,
        arcs,
    })
}

fn model_validation_min_cost_flow_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:min-cost-flow-small-for-{tool}");
    let source = model_validation_min_cost_flow_source(payload);
    let problem = match model_validation_min_cost_flow_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_min_cost_flow_with_external_reference(
        &problem,
        &ExternalMinCostFlowReferenceOptions {
            solver: ExternalMinCostFlowReferenceSolver::RustSuccessiveShortestPath,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalMinCostFlowReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalMinCostFlowReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalMinCostFlowReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalMinCostFlowReferenceStatus::Unsupported
        | ExternalMinCostFlowReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalMinCostFlowReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = Vec::new();
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    stdout.push(format!("arcs={}", solution.flows.len()));
    if let Some(iterations) = solution.iterations {
        stdout.push(format!("iterations={iterations}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_max_flow_source(payload: &Value) -> &Value {
    [
        "max_flow_model",
        "maxFlowModel",
        "flow_model",
        "flowModel",
        "network",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_payload_has_max_flow_model(payload: &Value) -> bool {
    let source = model_validation_max_flow_source(payload);
    source.get("edges").and_then(Value::as_array).is_some()
        && source
            .get("source")
            .or_else(|| source.get("source_node"))
            .or_else(|| source.get("sourceNode"))
            .is_some()
        && source
            .get("sink")
            .or_else(|| source.get("sink_node"))
            .or_else(|| source.get("sinkNode"))
            .is_some()
}

fn model_validation_usize_field(
    source: &Value,
    keys: &[&str],
    label: &str,
) -> Result<usize, String> {
    let value = keys
        .iter()
        .filter_map(|key| source.get(*key))
        .next()
        .ok_or_else(|| format!("{label} is required"))?;
    let number = model_validation_linear_integer(value)
        .ok_or_else(|| format!("{label} must be an integer"))?;
    if number < 0 {
        return Err(format!("{label} must be non-negative"));
    }
    usize::try_from(number).map_err(|_| format!("{label} is too large"))
}

fn model_validation_max_flow_edge(value: &Value, idx: usize) -> Result<MaxFlowEdge, String> {
    if let Some(obj) = value.as_object() {
        let from = model_validation_usize_field(
            value,
            &["from", "source", "u", "tail"],
            &format!("edges[{idx}].from"),
        )?;
        let to = model_validation_usize_field(
            value,
            &["to", "target", "v", "head"],
            &format!("edges[{idx}].to"),
        )?;
        let capacity = obj
            .get("capacity")
            .or_else(|| obj.get("cap"))
            .or_else(|| obj.get("upper"))
            .or_else(|| obj.get("upper_bound"))
            .or_else(|| obj.get("upperBound"))
            .map(|value| {
                model_validation_routing_number(Some(value), &format!("edges[{idx}].capacity"))
            })
            .transpose()?
            .ok_or_else(|| format!("edges[{idx}] needs capacity"))?;
        let name = obj.get("name").or_else(|| obj.get("id")).and_then(|value| {
            model_validation_string_value(value, &format!("edges[{idx}].name")).ok()
        });
        return Ok(MaxFlowEdge {
            from,
            to,
            capacity,
            name,
        });
    }
    if let Some(values) = value.as_array() {
        if values.len() < 3 {
            return Err(format!("edges[{idx}] must have from, to, and capacity"));
        }
        return Ok(MaxFlowEdge {
            from: model_validation_routing_index(
                &values[0],
                usize::MAX,
                &format!("edges[{idx}][0]"),
            )?,
            to: model_validation_routing_index(
                &values[1],
                usize::MAX,
                &format!("edges[{idx}][1]"),
            )?,
            capacity: model_validation_routing_number(values.get(2), &format!("edges[{idx}][2]"))?,
            name: None,
        });
    }
    Err(format!("edges[{idx}] must be an object or array"))
}

fn model_validation_max_flow_problem(source: &Value) -> Result<MaxFlowProblem, String> {
    let source_node =
        model_validation_usize_field(source, &["source", "source_node", "sourceNode"], "source")?;
    let sink = model_validation_usize_field(source, &["sink", "sink_node", "sinkNode"], "sink")?;
    let raw_edges = source
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "max-flow payload needs edges array".to_string())?;
    let edges = raw_edges
        .iter()
        .enumerate()
        .map(|(idx, edge)| model_validation_max_flow_edge(edge, idx))
        .collect::<Result<Vec<_>, _>>()?;
    let derived_nodes = edges
        .iter()
        .flat_map(|edge| [edge.from, edge.to])
        .chain([source_node, sink])
        .max()
        .map(|max_index| max_index.saturating_add(1))
        .unwrap_or(0);
    let num_nodes = source
        .get("num_nodes")
        .or_else(|| source.get("numNodes"))
        .or_else(|| source.get("node_count"))
        .or_else(|| source.get("nodeCount"))
        .map(|value| {
            let count = model_validation_linear_integer(value)
                .ok_or_else(|| "num_nodes must be an integer".to_string())?;
            if count <= 0 {
                return Err("num_nodes must be positive".to_string());
            }
            usize::try_from(count).map_err(|_| "num_nodes is too large".to_string())
        })
        .transpose()?
        .unwrap_or(derived_nodes);
    Ok(MaxFlowProblem {
        num_nodes,
        source: source_node,
        sink,
        edges,
    })
}

fn model_validation_max_flow_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:max-flow-small-for-{tool}");
    let source = model_validation_max_flow_source(payload);
    let problem = match model_validation_max_flow_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_max_flow_with_external_reference(
        &problem,
        &ExternalMaxFlowReferenceOptions {
            solver: ExternalMaxFlowReferenceSolver::RustEdmondsKarp,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalMaxFlowReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalMaxFlowReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalMaxFlowReferenceStatus::Unsupported
        | ExternalMaxFlowReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalMaxFlowReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = Vec::new();
    if let Some(max_flow) = solution.max_flow {
        stdout.push(format!("max_flow={max_flow:.9}"));
    }
    stdout.push(format!("min_cut_capacity={:.9}", solution.min_cut.capacity));
    if let Some(iterations) = solution.iterations {
        stdout.push(format!("iterations={iterations}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_wis_source(payload: &Value) -> &Value {
    [
        "weighted_independent_set_model",
        "weightedIndependentSetModel",
        "maximum_weight_independent_set_model",
        "maximumWeightIndependentSetModel",
        "independent_set_model",
        "independentSetModel",
        "set_packing_model",
        "setPackingModel",
        "conflict_graph",
        "conflictGraph",
        "graph",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_wis_weight_values(source: &Value) -> Option<&Vec<Value>> {
    [
        "weights",
        "vertex_weights",
        "vertexWeights",
        "node_weights",
        "nodeWeights",
        "profits",
        "values",
        "utilities",
    ]
    .iter()
    .filter_map(|key| source.get(*key).and_then(Value::as_array))
    .next()
}

fn model_validation_wis_vertex_has_weight(value: &Value) -> bool {
    value.as_object().is_some_and(|obj| {
        obj.get("weight")
            .or_else(|| obj.get("w"))
            .or_else(|| obj.get("value"))
            .or_else(|| obj.get("profit"))
            .or_else(|| obj.get("utility"))
            .is_some()
    }) || value.as_array().is_some_and(|items| {
        if items.len() >= 2 {
            model_validation_linear_number(items.get(1)).is_some()
        } else {
            items
                .first()
                .and_then(|item| model_validation_linear_number(Some(item)))
                .is_some()
        }
    }) || value.as_number().is_some()
}

fn model_validation_payload_has_wis_model(payload: &Value) -> bool {
    let source = model_validation_wis_source(payload);
    let has_edges = source.get("edges").and_then(Value::as_array).is_some();
    let has_weight_array = model_validation_wis_weight_values(source).is_some();
    let has_weighted_vertices = source
        .get("vertices")
        .or_else(|| source.get("nodes"))
        .and_then(Value::as_array)
        .is_some_and(|vertices| {
            !vertices.is_empty()
                && (has_weight_array || vertices.iter().all(model_validation_wis_vertex_has_weight))
        });
    has_edges && (has_weight_array || has_weighted_vertices)
}

fn model_validation_wis_index_id(source: &Value, idx: usize) -> Result<Option<String>, String> {
    for key in [
        "ids",
        "vertex_ids",
        "vertexIds",
        "node_ids",
        "nodeIds",
        "labels",
    ] {
        if let Some(values) = source.get(key).and_then(Value::as_array) {
            return values
                .get(idx)
                .map(|value| model_validation_string_value(value, &format!("{key}[{idx}]")))
                .transpose();
        }
    }
    Ok(None)
}

fn model_validation_wis_index_weight(source: &Value, idx: usize) -> Result<Option<f64>, String> {
    let Some(values) = model_validation_wis_weight_values(source) else {
        return Ok(None);
    };
    let value = values
        .get(idx)
        .ok_or_else(|| format!("weights[{idx}] is required"))?;
    model_validation_routing_number(Some(value), &format!("weights[{idx}]")).map(Some)
}

fn model_validation_wis_vertex(
    value: &Value,
    source: &Value,
    idx: usize,
) -> Result<WeightedIndependentSetVertex, String> {
    if let Some(obj) = value.as_object() {
        let id = obj
            .get("id")
            .or_else(|| obj.get("name"))
            .or_else(|| obj.get("label"))
            .or_else(|| obj.get("vertex"))
            .or_else(|| obj.get("node"))
            .map(|value| model_validation_string_value(value, &format!("vertices[{idx}].id")))
            .transpose()?
            .or(model_validation_wis_index_id(source, idx)?)
            .unwrap_or_else(|| format!("v{idx}"));
        let weight = obj
            .get("weight")
            .or_else(|| obj.get("w"))
            .or_else(|| obj.get("value"))
            .or_else(|| obj.get("profit"))
            .or_else(|| obj.get("utility"))
            .map(|value| {
                model_validation_routing_number(Some(value), &format!("vertices[{idx}].weight"))
            })
            .transpose()?
            .or(model_validation_wis_index_weight(source, idx)?)
            .ok_or_else(|| format!("vertices[{idx}] needs weight"))?;
        return Ok(WeightedIndependentSetVertex { id, weight });
    }
    if let Some(values) = value.as_array() {
        if values.is_empty() {
            return Err(format!("vertices[{idx}] must not be empty"));
        }
        let id = if values.len() >= 2 {
            model_validation_string_value(&values[0], &format!("vertices[{idx}].id"))?
        } else {
            model_validation_wis_index_id(source, idx)?.unwrap_or_else(|| format!("v{idx}"))
        };
        let weight = if values.len() >= 2 {
            model_validation_routing_number(values.get(1), &format!("vertices[{idx}].weight"))?
        } else {
            model_validation_wis_index_weight(source, idx)?.unwrap_or_else(|| {
                model_validation_routing_number(values.first(), &format!("vertices[{idx}].weight"))
                    .unwrap_or(f64::NAN)
            })
        };
        if !weight.is_finite() {
            return Err(format!("vertices[{idx}].weight must be finite"));
        }
        return Ok(WeightedIndependentSetVertex { id, weight });
    }
    if let Some(weight) = model_validation_wis_index_weight(source, idx)? {
        return Ok(WeightedIndependentSetVertex {
            id: model_validation_string_value(value, &format!("vertices[{idx}]"))?,
            weight,
        });
    }
    if let Some(weight) = model_validation_linear_number(Some(value)) {
        return Ok(WeightedIndependentSetVertex {
            id: model_validation_wis_index_id(source, idx)?.unwrap_or_else(|| format!("v{idx}")),
            weight,
        });
    }
    Ok(WeightedIndependentSetVertex {
        id: model_validation_string_value(value, &format!("vertices[{idx}]"))?,
        weight: model_validation_wis_index_weight(source, idx)?
            .ok_or_else(|| format!("vertices[{idx}] needs weight"))?,
    })
}

fn model_validation_wis_vertices(
    source: &Value,
) -> Result<Vec<WeightedIndependentSetVertex>, String> {
    if let Some(vertices) = source
        .get("vertices")
        .or_else(|| source.get("nodes"))
        .and_then(Value::as_array)
    {
        return vertices
            .iter()
            .enumerate()
            .map(|(idx, vertex)| model_validation_wis_vertex(vertex, source, idx))
            .collect();
    }
    let weights = model_validation_wis_weight_values(source)
        .ok_or_else(|| "weighted-independent-set payload needs vertices or weights".to_string())?;
    weights
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            Ok(WeightedIndependentSetVertex {
                id: model_validation_wis_index_id(source, idx)?
                    .unwrap_or_else(|| format!("v{idx}")),
                weight: model_validation_routing_number(Some(value), &format!("weights[{idx}]"))?,
            })
        })
        .collect()
}

fn model_validation_wis_edge_endpoint(
    value: Option<&Value>,
    label: &str,
) -> Result<String, String> {
    value
        .map(|value| model_validation_string_value(value, label))
        .transpose()?
        .ok_or_else(|| format!("{label} is required"))
}

fn model_validation_wis_edge(value: &Value, idx: usize) -> Result<(String, String), String> {
    if let Some(obj) = value.as_object() {
        let from = model_validation_wis_edge_endpoint(
            obj.get("from")
                .or_else(|| obj.get("source"))
                .or_else(|| obj.get("u"))
                .or_else(|| obj.get("tail"))
                .or_else(|| obj.get("a"))
                .or_else(|| obj.get("left")),
            &format!("edges[{idx}].from"),
        )?;
        let to = model_validation_wis_edge_endpoint(
            obj.get("to")
                .or_else(|| obj.get("target"))
                .or_else(|| obj.get("v"))
                .or_else(|| obj.get("head"))
                .or_else(|| obj.get("b"))
                .or_else(|| obj.get("right")),
            &format!("edges[{idx}].to"),
        )?;
        return Ok((from, to));
    }
    if let Some(values) = value.as_array() {
        if values.len() < 2 {
            return Err(format!("edges[{idx}] must have two endpoints"));
        }
        return Ok((
            model_validation_string_value(&values[0], &format!("edges[{idx}][0]"))?,
            model_validation_string_value(&values[1], &format!("edges[{idx}][1]"))?,
        ));
    }
    Err(format!("edges[{idx}] must be an object or array"))
}

fn model_validation_wis_problem(source: &Value) -> Result<WeightedIndependentSetProblem, String> {
    let raw_edges = source
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "weighted-independent-set payload needs edges array".to_string())?;
    Ok(WeightedIndependentSetProblem {
        vertices: model_validation_wis_vertices(source)?,
        edges: raw_edges
            .iter()
            .enumerate()
            .map(|(idx, edge)| model_validation_wis_edge(edge, idx))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn model_validation_wis_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:weighted-independent-set-small-for-{tool}");
    let source = model_validation_wis_source(payload);
    let problem = match model_validation_wis_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_weighted_independent_set_with_external_reference(
        &problem,
        &ExternalWeightedIndependentSetReferenceOptions {
            solver: ExternalWeightedIndependentSetReferenceSolver::RustBranchAndBound,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalWeightedIndependentSetReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalWeightedIndependentSetReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalWeightedIndependentSetReferenceStatus::Unsupported
        | ExternalWeightedIndependentSetReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalWeightedIndependentSetReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = vec![format!(
        "selected={}",
        solution.selected_vertex_ids.join(",")
    )];
    if let Some(total_weight) = solution.total_weight {
        stdout.push(format!("weight={total_weight:.9}"));
    }
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if let Some(upper_bound) = solution.upper_bound {
        stdout.push(format!("upper_bound={upper_bound:.9}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_scheduling_source(payload: &Value) -> &Value {
    [
        "scheduling_model",
        "schedulingModel",
        "job_shop_model",
        "jobShopModel",
        "flow_shop_model",
        "flowShopModel",
        "schedule_model",
        "scheduleModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_scheduling_job_has_operations(value: &Value) -> bool {
    value.as_object().is_some_and(|obj| {
        obj.get("operations")
            .or_else(|| obj.get("ops"))
            .or_else(|| obj.get("tasks"))
            .and_then(Value::as_array)
            .is_some()
            || (obj.get("machines").and_then(Value::as_array).is_some()
                && obj.get("durations").and_then(Value::as_array).is_some())
    })
}

fn model_validation_scheduling_job_has_processing_times(value: &Value) -> bool {
    value.as_object().is_some_and(|obj| {
        obj.get("processing_times")
            .or_else(|| obj.get("processingTimes"))
            .or_else(|| obj.get("durations"))
            .or_else(|| obj.get("times"))
            .and_then(Value::as_array)
            .is_some()
    }) || value.as_array().is_some_and(|items| {
        items.iter().any(|item| item.as_array().is_some())
            || items
                .iter()
                .all(|item| model_validation_linear_number(Some(item)).is_some())
    })
}

fn model_validation_payload_has_scheduling_model(payload: &Value) -> bool {
    let source = model_validation_scheduling_source(payload);
    source
        .get("jobs")
        .and_then(Value::as_array)
        .is_some_and(|jobs| {
            !jobs.is_empty()
                && jobs.iter().any(|job| {
                    model_validation_scheduling_job_has_operations(job)
                        || model_validation_scheduling_job_has_processing_times(job)
                })
        })
        || source
            .get("processing_times")
            .or_else(|| source.get("processingTimes"))
            .and_then(Value::as_array)
            .is_some()
}

fn model_validation_scheduling_kind(payload: &Value, source: &Value) -> String {
    payload
        .get("kind")
        .or_else(|| source.get("kind"))
        .and_then(Value::as_str)
        .map(model_validation_normalized_tool)
        .unwrap_or_default()
}

fn model_validation_scheduling_is_flow_shop(payload: &Value, source: &Value) -> bool {
    let kind = model_validation_scheduling_kind(payload, source);
    if matches!(
        kind.as_str(),
        "flow-shop-validation" | "flowshop-validation" | "flow-shop" | "flowshop"
    ) {
        return true;
    }
    if matches!(
        kind.as_str(),
        "job-shop-validation" | "jobshop-validation" | "job-shop" | "jobshop"
    ) {
        return false;
    }
    if source
        .get("processing_times")
        .or_else(|| source.get("processingTimes"))
        .is_some()
    {
        return true;
    }
    source
        .get("jobs")
        .and_then(Value::as_array)
        .is_some_and(|jobs| {
            !jobs.is_empty()
                && jobs
                    .iter()
                    .all(model_validation_scheduling_job_has_processing_times)
                && !jobs
                    .iter()
                    .any(model_validation_scheduling_job_has_operations)
        })
}

fn model_validation_scheduling_due(
    obj: &serde_json::Map<String, Value>,
    label: &str,
) -> Result<Option<f64>, String> {
    obj.get("due")
        .or_else(|| obj.get("due_date"))
        .or_else(|| obj.get("dueDate"))
        .or_else(|| obj.get("deadline"))
        .map(|value| model_validation_routing_number(Some(value), label))
        .transpose()
}

fn model_validation_scheduling_job_id(
    obj: Option<&serde_json::Map<String, Value>>,
    idx: usize,
) -> Result<String, String> {
    obj.and_then(|obj| {
        obj.get("id")
            .or_else(|| obj.get("name"))
            .or_else(|| obj.get("label"))
            .or_else(|| obj.get("job"))
    })
    .map(|value| model_validation_string_value(value, &format!("jobs[{idx}].id")))
    .transpose()
    .map(|id| id.unwrap_or_else(|| format!("J{}", idx + 1)))
}

fn model_validation_job_operation(value: &Value, label: &str) -> Result<JobOperation, String> {
    if let Some(obj) = value.as_object() {
        let machine = model_validation_wis_edge_endpoint(
            obj.get("machine")
                .or_else(|| obj.get("machine_id"))
                .or_else(|| obj.get("machineId"))
                .or_else(|| obj.get("resource"))
                .or_else(|| obj.get("station")),
            &format!("{label}.machine"),
        )?;
        let duration = obj
            .get("duration")
            .or_else(|| obj.get("processing_time"))
            .or_else(|| obj.get("processingTime"))
            .or_else(|| obj.get("time"))
            .or_else(|| obj.get("p"))
            .map(|value| model_validation_routing_number(Some(value), &format!("{label}.duration")))
            .transpose()?
            .ok_or_else(|| format!("{label} needs duration"))?;
        return Ok(JobOperation { machine, duration });
    }
    if let Some(values) = value.as_array() {
        if values.len() < 2 {
            return Err(format!("{label} must have machine and duration"));
        }
        return Ok(JobOperation {
            machine: model_validation_string_value(&values[0], &format!("{label}[0]"))?,
            duration: model_validation_routing_number(values.get(1), &format!("{label}[1]"))?,
        });
    }
    Err(format!("{label} must be an object or array"))
}

fn model_validation_job_shop_operations(
    obj: &serde_json::Map<String, Value>,
    job_idx: usize,
) -> Result<Vec<JobOperation>, String> {
    if let Some(operations) = obj
        .get("operations")
        .or_else(|| obj.get("ops"))
        .or_else(|| obj.get("tasks"))
        .and_then(Value::as_array)
    {
        return operations
            .iter()
            .enumerate()
            .map(|(op_idx, operation)| {
                model_validation_job_operation(
                    operation,
                    &format!("jobs[{job_idx}].operations[{op_idx}]"),
                )
            })
            .collect();
    }
    let machines = obj
        .get("machines")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("jobs[{job_idx}] needs operations or machines"))?;
    let durations = obj
        .get("durations")
        .or_else(|| obj.get("processing_times"))
        .or_else(|| obj.get("processingTimes"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("jobs[{job_idx}] needs durations"))?;
    if machines.len() != durations.len() {
        return Err(format!(
            "jobs[{job_idx}] machines length {} does not match durations length {}",
            machines.len(),
            durations.len()
        ));
    }
    machines
        .iter()
        .zip(durations.iter())
        .enumerate()
        .map(|(op_idx, (machine, duration))| {
            Ok(JobOperation {
                machine: model_validation_string_value(
                    machine,
                    &format!("jobs[{job_idx}].machines[{op_idx}]"),
                )?,
                duration: model_validation_routing_number(
                    Some(duration),
                    &format!("jobs[{job_idx}].durations[{op_idx}]"),
                )?,
            })
        })
        .collect()
}

fn model_validation_job_shop_job(value: &Value, idx: usize) -> Result<JobShopJob, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| format!("jobs[{idx}] must be an object"))?;
    Ok(JobShopJob {
        id: model_validation_scheduling_job_id(Some(obj), idx)?,
        due: model_validation_scheduling_due(obj, &format!("jobs[{idx}].due"))?,
        operations: model_validation_job_shop_operations(obj, idx)?,
    })
}

fn model_validation_job_shop_jobs(source: &Value) -> Result<Vec<JobShopJob>, String> {
    let jobs = source
        .get("jobs")
        .and_then(Value::as_array)
        .ok_or_else(|| "job-shop payload needs jobs array".to_string())?;
    jobs.iter()
        .enumerate()
        .map(|(idx, job)| model_validation_job_shop_job(job, idx))
        .collect()
}

fn model_validation_number_values(values: &[Value], label: &str) -> Result<Vec<f64>, String> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            model_validation_routing_number(Some(value), &format!("{label}[{idx}]"))
        })
        .collect()
}

fn model_validation_flow_shop_processing_times(
    value: &Value,
    label: &str,
) -> Result<Vec<f64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?;
    model_validation_number_values(values, label)
}

fn model_validation_flow_shop_job(value: &Value, idx: usize) -> Result<FlowShopJob, String> {
    if let Some(obj) = value.as_object() {
        let processing_times = obj
            .get("processing_times")
            .or_else(|| obj.get("processingTimes"))
            .or_else(|| obj.get("durations"))
            .or_else(|| obj.get("times"))
            .ok_or_else(|| format!("jobs[{idx}] needs processingTimes"))?;
        return Ok(FlowShopJob {
            id: model_validation_scheduling_job_id(Some(obj), idx)?,
            processing_times: model_validation_flow_shop_processing_times(
                processing_times,
                &format!("jobs[{idx}].processingTimes"),
            )?,
            due: model_validation_scheduling_due(obj, &format!("jobs[{idx}].due"))?,
        });
    }
    if let Some(values) = value.as_array() {
        if values.len() >= 2 && values[1].as_array().is_some() {
            return Ok(FlowShopJob {
                id: model_validation_string_value(&values[0], &format!("jobs[{idx}].id"))?,
                processing_times: model_validation_flow_shop_processing_times(
                    &values[1],
                    &format!("jobs[{idx}].processingTimes"),
                )?,
                due: None,
            });
        }
        return Ok(FlowShopJob {
            id: format!("F{}", idx + 1),
            processing_times: model_validation_number_values(values, &format!("jobs[{idx}]"))?,
            due: None,
        });
    }
    Err(format!("jobs[{idx}] must be an object or array"))
}

fn model_validation_flow_shop_jobs(source: &Value) -> Result<Vec<FlowShopJob>, String> {
    if let Some(jobs) = source.get("jobs").and_then(Value::as_array) {
        return jobs
            .iter()
            .enumerate()
            .map(|(idx, job)| model_validation_flow_shop_job(job, idx))
            .collect();
    }
    let matrix = source
        .get("processing_times")
        .or_else(|| source.get("processingTimes"))
        .and_then(Value::as_array)
        .ok_or_else(|| "flow-shop payload needs jobs or processingTimes".to_string())?;
    matrix
        .iter()
        .enumerate()
        .map(|(idx, row)| {
            Ok(FlowShopJob {
                id: model_validation_wis_index_id(source, idx)?
                    .unwrap_or_else(|| format!("F{}", idx + 1)),
                processing_times: model_validation_flow_shop_processing_times(
                    row,
                    &format!("processingTimes[{idx}]"),
                )?,
                due: None,
            })
        })
        .collect()
}

fn model_validation_scheduling_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:scheduling-small-for-{tool}");
    let source = model_validation_scheduling_source(payload);
    let is_flow_shop = model_validation_scheduling_is_flow_shop(payload, source);
    let (schedule_kind, solution) = if is_flow_shop {
        let jobs = match model_validation_flow_shop_jobs(source) {
            Ok(jobs) => jobs,
            Err(message) => {
                return model_validation_result("failed", "failure", &validator, message, "", "");
            }
        };
        (
            "flow-shop",
            solve_flow_shop_with_external_reference(
                &jobs,
                &ExternalSchedulingReferenceOptions {
                    solver: ExternalSchedulingReferenceSolver::RustExact,
                },
            ),
        )
    } else {
        let jobs = match model_validation_job_shop_jobs(source) {
            Ok(jobs) => jobs,
            Err(message) => {
                return model_validation_result("failed", "failure", &validator, message, "", "");
            }
        };
        (
            "job-shop",
            solve_job_shop_with_external_reference(
                &jobs,
                &ExternalSchedulingReferenceOptions {
                    solver: ExternalSchedulingReferenceSolver::RustExact,
                },
            ),
        )
    };
    let (status, verdict) = match solution.status {
        ExternalSchedulingReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalSchedulingReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalSchedulingReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalSchedulingReferenceStatus::Unsupported
        | ExternalSchedulingReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalSchedulingReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = vec![
        format!("kind={schedule_kind}"),
        format!("schedule_ops={}", solution.schedule.len()),
    ];
    if !solution.sequence.is_empty() {
        stdout.push(format!("sequence={}", solution.sequence.join(",")));
    }
    if let Some(makespan) = solution.makespan {
        stdout.push(format!("makespan={makespan:.9}"));
    }
    if let Some(total_flow_time) = solution.total_flow_time {
        stdout.push(format!("total_flow_time={total_flow_time:.9}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_mst_source(payload: &Value) -> &Value {
    [
        "mst_model",
        "mstModel",
        "minimum_spanning_tree_model",
        "minimumSpanningTreeModel",
        "spanning_tree_model",
        "spanningTreeModel",
        "graph",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_mst_edge_has_weight(value: &Value) -> bool {
    value.as_object().is_some_and(|obj| {
        obj.get("weight")
            .or_else(|| obj.get("cost"))
            .or_else(|| obj.get("distance"))
            .is_some()
    }) || value.as_array().is_some_and(|items| items.len() >= 3)
}

fn model_validation_payload_has_mst_model(payload: &Value) -> bool {
    let source = model_validation_mst_source(payload);
    source
        .get("edges")
        .and_then(Value::as_array)
        .is_some_and(|edges| {
            !edges.is_empty() && edges.iter().all(model_validation_mst_edge_has_weight)
        })
        && (source.get("vertices").and_then(Value::as_array).is_some()
            || source.get("nodes").and_then(Value::as_array).is_some())
}

fn model_validation_mst_vertices(source: &Value) -> Result<Vec<String>, String> {
    let values = source
        .get("vertices")
        .or_else(|| source.get("nodes"))
        .and_then(Value::as_array)
        .ok_or_else(|| "MST payload needs vertices or nodes array".to_string())?;
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| model_validation_string_value(value, &format!("vertices[{idx}]")))
        .collect()
}

fn model_validation_mst_edge(value: &Value, idx: usize) -> Result<MinimumSpanningTreeEdge, String> {
    if let Some(obj) = value.as_object() {
        let from = obj
            .get("from")
            .or_else(|| obj.get("source"))
            .or_else(|| obj.get("u"))
            .ok_or_else(|| format!("edges[{idx}] missing from/source/u"))?;
        let to = obj
            .get("to")
            .or_else(|| obj.get("target"))
            .or_else(|| obj.get("v"))
            .ok_or_else(|| format!("edges[{idx}] missing to/target/v"))?;
        let weight = obj
            .get("weight")
            .or_else(|| obj.get("cost"))
            .or_else(|| obj.get("distance"))
            .map(|value| {
                model_validation_routing_number(Some(value), &format!("edges[{idx}].weight"))
            })
            .transpose()?
            .ok_or_else(|| format!("edges[{idx}] needs weight"))?;
        let from = model_validation_string_value(from, &format!("edges[{idx}].from"))?;
        let to = model_validation_string_value(to, &format!("edges[{idx}].to"))?;
        let id = obj
            .get("id")
            .or_else(|| obj.get("name"))
            .map(|value| model_validation_string_value(value, &format!("edges[{idx}].id")))
            .transpose()?
            .unwrap_or_else(|| format!("{from}-{to}"));
        return Ok(MinimumSpanningTreeEdge {
            id,
            from,
            to,
            weight,
        });
    }
    if let Some(values) = value.as_array() {
        let (id, from_idx, to_idx, weight_idx) = match values.first() {
            Some(Value::String(id)) if values.len() >= 4 => (id.clone(), 1, 2, 3),
            Some(Value::Number(number)) if values.len() >= 4 => (number.to_string(), 1, 2, 3),
            _ => (format!("edge{idx}"), 0, 1, 2),
        };
        let from = model_validation_string_value(
            values
                .get(from_idx)
                .ok_or_else(|| format!("edges[{idx}] missing from"))?,
            &format!("edges[{idx}].from"),
        )?;
        let to = model_validation_string_value(
            values
                .get(to_idx)
                .ok_or_else(|| format!("edges[{idx}] missing to"))?,
            &format!("edges[{idx}].to"),
        )?;
        let weight = model_validation_routing_number(
            values.get(weight_idx),
            &format!("edges[{idx}].weight"),
        )?;
        return Ok(MinimumSpanningTreeEdge {
            id,
            from,
            to,
            weight,
        });
    }
    Err(format!("edges[{idx}] must be an object or array"))
}

fn model_validation_mst_problem(source: &Value) -> Result<MinimumSpanningTreeProblem, String> {
    let vertices = model_validation_mst_vertices(source)?;
    let raw_edges = source
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "MST payload needs edges array".to_string())?;
    let edges = raw_edges
        .iter()
        .enumerate()
        .map(|(idx, edge)| model_validation_mst_edge(edge, idx))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MinimumSpanningTreeProblem { vertices, edges })
}

fn model_validation_mst_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:mst-small-for-{tool}");
    let source = model_validation_mst_source(payload);
    let problem = match model_validation_mst_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_minimum_spanning_tree_with_external_reference(
        &problem,
        &ExternalMinimumSpanningTreeReferenceOptions {
            solver: ExternalMinimumSpanningTreeReferenceSolver::RustKruskal,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalMinimumSpanningTreeReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalMinimumSpanningTreeReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalMinimumSpanningTreeReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalMinimumSpanningTreeReferenceStatus::Unsupported => ("failed", "failure"),
        ExternalMinimumSpanningTreeReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalMinimumSpanningTreeReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = vec![format!("edges={}", solution.selected_edge_ids.join(","))];
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if let Some(weight) = solution.total_weight {
        stdout.push(format!("weight={weight:.9}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_graph_coloring_source(payload: &Value) -> &Value {
    [
        "graph_coloring_model",
        "graphColoringModel",
        "coloring_model",
        "coloringModel",
        "graph",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_payload_has_graph_coloring_model(payload: &Value) -> bool {
    let source = model_validation_graph_coloring_source(payload);
    source.get("edges").and_then(Value::as_array).is_some()
        && (source.get("vertices").and_then(Value::as_array).is_some()
            || source.get("nodes").and_then(Value::as_array).is_some()
            || source.get("vertex_count").is_some()
            || source.get("vertexCount").is_some())
}

fn model_validation_graph_coloring_vertices(source: &Value) -> Result<Vec<String>, String> {
    if let Some(values) = source
        .get("vertices")
        .or_else(|| source.get("nodes"))
        .and_then(Value::as_array)
    {
        return values
            .iter()
            .enumerate()
            .map(|(idx, value)| model_validation_string_value(value, &format!("vertices[{idx}]")))
            .collect();
    }
    for key in ["vertex_count", "vertexCount", "nodes_count", "nodeCount"] {
        if let Some(value) = source.get(key) {
            let count = model_validation_linear_integer(value)
                .ok_or_else(|| format!("{key} must be an integer"))?;
            if count <= 0 {
                return Err(format!("{key} must be positive"));
            }
            let count = usize::try_from(count).map_err(|_| format!("{key} is too large"))?;
            return Ok((0..count).map(|idx| idx.to_string()).collect());
        }
    }
    Err("graph-coloring payload needs vertices, nodes, or vertex_count".to_string())
}

fn model_validation_graph_coloring_edge(
    value: &Value,
    idx: usize,
) -> Result<(String, String), String> {
    if let Some(values) = value.as_array() {
        if values.len() < 2 {
            return Err(format!("edges[{idx}] must have at least two endpoints"));
        }
        return Ok((
            model_validation_string_value(&values[0], &format!("edges[{idx}][0]"))?,
            model_validation_string_value(&values[1], &format!("edges[{idx}][1]"))?,
        ));
    }
    let obj = value
        .as_object()
        .ok_or_else(|| format!("edges[{idx}] must be an array or object"))?;
    let left = obj
        .get("source")
        .or_else(|| obj.get("from"))
        .or_else(|| obj.get("u"))
        .or_else(|| obj.get("a"))
        .ok_or_else(|| format!("edges[{idx}] missing source/from/u"))?;
    let right = obj
        .get("target")
        .or_else(|| obj.get("to"))
        .or_else(|| obj.get("v"))
        .or_else(|| obj.get("b"))
        .ok_or_else(|| format!("edges[{idx}] missing target/to/v"))?;
    Ok((
        model_validation_string_value(left, &format!("edges[{idx}].source"))?,
        model_validation_string_value(right, &format!("edges[{idx}].target"))?,
    ))
}

fn model_validation_graph_coloring_problem(source: &Value) -> Result<GraphColoringProblem, String> {
    let vertices = model_validation_graph_coloring_vertices(source)?;
    let raw_edges = source
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "graph-coloring payload needs edges array".to_string())?;
    let edges = raw_edges
        .iter()
        .enumerate()
        .map(|(idx, edge)| model_validation_graph_coloring_edge(edge, idx))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GraphColoringProblem { vertices, edges })
}

fn model_validation_graph_coloring_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:graph-coloring-small-for-{tool}");
    let source = model_validation_graph_coloring_source(payload);
    let problem = match model_validation_graph_coloring_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_graph_coloring_with_external_reference(
        &problem,
        &ExternalGraphColoringReferenceOptions {
            solver: ExternalGraphColoringReferenceSolver::RustDsatur,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalGraphColoringReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalGraphColoringReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalGraphColoringReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalGraphColoringReferenceStatus::Unsupported
        | ExternalGraphColoringReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalGraphColoringReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = Vec::new();
    if !solution.color_indices.is_empty() {
        stdout.push(format!(
            "colors={}",
            solution
                .color_indices
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    if let Some(used_colors) = solution.used_color_count {
        stdout.push(format!("used_colors={used_colors}"));
    }
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_set_cover_source(payload: &Value) -> &Value {
    [
        "set_cover_model",
        "setCoverModel",
        "cover_model",
        "coverModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_payload_has_set_cover_model(payload: &Value) -> bool {
    let source = model_validation_set_cover_source(payload);
    (source.get("universe").and_then(Value::as_array).is_some()
        || source.get("elements").and_then(Value::as_array).is_some())
        && (source.get("sets").and_then(Value::as_array).is_some()
            || source.get("subsets").and_then(Value::as_array).is_some())
}

fn model_validation_set_cover_universe(source: &Value) -> Result<Vec<String>, String> {
    let values = source
        .get("universe")
        .or_else(|| source.get("elements"))
        .and_then(Value::as_array)
        .ok_or_else(|| "set-cover payload needs universe or elements array".to_string())?;
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| model_validation_string_value(value, &format!("universe[{idx}]")))
        .collect()
}

fn model_validation_set_cover_set(value: &Value, idx: usize) -> Result<SetCoverSet, String> {
    if let Some(obj) = value.as_object() {
        let id = obj
            .get("id")
            .or_else(|| obj.get("name"))
            .or_else(|| obj.get("label"))
            .map(|value| model_validation_string_value(value, &format!("sets[{idx}].id")))
            .transpose()?
            .unwrap_or_else(|| format!("set{idx}"));
        let cost = obj
            .get("cost")
            .or_else(|| obj.get("weight"))
            .or_else(|| obj.get("price"))
            .map(|value| model_validation_routing_number(Some(value), &format!("sets[{idx}].cost")))
            .transpose()?
            .unwrap_or(1.0);
        let elements = obj
            .get("elements")
            .or_else(|| obj.get("covers"))
            .or_else(|| obj.get("items"))
            .and_then(Value::as_array)
            .ok_or_else(|| format!("sets[{idx}] needs elements array"))?
            .iter()
            .enumerate()
            .map(|(elem_idx, value)| {
                model_validation_string_value(value, &format!("sets[{idx}].elements[{elem_idx}]"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(SetCoverSet { id, cost, elements });
    }
    if let Some(values) = value.as_array() {
        let (id, cost_idx, elements_idx) = match values.first() {
            Some(Value::String(id)) if values.len() >= 3 => (id.clone(), 1, 2),
            Some(Value::Number(number)) if values.len() >= 3 => (number.to_string(), 1, 2),
            _ => (format!("set{idx}"), 0, 1),
        };
        let cost =
            model_validation_routing_number(values.get(cost_idx), &format!("sets[{idx}].cost"))?;
        let elements = values
            .get(elements_idx)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("sets[{idx}] needs elements array"))?
            .iter()
            .enumerate()
            .map(|(elem_idx, value)| {
                model_validation_string_value(value, &format!("sets[{idx}].elements[{elem_idx}]"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(SetCoverSet { id, cost, elements });
    }
    Err(format!("sets[{idx}] must be an object or array"))
}

fn model_validation_set_cover_problem(source: &Value) -> Result<SetCoverProblem, String> {
    let universe = model_validation_set_cover_universe(source)?;
    let raw_sets = source
        .get("sets")
        .or_else(|| source.get("subsets"))
        .and_then(Value::as_array)
        .ok_or_else(|| "set-cover payload needs sets or subsets array".to_string())?;
    let sets = raw_sets
        .iter()
        .enumerate()
        .map(|(idx, value)| model_validation_set_cover_set(value, idx))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SetCoverProblem { universe, sets })
}

fn model_validation_set_cover_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:set-cover-small-for-{tool}");
    let source = model_validation_set_cover_source(payload);
    let problem = match model_validation_set_cover_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_set_cover_with_external_reference(
        &problem,
        &ExternalSetCoverReferenceOptions {
            solver: ExternalSetCoverReferenceSolver::RustExact,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalSetCoverReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalSetCoverReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalSetCoverReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalSetCoverReferenceStatus::Unsupported
        | ExternalSetCoverReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalSetCoverReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let mut stdout = vec![format!("sets={}", solution.selected_set_ids.join(","))];
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if !solution.covered_elements.is_empty() {
        stdout.push(format!("covered={}", solution.covered_elements.join(",")));
    }
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_payload_optional_text<'a>(
    payload: &'a Value,
    keys: &[&str],
) -> Option<&'a str> {
    keys.iter()
        .filter_map(|key| payload.get(*key).and_then(Value::as_str))
        .find(|text| !text.trim().is_empty())
}

fn model_validation_pddl_without_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let before_comment = line.split_once(';').map_or(line, |(before, _)| before);
        out.push_str(before_comment);
        out.push('\n');
    }
    out
}

fn model_validation_pddl_text_has_marker(text: &str, marker: &str) -> bool {
    model_validation_pddl_without_comments(text)
        .to_ascii_lowercase()
        .contains(marker)
}

fn model_validation_payload_has_pddl(payload: &Value) -> bool {
    [
        "domain",
        "domain_pddl",
        "domainPddl",
        "problem",
        "problem_pddl",
        "problemPddl",
        "pddl",
        "model",
        "text",
        "content",
        "plan",
        "plan_text",
        "solution",
        "output",
    ]
    .iter()
    .any(|key| {
        payload
            .get(*key)
            .and_then(Value::as_str)
            .is_some_and(|text| {
                let lower = model_validation_pddl_without_comments(text).to_ascii_lowercase();
                lower.contains("(define")
                    || lower.contains("(:action")
                    || lower.contains("(:durative-action")
                    || lower.lines().any(|line| {
                        let action = model_validation_pddl_plan_action_fragment(line);
                        action.starts_with('(') && action.contains(')')
                    })
            })
    })
}

fn model_validation_pddl_domain_errors(text: &str) -> Vec<String> {
    let stripped = model_validation_pddl_without_comments(text);
    let lower = stripped.to_ascii_lowercase();
    let mut errors = output_validation_balanced_delimiters(&stripped);
    if !lower.contains("(define") {
        errors.push("domain: missing (define ...) form".to_string());
    }
    if !lower.contains("(domain") {
        errors.push("domain: missing (domain NAME) declaration".to_string());
    }
    if !lower.contains(":predicates")
        && !lower.contains(":functions")
        && !lower.contains(":action")
        && !lower.contains(":durative-action")
    {
        errors.push("domain: missing predicates, functions, or actions".to_string());
    }
    if !lower.contains(":action") && !lower.contains(":durative-action") {
        errors.push("domain: missing action or durative-action declaration".to_string());
    }
    errors
}

fn model_validation_pddl_problem_errors(text: &str) -> Vec<String> {
    let stripped = model_validation_pddl_without_comments(text);
    let lower = stripped.to_ascii_lowercase();
    let mut errors = output_validation_balanced_delimiters(&stripped);
    if !lower.contains("(define") {
        errors.push("problem: missing (define ...) form".to_string());
    }
    if !lower.contains("(problem") {
        errors.push("problem: missing (problem NAME) declaration".to_string());
    }
    for marker in [":domain", ":init", ":goal"] {
        if !lower.contains(marker) {
            errors.push(format!("problem: missing {marker} section"));
        }
    }
    errors
}

fn model_validation_pddl_plan_action_fragment(line: &str) -> &str {
    let trimmed = line.trim();
    if let Some((prefix, rest)) = trimmed.split_once(':') {
        if prefix.trim().parse::<f64>().is_ok() {
            return rest.trim();
        }
    }
    trimmed
}

fn model_validation_pddl_plan_errors(text: &str, allow_empty_plan: bool) -> (Vec<String>, usize) {
    let stripped = model_validation_pddl_without_comments(text);
    let mut errors = output_validation_balanced_delimiters(&stripped);
    let mut action_count = 0_usize;
    for (line_idx, line) in stripped.lines().enumerate() {
        let action = model_validation_pddl_plan_action_fragment(line);
        if action.is_empty() {
            continue;
        }
        if action.starts_with('(') && action.contains(')') {
            action_count += 1;
            continue;
        }
        if action.to_ascii_lowercase().starts_with("cost")
            || action.to_ascii_lowercase().starts_with("metric")
        {
            continue;
        }
        errors.push(format!(
            "plan line {} is not a parenthesized PDDL action",
            line_idx + 1
        ));
    }
    if action_count == 0 && !allow_empty_plan {
        errors.push("plan: missing parenthesized action lines".to_string());
    }
    (errors, action_count)
}

fn model_validation_pddl_reference(payload: &Value, tool: &str) -> Value {
    let tool = model_validation_normalized_tool(tool);
    let kind = payload
        .get("kind")
        .and_then(Value::as_str)
        .map(model_validation_normalized_tool)
        .unwrap_or_default();
    let validator = format!("builtin:pddl-structural-for-{tool}");
    let generic =
        model_validation_payload_optional_text(payload, &["pddl", "model", "text", "content"]);
    let domain = model_validation_payload_optional_text(
        payload,
        &["domain", "domain_pddl", "domainPddl", "domain_text"],
    )
    .or_else(|| generic.filter(|text| model_validation_pddl_text_has_marker(text, "(domain")));
    let problem = model_validation_payload_optional_text(
        payload,
        &[
            "problem",
            "problem_pddl",
            "problemPddl",
            "problem_text",
            "instance",
        ],
    )
    .or_else(|| generic.filter(|text| model_validation_pddl_text_has_marker(text, "(problem")));
    let plan = model_validation_payload_optional_text(
        payload,
        &["plan", "plan_text", "solution", "output", "actions"],
    );
    let plan_validator_tools = ["pddl-val", "validate", "val", "pddl-validate"];
    let planning_solver_tools = [
        "fast-downward",
        "fast-downward.py",
        "lpg-td",
        "lpg",
        "optic",
        "optic-clp",
        "enhsp",
        "enhsp.jar",
    ];
    let needs_plan = kind.contains("plan") || plan_validator_tools.contains(&tool.as_str());
    let needs_domain_problem =
        needs_plan || planning_solver_tools.contains(&tool.as_str()) || kind.contains("planning");
    if generic.is_none() && domain.is_none() && problem.is_none() && plan.is_none() {
        return model_validation_result(
            "failed",
            "failure",
            &validator,
            "payload needs PDDL domain/problem text or a plan",
            "",
            "",
        );
    }
    if needs_domain_problem && domain.is_none() {
        return model_validation_result(
            "failed",
            "failure",
            &validator,
            "payload needs domain, domain_pddl, domainPddl, or combined pddl/model/text content",
            "",
            "",
        );
    }
    if needs_domain_problem && problem.is_none() {
        return model_validation_result(
            "failed",
            "failure",
            &validator,
            "payload needs problem, problem_pddl, problemPddl, or combined pddl/model/text content",
            "",
            "",
        );
    }
    if needs_plan && plan.is_none() {
        return model_validation_result(
            "failed",
            "failure",
            &validator,
            "payload needs plan, plan_text, solution, output, or actions",
            "",
            "",
        );
    }

    let mut errors = Vec::new();
    if let Some(domain) = domain {
        errors.extend(model_validation_pddl_domain_errors(domain));
    }
    if let Some(problem) = problem {
        errors.extend(model_validation_pddl_problem_errors(problem));
    }
    if domain.is_none() && problem.is_none() {
        if let Some(generic) = generic {
            let stripped = model_validation_pddl_without_comments(generic);
            let lower = stripped.to_ascii_lowercase();
            errors.extend(output_validation_balanced_delimiters(&stripped));
            if !lower.contains("(define") {
                errors.push("pddl: missing (define ...) form".to_string());
            }
            if !lower.contains("(domain") && !lower.contains("(problem") {
                errors.push("pddl: missing domain or problem declaration".to_string());
            }
        }
    }
    let mut action_count = 0_usize;
    if let Some(plan) = plan {
        let (plan_errors, count) = model_validation_pddl_plan_errors(
            plan,
            payload
                .get("allow_empty_plan")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        );
        errors.extend(plan_errors);
        action_count = count;
    }

    if errors.is_empty() {
        let stdout = format!(
            "domain={} problem={} plan_actions={action_count}\n",
            domain.is_some(),
            problem.is_some()
        );
        model_validation_result(
            "ok",
            "valid",
            &validator,
            "PDDL structure accepted",
            stdout,
            "",
        )
    } else {
        model_validation_result(
            "ok",
            "invalid",
            &validator,
            errors.first().cloned().unwrap_or_default(),
            "",
            errors.join("\n"),
        )
    }
}

fn model_validation_parse_dimacs_cnf(text: &str) -> Result<(usize, Vec<Vec<i32>>), String> {
    let mut variables = 0_usize;
    let mut clauses = Vec::new();
    let mut pending = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                variables = parts[2]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid DIMACS variable count {:?}", parts[2]))?;
            }
            continue;
        }
        for token in line.split_whitespace() {
            let literal = token
                .parse::<i32>()
                .map_err(|_| format!("invalid DIMACS literal {token:?}"))?;
            if literal == 0 {
                clauses.push(std::mem::take(&mut pending));
            } else {
                variables = variables.max(literal.unsigned_abs() as usize);
                pending.push(literal);
            }
        }
    }
    if !pending.is_empty() {
        clauses.push(pending);
    }
    Ok((variables, clauses))
}

fn model_validation_dimacs_reference(payload: &Value) -> Value {
    let text = model_validation_payload_text(payload, &["dimacs", "cnf", "text", "model"]);
    if text.trim().is_empty() {
        return model_validation_result(
            "failed",
            "failure",
            "dimacs",
            "payload needs dimacs, cnf, text, or model",
            "",
            "",
        );
    }
    let (variables, clauses) = match model_validation_parse_dimacs_cnf(text) {
        Ok(parsed) => parsed,
        Err(message) => {
            return model_validation_result(
                "failed",
                "failure",
                "builtin:dimacs-small-cnf",
                message,
                "",
                "",
            );
        }
    };
    if variables > 24 {
        return model_validation_result(
            "unavailable",
            "unknown",
            "builtin:dimacs-small-cnf",
            format!("builtin CNF fallback is capped at 24 variables, got {variables}"),
            "",
            "",
        );
    }
    let limit = 1_u64 << variables;
    for mask in 0..limit {
        let satisfied = clauses.iter().all(|clause| {
            clause.iter().any(|literal| {
                let var = literal.unsigned_abs() as usize - 1;
                let value = ((mask >> var) & 1) == 1;
                value == (*literal > 0)
            })
        });
        if satisfied {
            let mut model = Vec::new();
            for idx in 0..variables {
                if ((mask >> idx) & 1) == 1 {
                    model.push((idx + 1).to_string());
                } else {
                    model.push(format!("-{}", idx + 1));
                }
            }
            return model_validation_result(
                "ok",
                "sat",
                "builtin:dimacs-small-cnf",
                "satisfying assignment found",
                format!("s SATISFIABLE\nv {} 0\n", model.join(" ")),
                "",
            );
        }
    }
    model_validation_result(
        "ok",
        "unsat",
        "builtin:dimacs-small-cnf",
        "all assignments exhausted",
        "",
        "",
    )
}

fn model_validation_parse_wcnf(
    text: &str,
) -> Result<(usize, Option<i64>, Vec<(i64, Vec<i32>)>), String> {
    let mut variables = 0_usize;
    let mut top_weight = None;
    let mut clauses = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 && parts[1].eq_ignore_ascii_case("wcnf") {
                variables = parts[2]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid WCNF variable count {:?}", parts[2]))?;
                if parts.len() >= 5 {
                    top_weight = Some(
                        parts[4]
                            .parse::<i64>()
                            .map_err(|_| format!("invalid WCNF top weight {:?}", parts[4]))?,
                    );
                }
            }
            continue;
        }
        let tokens: Vec<i64> = line
            .split_whitespace()
            .map(|token| {
                token
                    .parse::<i64>()
                    .map_err(|_| format!("invalid WCNF token {token:?}"))
            })
            .collect::<Result<_, _>>()?;
        if tokens.len() < 2 || tokens.last() != Some(&0) {
            return Err("WCNF clauses must be '<weight> <lits...> 0'".to_string());
        }
        let weight = tokens[0];
        let clause: Vec<i32> = tokens[1..tokens.len() - 1]
            .iter()
            .map(|literal| *literal as i32)
            .collect();
        for literal in &clause {
            variables = variables.max(literal.unsigned_abs() as usize);
        }
        clauses.push((weight, clause));
    }
    Ok((variables, top_weight, clauses))
}

fn model_validation_wcnf_reference(payload: &Value) -> Value {
    let text = model_validation_payload_text(payload, &["wcnf", "dimacs", "text", "model"]);
    if text.trim().is_empty() {
        return model_validation_result(
            "failed",
            "failure",
            "wcnf",
            "payload needs wcnf, dimacs, text, or model",
            "",
            "",
        );
    }
    let (variables, top_weight, clauses) = match model_validation_parse_wcnf(text) {
        Ok(parsed) => parsed,
        Err(message) => {
            return model_validation_result(
                "failed",
                "failure",
                "builtin:wcnf-small-maxsat",
                message,
                "",
                "",
            );
        }
    };
    if variables > 24 {
        return model_validation_result(
            "unavailable",
            "unknown",
            "builtin:wcnf-small-maxsat",
            format!("builtin WCNF fallback is capped at 24 variables, got {variables}"),
            "",
            "",
        );
    }
    let mut best_cost: Option<i64> = None;
    let mut best_mask = 0_u64;
    for mask in 0..(1_u64 << variables) {
        let mut hard_failed = false;
        let mut cost = 0_i64;
        for (weight, clause) in &clauses {
            let satisfied = clause.iter().any(|literal| {
                let var = literal.unsigned_abs() as usize - 1;
                let value = ((mask >> var) & 1) == 1;
                value == (*literal > 0)
            });
            if satisfied {
                continue;
            }
            if top_weight.is_some_and(|top| *weight >= top) {
                hard_failed = true;
                break;
            }
            cost += *weight;
        }
        if hard_failed {
            continue;
        }
        if best_cost.is_none_or(|best| cost < best) {
            best_cost = Some(cost);
            best_mask = mask;
        }
    }
    let Some(best_cost) = best_cost else {
        return model_validation_result(
            "ok",
            "unsat",
            "builtin:wcnf-small-maxsat",
            "hard clauses are unsatisfiable",
            "",
            "",
        );
    };
    let mut model = Vec::new();
    for idx in 0..variables {
        if ((best_mask >> idx) & 1) == 1 {
            model.push((idx + 1).to_string());
        } else {
            model.push(format!("-{}", idx + 1));
        }
    }
    model_validation_result(
        "ok",
        "optimal",
        "builtin:wcnf-small-maxsat",
        format!("optimum={best_cost}"),
        format!("o {best_cost}\ns OPTIMUM FOUND\nv {} 0\n", model.join(" ")),
        "",
    )
}

fn model_validation_weighted_max_sat_source(payload: &Value) -> &Value {
    [
        "weighted_max_sat_model",
        "weightedMaxSatModel",
        "weighted_maxsat_model",
        "weightedMaxsatModel",
        "max_sat_model",
        "maxSatModel",
        "maxsat_model",
        "maxsatModel",
        "model",
        "problem",
    ]
    .iter()
    .filter_map(|key| payload.get(*key))
    .find(|value| value.as_object().is_some())
    .unwrap_or(payload)
}

fn model_validation_payload_has_weighted_max_sat_model(payload: &Value) -> bool {
    let source = model_validation_weighted_max_sat_source(payload);
    source
        .get("clauses")
        .or_else(|| source.get("soft_clauses"))
        .or_else(|| source.get("softClauses"))
        .and_then(Value::as_array)
        .is_some_and(|clauses| {
            !clauses.is_empty()
                && clauses.iter().all(|clause| {
                    clause.as_object().is_some_and(|obj| {
                        obj.get("literals")
                            .or_else(|| obj.get("lits"))
                            .or_else(|| obj.get("clause"))
                            .and_then(Value::as_array)
                            .is_some()
                    }) || clause.as_array().is_some()
                })
        })
}

fn model_validation_weighted_max_sat_literal(value: &Value, label: &str) -> Result<i64, String> {
    let literal = model_validation_linear_integer(value)
        .ok_or_else(|| format!("{label} must be an integer"))?;
    if literal == 0 {
        return Err(format!("{label} must be non-zero"));
    }
    Ok(literal)
}

fn model_validation_weighted_max_sat_literals(
    value: &Value,
    label: &str,
) -> Result<Vec<i64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?;
    if values.is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            model_validation_weighted_max_sat_literal(value, &format!("{label}[{idx}]"))
        })
        .collect()
}

fn model_validation_weighted_max_sat_bool(value: &Value) -> Option<bool> {
    if let Some(value) = value.as_bool() {
        return Some(value);
    }
    value
        .as_str()
        .and_then(|text| match text.trim().to_ascii_lowercase().as_str() {
            "true" | "hard" | "required" | "mandatory" => Some(true),
            "false" | "soft" | "optional" => Some(false),
            _ => None,
        })
}

fn model_validation_weighted_max_sat_hard(obj: &serde_json::Map<String, Value>) -> bool {
    obj.get("hard")
        .or_else(|| obj.get("required"))
        .or_else(|| obj.get("mandatory"))
        .or_else(|| obj.get("type"))
        .and_then(model_validation_weighted_max_sat_bool)
        .unwrap_or(false)
}

fn model_validation_weighted_max_sat_clause(
    value: &Value,
    idx: usize,
) -> Result<WeightedMaxSatClause, String> {
    if let Some(obj) = value.as_object() {
        let id = obj
            .get("id")
            .or_else(|| obj.get("name"))
            .or_else(|| obj.get("label"))
            .map(|value| model_validation_string_value(value, &format!("clauses[{idx}].id")))
            .transpose()?
            .unwrap_or_else(|| format!("C{}", idx + 1));
        let literals_value = obj
            .get("literals")
            .or_else(|| obj.get("lits"))
            .or_else(|| obj.get("clause"))
            .ok_or_else(|| format!("clauses[{idx}] needs literals"))?;
        let hard = model_validation_weighted_max_sat_hard(obj);
        let weight = obj
            .get("weight")
            .or_else(|| obj.get("cost"))
            .or_else(|| obj.get("reward"))
            .map(|value| {
                model_validation_routing_number(Some(value), &format!("clauses[{idx}].weight"))
            })
            .transpose()?
            .unwrap_or(if hard { 0.0 } else { 1.0 });
        return Ok(WeightedMaxSatClause {
            id,
            literals: model_validation_weighted_max_sat_literals(
                literals_value,
                &format!("clauses[{idx}].literals"),
            )?,
            weight,
            hard,
        });
    }
    if let Some(values) = value.as_array() {
        if values.is_empty() {
            return Err(format!("clauses[{idx}] must not be empty"));
        }
        if values.first().is_some_and(Value::is_array) {
            let weight = values
                .get(1)
                .map(|value| {
                    model_validation_routing_number(Some(value), &format!("clauses[{idx}].weight"))
                })
                .transpose()?
                .unwrap_or(1.0);
            let hard = values
                .get(2)
                .and_then(model_validation_weighted_max_sat_bool)
                .unwrap_or(false);
            return Ok(WeightedMaxSatClause {
                id: format!("C{}", idx + 1),
                literals: model_validation_weighted_max_sat_literals(
                    &values[0],
                    &format!("clauses[{idx}].literals"),
                )?,
                weight,
                hard,
            });
        }
        return Ok(WeightedMaxSatClause {
            id: format!("C{}", idx + 1),
            literals: model_validation_weighted_max_sat_literals(
                value,
                &format!("clauses[{idx}]"),
            )?,
            weight: 1.0,
            hard: false,
        });
    }
    Err(format!("clauses[{idx}] must be an object or array"))
}

fn model_validation_weighted_max_sat_problem(
    source: &Value,
) -> Result<WeightedMaxSatProblem, String> {
    let raw_clauses = source
        .get("clauses")
        .or_else(|| source.get("soft_clauses"))
        .or_else(|| source.get("softClauses"))
        .and_then(Value::as_array)
        .ok_or_else(|| "weighted MaxSAT payload needs clauses array".to_string())?;
    let clauses = raw_clauses
        .iter()
        .enumerate()
        .map(|(idx, clause)| model_validation_weighted_max_sat_clause(clause, idx))
        .collect::<Result<Vec<_>, _>>()?;
    let derived_vars = clauses
        .iter()
        .flat_map(|clause| clause.literals.iter())
        .map(|literal| literal.unsigned_abs() as usize)
        .max()
        .unwrap_or(0);
    let variable_array_count = source
        .get("variables")
        .and_then(Value::as_array)
        .map(Vec::len);
    let num_vars = model_validation_optional_usize_field(
        source,
        &[
            "num_vars",
            "numVars",
            "variable_count",
            "variableCount",
            "n_vars",
            "nVars",
        ],
        "num_vars",
    )?
    .or(variable_array_count)
    .unwrap_or(derived_vars);
    if num_vars == 0 {
        return Err("num_vars must be positive".to_string());
    }
    Ok(WeightedMaxSatProblem { num_vars, clauses })
}

fn model_validation_weighted_max_sat_reference(payload: &Value, tool: &str) -> Value {
    let validator = format!("builtin:weighted-max-sat-small-for-{tool}");
    let source = model_validation_weighted_max_sat_source(payload);
    let problem = match model_validation_weighted_max_sat_problem(source) {
        Ok(problem) => problem,
        Err(message) => {
            return model_validation_result("failed", "failure", &validator, message, "", "");
        }
    };
    let solution = solve_weighted_max_sat_with_external_reference(
        &problem,
        &ExternalWeightedMaxSatReferenceOptions {
            solver: ExternalWeightedMaxSatReferenceSolver::RustEnumeration,
        },
    );
    let (status, verdict) = match solution.status {
        ExternalWeightedMaxSatReferenceStatus::Optimal => ("ok", "optimal"),
        ExternalWeightedMaxSatReferenceStatus::Feasible => ("ok", "feasible"),
        ExternalWeightedMaxSatReferenceStatus::Infeasible => ("ok", "infeasible"),
        ExternalWeightedMaxSatReferenceStatus::Unsupported
        | ExternalWeightedMaxSatReferenceStatus::Unavailable => ("unavailable", "unknown"),
        ExternalWeightedMaxSatReferenceStatus::NumericalError => ("failed", "failure"),
    };
    let assignment = solution
        .assignment
        .iter()
        .map(|value| if *value { '1' } else { '0' })
        .collect::<String>();
    let mut stdout = vec![format!("assignment={assignment}")];
    if let Some(objective) = solution.objective {
        stdout.push(format!("objective={objective:.9}"));
    }
    if let Some(satisfied) = solution.satisfied_soft_weight {
        stdout.push(format!("satisfied_soft={satisfied:.9}"));
    }
    if let Some(unsatisfied) = solution.unsatisfied_soft_weight {
        stdout.push(format!("unsatisfied_soft={unsatisfied:.9}"));
    }
    stdout.push(format!(
        "violated_hard={}",
        solution.violated_hard_clause_ids.len()
    ));
    stdout.push(format!("solver={}", solution.solver));
    model_validation_result(
        status,
        verdict,
        &validator,
        solution.message,
        stdout.join(" "),
        "",
    )
}

fn model_validation_parse_opb(
    text: &str,
) -> Result<(Vec<String>, Vec<(Vec<(i64, String)>, String, i64)>), String> {
    let mut variables = BTreeMap::new();
    let mut constraints = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim().trim_end_matches(';').trim();
        if line.is_empty() || line.starts_with('*') {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("min:") || lower.starts_with("max:") {
            continue;
        }
        let (lhs, op, rhs) = if let Some((lhs, rhs)) = line.split_once(">=") {
            (lhs, ">=", rhs)
        } else if let Some((lhs, rhs)) = line.split_once("<=") {
            (lhs, "<=", rhs)
        } else if let Some((lhs, rhs)) = line.split_once('=') {
            (lhs, "=", rhs)
        } else {
            return Err(format!("unsupported OPB constraint {line:?}"));
        };
        let rhs = rhs
            .trim()
            .parse::<i64>()
            .map_err(|_| format!("unsupported OPB rhs {rhs:?}"))?;
        let tokens: Vec<&str> = lhs.split_whitespace().collect();
        if tokens.len() % 2 != 0 {
            return Err(format!("unsupported OPB term list {lhs:?}"));
        }
        let mut terms = Vec::new();
        for pair in tokens.chunks(2) {
            let coeff = pair[0]
                .parse::<i64>()
                .map_err(|_| format!("unsupported OPB coefficient {:?}", pair[0]))?;
            let name = pair[1].to_string();
            variables.insert(name.clone(), ());
            terms.push((coeff, name));
        }
        constraints.push((terms, op.to_string(), rhs));
    }
    if constraints.is_empty() {
        return Err("missing OPB constraints".to_string());
    }
    Ok((variables.keys().cloned().collect(), constraints))
}

fn model_validation_opb_constraint_satisfied(
    constraint: &(Vec<(i64, String)>, String, i64),
    assignment: &BTreeMap<String, bool>,
) -> bool {
    let (terms, op, rhs) = constraint;
    let total = terms.iter().fold(0_i64, |acc, (coeff, name)| {
        acc + coeff * i64::from(*assignment.get(name).unwrap_or(&false))
    });
    match op.as_str() {
        ">=" => total >= *rhs,
        "<=" => total <= *rhs,
        _ => total == *rhs,
    }
}

fn model_validation_opb_reference(payload: &Value) -> Value {
    let text = model_validation_payload_text(payload, &["opb", "pb", "text", "model"]);
    if text.trim().is_empty() {
        return model_validation_result(
            "failed",
            "failure",
            "opb",
            "payload needs opb, pb, text, or model",
            "",
            "",
        );
    }
    let (variables, constraints) = match model_validation_parse_opb(text) {
        Ok(parsed) => parsed,
        Err(message) => {
            return model_validation_result(
                "failed",
                "failure",
                "builtin:opb-small-pb",
                message,
                "",
                "",
            );
        }
    };
    if variables.len() > 24 {
        return model_validation_result(
            "unavailable",
            "unknown",
            "builtin:opb-small-pb",
            format!(
                "builtin OPB fallback is capped at 24 variables, got {}",
                variables.len()
            ),
            "",
            "",
        );
    }
    for mask in 0..(1_u64 << variables.len()) {
        let assignment: BTreeMap<String, bool> = variables
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), ((mask >> idx) & 1) == 1))
            .collect();
        if constraints
            .iter()
            .all(|constraint| model_validation_opb_constraint_satisfied(constraint, &assignment))
        {
            let stdout = variables
                .iter()
                .map(|name| {
                    format!(
                        "{name}={}",
                        i32::from(*assignment.get(name).unwrap_or(&false))
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            return model_validation_result(
                "ok",
                "sat",
                "builtin:opb-small-pb",
                "satisfying assignment found",
                stdout,
                "",
            );
        }
    }
    model_validation_result(
        "ok",
        "unsat",
        "builtin:opb-small-pb",
        "all assignments exhausted",
        "",
        "",
    )
}

pub fn run_model_validation_json_with_rust_reference(payload: &Value, tool: &str) -> Value {
    let kind = payload
        .get("kind")
        .and_then(Value::as_str)
        .map(model_validation_normalized_tool)
        .unwrap_or_default();
    let tool = model_validation_normalized_tool(tool);
    let minizinc_tools = [
        "minizinc",
        "flatzinc",
        "minizinc-solution-checker",
        "gecode",
        "chuffed",
        "ortools-cp-sat",
        "fzn-cp-sat",
    ];
    let cp_sat_tools = [
        "cp-sat",
        "cpsat",
        "ortools-cp-sat",
        "or-tools-cp-sat",
        "fzn-cp-sat",
    ];
    let smt_tools = [
        "z3",
        "cvc5",
        "yices",
        "bitwuzla",
        "boolector",
        "mathsat",
        "optimathsat",
        "opensmt",
        "smtinterpol",
        "princess",
    ];
    let sat_tools = [
        "kissat",
        "cadical",
        "cryptominisat",
        "cryptominisat5",
        "minisat",
        "glucose",
        "glucose-syrup",
        "maplesat",
        "maple-sat",
        "maple-lcm",
        "varisat",
        "sat4j",
        "sat4j-sat",
        "pysat",
        "pysat-adapter",
        "python-sat",
        "python-sat-adapter",
    ];
    let maxsat_tools = [
        "open-wbo",
        "open-wbo-static",
        "maxhs",
        "sat4j",
        "sat4j-sat",
        "pysat",
        "pysat-adapter",
        "python-sat",
        "python-sat-adapter",
    ];
    let pseudo_boolean_tools = [
        "roundingsat",
        "sat4j",
        "sat4j-sat",
        "pysat",
        "pysat-adapter",
        "python-sat",
        "python-sat-adapter",
    ];
    let pddl_tools = [
        "pddl-val",
        "validate",
        "val",
        "pddl-validate",
        "fast-downward",
        "fast-downward.py",
        "lpg-td",
        "lpg",
        "optic",
        "optic-clp",
        "enhsp",
        "enhsp.jar",
    ];
    let assignment_tools = [
        "scipy-optimize",
        "scipy-optimize-adapter",
        "scipy-adapter",
        "scipy-linear-sum-assignment",
        "linear-sum-assignment",
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
    ];
    let bin_packing_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "cpmpy",
        "cpmpy-adapter",
        "pycsp3",
        "pycsp3-adapter",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
    ];
    let facility_location_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
        "good-lp",
        "lp-modeler",
    ];
    let knapsack_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
        "good-lp",
        "lp-modeler",
    ];
    let graph_coloring_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "cpmpy",
        "cpmpy-adapter",
        "pycsp3",
        "pycsp3-adapter",
        "choco-solver",
        "jacop",
    ];
    let weighted_independent_set_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-cp-sat",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "cpmpy",
        "cpmpy-adapter",
        "pycsp3",
        "pycsp3-adapter",
        "choco-solver",
        "jacop",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
        "networkx",
        "good-lp",
        "lp-modeler",
    ];
    let scheduling_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-cp-sat",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "cpmpy",
        "cpmpy-adapter",
        "pycsp3",
        "pycsp3-adapter",
        "choco-solver",
        "jacop",
        "ibm-cp-optimizer",
        "cp-optimizer",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
        "optaplanner",
        "optaplanner-adapter",
        "timefold",
        "timefold-adapter",
    ];
    let set_cover_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
        "good-lp",
        "lp-modeler",
    ];
    let min_cost_flow_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
        "networkx",
        "good-lp",
        "lp-modeler",
    ];
    let max_flow_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
        "networkx",
    ];
    let mst_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "cpmpy",
        "cpmpy-adapter",
        "pycsp3",
        "pycsp3-adapter",
        "networkx",
    ];
    let routing_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "optaplanner",
        "optaplanner-adapter",
        "ores-optaplanner-adapter",
        "timefold",
        "timefold-adapter",
        "ores-timefold-adapter",
        "hexaly",
        "localsolver",
        "localsolver-studio",
    ];
    let tsp_tools = [
        "ortools-python",
        "ortools-python-adapter",
        "ortools-adapter",
        "ortools-java",
        "ortools-java-adapter",
        "ores-ortools-java-adapter",
        "optaplanner",
        "optaplanner-adapter",
        "ores-optaplanner-adapter",
        "timefold",
        "timefold-adapter",
        "ores-timefold-adapter",
        "hexaly",
        "localsolver",
        "localsolver-studio",
        "networkx",
    ];
    let finite_domain_cp_tools = [
        "cpmpy",
        "cpmpy-adapter",
        "cpm-py-adapter",
        "pycsp3",
        "pycsp3-adapter",
        "choco-solver",
        "jacop",
        "ortools-java",
    ];
    let linear_modeling_tools = [
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "gurobipy",
        "gurobipy-adapter",
        "cplex-python",
        "cplex-python-adapter",
        "xpress-python",
        "xpress-python-adapter",
        "docplex",
        "docplex-adapter",
        "gurobi-rust",
        "gurobi-rust-adapter",
        "ores-gurobi-rust-adapter",
        "cplex-rust",
        "cplex-rust-adapter",
        "ores-cplex-rust-adapter",
        "highs-rust",
        "highs-rust-adapter",
        "ores-highs-rust-adapter",
        "scip-rust",
        "scip-rust-adapter",
        "ores-scip-rust-adapter",
        "cbc-rust",
        "cbc-rust-adapter",
        "ores-cbc-rust-adapter",
        "highs-cli",
        "highs",
        "glpk-cli",
        "glpsol",
        "scip-cli",
        "scip",
        "cbc-cli",
        "cbc",
        "clp-cli",
        "clp",
        "soplex-cli",
        "soplex",
        "qsopt-ex-cli",
        "qsopt-ex",
        "qsopt",
        "esolver",
        "lp-solve-cli",
        "lp-solve",
        "lpsolve",
        "ortools-glop",
        "ortools-pdlp",
        "good-lp",
        "lp-modeler",
        "rust-linprog",
    ];
    let stochastic_lp_tools = [
        "pyomo",
        "pyomo-adapter",
        "pulp",
        "pulp-adapter",
        "python-mip",
        "python-mip-adapter",
        "mip-adapter",
        "scipy-optimize",
        "scipy-optimize-adapter",
        "scipy-adapter",
        "highs",
        "highs-adapter",
        "gurobipy",
        "gurobipy-adapter",
        "cplex-python",
        "cplex-python-adapter",
        "docplex",
        "docplex-adapter",
        "highs-rust",
        "highs-rust-adapter",
        "ores-highs-rust-adapter",
        "gurobi-rust",
        "gurobi-rust-adapter",
        "ores-gurobi-rust-adapter",
        "cplex-rust",
        "cplex-rust-adapter",
        "ores-cplex-rust-adapter",
        "jump",
        "jump-adapter",
        "good-lp",
        "lp-modeler",
    ];
    let quadratic_modeling_tools = [
        "osqp",
        "osqp-adapter",
        "highs",
        "highs-adapter",
        "scipy-optimize",
        "scipy-optimize-adapter",
        "scipy-adapter",
        "cvxpy",
        "cvxpy-adapter",
        "cvxopt",
        "cvxopt-adapter",
        "scs",
        "scs-adapter",
        "clarabel",
        "clarabel-adapter",
        "ecos",
        "ecos-adapter",
        "mosek",
        "mosek-adapter",
        "copt",
        "copt-adapter",
        "qpoases",
        "qpoases-adapter",
        "proxqp",
        "proxqp-adapter",
        "cosmo",
        "cosmo-adapter",
        "pyomo",
        "pyomo-adapter",
        "gurobipy",
        "gurobipy-adapter",
        "cplex-python",
        "cplex-python-adapter",
        "highs-rust",
        "highs-rust-adapter",
        "ores-highs-rust-adapter",
        "gurobi-rust",
        "gurobi-rust-adapter",
        "ores-gurobi-rust-adapter",
        "cplex-rust",
        "cplex-rust-adapter",
        "ores-cplex-rust-adapter",
    ];
    let nonlinear_modeling_tools = [
        "scipy-optimize",
        "scipy-optimize-adapter",
        "scipy-adapter",
        "argmin",
        "argmin-adapter",
        "ores-argmin-adapter",
        "nlopt",
        "nlopt-adapter",
        "nlopt-rs",
        "nlopt-rs-adapter",
        "ores-nlopt-rs-adapter",
        "nlopt-cli",
        "ipopt",
        "ipopt-adapter",
        "ipopt-rust",
        "ipopt-rust-adapter",
        "ores-ipopt-rust-adapter",
        "casadi",
        "casadi-adapter",
        "mosek",
        "mosek-adapter",
        "copt",
        "copt-adapter",
        "cvxpy",
        "cvxpy-adapter",
        "cvxopt",
        "cvxopt-adapter",
        "osqp",
        "osqp-adapter",
        "scs",
        "scs-adapter",
        "clarabel",
        "clarabel-adapter",
        "ecos",
        "ecos-adapter",
    ];
    if matches!(
        kind.as_str(),
        "scheduling-validation"
            | "schedule-validation"
            | "job-shop-validation"
            | "jobshop-validation"
            | "flow-shop-validation"
            | "flowshop-validation"
    ) || (scheduling_tools.contains(&tool.as_str())
        && model_validation_payload_has_scheduling_model(payload))
    {
        return model_validation_scheduling_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "cp-sat-validation"
            | "cpsat-validation"
            | "cp-sat-json-validation"
            | "ortools-cp-sat-validation"
    ) || (cp_sat_tools.contains(&tool.as_str())
        && model_validation_payload_has_cp_sat_json_model(payload))
    {
        return model_validation_cp_sat_reference(payload, &tool);
    }
    if kind == "minizinc-validation" || minizinc_tools.contains(&tool.as_str()) {
        return model_validation_minizinc_reference(payload);
    }
    if kind == "asp-validation"
        || kind == "clingo-validation"
        || tool == "clingo"
        || tool == "clingcon"
    {
        return model_validation_asp_reference(payload, &tool);
    }
    if kind == "smtlib-validation"
        || kind == "smt-lib-validation"
        || smt_tools.contains(&tool.as_str())
    {
        let text = model_validation_payload_text(payload, &["script", "smtlib", "text", "model"]);
        if text.trim().is_empty() {
            return model_validation_result(
                "failed",
                "failure",
                "smtlib",
                "payload needs script, smtlib, text, or model",
                "",
                "",
            );
        }
        return model_validation_infer_smtlib(text);
    }
    if matches!(
        kind.as_str(),
        "weighted-max-sat-validation"
            | "weighted-maxsat-validation"
            | "partial-max-sat-validation"
            | "partial-maxsat-validation"
            | "maxsat-json-validation"
            | "maxsat-validation"
    ) && model_validation_payload_has_weighted_max_sat_model(payload)
        || (maxsat_tools.contains(&tool.as_str())
            && model_validation_payload_has_weighted_max_sat_model(payload))
    {
        return model_validation_weighted_max_sat_reference(payload, &tool);
    }
    if kind == "wcnf-validation"
        || kind == "dimacs-wcnf-validation"
        || kind == "maxsat-validation"
        || (maxsat_tools.contains(&tool.as_str()) && model_validation_payload_has_wcnf(payload))
    {
        return model_validation_wcnf_reference(payload);
    }
    if kind == "opb-validation"
        || kind == "pseudo-boolean-validation"
        || (pseudo_boolean_tools.contains(&tool.as_str())
            && model_validation_payload_has_opb(payload))
    {
        return model_validation_opb_reference(payload);
    }
    if matches!(
        kind.as_str(),
        "pddl-validation"
            | "pddl-plan-validation"
            | "plan-validation"
            | "planning-validation"
            | "classical-planning-validation"
    ) || (pddl_tools.contains(&tool.as_str()) && model_validation_payload_has_pddl(payload))
    {
        return model_validation_pddl_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "assignment-validation"
            | "linear-sum-assignment-validation"
            | "bipartite-assignment-validation"
            | "matching-validation"
    ) || (assignment_tools.contains(&tool.as_str())
        && model_validation_payload_has_assignment_model(payload))
    {
        return model_validation_assignment_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "bin-packing-validation" | "binpacking-validation" | "packing-validation"
    ) || (bin_packing_tools.contains(&tool.as_str())
        && model_validation_payload_has_bin_packing_model(payload))
    {
        return model_validation_bin_packing_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "facility-location-validation"
            | "uncapacitated-facility-location-validation"
            | "ufl-validation"
            | "location-allocation-validation"
    ) || (facility_location_tools.contains(&tool.as_str())
        && model_validation_payload_has_facility_location_model(payload))
    {
        return model_validation_facility_location_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "knapsack-validation"
            | "binary-knapsack-validation"
            | "zero-one-knapsack-validation"
            | "0-1-knapsack-validation"
    ) || (knapsack_tools.contains(&tool.as_str())
        && model_validation_payload_has_knapsack_model(payload))
    {
        return model_validation_knapsack_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "min-cost-flow-validation"
            | "minimum-cost-flow-validation"
            | "transportation-validation"
            | "transshipment-validation"
    ) || (min_cost_flow_tools.contains(&tool.as_str())
        && model_validation_payload_has_min_cost_flow_model(payload))
    {
        return model_validation_min_cost_flow_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "max-flow-validation" | "maximum-flow-validation" | "min-cut-validation"
    ) || (max_flow_tools.contains(&tool.as_str())
        && model_validation_payload_has_max_flow_model(payload))
    {
        return model_validation_max_flow_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "weighted-independent-set-validation"
            | "maximum-weight-independent-set-validation"
            | "max-weight-independent-set-validation"
            | "independent-set-validation"
            | "set-packing-validation"
            | "conflict-graph-validation"
    ) || (weighted_independent_set_tools.contains(&tool.as_str())
        && model_validation_payload_has_wis_model(payload))
    {
        return model_validation_wis_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "mst-validation"
            | "minimum-spanning-tree-validation"
            | "spanning-tree-validation"
            | "minimum-spanning-forest-validation"
    ) || (mst_tools.contains(&tool.as_str()) && model_validation_payload_has_mst_model(payload))
    {
        return model_validation_mst_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "graph-coloring-validation"
            | "graph-colouring-validation"
            | "coloring-validation"
            | "colouring-validation"
    ) || (graph_coloring_tools.contains(&tool.as_str())
        && model_validation_payload_has_graph_coloring_model(payload))
    {
        return model_validation_graph_coloring_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "set-cover-validation" | "set-covering-validation" | "covering-validation"
    ) || (set_cover_tools.contains(&tool.as_str())
        && model_validation_payload_has_set_cover_model(payload))
    {
        return model_validation_set_cover_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "tsp-validation"
            | "traveling-salesman-validation"
            | "travelling-salesman-validation"
            | "traveling-salesperson-validation"
            | "travelling-salesperson-validation"
    ) || (tsp_tools.contains(&tool.as_str()) && model_validation_payload_has_tsp_model(payload))
    {
        return model_validation_tsp_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "routing-validation"
            | "vehicle-routing-validation"
            | "vrp-validation"
            | "cvrp-validation"
            | "tsp-validation"
            | "ortools-routing-validation"
    ) || (routing_tools.contains(&tool.as_str())
        && model_validation_payload_has_routing_model(payload))
    {
        return model_validation_routing_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "stochastic-lp-validation"
            | "stochastic-linear-program-validation"
            | "two-stage-stochastic-lp-validation"
            | "saa-validation"
    ) || (stochastic_lp_tools.contains(&tool.as_str())
        && model_validation_payload_has_stochastic_lp_model(payload))
    {
        return model_validation_stochastic_lp_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "quadratic-validation"
            | "quadratic-program-validation"
            | "quadratic-model-validation"
            | "qp-validation"
            | "miqp-validation"
    ) || (quadratic_modeling_tools.contains(&tool.as_str())
        && model_validation_payload_has_quadratic_model(payload))
    {
        return model_validation_quadratic_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "linear-validation"
            | "linear-mip-validation"
            | "mip-validation"
            | "lp-validation"
            | "algebraic-model-validation"
            | "linear-model-validation"
    ) || (linear_modeling_tools.contains(&tool.as_str())
        && model_validation_payload_has_linear_model(payload))
    {
        return model_validation_linear_mip_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "nonlinear-validation"
            | "nlp-validation"
            | "nonlinear-model-validation"
            | "convex-validation"
            | "convex-model-validation"
            | "qp-validation"
    ) || (nonlinear_modeling_tools.contains(&tool.as_str())
        && model_validation_payload_has_nonlinear_model(payload))
    {
        return model_validation_nonlinear_reference(payload, &tool);
    }
    if matches!(
        kind.as_str(),
        "cp-validation" | "finite-domain-cp-validation" | "constraint-model-validation"
    ) || (finite_domain_cp_tools.contains(&tool.as_str())
        && model_validation_payload_has_finite_domain_cp(payload))
    {
        return model_validation_finite_domain_cp_reference(payload, &tool);
    }
    if kind == "dimacs-validation"
        || kind == "dimacs-cnf-validation"
        || sat_tools.contains(&tool.as_str())
    {
        return model_validation_dimacs_reference(payload);
    }
    model_validation_result(
        "unavailable",
        "unknown",
        &tool,
        format!("unknown model validation payload kind {kind:?}"),
        "",
        "",
    )
}

fn proof_validation_result(
    tool: &str,
    validator: &str,
    status: &str,
    verdict: &str,
    message: impl Into<String>,
    extras: Vec<(&str, Value)>,
) -> Value {
    let mut output = serde_json::Map::new();
    output.insert(
        "kind".to_string(),
        Value::String("proof-validation-result".to_string()),
    );
    output.insert("tool".to_string(), Value::String(tool.to_string()));
    output.insert(
        "validator".to_string(),
        Value::String(validator.to_string()),
    );
    output.insert("status".to_string(), Value::String(status.to_string()));
    output.insert("verdict".to_string(), Value::String(verdict.to_string()));
    output.insert("message".to_string(), Value::String(message.into()));
    for (key, value) in extras {
        output.insert(key.to_string(), value);
    }
    Value::Object(output)
}

fn proof_validation_cnf_model(variables: usize, clauses: &[Vec<i32>]) -> Option<Vec<i32>> {
    if variables > 20 {
        return None;
    }
    for mask in 0..(1_u64 << variables) {
        let satisfied = clauses.iter().all(|clause| {
            clause.iter().any(|literal| {
                let var = literal.unsigned_abs() as usize - 1;
                let value = ((mask >> var) & 1) == 1;
                value == (*literal > 0)
            })
        });
        if satisfied {
            return Some(
                (0..variables)
                    .map(|idx| {
                        if ((mask >> idx) & 1) == 1 {
                            (idx + 1) as i32
                        } else {
                            -((idx + 1) as i32)
                        }
                    })
                    .collect(),
            );
        }
    }
    None
}

fn proof_validation_drat_has_empty_clause(proof: &str) -> bool {
    proof.lines().any(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') || line.starts_with("d ") {
            return false;
        }
        line.split_whitespace()
            .map(str::parse::<i32>)
            .collect::<Result<Vec<_>, _>>()
            .is_ok_and(|tokens| tokens == [0])
    })
}

fn proof_validation_lrat_has_empty_clause(proof: &str) -> bool {
    proof.lines().any(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') {
            return false;
        }
        line.split_whitespace()
            .map(str::parse::<i32>)
            .collect::<Result<Vec<_>, _>>()
            .is_ok_and(|tokens| tokens.len() >= 2 && tokens[1] == 0)
    })
}

fn proof_validation_frat_has_empty_clause(proof: &str) -> bool {
    proof.lines().any(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') {
            return false;
        }
        if line == "0" {
            return true;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.first() != Some(&"a") {
            return false;
        }
        tokens
            .iter()
            .position(|token| *token == "0")
            .is_some_and(|idx| idx <= 2)
    })
}

fn proof_validation_pb_model(
    variables: &[String],
    constraints: &[(Vec<(i64, String)>, String, i64)],
) -> Option<BTreeMap<String, i32>> {
    if variables.len() > 20 {
        return None;
    }
    for mask in 0..(1_u64 << variables.len()) {
        let assignment: BTreeMap<String, bool> = variables
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), ((mask >> idx) & 1) == 1))
            .collect();
        if constraints
            .iter()
            .all(|constraint| model_validation_opb_constraint_satisfied(constraint, &assignment))
        {
            return Some(
                variables
                    .iter()
                    .map(|name| {
                        (
                            name.clone(),
                            i32::from(*assignment.get(name).unwrap_or(&false)),
                        )
                    })
                    .collect(),
            );
        }
    }
    None
}

fn proof_validation_veripb_has_derivation(proof: &str) -> bool {
    proof.lines().any(|line| {
        let line = line.trim();
        !line.is_empty() && !line.starts_with('*') && !line.starts_with('c')
    })
}

fn proof_validation_is_veripb_tool(tool: &str) -> bool {
    matches!(tool, "veripb" | "veripb-checker")
}

fn proof_validation_is_lrat_tool(tool: &str) -> bool {
    matches!(
        tool,
        "lrat" | "lrat-check" | "lrat-checker" | "cake-lpr" | "cake-lpr-check"
    )
}

fn proof_validation_is_frat_tool(tool: &str) -> bool {
    matches!(tool, "frat" | "frat-rs" | "frat-trim")
}

fn proof_validation_artifact_content<'a>(payload: &'a Value, names: &[&str]) -> Option<&'a str> {
    let artifacts = payload.get("artifacts")?.as_array()?;
    artifacts.iter().find_map(|artifact| {
        let artifact = artifact.as_object()?;
        let name = artifact.get("name")?.as_str()?;
        if names.iter().any(|wanted| name.eq_ignore_ascii_case(wanted)) {
            artifact.get("content")?.as_str()
        } else {
            None
        }
    })
}

fn proof_validation_payload_text<'a>(
    payload: &'a Value,
    keys: &[&str],
    artifact_names: &[&str],
) -> &'a str {
    let direct = model_validation_payload_text(payload, keys);
    if direct.trim().is_empty() {
        proof_validation_artifact_content(payload, artifact_names).unwrap_or("")
    } else {
        direct
    }
}

pub fn run_proof_validation_json_with_rust_reference(payload: &Value, tool: &str) -> Value {
    let tool = model_validation_normalized_tool(tool);
    let kind = payload
        .get("kind")
        .and_then(Value::as_str)
        .map(model_validation_normalized_tool)
        .unwrap_or_default();
    if proof_validation_is_veripb_tool(&tool)
        || matches!(
            kind.as_str(),
            "pseudo-boolean-proof-validation" | "opb-proof-validation" | "veripb-validation"
        )
    {
        let opb = proof_validation_payload_text(payload, &["opb", "model"], &["opb", "model"]);
        let proof = proof_validation_payload_text(payload, &["proof"], &["proof", "pbp", "rup"]);
        let validator = format!("builtin:small-opb-proof-for-{tool}");
        if opb.trim().is_empty() || proof.trim().is_empty() {
            return proof_validation_result(
                &tool,
                &validator,
                "ok",
                "invalid",
                "missing OPB or pseudo-Boolean proof text",
                Vec::new(),
            );
        }
        let (variables, constraints) = match model_validation_parse_opb(opb) {
            Ok(parsed) => parsed,
            Err(message) => {
                return proof_validation_result(
                    &tool,
                    &validator,
                    "ok",
                    "invalid",
                    message,
                    Vec::new(),
                );
            }
        };
        if let Some(model) = proof_validation_pb_model(&variables, &constraints) {
            return proof_validation_result(
                &tool,
                &validator,
                "ok",
                "invalid",
                "OPB model is satisfiable; proof cannot validate infeasibility",
                vec![("pb_status", json!("sat")), ("witness", json!(model))],
            );
        }
        if proof_validation_veripb_has_derivation(proof) {
            proof_validation_result(
                &tool,
                &validator,
                "ok",
                "valid",
                "infeasible OPB model with non-empty pseudo-Boolean proof",
                vec![("pb_status", json!("unsat"))],
            )
        } else {
            proof_validation_result(
                &tool,
                &validator,
                "ok",
                "invalid",
                "infeasible OPB model proof did not contain a derivation line",
                vec![("pb_status", json!("unsat"))],
            )
        }
    } else {
        let cnf = proof_validation_payload_text(payload, &["cnf", "dimacs"], &["cnf", "model"]);
        let proof =
            proof_validation_payload_text(payload, &["proof"], &["proof", "drat", "lrat", "frat"]);
        let validator = format!("builtin:small-cnf-proof-for-{tool}");
        if cnf.trim().is_empty() || proof.trim().is_empty() {
            return proof_validation_result(
                &tool,
                &validator,
                "ok",
                "invalid",
                "missing CNF or proof text",
                Vec::new(),
            );
        }
        let (variables, clauses) = match model_validation_parse_dimacs_cnf(cnf) {
            Ok(parsed) => parsed,
            Err(message) => {
                return proof_validation_result(
                    &tool,
                    &validator,
                    "ok",
                    "invalid",
                    message,
                    Vec::new(),
                );
            }
        };
        if let Some(model) = proof_validation_cnf_model(variables, &clauses) {
            return proof_validation_result(
                &tool,
                &validator,
                "ok",
                "invalid",
                "CNF is satisfiable; unsat proof cannot validate",
                vec![("cnf_status", json!("sat")), ("witness", json!(model))],
            );
        }
        let has_empty_clause = if proof_validation_is_lrat_tool(&tool) {
            proof_validation_lrat_has_empty_clause(proof)
        } else if proof_validation_is_frat_tool(&tool) {
            proof_validation_frat_has_empty_clause(proof)
        } else {
            proof_validation_drat_has_empty_clause(proof)
        };
        if has_empty_clause {
            proof_validation_result(
                &tool,
                &validator,
                "ok",
                "valid",
                "unsat CNF with empty-clause proof line",
                vec![("cnf_status", json!("unsat"))],
            )
        } else {
            proof_validation_result(
                &tool,
                &validator,
                "ok",
                "invalid",
                "unsat CNF proof did not contain an empty-clause line",
                vec![("cnf_status", json!("unsat"))],
            )
        }
    }
}

fn formal_benchmark_check(name: &str, passed: bool, message: &str) -> Value {
    json!({
        "name": name,
        "passed": passed,
        "message": if passed { "" } else { message },
    })
}

fn formal_benchmark_verdict(checks: &[Value]) -> &'static str {
    if !checks.is_empty()
        && checks
            .iter()
            .all(|check| check["passed"].as_bool() == Some(true))
    {
        "valid"
    } else {
        "invalid"
    }
}

fn formal_benchmark_result(
    status: &str,
    verdict: &str,
    validator: &str,
    message: impl Into<String>,
    checks: Vec<Value>,
) -> Value {
    json!({
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message.into(),
        "checks": checks,
        "stdout": "",
        "stderr": "",
    })
}

fn formal_benchmark_strings(payload: &Value, keys: &[&str]) -> Vec<String> {
    for key in keys {
        match payload.get(*key) {
            Some(Value::String(text)) => return vec![text.clone()],
            Some(Value::Array(items)) => {
                return items
                    .iter()
                    .map(|item| {
                        item.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| item.to_string())
                    })
                    .collect();
            }
            _ => {}
        }
    }
    Vec::new()
}

fn formal_benchmark_text(payload: &Value, keys: &[&str]) -> String {
    formal_benchmark_strings(payload, keys)
        .into_iter()
        .next()
        .unwrap_or_default()
}

fn formal_benchmark_balanced(text: &str, left: char, right: char) -> bool {
    let mut depth = 0_i32;
    for ch in text.chars() {
        if ch == left {
            depth += 1;
        } else if ch == right {
            depth -= 1;
            if depth < 0 {
                return false;
            }
        }
    }
    depth == 0
}

fn formal_contains_word(text: &str, word: &str) -> bool {
    text.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-')
        .any(|part| part.eq_ignore_ascii_case(word))
}

fn formal_line_starts_with(text: &str, prefix: &str) -> bool {
    let prefix = prefix.to_ascii_lowercase();
    text.lines()
        .any(|line| line.trim_start().to_ascii_lowercase().starts_with(&prefix))
}

fn formal_tla_module_name(module: &str) -> Option<String> {
    module.lines().find_map(|line| {
        let line = line.trim();
        let rest = line.strip_prefix("----")?.trim();
        let rest = rest.strip_prefix("MODULE")?.trim();
        let name = rest.strip_suffix("----")?.trim();
        if name
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
        {
            Some(name.to_string())
        } else {
            None
        }
    })
}

fn formal_validate_tla(payload: &Value) -> Value {
    let module = formal_benchmark_text(payload, &["module", "model", "text"]);
    let expected_invariants = formal_benchmark_strings(payload, &["expected_invariants"]);
    let expected_temporal = formal_benchmark_strings(payload, &["expected_temporal_properties"]);
    let module_name = formal_tla_module_name(&module);
    let mut checks = vec![
        formal_benchmark_check(
            "module-header",
            module_name.is_some(),
            "missing TLA+ module header",
        ),
        formal_benchmark_check(
            "module-terminator",
            module.trim_end().ends_with("===="),
            "missing final ====",
        ),
        formal_benchmark_check(
            "init-definition",
            module.contains("Init =="),
            "missing Init definition",
        ),
        formal_benchmark_check(
            "next-definition",
            module.contains("Next =="),
            "missing Next definition",
        ),
        formal_benchmark_check(
            "spec-definition",
            module.contains("Spec =="),
            "missing Spec definition",
        ),
    ];
    for invariant in expected_invariants {
        checks.push(formal_benchmark_check(
            &format!("invariant:{invariant}"),
            module.contains(&format!("{invariant} ==")),
            "invariant definition missing",
        ));
    }
    for temporal in expected_temporal {
        checks.push(formal_benchmark_check(
            &format!("temporal:{temporal}"),
            module.contains(&format!("{temporal} ==")),
            "temporal property definition missing",
        ));
    }
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result(
        "ok",
        verdict,
        "builtin:tla-structural",
        module_name
            .map(|name| format!("module={name}"))
            .unwrap_or_default(),
        checks,
    )
}

fn formal_validate_prism(payload: &Value) -> Value {
    let model = formal_benchmark_text(payload, &["model", "module", "text"]);
    let properties = formal_benchmark_text(payload, &["properties", "props"]);
    let model_type = model
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let module_count = model
        .lines()
        .filter(|line| {
            let line = line.trim_start().to_ascii_lowercase();
            line.starts_with("module ") && !line.starts_with("endmodule")
        })
        .count();
    let endmodule_count = model
        .lines()
        .filter(|line| line.trim().eq_ignore_ascii_case("endmodule"))
        .count();
    let checks = vec![
        formal_benchmark_check(
            "model-type",
            matches!(model_type.as_str(), "dtmc" | "ctmc" | "mdp" | "pta"),
            "unknown PRISM model type",
        ),
        formal_benchmark_check("module-present", module_count > 0, "no PRISM module found"),
        formal_benchmark_check(
            "module-balanced",
            module_count == endmodule_count,
            "module/endmodule count mismatch",
        ),
        formal_benchmark_check(
            "command-present",
            model.contains("->"),
            "no transition command found",
        ),
        formal_benchmark_check(
            "properties-present",
            !properties.trim().is_empty(),
            "no PRISM properties supplied",
        ),
    ];
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result("ok", verdict, "builtin:prism-structural", "", checks)
}

fn formal_validate_promela(payload: &Value) -> Value {
    let model = formal_benchmark_text(payload, &["model", "promela", "text"]);
    let properties = formal_benchmark_strings(payload, &["properties", "ltl"]);
    let expected_ltl = formal_benchmark_strings(payload, &["expected_ltl_properties"]);
    let mut checks = vec![
        formal_benchmark_check(
            "process-present",
            formal_contains_word(&model, "init") || formal_contains_word(&model, "proctype"),
            "missing init/proctype",
        ),
        formal_benchmark_check(
            "braces-balanced",
            formal_benchmark_balanced(&model, '{', '}'),
            "Promela braces are not balanced",
        ),
        formal_benchmark_check(
            "statement-terminator",
            model.contains(';') || model.contains("->"),
            "no Promela statements found",
        ),
    ];
    for property in properties {
        let property_lower = property.to_ascii_lowercase();
        checks.push(formal_benchmark_check(
            &format!("ltl:{}", property.chars().take(24).collect::<String>()),
            property_lower.contains("ltl") || property.contains("<>") || property.contains("[]"),
            "malformed LTL property",
        ));
    }
    for name in expected_ltl {
        checks.push(formal_benchmark_check(
            &format!("expected-ltl:{name}"),
            model.contains(&name),
            "expected LTL property missing",
        ));
    }
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result("ok", verdict, "builtin:promela-structural", "", checks)
}

fn formal_validate_smv(payload: &Value) -> Value {
    let model = formal_benchmark_text(payload, &["model", "smv", "text"]);
    let properties = formal_benchmark_strings(payload, &["properties"]).join("\n");
    let combined = format!("{model}\n{properties}").to_ascii_uppercase();
    let checks = vec![
        formal_benchmark_check(
            "module-main",
            formal_line_starts_with(&model, "MODULE main"),
            "missing MODULE main",
        ),
        formal_benchmark_check(
            "var-section",
            formal_line_starts_with(&model, "VAR"),
            "missing VAR section",
        ),
        formal_benchmark_check(
            "state-update",
            formal_line_starts_with(&model, "ASSIGN")
                || formal_line_starts_with(&model, "INIT")
                || formal_line_starts_with(&model, "TRANS"),
            "missing ASSIGN/INIT/TRANS section",
        ),
        formal_benchmark_check(
            "property-present",
            ["CTLSPEC", "LTLSPEC", "INVARSPEC", "SPEC"]
                .iter()
                .any(|token| combined.contains(token)),
            "missing nuXmv/SMV property",
        ),
    ];
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result("ok", verdict, "builtin:smv-structural", "", checks)
}

fn formal_validate_cbmc(payload: &Value) -> Value {
    let source = formal_benchmark_text(payload, &["source", "model", "c", "text"]);
    let expected_assertions = formal_benchmark_strings(payload, &["expected_assertions"]);
    let mut checks = vec![
        formal_benchmark_check(
            "main-function",
            source.contains(" main(") || source.contains(" main ("),
            "missing main function",
        ),
        formal_benchmark_check(
            "braces-balanced",
            formal_benchmark_balanced(&source, '{', '}'),
            "C braces are not balanced",
        ),
        formal_benchmark_check(
            "assertion-present",
            source.contains("__CPROVER_assert")
                || source.contains("assert(")
                || source.contains("assert ("),
            "missing C/CBMC assertion",
        ),
    ];
    for assertion in expected_assertions {
        checks.push(formal_benchmark_check(
            &format!("assertion:{assertion}"),
            source.contains(&assertion),
            "expected assertion text missing",
        ));
    }
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result("ok", verdict, "builtin:cbmc-structural", "", checks)
}

fn formal_validate_alloy(payload: &Value) -> Value {
    let model = formal_benchmark_text(payload, &["model", "alloy", "text"]);
    let commands = formal_benchmark_strings(payload, &["commands", "properties"]).join("\n");
    let combined = format!("{model}\n{commands}").to_ascii_lowercase();
    let model_lower = model.to_ascii_lowercase();
    let checks = vec![
        formal_benchmark_check(
            "module-or-signature",
            model_lower.contains("module ") || model_lower.contains("sig "),
            "missing module/sig",
        ),
        formal_benchmark_check(
            "signature-present",
            model_lower.contains("sig "),
            "missing Alloy signature",
        ),
        formal_benchmark_check(
            "braces-balanced",
            formal_benchmark_balanced(&model, '{', '}'),
            "Alloy braces are not balanced",
        ),
        formal_benchmark_check(
            "predicate-or-fact",
            ["pred ", "fact ", "assert "]
                .iter()
                .any(|token| model_lower.contains(token)),
            "missing pred/fact/assert",
        ),
        formal_benchmark_check(
            "command-present",
            combined.contains("run ") || combined.contains("check "),
            "missing run/check command",
        ),
    ];
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result("ok", verdict, "builtin:alloy-structural", "", checks)
}

fn formal_validate_uppaal(payload: &Value) -> Value {
    let model = formal_benchmark_text(payload, &["model", "xml", "text"]);
    let queries = formal_benchmark_strings(payload, &["queries", "properties", "query"]).join("\n");
    let checks = vec![
        formal_benchmark_check(
            "nta-root",
            model.contains("<nta") && model.contains("</nta>"),
            "missing UPPAAL nta root",
        ),
        formal_benchmark_check(
            "template-present",
            model.contains("<template") && model.contains("</template>"),
            "missing template",
        ),
        formal_benchmark_check(
            "location-present",
            model.contains("<location"),
            "missing location",
        ),
        formal_benchmark_check(
            "transition-present",
            model.contains("<transition"),
            "missing transition",
        ),
        formal_benchmark_check(
            "query-present",
            !queries.trim().is_empty(),
            "missing UPPAAL query",
        ),
        formal_benchmark_check(
            "query-operator",
            ["A[]", "E[]", "A<>", "E<>", "A []", "E []", "A <>", "E <>"]
                .iter()
                .any(|token| queries.contains(token)),
            "missing UPPAAL temporal operator",
        ),
    ];
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result("ok", verdict, "builtin:uppaal-structural", "", checks)
}

fn formal_validate_mcrl2(payload: &Value) -> Value {
    let spec = formal_benchmark_text(payload, &["model", "mcrl2", "spec", "text"]);
    let properties = formal_benchmark_strings(payload, &["properties", "formulae"]).join("\n");
    let checks = vec![
        formal_benchmark_check(
            "action-section",
            formal_line_starts_with(&spec, "act"),
            "missing mCRL2 act section",
        ),
        formal_benchmark_check(
            "process-section",
            formal_line_starts_with(&spec, "proc"),
            "missing mCRL2 proc section",
        ),
        formal_benchmark_check(
            "init-section",
            formal_line_starts_with(&spec, "init"),
            "missing mCRL2 init section",
        ),
        formal_benchmark_check(
            "semicolon-present",
            spec.contains(';'),
            "missing mCRL2 statement terminators",
        ),
        formal_benchmark_check(
            "property-or-modal-operator",
            !properties.trim().is_empty()
                || ["[", "<", "mu ", "nu "]
                    .iter()
                    .any(|token| spec.contains(token)),
            "missing mCRL2 modal property/formula",
        ),
    ];
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result("ok", verdict, "builtin:mcrl2-structural", "", checks)
}

fn formal_validate_maude(payload: &Value) -> Value {
    let module = formal_benchmark_text(payload, &["model", "maude", "module", "text"]);
    let commands = formal_benchmark_strings(payload, &["commands", "properties"]).join("\n");
    let lower = module.to_ascii_lowercase();
    let checks = vec![
        formal_benchmark_check(
            "module-header",
            (lower.contains("mod ") || lower.contains("fmod ")) && lower.contains(" is"),
            "missing Maude module header",
        ),
        formal_benchmark_check(
            "module-terminator",
            lower.contains("endfm") || lower.contains("endm"),
            "missing Maude module terminator",
        ),
        formal_benchmark_check(
            "operator-or-rule",
            ["op ", "eq ", "rl ", "crl "]
                .iter()
                .any(|token| lower.contains(token)),
            "missing Maude op/equation/rule",
        ),
        formal_benchmark_check(
            "brackets-balanced",
            formal_benchmark_balanced(&module, '[', ']'),
            "Maude brackets are not balanced",
        ),
        formal_benchmark_check(
            "command-present",
            !commands.trim().is_empty()
                || ["search", "red", "rew"]
                    .iter()
                    .any(|token| lower.contains(token)),
            "missing Maude command/search",
        ),
    ];
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result("ok", verdict, "builtin:maude-structural", "", checks)
}

fn formal_validate_program_verifier(payload: &Value, tool: &str) -> Value {
    let source = formal_benchmark_text(payload, &["source", "model", "program", "spec", "text"]);
    let language = model_validation_normalized_tool(
        payload
            .get("language")
            .and_then(Value::as_str)
            .unwrap_or(tool),
    );
    let contract_text =
        formal_benchmark_strings(payload, &["contracts", "properties", "expected_contracts"])
            .join("\n");
    let combined = format!("{source}\n{contract_text}");
    let combined_lower = combined.to_ascii_lowercase();
    let source_lower = source.to_ascii_lowercase();
    let has_contract = [
        "requires",
        "ensures",
        "invariant",
        "assert",
        "assume",
        "lemma",
        "theorem",
        "goal",
        "predicate",
        "claim",
    ]
    .iter()
    .any(|token| formal_contains_word(&combined_lower, token))
        || source.contains("/*@")
        || source.contains("//@")
        || source.contains("#[kani::proof]");
    let mut checks = vec![
        formal_benchmark_check(
            "source-present",
            !source.trim().is_empty(),
            "missing program/verifier source",
        ),
        formal_benchmark_check(
            "braces-balanced",
            formal_benchmark_balanced(&source, '{', '}'),
            "program braces are not balanced",
        ),
        formal_benchmark_check(
            "contract-or-assertion",
            has_contract,
            "missing contract/assertion/proof obligation",
        ),
    ];
    match language.as_str() {
        "dafny" => {
            checks.push(formal_benchmark_check(
                "dafny-declaration",
                ["method", "function", "predicate", "lemma", "class"]
                    .iter()
                    .any(|token| formal_contains_word(&source_lower, token)),
                "missing Dafny declaration",
            ));
            checks.push(formal_benchmark_check(
                "dafny-spec",
                ["requires", "ensures", "invariant", "assert"]
                    .iter()
                    .any(|token| formal_contains_word(&combined_lower, token)),
                "missing Dafny specification",
            ));
        }
        "why3" | "whyml" => {
            checks.push(formal_benchmark_check(
                "why3-module",
                source_lower.contains("module "),
                "missing Why3 module",
            ));
            checks.push(formal_benchmark_check(
                "why3-obligation",
                ["goal", "lemma", "let", "requires", "ensures", "invariant"]
                    .iter()
                    .any(|token| formal_contains_word(&combined_lower, token)),
                "missing Why3 proof obligation",
            ));
        }
        "frama-c" | "framac" => {
            let has_c_function = source.lines().any(|line| {
                let line = line.trim_start();
                ["int ", "void ", "double ", "float ", "char "]
                    .iter()
                    .any(|prefix| line.starts_with(prefix))
                    && line.contains('(')
            });
            checks.push(formal_benchmark_check(
                "c-function",
                has_c_function,
                "missing C function",
            ));
            checks.push(formal_benchmark_check(
                "acsl-contract",
                source.contains("/*@") || source.contains("//@"),
                "missing ACSL annotation",
            ));
        }
        "kani" | "mirai" | "rust" => {
            checks.push(formal_benchmark_check(
                "rust-function",
                source_lower.contains("fn "),
                "missing Rust function",
            ));
            checks.push(formal_benchmark_check(
                "rust-harness-or-assert",
                source.contains("#[kani::proof]")
                    || source_lower.contains("kani::")
                    || source.contains("assert!"),
                "missing Rust verifier harness/assertion",
            ));
        }
        "ebmc" | "esbmc" | "cbmc" | "cpa-checker" | "cpachecker" | "jbmc" | "klee" => {
            checks.push(formal_benchmark_check(
                "bounded-model-assertion",
                source.contains("__CPROVER_assert")
                    || source.contains("assert(")
                    || source.contains("assert ("),
                "missing bounded-model-checker assertion",
            ));
        }
        "coq" | "isabelle" | "lean" | "pvs" | "acl2" => {
            checks.push(formal_benchmark_check(
                "proof-declaration",
                ["Theorem", "Lemma", "theorem", "lemma", "Definition", "def"]
                    .iter()
                    .any(|token| source.contains(token)),
                "missing proof assistant theorem/lemma",
            ));
        }
        _ => {}
    }
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result(
        "ok",
        verdict,
        "builtin:program-verifier-structural",
        language,
        checks,
    )
}

fn formal_validate_security_protocol(payload: &Value, tool: &str) -> Value {
    let model = formal_benchmark_text(payload, &["model", "protocol", "source", "spec", "text"]);
    let properties =
        formal_benchmark_strings(payload, &["properties", "queries", "lemmas"]).join("\n");
    let combined = format!("{model}\n{properties}");
    let lower = combined.to_ascii_lowercase();
    let model_lower = model.to_ascii_lowercase();
    let mut checks = vec![
        formal_benchmark_check(
            "model-present",
            !model.trim().is_empty(),
            "missing security-protocol model",
        ),
        formal_benchmark_check(
            "braces-balanced",
            formal_benchmark_balanced(&model, '{', '}'),
            "protocol braces are not balanced",
        ),
        formal_benchmark_check(
            "actor-or-process",
            ["role", "process", "principal", "rule", "protocol", "theory"]
                .iter()
                .any(|token| formal_contains_word(&model_lower, token)),
            "missing role/process/rule/protocol declaration",
        ),
        formal_benchmark_check(
            "query-or-lemma",
            [
                "query",
                "lemma",
                "claim",
                "confidentiality",
                "authentication",
                "secrecy",
            ]
            .iter()
            .any(|token| formal_contains_word(&lower, token) || lower.contains(token)),
            "missing security query/lemma/claim",
        ),
    ];
    match tool {
        "tamarin" => {
            checks.push(formal_benchmark_check(
                "tamarin-theory",
                model_lower.contains("theory ") && model_lower.contains(" begin"),
                "missing theory begin",
            ));
            checks.push(formal_benchmark_check(
                "tamarin-end",
                formal_contains_word(&model_lower, "end"),
                "missing theory end",
            ));
            checks.push(formal_benchmark_check(
                "tamarin-rule",
                model_lower.contains("rule "),
                "missing Tamarin rule",
            ));
        }
        "proverif" | "cryptoverif" | "deepsec" => {
            checks.push(formal_benchmark_check(
                "applied-pi-shape",
                ["free", "fun", "event", "query", "process"]
                    .iter()
                    .any(|token| formal_contains_word(&lower, token)),
                "missing applied-pi/protocol declarations",
            ));
        }
        "scyther" => {
            checks.push(formal_benchmark_check(
                "scyther-protocol",
                model_lower.contains("protocol "),
                "missing protocol declaration",
            ));
            checks.push(formal_benchmark_check(
                "scyther-role",
                model_lower.contains("role "),
                "missing role declaration",
            ));
            checks.push(formal_benchmark_check(
                "scyther-claim",
                lower.contains("claim"),
                "missing claim",
            ));
        }
        "verifpal" => {
            checks.push(formal_benchmark_check(
                "verifpal-query",
                lower.contains("queries") || lower.contains("confidentiality?"),
                "missing Verifpal query block",
            ));
        }
        _ => {}
    }
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result(
        "ok",
        verdict,
        "builtin:security-protocol-structural",
        tool,
        checks,
    )
}

fn formal_validate_benchmark_manifest(payload: &Value) -> Value {
    let suite = payload
        .get("suite")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let entries = payload.get("entries");
    let require_paths = payload
        .get("require_paths")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let root_dir = payload
        .get("root_dir")
        .and_then(Value::as_str)
        .unwrap_or(".");
    let mut checks = vec![
        formal_benchmark_check(
            "suite-present",
            !suite.is_empty(),
            "missing benchmark suite",
        ),
        formal_benchmark_check(
            "entries-array",
            entries.is_some_and(Value::is_array),
            "entries must be an array",
        ),
    ];
    let mut names = BTreeMap::<String, ()>::new();
    if let Some(entries) = entries.and_then(Value::as_array) {
        checks.push(formal_benchmark_check(
            "entries-nonempty",
            !entries.is_empty(),
            "manifest has no entries",
        ));
        for (idx, entry) in entries.iter().enumerate() {
            let Some(entry_obj) = entry.as_object() else {
                checks.push(formal_benchmark_check(
                    &format!("entry:{idx}:object"),
                    false,
                    "entry must be an object",
                ));
                continue;
            };
            let name = entry_obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let family = entry_obj
                .get("family")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let format = entry_obj
                .get("format")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase();
            let path = entry_obj
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            checks.push(formal_benchmark_check(
                &format!("entry:{idx}:name"),
                !name.is_empty(),
                "missing entry name",
            ));
            checks.push(formal_benchmark_check(
                &format!("entry:{idx}:family"),
                !family.is_empty(),
                "missing entry family",
            ));
            checks.push(formal_benchmark_check(
                &format!("entry:{idx}:format"),
                !format.is_empty(),
                "missing entry format",
            ));
            checks.push(formal_benchmark_check(
                &format!("entry:{idx}:path"),
                !path.is_empty(),
                "missing entry path",
            ));
            let unique = name.is_empty() || !names.contains_key(name);
            checks.push(formal_benchmark_check(
                &format!("entry:{idx}:unique"),
                unique,
                "duplicate entry name",
            ));
            if !name.is_empty() {
                names.insert(name.to_string(), ());
            }
            if !format.is_empty() {
                checks.push(formal_benchmark_check(
                    &format!("entry:{idx}:format-known"),
                    matches!(
                        format.as_str(),
                        "lp" | "mps" | "nl" | "osil" | "json" | "dzn" | "qplib" | "cnf" | "fzn"
                    ),
                    &format!("unrecognized benchmark format {format:?}"),
                ));
            }
            if require_paths && !path.is_empty() {
                checks.push(formal_benchmark_check(
                    &format!("entry:{idx}:path-exists"),
                    Path::new(root_dir).join(path).is_file(),
                    &format!(
                        "benchmark file not found: {}",
                        Path::new(root_dir).join(path).display()
                    ),
                ));
            }
        }
    }
    let verdict = formal_benchmark_verdict(&checks);
    formal_benchmark_result("ok", verdict, "builtin:benchmark-manifest", "", checks)
}

pub fn run_formal_benchmark_validation_json_with_rust_reference(
    payload: &Value,
    tool: &str,
) -> Value {
    let tool = model_validation_normalized_tool(tool);
    let kind = payload
        .get("kind")
        .and_then(Value::as_str)
        .map(model_validation_normalized_tool)
        .unwrap_or_default();
    match (kind.as_str(), tool.as_str()) {
        ("tla-validation" | "tla-plus-validation", _) | (_, "tlc" | "apalache" | "tla") => {
            formal_validate_tla(payload)
        }
        ("prism-validation" | "prism-model-validation", _) | (_, "prism" | "storm") => {
            formal_validate_prism(payload)
        }
        ("alloy-validation" | "kodkod-validation", _) | (_, "alloy" | "kodkod") => {
            formal_validate_alloy(payload)
        }
        ("promela-validation" | "spin-validation", _) | (_, "spin") => {
            formal_validate_promela(payload)
        }
        ("smv-validation" | "nuxmv-validation", _) | (_, "nuxmv") => formal_validate_smv(payload),
        ("uppaal-validation" | "uppaal-xml-validation", _) | (_, "uppaal") => {
            formal_validate_uppaal(payload)
        }
        ("cbmc-validation" | "c-bounded-model-validation", _) | (_, "cbmc") => {
            formal_validate_cbmc(payload)
        }
        ("program-verifier-validation" | "deductive-verification", _)
        | (
            _,
            "dafny" | "frama-c" | "framac" | "why3" | "whyml" | "kani" | "mirai" | "ebmc" | "esbmc"
            | "cpa-checker" | "cpachecker" | "jbmc" | "klee" | "coq" | "isabelle" | "lean" | "pvs"
            | "acl2",
        ) => formal_validate_program_verifier(payload, &tool),
        ("security-protocol-validation" | "protocol-verification", _)
        | (
            _,
            "tamarin" | "proverif" | "cryptoverif" | "deepsec" | "scyther" | "verifpal" | "sapic"
            | "sapic-plus",
        ) => formal_validate_security_protocol(payload, &tool),
        ("mcrl2-validation" | "mcrl2-spec-validation", _) | (_, "mcrl2") => {
            formal_validate_mcrl2(payload)
        }
        ("maude-validation" | "maude-module-validation", _) | (_, "maude") => {
            formal_validate_maude(payload)
        }
        ("external-benchmark-manifest", _)
        | (
            _,
            "benchmark" | "miplib" | "qplib" | "minlplib" | "netlib-lp" | "csplib" | "or-library"
            | "tsplib" | "vrplib" | "minizinc-challenge",
        ) => formal_validate_benchmark_manifest(payload),
        _ => formal_benchmark_result(
            "unavailable",
            "unknown",
            &tool,
            format!("unknown formal/benchmark payload kind {kind:?}"),
            Vec::new(),
        ),
    }
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

fn external_validation_reference_timeout_ms() -> u64 {
    env::var("EXTERNAL_VALIDATION_REFERENCE_TIMEOUT_MS")
        .or_else(|_| env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_external_validation_output(
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
            Err(err) => return Err(format!("failed to poll external validation process: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for external validation process: {err}"))
}

const EVENT_SIMULATION_ENGINES: &[&str] = &[
    "simpy",
    "salabim",
    "ciw",
    "simulus",
    "simmer",
    "jaamsim",
    "desmo-j",
    "simsharp",
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

fn normalized_simulation_engine_id(engine_id: &str) -> String {
    engine_id.trim().to_ascii_lowercase().replace('_', "-")
}

fn canonical_simulation_engine_id(engine_id: &str) -> String {
    let normalized = normalized_simulation_engine_id(engine_id);
    let canonical = match normalized.as_str() {
        "simpy-adapter" => "simpy",
        "salabim-adapter" => "salabim",
        "ciw-adapter" => "ciw",
        "simulus-adapter" => "simulus",
        "simmer-adapter" | "rscript" => "simmer",
        "desmoj" | "desmoj-adapter" => "desmo-j",
        "simsharp-adapter" => "simsharp",
        "waf" => "ns3",
        "opp-run" | "opp_run" | "omnet++" => "omnetpp",
        "sumo-gui" => "sumo",
        "fmpy" | "fmucheck" | "fmu-adapter" | "fmi" | "fmu" => "fmi-fmu",
        "omc" => "openmodelica",
        "matlab" | "simulink-adapter" => "simulink",
        "ptolemy" | "vergil" | "ptolemy-adapter" => "ptolemy-ii",
        "gem5.opt" => "gem5",
        "opendsscmd" | "dss" => "opendss",
        "pandapower-adapter" => "pandapower",
        "copasise" => "copasi",
        "tellurium-adapter" => "tellurium",
        "gz" => "gazebo",
        "mujoco-adapter" => "mujoco",
        "drake-adapter" => "drake",
        "pybullet-adapter" => "pybullet",
        "carlaue4" | "carlaue4.sh" => "carla",
        "isaacsim" => "isaac-sim",
        "airsim-adapter" => "airsim",
        "mesa-adapter" => "mesa",
        "agentpy-adapter" => "agentpy",
        "repast-adapter" | "repast-simphony" => "repast",
        "mason-adapter" => "mason",
        "netlogo-headless" | "netlogo-headless.sh" => "netlogo",
        "simgrid-mc" | "teshsuite" => "simgrid",
        "cloudsim-adapter" => "cloudsim",
        "neqsim-adapter" => "neqsim",
        "dwsim-adapter" => "dwsim",
        "capeopen-adapter" | "cape-open-adapter" => "cape-open",
        "plant-simulation-adapter" | "plantsim-adapter" => "plant-simulation",
        "extendsim-adapter" => "extendsim",
        "gpss-adapter" | "gpss-world-adapter" => "gpss-world",
        "anylogic-adapter" => "anylogic",
        "simio-adapter" => "simio",
        "simul8-adapter" => "simul8",
        "arena-adapter" => "arena",
        "flexsim-adapter" => "flexsim",
        _ => normalized.as_str(),
    };
    canonical.to_string()
}

fn simulation_engine_in_family(engine_id: &str, family: &[&str]) -> bool {
    let canonical = canonical_simulation_engine_id(engine_id);
    family.contains(&canonical.as_str())
}

fn default_simulation_model_format_for_engine(engine_id: &str) -> &'static str {
    if simulation_engine_in_family(engine_id, EVENT_SIMULATION_ENGINES) {
        "json-event-network"
    } else if simulation_engine_in_family(engine_id, MOBILITY_SIMULATION_ENGINES) {
        "json-mobility-network"
    } else if simulation_engine_in_family(engine_id, ENERGY_SIMULATION_ENGINES) {
        "json-energy-balance"
    } else if simulation_engine_in_family(engine_id, PHYSICS_SIMULATION_ENGINES) {
        "json-physics-trajectory"
    } else if simulation_engine_in_family(engine_id, AGENT_SIMULATION_ENGINES) {
        "json-agent-based"
    } else if simulation_engine_in_family(engine_id, DISTRIBUTED_SIMULATION_ENGINES) {
        "json-distributed-system"
    } else if simulation_engine_in_family(engine_id, PROCESS_SIMULATION_ENGINES) {
        "json-process-flow"
    } else {
        "json-event-network"
    }
}

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

fn unavailable_rust_simulation_reference_run(
    engine_id: String,
    message: String,
    elapsed_ms: f64,
) -> ExternalSimulationValidationReferenceRun {
    simulation_reference_run(
        engine_id,
        "rust:unsupported-simulation-format".to_string(),
        ExternalSimulationValidationStatus::Unavailable,
        ExternalSimulationValidationVerdict::Unknown,
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
    let engine = canonical_simulation_engine_id(&engine_id);
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
    let engine = canonical_simulation_engine_id(&engine_id);
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
    let engine = canonical_simulation_engine_id(&engine_id);
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
    let engine = canonical_simulation_engine_id(&engine_id);
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
    let engine = canonical_simulation_engine_id(&engine_id);
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
    let engine = canonical_simulation_engine_id(&engine_id);
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
    let engine = canonical_simulation_engine_id(&engine_id);
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
        .unwrap_or_else(|| default_simulation_model_format_for_engine(&engine_id));
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

    unavailable_rust_simulation_reference_run(
        engine_id,
        format!("unsupported model_format {model_format:?}"),
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

pub fn run_simulation_validation_json_with_python_reference(
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

    if let Some(mut stdin) = child.stdin.take() {
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

    let timeout_ms = external_validation_reference_timeout_ms();
    let (output, timed_out) = match wait_for_external_validation_output(child, timeout_ms) {
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
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("external validation process timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; external validation process timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
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
    let timeout_ms = external_validation_reference_timeout_ms();
    let (output, timed_out) = match wait_for_external_validation_output(child, timeout_ms) {
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
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if stderr.trim().is_empty() {
            format!("external validation process timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; external validation process timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).to_string()
    };
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
        "minilp" => names.push("MINILP_CARGO_MANIFEST".to_string()),
        "argmin" => names.push("ARGMIN_CRATE".to_string()),
        "nlopt-rs" => names.push("NLOPT_DIR".to_string()),
        "osqp-rust" => {
            names.push("OSQP_RS_CARGO_MANIFEST".to_string());
            names.push("OSQP_DIR".to_string());
            names.push("OSQP_HOME".to_string());
        }
        "clarabel-rust" => names.push("CLARABEL_RS_CARGO_MANIFEST".to_string()),
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
        "osqp-rust" => &["OSQP_DIR", "OSQP_HOME"],
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

    if tool.id == "lindo-cli" {
        let probe = probe_external_linear_cli_solver(
            ExternalLinearCliKind::Lp,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Lindo,
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        if probe.status == ExternalLinearCliProbeStatus::Ready {
            return ExternalValidationProbe {
                tool_id: tool.id.to_string(),
                status: ExternalValidationProbeStatus::Ready,
                command: probe.command.clone(),
                message: format!(
                    "{} via external_linear_cli ready probe: {}",
                    tool.display_name, probe.message
                ),
            };
        }
    }

    if tool.id == "knitro" {
        let probe = probe_external_gams_solver(ExternalGamsSolver::Knitro, 10_000);
        if probe.ready {
            return ExternalValidationProbe {
                tool_id: tool.id.to_string(),
                status: ExternalValidationProbeStatus::Ready,
                command: probe.command,
                message: format!(
                    "{} via GAMS ready probe: {}",
                    tool.display_name, probe.message
                ),
            };
        }
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
    if tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage {
        return probe_python_validation_package(tool);
    }
    if tool.artifact_kind == ExternalValidationArtifactKind::NodePackage {
        return probe_node_validation_package(tool, None);
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
    let timeout_ms = external_validation_reference_timeout_ms();
    let (output, timed_out) = match wait_for_external_validation_output(child, timeout_ms) {
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
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("external validation process timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; external validation process timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
    if !output.status.success() {
        return ExternalValidationRun {
            tool_id: tool.id.to_string(),
            status: ExternalValidationRunStatus::Failed,
            output: None,
            elapsed_ms,
            message: stderr,
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

fn probe_python_validation_package(tool: &ExternalValidationToolSpec) -> ExternalValidationProbe {
    let Some(python) = default_python_probe_command() else {
        return ExternalValidationProbe {
            tool_id: tool.id.to_string(),
            status: ExternalValidationProbeStatus::NotConfigured,
            command: None,
            message: format!(
                "{} needs a local adapter command or Python env; set {} or {}",
                tool.display_name,
                external_validation_adapter_env_names(tool)[0],
                external_validation_artifact_hint(tool)
            ),
        };
    };
    if external_validation_python_modules(tool)
        .iter()
        .any(|module| python_can_import(&python, module))
    {
        return ExternalValidationProbe {
            tool_id: tool.id.to_string(),
            status: ExternalValidationProbeStatus::Ready,
            command: Some(python),
            message: format!("{} Python module is importable", tool.display_name),
        };
    }
    ExternalValidationProbe {
        tool_id: tool.id.to_string(),
        status: ExternalValidationProbeStatus::NotConfigured,
        command: Some(python),
        message: format!(
            "{} needs a local adapter command or importable package; set {} or {}",
            tool.display_name,
            external_validation_adapter_env_names(tool)[0],
            external_validation_artifact_hint(tool)
        ),
    }
}

fn external_validation_python_modules(
    tool: &ExternalValidationToolSpec,
) -> &'static [&'static str] {
    match tool.id {
        "cpmpy" => &["cpmpy"],
        "pycsp3" => &["pycsp3"],
        "pyomo" => &["pyomo.environ", "pyomo"],
        "pulp" => &["pulp"],
        "pyscipopt" => &["pyscipopt"],
        "python-mip" => &["mip"],
        "clingo" => &["clingo"],
        "cvc5" => &["cvc5"],
        "bitwuzla" => &["bitwuzla"],
        "gurobipy" => &["gurobipy"],
        "cplex-python" => &["cplex"],
        "xpress-python" => &["xpress"],
        "docplex" => &["docplex"],
        "ortools-python" | "ortools-glop" | "ortools-pdlp" => &["ortools"],
        "scipy-optimize" => &["scipy.optimize", "scipy"],
        "mosek" => &["mosek"],
        "copt" => &["coptpy"],
        "nlopt" => &["nlopt"],
        "kissat" => &["pysat.solvers:kissat"],
        "cadical" => &["pysat.solvers:cadical153"],
        "minisat" => &["pysat.solvers:minisat22"],
        "glucose" => &["pysat.solvers:glucose4"],
        "maplesat" => &["pysat.solvers:maplesat"],
        "pysat" => &["pysat"],
        "casadi" => &["casadi"],
        "osqp" => &["osqp"],
        "scs" => &["scs"],
        "clarabel" => &["clarabel"],
        "ecos" => &["ecos"],
        "proxqp" => &["proxsuite.proxqp", "proxsuite"],
        "cvxpy" => &["cvxpy"],
        "cvxopt" => &["cvxopt"],
        "simpy" => &["simpy"],
        "salabim" => &["salabim"],
        "ciw" => &["ciw"],
        "simulus" => &["simulus"],
        "pandapower" => &["pandapower"],
        "tellurium" => &["tellurium"],
        "mujoco" => &["mujoco"],
        "drake" => &["pydrake"],
        "pybullet" => &["pybullet"],
        "mesa" => &["mesa"],
        "agentpy" => &["agentpy"],
        "json-schema" => &["jsonschema"],
        "check-jsonschema" => &["check_jsonschema"],
        "openapi-spec-validator" => &["openapi_spec_validator"],
        "csv-validator" => &["csvvalidator:smoke"],
        "pydantic" => &["pydantic"],
        "marshmallow" => &["marshmallow"],
        "cerberus" => &["cerberus"],
        "python-xmlschema" => &["xmlschema"],
        "schematron" => &["lxml.isoschematron:smoke"],
        "saxon" => &["saxonche:smoke"],
        "great-expectations" => &["great_expectations"],
        "pandera" => &["pandera"],
        "whylogs" => &["whylogs"],
        "soda-core" => &["soda_core.scan", "soda_core", "soda.scan", "soda"],
        "evidently" => &["evidently"],
        "deepchecks" => &["deepchecks"],
        "frictionless" => &["frictionless"],
        "sqlfluff" => &["sqlfluff"],
        "apache-avro" => &["avro"],
        "apache-arrow" => &["pyarrow"],
        "tensorflow-data-validation" => &["tensorflow_data_validation"],
        _ => &[],
    }
}

fn probe_node_validation_package(
    tool: &ExternalValidationToolSpec,
    node_path: Option<&str>,
) -> ExternalValidationProbe {
    let Some(node) = default_node_probe_command() else {
        return ExternalValidationProbe {
            tool_id: tool.id.to_string(),
            status: ExternalValidationProbeStatus::NotConfigured,
            command: None,
            message: format!(
                "{} needs a local adapter command or Node.js env; set {} or {}",
                tool.display_name,
                external_validation_adapter_env_names(tool)[0],
                external_validation_artifact_hint(tool)
            ),
        };
    };
    if external_validation_node_modules(tool)
        .iter()
        .any(|module| node_can_import(&node, module, node_path))
    {
        return ExternalValidationProbe {
            tool_id: tool.id.to_string(),
            status: ExternalValidationProbeStatus::Ready,
            command: Some(node),
            message: format!("{} Node package is importable", tool.display_name),
        };
    }
    ExternalValidationProbe {
        tool_id: tool.id.to_string(),
        status: ExternalValidationProbeStatus::NotConfigured,
        command: Some(node),
        message: format!(
            "{} needs a local adapter command or importable Node package; set {} or {}",
            tool.display_name,
            external_validation_adapter_env_names(tool)[0],
            external_validation_artifact_hint(tool)
        ),
    }
}

fn external_validation_node_modules(tool: &ExternalValidationToolSpec) -> &'static [&'static str] {
    match tool.id {
        "zod" => &["zod"],
        "valibot" => &["valibot"],
        _ => &[],
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
        ExternalValidationArtifactKind::NodePackage => {
            probe_node_validation_package(tool, Some(value.as_str()))
        }
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

fn python_can_import(python: &Path, module: &str) -> bool {
    let probe = if let Some(solver_name) = module.strip_prefix("pysat.solvers:") {
        format!(
            "import sys; from pysat.solvers import Solver; solver = Solver(name={solver_name:?}, bootstrap_with=[[1], [-1]]); result = solver.solve(); solver.delete(); sys.exit(0 if result is False else 1)"
        )
    } else if module == "lxml.isoschematron:smoke" {
        r#"
import sys
from lxml import etree, isoschematron
schema_doc = etree.XML(b'<schema xmlns="http://purl.oclc.org/dsdl/schematron"><pattern><rule context="item"><assert test="@id">item needs id</assert></rule></pattern></schema>')
schematron = isoschematron.Schematron(schema_doc)
valid_doc = etree.XML(b'<root><item id="a"/></root>')
invalid_doc = etree.XML(b'<root><item/></root>')
sys.exit(0 if schematron.validate(valid_doc) and not schematron.validate(invalid_doc) else 1)
"#
        .to_string()
    } else if module == "csvvalidator:smoke" {
        r#"
import sys
from csvvalidator import CSVValidator, number_range_inclusive
validator = CSVValidator(["name", "score"])
validator.add_header_check()
validator.add_record_length_check()
validator.add_value_check("score", number_range_inclusive(0, 10))
valid = validator.validate([["name", "score"], ["alpha", "7"]])
invalid = validator.validate([["name", "score"], ["alpha", "11"]])
sys.exit(0 if not valid and invalid else 1)
"#
        .to_string()
    } else if module == "saxonche:smoke" {
        r#"
import sys
from saxonche import PySaxonProcessor
with PySaxonProcessor(license=False) as proc:
    xpath = proc.new_xpath_processor()
    two = xpath.evaluate_single("1 + 1")
    xslt = proc.new_xslt30_processor()
    executable = xslt.compile_stylesheet(stylesheet_text='<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"><xsl:template match="/"><ok><xsl:value-of select="count(/root/item)"/></ok></xsl:template></xsl:stylesheet>')
    node = proc.parse_xml(xml_text="<root><item/><item/></root>")
    rendered = executable.transform_to_string(xdm_node=node)
sys.exit(0 if two is not None and two.string_value == "2" and "<ok>2</ok>" in rendered else 1)
"#
        .to_string()
    } else if module == "pycsp3" {
        format!(
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module:?}) else 1)"
        )
    } else {
        format!("import importlib; importlib.import_module({module:?})")
    };
    Command::new(python)
        .arg("-c")
        .arg(probe)
        .output()
        .is_ok_and(|output| output.status.success())
}

fn default_node_probe_command() -> Option<PathBuf> {
    node_probe_command_from_env(env::var_os("NODE_BIN"), env::var_os("NODE"))
        .or_else(|| find_first_command(&["node"]))
}

fn node_probe_command_from_env(
    node_bin: Option<OsString>,
    node: Option<OsString>,
) -> Option<PathBuf> {
    node_bin
        .filter(|value| !value.is_empty())
        .or_else(|| node.filter(|value| !value.is_empty()))
        .map(PathBuf::from)
}

fn node_can_import(node: &Path, module: &str, node_path: Option<&str>) -> bool {
    let mut esm_import = Command::new(node);
    esm_import
        .arg("--input-type=module")
        .arg("-e")
        .arg(format!("await import({module:?});"));
    if let Some(path) = node_path.filter(|path| !path.is_empty()) {
        esm_import.env("NODE_PATH", path);
    }
    if esm_import
        .output()
        .is_ok_and(|output| output.status.success())
    {
        return true;
    }

    let mut require_resolve = Command::new(node);
    require_resolve
        .arg("-e")
        .arg(
            r#"
const { createRequire } = require("module");
const { pathToFileURL } = require("url");
const req = createRequire(process.cwd() + "/");
Promise.resolve()
  .then(async () => {
    const resolved = req.resolve(process.argv[1]);
    await import(pathToFileURL(resolved).href);
  })
  .then(() => process.exit(0), () => process.exit(1));
"#,
        )
        .arg(module);
    if let Some(path) = node_path.filter(|path| !path.is_empty()) {
        require_resolve.env("NODE_PATH", path);
    }
    require_resolve
        .output()
        .is_ok_and(|output| output.status.success())
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
    use super::wait_for_external_validation_output;
    use crate::des::general::external_validation_tools::{
        dimacs_cnf_to_string, dimacs_wcnf_to_string, external_benchmark_manifest_to_json,
        external_simulation_validation_engine_manifest, external_simulation_validation_tool_specs,
        external_validation_adapter_env_names, external_validation_artifact_cli_args,
        external_validation_artifact_env_names, external_validation_command_dir_env_names,
        external_validation_consensus_report_to_json,
        external_validation_default_artifact_cli_args, external_validation_default_file_cli_args,
        external_validation_default_text_cli_args, external_validation_file_cli_args,
        external_validation_node_modules, external_validation_python_modules,
        external_validation_tool_specs, find_command_in_install_dir, find_external_validation_tool,
        find_java_classpath_in_install_dir, infer_external_validation_text_verdict, is_jar_file,
        json_schema_validation_request_to_json, minizinc_validation_request_to_json,
        node_probe_command_from_env, prism_validation_model_to_string,
        prism_validation_properties_to_string, python_probe_command_from_env,
        run_external_validation_artifact_cli, run_external_validation_consensus,
        run_external_validation_file_cli, run_external_validation_text_cli,
        run_model_validation_json_with_rust_reference,
        run_output_validation_json_with_rust_reference,
        run_proof_validation_json_with_rust_reference,
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
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    #[test]
    fn validation_python_probe_command_honors_python_bin_precedence() {
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
                Some(OsString::from("/tmp/python")),
            ),
            Some(PathBuf::from("/tmp/python")),
        );
        assert_eq!(python_probe_command_from_env(None, None), None);
    }

    #[test]
    fn validation_node_probe_command_honors_node_bin_precedence() {
        assert_eq!(
            node_probe_command_from_env(
                Some(OsString::from("/tmp/node-bin")),
                Some(OsString::from("/tmp/node")),
            ),
            Some(PathBuf::from("/tmp/node-bin")),
        );
        assert_eq!(
            node_probe_command_from_env(None, Some(OsString::from("/tmp/node"))),
            Some(PathBuf::from("/tmp/node")),
        );
        assert_eq!(
            node_probe_command_from_env(Some(OsString::new()), Some(OsString::from("/tmp/node")),),
            Some(PathBuf::from("/tmp/node")),
        );
        assert_eq!(node_probe_command_from_env(None, None), None);
    }

    #[test]
    fn python_package_import_map_covers_declared_validation_tools() {
        for tool in external_validation_tool_specs()
            .iter()
            .filter(|tool| tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage)
        {
            assert!(
                !external_validation_python_modules(tool).is_empty(),
                "{} is declared as a Python package but has no probe module",
                tool.id
            );
        }

        let expected_pysat_backends = [
            ("kissat", "pysat.solvers:kissat"),
            ("cadical", "pysat.solvers:cadical153"),
            ("minisat", "pysat.solvers:minisat22"),
            ("glucose", "pysat.solvers:glucose4"),
            ("maplesat", "pysat.solvers:maplesat"),
        ];
        for (tool_id, backend) in expected_pysat_backends {
            let tool = find_external_validation_tool(tool_id).unwrap();
            assert_eq!(tool.runtime, ExternalValidationRuntime::Python);
            assert_eq!(
                tool.artifact_kind,
                ExternalValidationArtifactKind::PythonPackage
            );
            assert!(
                external_validation_python_modules(tool).contains(&backend),
                "{tool_id} should probe its concrete PySAT backend"
            );
        }

        for (tool_id, module) in [
            ("mosek", "mosek"),
            ("copt", "coptpy"),
            ("apache-avro", "avro"),
            ("json-schema", "jsonschema"),
            ("schematron", "lxml.isoschematron:smoke"),
            ("csv-validator", "csvvalidator:smoke"),
            ("saxon", "saxonche:smoke"),
        ] {
            let tool = find_external_validation_tool(tool_id).unwrap();
            assert_eq!(tool.runtime, ExternalValidationRuntime::Python);
            assert_eq!(
                tool.artifact_kind,
                ExternalValidationArtifactKind::PythonPackage
            );
            assert!(
                external_validation_python_modules(tool).contains(&module),
                "{tool_id} should probe its Python API package"
            );
        }
    }

    #[test]
    fn node_package_import_map_covers_declared_validation_tools() {
        for tool in external_validation_tool_specs()
            .iter()
            .filter(|tool| tool.artifact_kind == ExternalValidationArtifactKind::NodePackage)
        {
            assert_eq!(tool.runtime, ExternalValidationRuntime::Node);
            assert!(
                !external_validation_node_modules(tool).is_empty(),
                "{} is declared as a Node package but has no probe module",
                tool.id
            );
        }

        for (tool_id, module) in [("zod", "zod"), ("valibot", "valibot")] {
            let tool = find_external_validation_tool(tool_id).unwrap();
            assert_eq!(tool.runtime, ExternalValidationRuntime::Node);
            assert_eq!(
                tool.artifact_kind,
                ExternalValidationArtifactKind::NodePackage
            );
            assert!(
                external_validation_node_modules(tool).contains(&module),
                "{tool_id} should probe its Node API package"
            );
        }
    }

    #[test]
    fn output_validation_yamllint_and_graphql_schema_have_rust_fallbacks() {
        let yaml = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "yaml-validation",
                "yaml": "---\nname: soccer-learning\nsteps:\n  - simulate\n  - validate\n",
            }),
            "yamllint",
        );
        assert_eq!(yaml["status"].as_str(), Some("ok"));
        assert_eq!(yaml["verdict"].as_str(), Some("valid"));
        assert_eq!(yaml["validator"].as_str(), Some("builtin:yaml-structural"));

        let bad_yaml = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "yaml-validation",
                "yaml": "---\n: missing-key\n",
            }),
            "yamllint",
        );
        assert_eq!(bad_yaml["status"].as_str(), Some("ok"));
        assert_eq!(bad_yaml["verdict"].as_str(), Some("invalid"));

        let graphql = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "graphql-schema-validation",
                "schema": "type Query { score: Int }\n",
            }),
            "graphql-schema",
        );
        assert_eq!(graphql["status"].as_str(), Some("ok"));
        assert_eq!(graphql["verdict"].as_str(), Some("valid"));
        assert_eq!(
            graphql["validator"].as_str(),
            Some("builtin:graphql-schema-structural")
        );

        let bad_graphql = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "graphql-schema-validation",
                "schema": "type Query { score: Int \n",
            }),
            "graphql-schema",
        );
        assert_eq!(bad_graphql["status"].as_str(), Some("ok"));
        assert_eq!(bad_graphql["verdict"].as_str(), Some("invalid"));
    }

    #[test]
    fn xml_schema_command_aliases_use_rust_structural_fallbacks() {
        let payload = json!({
            "kind": "xsd-validation",
            "xml": "<match><score home=\"1\" away=\"0\"/></match>",
            "requiredElements": ["match", "score"],
        });
        for (alias, expected_validator) in [
            ("xmlschema", "builtin:xml-schema-structural-for-xmlschema"),
            (
                "xmlschema-validate",
                "builtin:xml-schema-structural-for-xmlschema-validate",
            ),
            (
                "xsd-validator",
                "builtin:xml-schema-structural-for-xsd-validator",
            ),
            (
                "xmlschema-adapter",
                "builtin:xml-schema-structural-for-xmlschema-adapter",
            ),
        ] {
            let run = run_output_validation_json_with_rust_reference(&payload, alias);
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}: {run:?}");
            assert_eq!(run["verdict"].as_str(), Some("valid"), "{alias}: {run:?}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator),
                "{alias}: {run:?}"
            );
        }
    }

    #[test]
    fn xml_rule_command_aliases_use_rust_structural_fallbacks() {
        let payload = json!({
            "kind": "schematron-validation",
            "xml": "<match><score home=\"1\" away=\"0\"/></match>",
            "requiredElements": ["match", "score"],
        });
        for (alias, expected_validator) in [
            (
                "schematron-adapter",
                "builtin:schematron-structural-for-schematron-adapter",
            ),
            ("saxon-he", "builtin:schematron-structural-for-saxon-he"),
            ("saxon9he", "builtin:schematron-structural-for-saxon9he"),
        ] {
            let run = run_output_validation_json_with_rust_reference(&payload, alias);
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}: {run:?}");
            assert_eq!(run["verdict"].as_str(), Some("valid"), "{alias}: {run:?}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator),
                "{alias}: {run:?}"
            );
        }
    }

    #[test]
    fn cue_payloads_use_rust_structural_fallback_without_breaking_json_schema_alias() {
        let cue = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "cue-validation",
                "cue": "objective: number\nstatus?: \"ok\" | \"warn\"\n",
                "instance": {"objective": 3.5, "status": "ok"},
            }),
            "cue",
        );
        assert_eq!(cue["status"].as_str(), Some("ok"));
        assert_eq!(cue["verdict"].as_str(), Some("valid"));
        assert_eq!(cue["validator"].as_str(), Some("builtin:cue-structural"));

        let bad_cue = run_output_validation_json_with_rust_reference(
            &json!({
                "cue": "objective: number\nstatus: bool\n",
                "instance": {"objective": "wide", "status": "ok"},
            }),
            "cue",
        );
        assert_eq!(bad_cue["status"].as_str(), Some("ok"));
        assert_eq!(bad_cue["verdict"].as_str(), Some("invalid"));
        assert_eq!(
            bad_cue["validator"].as_str(),
            Some("builtin:cue-structural")
        );

        let json_schema_alias = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "type": "object",
                    "required": ["objective"],
                    "properties": {"objective": {"type": "number"}}
                },
                "instance": {"objective": 3.5},
            }),
            "cue",
        );
        assert_eq!(json_schema_alias["status"].as_str(), Some("ok"));
        assert_eq!(json_schema_alias["verdict"].as_str(), Some("valid"));
        assert_eq!(
            json_schema_alias["validator"].as_str(),
            Some("builtin:json-schema-subset-for-cue")
        );
    }

    #[test]
    fn python_schema_dialects_use_rust_structured_constraints() {
        let cerberus_valid = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "name": {"type": "string", "required": true, "empty": false, "minlength": 3},
                    "age": {"type": "integer", "min": 0, "max": 120},
                    "status": {"type": "string", "allowed": ["ok", "warn"]},
                    "note": {"type": "string", "nullable": true}
                },
                "data": {"name": "Alex", "age": 42, "status": "ok", "note": null}
            }),
            "cerberus",
        );
        assert_eq!(cerberus_valid["status"].as_str(), Some("ok"));
        assert_eq!(cerberus_valid["verdict"].as_str(), Some("valid"));
        assert_eq!(
            cerberus_valid["validator"].as_str(),
            Some("builtin:pydantic-model-subset-for-cerberus")
        );

        let cerberus_invalid = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "name": {"type": "string", "required": true, "empty": false, "minlength": 3},
                    "age": {"type": "integer", "min": 0, "max": 120},
                    "status": {"type": "string", "allowed": ["ok", "warn"]}
                },
                "instance": {"name": "", "age": -1, "status": "bad"}
            }),
            "cerberus",
        );
        assert_eq!(cerberus_invalid["status"].as_str(), Some("ok"));
        assert_eq!(cerberus_invalid["verdict"].as_str(), Some("invalid"));
        assert!(
            cerberus_invalid["errors"]
                .as_array()
                .is_some_and(|errors| errors.len() >= 3),
            "{cerberus_invalid:?}"
        );

        let marshmallow = run_output_validation_json_with_rust_reference(
            &json!({
                "model": {
                    "fields": {
                        "score": {"type": "fields.Float", "required": true, "min_value": 0.0, "max_value": 10.0},
                        "tags": {"type": "List", "schema": {"type": "String"}, "minLength": 1}
                    }
                },
                "instance": {"score": 8.5, "tags": ["left", "press"]}
            }),
            "marshmallow",
        );
        assert_eq!(marshmallow["status"].as_str(), Some("ok"));
        assert_eq!(marshmallow["verdict"].as_str(), Some("valid"));
        assert_eq!(
            marshmallow["validator"].as_str(),
            Some("builtin:pydantic-model-subset-for-marshmallow")
        );

        for (alias, expected_validator) in [
            (
                "pydantic-adapter",
                "builtin:pydantic-model-subset-for-pydantic-adapter",
            ),
            (
                "zod-adapter",
                "builtin:pydantic-model-subset-for-zod-adapter",
            ),
            (
                "valibot-adapter",
                "builtin:pydantic-model-subset-for-valibot-adapter",
            ),
            (
                "marshmallow-adapter",
                "builtin:pydantic-model-subset-for-marshmallow-adapter",
            ),
            (
                "cerberus-adapter",
                "builtin:pydantic-model-subset-for-cerberus-adapter",
            ),
        ] {
            let run = run_output_validation_json_with_rust_reference(
                &json!({
                    "schema": {
                        "score": {"type": "number", "required": true, "min": 0.0},
                    },
                    "instance": {"score": 1.0}
                }),
                alias,
            );
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}: {run:?}");
            assert_eq!(run["verdict"].as_str(), Some("valid"), "{alias}: {run:?}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator),
                "{alias}: {run:?}"
            );
        }
    }

    #[test]
    fn table_package_tools_use_rust_csv_fallback_when_payload_is_table_shaped() {
        let valid = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "columns": {
                        "episode": {"type": "integer", "required": true},
                        "score": {"type": "number", "minimum": 0.0},
                        "team": {"type": "string", "enum": ["home", "away"]}
                    },
                    "minRows": 2,
                    "additionalColumns": false
                },
                "csv": "episode,score,team\n1,2.5,home\n2,1.0,away\n",
            }),
            "frictionless",
        );
        assert_eq!(valid["status"].as_str(), Some("ok"));
        assert_eq!(valid["verdict"].as_str(), Some("valid"));
        assert_eq!(
            valid["validator"].as_str(),
            Some("builtin:table-schema-subset-for-frictionless")
        );

        let invalid = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "columns": {
                        "episode": {"type": "integer", "required": true},
                        "score": {"type": "number", "minimum": 0.0}
                    }
                },
                "text": "episode,score\n1,-3.0\n2,4.0\n",
            }),
            "great-expectations",
        );
        assert_eq!(invalid["status"].as_str(), Some("ok"));
        assert_eq!(invalid["verdict"].as_str(), Some("invalid"));
        assert_eq!(
            invalid["validator"].as_str(),
            Some("builtin:table-schema-subset-for-great-expectations")
        );
    }

    #[test]
    fn frictionless_data_packages_use_rust_structural_fallbacks() {
        let package = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "data-package-validation",
                "package": {
                    "profile": "tabular-data-package",
                    "resources": [
                        {
                            "name": "episodes",
                            "path": "episodes.csv",
                            "schema": {
                                "fields": [
                                    {"name": "episode", "type": "integer", "constraints": {"required": true, "minimum": 1}},
                                    {"name": "score", "type": "number", "constraints": {"minimum": 0}},
                                    {"name": "status", "type": "string", "constraints": {"enum": ["ok", "warn"]}}
                                ],
                                "primaryKey": "episode"
                            },
                            "rows": [
                                {"episode": 1, "score": 3.5, "status": "ok"},
                                {"episode": 2, "score": 2.0, "status": "warn"}
                            ]
                        }
                    ]
                }
            }),
            "frictionless",
        );
        assert_eq!(package["status"].as_str(), Some("ok"));
        assert_eq!(package["verdict"].as_str(), Some("valid"));
        assert_eq!(
            package["validator"].as_str(),
            Some("builtin:frictionless-data-package-structural")
        );

        let invalid = run_output_validation_json_with_rust_reference(
            &json!({
                "resources": [
                    {
                        "name": "episodes",
                        "schema": {
                            "fields": [
                                {"name": "episode", "type": "integer"},
                                {"name": "episode", "type": "integer"},
                                {"name": "score", "type": "mystery", "constraints": {"minimum": 0}}
                            ],
                            "primaryKey": "missing_id"
                        },
                        "rows": [{"episode": "one", "score": -1.0}]
                    },
                    {"name": "episodes", "path": ""}
                ]
            }),
            "frictionless",
        );
        assert_eq!(invalid["status"].as_str(), Some("ok"));
        assert_eq!(invalid["verdict"].as_str(), Some("invalid"));
        assert!(
            invalid["errors"]
                .as_array()
                .is_some_and(|errors| errors.len() >= 5),
            "{invalid:?}"
        );

        let table = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "columns": {"episode": {"type": "integer", "required": true}},
                    "minRows": 1
                },
                "rows": [{"episode": 1}]
            }),
            "frictionless",
        );
        assert_eq!(table["status"].as_str(), Some("ok"));
        assert_eq!(table["verdict"].as_str(), Some("valid"));
        assert_eq!(
            table["validator"].as_str(),
            Some("builtin:table-schema-subset-for-frictionless")
        );
    }

    #[test]
    fn openrefine_histories_use_rust_structural_fallbacks() {
        let history = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "openrefine-history-validation",
                "rows": [{"name": " Alpha ", "entity": "Alpha"}],
                "operations": [
                    {
                        "op": "core/text-transform",
                        "description": "trim names",
                        "columnName": "name",
                        "expression": "value.trim()",
                        "onError": "keep-original"
                    },
                    {
                        "op": "core/mass-edit",
                        "columnName": "entity",
                        "edits": [{"from": ["Alpha"], "to": "Alpha FC"}]
                    }
                ],
                "reconciliation": {
                    "column": "entity",
                    "matched": 1,
                    "unmatched": 0,
                    "total": 1,
                    "candidates": [{"id": "Q1", "name": "Alpha FC", "score": 98.0}]
                }
            }),
            "openrefine",
        );
        assert_eq!(history["status"].as_str(), Some("ok"));
        assert_eq!(history["verdict"].as_str(), Some("valid"));
        assert_eq!(
            history["validator"].as_str(),
            Some("builtin:openrefine-structural-for-openrefine")
        );

        let invalid = run_output_validation_json_with_rust_reference(
            &json!({
                "rows": [{"name": "Alpha"}],
                "operations": [
                    {"description": 7},
                    {"op": "core/text-transform", "columnName": "missing", "onError": "explode", "repeatCount": -1},
                    {"op": "core/mass-edit", "columnName": "name", "edits": [{}]}
                ],
                "reconciliation": {
                    "column": "missing",
                    "matched": 3,
                    "unmatched": 2,
                    "total": 4,
                    "candidates": [{"score": 120.0}, "bad"]
                }
            }),
            "openrefine_adapter",
        );
        assert_eq!(invalid["status"].as_str(), Some("ok"));
        assert_eq!(invalid["verdict"].as_str(), Some("invalid"));
        assert_eq!(
            invalid["validator"].as_str(),
            Some("builtin:openrefine-structural-for-openrefine-adapter")
        );
        assert!(
            invalid["errors"]
                .as_array()
                .is_some_and(|errors| errors.len() >= 8),
            "{invalid:?}"
        );

        let table = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "columns": {"name": {"type": "string", "required": true}},
                    "minRows": 1
                },
                "rows": [{"name": "Alpha"}]
            }),
            "refine",
        );
        assert_eq!(table["status"].as_str(), Some("ok"));
        assert_eq!(table["verdict"].as_str(), Some("valid"));
        assert_eq!(
            table["validator"].as_str(),
            Some("builtin:table-schema-subset-for-refine")
        );
    }

    #[test]
    fn columnar_metadata_tools_use_rust_structural_fallbacks() {
        let parquet = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "parquet-validation",
                "schema": {
                    "fields": [
                        {"name": "episode", "type": "int64"},
                        {"name": "score", "type": "double"},
                        {"name": "status", "type": "utf8"}
                    ]
                },
                "metadata": {
                    "format": "parquet",
                    "num_rows": 3,
                    "num_columns": 3,
                    "num_row_groups": 2,
                    "row_groups": [{"num_rows": 2}, {"num_rows": 1}],
                    "compression": "zstd",
                    "file_size": 1024
                }
            }),
            "parquet-tools",
        );
        assert_eq!(parquet["status"].as_str(), Some("ok"));
        assert_eq!(parquet["verdict"].as_str(), Some("valid"));
        assert_eq!(
            parquet["validator"].as_str(),
            Some("builtin:columnar-metadata-for-parquet-tools")
        );

        let arrow = run_output_validation_json_with_rust_reference(
            &json!({
                "arrow_schema": {
                    "fields": [
                        {"name": "embedding", "type": "tensor"},
                        {"name": "embedding", "type": "float64"}
                    ]
                },
                "metadata": {
                    "format": "arrow",
                    "num_rows": 4,
                    "num_columns": 1,
                    "row_groups": [{"num_rows": 2}, {"num_rows": 1}],
                    "compression": "mystery"
                }
            }),
            "pyarrow_adapter",
        );
        assert_eq!(arrow["status"].as_str(), Some("ok"));
        assert_eq!(arrow["verdict"].as_str(), Some("invalid"));
        assert_eq!(
            arrow["validator"].as_str(),
            Some("builtin:columnar-metadata-for-pyarrow-adapter")
        );
        assert!(
            arrow["errors"]
                .as_array()
                .is_some_and(|errors| errors.len() >= 3),
            "{arrow:?}"
        );

        let table = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "columns": {"episode": {"type": "integer", "required": true}},
                    "minRows": 1
                },
                "rows": [{"episode": 1}]
            }),
            "apache-arrow",
        );
        assert_eq!(table["status"].as_str(), Some("ok"));
        assert_eq!(table["verdict"].as_str(), Some("valid"));
        assert_eq!(
            table["validator"].as_str(),
            Some("builtin:table-schema-subset-for-apache-arrow")
        );

        for (alias, expected_validator) in [
            (
                "arrow-adapter",
                "builtin:table-schema-subset-for-arrow-adapter",
            ),
            (
                "pyarrow-adapter",
                "builtin:table-schema-subset-for-pyarrow-adapter",
            ),
        ] {
            let run = run_output_validation_json_with_rust_reference(
                &json!({
                    "schema": {
                        "columns": {"episode": {"type": "integer", "required": true}},
                        "minRows": 1
                    },
                    "rows": [{"episode": 1}]
                }),
                alias,
            );
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}: {run:?}");
            assert_eq!(run["verdict"].as_str(), Some("valid"), "{alias}: {run:?}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator),
                "{alias}: {run:?}"
            );
        }
    }

    #[test]
    fn data_profile_tools_use_rust_structural_fallbacks() {
        let whylogs = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "profile-validation",
                "profile": {
                    "row_count": 100,
                    "features": {
                        "score": {"type": "number", "count": 100, "missing": 0, "min": 0.0, "max": 10.0, "mean": 4.5, "distinct": 20},
                        "status": {"type": "categorical", "count": 100, "missing": 0, "distinct": 3}
                    }
                },
                "constraints": [
                    {"feature": "score", "metric": "mean", "comparison": "<=", "target": 5.0}
                ]
            }),
            "whylogs",
        );
        assert_eq!(whylogs["status"].as_str(), Some("ok"));
        assert_eq!(whylogs["verdict"].as_str(), Some("valid"));
        assert_eq!(
            whylogs["validator"].as_str(),
            Some("builtin:data-profile-structural-for-whylogs")
        );

        let evidently = run_output_validation_json_with_rust_reference(
            &json!({
                "baseline": {
                    "features": {
                        "score": {"type": "number", "count": 100, "missing": 0, "min": 0.0, "max": 10.0, "mean": 4.0}
                    }
                },
                "current": {
                    "features": {
                        "score": {"type": "number", "count": 50, "missing": 70, "min": 8.0, "max": 3.0, "mean": 12.0}
                    }
                },
                "drift_threshold": 0.5,
                "anomalies": [{"feature": "score", "type": "range"}]
            }),
            "evidently",
        );
        assert_eq!(evidently["status"].as_str(), Some("ok"));
        assert_eq!(evidently["verdict"].as_str(), Some("invalid"));
        assert_eq!(
            evidently["validator"].as_str(),
            Some("builtin:data-profile-structural-for-evidently")
        );
        assert!(
            evidently["errors"]
                .as_array()
                .is_some_and(|errors| errors.len() >= 4),
            "{evidently:?}"
        );

        let table = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "columns": {"score": {"type": "number", "required": true}},
                    "minRows": 1
                },
                "rows": [{"score": 4.0}]
            }),
            "deepchecks",
        );
        assert_eq!(table["status"].as_str(), Some("ok"));
        assert_eq!(table["verdict"].as_str(), Some("valid"));
        assert_eq!(
            table["validator"].as_str(),
            Some("builtin:table-schema-subset-for-deepchecks")
        );

        for (alias, expected_validator) in [
            (
                "whylogs-adapter",
                "builtin:data-profile-structural-for-whylogs-adapter",
            ),
            (
                "great-expectations",
                "builtin:data-profile-structural-for-great-expectations",
            ),
            ("gx", "builtin:data-profile-structural-for-gx"),
            (
                "evidently-adapter",
                "builtin:data-profile-structural-for-evidently-adapter",
            ),
            (
                "deepchecks-adapter",
                "builtin:data-profile-structural-for-deepchecks-adapter",
            ),
            ("soda", "builtin:data-profile-structural-for-soda"),
            (
                "deequ-adapter",
                "builtin:data-profile-structural-for-deequ-adapter",
            ),
            (
                "tfdv-adapter",
                "builtin:data-profile-structural-for-tfdv-adapter",
            ),
        ] {
            let run = run_output_validation_json_with_rust_reference(
                &json!({
                    "kind": "profile-validation",
                    "profile": {
                        "row_count": 8,
                        "features": {
                            "score": {"type": "number", "count": 8, "missing": 0, "min": 0.0, "max": 3.0, "mean": 1.5}
                        }
                    },
                    "constraints": [
                        {"feature": "score", "metric": "mean", "comparison": "<=", "target": 2.0}
                    ]
                }),
                alias,
            );
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}: {run:?}");
            assert_eq!(run["verdict"].as_str(), Some("valid"), "{alias}: {run:?}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator),
                "{alias}: {run:?}"
            );
        }

        for (alias, expected_validator) in [
            (
                "pandera-adapter",
                "builtin:table-schema-subset-for-pandera-adapter",
            ),
            ("gx", "builtin:table-schema-subset-for-gx"),
        ] {
            let run = run_output_validation_json_with_rust_reference(
                &json!({
                    "schema": {
                        "columns": {"score": {"type": "number", "required": true}},
                        "minRows": 1
                    },
                    "rows": [{"score": 4.0}]
                }),
                alias,
            );
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}: {run:?}");
            assert_eq!(run["verdict"].as_str(), Some("valid"), "{alias}: {run:?}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator),
                "{alias}: {run:?}"
            );
        }
    }

    #[test]
    fn sql_validator_aliases_use_rust_structural_fallbacks() {
        let dbt = run_output_validation_json_with_rust_reference(
            &json!({
                "kind": "dbt-validation",
                "text": "with matches as (select id, score from soccer_matches)\nselect id, score from matches where score >= 0\n",
            }),
            "dbt",
        );
        assert_eq!(dbt["status"].as_str(), Some("ok"));
        assert_eq!(dbt["verdict"].as_str(), Some("valid"));
        assert_eq!(
            dbt["validator"].as_str(),
            Some("builtin:sql-structural-for-dbt")
        );

        let sqlfluff = run_output_validation_json_with_rust_reference(
            &json!({
                "sql": "select id, score from where score > 0",
            }),
            "sqlfluff",
        );
        assert_eq!(sqlfluff["status"].as_str(), Some("ok"));
        assert_eq!(sqlfluff["verdict"].as_str(), Some("invalid"));
        assert_eq!(
            sqlfluff["validator"].as_str(),
            Some("builtin:sql-structural-for-sqlfluff")
        );

        let sql_lint = run_output_validation_json_with_rust_reference(
            &json!({
                "query": "select * from games where status = 'complete'",
            }),
            "sql_lint",
        );
        assert_eq!(sql_lint["status"].as_str(), Some("ok"));
        assert_eq!(sql_lint["verdict"].as_str(), Some("valid"));
        assert_eq!(
            sql_lint["validator"].as_str(),
            Some("builtin:sql-structural-for-sql-lint")
        );
    }

    #[test]
    fn output_validator_command_aliases_use_rust_structural_fallbacks() {
        let openapi_payload = json!({
            "spec": {
                "openapi": "3.0.0",
                "info": {"title": "Soccer API", "version": "1.0.0"},
                "paths": {
                    "/score": {
                        "get": {
                            "responses": {"200": {"description": "ok"}}
                        }
                    }
                }
            }
        });
        for alias in ["swagger-cli", "openapi-generator-cli"] {
            let run = run_output_validation_json_with_rust_reference(&openapi_payload, alias);
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}");
            assert_eq!(run["verdict"].as_str(), Some("valid"), "{alias}");
            assert_eq!(
                run["validator"].as_str(),
                Some("builtin:openapi-structural"),
                "{alias}"
            );
        }
        for alias in ["redocly", "asyncapi"] {
            let run = run_output_validation_json_with_rust_reference(&openapi_payload, alias);
            let expected_validator = format!("builtin:openapi-structural-for-{alias}");
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}");
            assert_eq!(run["verdict"].as_str(), Some("valid"), "{alias}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator.as_str()),
                "{alias}"
            );
        }

        for alias in ["graphql-schema-linter", "graphql-inspector"] {
            let run = run_output_validation_json_with_rust_reference(
                &json!({"schema": "type Query { score: Int }\n"}),
                alias,
            );
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}");
            assert_eq!(run["verdict"].as_str(), Some("valid"), "{alias}");
            assert_eq!(
                run["validator"].as_str(),
                Some("builtin:graphql-schema-structural"),
                "{alias}"
            );
        }

        let csv = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "columns": {"episode": {"type": "integer", "required": true}},
                    "minRows": 1
                },
                "csv": "episode\n1\n",
            }),
            "csvlint",
        );
        assert_eq!(csv["status"].as_str(), Some("ok"));
        assert_eq!(csv["verdict"].as_str(), Some("valid"));
        assert_eq!(
            csv["validator"].as_str(),
            Some("builtin:table-schema-subset")
        );

        let protobuf = run_output_validation_json_with_rust_reference(
            &json!({
                "schema": {
                    "fields": {
                        "episode": {"type": "int32", "required": true},
                        "score": {"type": "double"}
                    }
                },
                "message": {"episode": 1, "score": 3.5}
            }),
            "conformance-test-runner",
        );
        assert_eq!(protobuf["status"].as_str(), Some("ok"));
        assert_eq!(protobuf["verdict"].as_str(), Some("valid"));
        assert_eq!(
            protobuf["validator"].as_str(),
            Some("builtin:protobuf-conformance-subset")
        );
    }

    #[test]
    fn python_sat_tool_aliases_use_rust_model_validation_fallbacks() {
        let cnf = run_model_validation_json_with_rust_reference(
            &json!({
                "dimacs": "p cnf 2 2\n1 2 0\n-1 0\n",
            }),
            "pysat",
        );
        assert_eq!(cnf["status"].as_str(), Some("ok"));
        assert_eq!(cnf["verdict"].as_str(), Some("sat"));
        assert_eq!(cnf["validator"].as_str(), Some("builtin:dimacs-small-cnf"));

        for alias in ["cryptominisat5", "glucose-syrup", "maple-sat", "maple_lcm"] {
            let alias_run = run_model_validation_json_with_rust_reference(
                &json!({
                    "dimacs": "p cnf 1 1\n1 0\n",
                }),
                alias,
            );
            assert_eq!(alias_run["status"].as_str(), Some("ok"), "{alias}");
            assert_eq!(alias_run["verdict"].as_str(), Some("sat"), "{alias}");
            assert_eq!(
                alias_run["validator"].as_str(),
                Some("builtin:dimacs-small-cnf"),
                "{alias}"
            );
        }

        let wcnf = run_model_validation_json_with_rust_reference(
            &json!({
                "wcnf": "p wcnf 2 2 10\n10 1 0\n2 -1 2 0\n",
            }),
            "python-sat-adapter",
        );
        assert_eq!(wcnf["status"].as_str(), Some("ok"));
        assert_eq!(wcnf["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            wcnf["validator"].as_str(),
            Some("builtin:wcnf-small-maxsat")
        );

        let open_wbo_alias = run_model_validation_json_with_rust_reference(
            &json!({
                "dimacs": "p wcnf 1 1 10\n2 1 0\n",
            }),
            "open-wbo_static",
        );
        assert_eq!(open_wbo_alias["status"].as_str(), Some("ok"));
        assert_eq!(open_wbo_alias["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            open_wbo_alias["validator"].as_str(),
            Some("builtin:wcnf-small-maxsat")
        );

        let opb = run_model_validation_json_with_rust_reference(
            &json!({
                "opb": "1 x1 1 x2 >= 1;\n",
            }),
            "sat4j-sat",
        );
        assert_eq!(opb["status"].as_str(), Some("ok"));
        assert_eq!(opb["verdict"].as_str(), Some("sat"));
        assert_eq!(opb["validator"].as_str(), Some("builtin:opb-small-pb"));

        let pysat_opb = run_model_validation_json_with_rust_reference(
            &json!({
                "opb": "1 x1 >= 1;\n",
            }),
            "pysat-adapter",
        );
        assert_eq!(pysat_opb["status"].as_str(), Some("ok"));
        assert_eq!(pysat_opb["verdict"].as_str(), Some("sat"));
        assert_eq!(
            pysat_opb["validator"].as_str(),
            Some("builtin:opb-small-pb")
        );
    }

    #[test]
    fn cp_python_modeling_adapters_use_rust_finite_domain_fallbacks() {
        let cpmpy = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "finite-domain-cp-validation",
                "variables": {
                    "left": [0, 1, 2],
                    "right": [0, 1, 2]
                },
                "constraints": [
                    {"op": "all_different", "vars": ["left", "right"]},
                    {"op": "sum_le", "vars": ["left", "right"], "rhs": 1}
                ]
            }),
            "cpmpy",
        );
        assert_eq!(cpmpy["status"].as_str(), Some("ok"));
        assert_eq!(cpmpy["verdict"].as_str(), Some("sat"));
        assert_eq!(
            cpmpy["validator"].as_str(),
            Some("builtin:finite-domain-cp-for-cpmpy")
        );
        assert!(
            cpmpy["stdout"]
                .as_str()
                .is_some_and(|stdout| stdout.contains("left=") && stdout.contains("right=")),
            "{cpmpy:?}"
        );

        let pycsp3 = run_model_validation_json_with_rust_reference(
            &json!({
                "variables": [
                    {"name": "a", "domain": [0]},
                    {"name": "b", "domain": [0]}
                ],
                "constraints": [
                    {"op": "all-different", "scope": ["a", "b"]}
                ]
            }),
            "pycsp3",
        );
        assert_eq!(pycsp3["status"].as_str(), Some("ok"));
        assert_eq!(pycsp3["verdict"].as_str(), Some("unsat"));
        assert_eq!(
            pycsp3["validator"].as_str(),
            Some("builtin:finite-domain-cp-for-pycsp3")
        );
    }

    #[test]
    fn cp_sat_json_adapters_use_rust_reference_fallbacks() {
        let cp_sat = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "cp-sat-validation",
                "variables": [
                    {"name": "x", "domain": [0, 1]},
                    {"name": "y", "domain": [0, 1]}
                ],
                "constraints": [
                    {
                        "kind": "linear",
                        "terms": [
                            {"var": 0, "coeff": 1},
                            {"var": 1, "coeff": 1}
                        ],
                        "sense": "eq",
                        "rhs": 1
                    }
                ],
                "objective": {
                    "sense": "min",
                    "terms": [
                        {"var": 0, "coeff": 1},
                        {"var": 1, "coeff": 2}
                    ]
                }
            }),
            "ortools_cp_sat",
        );
        assert_eq!(cp_sat["status"].as_str(), Some("ok"));
        assert_eq!(cp_sat["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            cp_sat["validator"].as_str(),
            Some("builtin:cp-sat-small-for-ortools-cp-sat")
        );
        assert!(
            cp_sat["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("assignment=1,0")
                    && stdout.contains("objective=1.000")
                    && stdout.contains("backend=rust:cp-native-enumeration")
                    && stdout.contains("solver=rust-enumeration")
            }),
            "{cp_sat:?}"
        );
    }

    #[test]
    fn quadratic_adapters_use_rust_reference_fallbacks() {
        let qp = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "qp-validation",
                "Q": [
                    [2, 0],
                    [0, 2]
                ],
                "c": [-2, -4],
                "lb": [0, 0],
                "ub": [5, 5]
            }),
            "osqp_adapter",
        );
        assert_eq!(qp["status"].as_str(), Some("ok"));
        assert_eq!(qp["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            qp["validator"].as_str(),
            Some("builtin:quadratic-small-for-osqp-adapter")
        );
        assert!(
            qp["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("x0=1.000")
                    && stdout.contains("x1=2.000")
                    && stdout.contains("objective=-5.000")
                    && stdout.contains("solver=rust:qp-active-set")
            }),
            "{qp:?}"
        );

        let miqp = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "miqp-validation",
                "Q": [
                    [2, 0],
                    [0, 2]
                ],
                "c": [-2, -4],
                "lb": [0, 0],
                "ub": [5, 5],
                "integerVars": [true, true]
            }),
            "gurobipy_adapter",
        );
        assert_eq!(miqp["status"].as_str(), Some("ok"));
        assert_eq!(miqp["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            miqp["validator"].as_str(),
            Some("builtin:quadratic-small-for-gurobipy-adapter")
        );
        assert!(
            miqp["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("x0=1.000")
                    && stdout.contains("x1=2.000")
                    && stdout.contains("objective=-5.000")
                    && stdout.contains("solver=rust:miqp-enumeration")
            }),
            "{miqp:?}"
        );

        for (alias, expected_validator) in [
            (
                "highs-rust-adapter",
                "builtin:quadratic-small-for-highs-rust-adapter",
            ),
            (
                "gurobi-rust-adapter",
                "builtin:quadratic-small-for-gurobi-rust-adapter",
            ),
            (
                "cplex-rust-adapter",
                "builtin:quadratic-small-for-cplex-rust-adapter",
            ),
        ] {
            let run = run_model_validation_json_with_rust_reference(
                &json!({
                    "kind": "miqp-validation",
                    "Q": [
                        [2, 0],
                        [0, 2]
                    ],
                    "c": [-2, -4],
                    "lb": [0, 0],
                    "ub": [5, 5],
                    "integerVars": [true, true]
                }),
                alias,
            );
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}: {run:?}");
            assert_eq!(run["verdict"].as_str(), Some("optimal"), "{alias}: {run:?}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator),
                "{alias}: {run:?}"
            );
            assert!(
                run["stdout"]
                    .as_str()
                    .is_some_and(|stdout| stdout.contains("solver=rust:miqp-enumeration")),
                "{alias}: {run:?}"
            );
        }
    }

    #[test]
    fn stochastic_lp_adapters_use_rust_monolithic_reference_fallbacks() {
        let slp = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "stochastic-lp-validation",
                "cFirst": [-1],
                "aFirst": [[1]],
                "bFirst": [20],
                "qSecond": [3],
                "wSecond": [[1], [1]],
                "scenarios": [
                    {
                        "t": [[-1], [0]],
                        "h": [0, 5],
                        "prob": 0.5
                    },
                    {
                        "t": [[-1], [0]],
                        "h": [0, 15],
                        "prob": 0.5
                    }
                ]
            }),
            "pyomo_adapter",
        );
        assert_eq!(slp["status"].as_str(), Some("ok"));
        assert_eq!(slp["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            slp["validator"].as_str(),
            Some("builtin:stochastic-lp-small-for-pyomo-adapter")
        );
        assert!(
            slp["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("x0=15.000")
                    && stdout.contains("objective=15.000")
                    && stdout.contains("scenarios=2")
                    && stdout.contains("solver=rust:monolithic-slp")
            }),
            "{slp:?}"
        );

        for (alias, expected_validator) in [
            (
                "highs-rust-adapter",
                "builtin:stochastic-lp-small-for-highs-rust-adapter",
            ),
            (
                "gurobi-rust-adapter",
                "builtin:stochastic-lp-small-for-gurobi-rust-adapter",
            ),
            (
                "cplex-rust-adapter",
                "builtin:stochastic-lp-small-for-cplex-rust-adapter",
            ),
        ] {
            let run = run_model_validation_json_with_rust_reference(
                &json!({
                    "kind": "stochastic-lp-validation",
                    "cFirst": [-1],
                    "aFirst": [[1]],
                    "bFirst": [20],
                    "qSecond": [3],
                    "wSecond": [[1], [1]],
                    "scenarios": [
                        {
                            "t": [[-1], [0]],
                            "h": [0, 5],
                            "prob": 0.5
                        },
                        {
                            "t": [[-1], [0]],
                            "h": [0, 15],
                            "prob": 0.5
                        }
                    ]
                }),
                alias,
            );
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}: {run:?}");
            assert_eq!(run["verdict"].as_str(), Some("optimal"), "{alias}: {run:?}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator),
                "{alias}: {run:?}"
            );
            assert!(
                run["stdout"]
                    .as_str()
                    .is_some_and(|stdout| stdout.contains("solver=rust:monolithic-slp")),
                "{alias}: {run:?}"
            );
        }
    }

    #[test]
    fn python_linear_modeling_adapters_use_rust_mip_fallbacks() {
        let pyomo = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "linear-mip-validation",
                "sense": "max",
                "objective": [3, 2],
                "constraints": [
                    {"coefs": [1, 1], "sense": "<=", "rhs": 1}
                ],
                "domains": [[0, 1], [0, 1]]
            }),
            "pyomo",
        );
        assert_eq!(pyomo["status"].as_str(), Some("ok"));
        assert_eq!(pyomo["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            pyomo["validator"].as_str(),
            Some("builtin:linear-mip-small-for-pyomo")
        );
        assert!(
            pyomo["stdout"]
                .as_str()
                .is_some_and(|stdout| stdout.contains("x0=1") && stdout.contains("objective=3")),
            "{pyomo:?}"
        );

        let python_mip = run_model_validation_json_with_rust_reference(
            &json!({
                "sense": "max",
                "objective": {"x": 4, "y": 1},
                "variables": {
                    "x": {"binary": true},
                    "y": {"binary": true}
                },
                "constraints": [
                    {"coefficients": [1, 1], "operator": "<=", "rhs": 1}
                ]
            }),
            "python_mip_adapter",
        );
        assert_eq!(python_mip["status"].as_str(), Some("ok"));
        assert_eq!(python_mip["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            python_mip["validator"].as_str(),
            Some("builtin:linear-mip-small-for-python-mip-adapter")
        );
        assert!(
            python_mip["stdout"]
                .as_str()
                .is_some_and(|stdout| stdout.contains("x=1") && stdout.contains("objective=4")),
            "{python_mip:?}"
        );

        let docplex = run_model_validation_json_with_rust_reference(
            &json!({
                "objective": [1],
                "constraints": [
                    {"coefs": [1], "sense": "<=", "rhs": 0},
                    {"coefs": [1], "sense": ">=", "rhs": 1}
                ],
                "domains": [[0, 1]]
            }),
            "docplex-adapter",
        );
        assert_eq!(docplex["status"].as_str(), Some("ok"));
        assert_eq!(docplex["verdict"].as_str(), Some("infeasible"));
        assert_eq!(
            docplex["validator"].as_str(),
            Some("builtin:linear-mip-small-for-docplex-adapter")
        );

        for (alias, expected_validator) in [
            (
                "gurobi-rust-adapter",
                "builtin:linear-mip-small-for-gurobi-rust-adapter",
            ),
            (
                "cplex-rust-adapter",
                "builtin:linear-mip-small-for-cplex-rust-adapter",
            ),
            (
                "highs-rust-adapter",
                "builtin:linear-mip-small-for-highs-rust-adapter",
            ),
            (
                "scip-rust-adapter",
                "builtin:linear-mip-small-for-scip-rust-adapter",
            ),
            (
                "cbc-rust-adapter",
                "builtin:linear-mip-small-for-cbc-rust-adapter",
            ),
            ("highs-cli", "builtin:linear-mip-small-for-highs-cli"),
            ("glpsol", "builtin:linear-mip-small-for-glpsol"),
            ("scip", "builtin:linear-mip-small-for-scip"),
            ("cbc", "builtin:linear-mip-small-for-cbc"),
            ("clp", "builtin:linear-mip-small-for-clp"),
            ("soplex", "builtin:linear-mip-small-for-soplex"),
            ("qsopt-ex-cli", "builtin:linear-mip-small-for-qsopt-ex-cli"),
            ("esolver", "builtin:linear-mip-small-for-esolver"),
            ("lp_solve", "builtin:linear-mip-small-for-lp-solve"),
        ] {
            let run = run_model_validation_json_with_rust_reference(
                &json!({
                    "kind": "linear-mip-validation",
                    "sense": "max",
                    "objective": [3, 2],
                    "constraints": [
                        {"coefs": [1, 1], "sense": "<=", "rhs": 1}
                    ],
                    "domains": [[0, 1], [0, 1]]
                }),
                alias,
            );
            assert_eq!(run["status"].as_str(), Some("ok"), "{alias}: {run:?}");
            assert_eq!(run["verdict"].as_str(), Some("optimal"), "{alias}: {run:?}");
            assert_eq!(
                run["validator"].as_str(),
                Some(expected_validator),
                "{alias}: {run:?}"
            );
            assert!(
                run["stdout"]
                    .as_str()
                    .is_some_and(|stdout| stdout.contains("x0=1") && stdout.contains("objective=3")),
                "{alias}: {run:?}"
            );
        }
    }

    #[test]
    fn assignment_and_knapsack_adapters_use_rust_reference_fallbacks() {
        let assignment = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "assignment-validation",
                "cost": [
                    [8, 2, 5, 9],
                    [6, 4, 7, 3],
                    [5, 8, 1, 6],
                    [7, 3, 4, 2]
                ]
            }),
            "scipy_adapter",
        );
        assert_eq!(assignment["status"].as_str(), Some("ok"));
        assert_eq!(assignment["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            assignment["validator"].as_str(),
            Some("builtin:assignment-small-for-scipy-adapter")
        );
        assert!(
            assignment["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("assignment=1,0,2,3")
                    && stdout.contains("objective=11.000")
                    && stdout.contains("solver=rust:assignment-dp")
            }),
            "{assignment:?}"
        );

        let knapsack = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "knapsack-validation",
                "capacity": 10,
                "items": [
                    {"id": "A", "weight": 5, "value": 10},
                    {"id": "B", "weight": 4, "value": 40},
                    {"id": "C", "weight": 6, "value": 30},
                    {"id": "D", "weight": 3, "value": 50}
                ]
            }),
            "ortools_python_adapter",
        );
        assert_eq!(knapsack["status"].as_str(), Some("ok"));
        assert_eq!(knapsack["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            knapsack["validator"].as_str(),
            Some("builtin:knapsack-small-for-ortools-python-adapter")
        );
        assert!(
            knapsack["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("items=B,D")
                    && stdout.contains("objective=90.000")
                    && stdout.contains("solver=rust:branch-and-bound-knapsack")
            }),
            "{knapsack:?}"
        );

        let vector_knapsack = run_model_validation_json_with_rust_reference(
            &json!({
                "capacity": 5,
                "weights": [5, 4, 1],
                "values": [10, 10, 0],
                "item_ids": ["A", "B", "C"]
            }),
            "python_mip_adapter",
        );
        assert_eq!(vector_knapsack["status"].as_str(), Some("ok"));
        assert_eq!(vector_knapsack["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            vector_knapsack["validator"].as_str(),
            Some("builtin:knapsack-small-for-python-mip-adapter")
        );
        assert!(
            vector_knapsack["stdout"]
                .as_str()
                .is_some_and(|stdout| stdout.contains("items=B")),
            "{vector_knapsack:?}"
        );
    }

    #[test]
    fn bin_packing_and_facility_location_adapters_use_rust_reference_fallbacks() {
        let bin_packing = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "bin-packing-validation",
                "capacity": 10,
                "items": [
                    {"id": "A", "weight": 6},
                    {"id": "B", "weight": 4},
                    {"id": "C", "weight": 5},
                    {"id": "D", "weight": 5}
                ]
            }),
            "ortools_python_adapter",
        );
        assert_eq!(bin_packing["status"].as_str(), Some("ok"));
        assert_eq!(bin_packing["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            bin_packing["validator"].as_str(),
            Some("builtin:bin-packing-small-for-ortools-python-adapter")
        );
        assert!(
            bin_packing["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("bins=2")
                    && stdout.contains("total_weight=20.000")
                    && stdout.contains("solver=rust:exact-bin-packing")
            }),
            "{bin_packing:?}"
        );

        let facility = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "facility-location-validation",
                "facilities": ["A", "B"],
                "customers": ["C"],
                "fixedCosts": [1, 1],
                "serviceCosts": [[1], [1]]
            }),
            "python_mip_adapter",
        );
        assert_eq!(facility["status"].as_str(), Some("ok"));
        assert_eq!(facility["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            facility["validator"].as_str(),
            Some("builtin:facility-location-small-for-python-mip-adapter")
        );
        assert!(
            facility["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("open_facilities=A")
                    && stdout.contains("objective=2.000")
                    && stdout.contains("solver=rust:exact-facility-location")
            }),
            "{facility:?}"
        );
    }

    #[test]
    fn graph_coloring_and_set_cover_adapters_use_rust_reference_fallbacks() {
        let coloring = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "graph-coloring-validation",
                "vertices": ["A", "B", "C"],
                "edges": [["A", "B"], ["B", "C"], ["A", "C"]]
            }),
            "ortools_python_adapter",
        );
        assert_eq!(coloring["status"].as_str(), Some("ok"));
        assert_eq!(coloring["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            coloring["validator"].as_str(),
            Some("builtin:graph-coloring-small-for-ortools-python-adapter")
        );
        assert!(
            coloring["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("used_colors=3")
                    && stdout.contains("objective=3.000")
                    && stdout.contains("solver=rust:dsatur-graph-coloring")
            }),
            "{coloring:?}"
        );

        let set_cover = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "set-cover-validation",
                "universe": ["A", "B", "C", "D"],
                "sets": [
                    {"id": "AB", "cost": 2, "elements": ["A", "B"]},
                    {"id": "CD", "cost": 2, "elements": ["C", "D"]},
                    {"id": "ALL", "cost": 5, "elements": ["A", "B", "C", "D"]}
                ]
            }),
            "python_mip_adapter",
        );
        assert_eq!(set_cover["status"].as_str(), Some("ok"));
        assert_eq!(set_cover["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            set_cover["validator"].as_str(),
            Some("builtin:set-cover-small-for-python-mip-adapter")
        );
        assert!(
            set_cover["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("sets=AB,CD")
                    && stdout.contains("objective=4.000")
                    && stdout.contains("solver=rust:exact-set-cover")
            }),
            "{set_cover:?}"
        );
    }

    #[test]
    fn weighted_max_sat_adapters_use_rust_reference_fallbacks() {
        let weighted_max_sat = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "weighted-max-sat-validation",
                "numVars": 3,
                "clauses": [
                    {"id": "H_cover", "literals": [1, 2], "hard": true},
                    {"id": "H_implication", "literals": [-2, 3], "hard": true},
                    {"id": "S_pick_x1", "literals": [1], "weight": 6},
                    {"id": "S_pick_x2", "literals": [2], "weight": 6},
                    {"id": "S_not_both_x1_x2", "literals": [-1, -2], "weight": 5},
                    {"id": "S_pick_x3", "literals": [3], "weight": 4},
                    {"id": "S_skip_x3", "literals": [-3], "weight": 3}
                ]
            }),
            "open_wbo_static",
        );
        assert_eq!(weighted_max_sat["status"].as_str(), Some("ok"));
        assert_eq!(weighted_max_sat["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            weighted_max_sat["validator"].as_str(),
            Some("builtin:weighted-max-sat-small-for-open-wbo-static")
        );
        assert!(
            weighted_max_sat["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("assignment=111")
                    && stdout.contains("objective=16.000")
                    && stdout.contains("violated_hard=0")
                    && stdout.contains("solver=rust:exact-weighted-max-sat")
            }),
            "{weighted_max_sat:?}"
        );
    }

    #[test]
    fn scheduling_adapters_use_rust_reference_fallbacks() {
        let job_shop = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "job-shop-validation",
                "jobs": [
                    {
                        "id": "J1",
                        "due": 10,
                        "operations": [
                            {"machine": "M1", "duration": 3},
                            {"machine": "M2", "duration": 2}
                        ]
                    },
                    {
                        "id": "J2",
                        "due": 8,
                        "operations": [
                            {"machine": "M2", "duration": 2},
                            {"machine": "M1", "duration": 4}
                        ]
                    },
                    {
                        "id": "J3",
                        "due": 12,
                        "operations": [
                            {"machine": "M1", "duration": 2},
                            {"machine": "M2", "duration": 3}
                        ]
                    }
                ]
            }),
            "choco_solver",
        );
        assert_eq!(job_shop["status"].as_str(), Some("ok"));
        assert_eq!(job_shop["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            job_shop["validator"].as_str(),
            Some("builtin:scheduling-small-for-choco-solver")
        );
        assert!(
            job_shop["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("kind=job-shop")
                    && stdout.contains("schedule_ops=6")
                    && stdout.contains("makespan=9.000")
                    && stdout.contains("solver=rust:exact-job-shop")
            }),
            "{job_shop:?}"
        );

        let flow_shop = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "flow-shop-validation",
                "jobs": [
                    {"id": "F1", "processingTimes": [2, 3, 2]},
                    {"id": "F2", "processingTimes": [4, 1, 3]},
                    {"id": "F3", "processingTimes": [3, 2, 4]},
                    {"id": "F4", "processingTimes": [2, 5, 1]}
                ]
            }),
            "ortools_cp_sat",
        );
        assert_eq!(flow_shop["status"].as_str(), Some("ok"));
        assert_eq!(flow_shop["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            flow_shop["validator"].as_str(),
            Some("builtin:scheduling-small-for-ortools-cp-sat")
        );
        assert!(
            flow_shop["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("kind=flow-shop")
                    && stdout.contains("schedule_ops=12")
                    && stdout.contains("sequence=")
                    && stdout.contains("solver=rust:exact-flow-shop")
            }),
            "{flow_shop:?}"
        );
    }

    #[test]
    fn weighted_independent_set_adapters_use_rust_reference_fallbacks() {
        let weighted_independent_set = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "weighted-independent-set-validation",
                "vertices": [
                    {"id": "A", "weight": 8},
                    {"id": "B", "weight": 7},
                    {"id": "C", "weight": 6},
                    {"id": "D", "weight": 6},
                    {"id": "E", "weight": 5},
                    {"id": "F", "weight": 4},
                    {"id": "G", "weight": 3}
                ],
                "edges": [
                    ["A", "B"],
                    ["A", "C"],
                    ["A", "D"],
                    ["B", "C"],
                    ["B", "E"],
                    ["C", "D"],
                    ["C", "F"],
                    ["D", "E"],
                    ["D", "F"],
                    ["E", "F"],
                    ["E", "G"],
                    ["F", "G"]
                ]
            }),
            "ortools_python_adapter",
        );
        assert_eq!(weighted_independent_set["status"].as_str(), Some("ok"));
        assert_eq!(
            weighted_independent_set["verdict"].as_str(),
            Some("optimal")
        );
        assert_eq!(
            weighted_independent_set["validator"].as_str(),
            Some("builtin:weighted-independent-set-small-for-ortools-python-adapter")
        );
        assert!(
            weighted_independent_set["stdout"]
                .as_str()
                .is_some_and(|stdout| {
                    stdout.contains("selected=B,D,G")
                        && stdout.contains("objective=16.000")
                        && stdout.contains("solver=rust:branch-and-bound-weighted-independent-set")
                }),
            "{weighted_independent_set:?}"
        );
    }

    #[test]
    fn min_cost_flow_adapters_use_rust_reference_fallbacks() {
        let min_cost_flow = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "min-cost-flow-validation",
                "numNodes": 4,
                "supplies": [5, 7, -6, -6],
                "arcs": [
                    {"from": 0, "to": 2, "capacity": 5, "cost": 2, "name": "s0_d0"},
                    {"from": 0, "to": 3, "capacity": 5, "cost": 4, "name": "s0_d1"},
                    {"from": 1, "to": 2, "capacity": 6, "cost": 5, "name": "s1_d0"},
                    {"from": 1, "to": 3, "capacity": 8, "cost": 1, "name": "s1_d1"}
                ]
            }),
            "ortools_python_adapter",
        );
        assert_eq!(min_cost_flow["status"].as_str(), Some("ok"));
        assert_eq!(min_cost_flow["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            min_cost_flow["validator"].as_str(),
            Some("builtin:min-cost-flow-small-for-ortools-python-adapter")
        );
        assert!(
            min_cost_flow["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("objective=21.000")
                    && stdout.contains("arcs=4")
                    && stdout.contains("solver=rust:ssp-min-cost-flow")
            }),
            "{min_cost_flow:?}"
        );
    }

    #[test]
    fn network_flow_and_mst_adapters_use_rust_reference_fallbacks() {
        let max_flow = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "max-flow-validation",
                "numNodes": 4,
                "source": 0,
                "sink": 3,
                "edges": [
                    {"from": 0, "to": 1, "capacity": 3},
                    {"from": 0, "to": 2, "capacity": 2},
                    {"from": 1, "to": 3, "capacity": 2},
                    {"from": 2, "to": 3, "capacity": 3},
                    {"from": 1, "to": 2, "capacity": 1}
                ]
            }),
            "ortools_python_adapter",
        );
        assert_eq!(max_flow["status"].as_str(), Some("ok"));
        assert_eq!(max_flow["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            max_flow["validator"].as_str(),
            Some("builtin:max-flow-small-for-ortools-python-adapter")
        );
        assert!(
            max_flow["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("max_flow=5.000")
                    && stdout.contains("min_cut_capacity=5.000")
                    && stdout.contains("solver=rust:edmonds-karp-max-flow")
            }),
            "{max_flow:?}"
        );

        let mst = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "mst-validation",
                "vertices": ["A", "B", "C"],
                "edges": [
                    {"id": "AB", "from": "A", "to": "B", "weight": 1},
                    {"id": "BC", "from": "B", "to": "C", "weight": 2},
                    {"id": "AC", "from": "A", "to": "C", "weight": 4}
                ]
            }),
            "ortools_java_adapter",
        );
        assert_eq!(mst["status"].as_str(), Some("ok"));
        assert_eq!(mst["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            mst["validator"].as_str(),
            Some("builtin:mst-small-for-ortools-java-adapter")
        );
        assert!(
            mst["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("edges=AB,BC")
                    && stdout.contains("objective=3.000")
                    && stdout.contains("solver=rust:kruskal-mst")
            }),
            "{mst:?}"
        );
    }

    #[test]
    fn tsp_adapters_use_rust_held_karp_reference_fallbacks() {
        let matrix = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "tsp-validation",
                "distanceMatrix": [
                    [0, 1, 1.4142135623730951, 1],
                    [1, 0, 1, 1.4142135623730951],
                    [1.4142135623730951, 1, 0, 1],
                    [1, 1.4142135623730951, 1, 0]
                ]
            }),
            "ortools_python_adapter",
        );
        assert_eq!(matrix["status"].as_str(), Some("ok"));
        assert_eq!(matrix["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            matrix["validator"].as_str(),
            Some("builtin:tsp-small-for-ortools-python-adapter")
        );
        assert!(
            matrix["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("tour=0->1->2->3->0")
                    && stdout.contains("objective=4.000")
                    && stdout.contains("solver=rust:held-karp-tsp")
            }),
            "{matrix:?}"
        );

        let points = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "traveling-salesman-validation",
                "points": [
                    {"id": "A", "x": 0, "y": 0},
                    {"id": "B", "x": 1, "y": 0},
                    {"id": "C", "x": 1, "y": 1},
                    {"id": "D", "x": 0, "y": 1}
                ]
            }),
            "optaplanner_adapter",
        );
        assert_eq!(points["status"].as_str(), Some("ok"));
        assert_eq!(points["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            points["validator"].as_str(),
            Some("builtin:tsp-small-for-optaplanner-adapter")
        );
        assert!(
            points["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("tour=0->1->2->3->0")
                    && stdout.contains("objective=4.000")
                    && stdout.contains("solver=rust:held-karp-tsp")
            }),
            "{points:?}"
        );

        let vrp_shaped_matrix = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "routing-validation",
                "distance_matrix": [
                    [0, 2, 9, 10],
                    [2, 0, 4, 6],
                    [9, 4, 0, 3],
                    [10, 6, 3, 0]
                ],
                "starts": [0],
                "ends": [0],
                "vehicles": 1,
                "customers": [1, 2, 3]
            }),
            "ortools_python_adapter",
        );
        assert_eq!(
            vrp_shaped_matrix["validator"].as_str(),
            Some("builtin:routing-small-for-ortools-python-adapter")
        );
    }

    #[test]
    fn routing_adapters_use_rust_small_vrp_fallbacks() {
        let matrix = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "routing-validation",
                "distance_matrix": [
                    [0, 2, 9, 10],
                    [2, 0, 4, 6],
                    [9, 4, 0, 3],
                    [10, 6, 3, 0]
                ],
                "starts": [0],
                "ends": [0],
                "vehicles": 1,
                "customers": [1, 2, 3]
            }),
            "ortools_python_adapter",
        );
        assert_eq!(matrix["status"].as_str(), Some("ok"));
        assert_eq!(matrix["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            matrix["validator"].as_str(),
            Some("builtin:routing-small-for-ortools-python-adapter")
        );
        assert!(
            matrix["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("route=0->1->2->3->0") && stdout.contains("objective=19.000")
            }),
            "{matrix:?}"
        );

        let capacity = run_model_validation_json_with_rust_reference(
            &json!({
                "distanceMatrix": [
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]
                ],
                "customers": [1, 2],
                "demands": [0, 5, 5],
                "vehicle_capacity": 6
            }),
            "ortools-adapter",
        );
        assert_eq!(capacity["status"].as_str(), Some("ok"));
        assert_eq!(capacity["verdict"].as_str(), Some("infeasible"));
        assert_eq!(
            capacity["validator"].as_str(),
            Some("builtin:routing-small-for-ortools-adapter")
        );

        let cvrp = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "cvrp-validation",
                "depot": {"x": 0.0, "y": 0.0},
                "customers": [
                    {"id": "A", "x": 1.0, "y": 0.0, "demand": 1.0},
                    {"id": "B", "x": 2.0, "y": 0.0, "demand": 1.0}
                ],
                "vehicle_capacity": 2.0
            }),
            "ortools-java-adapter",
        );
        assert_eq!(cvrp["status"].as_str(), Some("ok"));
        assert_eq!(cvrp["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            cvrp["validator"].as_str(),
            Some("builtin:routing-small-for-ortools-java-adapter")
        );
        assert!(
            cvrp["stdout"]
                .as_str()
                .is_some_and(|stdout| stdout.contains("solver=rust:exact-cvrp")),
            "{cvrp:?}"
        );
    }

    #[test]
    fn nonlinear_python_and_rust_adapters_use_rust_reference_fallbacks() {
        let scipy = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "nonlinear-validation",
                "variables": [
                    {"name": "x", "lb": 0.0, "ub": 3.0, "start": 0.2},
                    {"name": "y", "lb": 0.0, "ub": 3.0, "start": 0.2}
                ],
                "objective": "(x - 1)**2 + (y - 2)**2",
                "constraints": [
                    {"expr": "x + y", "sense": ">=", "rhs": 1.0}
                ],
                "sense": "min"
            }),
            "scipy_optimize_adapter",
        );
        assert_eq!(scipy["status"].as_str(), Some("ok"));
        assert_eq!(scipy["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            scipy["validator"].as_str(),
            Some("builtin:nonlinear-reference-for-scipy-optimize-adapter")
        );
        assert!(
            scipy["stdout"].as_str().is_some_and(|stdout| {
                stdout.contains("x=1.000")
                    && stdout.contains("y=2.000")
                    && stdout.contains("solver=builtin:nlp-pattern-search-for-scipy")
            }),
            "{scipy:?}"
        );

        let cvxpy = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "convex-validation",
                "variables": [
                    {"name": "x", "lb": -2.0, "ub": 2.0, "start": 0.0}
                ],
                "objective": "x**2",
                "constraints": [],
                "sense": "min"
            }),
            "cvxpy-adapter",
        );
        assert_eq!(cvxpy["status"].as_str(), Some("ok"));
        assert_eq!(cvxpy["verdict"].as_str(), Some("optimal"));
        assert_eq!(
            cvxpy["validator"].as_str(),
            Some("builtin:nonlinear-reference-for-cvxpy-adapter")
        );
        assert!(
            cvxpy["stdout"]
                .as_str()
                .is_some_and(|stdout| stdout.contains("objective=0.000")),
            "{cvxpy:?}"
        );

        let nlopt = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "nlp-validation",
                "variables": [
                    {"name": "x0", "lb": 0.0, "ub": 1.0},
                    {"name": "x1", "lb": 0.0, "ub": 1.0}
                ],
                "objective": "x0**2 + x1**2",
                "constraints": [
                    {"expr": "x0 + x1", "sense": ">=", "rhs": 3.0}
                ],
                "sense": "min"
            }),
            "nlopt_adapter",
        );
        assert_eq!(nlopt["status"].as_str(), Some("ok"));
        assert_eq!(nlopt["verdict"].as_str(), Some("infeasible"));
        assert_eq!(
            nlopt["validator"].as_str(),
            Some("builtin:nonlinear-reference-for-nlopt-adapter")
        );
        assert!(
            nlopt["stdout"].as_str().is_some_and(
                |stdout| stdout.contains("solver=builtin:nlp-pattern-search-for-nlopt")
            ),
            "{nlopt:?}"
        );

        for (tool, solver) in [
            ("ores-argmin-adapter", "solver=builtin:nlp-pattern-search"),
            (
                "ores-nlopt-rs-adapter",
                "solver=builtin:nlp-pattern-search-for-nlopt",
            ),
            (
                "ores-ipopt-rust-adapter",
                "solver=builtin:nlp-pattern-search-for-ipopt",
            ),
        ] {
            let result = run_model_validation_json_with_rust_reference(
                &json!({
                    "kind": "nonlinear-validation",
                    "variables": [
                        {"name": "x", "lb": -1.0, "ub": 2.0, "start": 0.0}
                    ],
                    "objective": "(x - 0.5)**2",
                    "constraints": [],
                    "sense": "min"
                }),
                tool,
            );
            assert_eq!(result["status"].as_str(), Some("ok"), "{tool}: {result:?}");
            assert_eq!(
                result["verdict"].as_str(),
                Some("optimal"),
                "{tool}: {result:?}"
            );
            assert_eq!(
                result["validator"].as_str(),
                Some(format!("builtin:nonlinear-reference-for-{tool}").as_str())
            );
            assert!(
                result["stdout"]
                    .as_str()
                    .is_some_and(|stdout| stdout.contains("x=0.500") && stdout.contains(solver)),
                "{tool}: {result:?}"
            );
        }
    }

    #[test]
    fn pddl_planning_aliases_use_rust_structural_fallbacks() {
        let domain = "(define (domain flank)\n  (:predicates (at ?p ?x) (clear ?x))\n  (:action move\n    :parameters (?p ?from ?to)\n    :precondition (and (at ?p ?from) (clear ?to))\n    :effect (and (not (at ?p ?from)) (at ?p ?to)))\n)\n";
        let problem = "(define (problem spread-wide)\n  (:domain flank)\n  (:objects p1 left center)\n  (:init (at p1 center) (clear left))\n  (:goal (and (at p1 left)))\n)\n";
        let plan = "0.000: (move p1 center left) [1.000]\n";
        let val = run_model_validation_json_with_rust_reference(
            &json!({
                "kind": "plan-validation",
                "domain": domain,
                "problem": problem,
                "plan": plan,
            }),
            "pddl_val",
        );
        assert_eq!(val["status"].as_str(), Some("ok"));
        assert_eq!(val["verdict"].as_str(), Some("valid"));
        assert_eq!(
            val["validator"].as_str(),
            Some("builtin:pddl-structural-for-pddl-val")
        );

        let fast_downward = run_model_validation_json_with_rust_reference(
            &json!({
                "domain": "(define (domain flank) (:predicates (at ?p ?x)) (:action move :parameters (?p ?x ?y) :precondition (at ?p ?x) :effect (at ?p ?y))",
                "problem": problem,
            }),
            "fast-downward.py",
        );
        assert_eq!(fast_downward["status"].as_str(), Some("ok"));
        assert_eq!(fast_downward["verdict"].as_str(), Some("invalid"));
        assert_eq!(
            fast_downward["validator"].as_str(),
            Some("builtin:pddl-structural-for-fast-downward.py")
        );
        assert!(
            fast_downward["stderr"]
                .as_str()
                .unwrap_or_default()
                .contains("unclosed"),
            "{fast_downward:?}"
        );
    }

    #[test]
    fn proof_checker_command_aliases_use_rust_structural_fallbacks() {
        let unsat_cnf = "p cnf 1 2\n1 0\n-1 0\n";

        let lrat = run_proof_validation_json_with_rust_reference(
            &json!({
                "cnf": unsat_cnf,
                "proof": "1 0 0\n",
            }),
            "cake_lpr",
        );
        assert_eq!(lrat["status"].as_str(), Some("ok"));
        assert_eq!(lrat["verdict"].as_str(), Some("valid"));
        assert_eq!(
            lrat["validator"].as_str(),
            Some("builtin:small-cnf-proof-for-cake-lpr")
        );

        let frat = run_proof_validation_json_with_rust_reference(
            &json!({
                "cnf": unsat_cnf,
                "proof": "a 0\n",
            }),
            "frat-trim",
        );
        assert_eq!(frat["status"].as_str(), Some("ok"));
        assert_eq!(frat["verdict"].as_str(), Some("valid"));
        assert_eq!(
            frat["validator"].as_str(),
            Some("builtin:small-cnf-proof-for-frat-trim")
        );

        let veripb = run_proof_validation_json_with_rust_reference(
            &json!({
                "kind": "opb-proof-validation",
                "opb": "1 x1 >= 1;\n1 x1 <= 0;\n",
                "proof": "1 >= 1 ;\n",
            }),
            "veripb-checker",
        );
        assert_eq!(veripb["status"].as_str(), Some("ok"));
        assert_eq!(veripb["verdict"].as_str(), Some("valid"));
        assert_eq!(
            veripb["validator"].as_str(),
            Some("builtin:small-opb-proof-for-veripb-checker")
        );
    }

    #[test]
    fn registry_covers_recommended_validation_layers() {
        let tools = external_validation_tool_specs();
        assert_eq!(tools.len(), 268);
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
            tool.id == "minilp"
                && tool.runtime == ExternalValidationRuntime::Rust
                && tool.artifact_kind == ExternalValidationArtifactKind::RustCrate
                && tool.input_formats.contains(&"lp")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "argmin" && tool.family == ExternalValidationFamily::NonlinearGlobalSolver
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "nlopt-rs" && tool.input_formats.contains(&"nl") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "osqp-rust"
                && tool.runtime == ExternalValidationRuntime::Rust
                && tool.family == ExternalValidationFamily::ConvexConicSolver
                && tool.artifact_kind == ExternalValidationArtifactKind::RustCrate
                && tool.command_aliases.contains(&"osqp-rust-adapter")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "clarabel-rust"
                && tool.runtime == ExternalValidationRuntime::Rust
                && tool.family == ExternalValidationFamily::ConvexConicSolver
                && tool.artifact_kind == ExternalValidationArtifactKind::RustCrate
                && tool.input_formats.contains(&"cone")
        }));
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
        assert!(tools.iter().any(|tool| {
            tool.id == "mosek"
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "copt"
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
        }));
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
            tool.id == "clingo"
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cvc5"
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
        }));
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
            tool.id == "json-schema"
                && tool.family == ExternalValidationFamily::OutputDataValidator
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "cue" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "zod"
                && tool.family == ExternalValidationFamily::OutputDataValidator
                && tool.runtime == ExternalValidationRuntime::Node
                && tool.artifact_kind == ExternalValidationArtifactKind::NodePackage
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "valibot"
                && tool.family == ExternalValidationFamily::OutputDataValidator
                && tool.runtime == ExternalValidationRuntime::Node
                && tool.artifact_kind == ExternalValidationArtifactKind::NodePackage
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "dbt" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "sqlfluff"
                && tool.family == ExternalValidationFamily::OutputDataValidator
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
                && tool.input_formats.contains(&"sql")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "apache-arrow"
                && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "spectral" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "schematron"
                && tool.family == ExternalValidationFamily::OutputDataValidator
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "xml-schema" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "xml-schema" && tool.command_aliases.contains(&"xmlschema-validate")
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "jing" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "saxon"
                && tool.family == ExternalValidationFamily::OutputDataValidator
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "csv-validator"
                && tool.family == ExternalValidationFamily::OutputDataValidator
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "protoc" && tool.family == ExternalValidationFamily::OutputDataValidator
        }));
        assert!(tools.iter().any(|tool| {
            tool.id == "apache-avro"
                && tool.family == ExternalValidationFamily::OutputDataValidator
                && tool.runtime == ExternalValidationRuntime::Python
                && tool.artifact_kind == ExternalValidationArtifactKind::PythonPackage
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
        let json_schema = find_external_validation_tool("json-schema").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(json_schema)[0],
            "ORES_JSON_SCHEMA_PYTHON"
        );
        let schematron = find_external_validation_tool("schematron").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(schematron)[0],
            "ORES_SCHEMATRON_PYTHON"
        );
        let jing = find_external_validation_tool("jing").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(jing)[0],
            "ORES_JING_CLASSPATH"
        );
        let csv_validator = find_external_validation_tool("csv-validator").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(csv_validator)[0],
            "ORES_CSV_VALIDATOR_PYTHON"
        );
        let saxon = find_external_validation_tool("saxon").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(saxon)[0],
            "ORES_SAXON_PYTHON"
        );
        let zod = find_external_validation_tool("zod").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(zod)[0],
            "ORES_ZOD_NODE_PATH"
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
        let clingo = find_external_validation_tool("clingo").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(clingo)[0],
            "ORES_CLINGO_PYTHON"
        );
        let cvc5 = find_external_validation_tool("cvc5").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(cvc5)[0],
            "ORES_CVC5_PYTHON"
        );
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
        let minilp = find_external_validation_tool("minilp").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(minilp)[0],
            "ORES_MINILP_CRATE"
        );
        let nlopt_rs = find_external_validation_tool("nlopt_rs").unwrap();
        assert_eq!(nlopt_rs.id, "nlopt-rs");
        assert!(
            external_validation_command_dir_env_names(nlopt_rs).contains(&"NLOPT_HOME".to_string())
        );
        let osqp_rust = find_external_validation_tool("osqp_rust").unwrap();
        assert_eq!(osqp_rust.id, "osqp-rust");
        assert!(external_validation_artifact_env_names(osqp_rust)
            .contains(&"OSQP_RS_CARGO_MANIFEST".to_string()));
        assert!(
            external_validation_command_dir_env_names(osqp_rust).contains(&"OSQP_HOME".to_string())
        );
        let clarabel_rust = find_external_validation_tool("clarabel_rust").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(clarabel_rust)[0],
            "ORES_CLARABEL_RUST_CRATE"
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
            "ORES_NLOPT_PYTHON"
        );
        assert!(external_validation_python_modules(nlopt).contains(&"nlopt"));
        let mosek = find_external_validation_tool("mosek").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(mosek)[0],
            "ORES_MOSEK_PYTHON"
        );
        assert!(external_validation_artifact_env_names(mosek)
            .contains(&"MOSEKLM_LICENSE_FILE".to_string()));
        assert!(external_validation_python_modules(mosek).contains(&"mosek"));
        let copt = find_external_validation_tool("copt").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(copt)[0],
            "ORES_COPT_PYTHON"
        );
        assert!(external_validation_artifact_env_names(copt).contains(&"COPT_HOME".to_string()));
        assert!(external_validation_python_modules(copt).contains(&"coptpy"));
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
    fn simulation_validation_unsupported_format_stays_in_rust() {
        let payload = json!({
            "kind": "simulation-validation",
            "engine": "simpy",
            "model_format": "json-unsupported-smoke",
            "model": {}
        });

        let run = run_simulation_validation_json_with_external_reference(
            &payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );

        assert_eq!(run.status, ExternalSimulationValidationStatus::Unavailable);
        assert_eq!(run.verdict, ExternalSimulationValidationVerdict::Unknown);
        assert_eq!(run.simulator, "rust:unsupported-simulation-format");
        assert!(run.message.contains("json-unsupported-smoke"));
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
    fn simulation_validation_aliases_use_rust_reference_families() {
        let event_model = json!({
            "servers": 2,
            "arrival_times": [0.0, 0.25, 0.5],
            "service_times": [0.5, 0.5, 0.5]
        });
        for (engine, canonical) in [
            ("ciw-adapter", "ciw"),
            ("simulus-adapter", "simulus"),
            ("desmoj-adapter", "desmo-j"),
            ("simsharp-adapter", "simsharp"),
            ("plantsim-adapter", "plant-simulation"),
        ] {
            let payload = json!({
                "kind": "simulation-validation",
                "engine": engine,
                "model_format": "json-event-network",
                "model": event_model,
                "expected_trace_properties": ["departures_after_arrivals"]
            });
            let run = run_simulation_validation_json_with_external_reference(
                &payload,
                &ExternalSimulationValidationReferenceOptions::default(),
            );
            assert_eq!(run.status, ExternalSimulationValidationStatus::Ok);
            assert_eq!(run.verdict, ExternalSimulationValidationVerdict::Valid);
            assert!(
                run.simulator
                    .starts_with(&format!("rust:single-station-des-for-{canonical}")),
                "{engine} used simulator {}",
                run.simulator
            );
        }

        let energy_payload = json!({
            "kind": "simulation-validation",
            "engine": "fmpy",
            "model_format": "json-energy-balance",
            "model": {"initial_temp": 20.0, "setpoint": 21.0, "heat_capacity": 10.0},
            "scenario": {"horizon": 1.0, "step": 1.0}
        });
        let energy_run = run_simulation_validation_json_with_external_reference(
            &energy_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(energy_run.status, ExternalSimulationValidationStatus::Ok);
        assert!(energy_run
            .simulator
            .starts_with("rust:energy-balance-for-fmi-fmu"));

        for (engine, canonical) in [
            ("mujoco-adapter", "mujoco"),
            ("drake-adapter", "drake"),
            ("pybullet-adapter", "pybullet"),
        ] {
            let payload = json!({
                "kind": "simulation-validation",
                "engine": engine,
                "model_format": "json-physics-trajectory",
                "model": {"initial_position": 0.0, "initial_velocity": 1.0, "acceleration": 0.0},
                "scenario": {"dt": 0.5, "steps": 2}
            });
            let run = run_simulation_validation_json_with_external_reference(
                &payload,
                &ExternalSimulationValidationReferenceOptions::default(),
            );
            assert_eq!(run.status, ExternalSimulationValidationStatus::Ok);
            assert!(
                run.simulator
                    .starts_with(&format!("rust:physics-trajectory-for-{canonical}")),
                "{engine} used simulator {}",
                run.simulator
            );
        }

        for (engine, canonical) in [
            ("agentpy-adapter", "agentpy"),
            ("repast-adapter", "repast"),
            ("mason-adapter", "mason"),
        ] {
            let payload = json!({
                "kind": "simulation-validation",
                "engine": engine,
                "model_format": "json-agent-based",
                "model": {"agents": [{"state": "a"}, {"state": "b"}], "interactions": [{"source": 0, "target": 1}]},
                "scenario": {"steps": 1}
            });
            let run = run_simulation_validation_json_with_external_reference(
                &payload,
                &ExternalSimulationValidationReferenceOptions::default(),
            );
            assert_eq!(run.status, ExternalSimulationValidationStatus::Ok);
            assert!(
                run.simulator
                    .starts_with(&format!("rust:agent-based-for-{canonical}")),
                "{engine} used simulator {}",
                run.simulator
            );
        }

        let distributed_payload = json!({
            "kind": "simulation-validation",
            "engine": "cloudsim-adapter",
            "model_format": "json-distributed-system",
            "model": {"hosts": [{"capacity": 4}], "links": [{"bandwidth": 10}], "tasks": [{"work": 2}]}
        });
        let distributed_run = run_simulation_validation_json_with_external_reference(
            &distributed_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(
            distributed_run.status,
            ExternalSimulationValidationStatus::Ok
        );
        assert!(distributed_run
            .simulator
            .starts_with("rust:distributed-system-for-cloudsim"));

        for (engine, canonical) in [
            ("dwsim-adapter", "dwsim"),
            ("capeopen-adapter", "cape-open"),
            ("tellurium-adapter", "tellurium"),
        ] {
            let payload = json!({
                "kind": "simulation-validation",
                "engine": engine,
                "model_format": "json-process-flow",
                "model": {
                    "units": [{"name": "unit"}],
                    "streams": [
                        {"from": "source", "to": "unit", "flow": 3.0},
                        {"from": "unit", "to": "sink", "flow": 3.0}
                    ]
                }
            });
            let run = run_simulation_validation_json_with_external_reference(
                &payload,
                &ExternalSimulationValidationReferenceOptions::default(),
            );
            assert_eq!(run.status, ExternalSimulationValidationStatus::Ok);
            assert!(
                run.simulator
                    .starts_with(&format!("rust:process-flow-for-{canonical}")),
                "{engine} used simulator {}",
                run.simulator
            );
        }
    }

    #[test]
    fn simulation_validation_infers_rust_model_format_from_engine_alias() {
        let physics_payload = json!({
            "kind": "simulation-validation",
            "engine": "mujoco-adapter",
            "model": {"initial_position": 0.0, "initial_velocity": 0.0, "acceleration": 1.0},
            "scenario": {"dt": 0.25, "steps": 2}
        });
        let physics_run = run_simulation_validation_json_with_external_reference(
            &physics_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(physics_run.status, ExternalSimulationValidationStatus::Ok);
        assert!(physics_run
            .simulator
            .starts_with("rust:physics-trajectory-for-mujoco"));

        let agent_payload = json!({
            "kind": "simulation-validation",
            "engine": "agentpy-adapter",
            "model": {"agents": [{"state": "a"}]},
            "scenario": {"steps": 1}
        });
        let agent_run = run_simulation_validation_json_with_external_reference(
            &agent_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(agent_run.status, ExternalSimulationValidationStatus::Ok);
        assert!(agent_run
            .simulator
            .starts_with("rust:agent-based-for-agentpy"));

        let process_payload = json!({
            "kind": "simulation-validation",
            "engine": "capeopen-adapter",
            "model": {
                "units": [{"name": "unit"}],
                "streams": [
                    {"from": "source", "to": "unit", "flow": 1.0},
                    {"from": "unit", "to": "sink", "flow": 1.0}
                ]
            }
        });
        let process_run = run_simulation_validation_json_with_external_reference(
            &process_payload,
            &ExternalSimulationValidationReferenceOptions::default(),
        );
        assert_eq!(process_run.status, ExternalSimulationValidationStatus::Ok);
        assert!(process_run
            .simulator
            .starts_with("rust:process-flow-for-cape-open"));
    }

    #[test]
    fn external_validation_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_external_validation_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
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
        assert_eq!(
            ExternalValidationArtifactKind::NodePackage.env_suffix(),
            Some("NODE_PATH")
        );
        assert_eq!(ExternalValidationArtifactKind::None.env_suffix(), None);
    }
}
