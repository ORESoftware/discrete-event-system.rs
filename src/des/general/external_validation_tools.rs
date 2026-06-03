//! Local adapter/probe surface for external model-validation, output-validation,
//! benchmark, proof-checking, and simulation engines.
//!
//! These tools are intentionally represented as local adapters: the crate knows
//! stable names, capabilities, command aliases, and environment-variable hooks,
//! but it does not vendor jars, native libraries, solver binaries, benchmark
//! corpora, or simulator installations.

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
const SMT_FORMATS: &[&str] = &["smt2"];
const SAT_FORMATS: &[&str] = &["cnf", "wcnf"];
const PROOF_FORMATS: &[&str] = &["drat", "lrat", "grat"];
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
        (
            "kissat" | "cadical" | "cryptominisat",
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
        (
            "kissat" | "cadical" | "cryptominisat",
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
        "cpmpy" => names.push("CPMPY_PYTHON".to_string()),
        "pycsp3" => names.push("PYCSP3_PYTHON".to_string()),
        "conjure" => names.push("CONJURE_HOME".to_string()),
        "savile-row" => names.push("SAVILEROW_HOME".to_string()),
        "picat" => names.push("PICAT_HOME".to_string()),
        "clingo" => names.push("CLINGO_HOME".to_string()),
        "clingcon" => names.push("CLINGCON_HOME".to_string()),
        "sat4j" => names.push("SAT4J_HOME".to_string()),
        "pysat" => names.push("PYSAT_PYTHON".to_string()),
        "open-wbo" => names.push("OPEN_WBO_HOME".to_string()),
        "ipopt" => names.push("IPOPT_DIR".to_string()),
        "bonmin" => names.push("BONMIN_DIR".to_string()),
        "couenne" => names.push("COUENNE_DIR".to_string()),
        "knitro" => names.push("ARTELYS_LICENSE".to_string()),
        "mosek" => names.push("MOSEKLM_LICENSE_FILE".to_string()),
        "baron" => names.push("BARON_LICENSE".to_string()),
        "copt" => names.push("COPT_HOME".to_string()),
        "java-pathfinder" => names.push("JPF_HOME".to_string()),
        "key" => names.push("KEY_HOME".to_string()),
        "viper" => names.push("VIPER_HOME".to_string()),
        "fstar" => names.push("FSTAR_HOME".to_string()),
        "gnatprove" => names.push("GNATPROVE_HOME".to_string()),
        "seahorn" => names.push("SEAHORN_DIR".to_string()),
        "smack" => names.push("SMACK_HOME".to_string()),
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
        "cpmpy" => &["CPMPY_HOME", "CPMPY_DIR"],
        "pycsp3" => &["PYCSP3_HOME", "PYCSP3_DIR"],
        "conjure" => &["CONJURE_HOME", "CONJURE_DIR"],
        "savile-row" => &["SAVILE_ROW_HOME", "SAVILE_ROW_DIR", "SAVILEROW_HOME"],
        "picat" => &["PICAT_HOME", "PICAT_DIR"],
        "clingo" => &["CLINGO_HOME", "CLINGO_DIR", "POTASSCO_HOME"],
        "clingcon" => &["CLINGCON_HOME", "CLINGCON_DIR", "POTASSCO_HOME"],
        "sat4j" => &["SAT4J_HOME", "SAT4J_DIR"],
        "pysat" => &["PYSAT_HOME", "PYSAT_DIR"],
        "open-wbo" => &["OPEN_WBO_HOME", "OPEN_WBO_DIR", "OPENWBO_HOME"],
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
        "dafny" => &["DAFNY_HOME", "DAFNY_DIR"],
        "frama-c" => &["FRAMA_C_HOME", "FRAMA_C_DIR"],
        "why3" => &["WHY3_HOME", "WHY3_DIR"],
        "esbmc" => &["ESBMC_HOME", "ESBMC_DIR"],
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
        "nlopt-cli" => &["NLOPT_DIR", "NLOPT_HOME"],
        "casadi" => &["CASADI_DIR", "CASADI_HOME"],
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
        simulation_validation_request_to_json, smtlib_validation_script_to_string,
        tla_validation_module_to_string, DimacsCnf, DimacsWcnf, DimacsWeightedClause,
        ExternalBenchmarkManifest, ExternalBenchmarkManifestEntry, ExternalValidationArtifact,
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
        assert_eq!(tools.len(), 190);
        assert!(tools
            .iter()
            .any(|tool| tool.id == "minizinc" && tool.input_formats.contains(&"mzn")));
        assert!(tools.iter().any(|tool| {
            tool.id == "cpmpy" && tool.family == ExternalValidationFamily::ConstraintModeling
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "conjure" && tool.input_formats.contains(&"essence") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "clingo" && tool.input_formats.contains(&"asp") }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "z3" && tool.family == ExternalValidationFamily::SmtSolver }));
        assert!(tools.iter().any(|tool| {
            tool.id == "sat4j" && tool.family == ExternalValidationFamily::SatSolver
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool.id == "open-wbo" && tool.input_formats.contains(&"wcnf") }));
        assert!(tools.iter().any(|tool| {
            tool.id == "drat-trim" && tool.family == ExternalValidationFamily::ProofChecker
        }));
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
            tool.id == "cbmc" && tool.family == ExternalValidationFamily::FormalModelChecker
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
        let cpmpy = find_external_validation_tool("cpmpy").unwrap();
        assert_eq!(
            external_validation_artifact_env_names(cpmpy)[0],
            "ORES_CPMPY_PYTHON"
        );
        assert!(
            external_validation_command_dir_env_names(cpmpy).contains(&"CPMPY_HOME".to_string())
        );
        let conjure = find_external_validation_tool("conjure").unwrap();
        assert!(external_validation_command_dir_env_names(conjure)
            .contains(&"ORES_CONJURE_DIR".to_string()));
        assert!(external_validation_command_dir_env_names(conjure)
            .contains(&"CONJURE_HOME".to_string()));
        let open_wbo = find_external_validation_tool("open_wbo").unwrap();
        assert_eq!(open_wbo.id, "open-wbo");
        assert!(external_validation_command_dir_env_names(open_wbo)
            .contains(&"OPEN_WBO_HOME".to_string()));
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
    fn text_cli_profiles_and_verdict_inference_are_stable() {
        assert_eq!(
            external_validation_default_text_cli_args("z3", ExternalValidationTextFormat::SmtLib2),
            &["-in", "-smt2"]
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
