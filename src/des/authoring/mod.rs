//! JSON-first model authoring for Simulink/Modelica-style workflows.
//!
//! This module is the typed boundary above the existing engines. It provides:
//!
//! * a Rust type model that derives JSON Schema;
//! * semantic validation for graph/spec references;
//! * compilation of the supported causal block subset into `des::hybrid`;
//! * a first acausal physical-network flattening pass into hybrid DAE text;
//! * state-machine blocks, trace comparison, dependency/catalog descriptors;
//! * Rust code generation that embeds and runs a checked JSON model spec.

use std::collections::{HashMap, HashSet};

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::des::hybrid::block::{Block, PortSpec, SampleTime, Signal};
use crate::des::hybrid::blocks::{
    BouncingBall, Constant, Counter, DiscretePi, Gain, Integrator, Saturation, StateSpace, Sum,
};
use crate::des::hybrid::{simulate, Compiled, Diagram, HybridError, SimOptions, Trace};
use crate::des::model::{CitizenError, ModelCitizen, ModelDescriptor, RunArtifact};
use crate::des::plugin::UiControl;

/// Stable schema id for the graph authoring contract.
pub const AUTHORING_SCHEMA: &str = "des/model-graph/v1";

/// Return the generated JSON Schema for [`AuthoringSpec`].
///
/// The schema is derived from Rust types, so schema drift becomes a compile-time
/// concern: fields, enum tags, renames, and nested structs all come from one
/// typed source of truth.
pub fn authoring_json_schema() -> Value {
    serde_json::to_value(schema_for!(AuthoringSpec)).unwrap_or_else(|_| json!({}))
}

/// A complete model document. It intentionally carries more than the executable
/// subset: some sections are for tooling, validation, interchange, and future
/// compiler passes, while `blocks` and `connections` can already compile to the
/// hybrid executive when they use supported block kinds.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuthoringSpec {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub solver: SolverSpec,
    #[serde(default)]
    pub units: Vec<UnitSpec>,
    #[serde(default)]
    pub data_dictionary: Vec<DataDictionaryEntry>,
    #[serde(default)]
    pub variants: Vec<VariantSpec>,
    #[serde(default)]
    pub active_variants: Vec<String>,
    #[serde(default)]
    pub blocks: Vec<BlockSpec>,
    #[serde(default)]
    pub connections: Vec<ConnectionSpec>,
    #[serde(default)]
    pub submodels: Vec<SubmodelSpec>,
    #[serde(default)]
    pub model_references: Vec<ModelReferenceSpec>,
    #[serde(default)]
    pub physical_networks: Vec<PhysicalNetworkSpec>,
    #[serde(default)]
    pub state_machines: Vec<StateMachineSpec>,
    #[serde(default)]
    pub fmi: FmiInteropSpec,
    #[serde(default)]
    pub codegen: CodegenSpec,
    #[serde(default)]
    pub verification: VerificationSpec,
    #[serde(default)]
    pub tooling: ToolingSpec,
}

impl Default for AuthoringSpec {
    fn default() -> Self {
        example_spec()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SolverSpec {
    #[serde(default)]
    pub mode: SolverMode,
    #[serde(default = "default_t_end")]
    pub t_end: f64,
    #[serde(default = "default_max_step")]
    pub max_step: f64,
    #[serde(default = "default_rel_tol")]
    pub rel_tol: f64,
    #[serde(default = "default_abs_tol")]
    pub abs_tol: f64,
    #[serde(default)]
    pub algebraic_loops: AlgebraicLoopPolicy,
    #[serde(default)]
    pub deployment: DeploymentProfile,
}

impl Default for SolverSpec {
    fn default() -> Self {
        SolverSpec {
            mode: SolverMode::FixedStepRk4,
            t_end: default_t_end(),
            max_step: default_max_step(),
            rel_tol: default_rel_tol(),
            abs_tol: default_abs_tol(),
            algebraic_loops: AlgebraicLoopPolicy::default(),
            deployment: DeploymentProfile::default(),
        }
    }
}

fn default_t_end() -> f64 {
    5.0
}

fn default_max_step() -> f64 {
    0.01
}

fn default_rel_tol() -> f64 {
    1e-6
}

fn default_abs_tol() -> f64 {
    1e-9
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "kebab-case")]
pub enum SolverMode {
    #[default]
    FixedStepRk4,
    VariableStepRk45,
    BackwardEuler,
    DiscreteOnly,
    DaeResidual,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "kebab-case")]
pub enum AlgebraicLoopPolicy {
    #[default]
    Reject,
    EmitDae,
    Newton,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "kebab-case")]
pub enum DeploymentProfile {
    #[default]
    Desktop,
    RealtimeFixedStep,
    Hil,
    CloudBatch,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct UnitSpec {
    pub name: String,
    pub quantity: String,
    pub unit: String,
    #[serde(default)]
    pub display_unit: Option<String>,
    #[serde(default)]
    pub scale_to_si: Option<f64>,
    #[serde(default)]
    pub offset_to_si: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DataDictionaryEntry {
    pub name: String,
    pub value: Value,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub protected: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct VariantSpec {
    pub id: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub default: bool,
    #[serde(default)]
    pub constraints: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct BlockSpec {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub variant: Option<String>,
    #[serde(default)]
    pub position: Option<PositionSpec>,
    pub kind: BlockKindSpec,
    #[serde(default)]
    pub requirements: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PositionSpec {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub w: Option<f64>,
    #[serde(default)]
    pub h: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum BlockKindSpec {
    Constant {
        value: Vec<f64>,
    },
    Gain {
        width: usize,
        k: f64,
    },
    Sum {
        width: usize,
        signs: Vec<f64>,
    },
    Saturation {
        lo: f64,
        hi: f64,
    },
    Integrator {
        initial: Vec<f64>,
    },
    StateSpace {
        a: Vec<Vec<f64>>,
        b: Vec<Vec<f64>>,
        c: Vec<Vec<f64>>,
        d: Vec<Vec<f64>>,
        #[serde(default)]
        x0: Option<Vec<f64>>,
    },
    BouncingBall {
        height: f64,
        velocity: f64,
        restitution: f64,
    },
    DiscretePi {
        kp: f64,
        ki: f64,
        period: f64,
    },
    Counter {
        period: f64,
    },
    StateMachine {
        machine: String,
        period: f64,
        #[serde(default)]
        input_width: usize,
    },
    Terminator {
        width: usize,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConnectionSpec {
    pub from: PortRef,
    pub to: PortRef,
    #[serde(default)]
    pub variant: Option<String>,
    #[serde(default)]
    pub requirements: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PortRef {
    pub block: String,
    #[serde(default)]
    pub port: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SubmodelSpec {
    pub id: String,
    #[serde(default)]
    pub description: Option<String>,
    pub model: Box<AuthoringSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelReferenceSpec {
    pub id: String,
    pub target: String,
    #[serde(default)]
    pub parameters: HashMap<String, Value>,
    #[serde(default)]
    pub protected: bool,
    #[serde(default)]
    pub variant: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalNetworkSpec {
    pub id: String,
    pub domain: PhysicalDomain,
    #[serde(default)]
    pub nodes: Vec<PhysicalNodeSpec>,
    #[serde(default)]
    pub components: Vec<PhysicalComponentSpec>,
    #[serde(default)]
    pub connections: Vec<PhysicalConnectionSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum PhysicalDomain {
    Electrical,
    TranslationalMechanical,
    RotationalMechanical,
    Thermal,
    Fluid,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalNodeSpec {
    pub id: String,
    #[serde(default)]
    pub quantity: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalComponentSpec {
    pub id: String,
    pub kind: PhysicalComponentKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum PhysicalComponentKind {
    ElectricalResistor {
        resistance: f64,
    },
    ElectricalCapacitor {
        capacitance: f64,
        #[serde(default)]
        initial_voltage: f64,
    },
    ElectricalInductor {
        inductance: f64,
        #[serde(default)]
        initial_current: f64,
    },
    ElectricalVoltageSource {
        voltage: f64,
    },
    ElectricalCurrentSource {
        current: f64,
    },
    ElectricalGround,
    TranslationalMass {
        mass: f64,
    },
    TranslationalSpring {
        stiffness: f64,
    },
    TranslationalDamper {
        damping: f64,
    },
    ThermalCapacitor {
        heat_capacity: f64,
    },
    ThermalConductor {
        conductance: f64,
    },
    FluidReservoir {
        pressure: f64,
    },
    FluidResistance {
        resistance: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalConnectionSpec {
    pub component: String,
    pub connector: String,
    pub node: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StateMachineSpec {
    pub id: String,
    #[serde(default)]
    pub initial: Option<String>,
    #[serde(default)]
    pub states: Vec<StateSpec>,
    #[serde(default)]
    pub transitions: Vec<TransitionSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StateSpec {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TransitionSpec {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub guard: GuardSpec,
    #[serde(default)]
    pub actions: Vec<String>,
    #[serde(default)]
    pub requirement: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum GuardSpec {
    #[default]
    Always,
    After {
        time: f64,
    },
    InputGreater {
        index: usize,
        threshold: f64,
    },
    InputLess {
        index: usize,
        threshold: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct FmiInteropSpec {
    #[serde(default)]
    pub imports: Vec<FmuImportSpec>,
    #[serde(default)]
    pub exports: Vec<FmuExportSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FmuImportSpec {
    pub id: String,
    pub path: String,
    #[serde(default)]
    pub interface: FmiInterfaceKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FmuExportSpec {
    pub id: String,
    #[serde(default)]
    pub interface: FmiInterfaceKind,
    #[serde(default)]
    pub include_rust_source: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "kebab-case")]
pub enum FmiInterfaceKind {
    #[default]
    CoSimulation,
    ModelExchange,
    ScheduledExecution,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct CodegenSpec {
    #[serde(default)]
    pub target: CodegenTarget,
    #[serde(default)]
    pub module_name: Option<String>,
    #[serde(default)]
    pub deterministic: bool,
    #[serde(default)]
    pub include_schema: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CodegenTarget {
    #[default]
    Rust,
    RustRealtimeFixedStep,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct VerificationSpec {
    #[serde(default)]
    pub requirements: Vec<RequirementSpec>,
    #[serde(default)]
    pub trace_links: Vec<TraceLinkSpec>,
    #[serde(default)]
    pub coverage: CoverageSpec,
    #[serde(default)]
    pub properties: Vec<FormalPropertySpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RequirementSpec {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub safety_level: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TraceLinkSpec {
    pub requirement: String,
    pub element: String,
    #[serde(default)]
    pub relation: TraceRelation,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "kebab-case")]
pub enum TraceRelation {
    #[default]
    Satisfies,
    Verifies,
    Refines,
    Tests,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct CoverageSpec {
    #[serde(default)]
    pub require_block_coverage: bool,
    #[serde(default)]
    pub require_transition_coverage: bool,
    #[serde(default)]
    pub target_mcdc: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FormalPropertySpec {
    pub id: String,
    pub expression: String,
    #[serde(default)]
    pub severity: PropertySeverity,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "kebab-case")]
pub enum PropertySeverity {
    Info,
    Warning,
    #[default]
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolingSpec {
    #[serde(default)]
    pub enable_model_browser: bool,
    #[serde(default)]
    pub enable_library_browser: bool,
    #[serde(default)]
    pub enable_signal_inspector: bool,
    #[serde(default)]
    pub enable_variant_manager: bool,
    #[serde(default)]
    pub enable_dependency_analyzer: bool,
    #[serde(default)]
    pub enable_parameter_estimation: bool,
    #[serde(default)]
    pub enable_protected_models: bool,
    #[serde(default)]
    pub enable_collaboration: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub path: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthoringError {
    InvalidSpec(String),
    Compile(String),
    Codegen(String),
}

impl std::fmt::Display for AuthoringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthoringError::InvalidSpec(m) => write!(f, "invalid authoring spec: {m}"),
            AuthoringError::Compile(m) => write!(f, "compile failed: {m}"),
            AuthoringError::Codegen(m) => write!(f, "code generation failed: {m}"),
        }
    }
}

impl std::error::Error for AuthoringError {}

impl From<HybridError> for AuthoringError {
    fn from(value: HybridError) -> Self {
        AuthoringError::Compile(value.to_string())
    }
}

/// Parse JSON into the compile-time typed authoring contract and run semantic
/// validation that JSON Schema alone cannot express.
pub fn parse_authoring_spec(value: &Value) -> Result<AuthoringSpec, AuthoringError> {
    let spec: AuthoringSpec = serde_json::from_value(value.clone())
        .map_err(|e| AuthoringError::InvalidSpec(e.to_string()))?;
    let failures: Vec<String> = validate_authoring_spec(&spec)
        .into_iter()
        .filter(|i| i.severity == ValidationSeverity::Error)
        .map(|i| format!("{}: {}", i.path, i.message))
        .collect();
    if failures.is_empty() {
        Ok(spec)
    } else {
        Err(AuthoringError::InvalidSpec(failures.join("; ")))
    }
}

/// Semantic checks: uniqueness, port endpoints, variant references, statechart
/// references, physical node/component references, and requirement trace links.
pub fn validate_authoring_spec(spec: &AuthoringSpec) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if spec.schema != AUTHORING_SCHEMA {
        issues.push(err(
            "$schema",
            format!("expected `{AUTHORING_SCHEMA}`, got `{}`", spec.schema),
        ));
    }
    validate_solver_spec(&spec.solver, &mut issues);

    let mut variant_ids = HashSet::new();
    for v in &spec.variants {
        validate_nonempty_id(&v.id, format!("variants.{}", v.id), &mut issues);
        if !variant_ids.insert(v.id.clone()) {
            issues.push(err(
                format!("variants.{}", v.id),
                "duplicate variant id".to_string(),
            ));
        }
    }
    for active in &spec.active_variants {
        if !variant_ids.contains(active) {
            issues.push(err(
                format!("activeVariants.{active}"),
                "active variant is not declared".to_string(),
            ));
        }
    }

    validate_unique_named(
        spec.units.iter().map(|u| u.name.as_str()),
        "units",
        &mut issues,
    );
    validate_unique_named(
        spec.data_dictionary.iter().map(|d| d.name.as_str()),
        "dataDictionary",
        &mut issues,
    );
    validate_unique_named(
        spec.submodels.iter().map(|s| s.id.as_str()),
        "submodels",
        &mut issues,
    );
    validate_unique_named(
        spec.model_references.iter().map(|m| m.id.as_str()),
        "modelReferences",
        &mut issues,
    );

    let mut block_ids = HashSet::new();
    let mut block_map = HashMap::new();
    for b in &spec.blocks {
        validate_nonempty_id(&b.id, format!("blocks.{}", b.id), &mut issues);
        if !block_ids.insert(b.id.clone()) {
            issues.push(err(
                format!("blocks.{}", b.id),
                "duplicate block id".to_string(),
            ));
        }
        block_map.insert(b.id.as_str(), b);
        validate_block_kind(&b.id, &b.kind, &mut issues);
        if let Some(v) = &b.variant {
            if !variant_ids.contains(v) {
                issues.push(err(
                    format!("blocks.{}.variant", b.id),
                    format!("unknown variant `{v}`"),
                ));
            }
        }
    }

    for (idx, c) in spec.connections.iter().enumerate() {
        if !block_ids.contains(&c.from.block) {
            issues.push(err(
                format!("connections.{idx}.from.block"),
                format!("unknown source block `{}`", c.from.block),
            ));
        }
        if !block_ids.contains(&c.to.block) {
            issues.push(err(
                format!("connections.{idx}.to.block"),
                format!("unknown destination block `{}`", c.to.block),
            ));
        }
        if let Some(v) = &c.variant {
            if !variant_ids.contains(v) {
                issues.push(err(
                    format!("connections.{idx}.variant"),
                    format!("unknown variant `{v}`"),
                ));
            }
        }
        if let Some(src) = block_map.get(c.from.block.as_str()) {
            let ports = block_ports(&src.kind);
            if c.from.port >= ports.outputs.len() {
                issues.push(err(
                    format!("connections.{idx}.from.port"),
                    format!(
                        "source block `{}` has {} output port(s), cannot use port {}",
                        c.from.block,
                        ports.outputs.len(),
                        c.from.port
                    ),
                ));
            }
        }
        if let Some(dst) = block_map.get(c.to.block.as_str()) {
            let ports = block_ports(&dst.kind);
            if c.to.port >= ports.inputs.len() {
                issues.push(err(
                    format!("connections.{idx}.to.port"),
                    format!(
                        "destination block `{}` has {} input port(s), cannot use port {}",
                        c.to.block,
                        ports.inputs.len(),
                        c.to.port
                    ),
                ));
            }
        }
        if let (Some(src), Some(dst)) = (
            block_map.get(c.from.block.as_str()),
            block_map.get(c.to.block.as_str()),
        ) {
            let src_ports = block_ports(&src.kind);
            let dst_ports = block_ports(&dst.kind);
            if let (Some(src_width), Some(dst_width)) = (
                src_ports.outputs.get(c.from.port),
                dst_ports.inputs.get(c.to.port),
            ) {
                if src_width != dst_width {
                    issues.push(err(
                        format!("connections.{idx}"),
                        format!(
                            "width mismatch: `{}` port {} has width {}, `{}` port {} expects width {}",
                            c.from.block,
                            c.from.port,
                            src_width,
                            c.to.block,
                            c.to.port,
                            dst_width
                        ),
                    ));
                }
            }
        }
    }

    let requirement_ids: HashSet<_> = spec
        .verification
        .requirements
        .iter()
        .map(|r| r.id.clone())
        .collect();
    validate_unique_named(
        spec.verification.requirements.iter().map(|r| r.id.as_str()),
        "verification.requirements",
        &mut issues,
    );

    let mut machine_ids = HashSet::new();
    let mut machine_map = HashMap::new();
    for sm in &spec.state_machines {
        validate_nonempty_id(&sm.id, format!("stateMachines.{}", sm.id), &mut issues);
        if !machine_ids.insert(sm.id.clone()) {
            issues.push(err(
                format!("stateMachines.{}", sm.id),
                "duplicate state machine id".to_string(),
            ));
        }
        machine_map.insert(sm.id.as_str(), sm);
        let mut states = HashSet::new();
        for s in &sm.states {
            validate_nonempty_id(
                &s.id,
                format!("stateMachines.{}.states.{}", sm.id, s.id),
                &mut issues,
            );
            if !states.insert(s.id.clone()) {
                issues.push(err(
                    format!("stateMachines.{}.states.{}", sm.id, s.id),
                    "duplicate state id".to_string(),
                ));
            }
        }
        if sm.states.is_empty() {
            issues.push(err(
                format!("stateMachines.{}.states", sm.id),
                "state machine must declare at least one state".to_string(),
            ));
        }
        if let Some(initial) = &sm.initial {
            if !states.contains(initial) {
                issues.push(err(
                    format!("stateMachines.{}.initial", sm.id),
                    format!("unknown initial state `{initial}`"),
                ));
            }
        }
        for (idx, t) in sm.transitions.iter().enumerate() {
            if !states.contains(&t.from) {
                issues.push(err(
                    format!("stateMachines.{}.transitions.{idx}.from", sm.id),
                    format!("unknown transition source `{}`", t.from),
                ));
            }
            if !states.contains(&t.to) {
                issues.push(err(
                    format!("stateMachines.{}.transitions.{idx}.to", sm.id),
                    format!("unknown transition target `{}`", t.to),
                ));
            }
            if let Some(req) = &t.requirement {
                if !requirement_ids.contains(req) {
                    issues.push(err(
                        format!("stateMachines.{}.transitions.{idx}.requirement", sm.id),
                        format!("unknown requirement `{req}`"),
                    ));
                }
            }
        }
    }
    for b in &spec.blocks {
        if let BlockKindSpec::StateMachine { machine, .. } = &b.kind {
            if !machine_ids.contains(machine) {
                issues.push(err(
                    format!("blocks.{}.kind.machine", b.id),
                    format!("unknown state machine `{machine}`"),
                ));
            }
        }
        if let BlockKindSpec::StateMachine {
            machine,
            input_width,
            ..
        } = &b.kind
        {
            if let Some(sm) = machine_map.get(machine.as_str()) {
                validate_state_machine_guards(&b.id, sm, *input_width, &mut issues);
            }
        }
    }

    validate_unique_named(
        spec.physical_networks.iter().map(|n| n.id.as_str()),
        "physicalNetworks",
        &mut issues,
    );
    for pn in &spec.physical_networks {
        validate_unique_named(
            pn.nodes.iter().map(|n| n.id.as_str()),
            format!("physicalNetworks.{}.nodes", pn.id),
            &mut issues,
        );
        validate_unique_named(
            pn.components.iter().map(|c| c.id.as_str()),
            format!("physicalNetworks.{}.components", pn.id),
            &mut issues,
        );
        for component in &pn.components {
            validate_physical_component(&pn.id, component, &mut issues);
        }
        let component_map: HashMap<_, _> =
            pn.components.iter().map(|c| (c.id.as_str(), c)).collect();
        let component_ids: HashSet<_> = pn.components.iter().map(|c| c.id.clone()).collect();
        let node_ids: HashSet<_> = pn.nodes.iter().map(|n| n.id.clone()).collect();
        let mut bound_connectors = HashSet::new();
        for (idx, c) in pn.connections.iter().enumerate() {
            if !component_ids.contains(&c.component) {
                issues.push(err(
                    format!("physicalNetworks.{}.connections.{idx}.component", pn.id),
                    format!("unknown component `{}`", c.component),
                ));
            }
            if !node_ids.contains(&c.node) {
                issues.push(err(
                    format!("physicalNetworks.{}.connections.{idx}.node", pn.id),
                    format!("unknown node `{}`", c.node),
                ));
            }
            if let Some(component) = component_map.get(c.component.as_str()) {
                let connectors = component_connectors(&component.kind);
                if !connectors.contains(&c.connector.as_str()) {
                    issues.push(err(
                        format!("physicalNetworks.{}.connections.{idx}.connector", pn.id),
                        format!(
                            "component `{}` has connector(s) {}, not `{}`",
                            c.component,
                            connectors.join(", "),
                            c.connector
                        ),
                    ));
                }
                if !bound_connectors.insert((c.component.as_str(), c.connector.as_str())) {
                    issues.push(err(
                        format!("physicalNetworks.{}.connections.{idx}.connector", pn.id),
                        format!(
                            "connector `{}.{}` is connected to more than one node",
                            c.component, c.connector
                        ),
                    ));
                }
            }
        }
    }

    for (idx, link) in spec.verification.trace_links.iter().enumerate() {
        if !requirement_ids.contains(&link.requirement) {
            issues.push(err(
                format!("verification.traceLinks.{idx}.requirement"),
                format!("unknown requirement `{}`", link.requirement),
            ));
        }
        if !model_element_exists(spec, &link.element) {
            issues.push(err(
                format!("verification.traceLinks.{idx}.element"),
                format!("unknown model element `{}`", link.element),
            ));
        }
    }

    issues
}

fn err(path: impl Into<String>, message: String) -> ValidationIssue {
    ValidationIssue {
        severity: ValidationSeverity::Error,
        path: path.into(),
        message,
    }
}

fn validate_nonempty_id(id: &str, path: impl Into<String>, issues: &mut Vec<ValidationIssue>) {
    if id.trim().is_empty() {
        issues.push(err(path, "id must not be empty".to_string()));
    }
}

fn validate_unique_named<'a>(
    ids: impl Iterator<Item = &'a str>,
    path: impl Into<String>,
    issues: &mut Vec<ValidationIssue>,
) {
    let path = path.into();
    let mut seen = HashSet::new();
    for id in ids {
        if id.trim().is_empty() {
            issues.push(err(
                format!("{path}.{id}"),
                "id must not be empty".to_string(),
            ));
        }
        if !seen.insert(id.to_string()) {
            issues.push(err(format!("{path}.{id}"), "duplicate id".to_string()));
        }
    }
}

fn validate_solver_spec(solver: &SolverSpec, issues: &mut Vec<ValidationIssue>) {
    validate_positive_finite("solver.tEnd", solver.t_end, issues);
    validate_positive_finite("solver.maxStep", solver.max_step, issues);
    validate_positive_finite("solver.relTol", solver.rel_tol, issues);
    validate_positive_finite("solver.absTol", solver.abs_tol, issues);
    if solver.max_step > solver.t_end {
        issues.push(err(
            "solver.maxStep",
            "maxStep must not exceed tEnd".to_string(),
        ));
    }
}

fn validate_block_kind(id: &str, kind: &BlockKindSpec, issues: &mut Vec<ValidationIssue>) {
    match kind {
        BlockKindSpec::Constant { value } => {
            if value.is_empty() {
                issues.push(err(
                    format!("blocks.{id}.kind.value"),
                    "constant value must contain at least one scalar".to_string(),
                ));
            }
            validate_finite_slice(format!("blocks.{id}.kind.value"), value, issues);
        }
        BlockKindSpec::Gain { width, k } => {
            validate_positive_usize(format!("blocks.{id}.kind.width"), *width, issues);
            validate_finite(format!("blocks.{id}.kind.k"), *k, issues);
        }
        BlockKindSpec::Sum { width, signs } => {
            validate_positive_usize(format!("blocks.{id}.kind.width"), *width, issues);
            if signs.is_empty() {
                issues.push(err(
                    format!("blocks.{id}.kind.signs"),
                    "sum must have at least one input sign".to_string(),
                ));
            }
            validate_finite_slice(format!("blocks.{id}.kind.signs"), signs, issues);
        }
        BlockKindSpec::Saturation { lo, hi } => {
            validate_finite(format!("blocks.{id}.kind.lo"), *lo, issues);
            validate_finite(format!("blocks.{id}.kind.hi"), *hi, issues);
            if lo > hi {
                issues.push(err(
                    format!("blocks.{id}.kind"),
                    "saturation lower bound must be <= upper bound".to_string(),
                ));
            }
        }
        BlockKindSpec::Integrator { initial } => {
            if initial.is_empty() {
                issues.push(err(
                    format!("blocks.{id}.kind.initial"),
                    "integrator initial state must not be empty".to_string(),
                ));
            }
            validate_finite_slice(format!("blocks.{id}.kind.initial"), initial, issues);
        }
        BlockKindSpec::StateSpace { a, b, c, d, x0 } => {
            validate_state_space(id, a, b, c, d, x0.as_deref(), issues);
        }
        BlockKindSpec::BouncingBall {
            height,
            velocity,
            restitution,
        } => {
            validate_finite(format!("blocks.{id}.kind.height"), *height, issues);
            validate_finite(format!("blocks.{id}.kind.velocity"), *velocity, issues);
            validate_finite(
                format!("blocks.{id}.kind.restitution"),
                *restitution,
                issues,
            );
            if *restitution < 0.0 {
                issues.push(err(
                    format!("blocks.{id}.kind.restitution"),
                    "restitution must be non-negative".to_string(),
                ));
            }
        }
        BlockKindSpec::DiscretePi { kp, ki, period } => {
            validate_finite(format!("blocks.{id}.kind.kp"), *kp, issues);
            validate_finite(format!("blocks.{id}.kind.ki"), *ki, issues);
            validate_positive_finite(format!("blocks.{id}.kind.period"), *period, issues);
        }
        BlockKindSpec::Counter { period } => {
            validate_positive_finite(format!("blocks.{id}.kind.period"), *period, issues);
        }
        BlockKindSpec::StateMachine {
            period,
            input_width,
            ..
        } => {
            validate_positive_finite(format!("blocks.{id}.kind.period"), *period, issues);
            if *input_width > 100_000 {
                issues.push(err(
                    format!("blocks.{id}.kind.inputWidth"),
                    "inputWidth is unreasonably large".to_string(),
                ));
            }
        }
        BlockKindSpec::Terminator { width } => {
            validate_positive_usize(format!("blocks.{id}.kind.width"), *width, issues);
        }
    }
}

fn validate_state_space(
    id: &str,
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: &[Vec<f64>],
    d: &[Vec<f64>],
    x0: Option<&[f64]>,
    issues: &mut Vec<ValidationIssue>,
) {
    let n = a.len();
    if n == 0 {
        issues.push(err(
            format!("blocks.{id}.kind.a"),
            "state-space matrix A must not be empty".to_string(),
        ));
        return;
    }
    validate_matrix_shape(format!("blocks.{id}.kind.a"), a, n, n, issues);
    let m = b.first().map_or(0, Vec::len);
    validate_matrix_shape(format!("blocks.{id}.kind.b"), b, n, m, issues);
    let p = c.len();
    if p == 0 {
        issues.push(err(
            format!("blocks.{id}.kind.c"),
            "state-space matrix C must not be empty".to_string(),
        ));
    }
    validate_matrix_shape(format!("blocks.{id}.kind.c"), c, p, n, issues);
    validate_matrix_shape(format!("blocks.{id}.kind.d"), d, p, m, issues);
    if let Some(x0) = x0 {
        if x0.len() != n {
            issues.push(err(
                format!("blocks.{id}.kind.x0"),
                format!("x0 length must be {n}, got {}", x0.len()),
            ));
        }
        validate_finite_slice(format!("blocks.{id}.kind.x0"), x0, issues);
    }
}

fn validate_matrix_shape(
    path: impl Into<String>,
    matrix: &[Vec<f64>],
    rows: usize,
    cols: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    let path = path.into();
    if matrix.len() != rows {
        issues.push(err(
            path.clone(),
            format!("expected {rows} row(s), got {}", matrix.len()),
        ));
        return;
    }
    for (idx, row) in matrix.iter().enumerate() {
        if row.len() != cols {
            issues.push(err(
                format!("{path}.{idx}"),
                format!("expected {cols} column(s), got {}", row.len()),
            ));
        }
        validate_finite_slice(format!("{path}.{idx}"), row, issues);
    }
}

fn validate_state_machine_guards(
    block_id: &str,
    machine: &StateMachineSpec,
    input_width: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    for (idx, transition) in machine.transitions.iter().enumerate() {
        let guard_index = match &transition.guard {
            GuardSpec::InputGreater { index, .. } | GuardSpec::InputLess { index, .. } => {
                Some(*index)
            }
            _ => None,
        };
        if let Some(index) = guard_index {
            if input_width == 0 || index >= input_width {
                issues.push(err(
                    format!("blocks.{block_id}.kind.machine"),
                    format!(
                        "transition {idx} guard reads input {index}, but block inputWidth is {input_width}"
                    ),
                ));
            }
        }
    }
}

fn validate_physical_component(
    network_id: &str,
    component: &PhysicalComponentSpec,
    issues: &mut Vec<ValidationIssue>,
) {
    let base = format!("physicalNetworks.{network_id}.components.{}", component.id);
    match &component.kind {
        PhysicalComponentKind::ElectricalResistor { resistance } => {
            validate_positive_finite(format!("{base}.resistance"), *resistance, issues);
        }
        PhysicalComponentKind::ElectricalCapacitor {
            capacitance,
            initial_voltage,
        } => {
            validate_positive_finite(format!("{base}.capacitance"), *capacitance, issues);
            validate_finite(format!("{base}.initialVoltage"), *initial_voltage, issues);
        }
        PhysicalComponentKind::ElectricalInductor {
            inductance,
            initial_current,
        } => {
            validate_positive_finite(format!("{base}.inductance"), *inductance, issues);
            validate_finite(format!("{base}.initialCurrent"), *initial_current, issues);
        }
        PhysicalComponentKind::ElectricalVoltageSource { voltage } => {
            validate_finite(format!("{base}.voltage"), *voltage, issues);
        }
        PhysicalComponentKind::ElectricalCurrentSource { current } => {
            validate_finite(format!("{base}.current"), *current, issues);
        }
        PhysicalComponentKind::ElectricalGround => {}
        PhysicalComponentKind::TranslationalMass { mass } => {
            validate_positive_finite(format!("{base}.mass"), *mass, issues);
        }
        PhysicalComponentKind::TranslationalSpring { stiffness } => {
            validate_positive_finite(format!("{base}.stiffness"), *stiffness, issues);
        }
        PhysicalComponentKind::TranslationalDamper { damping } => {
            validate_positive_finite(format!("{base}.damping"), *damping, issues);
        }
        PhysicalComponentKind::ThermalCapacitor { heat_capacity } => {
            validate_positive_finite(format!("{base}.heatCapacity"), *heat_capacity, issues);
        }
        PhysicalComponentKind::ThermalConductor { conductance } => {
            validate_positive_finite(format!("{base}.conductance"), *conductance, issues);
        }
        PhysicalComponentKind::FluidReservoir { pressure } => {
            validate_finite(format!("{base}.pressure"), *pressure, issues);
        }
        PhysicalComponentKind::FluidResistance { resistance } => {
            validate_positive_finite(format!("{base}.resistance"), *resistance, issues);
        }
    }
}

fn validate_positive_usize(
    path: impl Into<String>,
    value: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    if value == 0 {
        issues.push(err(path, "value must be positive".to_string()));
    }
}

fn validate_positive_finite(
    path: impl Into<String>,
    value: f64,
    issues: &mut Vec<ValidationIssue>,
) {
    let path = path.into();
    if !value.is_finite() || value <= 0.0 {
        issues.push(err(path, "value must be finite and positive".to_string()));
    }
}

fn validate_finite(path: impl Into<String>, value: f64, issues: &mut Vec<ValidationIssue>) {
    if !value.is_finite() {
        issues.push(err(path, "value must be finite".to_string()));
    }
}

fn validate_finite_slice(
    path: impl Into<String>,
    values: &[f64],
    issues: &mut Vec<ValidationIssue>,
) {
    let path = path.into();
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            issues.push(err(
                format!("{path}.{idx}"),
                "value must be finite".to_string(),
            ));
        }
    }
}

fn model_element_exists(spec: &AuthoringSpec, element: &str) -> bool {
    spec.blocks.iter().any(|b| b.id == element)
        || spec.state_machines.iter().any(|m| m.id == element)
        || spec.physical_networks.iter().any(|n| n.id == element)
        || spec.submodels.iter().any(|s| s.id == element)
        || spec.model_references.iter().any(|m| m.id == element)
}

#[derive(Clone, Debug)]
struct BlockPorts {
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

fn block_ports(kind: &BlockKindSpec) -> BlockPorts {
    match kind {
        BlockKindSpec::Constant { value } => BlockPorts {
            inputs: Vec::new(),
            outputs: vec![value.len()],
        },
        BlockKindSpec::Gain { width, .. } => BlockPorts {
            inputs: vec![*width],
            outputs: vec![*width],
        },
        BlockKindSpec::Sum { width, signs } => BlockPorts {
            inputs: vec![*width; signs.len()],
            outputs: vec![*width],
        },
        BlockKindSpec::Saturation { .. } => BlockPorts {
            inputs: vec![1],
            outputs: vec![1],
        },
        BlockKindSpec::Integrator { initial } => BlockPorts {
            inputs: vec![initial.len()],
            outputs: vec![initial.len()],
        },
        BlockKindSpec::StateSpace { b, c, d, .. } => {
            let m = b
                .first()
                .map_or_else(|| d.first().map_or(0, Vec::len), Vec::len);
            BlockPorts {
                inputs: vec![m],
                outputs: vec![c.len()],
            }
        }
        BlockKindSpec::BouncingBall { .. } => BlockPorts {
            inputs: Vec::new(),
            outputs: vec![2],
        },
        BlockKindSpec::DiscretePi { .. } => BlockPorts {
            inputs: vec![1],
            outputs: vec![1],
        },
        BlockKindSpec::Counter { .. } => BlockPorts {
            inputs: Vec::new(),
            outputs: vec![1],
        },
        BlockKindSpec::StateMachine { input_width, .. } => {
            if *input_width == 0 {
                BlockPorts {
                    inputs: Vec::new(),
                    outputs: vec![1],
                }
            } else {
                BlockPorts {
                    inputs: vec![*input_width],
                    outputs: vec![1],
                }
            }
        }
        BlockKindSpec::Terminator { width } => BlockPorts {
            inputs: vec![*width],
            outputs: Vec::new(),
        },
    }
}

fn effective_active_variants(spec: &AuthoringSpec) -> HashSet<String> {
    if !spec.active_variants.is_empty() {
        return spec.active_variants.iter().cloned().collect();
    }
    spec.variants
        .iter()
        .filter(|v| v.default)
        .map(|v| v.id.clone())
        .collect()
}

fn variant_allowed(active: &HashSet<String>, variant: &Option<String>) -> bool {
    match variant {
        Some(v) => active.contains(v),
        None => true,
    }
}

/// Compile the causal executable subset of an authoring spec into the existing
/// hybrid executive. Unsupported model sections remain available in the result
/// document and DAE flattening output.
pub fn compile_hybrid_graph(
    spec: &AuthoringSpec,
) -> Result<(Compiled, SimOptions), AuthoringError> {
    let failures: Vec<String> = validate_authoring_spec(spec)
        .into_iter()
        .filter(|i| i.severity == ValidationSeverity::Error)
        .map(|i| format!("{}: {}", i.path, i.message))
        .collect();
    if !failures.is_empty() {
        return Err(AuthoringError::InvalidSpec(failures.join("; ")));
    }
    match &spec.solver.mode {
        SolverMode::FixedStepRk4 | SolverMode::DiscreteOnly => {}
        SolverMode::VariableStepRk45 | SolverMode::BackwardEuler | SolverMode::DaeResidual => {
            return Err(AuthoringError::Compile(format!(
                "solver mode `{:?}` is declared but the causal hybrid compiler currently executes only fixed-step RK4/discrete graphs",
                spec.solver.mode
            )));
        }
    }
    if !matches!(&spec.solver.algebraic_loops, AlgebraicLoopPolicy::Reject) {
        return Err(AuthoringError::Compile(format!(
            "algebraic loop policy `{:?}` is declared but the causal hybrid compiler currently supports only `reject`",
            spec.solver.algebraic_loops
        )));
    }

    let active = effective_active_variants(spec);
    let mut d = Diagram::new();
    let machines: HashMap<String, StateMachineSpec> = spec
        .state_machines
        .iter()
        .map(|m| (m.id.clone(), m.clone()))
        .collect();
    let mut handles = HashMap::new();
    let mut active_blocks = HashSet::new();
    let mut driven_inputs = HashSet::new();

    for block in &spec.blocks {
        if !variant_allowed(&active, &block.variant) {
            continue;
        }
        let handle = d.add(build_block(block, &machines)?);
        handles.insert(block.id.clone(), handle);
        active_blocks.insert(block.id.clone());
    }

    for conn in &spec.connections {
        if !variant_allowed(&active, &conn.variant) {
            continue;
        }
        let Some(src) = handles.get(&conn.from.block).copied() else {
            return Err(AuthoringError::Compile(format!(
                "active connection references missing or inactive source block `{}`",
                conn.from.block
            )));
        };
        let Some(dst) = handles.get(&conn.to.block).copied() else {
            return Err(AuthoringError::Compile(format!(
                "active connection references missing or inactive destination block `{}`",
                conn.to.block
            )));
        };
        d.connect((src, conn.from.port), (dst, conn.to.port))?;
        driven_inputs.insert((conn.to.block.clone(), conn.to.port));
    }

    for block in spec
        .blocks
        .iter()
        .filter(|b| active_blocks.contains(b.id.as_str()))
    {
        let ports = block_ports(&block.kind);
        for (port, width) in ports.inputs.iter().enumerate() {
            if *width > 0 && !driven_inputs.contains(&(block.id.clone(), port)) {
                return Err(AuthoringError::Compile(format!(
                    "active block `{}` input port {port} is not connected",
                    block.id
                )));
            }
        }
    }

    let compiled = d.build()?;
    let opts = SimOptions {
        t_end: spec.solver.t_end,
        max_step: spec.solver.max_step,
        zc_tol: spec.solver.abs_tol.max(1e-12),
    };
    Ok((compiled, opts))
}

fn build_block(
    spec: &BlockSpec,
    machines: &HashMap<String, StateMachineSpec>,
) -> Result<Box<dyn Block>, AuthoringError> {
    let name = spec.label.as_deref().unwrap_or(&spec.id);
    let block: Box<dyn Block> = match &spec.kind {
        BlockKindSpec::Constant { value } => Box::new(Constant::new(name, value.clone())),
        BlockKindSpec::Gain { width, k } => Box::new(Gain::new(name, *width, *k)),
        BlockKindSpec::Sum { width, signs } => Box::new(Sum::new(name, *width, signs.clone())),
        BlockKindSpec::Saturation { lo, hi } => Box::new(Saturation::new(name, *lo, *hi)),
        BlockKindSpec::Integrator { initial } => Box::new(Integrator::new(name, initial.clone())),
        BlockKindSpec::StateSpace { a, b, c, d, x0 } => {
            let ss = StateSpace::new(name, a.clone(), b.clone(), c.clone(), d.clone());
            match x0 {
                Some(x0) => Box::new(ss.with_x0(x0.clone())),
                None => Box::new(ss),
            }
        }
        BlockKindSpec::BouncingBall {
            height,
            velocity,
            restitution,
        } => Box::new(BouncingBall::new(name, *height, *velocity, *restitution)),
        BlockKindSpec::DiscretePi { kp, ki, period } => {
            Box::new(DiscretePi::new(name, *kp, *ki, *period))
        }
        BlockKindSpec::Counter { period } => Box::new(Counter::new(name, *period)),
        BlockKindSpec::StateMachine {
            machine,
            period,
            input_width,
        } => {
            let Some(sm) = machines.get(machine) else {
                return Err(AuthoringError::Compile(format!(
                    "block `{}` references unknown state machine `{machine}`",
                    spec.id
                )));
            };
            Box::new(StateMachineBlock::new(
                name,
                sm.clone(),
                *period,
                *input_width,
            ))
        }
        BlockKindSpec::Terminator { width } => Box::new(TerminatorBlock::new(name, *width)),
    };
    Ok(block)
}

/// A small Stateflow-like discrete block. It exposes the active state's numeric
/// index and advances on its sample period according to declarative guards.
pub struct StateMachineBlock {
    name: String,
    machine: StateMachineSpec,
    period: f64,
    input_width: usize,
}

impl StateMachineBlock {
    pub fn new(name: &str, machine: StateMachineSpec, period: f64, input_width: usize) -> Self {
        StateMachineBlock {
            name: name.to_string(),
            machine,
            period,
            input_width,
        }
    }

    fn initial_index(&self) -> usize {
        let initial = self
            .machine
            .initial
            .as_deref()
            .or_else(|| self.machine.states.first().map(|s| s.id.as_str()));
        initial
            .and_then(|id| self.machine.states.iter().position(|s| s.id == id))
            .unwrap_or(0)
    }
}

impl Block for StateMachineBlock {
    fn name(&self) -> &str {
        &self.name
    }

    fn port_spec(&self) -> PortSpec {
        if self.input_width == 0 {
            PortSpec::source(1)
        } else {
            PortSpec::siso(self.input_width, 1)
        }
    }

    fn sample_time(&self) -> SampleTime {
        SampleTime::Discrete {
            period: self.period,
            offset: 0.0,
        }
    }

    fn n_disc(&self) -> usize {
        1
    }

    fn init_disc(&self) -> Vec<f64> {
        vec![self.initial_index() as f64]
    }

    fn feedthrough(&self) -> bool {
        false
    }

    fn outputs(&self, _t: f64, _xc: &[f64], xd: &[f64], _u: &[Signal]) -> Vec<Signal> {
        vec![vec![xd.first().copied().unwrap_or(0.0)]]
    }

    fn update(&self, t: f64, xd: &mut Vec<f64>, u: &[Signal]) {
        let idx = xd.first().copied().unwrap_or(0.0).round().max(0.0) as usize;
        let Some(active) = self.machine.states.get(idx).map(|s| s.id.as_str()) else {
            return;
        };
        for tr in &self.machine.transitions {
            if tr.from != active {
                continue;
            }
            if guard_matches(&tr.guard, t, u) {
                if let Some(next) = self.machine.states.iter().position(|s| s.id == tr.to) {
                    xd[0] = next as f64;
                    break;
                }
            }
        }
    }
}

fn guard_matches(guard: &GuardSpec, t: f64, u: &[Signal]) -> bool {
    match guard {
        GuardSpec::Always => true,
        GuardSpec::After { time } => t >= *time,
        GuardSpec::InputGreater { index, threshold } => u
            .first()
            .and_then(|sig| sig.get(*index))
            .is_some_and(|v| *v > *threshold),
        GuardSpec::InputLess { index, threshold } => u
            .first()
            .and_then(|sig| sig.get(*index))
            .is_some_and(|v| *v < *threshold),
    }
}

pub struct TerminatorBlock {
    name: String,
    width: usize,
}

impl TerminatorBlock {
    pub fn new(name: &str, width: usize) -> Self {
        TerminatorBlock {
            name: name.to_string(),
            width,
        }
    }
}

impl Block for TerminatorBlock {
    fn name(&self) -> &str {
        &self.name
    }

    fn port_spec(&self) -> PortSpec {
        PortSpec::sink(self.width)
    }

    fn sample_time(&self) -> SampleTime {
        SampleTime::Constant
    }

    fn feedthrough(&self) -> bool {
        false
    }

    fn outputs(&self, _t: f64, _xc: &[f64], _xd: &[f64], _u: &[Signal]) -> Vec<Signal> {
        Vec::new()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DaeModel {
    pub variables: Vec<DaeVariable>,
    pub equations: Vec<DaeEquation>,
    pub initial_equations: Vec<DaeEquation>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DaeVariable {
    pub name: String,
    pub kind: DaeVariableKind,
    #[serde(default)]
    pub unit: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum DaeVariableKind {
    Potential,
    Flow,
    State,
    Parameter,
    Algebraic,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DaeEquation {
    pub expression: String,
    pub source: String,
}

/// Flatten acausal physical connectors into textual hybrid DAE equations.
///
/// This is intentionally a compiler IR, not yet a full numerical DAE solver. It
/// captures the Modelica/Simscape semantics we need next: potentials equalize at
/// a node, flows sum to zero, and components contribute constitutive equations.
pub fn flatten_physical_networks(spec: &AuthoringSpec) -> DaeModel {
    let mut variables = Vec::new();
    let mut equations = Vec::new();
    let mut initial_equations = Vec::new();
    let mut diagnostics = Vec::new();

    for network in &spec.physical_networks {
        for node in &network.nodes {
            variables.push(DaeVariable {
                name: format!("{}.{}.potential", network.id, node.id),
                kind: DaeVariableKind::Potential,
                unit: node.quantity.clone(),
            });
        }

        for component in &network.components {
            emit_component_equations(
                network,
                component,
                &mut variables,
                &mut equations,
                &mut initial_equations,
            );
        }

        let mut node_flows: HashMap<String, Vec<String>> = HashMap::new();
        for conn in &network.connections {
            let potential = format!(
                "{}.{}.{}.potential",
                network.id, conn.component, conn.connector
            );
            let node_potential = format!("{}.{}.potential", network.id, conn.node);
            equations.push(DaeEquation {
                expression: format!("{potential} = {node_potential}"),
                source: format!(
                    "connect({}.{}, {})",
                    conn.component, conn.connector, conn.node
                ),
            });
            node_flows
                .entry(conn.node.clone())
                .or_default()
                .push(format!(
                    "{}.{}.{}.flow",
                    network.id, conn.component, conn.connector
                ));
        }
        for (node, flows) in node_flows {
            if !flows.is_empty() {
                equations.push(DaeEquation {
                    expression: format!("0 = {}", flows.join(" + ")),
                    source: format!("node-flow-balance({}.{})", network.id, node),
                });
            }
        }
    }

    if !spec.physical_networks.is_empty() {
        diagnostics.push(
            "physical networks are flattened to hybrid DAE IR; numeric DAE solve/index reduction is the next compiler pass"
                .to_string(),
        );
    }

    DaeModel {
        variables,
        equations,
        initial_equations,
        diagnostics,
    }
}

fn emit_component_equations(
    network: &PhysicalNetworkSpec,
    component: &PhysicalComponentSpec,
    variables: &mut Vec<DaeVariable>,
    equations: &mut Vec<DaeEquation>,
    initial_equations: &mut Vec<DaeEquation>,
) {
    let n = &network.id;
    let c = &component.id;
    let source = format!("{n}.{c}");
    for connector in component_connectors(&component.kind) {
        variables.push(DaeVariable {
            name: format!("{n}.{c}.{connector}.potential"),
            kind: DaeVariableKind::Potential,
            unit: None,
        });
        variables.push(DaeVariable {
            name: format!("{n}.{c}.{connector}.flow"),
            kind: DaeVariableKind::Flow,
            unit: None,
        });
    }

    let expr = match &component.kind {
        PhysicalComponentKind::ElectricalResistor { resistance } => {
            format!("{n}.{c}.p.flow = ({n}.{c}.p.potential - {n}.{c}.n.potential) / {resistance}")
        }
        PhysicalComponentKind::ElectricalCapacitor {
            capacitance,
            initial_voltage,
        } => {
            initial_equations.push(DaeEquation {
                expression: format!(
                    "{n}.{c}.p.potential - {n}.{c}.n.potential = {initial_voltage}"
                ),
                source: source.clone(),
            });
            format!(
                "{n}.{c}.p.flow = {capacitance} * der({n}.{c}.p.potential - {n}.{c}.n.potential)"
            )
        }
        PhysicalComponentKind::ElectricalInductor {
            inductance,
            initial_current,
        } => {
            initial_equations.push(DaeEquation {
                expression: format!("{n}.{c}.p.flow = {initial_current}"),
                source: source.clone(),
            });
            format!(
                "{n}.{c}.p.potential - {n}.{c}.n.potential = {inductance} * der({n}.{c}.p.flow)"
            )
        }
        PhysicalComponentKind::ElectricalVoltageSource { voltage } => {
            format!("{n}.{c}.p.potential - {n}.{c}.n.potential = {voltage}")
        }
        PhysicalComponentKind::ElectricalCurrentSource { current } => {
            format!("{n}.{c}.p.flow = {current}")
        }
        PhysicalComponentKind::ElectricalGround => format!("{n}.{c}.p.potential = 0"),
        PhysicalComponentKind::TranslationalMass { mass } => {
            format!("{n}.{c}.flange.flow = {mass} * der(der({n}.{c}.flange.potential))")
        }
        PhysicalComponentKind::TranslationalSpring { stiffness } => {
            format!("{n}.{c}.a.flow = {stiffness} * ({n}.{c}.a.potential - {n}.{c}.b.potential)")
        }
        PhysicalComponentKind::TranslationalDamper { damping } => {
            format!("{n}.{c}.a.flow = {damping} * der({n}.{c}.a.potential - {n}.{c}.b.potential)")
        }
        PhysicalComponentKind::ThermalCapacitor { heat_capacity } => {
            format!("{n}.{c}.p.flow = {heat_capacity} * der({n}.{c}.p.potential)")
        }
        PhysicalComponentKind::ThermalConductor { conductance } => {
            format!("{n}.{c}.a.flow = {conductance} * ({n}.{c}.a.potential - {n}.{c}.b.potential)")
        }
        PhysicalComponentKind::FluidReservoir { pressure } => {
            format!("{n}.{c}.p.potential = {pressure}")
        }
        PhysicalComponentKind::FluidResistance { resistance } => {
            format!("{n}.{c}.a.flow = ({n}.{c}.a.potential - {n}.{c}.b.potential) / {resistance}")
        }
    };
    equations.push(DaeEquation {
        expression: expr,
        source: source.clone(),
    });

    let connectors = component_connectors(&component.kind);
    if connectors.len() == 2 {
        equations.push(DaeEquation {
            expression: format!(
                "{n}.{c}.{}.flow + {n}.{c}.{}.flow = 0",
                connectors[0], connectors[1]
            ),
            source,
        });
    }
}

fn component_connectors(kind: &PhysicalComponentKind) -> &'static [&'static str] {
    match kind {
        PhysicalComponentKind::ElectricalGround
        | PhysicalComponentKind::ThermalCapacitor { .. }
        | PhysicalComponentKind::FluidReservoir { .. }
        | PhysicalComponentKind::TranslationalMass { .. } => &["p"],
        PhysicalComponentKind::TranslationalSpring { .. }
        | PhysicalComponentKind::TranslationalDamper { .. }
        | PhysicalComponentKind::ThermalConductor { .. }
        | PhysicalComponentKind::FluidResistance { .. } => &["a", "b"],
        _ => &["p", "n"],
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RustCodegenResult {
    pub module_name: String,
    pub source: String,
}

/// Generate Rust source that embeds this JSON spec, type-checks it through
/// [`AuthoringSpec`], compiles the supported subset, and runs the hybrid engine.
pub fn generate_rust(spec: &AuthoringSpec) -> Result<RustCodegenResult, AuthoringError> {
    let module_name = sanitize_ident(spec.codegen.module_name.as_deref().unwrap_or(&spec.name));
    let spec_text =
        serde_json::to_string_pretty(spec).map_err(|e| AuthoringError::Codegen(e.to_string()))?;
    let schema_text = if spec.codegen.include_schema {
        serde_json::to_string_pretty(&authoring_json_schema())
            .map_err(|e| AuthoringError::Codegen(e.to_string()))?
    } else {
        "{}".to_string()
    };
    let spec_literal = rust_string_literal(&spec_text);
    let schema_literal = rust_string_literal(&schema_text);
    let source = format!(
        r####"pub mod {module_name} {{
    pub const SPEC_JSON: &str = {spec_literal};
    pub const JSON_SCHEMA: &str = {schema_literal};

    pub fn run() -> Result<des_engine::des::hybrid::Trace, Box<dyn std::error::Error>> {{
        let value: serde_json::Value = serde_json::from_str(SPEC_JSON)?;
        let spec = des_engine::des::authoring::parse_authoring_spec(&value)?;
        let (compiled, opts) = des_engine::des::authoring::compile_hybrid_graph(&spec)?;
        Ok(des_engine::des::hybrid::simulate(&compiled, &opts))
    }}

    pub fn rust_codegen_target() -> &'static str {{
        "rust"
    }}
}}
"####
    );
    Ok(RustCodegenResult {
        module_name,
        source,
    })
}

fn sanitize_ident(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() || out.as_bytes().first().is_some_and(|b| b.is_ascii_digit()) {
        out.insert_str(0, "model_");
    }
    if is_rust_keyword(&out) {
        out.insert_str(0, "model_");
    }
    out
}

fn rust_string_literal(value: &str) -> String {
    format!("{value:?}")
}

fn is_rust_keyword(value: &str) -> bool {
    matches!(
        value,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
    )
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct LibraryBlockDescriptor {
    pub id: String,
    pub category: String,
    pub title: String,
    pub description: String,
    pub schema_fragment: Value,
}

/// Built-in authoring library catalog for a UI/library browser.
pub fn library_catalog() -> Vec<LibraryBlockDescriptor> {
    vec![
        lib("constant", "Sources", "Constant", "Fixed vector source"),
        lib("gain", "Math", "Gain", "Elementwise scalar gain"),
        lib("sum", "Math", "Sum", "Weighted sum of equal-width inputs"),
        lib(
            "saturation",
            "Math",
            "Saturation",
            "Clamp a scalar signal with event surfaces",
        ),
        lib(
            "integrator",
            "Continuous",
            "Integrator",
            "Continuous integrator",
        ),
        lib(
            "state-space",
            "Continuous",
            "State Space",
            "Continuous LTI state-space plant",
        ),
        lib(
            "discrete-pi",
            "Discrete",
            "Discrete PI",
            "Sampled PI controller with zero-order hold",
        ),
        lib("counter", "Discrete", "Counter", "Discrete sample counter"),
        lib(
            "bouncing-ball",
            "Hybrid",
            "Bouncing Ball",
            "Continuous dynamics with event reset",
        ),
        lib(
            "state-machine",
            "State Machines",
            "State Machine",
            "Stateflow-like sampled logic",
        ),
        lib(
            "physical-network",
            "Physical",
            "Physical Network",
            "Acausal connector network",
        ),
        lib("fmu", "FMI", "FMU", "FMI import/export declaration"),
        lib(
            "requirement",
            "Verification",
            "Requirement",
            "Traceable requirement item",
        ),
    ]
}

fn lib(id: &str, category: &str, title: &str, description: &str) -> LibraryBlockDescriptor {
    LibraryBlockDescriptor {
        id: id.to_string(),
        category: category.to_string(),
        title: title.to_string(),
        description: description.to_string(),
        schema_fragment: json!({ "type": id }),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelBrowser {
    pub model: String,
    pub blocks: Vec<String>,
    pub submodels: Vec<String>,
    pub model_references: Vec<String>,
    pub physical_networks: Vec<String>,
    pub state_machines: Vec<String>,
    pub variants: Vec<String>,
    pub requirements: Vec<String>,
}

pub fn model_browser(spec: &AuthoringSpec) -> ModelBrowser {
    ModelBrowser {
        model: spec.name.clone(),
        blocks: spec.blocks.iter().map(|b| b.id.clone()).collect(),
        submodels: spec.submodels.iter().map(|m| m.id.clone()).collect(),
        model_references: spec.model_references.iter().map(|m| m.id.clone()).collect(),
        physical_networks: spec
            .physical_networks
            .iter()
            .map(|n| n.id.clone())
            .collect(),
        state_machines: spec.state_machines.iter().map(|m| m.id.clone()).collect(),
        variants: spec.variants.iter().map(|v| v.id.clone()).collect(),
        requirements: spec
            .verification
            .requirements
            .iter()
            .map(|r| r.id.clone())
            .collect(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DependencyGraph {
    pub nodes: Vec<DependencyNode>,
    pub edges: Vec<DependencyEdge>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DependencyNode {
    pub id: String,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DependencyEdge {
    pub from: String,
    pub to: String,
    pub kind: String,
}

pub fn dependency_graph(spec: &AuthoringSpec) -> DependencyGraph {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for b in &spec.blocks {
        nodes.push(DependencyNode {
            id: b.id.clone(),
            kind: "block".to_string(),
        });
        for req in &b.requirements {
            edges.push(DependencyEdge {
                from: req.clone(),
                to: b.id.clone(),
                kind: "satisfies".to_string(),
            });
        }
    }
    for c in &spec.connections {
        edges.push(DependencyEdge {
            from: c.from.block.clone(),
            to: c.to.block.clone(),
            kind: "signal".to_string(),
        });
    }
    for r in &spec.verification.requirements {
        nodes.push(DependencyNode {
            id: r.id.clone(),
            kind: "requirement".to_string(),
        });
    }
    for link in &spec.verification.trace_links {
        edges.push(DependencyEdge {
            from: link.requirement.clone(),
            to: link.element.clone(),
            kind: format!("{:?}", link.relation).to_ascii_lowercase(),
        });
    }
    DependencyGraph { nodes, edges }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TraceCompareOptions {
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub time_tol: f64,
}

impl Default for TraceCompareOptions {
    fn default() -> Self {
        TraceCompareOptions {
            abs_tol: 1e-9,
            rel_tol: 1e-6,
            time_tol: 1e-9,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TraceComparisonReport {
    pub passed: bool,
    pub samples_compared: usize,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub mismatches: Vec<TraceMismatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TraceMismatch {
    pub sample: usize,
    pub column: String,
    pub lhs: f64,
    pub rhs: f64,
    pub abs_error: f64,
    pub rel_error: f64,
}

pub fn compare_traces(
    lhs: &Trace,
    rhs: &Trace,
    opts: TraceCompareOptions,
) -> TraceComparisonReport {
    let mut mismatches = Vec::new();
    let mut max_abs_error = 0.0;
    let mut max_rel_error = 0.0;
    let n = lhs.times.len().min(rhs.times.len());
    let columns: Vec<(usize, usize, String)> = lhs
        .columns
        .iter()
        .enumerate()
        .filter_map(|(li, name)| rhs.column_index(name).map(|ri| (li, ri, name.clone())))
        .collect();
    for name in &lhs.columns {
        if rhs.column_index(name).is_none() {
            mismatches.push(structural_mismatch(format!("missing rhs column `{name}`")));
        }
    }
    for name in &rhs.columns {
        if lhs.column_index(name).is_none() {
            mismatches.push(structural_mismatch(format!("missing lhs column `{name}`")));
        }
    }
    if lhs.times.len() != rhs.times.len() {
        mismatches.push(structural_mismatch(format!(
            "sample count differs: lhs={}, rhs={}",
            lhs.times.len(),
            rhs.times.len()
        )));
    }
    for k in 0..n {
        if (lhs.times[k] - rhs.times[k]).abs() > opts.time_tol {
            mismatches.push(TraceMismatch {
                sample: k,
                column: "t".to_string(),
                lhs: lhs.times[k],
                rhs: rhs.times[k],
                abs_error: (lhs.times[k] - rhs.times[k]).abs(),
                rel_error: 0.0,
            });
            continue;
        }
        for (li, ri, name) in &columns {
            let a = lhs.rows[k][*li];
            let b = rhs.rows[k][*ri];
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1.0);
            max_abs_error = f64::max(max_abs_error, abs);
            max_rel_error = f64::max(max_rel_error, rel);
            if abs > opts.abs_tol && rel > opts.rel_tol {
                mismatches.push(TraceMismatch {
                    sample: k,
                    column: name.clone(),
                    lhs: a,
                    rhs: b,
                    abs_error: abs,
                    rel_error: rel,
                });
            }
        }
    }
    TraceComparisonReport {
        passed: mismatches.is_empty() && lhs.times.len() == rhs.times.len(),
        samples_compared: n,
        max_abs_error,
        max_rel_error,
        mismatches,
    }
}

fn structural_mismatch(message: String) -> TraceMismatch {
    TraceMismatch {
        sample: 0,
        column: message,
        lhs: 0.0,
        rhs: 0.0,
        abs_error: 1.0,
        rel_error: 1.0,
    }
}

/// First-class citizen for arbitrary graph specs.
pub struct AuthoringCitizen;

impl ModelCitizen for AuthoringCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "model-graph".to_string(),
            title: "Typed Model Graph".to_string(),
            description: "JSON Schema-backed authoring format with causal blocks, variants, \
                          hierarchy metadata, acausal physical-network IR, state machines, \
                          V&V metadata, FMI declarations, and Rust code generation."
                .to_string(),
            spec_schema: AUTHORING_SCHEMA.to_string(),
            methods: vec![
                "simulate".to_string(),
                "flatten-dae".to_string(),
                "generate-rust".to_string(),
                "inspect".to_string(),
            ],
            example_spec: serde_json::to_value(example_spec()).unwrap_or_else(|_| json!({})),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        let spec =
            parse_authoring_spec(spec).map_err(|e| CitizenError::InvalidSpec(e.to_string()))?;
        let dae = flatten_physical_networks(&spec);
        let rust = generate_rust(&spec).map_err(|e| CitizenError::Run(e.to_string()))?;
        let browser = model_browser(&spec);
        let deps = dependency_graph(&spec);
        let schema = if spec.codegen.include_schema {
            authoring_json_schema()
        } else {
            json!({ "id": AUTHORING_SCHEMA })
        };

        if spec.blocks.is_empty() {
            let results = json!({
                "kind": "model-graph",
                "schema": schema,
                "model": spec.name,
                "dae": dae,
                "browser": browser,
                "dependencies": deps,
                "library": library_catalog(),
                "rust": rust,
                "verification": spec.verification,
                "fmi": spec.fmi,
                "tooling": spec.tooling,
            });
            return Ok(RunArtifact::results(
                "model-graph",
                "Typed Model Graph",
                "Validated non-simulated authoring document.",
                results,
                Vec::new(),
                "Authoring spec validated; no executable causal block graph was present.",
            ));
        }

        let (compiled, opts) =
            compile_hybrid_graph(&spec).map_err(|e| CitizenError::Run(e.to_string()))?;
        let trace = simulate(&compiled, &opts);
        let frames = trace.to_jsonl_frames();
        let results = json!({
            "kind": "model-graph",
            "schema": schema,
            "model": spec.name,
            "solver": spec.solver,
            "columns": trace.columns,
            "samples": trace.times.len(),
            "events": trace.events,
            "dae": dae,
            "browser": browser,
            "dependencies": deps,
            "library": library_catalog(),
            "rust": rust,
            "verification": spec.verification,
            "fmi": spec.fmi,
            "tooling": spec.tooling,
        });
        let summary = format!(
            "Model graph `{}` run: {} samples, {} event(s).",
            spec.name,
            trace.times.len(),
            trace.events
        );
        Ok(RunArtifact::sim(
            "model-graph",
            "Typed Model Graph",
            "Schema-backed model graph rendered through the standard sim player.",
            frames,
            results,
            vec![UiControl::range(
                "speed",
                "Speed (fps)",
                1.0,
                60.0,
                1.0,
                20.0,
            )],
            &summary,
        ))
    }
}

/// A small but representative model: a setpoint, state-space plant, discrete PI
/// controller, state-machine supervisor, requirement trace link, and an
/// acausal RC network flattened as DAE IR.
pub fn example_spec() -> AuthoringSpec {
    AuthoringSpec {
        schema: AUTHORING_SCHEMA.to_string(),
        name: "closed-loop-authoring-demo".to_string(),
        version: Some("1.0.0".to_string()),
        description: Some(
            "Typed graph spec exercising hybrid simulation and authoring metadata.".to_string(),
        ),
        solver: SolverSpec {
            t_end: 2.0,
            max_step: 0.01,
            ..SolverSpec::default()
        },
        units: vec![UnitSpec {
            name: "volt".to_string(),
            quantity: "ElectricPotential".to_string(),
            unit: "V".to_string(),
            display_unit: None,
            scale_to_si: Some(1.0),
            offset_to_si: Some(0.0),
        }],
        data_dictionary: vec![DataDictionaryEntry {
            name: "setpoint".to_string(),
            value: json!(1.0),
            unit: Some("1".to_string()),
            description: Some("Control target.".to_string()),
            protected: false,
        }],
        variants: vec![VariantSpec {
            id: "nominal".to_string(),
            description: Some("Nominal controller configuration.".to_string()),
            default: true,
            constraints: Vec::new(),
        }],
        active_variants: vec!["nominal".to_string()],
        blocks: vec![
            BlockSpec {
                id: "reference".to_string(),
                label: Some("reference".to_string()),
                variant: None,
                position: None,
                kind: BlockKindSpec::Constant { value: vec![1.0] },
                requirements: vec!["REQ-TRACK".to_string()],
            },
            BlockSpec {
                id: "error".to_string(),
                label: Some("error".to_string()),
                variant: None,
                position: None,
                kind: BlockKindSpec::Sum {
                    width: 1,
                    signs: vec![1.0, -1.0],
                },
                requirements: Vec::new(),
            },
            BlockSpec {
                id: "controller".to_string(),
                label: Some("controller".to_string()),
                variant: Some("nominal".to_string()),
                position: None,
                kind: BlockKindSpec::DiscretePi {
                    kp: 2.0,
                    ki: 1.5,
                    period: 0.1,
                },
                requirements: vec!["REQ-TRACK".to_string()],
            },
            BlockSpec {
                id: "plant".to_string(),
                label: Some("plant".to_string()),
                variant: None,
                position: None,
                kind: BlockKindSpec::StateSpace {
                    a: vec![vec![-1.0]],
                    b: vec![vec![1.0]],
                    c: vec![vec![1.0]],
                    d: vec![vec![0.0]],
                    x0: None,
                },
                requirements: vec!["REQ-TRACK".to_string()],
            },
            BlockSpec {
                id: "supervisor".to_string(),
                label: Some("supervisor".to_string()),
                variant: None,
                position: None,
                kind: BlockKindSpec::StateMachine {
                    machine: "mode".to_string(),
                    period: 0.5,
                    input_width: 0,
                },
                requirements: vec!["REQ-MODE".to_string()],
            },
        ],
        connections: vec![
            ConnectionSpec {
                from: PortRef {
                    block: "reference".to_string(),
                    port: 0,
                },
                to: PortRef {
                    block: "error".to_string(),
                    port: 0,
                },
                variant: None,
                requirements: Vec::new(),
            },
            ConnectionSpec {
                from: PortRef {
                    block: "plant".to_string(),
                    port: 0,
                },
                to: PortRef {
                    block: "error".to_string(),
                    port: 1,
                },
                variant: None,
                requirements: Vec::new(),
            },
            ConnectionSpec {
                from: PortRef {
                    block: "error".to_string(),
                    port: 0,
                },
                to: PortRef {
                    block: "controller".to_string(),
                    port: 0,
                },
                variant: Some("nominal".to_string()),
                requirements: Vec::new(),
            },
            ConnectionSpec {
                from: PortRef {
                    block: "controller".to_string(),
                    port: 0,
                },
                to: PortRef {
                    block: "plant".to_string(),
                    port: 0,
                },
                variant: Some("nominal".to_string()),
                requirements: Vec::new(),
            },
        ],
        submodels: Vec::new(),
        model_references: Vec::new(),
        physical_networks: vec![PhysicalNetworkSpec {
            id: "rc".to_string(),
            domain: PhysicalDomain::Electrical,
            nodes: vec![
                PhysicalNodeSpec {
                    id: "vin".to_string(),
                    quantity: Some("V".to_string()),
                },
                PhysicalNodeSpec {
                    id: "vout".to_string(),
                    quantity: Some("V".to_string()),
                },
                PhysicalNodeSpec {
                    id: "gnd".to_string(),
                    quantity: Some("V".to_string()),
                },
            ],
            components: vec![
                PhysicalComponentSpec {
                    id: "source".to_string(),
                    kind: PhysicalComponentKind::ElectricalVoltageSource { voltage: 5.0 },
                },
                PhysicalComponentSpec {
                    id: "r1".to_string(),
                    kind: PhysicalComponentKind::ElectricalResistor { resistance: 1000.0 },
                },
                PhysicalComponentSpec {
                    id: "c1".to_string(),
                    kind: PhysicalComponentKind::ElectricalCapacitor {
                        capacitance: 1e-6,
                        initial_voltage: 0.0,
                    },
                },
                PhysicalComponentSpec {
                    id: "ground".to_string(),
                    kind: PhysicalComponentKind::ElectricalGround,
                },
            ],
            connections: vec![
                PhysicalConnectionSpec {
                    component: "source".to_string(),
                    connector: "p".to_string(),
                    node: "vin".to_string(),
                },
                PhysicalConnectionSpec {
                    component: "source".to_string(),
                    connector: "n".to_string(),
                    node: "gnd".to_string(),
                },
                PhysicalConnectionSpec {
                    component: "r1".to_string(),
                    connector: "p".to_string(),
                    node: "vin".to_string(),
                },
                PhysicalConnectionSpec {
                    component: "r1".to_string(),
                    connector: "n".to_string(),
                    node: "vout".to_string(),
                },
                PhysicalConnectionSpec {
                    component: "c1".to_string(),
                    connector: "p".to_string(),
                    node: "vout".to_string(),
                },
                PhysicalConnectionSpec {
                    component: "c1".to_string(),
                    connector: "n".to_string(),
                    node: "gnd".to_string(),
                },
                PhysicalConnectionSpec {
                    component: "ground".to_string(),
                    connector: "p".to_string(),
                    node: "gnd".to_string(),
                },
            ],
        }],
        state_machines: vec![StateMachineSpec {
            id: "mode".to_string(),
            initial: Some("startup".to_string()),
            states: vec![
                StateSpec {
                    id: "startup".to_string(),
                    label: Some("Startup".to_string()),
                },
                StateSpec {
                    id: "run".to_string(),
                    label: Some("Run".to_string()),
                },
            ],
            transitions: vec![TransitionSpec {
                from: "startup".to_string(),
                to: "run".to_string(),
                guard: GuardSpec::After { time: 0.5 },
                actions: Vec::new(),
                requirement: Some("REQ-MODE".to_string()),
            }],
        }],
        fmi: FmiInteropSpec {
            imports: Vec::new(),
            exports: vec![FmuExportSpec {
                id: "closed-loop-fmu-plan".to_string(),
                interface: FmiInterfaceKind::CoSimulation,
                include_rust_source: true,
            }],
        },
        codegen: CodegenSpec {
            target: CodegenTarget::Rust,
            module_name: Some("closed_loop_authoring_demo".to_string()),
            deterministic: true,
            include_schema: false,
        },
        verification: VerificationSpec {
            requirements: vec![
                RequirementSpec {
                    id: "REQ-TRACK".to_string(),
                    text: "The plant shall track the unit reference.".to_string(),
                    source: Some("demo".to_string()),
                    safety_level: None,
                },
                RequirementSpec {
                    id: "REQ-MODE".to_string(),
                    text: "The supervisor shall leave startup after 0.5 seconds.".to_string(),
                    source: Some("demo".to_string()),
                    safety_level: None,
                },
            ],
            trace_links: vec![TraceLinkSpec {
                requirement: "REQ-TRACK".to_string(),
                element: "controller".to_string(),
                relation: TraceRelation::Satisfies,
            }],
            coverage: CoverageSpec {
                require_block_coverage: true,
                require_transition_coverage: true,
                target_mcdc: Some(0.8),
            },
            properties: vec![FormalPropertySpec {
                id: "PROP-NO-NAN".to_string(),
                expression: "always finite(plant.p0)".to_string(),
                severity: PropertySeverity::Error,
            }],
        },
        tooling: ToolingSpec {
            enable_model_browser: true,
            enable_library_browser: true,
            enable_signal_inspector: true,
            enable_variant_manager: true,
            enable_dependency_analyzer: true,
            enable_parameter_estimation: true,
            enable_protected_models: true,
            enable_collaboration: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_is_generated_from_rust_types() {
        let schema = authoring_json_schema();
        let text = serde_json::to_string(&schema).unwrap();
        assert!(text.contains("AuthoringSpec"));
        assert!(text.contains("BlockKindSpec"));
    }

    #[test]
    fn example_spec_validates_and_runs() {
        let value = serde_json::to_value(example_spec()).unwrap();
        let parsed = parse_authoring_spec(&value).unwrap();
        let (compiled, opts) = compile_hybrid_graph(&parsed).unwrap();
        let trace = simulate(&compiled, &opts);
        assert!(!trace.to_jsonl_frames().is_empty());
        assert!(trace.column_index("plant.p0").is_some());
    }

    #[test]
    fn physical_network_flattens_to_dae_equations() {
        let spec = example_spec();
        let dae = flatten_physical_networks(&spec);
        let joined = dae
            .equations
            .iter()
            .map(|e| e.expression.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("0 ="));
        assert!(joined.contains("potential"));
        assert!(!dae.diagnostics.is_empty());
    }

    #[test]
    fn inactive_variant_cannot_leave_required_inputs_unconnected() {
        let mut spec = example_spec();
        spec.variants[0].default = false;
        spec.active_variants.clear();
        let err = match compile_hybrid_graph(&spec) {
            Ok(_) => panic!("expected compile to reject unconnected input"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("input port 0 is not connected"));
    }

    #[test]
    fn validation_rejects_bad_numbers_and_shapes() {
        let mut spec = example_spec();
        spec.solver.max_step = 0.0;
        if let BlockKindSpec::StateSpace { a, .. } = &mut spec.blocks[3].kind {
            a[0].push(2.0);
        }
        let issues = validate_authoring_spec(&spec);
        let text = serde_json::to_string(&issues).unwrap();
        assert!(text.contains("solver.maxStep"));
        assert!(text.contains("expected 1 column"));
    }

    #[test]
    fn validation_rejects_bad_physical_connector_and_duplicate_binding() {
        let mut spec = example_spec();
        spec.physical_networks[0].connections[0].connector = "bad".to_string();
        let duplicate = spec.physical_networks[0].connections[1].clone();
        spec.physical_networks[0].connections.push(duplicate);
        let issues = validate_authoring_spec(&spec);
        let text = serde_json::to_string(&issues).unwrap();
        assert!(text.contains("has connector"));
        assert!(text.contains("connected to more than one node"));
    }

    #[test]
    fn unsupported_solver_modes_fail_before_simulation() {
        let mut spec = example_spec();
        spec.solver.mode = SolverMode::VariableStepRk45;
        let err = match compile_hybrid_graph(&spec) {
            Ok(_) => panic!("expected compile to reject unsupported solver"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("fixed-step RK4"));
    }

    #[test]
    fn rust_codegen_embeds_checked_spec() {
        let mut spec = example_spec();
        spec.name = "mod".to_string();
        spec.codegen.module_name = Some("fn".to_string());
        spec.description = Some("contains raw delimiter \"## safely".to_string());
        let generated = generate_rust(&spec).unwrap();
        assert!(generated.source.contains("parse_authoring_spec"));
        assert!(generated.source.contains("compile_hybrid_graph"));
        assert!(generated.source.contains("\\\"##"));
        assert_eq!(generated.module_name, "model_fn");
    }

    #[test]
    fn trace_compare_accepts_identical_runs() {
        let spec = example_spec();
        let (compiled_a, opts_a) = compile_hybrid_graph(&spec).unwrap();
        let (compiled_b, opts_b) = compile_hybrid_graph(&spec).unwrap();
        let a = simulate(&compiled_a, &opts_a);
        let b = simulate(&compiled_b, &opts_b);
        let report = compare_traces(&a, &b, TraceCompareOptions::default());
        assert!(report.passed);
        assert_eq!(report.max_abs_error, 0.0);
    }

    #[test]
    fn trace_compare_fails_on_missing_columns() {
        let spec = example_spec();
        let (compiled_a, opts_a) = compile_hybrid_graph(&spec).unwrap();
        let (compiled_b, opts_b) = compile_hybrid_graph(&spec).unwrap();
        let a = simulate(&compiled_a, &opts_a);
        let mut b = simulate(&compiled_b, &opts_b);
        b.columns.pop();
        for row in &mut b.rows {
            row.pop();
        }
        let report = compare_traces(&a, &b, TraceCompareOptions::default());
        assert!(!report.passed);
        assert!(report
            .mismatches
            .iter()
            .any(|m| m.column.contains("missing rhs column")));
    }
}
